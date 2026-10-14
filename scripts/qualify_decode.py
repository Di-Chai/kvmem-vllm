"""Measure concurrent decode in shared, fully active token-arrival windows.

Exact output counts require streamed token IDs AND matching final usage. SSE
timings resolve speculative bursts, not individual token emission within a burst.
Optional native snapshots independently corroborate stable decode ownership.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import httpx

from qualify_multi_user import distribution, load_users, redact_strings, validate_api_base
from sample_decode_progress import read_sample, steady_interval


DEFAULT_PROMPT = """Write a comprehensive engineering design review for a multi-user inference
service in at least 36 numbered sections. Each section must contain at least 150
words of substantive discussion plus relevant Python pseudocode or concrete
failure examples. Cover admission control, per-user fairness, bounded queues,
streaming, prefill versus decode, memory accounting, KV reuse, authentication,
cache isolation, cancellation, timeouts, overload, storage, observability,
recovery, capacity planning, and meaningful tests. Explain alternatives and their
tradeoffs in every section. Start directly with section 1. Continue sequentially
through all 36 sections; do not summarize, conclude, or stop early."""


class DecodeError(ValueError):
    pass


async def stream_events(response):
    lines, size = [], 0
    async for line in response.aiter_lines():
        size += len(line)
        if size > 1024 * 1024:
            raise DecodeError('sse_event_too_large')
        if not line:
            if lines:
                data = '\n'.join(lines)
                if data == '[DONE]':
                    yield None
                    return
                try:
                    event = json.loads(data)
                except ValueError:
                    raise DecodeError('invalid_sse_json') from None
                if not isinstance(event, dict) or event.get('error'):
                    raise DecodeError('sse_error')
                yield event
            lines, size = [], 0
        elif line.startswith('data:'):
            lines.append(line[5:].removeprefix(' '))
    raise DecodeError('missing_done')


def consume_event(trace, event, at):
    """Keep exact token IDs/counts so fixed-output content can be audited."""
    if isinstance(event.get('id'), str):
        if trace.get('_response_id') not in (None, event['id']):
            raise DecodeError('response_identity_changed')
        trace['_response_id'] = event['id']
    usage = event.get('usage')
    if usage is not None:
        if not isinstance(usage, dict):
            raise DecodeError('invalid_usage')
        for field in ('prompt_tokens', 'completion_tokens'):
            value = usage.get(field)
            if value is not None:
                if type(value) is not int or value < 0:
                    raise DecodeError('invalid_usage')
                trace[field] = value
    choices = event.get('choices', [])
    if not isinstance(choices, list) or len(choices) > 1:
        raise DecodeError('unexpected_choices')
    for choice in choices:
        if not isinstance(choice, dict) or choice.get('index', 0) != 0:
            raise DecodeError('unexpected_choice')
        tokens = choice.get('token_ids')
        if tokens is not None:
            if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
                raise DecodeError('invalid_token_ids')
            trace['saw_token_ids'] = True
            if tokens:
                trace['observed_tokens'] += len(tokens)
                trace['timeline'].append({'at': at, 'tokens': trace['observed_tokens'],
                                          'delta_tokens': len(tokens), 'token_ids': tokens})
                trace['_token_digest'].update(''.join(f'{token},' for token in tokens).encode())
        if choice.get('finish_reason') is not None:
            trace['finish_reason'] = choice['finish_reason']
            trace['finish_at'] = at


def finish_trace(trace, expected_tokens):
    if not trace['saw_token_ids'] or not trace['timeline']:
        raise DecodeError('missing_streamed_token_ids')
    if trace.get('completion_tokens') is None or trace.get('prompt_tokens') is None:
        raise DecodeError('missing_final_usage')
    if trace['observed_tokens'] != trace['completion_tokens']:
        raise DecodeError('token_ids_usage_mismatch')
    if trace['completion_tokens'] != expected_tokens:
        raise DecodeError('fixed_output_length_not_reached')
    if trace.get('finish_reason') != 'length':
        raise DecodeError('unexpected_finish_reason')
    first, last = trace['timeline'][0], trace['timeline'][-1]
    remaining = last['tokens'] - first['tokens']
    elapsed = last['at'] - first['at']
    trace.update(first_token_at=first['at'], last_token_at=last['at'],
                 ttft_seconds=first['at'] - trace['started_at'],
                 first_burst_tokens=first['tokens'], decode_seconds=elapsed,
                 tokens_after_first_burst=remaining,
                 tpot_seconds=elapsed / remaining if remaining > 0 else None)


def count_at(trace, at, *, include_boundary=True):
    points = [point['tokens'] for point in trace['timeline']
              if point['at'] <= at] if include_boundary else [point['tokens'] for point in trace['timeline']
                                                             if point['at'] < at]
    return points[-1] if points else 0


def decode_window(traces, trim_tokens, min_seconds, min_events):
    """Intersect per-request decode intervals; keep all stalls inside them."""
    if not traces or any(trace['outcome'] != 'success' for trace in traces):
        return {'status': 'failed', 'error': 'incomplete_requests'}
    starts, ends = [], []
    for trace in traces:
        timeline, total = trace['timeline'], trace['observed_tokens']
        if not timeline or total <= 2 * trim_tokens:
            return {'status': 'failed', 'error': 'insufficient_output_tokens'}
        if trim_tokens:
            start = next((point['at'] for point in timeline if point['tokens'] >= trim_tokens), None)
            safe_ends = [point['at'] for point in timeline if point['tokens'] <= total - trim_tokens]
            end = safe_ends[-1] if safe_ends else None
        else:
            start, end = timeline[0]['at'], timeline[-1]['at']
        if start is None or end is None:
            return {'status': 'failed', 'error': 'no_shared_decode_window'}
        starts.append(start)
        ends.append(end)
    start, end = max(starts), min(ends)
    seconds = end - start
    result = {'status': 'failed', 'start': start, 'end': end, 'seconds': max(0, seconds),
              'trim_tokens_each_end': trim_tokens, 'active_streams': len(traces),
              'end_inclusive': bool(trim_tokens), 'per_request': []}
    if seconds <= 0:
        return dict(result, error='no_shared_decode_window')
    for trace in traces:
        before = count_at(trace, start)
        after = count_at(trace, end, include_boundary=bool(trim_tokens))
        points = [point for point in trace['timeline'] if start < point['at'] and
                  (point['at'] <= end if trim_tokens else point['at'] < end)]
        result['per_request'].append({'client_index': trace['client_index'],
            'start_tokens': before, 'end_tokens': after, 'tokens': after - before,
            'token_events': len(points)})
    tokens = sum(item['tokens'] for item in result['per_request'])
    result.update(output_tokens=tokens, output_tokens_per_second=tokens / seconds)
    if seconds < min_seconds:
        return dict(result, error='window_too_short')
    if any(item['tokens'] <= 0 or item['token_events'] < min_events for item in result['per_request']):
        return dict(result, error='not_all_requests_have_enough_progress')
    result['status'] = 'passed'
    return result


def native_confirmation(samples, traces, window, concurrency, min_intervals, min_coverage=.9):
    if window.get('status') != 'passed':
        return {'status': 'unavailable', 'error': 'no_valid_sse_window', 'intervals': []}
    response_ids = [trace.get('_response_id') for trace in traces]
    violations, boundary_observations = [], []
    previous_counter = None

    def matches_owners(owners):
        return len(owners) == concurrency and all(isinstance(rid, str) and
            sum(owner == rid or owner.startswith(rid + '-') for owner in owners) == 1
            for rid in response_ids)

    # The capture files are separate snapshots. Only attribute a state change
    # to this window when the whole read is strictly inside its boundaries.
    # Boundary-spanning observations are kept and remove coverage; they cannot
    # prove contamination happened before or after the exact SSE boundary.
    for index, sample in enumerate(samples):
        at = sample['at']
        read_started = sample.get('read_started_at', at)
        overlaps = at >= window['start'] and read_started <= window['end']
        inside = window['start'] < read_started <= at < window['end']
        reasons = []
        if sample.get('error'):
            reasons.append('sample_read_error')
        else:
            try:
                if sample['prefill_raced']:
                    reasons.append('prefill_raced_read')
                if sample['scheduler']['refresh_pending']:
                    reasons.append('refresh_during_window')
                if not matches_owners(sample['progress']['accepted_history']):
                    reasons.append('request_ownership_mismatch')
                if previous_counter is not None:
                    delta = sample['prefill'] - previous_counter['prefill']
                    previous_start = previous_counter.get('read_started_at', previous_counter['at'])
                    if delta and at >= window['start'] and previous_start <= window['end']:
                        counter_observation = {'sample_index': index, 'read_started_at': previous_start,
                            'read_finished_at': at, 'reasons': ['prefill_increment_during_window'
                            if delta > 0 else 'prefill_counter_regressed'], 'prefill_delta': delta}
                        counter_inside = previous_start >= window['start'] and at <= window['end']
                        (violations if counter_inside else boundary_observations).append(counter_observation)
                previous_counter = sample
            except (KeyError, TypeError, AttributeError):
                reasons.append('invalid_native_sample')
        if reasons and overlaps:
            observation = {'sample_index': index, 'read_started_at': read_started,
                           'read_finished_at': at, 'reasons': reasons}
            if sample.get('error'):
                observation['error'] = sample['error']
            (violations if inside else boundary_observations).append(observation)
    intervals = []
    for index in range(2, len(samples)):
        group = samples[index - 2:index + 1]
        try:
            interval = steady_interval(*group, concurrency)
        except (KeyError, TypeError, AttributeError):
            continue
        if interval is None or interval['start'] < window['start'] or interval['end'] > window['end']:
            continue
        if any(sample.get('read_started_at', sample['at']) < window['start'] for sample in group[1:]):
            continue
        owners = group[-1]['progress']['accepted_history']
        # Stable foreign traffic is not evidence that this batch ran at C.
        if not matches_owners(owners):
            continue
        intervals.append(interval)
    seconds = sum(interval['seconds'] for interval in intervals)
    tokens = sum(interval['accepted_input_tokens'] for interval in intervals)
    per_owner = Counter()
    for interval in intervals:
        per_owner.update(interval['per_request_tokens'])
    coverage = seconds / (window['end'] - window['start'])
    confirmed = (not violations and len(intervals) >= min_intervals and len(per_owner) == concurrency and
                 all(value > 0 for value in per_owner.values()) and seconds > 0 and coverage >= min_coverage)
    return {'status': 'passed' if confirmed else 'failed' if violations else 'insufficient_evidence',
            'actual_stable_owner_count': concurrency if intervals else None,
            'interval_count': len(intervals), 'seconds': seconds,
            'sse_window_coverage_fraction': coverage, 'required_coverage_fraction': min_coverage,
            'violations': violations, 'boundary_observations': boundary_observations,
            'accepted_input_tokens': tokens,
            'accepted_input_tokens_per_second': tokens / seconds if seconds else None,
            'sample_count': len(samples), 'sample_errors': sum('error' in sample for sample in samples),
            'intervals': intervals,
            'scope': 'Approximate archive-commit rate in qualified subintervals of the SSE window; '
                     'snapshot lag and gaps prevent treating it as exact arrival throughput.'}


def make_payload(args, prompt):
    payload = {'model': args.model, 'max_tokens': args.output_tokens, 'temperature': 0,
               'seed': args.seed, 'ignore_eos': True, 'stream': True, 'return_token_ids': True,
               'stream_options': {'include_usage': True}}
    if getattr(args, 'prompt_sessions', False):
        payload['session_id'] = 'decode-qualification-' + uuid.uuid4().hex
    if args.endpoint == 'chat/completions':
        payload.update(messages=[{'role': 'user', 'content': prompt}],
                       chat_template_kwargs={'enable_thinking': False})
    else:
        payload['prompt'] = prompt
    return payload


def load_prompt(args):
    """Return exact input plus metadata; token-ID input never uses a template."""
    if args.prompt_token_file:
        raw = args.prompt_token_file.read_bytes()
        value = json.loads(raw)
        tokens = value.get('input_ids') if isinstance(value, dict) else None
        if (not isinstance(tokens, list) or not tokens or
                any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError('Token prompt file requires a nonempty input_ids array of nonnegative integers')
        canonical = json.dumps(tokens, separators=(',', ':')).encode()
        return tokens, {'kind': 'token_ids', 'input_tokens': len(tokens),
                        'input_file_sha256': hashlib.sha256(raw).hexdigest(),
                        'token_ids_sha256': hashlib.sha256(canonical).hexdigest()}, raw
    prompt = args.prompt_file.read_text() if args.prompt_file else DEFAULT_PROMPT
    if not prompt.strip():
        raise ValueError('Prompt must not be empty')
    return prompt, {'kind': 'text', 'text_sha256': hashlib.sha256(prompt.encode()).hexdigest()}, None


def end_to_end_summary(traces, wall):
    successes = [trace for trace in traces if trace['outcome'] == 'success']
    outcomes = Counter(trace['outcome'] for trace in traces)
    tokens = sum(trace['completion_tokens'] for trace in successes)
    return {'status': 'passed' if len(successes) == len(traces) and traces else 'failed',
            'attempted': len(traces), 'success': len(successes),
            'rejected': outcomes['rejected'], 'failure': outcomes['failure'],
            'wall_seconds': wall, 'successful_output_tokens': tokens,
            'successful_output_tokens_per_second': tokens / wall if wall > 0 else None,
            'successful_requests_per_second': len(successes) / wall if wall > 0 else None,
            'ttft_seconds': distribution([trace['ttft_seconds'] for trace in successes]),
            'request_latency_seconds': distribution([trace['elapsed_seconds'] for trace in successes]),
            'tpot_seconds': distribution([trace['tpot_seconds'] for trace in successes
                                          if trace['tpot_seconds'] is not None]),
            'scope': 'HTTP service throughput over the whole batch, including prefill, queuing and response drain.'}


def batch_windows(args, traces, samples, concurrency, wall):
    common = decode_window(traces, 0, args.min_window_seconds, args.min_events_per_request)
    steady = decode_window(traces, args.trim_tokens, args.min_window_seconds, args.min_events_per_request)
    service = end_to_end_summary(traces, wall)
    inapplicable = {'no_shared_decode_window', 'window_too_short', 'not_all_requests_have_enough_progress'}
    if args.allow_queued and service['status'] == 'passed':
        for window in (common, steady):
            if window.get('error') in inapplicable:
                window.update(status='not_applicable', reason='no_qualified_full_client_cohort_window')
    native = (native_confirmation(samples, traces, steady, concurrency, args.min_native_intervals,
                                  args.min_native_coverage)
              if args.capture else {'status': 'not_requested'})
    if steady['status'] == 'not_applicable':
        native = {'status': 'not_applicable', 'reason': 'no_qualified_full_client_cohort_window'}
        passed = service['status'] == 'passed'
    elif args.allow_queued and service['status'] == 'passed' and native['status'] not in ('passed', 'not_requested'):
        # Streaming lifetimes can overlap while only a smaller cohort is
        # scheduled. Keep the failed corroboration instead of calling that C
        # active decode or turning a successful queued service into a failure.
        steady.update(status='not_applicable', reason='full_client_cohort_not_native_confirmed')
        common.update(status='not_applicable', reason='full_client_cohort_not_native_confirmed')
        native = {'status': 'not_applicable', 'reason': 'full_client_cohort_not_native_confirmed',
                  'confirmation_attempt': native}
        passed = True
    else:
        passed = (service['status'] == 'passed' and steady['status'] == 'passed' and
                  native['status'] in ('passed', 'not_requested'))
    return {'status': 'passed' if passed else 'failed', 'end_to_end': service,
            'common_decode_window': common, 'steady_decode_window': steady,
            'native_confirmation': native}


async def run_batch(args, users, concurrency, prompt):
    origin = time.monotonic()
    samples, sampling_done = [], asyncio.Event()
    start_together = asyncio.Event()
    limits = httpx.Limits(max_connections=concurrency + 1, max_keepalive_connections=concurrency)

    async def sample_native():
        while not sampling_done.is_set():
            read_started = time.monotonic() - origin
            try:
                sample = read_sample(args.capture)
                sample['at'] -= origin
            except (OSError, ValueError, KeyError, TypeError) as error:
                sample = {'at': time.monotonic() - origin, 'error': type(error).__name__}
            sample['read_started_at'] = read_started
            samples.append(sample)
            try:
                await asyncio.wait_for(sampling_done.wait(), args.sample_interval)
            except TimeoutError:
                pass

    async with httpx.AsyncClient(timeout=args.timeout, limits=limits, trust_env=False) as client:
        async def request(index):
            trace = {'client_index': index, 'outcome': 'failure', 'http_status': None,
                     'timeline': [], 'observed_tokens': 0, 'saw_token_ids': False,
                     '_token_digest': hashlib.sha256()}
            await start_together.wait()
            trace['started_at'] = time.monotonic() - origin
            try:
                async with asyncio.timeout(args.timeout):
                    async with client.stream('POST', args.api_base + '/' + args.endpoint,
                        headers={'Authorization': 'Bearer ' + users[index]['api_key']},
                        json=make_payload(args, prompt)) as response:
                        trace['http_status'] = response.status_code
                        if response.status_code != 200:
                            trace['outcome'] = ('rejected' if response.status_code in (409, 429, 503)
                                                else 'failure')
                            trace['error'] = 'http_' + str(response.status_code)
                            return trace
                        if response.headers.get('content-type', '').split(';')[0].strip() != 'text/event-stream':
                            raise DecodeError('unexpected_content_type')
                        done = False
                        async for event in stream_events(response):
                            if event is None:
                                done = True
                                break
                            consume_event(trace, event, time.monotonic() - origin)
                        if not done:
                            raise DecodeError('missing_done')
                        finish_trace(trace, args.output_tokens)
                        if isinstance(prompt, list) and trace['prompt_tokens'] != len(prompt):
                            raise DecodeError('prompt_token_count_mismatch')
                        trace['outcome'] = 'success'
            except DecodeError as error:
                trace['error'] = str(error)
            except (httpx.HTTPError, TimeoutError) as error:
                trace['error'] = type(error).__name__
            finally:
                trace['ended_at'] = time.monotonic() - origin
                trace['elapsed_seconds'] = trace['ended_at'] - trace['started_at']
                trace['token_ids_sha256'] = trace.pop('_token_digest').hexdigest()
            return trace

        monitor = asyncio.create_task(sample_native()) if args.capture else None
        tasks = [asyncio.create_task(request(index)) for index in range(concurrency)]
        start_together.set()
        try:
            traces = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            sampling_done.set()
            if monitor:
                await monitor
    wall = time.monotonic() - origin
    return {**batch_windows(args, traces, samples, concurrency, wall), 'concurrency': concurrency,
            'distinct_users': concurrency, 'wall_seconds': wall,
            'outcomes': dict(Counter(trace['outcome'] for trace in traces)),
            'wall_output_tokens_per_second': sum(trace['observed_tokens'] for trace in traces) / wall,
            'requests': traces, 'native_samples': samples}


def public_evidence(batch):
    """Replace native request IDs with client aliases; retain raw count/timing data."""
    identities = {trace.get('_response_id'): f'client_{trace["client_index"]}'
                  for trace in batch['requests'] if trace.get('_response_id')}

    def alias(name):
        if not isinstance(name, str):
            return name
        for response_id, label in identities.items():
            if name == response_id or name.startswith(response_id + '-'):
                return label
        return name

    def walk(value):
        if isinstance(value, dict):
            return {alias(key): walk(item) for key, item in value.items() if not key.startswith('_')}
        if isinstance(value, list):
            return [walk(item) for item in value]
        return value
    return walk(batch)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='http://127.0.0.1:18380/v1')
    parser.add_argument('--users-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--model', default='kvmem')
    parser.add_argument('--endpoint', choices=('completions', 'chat/completions'))
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--prompt-sessions', action='store_true',
                        help='Include native session/checkpoint maintenance in the decode measurement')
    parser.add_argument('--trim-tokens', type=int, default=64)
    prompt_source = parser.add_mutually_exclusive_group()
    prompt_source.add_argument('--prompt-file', type=Path)
    prompt_source.add_argument('--prompt-token-file', type=Path,
                               help='JSON input_ids; forces completions with no re-tokenization')
    parser.add_argument('--allow-queued', action='store_true',
                        help='No shared full-client decode window is not_applicable when HTTP requests all succeed')
    parser.add_argument('--seed', type=int, default=104729)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--sample-interval', type=float, default=.25)
    parser.add_argument('--min-window-seconds', type=float, default=2)
    parser.add_argument('--min-events-per-request', type=int, default=4)
    parser.add_argument('--min-native-intervals', type=int, default=3)
    parser.add_argument('--min-native-coverage', type=float, default=.9,
                        help='Minimum fraction of the SSE steady window corroborated by native intervals')
    args = parser.parse_args(argv)
    if args.prompt_token_file and args.endpoint == 'chat/completions':
        parser.error('Token-ID prompts require the completions endpoint')
    args.endpoint = args.endpoint or ('completions' if args.prompt_token_file else 'chat/completions')
    try:
        args.api_base = validate_api_base(args.api_base)
    except ValueError:
        parser.error('Invalid API base')
    if any(not 1 <= value <= 128 for value in args.concurrency) or args.rounds < 1:
        parser.error('Need positive rounds and concurrency levels between 1 and 128')
    if args.trim_tokens < 1 or args.output_tokens <= 2 * args.trim_tokens:
        parser.error('Output length must exceed twice the positive token trim')
    if args.min_events_per_request < 2 or args.min_native_intervals < 1:
        parser.error('Need >=2 token events per request and >=1 native interval')
    if any(not math.isfinite(value) or value <= 0 for value in (
            args.timeout, args.sample_interval, args.min_window_seconds)):
        parser.error('Timeout and sampling/window durations must be positive and finite')
    if not math.isfinite(args.min_native_coverage) or not 0 < args.min_native_coverage <= 1:
        parser.error('Minimum native coverage must be in (0, 1]')
    return args


def main(argv=None):
    args = parse_args(argv)
    users = load_users(args.users_file)
    if len(users) < max(args.concurrency):
        raise ValueError('Need a distinct API user for every concurrent request')
    prompt, input_metadata, raw_input = load_prompt(args)
    args.output.mkdir(parents=True, exist_ok=False)
    if raw_input is not None:
        (args.output / 'input.json').write_bytes(raw_input)
        input_metadata['frozen_file'] = 'input.json'
    report = {'status': 'running', 'scope': __doc__, 'batches': [],
              'configuration': {key: value for key, value in vars(args).items()
                  if key not in ('users_file', 'capture', 'output', 'prompt_file', 'prompt_token_file')},
              'input': input_metadata,
              'prompt_sha256': input_metadata.get('input_file_sha256', input_metadata.get('text_sha256')),
              'prompt': prompt if isinstance(prompt, str) else None,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'metric_notes': [
                  'The same input is used across C and rounds; token-ID input is sent directly to completions '
                  'and its final prompt usage must match the supplied token count.',
                  'Token counts come from streamed choices.token_ids, checked against final usage.',
                  'TTFT is dispatch to first generated token-ID event, even if visible text is empty.',
                  'TPOT = (last token-event time - first token-event time)/(total tokens - first burst tokens).',
                  'Whole-request TPOT can include other requests prefill interference before the shared window starts.',
                  'Common window starts after every request has a generated token and excludes the first terminal burst.',
                  'Steady window starts after every request has >=trim tokens and ends at the earliest final '
                  'event that leaves >=trim tokens; boundary bursts can trim extra tokens.',
                  'Window rates use aggregate cumulative-count differences over one shared wall interval; '
                  'zero-progress intervals and all stalls within that interval are retained.',
                  'Every request must make progress and meet minimum duration/event counts; '
                  'supplied native capture must independently confirm stable ownership at requested concurrency.',
                  'Native confirmation requires >=90% coverage by default and rejects interior prefill increments, '
                  'refreshes, owner mismatches or read errors; boundary-spanning reads are retained as ambiguous.',
                  'Native snapshots are approximate commit evidence, not exact arrival timestamps.',
                  'With allow_queued, passed HTTP service can have no applicable full-client decode window; '
                  'an actual-owner cohort analysis is separate and must not relabel queued clients as active decode.',
                  'This is fixed-output synthetic decode performance, not task quality or mixed-prefill service latency.']}

    def save():
        public = redact_strings(report, [user['api_key'] for user in users])
        temporary = args.output / 'results.tmp'
        temporary.write_text(json.dumps(public, indent=2) + '\n')
        temporary.replace(args.output / 'results.json')

    try:
        save()
        for concurrency in args.concurrency:
            for round_index in range(args.rounds):
                batch = public_evidence(asyncio.run(run_batch(args, users, concurrency, prompt)))
                batch['round_index'] = round_index
                report['batches'].append(batch)
                save()
                print(json.dumps({'concurrency': concurrency, 'round_index': round_index,
                                  'status': batch['status'], 'steady': batch['steady_decode_window'],
                                  'native': {key: value for key, value in batch['native_confirmation'].items()
                                             if key != 'intervals'}}), flush=True)
        report['status'] = 'passed' if all(batch['status'] == 'passed' for batch in report['batches']) else 'failed'
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__)
        raise
    finally:
        save()
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
