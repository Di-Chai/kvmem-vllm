"""Authenticated streaming HTTP load and lifecycle checks (standard library only).

Measurements describe this closed-loop client workload, not model/agent quality.
Streaming chunks are not tokens; token rates require server-provided usage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import http.client
import json
import math
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import urlsplit
import uuid


class StreamError(ValueError):
    """A fixed, non-secret protocol error safe to include in a report."""


def load_users(path):
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise ValueError('Cannot read users file as JSON') from None
    users = data.get('users') if isinstance(data, dict) else None
    if not isinstance(users, list) or not users:
        raise ValueError('Users file needs a nonempty users list')
    ids, keys = set(), set()
    result = []
    for entry in users:
        if not isinstance(entry, dict):
            raise ValueError('Every user needs id and api_key strings')
        user_id, key = entry.get('id'), entry.get('api_key')
        if (not isinstance(user_id, str) or not user_id.strip() or
                not isinstance(key, str) or not key.strip() or
                any(ord(c) < 33 or ord(c) > 126 for c in key)):
            raise ValueError('Every user needs an id and a printable API key without spaces')
        if user_id in ids or key in keys:
            raise ValueError('User IDs and API keys must each be unique')
        ids.add(user_id)
        keys.add(key)
        result.append({'id': user_id, 'api_key': key})
    return result


def validate_api_base(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or
            parsed.username is not None or parsed.password is not None or
            parsed.query or parsed.fragment):
        raise ValueError('API base must be an HTTP(S) URL without credentials, query or fragment')
    # Force malformed port validation before requests begin.
    parsed.port
    return value.rstrip('/')


def iter_sse(readline, max_event_bytes=1024 * 1024):
    """Parse SSE framing, including comments, CRLF and multiline data fields."""
    data, size = [], 0
    while True:
        line = readline(max_event_bytes + 1)
        if not line:
            if data:
                raise StreamError('sse_event_truncated')
            return
        size += len(line)
        if size > max_event_bytes:
            raise StreamError('sse_event_too_large')
        if not line.endswith(b'\n'):
            raise StreamError('sse_line_truncated')
        try:
            line = line.rstrip(b'\r\n').decode('utf-8')
        except UnicodeDecodeError:
            raise StreamError('sse_invalid_utf8') from None
        if not line:
            if data:
                yield '\n'.join(data)
            data, size = [], 0
        elif line.startswith('data:'):
            value = line[5:]
            data.append(value[1:] if value.startswith(' ') else value)


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    return values[lower] + (values[math.ceil(position)] - values[lower]) * (position - lower)


def distribution(values):
    return {'count': len(values), 'p50': percentile(values, .5),
            'p95': percentile(values, .95), 'max': max(values) if values else None}


def summarize(results, wall_seconds, min_success_rate=1.0):
    counts = Counter(result['outcome'] for result in results)
    successes = [r for r in results if r['outcome'] == 'success']
    starts = [r['started_seconds'] for r in results]
    arrival_span = max(starts) - min(starts) if starts else 0
    measured = [r['completion_tokens'] for r in successes if r.get('completion_tokens') is not None]
    quality = [r['marker_exact'] for r in successes if r.get('marker_exact') is not None]
    success_rate = len(successes) / len(results) if results else 0
    if (not successes or counts['failure'] or any(value is False for value in quality) or
            any(r.get('foreign_marker_observed') for r in results)):
        status = 'failed'
    elif success_rate < min_success_rate:
        status = 'capacity_limited'
    else:
        status = 'passed'
    return {
        'status': status, 'attempted': len(results), 'success': counts['success'],
        'rejected': counts['rejected'], 'failure': counts['failure'],
        'success_rate': success_rate, 'required_success_rate': min_success_rate,
        'http_status_counts': dict(sorted(Counter(str(r['http_status']) for r in results).items())),
        'wall_seconds': wall_seconds, 'arrival_span_seconds': arrival_span,
        'attempts_per_second_over_wall': len(results) / wall_seconds if wall_seconds > 0 else None,
        'successful_requests_per_second_over_wall': len(successes) / wall_seconds if wall_seconds > 0 else None,
        'arrival_rate_over_start_span': (len(results) - 1) / arrival_span if arrival_span > 0 else None,
        'completion_tokens_observed': sum(measured),
        'successful_requests_with_token_usage': len(measured),
        'output_tokens_per_second_over_wall': (sum(measured) / wall_seconds
            if successes and len(measured) == len(successes) and wall_seconds > 0 else None),
        'success_latency_seconds': distribution([r['elapsed_seconds'] for r in successes]),
        'all_attempt_latency_seconds': distribution([r['elapsed_seconds'] for r in results]),
        'ttft_seconds': distribution([r['ttft_seconds'] for r in successes if r['ttft_seconds'] is not None]),
        'marker_checks': {'checked': len(quality), 'passed': sum(quality),
                          'failed': len(quality) - sum(quality)},
    }


def redact_strings(value, secrets):
    """Redact string values without corrupting JSON numbers or booleans."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, '[REDACTED]')
        return value
    if isinstance(value, dict):
        return {key: redact_strings(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_strings(item, secrets) for item in value]
    return value


def contains_marker(text, marker):
    """Match a whole ASCII identifier, never a different tenant's prefix."""
    return re.search(r'(?<![A-Za-z0-9_])' + re.escape(marker) +
                     r'(?![A-Za-z0-9_])', text) is not None


def make_payload(args, marker, session, *, output_tokens=None, force_marker=False):
    filler = ' '.join('context' for _ in range(args.prompt_words))
    if args.check_markers or force_marker:
        prompt = (f'{filler}\nThe private retrieval marker is {marker}.\n'
                  'Return only the private retrieval marker, exactly, with no other text.\nAnswer: ')
    else:
        prompt = (f'{filler}\nWrite a detailed numbered list of practical software testing steps. '
                  'Keep writing until the output limit.\nAnswer: ')
    payload = {'model': args.model, 'temperature': 0, 'seed': 104729,
               'max_tokens': output_tokens or args.output_tokens, 'stream': True,
               'stream_options': {'include_usage': True}, 'session_id': session}
    if args.endpoint == 'chat/completions':
        payload['messages'] = [{'role': 'user', 'content': prompt}]
        if args.disable_thinking:
            payload['chat_template_kwargs'] = {'enable_thinking': False}
    else:
        payload['prompt'] = prompt
    if args.ignore_eos:
        payload['ignore_eos'] = True
    return payload


def stream_request(api_base, endpoint, api_key, payload, *, timeout=600, origin=None,
                   marker=None, forbidden_markers=(), cancel_after_chunks=None, user_index=0):
    """Return bounded diagnostics; never retain headers, prompts or response text."""
    started = time.monotonic()
    origin = started if origin is None else origin
    result = {'user_index': user_index, 'outcome': 'failure', 'http_status': None,
              'started_seconds': started - origin, 'ttft_seconds': None,
              'completion_tokens': None, 'prompt_tokens': None, 'content_chunks': 0,
              'marker_exact': None, 'foreign_marker_observed': False, 'saw_done': False}
    parsed = urlsplit(api_base)
    connection_class = http.client.HTTPSConnection if parsed.scheme == 'https' else http.client.HTTPConnection
    connection = connection_class(parsed.hostname, parsed.port, timeout=timeout)
    response = transport = None
    output, output_size, finished_choice = [], 0, False
    deadline = started + timeout
    try:
        connection.request('POST', parsed.path.rstrip('/') + '/' + endpoint,
            json.dumps(payload).encode(), {'Authorization': 'Bearer ' + api_key,
            'Content-Type': 'application/json', 'Accept': 'text/event-stream'})
        # getresponse() detaches connection.sock for Connection: close; keep
        # the live transport so deadlines/disconnect also cover that response.
        transport = connection.sock
        response = connection.getresponse()
        result['http_status'] = response.status
        if response.status != 200:
            result['outcome'] = 'rejected' if response.status in (409, 429, 503) else 'failure'
            result['error'] = 'http_' + str(response.status)
            return result
        if response.getheader('Content-Type', '').split(';')[0].strip().lower() != 'text/event-stream':
            raise StreamError('unexpected_content_type')

        def read_line(limit):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            if transport is not None:
                transport.settimeout(remaining)
            return response.readline(limit)

        for event in iter_sse(read_line):
            if event == '[DONE]':
                result['saw_done'] = True
                break
            try:
                chunk = json.loads(event)
            except ValueError:
                raise StreamError('invalid_stream_json') from None
            if not isinstance(chunk, dict) or chunk.get('error'):
                raise StreamError('stream_error_event')
            usage = chunk.get('usage')
            if usage is not None:
                if not isinstance(usage, dict):
                    raise StreamError('invalid_usage')
                for key in ('completion_tokens', 'prompt_tokens'):
                    value = usage.get(key)
                    if value is not None:
                        if type(value) is not int or value < 0:
                            raise StreamError('invalid_usage')
                        result[key] = value
            choices = chunk.get('choices', [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise StreamError('unexpected_choices')
            for choice in choices:
                if not isinstance(choice, dict):
                    raise StreamError('unexpected_choice')
                delta = choice.get('delta') or {}
                if not isinstance(delta, dict):
                    raise StreamError('unexpected_delta')
                text = choice.get('text', delta.get('content')) or ''
                if not isinstance(text, str):
                    raise StreamError('non_text_content')
                if text:
                    if result['ttft_seconds'] is None:
                        result['ttft_seconds'] = time.monotonic() - started
                    result['content_chunks'] += 1
                    output_size += len(text)
                    if output_size > 16 * 1024 * 1024:
                        raise StreamError('output_too_large')
                    output.append(text)
                if choice.get('finish_reason') is not None:
                    finished_choice = True
            if cancel_after_chunks is not None and result['content_chunks'] >= cancel_after_chunks:
                # Close an active response, not a drained stream. This verifies
                # client disconnect/recovery; native abort needs backend evidence.
                if transport is not None:
                    transport.shutdown(socket.SHUT_RDWR)
                result['outcome'] = 'cancelled'
                return result
        if not result['saw_done']:
            raise StreamError('missing_done')
        if not finished_choice:
            raise StreamError('missing_finish_reason')
        if not result['content_chunks']:
            raise StreamError('empty_generation')
        text = ''.join(output).strip()
        result['marker_exact'] = text == marker if marker is not None else None
        result['foreign_marker_observed'] = any(contains_marker(text, value) for value in forbidden_markers)
        result['outcome'] = 'success'
        return result
    except StreamError as error:
        result['error'] = str(error)
        return result
    except (OSError, http.client.HTTPException, ValueError) as error:
        # Exception messages and upstream bodies may echo credentials.
        result['error'] = type(error).__name__
        return result
    finally:
        if response is not None:
            response.close()
        connection.close()
        result['elapsed_seconds'] = time.monotonic() - started


def run_level(args, users, concurrency, run_id):
    origin = time.monotonic()
    barrier = threading.Barrier(concurrency)

    def client(index):
        user_index = index % len(users)
        user = users[user_index]
        records = []
        barrier.wait(timeout=30)
        for round_index in range(args.rounds):
            marker = f'KVMEM_{run_id}_{user_index}_{index}_{round_index}'
            session = f'load-{run_id}-{concurrency}-{index}'
            payload = make_payload(args, marker, session)
            record = stream_request(args.api_base, args.endpoint, user['api_key'], payload,
                timeout=args.timeout, origin=origin, marker=marker if args.check_markers else None,
                user_index=user_index)
            record.update(client_index=index, round_index=round_index)
            records.append(record)
        return records

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = [record for group in executor.map(client, range(concurrency)) for record in group]
    wall = time.monotonic() - origin
    results.sort(key=lambda result: result['started_seconds'])
    return {'concurrency': concurrency, 'distinct_users': min(len(users), concurrency),
            'summary': summarize(results, wall, args.min_success_rate), 'requests': results}


def run_probes(args, users, run_id):
    """Synthetic output isolation and disconnect recovery, separate from load."""
    probe = {'scope': 'Output-marker isolation and HTTP disconnect recovery; '
             'does not prove native KV namespace isolation or native abort completion'}
    if len(users) < 2:
        return dict(probe, status='failed', error='two_distinct_users_required')
    origin = time.monotonic()
    markers = [f'KVMEM_{run_id}_PRIVATE_{index:03d}_END' for index in range(len(users))]
    session = 'adversarial-shared-session-' + run_id

    def collide(index):
        payload = make_payload(args, markers[index], session, force_marker=True)
        payload['cache_salt'] = 'adversarial-shared-salt-' + run_id
        payload['request_id'] = 'adversarial-shared-request-' + run_id
        return stream_request(args.api_base, args.endpoint, users[index]['api_key'], payload,
            timeout=args.timeout, origin=origin, marker=markers[index],
            forbidden_markers=markers[:index] + markers[index + 1:], user_index=index)

    # Repeat the same tenant prompts so the second wave can exercise reuse.
    waves = []
    for _ in range(2):
        with ThreadPoolExecutor(max_workers=len(users)) as executor:
            waves.append(list(executor.map(collide, range(len(users)))))
    probe['tenant_collision_waves'] = waves
    cancel_payload = make_payload(args, markers[0], 'cancel-' + run_id,
                                  output_tokens=max(args.output_tokens, 1024))
    # Cancellation needs a long continuation, independent of exact-marker mode.
    text = 'Write a very long numbered list of software testing ideas. Continue until the output limit.'
    if args.endpoint == 'completions':
        cancel_payload['prompt'] = text
    else:
        cancel_payload['messages'] = [{'role': 'user', 'content': text}]
    cancel_payload['ignore_eos'] = True
    probe['cancel'] = stream_request(args.api_base, args.endpoint, users[0]['api_key'],
        cancel_payload, timeout=args.timeout, origin=origin, cancel_after_chunks=1)
    recover_payload = make_payload(args, markers[0], 'cancel-' + run_id, force_marker=True)
    deadline = time.monotonic() + args.recovery_timeout
    recoveries = []
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        result = stream_request(args.api_base, args.endpoint, users[0]['api_key'],
            recover_payload, timeout=min(args.timeout, remaining), origin=origin, marker=markers[0])
        recoveries.append(result)
        if result['outcome'] != 'rejected':
            break
        time.sleep(min(.1, max(0, deadline - time.monotonic())))
    probe['recovery_attempts'] = recoveries
    checks = [record for wave in waves for record in wave]
    checks += recoveries[-1:]
    passed = (bool(recoveries) and probe['cancel']['outcome'] == 'cancelled' and
              all(r['outcome'] == 'success' and r['marker_exact'] and
                  not r['foreign_marker_observed'] for r in checks))
    probe['status'] = 'passed' if passed else 'failed'
    return probe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--users-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory; never overwrite a previous run')
    parser.add_argument('--model', default='kvmem')
    parser.add_argument('--endpoint', choices=('completions', 'chat/completions'), default='completions')
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--rounds', type=int, default=2, help='Requests per concurrent client (closed-loop)')
    parser.add_argument('--prompt-words', type=int, default=256, help='Filler words; actual tokens come from server usage')
    parser.add_argument('--output-tokens', type=int, default=128, help='Maximum output tokens per request')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--min-success-rate', type=float, default=1.0)
    parser.add_argument('--check-markers', action='store_true', help='Separate synthetic exact-retrieval check')
    parser.add_argument('--probes', action='store_true', help='Also collide tenant session IDs and test cancellation recovery')
    parser.add_argument('--recovery-timeout', type=float, default=60)
    parser.add_argument('--disable-thinking', action='store_true', help='Chat requests: enable_thinking=false')
    parser.add_argument('--ignore-eos', action='store_true', help='Force requested output budget (performance only)')
    args = parser.parse_args(argv)
    try:
        args.api_base = validate_api_base(args.api_base)
    except ValueError as error:
        parser.error(str(error))
    if any(not 1 <= value <= 1024 for value in args.concurrency):
        parser.error('Concurrency levels must be between 1 and 1024')
    if args.rounds < 1 or args.prompt_words < 0 or args.output_tokens < 1:
        parser.error('Rounds/output tokens must be positive; prompt words must be nonnegative')
    if (not math.isfinite(args.timeout) or args.timeout <= 0 or
            not math.isfinite(args.recovery_timeout) or args.recovery_timeout <= 0 or
            not 0 < args.min_success_rate <= 1):
        parser.error('Timeouts must be positive and finite; success rate must be in (0, 1]')
    if args.ignore_eos and (args.check_markers or args.probes):
        parser.error('Forced output length cannot be combined with exact-marker quality/probes')
    return args


def main(argv=None):
    args = parse_args(argv)
    users = load_users(args.users_file)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex[:12]
    report = {'status': 'running', 'scope': __doc__, 'run_id': run_id,
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'configuration': {key: value for key, value in vars(args).items()
                                if key not in ('users_file', 'output')},
              'user_count': len(users), 'levels': [],
              'measurement_notes': [
                  'Closed-loop clients submit their next request only after completion/rejection.',
                  'Wall rates include launch, rejection, response drain and last-request completion.',
                  'Arrival rate is (attempts - 1)/(last start - first start), not served throughput.',
                  'TTFT starts at client dispatch and ends at first nonempty content chunk; '
                  'a chunk may contain multiple tokens.',
                  'Output token rates require usage from every successful request; chunks are never counted as tokens.',
                  'Exact marker checks assess this synthetic task only; no agent-quality or native KV isolation claim.',
                  'Only indexed user references are saved; credentials and response bodies are omitted.']}

    def save():
        # Defense in depth if a supplied model/URL happens to contain a key.
        content = json.dumps(redact_strings(report, [user['api_key'] for user in users]), indent=2)
        temporary = out / 'results.tmp'
        temporary.write_text(content + '\n')
        temporary.replace(out / 'results.json')

    try:
        save()
        for level in args.concurrency:
            result = run_level(args, users, level, run_id)
            report['levels'].append(result)
            save()
            print(json.dumps({'concurrency': level, 'summary': result['summary']}), flush=True)
        if args.probes:
            report['probes'] = run_probes(args, users, run_id)
        statuses = [level['summary']['status'] for level in report['levels']]
        if args.probes:
            statuses.append(report['probes']['status'])
        report['status'] = ('failed' if 'failed' in statuses else
                            'capacity_limited' if 'capacity_limited' in statuses else 'passed')
    except BaseException as error:
        report['status'], report['error'] = 'failed', type(error).__name__
        raise
    finally:
        save()
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
