"""Bounded live overload/fairness/cancellation gate for one native HTTP service.

Uses two configured users, closes every owned stream, then verifies native
resource retirement. Run exclusively against an otherwise idle backend.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import httpx

from qualify_multi_user import load_users, redact_strings, validate_api_base


class GateFailure(RuntimeError):
    pass


def native_snapshot(capture):
    """Persist resource counts and abort ordinals, never native request IDs."""
    archive_path, scheduler_path = capture / 'archive.json', capture / 'scheduler_resources.json'
    archive = json.loads(archive_path.read_text())
    scheduler = json.loads(scheduler_path.read_text())
    fields = ('active_requests', 'retiring_requests', 'pending_downloads')
    archive_counts = {key: archive[key] for key in fields}
    archive_counts['session_retiring_requests'] = archive.get('session_retiring_requests', 0)
    scheduler_counts = {key: scheduler[key] for key in (
        'requests', 'owned_blocks', 'reservations', 'refresh_pending', 'free_blocks', 'total_blocks')}
    completed = archive.get('completed', [])
    total = archive.get('completed_requests', len(completed))
    if type(total) is not int or total < len(completed):
        raise ValueError('Invalid cumulative completion count')
    aborted_ordinals = [total - len(completed) + index + 1
                        for index, record in enumerate(completed)
                        if record.get('terminal_prefix', {}).get('status') == 'FINISHED_ABORTED']
    return {'archive': archive_counts, 'scheduler': scheduler_counts,
            'completed_count': total, 'completed_retained': len(completed),
            'completion_count_source': ('cumulative' if 'completed_requests' in archive
                                        else 'legacy_history_length'),
            'aborted_count': len(aborted_ordinals), 'aborted_count_scope': 'retained_history',
            'aborted_completion_ordinals': aborted_ordinals,
            'versions': [archive_path.stat().st_mtime_ns, scheduler_path.stat().st_mtime_ns]}


def completion_deltas(before, after):
    """Count only new retained completions, not a rolling-list length change."""
    completed = after['completed_count'] - before['completed_count']
    if completed < 0:
        raise GateFailure('native_completion_counter_moved_backwards')
    if completed > after['completed_retained']:
        raise GateFailure('new_native_completion_records_no_longer_retained')
    aborted = sum(ordinal > before['completed_count']
                  for ordinal in after['aborted_completion_ordinals'])
    return completed, aborted


def native_idle(snapshot):
    archive, scheduler = snapshot['archive'], snapshot['scheduler']
    return (not any(archive.values()) and not any(scheduler[key] for key in (
        'requests', 'owned_blocks', 'reservations', 'refresh_pending')) and
        scheduler['free_blocks'] == scheduler['total_blocks'] - 1)


async def settle(capture, deadline, *, after=None):
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            snapshot = native_snapshot(capture)
            fresh = after is None or all(current > previous for current, previous in
                                        zip(snapshot['versions'], after['versions']))
            if native_idle(snapshot) and fresh:
                consecutive += 1
                if consecutive >= 2:
                    return snapshot
            else:
                consecutive = 0
        except (OSError, ValueError, KeyError, TypeError):
            consecutive = 0
        await asyncio.sleep(.05)
    raise GateFailure('native_resources_did_not_settle')


async def events(response):
    data, size = [], 0
    async for line in response.aiter_lines():
        size += len(line)
        if size > 1024 * 1024:
            raise GateFailure('sse_event_too_large')
        if line == '':
            if data:
                event = '\n'.join(data)
                if event == '[DONE]':
                    yield None
                    return
                try:
                    value = json.loads(event)
                except ValueError:
                    raise GateFailure('invalid_sse_json') from None
                if not isinstance(value, dict) or value.get('error'):
                    raise GateFailure('sse_error')
                yield value
            data, size = [], 0
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    raise GateFailure('missing_sse_done')


def content_present(event):
    choices = event.get('choices', [])
    if not isinstance(choices, list):
        raise GateFailure('invalid_sse_choices')
    return any(isinstance(choice, dict) and (
        choice.get('text') or isinstance(choice.get('delta'), dict) and
        choice['delta'].get('content')) for choice in choices)


def payload(args, *, hold):
    prompt = ('Write a long numbered list of useful software testing techniques. Keep writing.'
              if hold else 'Reply with only the word OK.')
    value = {'model': args.model, 'max_tokens': 4096 if hold else 8,
             'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    if hold:
        value['ignore_eos'] = True
    if args.endpoint == 'chat/completions':
        value.update(messages=[{'role': 'user', 'content': prompt}],
                     chat_template_kwargs={'enable_thinking': False})
    else:
        value['prompt'] = prompt
    return value


async def short_request(client, args, user, origin, deadline):
    result = {'http_status': None, 'outcome': 'failure'}
    started = time.monotonic()
    try:
        async with asyncio.timeout_at(asyncio.get_running_loop().time() + max(0, deadline - started)):
            async with client.stream('POST', args.api_base + '/' + args.endpoint,
                headers={'Authorization': 'Bearer ' + user['api_key']}, json=payload(args, hold=False)) as response:
                result['http_status'] = response.status_code
                if response.status_code != 200:
                    result['error'] = 'http_' + str(response.status_code)
                    return result
                has_content, finished, done = False, False, False
                async for event in events(response):
                    if event is None:
                        done = True
                        break
                    if content_present(event):
                        if not has_content:
                            result['ttft_seconds'] = time.monotonic() - started
                        has_content = True
                    finished |= any(choice.get('finish_reason') is not None
                                    for choice in event.get('choices', []) if isinstance(choice, dict))
                if done and finished and has_content:
                    result['outcome'] = 'success'
                else:
                    result['error'] = 'incomplete_generation'
    except GateFailure as error:
        result['error'] = str(error)
    except (httpx.HTTPError, TimeoutError) as error:
        result['error'] = type(error).__name__
    finally:
        result.update(started_seconds=started - origin, elapsed_seconds=time.monotonic() - started)
    return result


async def run_gate(args, users):
    origin = time.monotonic()
    deadline = origin + args.scenario_timeout
    # Stop the overload phase early enough to close sockets, recover and retire.
    load_deadline = origin + args.scenario_timeout * .6
    state = {'active_content_streams': 0, 'peak_active_content_streams': 0,
             'retry_after_rejections': 0, 'peer_monitoring': False,
             'peer_min_active_content_streams': args.per_user_limit}
    changed = asyncio.Event()
    records, tasks = [], []
    before = None
    report = {'status': 'running', 'scope': __doc__, 'scenario_timeout_seconds': args.scenario_timeout,
              'fanout': args.fanout, 'expected_per_user_limit': args.per_user_limit,
              'expected_queue_capacity': args.queue_capacity, 'used_users': 2,
              'phases': {}, 'requests': records}
    phase = 'initial_native_settle'
    limits = httpx.Limits(max_connections=args.fanout + 4, max_keepalive_connections=0)

    def update_active(change):
        state['active_content_streams'] += change
        state['peak_active_content_streams'] = max(state['peak_active_content_streams'],
                                                   state['active_content_streams'])
        if state['peer_monitoring']:
            state['peer_min_active_content_streams'] = min(state['peer_min_active_content_streams'],
                                                           state['active_content_streams'])
        changed.set()

    async with httpx.AsyncClient(limits=limits, timeout=None, trust_env=False) as client:
        async def hold_stream():
            record = {'http_status': None, 'outcome': 'pending'}
            records.append(record)
            started, active = time.monotonic(), False
            try:
                async with client.stream('POST', args.api_base + '/' + args.endpoint,
                    headers={'Authorization': 'Bearer ' + users[0]['api_key']},
                    json=payload(args, hold=True)) as response:
                    record['http_status'] = response.status_code
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After', '')
                        retry_seconds = (int(retry_after) if len(retry_after) <= 10 and
                                         retry_after.isascii() and retry_after.isdigit() else 0)
                        record.update(outcome='rejected', retry_after_seconds=retry_seconds)
                        if retry_seconds > 0:
                            state['retry_after_rejections'] += 1
                        changed.set()
                        return
                    if response.status_code != 200:
                        record.update(outcome='failure', error='http_' + str(response.status_code))
                        changed.set()
                        return
                    async for event in events(response):
                        if event is None:
                            record['outcome'] = 'completed'
                            return
                        if not active and content_present(event):
                            active = True
                            record['ttft_seconds'] = time.monotonic() - started
                            update_active(1)
            except asyncio.CancelledError:
                record['outcome'] = 'client_cancelled'
                raise
            except GateFailure as error:
                record.update(outcome='failure', error=str(error))
                changed.set()
            except httpx.HTTPError as error:
                record.update(outcome='failure', error=type(error).__name__)
                changed.set()
            finally:
                if active:
                    update_active(-1)
                record['elapsed_seconds'] = time.monotonic() - started

        async def cleanup():
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            report['all_owned_requests_closed'] = all(task.done() for task in tasks)

        try:
            before = await settle(args.capture, min(deadline, origin + 3))
            report['phases']['before'] = before
            phase = 'overload_and_peer'
            tasks = [asyncio.create_task(hold_stream()) for _ in range(args.fanout)]
            async with asyncio.timeout(max(0, load_deadline - time.monotonic())):
                while (state['active_content_streams'] < args.per_user_limit or
                       state['retry_after_rejections'] < 1):
                    changed.clear()
                    if all(task.done() for task in tasks):
                        raise GateFailure('no_sustained_overload_observed')
                    await changed.wait()
                report['phases']['saturated'] = {
                    'at_seconds': time.monotonic() - origin,
                    'active_content_streams': state['active_content_streams'],
                    'retry_after_rejections': state['retry_after_rejections']}
                state['peer_monitoring'] = True
                peer = await short_request(client, args, users[1], origin, load_deadline)
                state['peer_monitoring'] = False
                peer['min_other_user_active_content_streams'] = state['peer_min_active_content_streams']
                report['phases']['peer'] = peer
                if (peer['outcome'] != 'success' or
                        peer['min_other_user_active_content_streams'] < args.per_user_limit):
                    raise GateFailure('peer_did_not_succeed_while_other_user_saturated')
            phase = 'close_owned_streams'
            await cleanup()
            report['phases']['closed_at_seconds'] = time.monotonic() - origin
            phase = 'same_user_recovery'
            attempts = []
            report['phases']['recovery_attempts'] = attempts
            while time.monotonic() < deadline - 1:
                recovered = await short_request(client, args, users[0], origin, deadline - 1)
                attempts.append(recovered)
                if recovered['outcome'] == 'success':
                    break
                if recovered['http_status'] != 429:
                    raise GateFailure('same_user_recovery_failed')
                await asyncio.sleep(.05)
            if not attempts or attempts[-1]['outcome'] != 'success':
                raise GateFailure('same_user_recovery_failed')
            phase = 'final_native_settle'
            after = await settle(args.capture, deadline, after=before)
            report['phases']['after'] = after
            report['native_completed_delta'], report['native_aborted_delta'] = completion_deltas(before, after)
            if any(record['outcome'] == 'failure' for record in records):
                raise GateFailure('unexpected_saturation_request_failure')
            if state['peak_active_content_streams'] > args.per_user_limit:
                raise GateFailure('per_user_limit_exceeded')
            report['status'] = 'passed'
        except (GateFailure, TimeoutError) as error:
            report.update(status='failed', phase=phase,
                          error=str(error) if isinstance(error, GateFailure) else 'phase_timeout')
        finally:
            await cleanup()
            # A failed fairness assertion must still leave resource evidence
            # after disconnecting every stream; never accept an old idle file.
            if before is not None and 'after' not in report['phases']:
                try:
                    report['phases']['after'] = await settle(args.capture, deadline, after=before)
                except GateFailure:
                    report['native_cleanup_error'] = 'native_resources_did_not_settle'
            report['wall_seconds'] = time.monotonic() - origin
            if report['wall_seconds'] > args.scenario_timeout:
                report.update(status='failed', error='scenario_deadline_exceeded')
            report['overload_summary'] = dict(Counter(record['outcome'] for record in records))
            report['retry_after_rejections'] = state['retry_after_rejections']
            report['peak_active_content_streams'] = state['peak_active_content_streams']
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api-base', default='http://127.0.0.1:8000/v1')
    parser.add_argument('--users-file', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='kvmem')
    parser.add_argument('--endpoint', choices=('completions', 'chat/completions'), default='chat/completions')
    parser.add_argument('--per-user-limit', type=int, default=2)
    parser.add_argument('--queue-capacity', type=int, default=64)
    parser.add_argument('--fanout', type=int, default=68)
    parser.add_argument('--scenario-timeout', type=float, default=30)
    args = parser.parse_args(argv)
    try:
        args.api_base = validate_api_base(args.api_base)
    except ValueError:
        parser.error('Invalid API base')
    if (args.per_user_limit < 1 or args.queue_capacity < 0 or
            not args.per_user_limit + args.queue_capacity < args.fanout <= 512):
        parser.error('Fanout must exceed per-user limit + queue capacity, and be <= 512')
    if not math.isfinite(args.scenario_timeout) or not 1 <= args.scenario_timeout <= 30:
        parser.error('Scenario timeout must be between 1 and 30 seconds')
    return args


def main(argv=None):
    args = parse_args(argv)
    users = load_users(args.users_file)
    if len(users) < 2:
        raise ValueError('Two distinct configured users are required')
    args.output.mkdir(parents=True, exist_ok=False)
    report = asyncio.run(run_gate(args, users))
    report['harness_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    serialized = json.dumps(redact_strings(report, [user['api_key'] for user in users]), indent=2)
    (args.output / 'results.json').write_text(serialized + '\n')
    print(json.dumps({key: report.get(key) for key in (
        'status', 'wall_seconds', 'overload_summary', 'retry_after_rejections',
        'native_completed_delta', 'native_aborted_delta', 'error')}), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
