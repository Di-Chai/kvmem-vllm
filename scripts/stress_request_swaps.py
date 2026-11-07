"""Finite cancellation qualification against an already running local server.

Requires KVMEM_LIFECYCLE_TRACE=1 and incremental admission. This client never
launches/reconfigures the server. Insufficient GPU pressure is a failed test,
not evidence of pause/cancel qualification. See --help for workload parameters.
"""
import argparse
import asyncio
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
import uuid

import httpx


PHASES = {'capturing', 'paused', 'restoring'}


@dataclass
class Stream:
    base_id: str
    response_id: str | None = None
    token_ids: list = field(default_factory=list)
    finish_reason: str | None = None
    opened: bool = False
    completed: bool = False
    cancel_requested: bool = False
    cancelled: bool = False
    closed_at: float | None = None


def internal_id_matches(base_id, internal_id):
    # Installed native completion serving adds cmpl-/prompt index; the input
    # processor optionally adds one eight-character UUID suffix for uniqueness.
    return re.fullmatch(re.escape(f'cmpl-{base_id}-0') + r'(?:-[0-9a-f]{8})?', internal_id) is not None


def cancellation_candidate(snapshot, streams):
    for internal_id, state in snapshot['request_swaps']['requests'].items():
        if state['phase'] not in PHASES:
            continue
        matching = [s for s in streams if internal_id_matches(s.base_id, internal_id)]
        if len(matching) > 1:
            raise ValueError('Ambiguous external/native request identity')
        if matching:
            stream = matching[0]
            if stream.opened and not stream.completed and stream.finish_reason is None and not stream.cancel_requested:
                return stream, internal_id, state
    return None


def outstanding(snapshot):
    """Require fenced ownership zero, not merely HTTP completion."""
    busy = {}
    for name in ('requests', 'owned_blocks', 'refresh_pending', 'connector_pending'):
        if snapshot[name]: busy[name] = snapshot[name]
    if snapshot['reservations']: busy['reservations'] = snapshot['reservations']
    if snapshot['publication_blocked']: busy['publication_blocked'] = snapshot['publication_blocked']
    if snapshot['free_blocks'] != snapshot['total_blocks'] - 1:
        busy['physical_pages'] = snapshot['total_blocks'] - 1 - snapshot['free_blocks']
    swaps = snapshot['request_swaps']
    for name in ('requests', 'inflight', 'pressure'):
        if swaps[name]: busy['swap_' + name] = swaps[name]
    for name in ('live', 'promised'):
        value = swaps['resources']['resources']['checkpoint_bytes'][name]
        if value: busy['checkpoint_' + name] = value
    if swaps['resources']['active']: busy['checkpoint_grants'] = swaps['resources']['active']
    history = snapshot['cpu_history_admission']
    for name in ('leases', 'active', 'retiring', 'reserved_bytes'):
        if history[name]: busy['history_' + name] = history[name]
    return busy


class Evidence:
    def __init__(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.file = (path / 'events.jsonl').open('x')

    def emit(self, kind, **data):
        self.file.write(json.dumps(dict(kind=kind, wall_time=time.time(), monotonic=time.monotonic(), **data)) + '\n')
        self.file.flush()

    def result(self, value):
        (self.path / 'result.json').write_text(json.dumps(value, indent=2) + '\n')


def load_snapshot(path):
    value = json.loads(path.read_text())
    # Required fields fail explicitly if pointed at a static or stale-schema run.
    outstanding(value)
    return value


def aborted_terminal(path, internal_id, *, after):
    matches = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    matches = [row for row in matches if row['request_id'] == internal_id]
    if len(matches) != 1 or matches[0]['status'] != 'FINISHED_ABORTED' or matches[0]['at'] < after:
        raise ValueError('Native terminal trace did not confirm the live request was aborted')
    return matches[0]


async def stream_request(client, url, body, state, emit):
    try:
        async with client.stream('POST', url + '/v1/completions', json={**body,
                'request_id': state.base_id, 'stream': True,
                'stream_options': {'include_usage': True}},
                headers={'X-Request-Id': state.base_id}) as response:
            response.raise_for_status()
            state.opened = True
            emit('http_open', base_id=state.base_id, headers=dict(response.headers))
            async for line in response.aiter_lines():
                if not line.startswith('data:'): continue
                data = line[5:].strip()
                if data == '[DONE]':
                    state.completed = True
                    break
                packet = json.loads(data)
                emit('sse', base_id=state.base_id, packet=packet)
                if 'error' in packet: raise RuntimeError(packet['error'])
                if packet['id'] != f'cmpl-{state.base_id}':
                    raise ValueError('Server did not preserve explicit completion request identity')
                state.response_id = packet['id']
                for choice in packet['choices']:
                    if choice['index'] != 0: raise ValueError('Unexpected completion choice')
                    state.token_ids.extend(choice.get('token_ids') or [])
                    if choice.get('finish_reason') is not None:
                        state.finish_reason = choice['finish_reason']
            if not state.completed or state.finish_reason is None:
                raise RuntimeError('Server closed stream without an explicit completion')
    except asyncio.CancelledError:
        if not state.cancel_requested:
            raise
        state.cancelled = True
    finally:
        state.closed_at = time.monotonic()
        emit('http_closed', base_id=state.base_id, cancelled=state.cancelled,
             completed=state.completed, finish_reason=state.finish_reason,
             tokens_received=len(state.token_ids))


async def wait_drained(path, evidence, *, after):
    while True:
        snapshot = load_snapshot(path)
        if snapshot['at'] >= after and not outstanding(snapshot):
            evidence.emit('resources_drained', snapshot=snapshot)
            return snapshot
        await asyncio.sleep(0.01)


async def completion(client, url, body, base_id, evidence):
    response = await client.post(url + '/v1/completions', json={**body, 'request_id': base_id},
                                 headers={'X-Request-Id': base_id})
    response.raise_for_status()
    packet = response.json()
    evidence.emit('completion', base_id=base_id, packet=packet)
    if packet['id'] != f'cmpl-{base_id}': raise ValueError('Completion identity mismatch')
    choice = packet['choices'][0]
    tokens = choice.get('token_ids')
    if (choice['finish_reason'] != 'length' or not isinstance(tokens, list) or
            len(tokens) != body['max_tokens'] or packet['usage']['completion_tokens'] != body['max_tokens']):
        raise ValueError('Completion did not produce the exact requested token count')
    return tokens


async def run(args, evidence):
    url = args.url.rstrip('/')
    path = args.capture_dir / 'scheduler_resources.json'
    start = load_snapshot(path)
    if outstanding(start): raise ValueError('Test requires an idle server with zero retained ownership')
    prefix = 'swap-stress-' + uuid.uuid4().hex
    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=8)) as client:
        model = args.model
        if model is None:
            response = await client.get(url + '/v1/models'); response.raise_for_status()
            model = response.json()['data'][0]['id']
        response = await client.post(url + '/tokenize', json={'model': model,
            'prompt': ' A quiet river flows past the village.', 'add_special_tokens': False})
        response.raise_for_status()
        unit = response.json()['tokens']
        if not unit: raise ValueError('Tokenizer returned no prompt tokens')
        prompt = (unit * ((args.prompt_tokens + len(unit) - 1) // len(unit)))[:args.prompt_tokens]
        body = dict(model=model, prompt=prompt, max_tokens=args.output_tokens,
                    ignore_eos=True, temperature=0, seed=args.seed, return_token_ids=True)
        evidence.emit('workload', body=body, concurrency=4, initial_snapshot=start)
        began = time.perf_counter()
        reference = await completion(client, url, body, prefix + '-reference', evidence)
        await wait_drained(path, evidence, after=began)
        streams = [Stream(f'{prefix}-{i}') for i in range(4)]
        tasks = [asyncio.create_task(stream_request(client, url, body, state, evidence.emit)) for state in streams]
        cancelled = None
        try:
            while cancelled is None:
                for task in tasks:
                    if task.done() and not task.cancelled() and task.exception(): raise task.exception()
                snapshot = load_snapshot(path)
                candidate = cancellation_candidate(snapshot, streams)
                if candidate is not None:
                    state, internal_id, swap = candidate
                    if tasks[streams.index(state)].done(): continue
                    # No await intervenes between this live state check and the
                    # client cancellation. The observed scheduler receipt is saved.
                    state.cancel_requested = True
                    cancel_at = time.perf_counter()
                    evidence.emit('cancel_live_swap', base_id=state.base_id, internal_id=internal_id,
                        response_id=state.response_id, swap=swap, snapshot=snapshot,
                        completed=state.completed, finish_reason=state.finish_reason,
                        received_tokens=len(state.token_ids))
                    task = tasks[streams.index(state)]
                    task.cancel()
                    await task
                    if not state.cancelled or state.completed or state.finish_reason is not None:
                        raise ValueError('Connection was not cancelled during a live request')
                    cancelled = state
                elif all(task.done() for task in tasks):
                    raise RuntimeError('No live swap observed; workload did not qualify cancellation')
                else:
                    await asyncio.sleep(0.002)
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        peers = [s for s in streams if s is not cancelled]
        if any(not s.completed or s.finish_reason != 'length' or s.token_ids != reference for s in peers):
            raise ValueError('Surviving peer token IDs or lengths differ from the isolated reference')
        drained = await wait_drained(path, evidence, after=cancel_at)
        terminal = aborted_terminal(args.capture_dir / 'scheduler_finished.jsonl', internal_id, after=cancel_at)
        evidence.emit('native_abort_confirmed', terminal=terminal)
        began = time.perf_counter()
        fresh = await completion(client, url, body, prefix + '-fresh', evidence)
        if fresh != reference: raise ValueError('Fresh request differs after swap cancellation')
        final = await wait_drained(path, evidence, after=began)
        return dict(status='passed', scope='finite live-swap cancellation, exact greedy outputs, fenced reclamation',
            cancelled=vars(cancelled), native_abort=terminal,
            peers=[vars(s) for s in peers], final_snapshot=final,
            prompt_tokens=args.prompt_tokens, output_tokens=args.output_tokens, seed=args.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model')
    parser.add_argument('--capture-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.output_tokens, args.timeout) <= 0: parser.error('Positive extents/timeout required')
    evidence = Evidence(args.output)
    async def bounded():
        async with asyncio.timeout(args.timeout): return await run(args, evidence)
    try:
        result = asyncio.run(bounded())
    except BaseException as error:
        evidence.result(dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise
    else:
        evidence.result(result)
        print(json.dumps(dict(status=result['status'], output=str(args.output))))
    finally:
        evidence.file.close()


if __name__ == '__main__': main()
