"""Finite exact-session/swap qualification against an already running server.

Requires eager incremental admission, synchronous PM sessions, terminal capture,
four session slots, and KVMEM_LIFECYCLE_TRACE=1. Never launches a server. A run
without an observed paused cancellation fails rather than claiming coverage.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import httpx

from stress_request_swaps import (Evidence, Stream, aborted_terminal, cancellation_candidate,
    completion, internal_id_matches, load_snapshot, outstanding, stream_request, wait_drained)


REPEAT_SENTENCE = ' A quiet river flows past the village.'


def prompt_layout(workload, size, block, unit, question):
    if size <= 0 or block <= 0 or not unit:
        raise ValueError('Prompt extent and tokenized source must be positive')
    if workload == 'repeat':
        prompt = (unit * ((size + len(unit) - 1) // len(unit)))[:size]
        query_begin = max(0, size - len(unit)) // block * block
    elif workload == 'code':
        if not question or len(question) >= size: raise ValueError('Prompt extent is too small')
        prompt = (unit * (size // len(unit) + 1))[:size-len(question)] + question
        query_begin = (size - len(question)) // block * block
    else:
        raise ValueError('Unknown session workload')
    return prompt, query_begin


def records(path):
    if not path.exists(): return []
    text = path.read_text()
    lines = text.splitlines()
    if text and not text.endswith('\n'): lines = lines[:-1]
    return [json.loads(line) for line in lines if line]


def matching(rows, base_id):
    return [row for row in rows if internal_id_matches(base_id, row.get('request_id', ''))]


def one(rows, label):
    if len(rows) != 1: raise ValueError(f'{label}: expected one exact request receipt, got {len(rows)}')
    return rows[0]


def active_resources(snapshot):
    # Idle owned P/M and their CPU history are intentionally retained between
    # calls. Everything else, including swap checkpoint credit, must drain.
    allowed = {'history_leases', 'history_reserved_bytes'}
    result = {key: value for key, value in outstanding(snapshot).items() if key not in allowed}
    owners = [entry for entry in snapshot['session_owners'] if entry['phase'] != 'idle']
    if owners: result['nonidle_sessions'] = owners
    return result


async def settled(probe, evidence, *, after):
    while True:
        snapshot = load_snapshot(probe / 'scheduler_resources.json')
        if snapshot['at'] >= after and not active_resources(snapshot):
            archive = json.loads((probe / 'archive.json').read_text())
            if not any(archive.get(name, 0) for name in
                       ('active_requests', 'retiring_requests', 'session_retiring_requests', 'pending_downloads')):
                evidence.emit('sessions_idle', snapshot=snapshot, archive=archive)
                return snapshot, archive
        await asyncio.sleep(.01)


def restore_proof(probe, base_id, session, kind):
    proof = one(matching(records(probe / 'sync_restore_timing.jsonl'), base_id), 'session import')
    if (proof.get('checkpoint_kind') != kind or proof.get('history_tokens', 0) <= 0 or
            proof.get('namespace') != [session, session] or proof.get('generation', 0) <= 0):
        raise ValueError('Session imported the wrong P/M checkpoint or namespace')
    return proof


def terminal_proof(archive, base_id, kind='M'):
    proof = one(matching(archive['completed'], base_id), 'retained terminal')
    if proof.get('checkpoint_kind') != kind or proof.get('checkpoint_tokens', 0) <= 0:
        raise ValueError('Request did not retain the expected model checkpoint')
    return proof


def serialization_proof(first_terminal, second_restore):
    retired = first_terminal.get('retirement_ack_ready_at')
    if retired is None or second_restore['started_at'] < retired:
        raise ValueError('Same-session successor imported before predecessor retirement acknowledgement')


def cancel_restore_proof(transitions, internal_id, history):
    rows = [row for row in transitions if row['request_id'] == internal_id and row['history_end'] == history]
    actions = [row['action'] for row in rows]
    if actions != ['capture', 'restore', 'drop']:
        raise ValueError(f'Cancelled session did not restore exact pages before drop: {actions}')
    return rows


def worker_checkpoint_outstanding(report):
    worker = report['request_checkpoints']
    ledger = worker['resources']
    resources = ledger['resources']['checkpoint_bytes']
    busy = {key: resources[key] for key in ('live', 'promised') if resources[key]}
    if ledger['active']: busy['active_grants'] = ledger['active']
    if worker['requests']: busy['requests'] = worker['requests']
    storage = worker['storage']
    if storage is not None:
        for key in ('entries', 'cpu_resident_bytes', 'cpu_charged_bytes', 'cpu_promised_bytes',
                    'logical_bytes', 'ssd_used_bytes', 'ssd_pending_write_bytes', 'io_pending'):
            if storage[key]: busy['storage_' + key] = storage[key]
    for key in ('archive_io_pending_bytes', 'cpu_authority_bytes', 'logical_allocated_bytes',
                'ssd_used_bytes', 'ssd_reserved_bytes', 'ssd_retired_pending_bytes', 'ssd_file_bytes'):
        if report.get(key, 0): busy['history_' + key] = report[key]
    return busy


def checkpoint_storage_proof(initial, final):
    if worker_checkpoint_outstanding(final):
        raise ValueError('Worker retained checkpoint or archive ownership after explicit release')
    before = initial['request_checkpoints']['storage']
    after = final['request_checkpoints']['storage']
    if (before is None) != (after is None):
        raise ValueError('Checkpoint storage mode changed during qualification')
    if after is None:
        return dict(storage='cpu', initial=before, final=after, deltas={})
    keys = ('disk_write_bytes', 'disk_read_bytes', 'spills', 'loads', 'io_failures')
    deltas = {key: after[key] - before[key] for key in keys}
    if any(value < 0 for value in deltas.values()) or deltas['io_failures']:
        raise ValueError('Checkpoint storage counters reset or I/O failed during session flow')
    if any(deltas[key] <= 0 for key in ('disk_write_bytes', 'disk_read_bytes', 'spills', 'loads')):
        raise ValueError('SSD checkpoint session qualification requires actual spill and restore reads')
    return dict(storage='ssd', initial=before, final=after, deltas=deltas)


async def wait_worker_drained(probe, evidence, *, after):
    while True:
        report = json.loads((probe / 'archive_resources.json').read_text())
        if report['at'] >= after and not worker_checkpoint_outstanding(report):
            evidence.emit('worker_checkpoint_storage_drained', report=report)
            return report
        await asyncio.sleep(.01)


async def run(args, evidence):
    url, probe = args.url.rstrip('/'), args.probe
    initial = load_snapshot(probe / 'scheduler_resources.json')
    if outstanding(initial): raise ValueError('Qualification requires an idle empty server')
    initial_worker = json.loads((probe / 'archive_resources.json').read_text())
    if worker_checkpoint_outstanding(initial_worker):
        raise ValueError('Qualification requires initially drained worker checkpoints and archive')
    evidence.emit('initial_checkpoint_storage', report=initial_worker)
    prefix = 'session-swap-' + uuid.uuid4().hex
    sessions = [prefix + '-ref'] + [prefix + f'-owner-{i}' for i in range(4)]
    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=8)) as client:
        async def tokenize(text):
            response = await client.post(url + '/tokenize', json=dict(model=args.model,
                prompt=text, add_special_tokens=False))
            response.raise_for_status()
            return response.json()['tokens']
        config = json.loads((probe / 'archive.json').read_text())['memory_config']
        block = config['block_tokens']
        if args.workload == 'repeat':
            unit = await tokenize(REPEAT_SENTENCE)
            question, appendix = [], unit
        else:
            facts = ''.join(f'Library maintenance note {i}: preserve the public API, use stable sorting, '
                            'check empty inputs, and document deterministic behavior.\n' for i in range(256))
            unit = await tokenize(facts)
            question = await tokenize('\nWrite a short Python function that merges overlapping intervals.\n')
            appendix = await tokenize('\nNow explain how this implementation handles empty input and equal endpoints.\n')
        prompt, query_begin = prompt_layout(args.workload, args.prompt_tokens, block, unit, question)
        def body(tokens, session=None, output=None):
            result = dict(model=args.model, prompt=tokens, max_tokens=output or args.output_tokens,
                ignore_eos=True, temperature=0, seed=args.seed, return_token_ids=True,
                kvmem_reselect='auto', kvmem_prefill_window='pressure',
                kvmem_query_span=[query_begin, len(prompt)], kvmem_query_id='swap-qualification-query',
                kvmem_mandatory_spans=[dict(begin=query_begin, end=len(prompt), kind='query')])
            if session is not None: result.update(session_id=session, cache_salt=session)
            return result
        evidence.emit('session_workload', prompt_ids=prompt, appendix_ids=appendix,
            output_tokens=args.output_tokens, seed=args.seed, namespaces=sessions,
            workload=args.workload, request_body=body(prompt))
        async def call(name, tokens, session, expected=None, imported=None):
            base = prefix + '-' + name
            started = time.perf_counter()
            output = await completion(client, url, body(tokens, session), base, evidence)
            _, archive = await settled(probe, evidence, after=started)
            terminal = terminal_proof(archive, base)
            proof = restore_proof(probe, base, session, imported) if imported else None
            if expected is not None and output != expected: raise ValueError(name + ': exact tokens differ from isolated session')
            evidence.emit('session_call_verified', name=name, base_id=base, terminal=terminal, restore=proof)
            return output, terminal, proof
        async def release(session, tag):
            payload = body([prompt[0]], session, 1)
            for key in ('kvmem_query_span', 'kvmem_query_id', 'kvmem_mandatory_spans'):
                payload.pop(key)
            payload['kvmem_reselect'] = 'off'
            payload['kv_transfer_params'] = {'kvmem_release_session': True}
            await completion(client, url, payload, prefix + '-release-' + tag, evidence)

        # Isolate the same P/M lineage used by the concurrent arms, rather than
        # comparing a rebuilt prefix to a different cold execution path.
        reference, _, _ = await call('reference-cold', prompt, sessions[0])
        continuation = prompt + reference + appendix
        expected_m, _, _ = await call('reference-m', continuation, sessions[0], imported='M')
        expected_duplicate, _, _ = await call('reference-duplicate', continuation, sessions[0], imported='P')
        expected_p, _, _ = await call('reference-p', prompt, sessions[0], imported='P')
        began = time.perf_counter()
        await release(sessions[0], 'reference')
        await wait_drained(probe / 'scheduler_resources.json', evidence, after=began)

        owners = sessions[1:]
        streams = [Stream(prefix + f'-cold-{i}') for i in range(4)]
        tasks = [asyncio.create_task(stream_request(client, url, body(prompt, session), state, evidence.emit))
                 for session, state in zip(owners, streams)]
        transitions, seen = [], set()
        cancelled = None
        cancel_receipt = None
        def observe(snapshot):
            for row in snapshot['request_swaps']['recent_transitions']:
                key = (row['step'], row['request_id'], row['action'], row['history_end'])
                if key not in seen:
                    seen.add(key); transitions.append(row)
                    evidence.emit('swap_transition', transition=row)
        try:
            while not all(task.done() for task in tasks):
                for task in tasks:
                    if task.done() and not task.cancelled() and task.exception(): raise task.exception()
                snapshot = load_snapshot(probe / 'scheduler_resources.json')
                observe(snapshot)
                if cancelled is None:
                    paused = {rid: row for rid, row in snapshot['request_swaps']['requests'].items()
                              if row['phase'] == 'paused' and row.get('session_bound')}
                    selection = {**snapshot, 'request_swaps': {**snapshot['request_swaps'], 'requests': paused}}
                    candidate = cancellation_candidate(selection, streams)
                    if candidate is not None:
                        state, internal_id, swap = candidate
                        task = tasks[streams.index(state)]
                        if task.done(): continue
                        state.cancel_requested = True
                        cancel_at = time.perf_counter()
                        cancel_receipt = dict(internal_id=internal_id, history_end=swap['history_end'],
                            namespace=owners[streams.index(state)], observed_at=cancel_at, snapshot=snapshot)
                        evidence.emit('cancel_paused_session', base_id=state.base_id, **cancel_receipt)
                        task.cancel(); await task
                        if not state.cancelled or state.completed or state.finish_reason is not None:
                            raise ValueError('Paused session connection was already complete')
                        cancelled = state
                await asyncio.sleep(.002)
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        if cancelled is None: raise ValueError('No paused owned session observed; cancellation is unqualified')
        for state in streams:
            if state is not cancelled and (state.token_ids != reference or state.finish_reason != 'length'):
                raise ValueError('Concurrent cold session differs from isolated output')
        snapshot, archive = await settled(probe, evidence, after=cancel_at)
        observe(snapshot)
        aborted = aborted_terminal(probe / 'scheduler_finished.jsonl', internal_id, after=cancel_at)
        restoration = cancel_restore_proof(transitions, internal_id, cancel_receipt['history_end'])
        evidence.emit('session_abort_restored', terminal=aborted, transitions=restoration)
        # Cancellation may happen before P exists. Rewarm the same namespace
        # with the original prefix; either a proven P hit or a fresh cold turn
        # must reproduce the isolated output and retain M.
        cancel_index = streams.index(cancelled)
        await call('recovered', prompt, owners[cancel_index], expected=reference)

        # Four M imports plus a same-namespace successor queued while owner 0
        # is visibly active. Its identical input falls back to original P.
        main = [Stream(prefix + f'-m-{i}') for i in range(4)]
        began = time.perf_counter()
        main_tasks = [asyncio.create_task(stream_request(client, url, body(continuation, session), state, evidence.emit))
                      for session, state in zip(owners, main)]
        follower = Stream(prefix + '-serialized')
        follower_task = None
        try:
            while follower_task is None:
                for task in main_tasks:
                    if task.done() and not task.cancelled() and task.exception(): raise task.exception()
                snapshot = load_snapshot(probe / 'scheduler_resources.json')
                active = [row for row in snapshot['session_owners'] if row['namespace'] == [owners[0], owners[0]]
                          and row['phase'] == 'active' and internal_id_matches(main[0].base_id, row['request_id'])]
                if active and not main_tasks[0].done():
                    evidence.emit('same_session_follower_queued', predecessor=main[0].base_id,
                                  successor=follower.base_id, owner=active[0], snapshot=snapshot)
                    follower_task = asyncio.create_task(stream_request(client, url,
                        body(continuation, owners[0]), follower, evidence.emit))
                elif main_tasks[0].done(): raise ValueError('Could not queue successor while predecessor was active')
                else: await asyncio.sleep(.002)
            await asyncio.gather(*main_tasks, follower_task)
        finally:
            cleanup = main_tasks + ([follower_task] if follower_task is not None else [])
            for task in cleanup:
                if not task.done(): task.cancel()
            await asyncio.gather(*cleanup, return_exceptions=True)
        if any(row.token_ids != expected_m or row.finish_reason != 'length' for row in main):
            raise ValueError('Concurrent M-import output differs from isolated M continuation')
        if follower.token_ids != expected_duplicate or follower.finish_reason != 'length':
            raise ValueError('Serialized same-session output differs from isolated P fallback')
        _, archive = await settled(probe, evidence, after=began)
        imports = [restore_proof(probe, row.base_id, session, 'M') for row, session in zip(main, owners)]
        follower_restore = restore_proof(probe, follower.base_id, owners[0], 'P')
        predecessor = terminal_proof(archive, main[0].base_id)
        serialization_proof(predecessor, follower_restore)
        evidence.emit('serialized_import_verified', imports=imports,
                      predecessor_terminal=predecessor, successor_restore=follower_restore)
        # Explicitly re-enter P after the longer completed trajectory.
        await call('concurrent-p', prompt, owners[0], expected=expected_p, imported='P')
        began = time.perf_counter()
        await asyncio.gather(*(release(session, str(i)) for i, session in enumerate(owners)))
        final = await wait_drained(probe / 'scheduler_resources.json', evidence, after=began)
        archive = json.loads((probe / 'archive.json').read_text())
        for name in ('session_checkpoint_payload_bytes', 'session_checkpoints', 'session_retiring_requests',
                     'active_requests', 'retiring_requests', 'archive_used_slots', 'pending_downloads'):
            if archive.get(name, 0): raise ValueError('Explicit session release retained ' + name)
        began = time.perf_counter()
        fresh = await completion(client, url, body(prompt), prefix + '-fresh', evidence)
        if fresh != reference: raise ValueError('Fresh request changed after session cleanup')
        final = await wait_drained(probe / 'scheduler_resources.json', evidence, after=began)
        final_worker = await wait_worker_drained(probe, evidence, after=began)
        storage_proof = checkpoint_storage_proof(initial_worker, final_worker)
        evidence.emit('checkpoint_storage_verified', proof=storage_proof)
        return dict(status='passed', scope='owned PM session exact outputs, serialized namespace import, paused cancel restore-first, explicit release',
            workload=args.workload, prompt_tokens=len(prompt), output_tokens=args.output_tokens, seed=args.seed,
            cancellation=cancel_receipt, native_abort=aborted, restore_first=restoration,
            serialized_predecessor=predecessor, serialized_successor=follower_restore,
            final_snapshot=final, final_worker_resources=final_worker,
            checkpoint_storage=storage_proof, namespaces=sessions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=900)
    parser.add_argument('--prompt-tokens', type=int, default=8192)
    parser.add_argument('--output-tokens', type=int, default=256)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--workload', choices=('code', 'repeat'), default='code')
    args = parser.parse_args()
    if min(args.timeout, args.prompt_tokens, args.output_tokens) <= 0: parser.error('Positive bounds required')
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
        print(json.dumps(dict(status='passed', output=str(args.output))))
    finally:
        evidence.file.close()


if __name__ == '__main__': main()
