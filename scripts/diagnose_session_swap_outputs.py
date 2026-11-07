"""Replay a failed session workload without intentional cancellation.

Run against an already running server. Two fresh isolated cold sessions and
four fresh concurrent cold sessions preserve the failed run's exact prompt,
query plan, sampling settings and output length. Output differences remain
failures, including when the observed waveform contains no request swaps.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

from qualify_session_swaps import (one, records, settled,
    terminal_proof, wait_worker_drained, worker_checkpoint_outstanding)
from stress_request_swaps import (Evidence, Stream, completion,
    load_snapshot, outstanding, stream_request, wait_drained)


QUESTION = '\nWrite a short Python function that merges overlapping intervals.\n'


def source_workload(path):
    rows = records(path)
    workload = one([row for row in rows if row['kind'] == 'session_workload'], 'source workload')
    reference = one([row for row in rows if row['kind'] == 'completion' and
        row['base_id'].endswith('-reference-cold')], 'source cold reference')
    packet = reference['packet']
    choice = one(packet['choices'], 'source completion choice')
    tokens = choice.get('token_ids')
    if (packet['id'] != 'cmpl-' + reference['base_id'] or choice['finish_reason'] != 'length' or
            not isinstance(tokens, list) or len(tokens) != workload['output_tokens'] or
            packet['usage']['completion_tokens'] != len(tokens)):
        raise ValueError('Source reference is not an exact, complete public token sequence')
    return workload, reference


def request_body(workload, model, question_tokens, block_tokens):
    prompt = workload['prompt_ids']
    if (not question_tokens or len(prompt) <= len(question_tokens) or block_tokens <= 0 or
            prompt[-len(question_tokens):] != question_tokens):
        raise ValueError('Current tokenizer cannot reconstruct the exact original query span')
    begin = (len(prompt) - len(question_tokens)) // block_tokens * block_tokens
    return dict(model=model, prompt=prompt, max_tokens=workload['output_tokens'],
        ignore_eos=True, temperature=0, seed=workload['seed'], return_token_ids=True,
        kvmem_reselect='auto', kvmem_prefill_window='pressure',
        kvmem_query_span=[begin, len(prompt)], kvmem_query_id='swap-qualification-query',
        kvmem_mandatory_spans=[dict(begin=begin, end=len(prompt), kind='query')])


def token_comparison(expected, actual):
    difference = next((i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]), None)
    if difference is None and len(expected) != len(actual):
        difference = min(len(expected), len(actual))
    start = max(0, (difference or 0) - 4)
    end = (difference or 0) + 8
    return dict(exact=expected == actual, expected_count=len(expected), actual_count=len(actual),
        first_difference=difference, expected_context=expected[start:end], actual_context=actual[start:end])


def compare_outputs(reference, isolated, concurrent):
    return dict(isolated_against_source=[token_comparison(reference, row) for row in isolated],
        isolated_repeat=token_comparison(isolated[0], isolated[1]),
        concurrent_against_source=[token_comparison(reference, row) for row in concurrent],
        concurrent_against_isolated=[[token_comparison(base, row) for row in concurrent] for base in isolated])


def all_exact(comparisons):
    return (comparisons['isolated_repeat']['exact'] and
        all(row['exact'] for key in ('isolated_against_source', 'concurrent_against_source')
            for row in comparisons[key]) and
        all(row['exact'] for baseline in comparisons['concurrent_against_isolated'] for row in baseline))


def swap_counts(snapshot):
    return {name: snapshot['request_swaps'][name]
            for name in ('capture_submissions', 'restore_submissions', 'pauses', 'restores', 'drops')}


async def run(args, evidence):
    probe, url = args.probe, args.url.rstrip('/')
    initial = load_snapshot(probe / 'scheduler_resources.json')
    initial_worker = json.loads((probe / 'archive_resources.json').read_text())
    if outstanding(initial) or worker_checkpoint_outstanding(initial_worker):
        raise ValueError('Diagnostic requires an idle empty server')
    workload, source = source_workload(args.source_events)
    prefix = 'session-diagnostic-' + uuid.uuid4().hex
    sessions = [prefix + f'-isolated-{i}' for i in range(2)] + [prefix + f'-owner-{i}' for i in range(4)]
    acquired, released, transitions, seen = [], [], [], set()
    outputs, terminals = {}, {}
    stop = asyncio.Event()
    async def observe():
        while not stop.is_set():
            snapshot = load_snapshot(probe / 'scheduler_resources.json')
            for row in snapshot['request_swaps']['recent_transitions']:
                if prefix not in row.get('request_id', ''): continue
                key = (row['step'], row['request_id'], row['action'], row['history_end'])
                if key not in seen:
                    seen.add(key); transitions.append(row)
                    evidence.emit('swap_transition', transition=row)
            try: await asyncio.wait_for(stop.wait(), timeout=.01)
            except TimeoutError: pass
    async with httpx.AsyncClient(timeout=args.timeout, limits=httpx.Limits(max_connections=8)) as client:
        response = await client.post(url + '/tokenize', json=dict(model=args.model, prompt=QUESTION,
                                                                  add_special_tokens=False))
        response.raise_for_status()
        archive = json.loads((probe / 'archive.json').read_text())
        body = request_body(workload, args.model, response.json()['tokens'], archive['memory_config']['block_tokens'])
        evidence.emit('diagnostic_workload', body=body, namespaces=sessions,
            source_events=str(args.source_events.resolve()),
            source_sha256=hashlib.sha256(args.source_events.read_bytes()).hexdigest(),
            source_reference=source, initial_snapshot=initial, initial_worker=initial_worker)
        def owned(session): return {**body, 'session_id': session, 'cache_salt': session}
        async def release(session):
            payload = dict(model=args.model, prompt=[body['prompt'][0]], max_tokens=1,
                ignore_eos=True, temperature=0, seed=body['seed'], return_token_ids=True,
                kvmem_reselect='off', kvmem_prefill_window='pressure',
                session_id=session, cache_salt=session,
                kv_transfer_params={'kvmem_release_session': True})
            await completion(client, url, payload, session + '-release', evidence)
            released.append(session)
        observer = asyncio.create_task(observe())
        failure = None
        try:
            async with asyncio.timeout(args.timeout):
                for i, session in enumerate(sessions[:2]):
                    base = prefix + f'-isolated-{i}'
                    acquired.append(session)
                    began = time.perf_counter()
                    outputs[base] = await completion(client, url, owned(session), base, evidence)
                    _, archive = await settled(probe, evidence, after=began)
                    terminals[base] = terminal_proof(archive, base)
                    evidence.emit('isolated_complete', base_id=base, terminal=terminals[base])
                    began = time.perf_counter()
                    await release(session)
                    await wait_drained(probe / 'scheduler_resources.json', evidence, after=began)
                streams = [Stream(prefix + f'-concurrent-{i}') for i in range(4)]
                acquired.extend(sessions[2:])
                began = time.perf_counter()
                completed = await asyncio.gather(*(stream_request(client, url, owned(session), stream, evidence.emit)
                    for session, stream in zip(sessions[2:], streams)), return_exceptions=True)
                for stream in streams: outputs[stream.base_id] = stream.token_ids
                errors = [result for result in completed if isinstance(result, BaseException)]
                if errors: raise RuntimeError(f'Concurrent HTTP completion failed: {errors!r}')
                if any(not row.completed or row.finish_reason != 'length' or len(row.token_ids) != body['max_tokens']
                       for row in streams):
                    raise ValueError('Concurrent request did not return the exact requested output count')
                _, archive = await settled(probe, evidence, after=began)
                for stream in streams: terminals[stream.base_id] = terminal_proof(archive, stream.base_id)
        except BaseException as error:
            failure = error
        finally:
            # Differences never short-circuit the waveform. Complete all calls,
            # then release each namespace; timeout cleanup is separately bounded.
            try:
                async with asyncio.timeout(60):
                    began = time.perf_counter()
                    await asyncio.gather(*(release(session) for session in acquired if session not in released))
                    final = await wait_drained(probe / 'scheduler_resources.json', evidence, after=began)
                    final_worker = await wait_worker_drained(probe, evidence, after=began)
            except BaseException as cleanup_error:
                evidence.emit('cleanup_failed', error_type=type(cleanup_error).__name__, error=str(cleanup_error))
                if failure is None: failure = cleanup_error
            stop.set()
            observed = await asyncio.gather(observer, return_exceptions=True)
            if isinstance(observed[0], BaseException) and failure is None: failure = observed[0]
        receipts = dict(terminals=terminals,
            imports=[row for row in records(probe / 'sync_restore_timing.jsonl') if prefix in row['request_id']],
            digests=[row for row in records(probe / 'verified_restores_digest.jsonl') if prefix in row['request_id']])
        evidence.emit('diagnostic_receipts', **receipts)
        if failure is not None:
            result = dict(status='failed', error_type=type(failure).__name__, error=str(failure),
                source_reference_ids=source['packet']['choices'][0]['token_ids'],
                outputs=outputs, released_namespaces=released, swap_transitions=transitions, **receipts)
            if len(outputs) == 6:
                result['comparisons'] = compare_outputs(result['source_reference_ids'],
                    [outputs[prefix + f'-isolated-{i}'] for i in range(2)],
                    [outputs[prefix + f'-concurrent-{i}'] for i in range(4)])
            try:
                result['latest_snapshot'] = load_snapshot(probe / 'scheduler_resources.json')
                before, after = swap_counts(initial), swap_counts(result['latest_snapshot'])
                result['swap_count_deltas'] = {key: after[key] - before[key] for key in before}
                result['no_swaps_observed'] = not any(result['swap_count_deltas'].values())
                result['latest_worker_resources'] = json.loads((probe / 'archive_resources.json').read_text())
            except Exception as telemetry_error:
                result['telemetry_error'] = str(telemetry_error)
            evidence.result(result)
            return None
    isolated = [outputs[prefix + f'-isolated-{i}'] for i in range(2)]
    concurrent = [outputs[prefix + f'-concurrent-{i}'] for i in range(4)]
    reference = source['packet']['choices'][0]['token_ids']
    comparisons = compare_outputs(reference, isolated, concurrent)
    before, after = swap_counts(initial), swap_counts(final)
    deltas = {key: after[key] - before[key] for key in before}
    if any(value < 0 for value in deltas.values()): raise ValueError('Swap counters reset during diagnostic')
    return dict(status='passed' if all_exact(comparisons) else 'failed',
        error=None if all_exact(comparisons) else 'Exact public output sequences differ',
        scope='two isolated cold sessions and four concurrent cold sessions; no intentional cancellation',
        source_reference_ids=reference, outputs=outputs, comparisons=comparisons,
        swap_counts_initial=before, swap_counts_final=after, swap_count_deltas=deltas,
        no_swaps_observed=not any(deltas.values()), swap_transitions=transitions,
        namespaces=sessions, released_namespaces=released, final_snapshot=final,
        final_worker_resources=final_worker, **receipts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--source-events', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--timeout', type=float, default=600)
    args = parser.parse_args()
    if args.timeout <= 0: parser.error('Positive timeout required')
    evidence = Evidence(args.output)
    try:
        result = asyncio.run(run(args, evidence))
        if result is not None: evidence.result(result)
        saved = json.loads((args.output / 'result.json').read_text())
        print(json.dumps(dict(status=saved['status'], output=str(args.output))))
        if saved['status'] != 'passed': raise SystemExit(1)
    except Exception as error:
        evidence.result(dict(status='failed', error_type=type(error).__name__, error=str(error)))
        raise
    finally:
        evidence.file.close()


if __name__ == '__main__': main()
