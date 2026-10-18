"""Fixed-work decode/prefill interference probe; the caller owns the server.

T1 ends observation, never the requests. The fixed resident cohort and seed wave
must drain their entire quotas before cleanup. This is not an agent-quality run.
"""
import array
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wave_timed_workload', ROOT / 'scripts/timed_agent_workload.py')
timed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timed)


def token_bytes_sha256(tokens):
    value = array.array('I', tokens)
    if value.itemsize != 4:
        raise RuntimeError('Require 32-bit token representation')
    if sys.byteorder != 'little':
        value.byteswap()
    return hashlib.sha256(value.tobytes()).hexdigest()


def validate_fixture(fixture, args, profile):
    if fixture.get('format') != 'fixed_prefill_wave_v1':
        raise ValueError('Require a frozen fixed_prefill_wave_v1 fixture')
    residents, seeds = fixture['residents'], fixture['seeds']
    if len(residents) != args.resident_requests or len(seeds) != args.seed_requests:
        raise ValueError('Fixture cohort sizes differ from the frozen protocol')
    if type(fixture.get('eos')) is not int or fixture['eos'] < 0:
        raise ValueError('A cleanup EOS token is required')
    for role, entries, length, quota in (
            ('resident', residents, args.context_tokens, args.output_tokens),
            ('seed', seeds, args.seed_context_tokens, args.seed_output_tokens)):
        for item in entries:
            tokens = item['tokens']
            if (len(tokens) != length or any(type(n) is not int or not 0 <= n < 2**32 for n in tokens)
                    or item['output_tokens'] != quota or item['input_sha256'] != timed.digest(tokens)
                    or item['input_bytes_sha256'] != token_bytes_sha256(tokens)):
                raise ValueError('Frozen token IDs, length, quota or fingerprint changed')
            if role == 'resident' and (type(item.get('suffix_token')) is not int or
                                      not 0 <= item['suffix_token'] < 2**32):
                raise ValueError('Resident suffix must be a fixed token ID')
            if quota > profile['kvmem']['gen_budget']:
                raise ValueError('Fixed output quota exceeds gen_budget')
    if len({item['input_sha256'] for item in seeds}) != 1:
        raise ValueError('The seed wave must use identical frozen input IDs')


def prepare_fixture(source, args, profile):
    """Derive controlled prompts once; never splice arm-specific model outputs."""
    if source.get('format') == 'fixed_prefill_wave_v1':
        result = json.loads(json.dumps(source))
    else:
        plans = source.get('agent_plans')
        if plans is not None and len(plans) < args.concurrency:
            raise ValueError('Need a source trace for each resident and seed')
        def source_tokens(i):
            return plans[i]['initial_tokens'] if plans is not None else source['prompt']
        def item(i, length, count):
            full = source_tokens(i)
            if len(full) < length:
                raise ValueError('Source fixture is shorter than the fixed input')
            tokens = list(full[:length])
            return dict(tokens=tokens, output_tokens=count,
                        input_sha256=timed.digest(tokens), input_bytes_sha256=token_bytes_sha256(tokens)), full
        residents = []
        for i in range(args.resident_requests):
            row, full = item(i, args.context_tokens, args.output_tokens)
            row['suffix_token'] = full[args.context_tokens] if len(full) > args.context_tokens else full[-1]
            residents.append(row)
        seed, _ = item(args.resident_requests, args.seed_context_tokens, args.seed_output_tokens)
        result = dict(format='fixed_prefill_wave_v1', source_sha256=timed.digest(source),
                      scope='Controlled fixed-work interference probe, not original task-quality inference',
                      eos=source.get('eos'), residents=residents,
                      seeds=[json.loads(json.dumps(seed)) for _ in range(args.seed_requests)])
    validate_fixture(result, args, profile)
    return result


def intervals_union(intervals):
    merged = []
    for start, stop in sorted((a, b) for a, b in intervals if b > a):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def overlap_summary(residents, seeds, begin, end):
    """No double counting when several seed prefills overlap the same decode."""
    left = max([begin] + [r['timeline'][0][0] for r in residents])
    right = min([end] + [r['end'] for r in residents])
    spans = intervals_union((max(left, r['begin']), min(right, r['timeline'][0][0])) for r in seeds)
    seconds = sum(b-a for a, b in spans)
    count = sum(n for r in residents for t, n in r['timeline'] if any(a <= t < b for a, b in spans))
    return dict(intervals=spans, seconds=seconds, common_decode_seconds=max(0., right-left),
                resident_output_tokens=count, tokens_per_second=count/seconds if seconds else None,
                scope='Client overlap: every resident streaming while at least one seed awaits first token; '
                      'includes seed admission/replay time, not proof of per-step GPU coexecution')


def native_matches(row, native_id):
    response = row.get('response_id')
    return isinstance(response, str) and re.fullmatch(re.escape(response) + r'-0-[0-9a-f]{8}', native_id) is not None


def seed_progress(seed, policy, deadline):
    """Conservative per-request completion lower bounds from sparse native state.

    In-flight cursors are optimistic scheduler positions and cannot prove GPU
    completion. Missing samples retain [known complete, full prompt] bounds.
    """
    length = seed['input_tokens']
    lower = 0
    evidence = None
    for row in policy:
        if row['at'] >= deadline:
            continue
        for item in row.get('candidates', []):
            if (native_matches(seed, item['request_id']) and item.get('reason') is None
                    and item['phase'] in ('prefill', 'decode')):
                progress = length-item['remaining_tokens']
                if not 0 <= progress <= length:
                    raise ValueError('Seed scheduler cursor outside the fixed prompt')
                if progress >= lower:
                    lower, evidence = progress, dict(at=row['at'], phase=item['phase'], native_id=item['request_id'])
    first = seed['timeline'][0][0] if seed['timeline'] else None
    if first is not None and first < deadline:
        lower = length
        evidence = dict(first_token_observed_at=first)
    return dict(completed_tokens_lower_bound=lower, completed_tokens_upper_bound=length,
                evidence=evidence, exact=lower == length,
                scope='Confirmed non-inflight native cursor/first-token evidence; sparse observations, no interpolation')


def restore_evidence(residents, restores, fixture):
    reports = []
    for row, item in zip(sorted(residents, key=lambda r:r['agent']), fixture['residents']):
        matches = [r for r in restores if native_matches(row, r['request_id'])]
        expected = len(item['tokens'])
        exact = (len(matches) == 1 and matches[0]['history_tokens'] == expected
                 and matches[0]['active_tokens'] == expected
                 and matches[0]['resident_active_tokens'] == expected
                 and matches[0]['uploaded_bytes'] == 0 and not matches[0].get('cancelled', False))
        reports.append(dict(request_id=row['request_id'], matched_restores=matches, passed=exact,
            prefix_bytes_sha256=item['input_bytes_sha256'], expected_history_tokens=expected))
    return dict(passed=all(r['passed'] for r in reports), requests=reports,
                scope='Exact fixed input-prefix bytes plus full-history resident restore metadata; '
                      'KV byte equality is qualified separately, not inferred from these performance counters')


def mixed_steps(rows, begin, end, cap):
    selected = [r for r in rows if begin <= r['started_at'] < end and
                r.get('decode_requests', 0) and r.get('prefill_tokens', 0)]
    ordinary = [r for r in selected if r.get('ordinary_prefill_only')]
    return dict(mixed_steps=len(selected), ordinary_only_mixed_steps=len(ordinary),
        mixed_with_multiple_prefills=sum(r.get('prefill_requests', 0) > 1 for r in selected),
        maximum_ordinary_tokens=max((r['ordinary_prefill_tokens'] for r in ordinary), default=None),
        ordinary_steps_exceeding_cap=(sum(r['ordinary_prefill_tokens'] > cap for r in ordinary) if cap else None),
        unknown_classification_steps=sum(bool(r.get('unknown_prefill_tokens')) for r in selected),
        atomic_query_steps=sum(bool(r.get('atomic_prefill_tokens')) for r in selected),
        scope='Recorded public mixed steps only; atomic Query exempt, no interpolation of absent CUDA-event samples')


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def wait_public_complete(capture, expected, timeout=120.):
    """Require the unsampled completed public-prompt counter after retirement."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        try:
            row = json.loads((capture/'public_work.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            row = None
        if row is not None:
            submitted = row['submitted']['public_prefill_tokens']
            completed = row['completed']['public_prefill_tokens']
            if submitted > expected:
                raise RuntimeError('Unexpected public re-prefill: fixed prompt demand was exceeded')
            if completed == submitted == expected:
                return dict(passed=True, completed_tokens=completed, expected_tokens=expected,
                    snapshot=row, scope='All public prompt forwards since this fresh server started; '
                                        'warmup and fixed resident suffix included, cleanup excluded')
        await asyncio.sleep(.05)
    raise TimeoutError('Completed public prefill demand did not reconcile')


def summarize(rows, window, fixture, policy, restores, dispatch_begin, work_finished):
    result = timed.summarize(rows, window, [])
    residents = [r for r in rows if r['kind'] == 'workload' and r['source_metadata']['role'] == 'resident']
    seeds = [r for r in rows if r['kind'] == 'workload' and r['source_metadata']['role'] == 'seed']
    t0, t1 = window['begin'], window['end']
    seeds_report = []
    for row in seeds:
        first = row['timeline'][0][0]
        seeds_report.append(dict(request_id=row['request_id'], input_sha256=row['input_sha256'],
            input_tokens=row['input_tokens'], quota=len(row['output_ids']),
            planned_dispatch=row['source_metadata']['planned_dispatch'], actual_dispatch=row['begin'],
            dispatch_lateness_seconds=row['begin']-row['source_metadata']['planned_dispatch'],
            full_ttft_seconds=first-row['begin'], finished=row['end'],
            ttft_observed_before_t1=first < t1,
            ttft_right_censored_at_least_seconds=max(0., t1-row['begin']) if first >= t1 else None,
            progress_at_t1=seed_progress(row, policy, t1)))
    t0_offsets = [dict(request_id=r['request_id'], agent=r['agent'],
        output_offset=sum(n for t, n in r['timeline'] if t < t0),
        accepted_context_tokens_at_t0=r['input_tokens']+sum(n for t, n in r['timeline'] if t < t0),
        input_sha256=r['input_sha256']) for r in residents]
    resident_gate = restore_evidence(residents, restores, fixture)
    seeds_dispatch = max(r['begin'] for r in seeds)
    coverage = all(r['end'] > seeds_dispatch and r['timeline'][0][0] <= t0 for r in residents)
    result.update(status='completed', drained=True, work_identity=timed.digest(fixture),
        fixed_work_output_tokens=sum(len(r['output_ids']) for r in residents+seeds),
        fixed_work_input_tokens=sum(r['input_tokens'] for r in residents+seeds),
        formal_dispatch_begin=dispatch_begin, full_work_completed_at=work_finished,
        full_work_makespan_seconds=work_finished-dispatch_begin,
        t0_to_work_complete_seconds=work_finished-t0,
        work_drain_after_t1_seconds=max(0., work_finished-t1),
        t0_resident_offsets=t0_offsets, seeds=seeds_report,
        decode_seed_overlap=overlap_summary(residents, seeds, t0, t1),
        full_decode_seed_overlap=overlap_summary(residents, seeds, dispatch_begin, work_finished),
        resident_restore=resident_gate,
        coverage_valid=coverage and resident_gate['passed'],
        resident_alive_at_all_seed_dispatches=coverage,
        seed_completed_tokens_lower_bound_at_t1=sum(s['progress_at_t1']['completed_tokens_lower_bound'] for s in seeds_report),
        seed_completed_tokens_upper_bound_at_t1=sum(s['progress_at_t1']['completed_tokens_upper_bound'] for s in seeds_report),
        scope='Identical finite inputs/quotas, fixed-offset arrivals; T1 never cancels unfinished work')
    return result


async def workload(client, profile, fixture, directory, args):
    validate_fixture(fixture, args, profile)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'events.jsonl').exists():
        raise FileExistsError('Use a fresh workload directory')
    capture = Path(args.capture_directory)
    recorder = timed.Recorder(directory, args.concurrency)
    tasks = []
    owned = []
    sessions = [f'wave-{args.run_tag}-{i}' for i in range(args.concurrency)]
    items = fixture['residents'] + fixture['seeds']
    settings = {key:value for key,value in vars(args).items() if not isinstance(value, Path)}
    timed.save(directory/'workload.json', dict(settings=settings, fixture_sha256=timed.digest(fixture)))

    async def call(i, tokens, count, kind, role, *, serial, planned=None):
        return await timed.stream_call(client, profile, tokens, sessions[i], count, recorder,
            agent=i, epoch=0, turn=serial, kind=kind,
            request_id=f'wave-{args.run_tag}-{i}-{serial}-{kind}',
            new_input_tokens=(1 if role == 'resident' and kind == 'workload' else len(tokens)),
            query=kind != 'cleanup', source_metadata=dict(role=role, planned_dispatch=planned))

    def spawn(coroutine):
        task = asyncio.create_task(coroutine)
        owned.append(task)
        return task

    try:
        warm = await asyncio.gather(*(spawn(call(i, item['tokens'], 1, 'warmup', 'resident', serial=0))
                                      for i, item in enumerate(fixture['residents'])))
        timed.save(directory/'resources-before.json', await timed.wait_retirement(capture,
            not_before=max(r['begin'] for r in warm),
            expected_views={sessions[i]:len(item['tokens']) for i,item in enumerate(fixture['residents'])}))
        # Warmup's final sampled token has no KV; the checkpoint is exactly the
        # original prefix. Use an explicit fixed suffix, never warmup output.
        prefix_proof = []
        dispatch_begin = time.monotonic()
        for i, item in enumerate(fixture['residents']):
            tokens = item['tokens'] + [item['suffix_token']]
            prefix_proof.append(dict(agent=i, warm_input_sha256=warm[i]['input_sha256'],
                formal_prefix_sha256=timed.digest(tokens[:-1]),
                formal_prefix_bytes_sha256=token_bytes_sha256(tokens[:-1]),
                equals_frozen_prefix=tokens[:-1] == item['tokens']))
            proof = prefix_proof[-1]
            if (not proof['equals_frozen_prefix'] or
                    proof['warm_input_sha256'] != proof['formal_prefix_sha256'] or
                    proof['formal_prefix_sha256'] != item['input_sha256'] or
                    proof['formal_prefix_bytes_sha256'] != item['input_bytes_sha256']):
                raise RuntimeError('Warmup and formal restore prefixes differ from the frozen input')
            tasks.append(spawn(call(i, tokens, item['output_tokens'], 'workload', 'resident', serial=1)))
        timed.save(directory/'prefix-proof.json', prefix_proof)
        while True:
            if any(t.done() for t in tasks):
                for task in tasks:
                    if task.done():
                        task.result()
                raise RuntimeError('Resident quota ended before the common first-token barrier')
            residents = [r for r in recorder.rows if r['kind'] == 'workload']
            if len(residents) == args.resident_requests and all(r['timeline'] for r in residents):
                break
            await asyncio.sleep(.005)
        t0, wall = time.monotonic(), time.time()
        recorder.window = dict(begin=t0, end=t0+args.duration_seconds, t0_monotonic=t0,
            t1_monotonic=t0+args.duration_seconds, t0_wall=wall, t1_wall=wall+args.duration_seconds,
            duration_seconds=args.duration_seconds, interval='[T0,T1)',
            formal_dispatch_begin=dispatch_begin, agents=args.concurrency,
            t0_definition='All resident formal requests have emitted at least one accepted token')
        timed.save(directory/'window.json', recorder.window)
        recorder.event('measurement_started', window=recorder.window)

        async def seed(i, item):
            planned = t0+args.seed_offset_seconds
            await asyncio.sleep(max(0., planned-time.monotonic()))
            return await call(i, item['tokens'], item['output_tokens'], 'workload', 'seed', serial=1, planned=planned)

        tasks.extend(spawn(seed(args.resident_requests+i, item))
                     for i,item in enumerate(fixture['seeds']))
        # No T1 cancellation: every originally offered quota must finish.
        await asyncio.gather(*tasks)
        work_finished = time.monotonic()
        recorder.event('specified_work_finished', formal_dispatch_begin=dispatch_begin)
        resources = await timed.wait_retirement(capture,
            not_before=max(r['begin'] for r in recorder.rows),
            expected_views={sessions[i]:len(item['tokens'])+(1 if i < args.resident_requests else 0)
                            +item['output_tokens']-1 for i,item in enumerate(items)})
        timed.save(directory/'resources-drained.json', resources)
        retired_at = time.monotonic()
        # Keep cleanup out of the declared measurement interval if all quotas
        # happen to finish early. Preserve the actual earlier retirement time.
        await asyncio.sleep(max(0., recorder.window['end']-time.monotonic()))
        result = summarize(recorder.rows, recorder.window, fixture,
            read_records(capture/'offline_decisions.jsonl'), read_records(capture/'async_restore.jsonl'),
            dispatch_begin, work_finished)
        result.update(full_work_retired_at=retired_at,
            prefill_cap_tokens=profile['kvmem']['prefill_batch_tokens'],
            full_work_retired_makespan_seconds=retired_at-dispatch_begin,
            retirement_after_output_seconds=retired_at-work_finished,
            retirement_scope='Actual accepted-output completion and published CPU/GPU ownership retirement',
            public_mixed_steps=mixed_steps(read_records(capture/'model_step_latency.jsonl'),
                t0, recorder.window['end'], profile['kvmem']['prefill_batch_tokens']))
        result['full_public_prefill'] = await wait_public_complete(capture,
            sum(len(item['tokens'])+1 for item in fixture['residents'])
            +sum(len(item['tokens']) for item in fixture['seeds']))
        cleanup_started = time.monotonic()
        await asyncio.gather(*(spawn(call(i, [fixture['eos']], 1, 'cleanup', 'cleanup', serial=2))
                               for i in range(args.concurrency)))
        timed.save(directory/'resources-after.json', await timed.wait_retirement(capture,
            not_before=cleanup_started, expected_views={session:1 for session in sessions}))
        # Include every stream in native/client reconciliation, but never count
        # cleanup as completed fixed work or normal decode output.
        result.update(total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            stream_output_tokens_by_kind={kind:sum(len(r['output_ids']) for r in recorder.rows if r['kind'] == kind)
                                          for kind in ('warmup', 'workload', 'cleanup')},
            cleanup_complete=True, cleanup_seconds=time.monotonic()-cleanup_started,
            cancelled_requests=[], finished_wall=time.time())
        timed.save(directory/'results.json', result)
        return result
    except BaseException as error:
        active = [r['request_id'] for r in recorder.rows if r['status'] == 'running']
        for task in owned:
            if not task.done():
                task.cancel()
        await asyncio.gather(*owned, return_exceptions=True)
        timed.save(directory/'results.json', dict(status='failed', drained=False, cleanup_complete=False,
            cancelled_requests=active, error=f'{type(error).__name__}: {error}', window=recorder.window,
            cleanup_scope='Owned server wrapper stops this failed arm; no successful retirement claim'))
        raise
    finally:
        timed.save(directory/'traces.json', recorder.rows)
        recorder.progress()
        recorder.stream.close()
