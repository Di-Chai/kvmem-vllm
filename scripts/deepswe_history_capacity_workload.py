"""Identical real-history capacity diagnostic, separate from live agent quality.

Prime eight full recorded histories, drain them, then offer eight identical
continuations together. Native vLLM owns admission/preemption. KVMem must retain
full CPU history and demonstrate a real eight-row GPU decode batch.
"""
import array
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/file)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


core = module('real_history_capacity_core', 'capacity_scaling_workload.py')
timed = core.timed
OUTPUT_TOKENS = 8192
SAMPLING = dict(temperature=.6, top_p=.95, top_k=20, min_p=0,
                presence_penalty=0, repetition_penalty=1, seed=1000, thinking_token_budget=4096)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_fixture(source_manifest, *, eos_token_id, warmup_tokens_path):
    source = json.loads(source_manifest.read_text())
    require(source['format'] == 'deepswe_capacity_history_v1' and len(source['samples']) == 8,
            'Require eight exact exported real DeepSWE histories')
    require(all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha
                for p, sha in source['source_sha256'].items()), 'Export source provenance changed')
    samples = []
    for row in source['samples']:
        raw = (source_manifest.parent/row['file']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == row['sha256'] and len(raw) % 4 == 0,
                'Exported token file changed')
        values = array.array('I')
        require(values.itemsize == 4, 'Require32-bit token encoding')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        require(all(row['original_sampling'].get(k) == SAMPLING[k] for k in
                    ('temperature', 'top_p', 'top_k', 'seed', 'thinking_token_budget')),
                'Recorded agent sampling differs from this protocol')
        samples.append(dict(row, id=f"{row['task']}-{row['turn']:05d}", input_ids=values.tolist()))
    raw = warmup_tokens_path.read_bytes()
    require(len(raw) == 67584*4, 'Use the same67584-token primary initialization input')
    values = array.array('I'); values.frombytes(raw)
    if sys.byteorder != 'little':
        values.byteswap()
    fixture = dict(format='deepswe_history_capacity_v1', capacity_probe_only=True,
        source_manifest=str(source_manifest.resolve()),
        source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        source_run=source['source_run'], samples=samples, sampling=SAMPLING,
        output_tokens=OUTPUT_TOKENS, generation_budget=32768, historical_ceiling=262144,
        eos=eos_token_id, warmup_ids=values.tolist(), warmup_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Real recorded full prompts, no crop/padding. Prime then resume; replay clients use independent salts. Fixed8192 output with ignore_eos, not live task scoring.')
    validate_fixture(fixture)
    return fixture


def validate_fixture(fixture):
    require(fixture.get('format') == 'deepswe_history_capacity_v1' and fixture.get('capacity_probe_only') is True
        and fixture.get('sampling') == SAMPLING and fixture.get('output_tokens') == OUTPUT_TOKENS
        and fixture.get('generation_budget') == 32768 and fixture.get('historical_ceiling') == 262144
        and type(fixture.get('eos')) is int and 0 <= fixture['eos'] < 2**31
        and len(fixture.get('samples', [])) == 8, 'Real-history protocol changed')
    ids, hashes = set(), set()
    for row in fixture['samples']:
        tokens = row['input_ids']; token = row['continuation_token_id']
        require(212992 <= len(tokens) == row['prompt_tokens'] and len(tokens)+1+OUTPUT_TOKENS <= 262144
            and all(type(t) is int and 0 <= t < 2**31 for t in tokens)
            and type(token) is int and 0 <= token < 2**31
            and core.token_bytes_sha256(tokens) == row['sha256'], 'Full real input or continuation changed')
        from vllm_kvmem.request_plan import RequestPlan
        RequestPlan.parse(row['request_plan'], len(tokens), 32, 32768)
        ids.add(row['id']); hashes.add(row['sha256'])
    require(len(ids) == len(hashes) == 8, 'Require eight distinct snapshots, no replicated prompts')
    require(len(fixture.get('warmup_ids', [])) == 67584 and
        core.token_bytes_sha256(fixture['warmup_ids']) == fixture['warmup_sha256'], 'Initialization input changed')


class SampledClient:
    def __init__(self, client):
        self.client = client

    def stream(self, method, url, **kwargs):
        # The legacy stream recorder remains unchanged. Only these frozen
        # sampling fields differ from its greedy diagnostic defaults.
        kwargs['json'] = dict(kwargs['json'], **SAMPLING)
        return self.client.stream(method, url, **kwargs)


async def warmup(client, profile, fixture, directory, args):
    validate_fixture(fixture)
    directory = Path(directory); directory.mkdir()
    path = directory/'input.i32'
    values = array.array('I', fixture['warmup_ids'])
    if sys.byteorder != 'little':
        values.byteswap()
    path.write_bytes(values.tobytes())
    first16 = module('real_history_primary_warmup', 'benchmark_deepswe_first16.py')
    options = SimpleNamespace(warmup_tokens=path, backend_kind='native' if profile['native_vllm'] else 'kvmem')
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    began = time.perf_counter()
    await asyncio.to_thread(first16.bounded_warmup, options, profile, directory.parent, http)
    proof = json.loads((directory.parent/'bounded-warmup.json').read_text())
    result = dict(status='completed', drained=True, cleanup_complete=True,
        warmup_begin=began, warmup_end=time.perf_counter(), input_tokens=67584,
        output_tokens=1, total_stream_output_tokens=2, kernel_load_proof=proof['kernel_load_proof'],
        scope='Exact same primary initialization input, independent salt; fully retired and excluded')
    timed.save(directory/'results.json', result)
    return result


def compression_evidence(prime, rows, resources, archive, geometry, refreshes, profile):
    reports = []
    limit = profile['kvmem']['select_budget']+profile['kvmem']['gen_budget']
    for old, row in zip(prime, rows):
        expected = row['input_tokens']+len(row['output_ids'])-1
        views = [v for v in resources['session_views'] if v['namespace'] == [row['session']]*2]
        completed = [r for r in archive['completed'] if core.native_matches(row, r['request_id'])]
        selections = [r for r in refreshes if (core.native_matches(old, r['request_id']) or
            core.native_matches(row, r['request_id'])) and r.get('selection') == 'pressure']
        passed = (len(views) == len(completed) == 1 and views[0]['history'] == expected
            and completed[0].get('history_end') == completed[0].get('checkpoint_tokens') == expected
            and completed[0].get('checkpoint_kind') == 'M' and 0 < views[0]['window'] <= limit < expected
            and selections and all(r.get('mean_k_backend') == 'qw3_cuda'
                and r.get('action_attention_enabled') is True and not r.get('cancelled')
                and 0 < r.get('selected_tokens', 0) <= 65536 < r.get('history_begin', 0)
                for r in selections))
        reports.append(dict(session=row['session'], passed=bool(passed), expected_history=expected,
            view=views[0] if len(views) == 1 else views, pressure_refreshes=len(selections)))
    record = 32*geometry.get('cpu_archive_bytes_per_token', 0)
    minimum = sum((r['input_tokens']+len(r['output_ids'])-1)//32*record for r in rows)
    storage = (record == 32*34816 and geometry.get('archive_codec') == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1'
        and geometry.get('select_budget') == 65536 and geometry.get('gen_budget') == 32768
        and geometry.get('cpu_offload_bytes') == 100_000_000_000
        and geometry.get('historical_tokens') == 262144 and archive.get('archive_backend') == 'cpu'
        and archive.get('repository_gc') is False and archive.get('cpu_authority_bytes', -1) >= minimum
        and archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0
        and not archive.get('pending_downloads') and not archive.get('session_retiring_requests'))
    return dict(passed=bool(storage and all(r['passed'] for r in reports)), requests=reports,
        cpu_storage_passed=storage, minimum_sealed_history_bytes=minimum,
        actual_cpu_authority_bytes=archive.get('cpu_authority_bytes'), geometry=geometry,
        scope='Full CPU publication plus compact active views after retirement; original Query/pins kept and pressure Action selection used. No byte oracle or task accuracy claim.')


async def workload(client, profile, fixture, directory, args):
    validate_fixture(fixture)
    require(args.offered_concurrency == args.admitted_concurrency == 8,
            'Offer all eight requests; native scheduler owns actual admission, no external5-user gate')
    native = profile['native_vllm']; directory = Path(directory); directory.mkdir()
    recorder = timed.Recorder(directory, 8); capture = Path(args.capture_directory)
    sessions = [f'real-history-{args.run_tag}-{i}' for i in range(8)]
    sampled = SampledClient(client); pending = []
    begin, wall = time.perf_counter(), time.time()
    try:
        before = await args.native_quiescent(stage='before_prime', not_before=begin)
        async def call(i, tokens, count, phase, plan):
            row = await timed.stream_call(sampled, profile, tokens, sessions[i], count, recorder,
                agent=i, epoch=0, turn=0 if phase == 'prime' else 1, kind=phase,
                request_id=f'real-history-{args.run_tag}-{i}-{phase}',
                new_input_tokens=len(tokens) if phase == 'prime' else 1, query=True,
                query_plan=plan, source_metadata=dict(sample_id=fixture['samples'][i]['id'],
                    source_input_sha256=fixture['samples'][i]['sha256']))
            require(row['usage'].get('prompt_tokens') == len(tokens),
                    'Server logical prompt count differs from the complete recorded input')
            return row
        pending = [asyncio.create_task(call(i, s['input_ids'], 1, 'prime', s['request_plan']))
                   for i, s in enumerate(fixture['samples'])]
        prime = await asyncio.gather(*pending)
        if not native:
            timed.save(directory/'resources-primed.json', await timed.wait_retirement(capture,
                not_before=begin, expected_views={r['session']:r['input_tokens'] for r in prime}))
        primed_at = time.perf_counter()
        await args.native_quiescent(stage='prime_drained', not_before=primed_at)
        prime_counts = await args.reconcile_outputs(args.snapshot, before['native_metrics'], 8,
            allow_preemptions=native, include_snapshot=True)
        resume_begin = time.perf_counter()
        # Prime's predicted token is uncomputed and absent from retained KV.
        # Supply the same recorded continuation token to both systems, even if
        # their prime predictions differed. The imported prefix is identical.
        pending = [asyncio.create_task(call(i, s['input_ids']+[s['continuation_token_id']],
                   OUTPUT_TOKENS, 'resume', s['request_plan'])) for i, s in enumerate(fixture['samples'])]
        rows = await asyncio.gather(*pending)
        end = time.perf_counter()
        require(all(len(r['output_ids']) == OUTPUT_TOKENS for r in rows), 'Resume quota incomplete')
        await args.native_quiescent(stage='resume_counters', not_before=end)
        resume_counts = await args.reconcile_outputs(args.snapshot, prime_counts['last_snapshot'],
            8*OUTPUT_TOKENS, allow_preemptions=native, include_snapshot=True)
        accounting = module('real_history_prefill_accounting', 'analyze_deepswe_capacity.py')
        computed = lambda snapshot: accounting.prefill_count(accounting.metrics(snapshot['raw_metrics']))
        prime_pf = computed(prime_counts['last_snapshot'])-computed(before['native_metrics'])
        resume_pf = computed(resume_counts['last_snapshot'])-computed(prime_counts['last_snapshot'])
        require(min(prime_pf, resume_pf) >= 0, 'Public prefill counter reset')
        result = dict(status='completed', drained=True, capacity_probe_only=True,
            workload_begin=begin, workload_end=end, resume_begin=resume_begin,
            offered_concurrency=8, admitted_concurrency=8, actual_output_quota=OUTPUT_TOKENS,
            fixed_work_makespan_seconds=end-begin, cold_prime_seconds=primed_at-begin,
            resume_seconds=end-resume_begin, output_tokens_per_second=8*OUTPUT_TOKENS/(end-resume_begin),
            cold_input_completion_rate_tokens_per_second=sum(r['input_tokens'] for r in prime)/
                (max(r['timeline'][0][0] for r in prime)-begin),
            prime_counter_reconciliation=prime_counts, fixed_work_output_tokens=8*OUTPUT_TOKENS,
            resume_counter_reconciliation=resume_counts,
            computed_public_prefill=dict(cold_prime_tokens=prime_pf, resume_tokens=resume_pf,
                cold_prime_tokens_per_second=prime_pf/(primed_at-begin),
                resume_tokens_per_second=resume_pf/(end-resume_begin),
                scope='Actual public computed prompt positions divided by phase wall time. Private Query replay and restored KV excluded; resume wall time includes decode and all interference.'),
            input_tokens_per_request=[r['input_tokens'] for r in prime],
            common_decode_window=core.decode_summary(rows, 8),
            client_decode_overlap_by_cohort={str(n):core.decode_summary(rows, n) for n in range(1,9)},
            ttft_seconds=[r['timeline'][0][0]-r['begin'] for r in rows],
            scope='Same real inputs and fixed continuation tokens. Cold prime measured separately. Resume rate includes native re-prefill, restore, queue and capacity interference; not a pure CUDA kernel rate. Fixed8192 outputs with ignore_eos, no live task quality claim.')
        if not native:
            state = await timed.wait_retirement(capture, not_before=resume_begin,
                expected_views={r['session']:r['input_tokens']+len(r['output_ids'])-1 for r in rows})
            archive = json.loads((capture/'archive.json').read_text())
            geometry = json.loads((capture/'cache-geometry.json').read_text())
            timed.save(directory/'resources-drained.json', state); timed.save(directory/'archive-drained.json', archive)
            result['compression'] = compression_evidence(prime, rows, state, archive, geometry,
                core.read_records(capture/'bounded_refresh.jsonl'), profile)
            result['gpu_decode_concurrency'] = core.gpu_decode_evidence(
                core.read_records(capture/'model_step_latency.jsonl'), result['common_decode_window'])
            cleanup_at = time.perf_counter()
            pending = [asyncio.create_task(timed.stream_call(sampled, profile, [fixture['eos']], sessions[i], 1,
                recorder, agent=i, epoch=0, turn=2, kind='cleanup', request_id=f'real-history-{args.run_tag}-{i}-cleanup',
                new_input_tokens=1, query=False)) for i in range(8)]
            await asyncio.gather(*pending)
            timed.save(directory/'resources-after.json', await timed.wait_retirement(capture,
                not_before=cleanup_at, expected_views={s:1 for s in sessions}))
        else:
            timed.save(directory/'resources-drained.json', await args.native_quiescent(stage='resume_drained', not_before=end))
            result['compression'] = dict(required=False)
            result['gpu_decode_concurrency'] = dict(passed=None, scope='Native histogram lower bound is added by wrapper; HTTP overlap alone is not same-step GPU concurrency')
        result.update(cleanup_complete=True, full_work_retired_at=time.perf_counter(),
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            stream_output_tokens_by_kind={k:sum(len(r['output_ids']) for r in recorder.rows if r['kind']==k)
                                          for k in ('prime','resume','cleanup')})
        timed.save(directory/'results.json', result)
        return result
    except BaseException as error:
        for job in pending:
            job.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        timed.save(directory/'results.json', dict(status='failed', error=repr(error), drained=False))
        raise
    finally:
        timed.save(directory/'traces.json', recorder.rows); recorder.progress(); recorder.stream.close()
