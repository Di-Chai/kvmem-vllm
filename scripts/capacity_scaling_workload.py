"""Fixed long-context capacity probe; the caller owns the inference server.

All offered requests finish their quota. This is a cropped, controlled capacity
load, not an original NIAH accuracy run. Client streaming overlap and observed
GPU batch concurrency are reported separately.
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
_spec = importlib.util.spec_from_file_location('capacity_timed_workload', ROOT / 'scripts/timed_agent_workload.py')
timed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(timed)
TASKS = ('niah_single_2', 'niah_multikey_2', 'niah_multivalue', 'niah_multiquery')
INPUT_TOKENS, GENERATION_BUDGET, OUTPUT_TOKENS = 229376, 32768, 8192


def token_bytes_sha256(tokens):
    values = array.array('I', tokens)
    if values.itemsize != 4:
        raise RuntimeError('Require 32-bit token IDs')
    if sys.byteorder != 'little':
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def crop_preserving_query(tokens, query_begin, query_end, target_tokens):
    """Keep the original prefix and complete Query-to-end suffix, without gold."""
    if (type(target_tokens) is not int or not 0 < target_tokens <= len(tokens)
            or type(query_begin) is not int or type(query_end) is not int
            or not 0 < query_begin < query_end <= len(tokens)):
        raise ValueError('Invalid original Query or crop extent')
    prefix_end = target_tokens - (len(tokens) - query_begin)
    if not 0 < prefix_end <= query_begin:
        raise ValueError('Crop cannot retain a nonempty original prefix and complete Query suffix')
    result = list(tokens[:prefix_end]) + list(tokens[query_begin:])
    return result, [prefix_end, prefix_end + query_end - query_begin], dict(
        method='prefix_then_complete_query_suffix', kept_ranges=[[0, prefix_end], [query_begin, len(tokens)]],
        removed_range=[prefix_end, query_begin], removed_tokens=len(tokens)-target_tokens,
        selection_uses_gold=False)


def prepare_fixture(source_manifest, *, context_tokens=INPUT_TOKENS,
                    output_tokens=OUTPUT_TOKENS, eos_token_id=248044):
    """Read the already frozen eight original token files; never synthesize text."""
    if context_tokens != INPUT_TOKENS or output_tokens != OUTPUT_TOKENS:
        raise ValueError('This protocol fixes 224K input and 8192 output tokens')
    if type(eos_token_id) is not int or not 0 <= eos_token_id < 2**31:
        raise ValueError('Require an explicit model cleanup token ID')
    path = Path(source_manifest).resolve()
    content = path.read_bytes()
    manifest = json.loads(content)
    samples = []
    # C4 is the first seed across all four task templates; C8 adds the second.
    for seed in (1000, 2000):
        for task in TASKS:
            matched = [r for r in manifest['rows'] if r['task'] == task
                       and r['nominal_input_tokens'] == 262144
                       and r['position_percent'] == 10 and r['seed'] == seed]
            if len(matched) != 1:
                raise ValueError('Frozen source must contain exactly one of each declared case')
            row = matched[0]
            token_path = path.parent / 'inputs' / (row['id'] + '.i32')
            raw = token_path.read_bytes()
            fingerprint = hashlib.sha256(raw).hexdigest()
            if (fingerprint != row['served_sha256'] or fingerprint != row['token_sha256']
                    or len(raw) % 4):
                raise ValueError('Original frozen token file differs from its manifest')
            values = array.array('i')
            if values.itemsize != 4:
                raise RuntimeError('Require 32-bit source token IDs')
            values.frombytes(raw)
            if sys.byteorder != 'little':
                values.byteswap()
            original = values.tolist()
            if len(original) != row['served_input_tokens'] or any(n < 0 for n in original):
                raise ValueError('Original token count or encoding changed')
            tokens, span, crop = crop_preserving_query(original, row['query_begin'], row['query_end'], context_tokens)
            samples.append(dict(id=row['id'], input_ids=tokens, input_sha256=timed.digest(tokens),
                input_bytes_sha256=token_bytes_sha256(tokens), query_begin=span[0], query_end=span[1],
                output_tokens=output_tokens, crop=crop,
                source=dict(path=str(token_path), sha256=fingerprint, tokens=len(original),
                    query_begin=row['query_begin'], query_end=row['query_end'], task=task,
                    nominal_input_tokens=262144, position_percent=10, seed=seed)))
    result = dict(format='capacity_scaling_v1', capacity_probe_only=True,
        scope='Cropped fixed-work capacity/performance load; no original NIAH accuracy claim',
        source_manifest=str(path), source_manifest_sha256=hashlib.sha256(content).hexdigest(),
        context_tokens=context_tokens, generation_budget=GENERATION_BUDGET,
        historical_ceiling=262144, output_tokens=output_tokens, eos=eos_token_id, samples=samples)
    validate_fixture(result)
    return result


def validate_fixture(fixture):
    if (fixture.get('format') != 'capacity_scaling_v1' or fixture.get('capacity_probe_only') is not True
            or fixture.get('context_tokens') != INPUT_TOKENS
            or fixture.get('generation_budget') != GENERATION_BUDGET
            or fixture.get('historical_ceiling') != INPUT_TOKENS + GENERATION_BUDGET
            or fixture.get('output_tokens') != OUTPUT_TOKENS
            or len(fixture.get('samples', [])) != 8
            or type(fixture.get('eos')) is not int or not 0 <= fixture['eos'] < 2**31):
        raise ValueError('Fixture differs from the fixed capacity protocol')
    identities, hashes = set(), set()
    for item in fixture['samples']:
        tokens = item['input_ids']
        if (len(tokens) != INPUT_TOKENS or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens)
                or item['output_tokens'] != OUTPUT_TOKENS
                or item['input_sha256'] != timed.digest(tokens)
                or item['input_bytes_sha256'] != token_bytes_sha256(tokens)
                or not 0 <= item['query_begin'] < item['query_end'] <= len(tokens)):
            raise ValueError('Frozen input, Query span, output quota or hash changed')
        identities.add(item['id'])
        hashes.add(item['input_bytes_sha256'])
    if len(identities) != 8 or len(hashes) != 8:
        raise ValueError('Require eight distinct frozen samples, not replicated prompts')


def read_records(path):
    if not path.exists():
        return []
    result = []
    lines = path.read_text().splitlines(keepends=True)
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines)-1 or line.endswith('\n'):
                raise
    return result


def native_matches(row, request_id):
    response = row.get('response_id')
    return isinstance(response, str) and re.fullmatch(re.escape(response)+r'-0-[0-9a-f]{8}', request_id) is not None


def decode_summary(rows, target):
    """Bound streams by first/last accepted events, excluding HTTP close delay."""
    spans = [(r['timeline'][0][0], r['timeline'][-1][0], r) for r in rows if r['timeline']]
    boundaries = sorted({t for start, end, _ in spans for t in (start, end)})
    intervals, histogram = [], {}
    for start, end in zip(boundaries, boundaries[1:]):
        active = [r for a, b, r in spans if a <= start and end <= b]
        histogram[len(active)] = histogram.get(len(active), 0.) + end-start
        if len(active) >= target:
            intervals.append(dict(begin=start, end=end, request_ids=[r['request_id'] for r in active]))
    seconds = sum(x['end']-x['begin'] for x in intervals)
    # Half-open event assignment avoids counting the same boundary event twice.
    count = sum(n for r in rows for t, n in r['timeline']
                if any(x['begin'] < t <= x['end'] and r['request_id'] in x['request_ids'] for x in intervals))
    left = max((a for a, _, _ in spans), default=0.)
    right = min((b for _, b, _ in spans), default=0.)
    return dict(target_concurrency=target, intervals=intervals, seconds=seconds,
        accepted_output_tokens=count, tokens_per_second=count/seconds if seconds else None,
        all_offered_common_seconds=max(0., right-left) if len(spans) == len(rows) else 0.,
        streaming_concurrency_seconds=histogram,
        scope='Client accepted-token overlap, not GPU same-step or pure-kernel throughput')


def gpu_decode_evidence(rows, decode):
    target = decode['target_concurrency']
    found = [r for r in rows if not r.get('prefill_tokens', 0)
             and r.get('decode_requests', 0) >= target
             and any(x['begin'] <= r['started_at'] < x['end'] for x in decode['intervals'])]
    return dict(passed=bool(found), target_concurrency=target, observed_steps=len(found),
        maximum_decode_requests=max((r.get('decode_requests', 0) for r in rows), default=None),
        example_steps=found[:4], scope='Recorded completed public GPU steps; native running gauges alone do not prove this')


def compression_evidence(rows, fixture, profile, resources, archive, geometry, refreshes):
    kv = profile['kvmem']
    limit = kv['select_budget'] + kv['gen_budget']
    reports = []
    for row in rows:
        item = fixture['samples'][row['agent']]
        expected = row['input_tokens'] + len(row['output_ids']) - 1
        views = [v for v in resources.get('session_views', []) if v.get('namespace') == [row['session']]*2]
        completed = [r for r in archive.get('completed', []) if native_matches(row, r['request_id'])]
        queries = [r for r in refreshes if native_matches(row, r['request_id']) and r.get('selection') == 'query']
        passed = (len(views) == len(completed) == len(queries) == 1
            and views[0]['history'] == completed[0]['history_end'] == expected
            and completed[0].get('checkpoint_tokens') == expected
            and completed[0].get('checkpoint_kind') == 'M'
            and 0 < views[0]['window'] <= limit < expected
            and queries[0].get('query_span') == [item['query_begin'], item['query_end']]
            and 0 < queries[0]['selected_tokens'] <= kv['select_budget'] < queries[0]['history_begin']
            and queries[0].get('mean_k_backend') == 'qw3_cuda'
            and queries[0].get('action_attention_enabled') is True
            and not queries[0].get('cancelled', False))
        reports.append(dict(request_id=row['request_id'], response_id=row['response_id'], passed=passed,
            expected_kv_history=expected, accepted_token_ledger=row['input_tokens']+len(row['output_ids']),
            session_view=views[0] if len(views) == 1 else views,
            completion=completed[0] if len(completed) == 1 else completed,
            query_selection=queries[0] if len(queries) == 1 else queries))
    record_bytes = geometry.get('cpu_archive_bytes_per_token', 0) * kv['block_tokens']
    minimum = sum((r['input_tokens']+len(r['output_ids'])-1)//kv['block_tokens'] * record_bytes for r in rows)
    storage_ok = (geometry.get('archive_codec') == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1'
        and geometry.get('historical_tokens') == 262144 and record_bytes > 0
        and archive.get('archive_backend') == 'cpu' and archive.get('repository_gc') is False
        and archive.get('cpu_authority_bytes', -1) >= minimum
        and archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0
        and not archive.get('pending_downloads') and not archive.get('session_retiring_requests'))
    return dict(passed=storage_ok and all(r['passed'] for r in reports), requests=reports,
        cpu_storage_passed=storage_ok, minimum_sealed_history_bytes=minimum,
        actual_cpu_authority_bytes=archive.get('cpu_authority_bytes'), geometry=geometry,
        scope='Generation-qualified retirement and CPU publication accounting; no per-byte oracle or task-quality claim')


def warmup_selection_evidence(row, profile, refreshes, query_span):
    """Require the scoring branch, not merely a configuration backend label.

    A short history <= K skips select_for_queries entirely. Beyond K, a
    nonempty eligible mask reaches cuda_scoring.kernel().score; completion of
    the refresh fence establishes successful extension loading and execution.
    """
    kv = profile['kvmem']
    matches = [r for r in refreshes if native_matches(row, r['request_id']) and r.get('selection') == 'query']
    if len(matches) != 1:
        return dict(passed=False, reason='missing_or_duplicate_query_refresh', matches=matches)
    query = matches[0]
    block = kv['block_tokens']
    pages = query.get('history_begin', 0)//block
    hard = set(query.get('mandatory_fit', {}).get('hard_blocks', []))
    sink, recent = kv.get('sink_blocks', 0), kv.get('recent_blocks', 0)
    eligible = sum(i >= sink and i < pages-recent and i not in hard for i in range(pages))
    passed = (query.get('mean_k_backend') == 'qw3_cuda' and query.get('query_span') == query_span
        and query.get('history_begin', 0) > kv['select_budget']
        and 0 < query.get('selected_tokens', 0) <= kv['select_budget']
        and query.get('query_score_tokens', 0) > 0 and eligible > 0
        and not query.get('cancelled', False)
        and query.get('refresh_finished_at', 0) >= query.get('refresh_started_at', 1))
    return dict(passed=passed, eligible_complete_pages=eligible, query_refresh=query,
        scope='Completed history>K Query scoring with nonempty eligible pages reaches qw3 CUDA kernel.score; no timing claim')


async def warmup(client, profile, fixture, directory, args):
    """Same 66K+1 initialization on both engines, fully outside formal counters."""
    validate_fixture(fixture)
    native = bool(profile.get('native_vllm', False))
    count = 65536+2048
    tokens = list(fixture['samples'][0]['input_ids'][:count])
    if len(tokens) != count:
        raise ValueError('Warmup needs the declared frozen source prefix')
    # This entire final 128-token span is a public initialization Query.
    query_span = [count-128, count]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'events.jsonl').exists():
        raise FileExistsError('Warmup must use a fresh directory')
    recorder = timed.Recorder(directory, 1)
    session = f'capacity-warmup-{args.run_tag}'
    began = time.perf_counter()
    capture = Path(args.capture_directory) if not native else None
    timed.save(directory/'workload.json', dict(input_tokens=count, output_tokens=1,
        query_span=query_span, input_bytes_sha256=token_bytes_sha256(tokens),
        source_sample=fixture['samples'][0]['id'], source_crop='first 67584 tokens of frozen capacity sample 0',
        native_vllm=native, warmup_begin=began,
        scope='Initialization only: identical formal-independent salt, prefix, output and 1-token cleanup on both engines'))
    try:
        row = await timed.stream_call(client, profile, tokens, session, 1, recorder,
            agent=0, epoch=0, turn=0, kind='warmup', request_id=f'capacity-warmup-{args.run_tag}',
            new_input_tokens=count, query_plan={'kvmem_query_span':query_span, 'kvmem_reselect':'force'})
        if row['usage'].get('prompt_tokens') != count:
            raise RuntimeError('Warmup input count changed')
        if native:
            state = await args.native_quiescent(stage='warmup_drained', not_before=row['end'])
            proof = dict(passed=True, required=False, scope='Native engine: same warmup input/output, no MeanK extension')
        else:
            state = await timed.wait_retirement(capture, not_before=row['begin'], expected_views={session:count})
            proof = warmup_selection_evidence(row, profile, read_records(capture/'bounded_refresh.jsonl'), query_span)
        timed.save(directory/'resources-drained.json', state)
        timed.save(directory/'kernel-load-proof.json', proof)
        # Clean even if proof fails, then fail closed before formal measurement.
        cleanup_begin = time.perf_counter()
        await timed.stream_call(client, profile, [fixture['eos']], session, 1, recorder,
            agent=0, epoch=1, turn=1, kind='cleanup', request_id=f'capacity-warmup-{args.run_tag}-cleanup',
            new_input_tokens=1, query=False)
        after = (await args.native_quiescent(stage='warmup_cleanup', not_before=cleanup_begin) if native else
                 await timed.wait_retirement(capture, not_before=cleanup_begin, expected_views={session:1}))
        timed.save(directory/'resources-after.json', after)
        result = dict(status='completed', drained=True, cleanup_complete=True,
            warmup_begin=began, warmup_end=time.perf_counter(), input_tokens=count, output_tokens=1,
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            kernel_load_proof=proof, cleanup_retained_session_tokens=None if native else 1,
            cleanup_scope=('Native requests quiescent; evictable APC entries may remain under isolated salts' if native else
                           'Warmup history reset to one token; one tiny session and its model checkpoint may remain, resources-after is authoritative'),
            scope='Initialization and cleanup excluded from all formal performance counters')
        if not proof['passed']:
            raise RuntimeError('Warmup did not prove nonempty MeanK CUDA scoring; do not start formal measurement')
        timed.save(directory/'results.json', result)
        return result
    except BaseException as error:
        timed.save(directory/'results.json', dict(status='failed', drained=False,
            error=f'{type(error).__name__}: {error}', scope='No automatic retry; owned server cleanup belongs to caller'))
        raise
    finally:
        timed.save(directory/'traces.json', recorder.rows)
        recorder.progress()
        recorder.stream.close()


async def workload(client, profile, fixture, directory, args):
    validate_fixture(fixture)
    offered, admitted = args.offered_concurrency, args.admitted_concurrency
    native = bool(profile.get('native_vllm', False))
    if (offered not in (4, 8) or type(admitted) is not int or not 1 <= admitted <= offered
            or args.output_tokens != OUTPUT_TOKENS or profile['max_model_len'] != 262144):
        raise ValueError('Invalid frozen offered/admitted concurrency or context/output quota')
    if not native and (profile['kvmem']['select_budget'] != 65536
            or profile['kvmem']['gen_budget'] != GENERATION_BUDGET
            or not profile['kvmem'].get('cpu_history_coordination')
            or profile['kvmem'].get('repository_gc', False)):
        raise ValueError('Require K64/G32 and complete coordinated CPU history without GC')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'events.jsonl').exists():
        raise FileExistsError('Capacity workload must use a fresh directory')
    recorder = timed.Recorder(directory, offered)
    sessions = [f'capacity-{args.run_tag}-{i}' for i in range(offered)]
    semaphore = asyncio.Semaphore(admitted)
    tasks, queue_rows = [], []
    capture = Path(args.capture_directory) if not native else None
    begin, wall_begin = time.perf_counter(), time.time()
    timed.save(directory/'workload.json', dict(fixture_sha256=timed.digest(fixture),
        offered_concurrency=offered, admitted_concurrency=admitted, input_tokens=INPUT_TOKENS,
        output_tokens=OUTPUT_TOKENS, generation_reserve=GENERATION_BUDGET, historical_ceiling=262144,
        native_vllm=native, workload_begin=begin, wall_begin=wall_begin,
        scope='All offered quotas drain; 32K is generation budget, this probe generates 8192 tokens'))

    async def call(i):
        item = fixture['samples'][i]
        enqueued = time.perf_counter()
        recorder.phase(i, 'ingress_waiting', 0, 0)
        async with semaphore:
            admitted_at = time.perf_counter()
            queue_rows.append(dict(agent=i, enqueued_at=enqueued, admitted_at=admitted_at,
                                   queue_seconds=admitted_at-enqueued))
            row = await timed.stream_call(client, profile, item['input_ids'], sessions[i], OUTPUT_TOKENS, recorder,
                agent=i, epoch=0, turn=0, kind='workload', request_id=f'capacity-{args.run_tag}-{i}',
                new_input_tokens=INPUT_TOKENS, query=True,
                query_plan={'kvmem_query_span':[item['query_begin'], item['query_end']], 'kvmem_reselect':'force'},
                source_metadata=dict(sample_id=item['id'], source_sha256=item['source']['sha256'],
                    input_bytes_sha256=item['input_bytes_sha256'], admitted_at=admitted_at, enqueued_at=enqueued))
            if row['usage'].get('prompt_tokens') != INPUT_TOKENS:
                raise RuntimeError('Server logical prompt count differs from the full frozen input')
            recorder.phase(i, 'completed', 0, 0)
            return row

    try:
        tasks = [asyncio.create_task(call(i)) for i in range(offered)]
        rows = await asyncio.gather(*tasks)
        end = time.perf_counter()
        decode = decode_summary(rows, admitted)
        recorder.window = dict(begin=begin, end=end, t0_monotonic=begin, t1_monotonic=end,
            t0_wall=wall_begin, t1_wall=time.time(), duration_seconds=end-begin,
            t0_definition='All finite requests offered; ingress waiting included, no early cancellation')
        timed.save(directory/'window.json', recorder.window)
        result = dict(status='completed', drained=True, capacity_probe_only=True,
            offered_concurrency=offered, admitted_concurrency=admitted,
            workload_begin=begin, workload_end=end, fixed_work_makespan_seconds=end-begin,
            fixed_work_input_tokens=offered*INPUT_TOKENS, fixed_work_output_tokens=offered*OUTPUT_TOKENS,
            output_tokens_per_second=offered*OUTPUT_TOKENS/(end-begin),
            input_completion_rate_tokens_per_second=offered*INPUT_TOKENS/(max(r['timeline'][0][0] for r in rows)-begin),
            input_completion_rate_scope='Aggregate input / offer-to-last-first-token; includes queue, selection and concurrent decode, not pure GPU prefill',
            common_decode_window=decode, ingress_queue=queue_rows,
            ttft_seconds=[r['timeline'][0][0]-r['begin'] for r in rows], cancelled_requests=[],
            generation_budget=GENERATION_BUDGET, actual_output_quota=OUTPUT_TOKENS,
            scope='Controlled fixed inputs/quotas; original NIAH accuracy and multi-round agent behavior are not measured')
        if not native:
            state = await timed.wait_retirement(capture, not_before=begin,
                expected_views={r['session']:r['input_tokens']+len(r['output_ids'])-1 for r in rows})
            archive = json.loads((capture/'archive.json').read_text())
            geometry = json.loads((capture/'cache-geometry.json').read_text())
            timed.save(directory/'resources-drained.json', state)
            timed.save(directory/'archive-drained.json', archive)
            result['compression'] = compression_evidence(rows, fixture, profile, state, archive, geometry,
                read_records(capture/'bounded_refresh.jsonl'))
            result['gpu_decode_concurrency'] = gpu_decode_evidence(read_records(capture/'model_step_latency.jsonl'), decode)
            retired = time.perf_counter()
            cleanup_begin = time.perf_counter()
            # Fixed work is complete; retire retained histories with owned-session resets.
            cleanup = [asyncio.create_task(timed.stream_call(client, profile, [fixture['eos']], sessions[i], 1, recorder,
                agent=i, epoch=0, turn=1, kind='cleanup', request_id=f'capacity-{args.run_tag}-{i}-cleanup',
                new_input_tokens=1, query=False)) for i in range(offered)]
            tasks.extend(cleanup)
            await asyncio.gather(*cleanup)
            timed.save(directory/'resources-after.json', await timed.wait_retirement(capture,
                not_before=cleanup_begin, expected_views={s:1 for s in sessions}))
            result['cleanup_seconds'] = time.perf_counter()-cleanup_begin
        else:
            if not callable(getattr(args, 'native_quiescent', None)):
                raise ValueError('Native arm requires server quiescence evidence')
            state = await args.native_quiescent(stage='fixed_work_drained', not_before=end)
            timed.save(directory/'resources-drained.json', state)
            retired = time.perf_counter()
            result['compression'] = dict(required=False, scope='Native full-context baseline')
            result['gpu_decode_concurrency'] = dict(passed=None, scope='No native step-level observer supplied; running gauges and client overlap do not establish same-batch execution')
        result.update(full_work_retired_at=retired, full_work_retired_makespan_seconds=retired-begin,
            cleanup_complete=True, total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            stream_output_tokens_by_kind={kind:sum(len(r['output_ids']) for r in recorder.rows if r['kind'] == kind)
                                          for kind in ('workload', 'cleanup')})
        timed.save(directory/'results.json', result)
        return result
    except BaseException as error:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        timed.save(directory/'results.json', dict(status='failed', drained=False, cleanup_complete=False,
            error=f'{type(error).__name__}: {error}', workload_begin=begin,
            scope='Failure preserved; caller must stop only its owned server, never count partial quota as complete'))
        raise
    finally:
        timed.save(directory/'traces.json', recorder.rows)
        recorder.progress()
        recorder.stream.close()


run_workload = workload
