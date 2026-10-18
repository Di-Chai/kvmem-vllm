"""Prepare an eight-user, two-round natural-output capture-slot coverage probe.

Run only the frozen entry after the exact early-release runtime passes its
separate GPU byte qualification. Recorded original histories, queries, external
tool messages and waits remain complete. This is a lifecycle/coverage probe,
not a throughput or on-policy DeepSWE quality result.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
KIND = 'early_capture_reuse_qualification_v1'
USERS, ROUNDS, LIMIT, CEILING = 8, 2, 32000, 262144
FEEDBACK_BYTES = 268435456


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


natural = module('capture_reuse_natural', 'benchmark_rolling_agents.py')
ready, qual, cap, work = natural.ready, natural.qual, natural.cap, natural.work
ARM = natural.ARMS[2]


def profile(source, port):
    p = copy.deepcopy(source)
    kv = p['kvmem']
    cap.require(p.get('max_num_seqs') == USERS and p.get('max_model_len') == CEILING and
        p.get('draft_method') == 'mtp' and p.get('draft_tokens') == 4 and
        p.get('model_format') == 'safetensors' and p.get('execution_mode') == 'compiled_graph',
        'Require original eight-slot compiled NVFP4 MTP(k=4) natural deployment')
    cap.require(kv.get('select_budget') == 65536 and kv.get('gen_budget') == 32768 and
        kv.get('block_tokens') == 32 and kv.get('archive_backend') == 'cpu' and
        kv.get('cpu_read_cache_bytes') == 100_000_000_000 and not kv.get('repository_gc', False),
        'Require K64/G32/full256K and complete CPU100GB history')
    cap.require(all(kv.get(k) is True for k in ('rolling_active_window', 'split_capture_dma',
        'async_cpu_scatter', 'request_planning', 'async_refresh', 'async_restore',
        'cpu_history_coordination', 'session_capture_terminal', 'action_pressure')) and
        kv.get('session_checkpoint_policy') == 'pm' and not kv.get('session_capture_decode', False),
        'Require existing coordinated async rolling/PM controls')
    p.update(port=port, verify_session_restore=False)
    kv.update(early_capture_slot_release=True, early_capture_feedback_bytes=FEEDBACK_BYTES)
    return p


def fixture_budget(fixture, p):
    budget = natural.fixture_bounds(fixture, p)
    # The entire immutable first-eight fixture is retained, including unused
    # later turns. Worst-case second-turn capacity is a reported condition, not
    # grounds for shortening the original prompt or changing generation limits.
    budget['two_round_worst_case_tokens'] = [len(plan['initial_tokens']) + ROUNDS*LIMIT +
        len(plan['turns'][1]['tool_tokens']) for plan in fixture['agent_plans']]
    budget['conditional_second_round_users'] = [i for i, n in enumerate(
        budget['two_round_worst_case_tokens']) if n > CEILING]
    return budget


def byte_identity(path, runtime_hashes, native_source):
    identity = natural.qualification_identity(path, ARM, runtime_hashes, native_source)
    kv = cap.read(path/'profile.json')['kvmem']
    cap.require(kv.get('early_capture_slot_release') is True and
        kv.get('early_capture_feedback_bytes') == FEEDBACK_BYTES and
        kv.get('split_capture_dma') is True and kv.get('async_cpu_scatter') is True,
        'Byte qualification did not enable the exact early-release candidate')
    return identity


def completed_byte_proof(m):
    expected = m['byte_qualification']
    path = Path(expected['path'])
    actual = byte_identity(path, m['runtime_sha256'], m['native_source'])
    cap.require(actual == expected, 'Byte qualification identity changed after preparation')
    return dict(identity=actual, evidence=natural.validate_completed_qualification(path, ARM))


def prepare(args):
    root = args.output.resolve()
    cap.require(not root.exists(), 'Use a fresh qualification directory')
    p, fixture = profile(cap.read(args.config), args.port), cap.read(args.fixture)
    from transformers import AutoTokenizer
    eos = AutoTokenizer.from_pretrained(p['tokenizer'], local_files_only=True).eos_token_id
    cap.require(type(eos) is int and fixture.get('eos') == eos, 'Original fixture/tokenizer EOS differs')
    budget = fixture_budget(fixture, p)
    natural.verify_provenance(fixture)
    runtime = args.candidate.resolve()
    hashes, native_source = cap.hashes(runtime), cap.native_source_identity()
    cap.require('src/vllm_kvmem/capture_feedback.py' in hashes, 'Early-release runtime is absent')
    identity = byte_identity(args.byte_qualification.resolve(), hashes, native_source)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    for directory in ('src', 'scripts', 'csrc'):
        shutil.copytree(ROOT/directory, root/'frozen'/directory, ignore=ignore)
    shutil.copytree(runtime, root/'runtime', ignore=ignore)
    cap.require(cap.hashes(root/'runtime') == hashes, 'Exact runtime changed during freeze')
    cap.save(root/'profile.json', p)
    # Byte-identical immutable input; no reserialization, truncation or reselection.
    shutil.copyfile(args.fixture, root/'fixture.json')
    cap.save(root/'manifest.json', dict(kind=KIND, at=time.time(),
        source_sha256=cap.hashes(root/'frozen'), runtime_sha256=hashes,
        runtime_source=str(runtime), native_source=native_source,
        profile_sha256=cap.sha(root/'profile.json'), profile_source=str(args.config.resolve()),
        profile_source_sha256=cap.sha(args.config), fixture_sha256=cap.sha(root/'fixture.json'),
        fixture_source=str(args.fixture.resolve()), fixture_source_sha256=cap.sha(args.fixture),
        byte_qualification=identity, budget=budget, port=args.port, idle_seconds=10,
        workload=dict(concurrency=USERS, complete_rounds=ROUNDS, generation_limit=LIMIT,
                      generation_mode='natural', sampling=natural.SAMPLING),
        scope='Two independent natural-output turns per original user, complete CPU histories and original '
              'external messages/waits; normal cleanup after all users drain. No output clipping, injected '
              'GPU delay, client cancellation, SSD or recomputation. No live agent quality claim. '
              'A context-capacity shortfall or absent real shared-slot reuse is coverage_insufficient.'))
    cap.save(root/'status.json', dict(status='prepared', at=time.time(),
        byte_qualification_may_be_pending=True, gpu_started=False))


def verify(root):
    m = cap.read(root/'manifest.json')
    cap.require(ROOT.resolve() == (root/'frozen').resolve(), 'Run the frozen entry point')
    cap.require(m.get('kind') == KIND and cap.hashes(root/'frozen') == m['source_sha256'], 'Frozen source changed')
    cap.require(cap.hashes(root/'runtime') == m['runtime_sha256'], 'Exact runtime changed')
    cap.require(cap.sha(root/'profile.json') == m['profile_sha256'], 'Frozen profile changed')
    cap.require(cap.sha(root/'fixture.json') == m['fixture_sha256'] == m['fixture_source_sha256'],
                'Complete immutable fixture changed')
    profile_source, fixture_source = Path(m['profile_source']), Path(m['fixture_source'])
    cap.require(cap.sha(profile_source) == m['profile_source_sha256'], 'Original natural profile changed')
    cap.require(cap.sha(fixture_source) == m['fixture_source_sha256'], 'Original natural fixture changed')
    cap.require(cap.native_source_identity() == m['native_source'], 'Installed native implementation changed')
    p, fixture = cap.read(root/'profile.json'), cap.read(root/'fixture.json')
    cap.require(p == profile(cap.read(profile_source), m['port']), 'Profile differs from original natural deployment')
    natural.verify_provenance(fixture)
    cap.require(fixture_budget(fixture, p) == m['budget'], 'Full history budget changed')
    cap.require(m['workload'] == dict(concurrency=USERS, complete_rounds=ROUNDS,
        generation_limit=LIMIT, generation_mode='natural', sampling=natural.SAMPLING), 'Workload controls changed')
    return m


def output_evidence(rows, events, fixture):
    """Reconstruct exact accepted IDs, their usage, and each original input chain."""
    proofs = []
    cap.require(len({r['request_id'] for r in rows}) == len(rows), 'Duplicate client request identity')
    for row in rows:
        rid, ids = row['request_id'], row['output_ids']
        chunks = [e for e in events if e.get('event') == 'stream_chunk' and e.get('request_id') == rid]
        streamed = [token for e in chunks for token in e['token_ids']]
        usage = [e['usage'] for e in chunks if e.get('usage')]
        finished = [e for e in events if e.get('event') == 'request_finished' and e.get('request_id') == rid]
        cap.require(row.get('status') == 'completed' and streamed == ids and
            all(type(token) is int and token >= 0 for token in ids) and
            row.get('output_sha256') == work.digest(ids) and
            (row.get('usage') or {}).get('completion_tokens') == len(ids) and
            usage and usage[-1] == row['usage'] and
            len(finished) == 1 and finished[0].get('status') == 'completed' and
            finished[0].get('accepted_tokens') == len(ids), 'Stream/token IDs/usage reconciliation failed')
        offset = 0
        for chunk in chunks:
            offset += len(chunk['token_ids'])
            cap.require(chunk.get('accepted_tokens') == len(chunk['token_ids']) and
                chunk.get('output_offset') == offset and
                chunk.get('response_id') in (None, row['response_id']), 'Stream chunk identity/count changed')
        cap.require(sum(n for _, n in row['timeline']) == len(ids), 'Token timeline differs from accepted IDs')
        if row['kind'] == 'workload':
            cap.require(row.get('generation_mode') == 'natural' and
                row.get('declared_generation_ceiling') == LIMIT and
                row.get('finish_reason') in ('stop', 'length') and len(ids) <= LIMIT,
                'Natural generation contract changed')
        else:
            cap.require(row['kind'] == 'cleanup' and len(ids) == 1 and row['input_tokens'] == 1,
                        'Normal one-token session cleanup missing')
        proofs.append(dict(request_id=rid, accepted_tokens=len(ids), output_sha256=work.digest(ids)))
    for user, plan in enumerate(fixture['agent_plans']):
        tokens = list(plan['initial_tokens'])
        turns = sorted((r for r in rows if r['agent'] == user and r['kind'] == 'workload'), key=lambda r:r['turn'])
        cap.require([r['turn'] for r in turns] == list(range(len(turns))) and 0 < len(turns) <= ROUNDS,
                    'Missing or repeated natural round')
        for turn, row in enumerate(turns):
            if turn:
                tokens.extend(plan['turns'][turn]['tool_tokens'])
                waits = [e for e in events if e.get('event') == 'tool_wait_completed' and
                         e.get('agent') == user and e.get('turn') == turn]
                declared = plan['turns'][turn]['tool_wait_seconds']
                cap.require(len(waits) == 1 and waits[0]['declared_seconds'] == declared and
                    waits[0]['elapsed_seconds'] + .01 >= declared, 'Original tool wait was shortened')
            cap.require(row['input_tokens'] == len(tokens) and row['input_sha256'] == work.digest(tokens) and
                row['query_plan'] == plan['query_plan'] and
                row['source_metadata']['source_task'] == plan['source_task'] and
                row['source_metadata']['source_turn'] == plan['turns'][turn]['source_turn'],
                'Full original history, Query or external-message chain changed')
            tokens.extend(row['output_ids'])
        cap.require(sum(r['kind'] == 'cleanup' and r['agent'] == user for r in rows) == 1,
                    'Each user needs one completed normal cleanup')
    return proofs


def reuse_evidence(events, samples, rows, *, not_before=0.):
    """A source release alone is not proof that independent work reused it."""
    stages = {}
    for e in events:
        cap.require(e.get('lane') == 'bulk' and type(e.get('slot')) is int and e['slot'] >= 0 and
            type(e.get('generation')) is int and e['generation'] > 0, 'Invalid physical source identity')
        stages.setdefault((e['lane'], e['slot'], e['generation']), []).append(e)
    workload = [r for r in rows if r['kind'] == 'workload']
    def workload_owners(e):
        return e.get('request_ids') and all(any(qual.core.native_matches(r, rid) for r in workload)
                                           for rid in e['request_ids'])
    proof = []
    for key, current in stages.items():
        submitted = [e for e in current if e['stage'] == 'source_submitted']
        cap.require(len(submitted) == 1, 'Missing or duplicate source generation submission')
        new = submitted[0]
        if not workload_owners(new):
            continue
        for old_generation in new.get('pending_previous_generations', []):
            old = stages.get((key[0], key[1], old_generation), [])
            starts = [e for e in old if e['stage'] == 'source_submitted']
            releases = [e for e in old if e['stage'] == 'source_released']
            completes = [e for e in old if e['stage'] == 'semantic_complete']
            cap.require(len(starts) == len(releases) == len(completes) == 1,
                        'A reused generation lacks complete source/semantic lifecycle')
            start, release, complete = starts[0], releases[0], completes[0]
            if len(start['request_ids']) < 2 or not workload_owners(start):
                continue
            copied = [e for e in old if e['stage'] == 'source_copied' and e['at'] <= release['at']]
            indices = [i for e in copied for i in e['copied_indices']]
            cap.require(sorted(indices) == list(range(len(start['request_ids']))) and
                old_generation < key[2] and start['at'] <= release['at'] <= new['at'] < complete['at'] and
                release.get('semantic_pending', 0) > 0,
                'Shared slot was reused before complete copy or after semantic completion')
            proof.append(dict(lane=key[0], slot=key[1], previous_generation=old_generation,
                next_generation=key[2], previous_owners=start['request_ids'], next_owners=new['request_ids'],
                source_released_at=release['at'], reused_at=new['at'], semantic_complete_at=complete['at'],
                feedback_bytes=release['feedback_bytes']))
    for sample in samples:
        state = sample['feedback']
        cap.require(0 <= state['detached_packets'] <= state['packet_limit'] and
            0 <= state['detached_owners'] <= state['owner_limit'] <= USERS and
            0 <= state['detached_bytes'] <= state['byte_limit'] == FEEDBACK_BYTES,
            'Detached feedback credit bound exceeded')
    if samples:
        final = samples[-1]
        cap.require(final.get('at', -1) >= not_before and all(final['feedback'][key] == 0 for key in
            ('detached_packets', 'detached_owners', 'detached_bytes')),
            'Final detached feedback credits are stale or not released after cleanup')
    counters = samples[-1]['publication_metrics'] if samples else {}
    count = lambda name: sum(value.get(name, 0) for key, value in counters.items() if key.endswith('/bulk'))
    observed = bool(proof and samples and count('source_releases_before_semantic') > 0 and
                    count('shared_source_reuse_before_semantic_packets') > 0)
    return dict(covered=observed, shared_slot_reuse_proofs=proof,
        source_releases_before_semantic=count('source_releases_before_semantic'),
        shared_source_reuse_before_semantic_packets=count('shared_source_reuse_before_semantic_packets'),
        feedback_samples=len(samples),
        scope='Only real workload packets, excluding cleanup; physical lane/slot/generation and event ordering. '
              'Counters are cumulative host observations and are not additive GPU-idle time.')


async def wait_feedback_released(capture, *, not_before, timeout=30.):
    deadline = time.monotonic()+timeout
    while True:
        try:
            progress = cap.read(capture/'progress.json')
        except (OSError, ValueError):
            progress = {}
        feedback = progress.get('early_capture_feedback', {})
        if progress.get('publication_observed_at', -1) >= not_before and all(
                feedback.get(key) == 0 for key in ('detached_packets', 'detached_owners', 'detached_bytes')):
            return progress
        if time.monotonic() >= deadline:
            raise TimeoutError('Detached feedback credit cleanup did not publish a fresh zero balance')
        await asyncio.sleep(.01)


async def workload(client, p, fixture, directory, capture):
    directory.mkdir()
    recorder = work.Recorder(directory, USERS)
    states = [dict(session=f'capture-reuse-{i}', tokens=list(plan['initial_tokens']))
              for i, plan in enumerate(fixture['agent_plans'])]
    publications, capacity = [], []
    async def call(i, turn, cleanup=False):
        state, plan = states[i], fixture['agent_plans'][i]
        return await work.stream_call(client, p, [fixture['eos']] if cleanup else state['tokens'],
            state['session'], 1 if cleanup else LIMIT, recorder, agent=i, epoch=0, turn=turn,
            kind='cleanup' if cleanup else 'workload', request_id=f'capture-reuse-{i}-{turn}-{int(cleanup)}',
            new_input_tokens=1 if cleanup else len(plan['initial_tokens']) if turn == 0 else
                len(plan['turns'][turn]['tool_tokens']), query=not cleanup,
            query_plan=None if cleanup else plan['query_plan'],
            source_metadata=None if cleanup else dict(source_task=plan['source_task'],
                source_turn=plan['turns'][turn]['source_turn'],
                recorded_output_quota=plan['turns'][turn]['output_tokens']),
            generation_mode='fixed' if cleanup else 'natural')
    async def agent(i):
        state, plan = states[i], fixture['agent_plans'][i]
        for turn in range(ROUNDS):
            addition = plan['turns'][turn]['tool_tokens'] if turn else []
            if len(state['tokens']) + len(addition) + LIMIT > CEILING:
                capacity.append(dict(agent=i, turn=turn, retained_tokens=len(state['tokens']),
                    original_tool_tokens=len(addition), generation_limit=LIMIT, reason='context_capacity'))
                return
            if turn:
                recorder.phase(i, 'tool_wait', 0, turn)
                delay = plan['turns'][turn]['tool_wait_seconds']
                begin = time.monotonic()
                await asyncio.sleep(delay)
                recorder.event('tool_wait_completed', agent=i, turn=turn,
                    declared_seconds=delay, elapsed_seconds=time.monotonic()-begin)
                state['tokens'].extend(addition)
            row = await call(i, turn)
            state['tokens'].extend(row['output_ids'])
        recorder.phase(i, 'rounds_completed', 0, ROUNDS)
    try:
        outcomes = await asyncio.gather(*(agent(i) for i in range(USERS)), return_exceptions=True)
        errors = [r for r in outcomes if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        before_cleanup = await work.wait_retirement(capture, timeout=180.,
            not_before=max(r['begin'] for r in recorder.rows),
            expected_views={s['session']:len(s['tokens'])-1 for s in states})
        cap.save(directory/'resources-before-cleanup.json', before_cleanup)
        archive = cap.read(capture/'archive.json')
        cap.save(directory/'archive-before-cleanup.json', archive)
        publications.extend(archive.get('completed', []))
        proofs = qual.history_evidence(recorder.rows, publications)
        outcomes = await asyncio.gather(*(call(i, ROUNDS, True) for i in range(USERS)), return_exceptions=True)
        errors = [r for r in outcomes if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        final = await work.wait_retirement(capture, timeout=180., not_before=max(r['begin'] for r in outcomes),
            expected_views={s['session']:1 for s in states})
        cap.save(directory/'resources-after.json', final)
        cleanup_begin = max(r['begin'] for r in outcomes)
        cap.save(directory/'publication-after-cleanup.json',
            await wait_feedback_released(capture, not_before=cleanup_begin))
        archive = cap.read(capture/'archive.json')
        cap.save(directory/'archive-after.json', archive)
        publications.extend(archive.get('completed', []))
        proofs.extend(qual.history_evidence(outcomes, publications))
        recorder.stream.flush()
        output = output_evidence(recorder.rows, ready.records(directory/'events.jsonl'), fixture)
        reuse = reuse_evidence(ready.records(capture/'capture_feedback_events.jsonl'),
                               ready.records(capture/'publication_samples.jsonl'), recorder.rows,
                               not_before=cleanup_begin)
        complete = sum(r['kind'] == 'workload' for r in recorder.rows) == USERS*ROUNDS
        return dict(status='passed' if complete and reuse['covered'] else 'coverage_insufficient',
            complete_rounds=complete, context_capacity_shortfalls=capacity, reuse=reuse,
            history_proofs=proofs, output_proofs=output, cleanup_completed=True,
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            actual_workload_requests=sum(r['kind'] == 'workload' for r in recorder.rows),
            archive_final=archive, byte_oracle_scope='Separate exact-runtime qualification required before launch',
            scope='Lifecycle coverage, not comparative throughput or live agent task quality')
    finally:
        cap.save(directory/'traces.json', recorder.rows)
        cap.save(directory/'history_publications.json', publications)
        recorder.progress()
        recorder.stream.close()


async def run_server(root, m):
    import httpx
    import psutil
    await ready.bench.q.wait_idle(root, m['idle_seconds'])
    # The idle wait may be long. A previously valid source or byte proof must
    # not authorize a different runtime that appeared while this owner waited.
    cap.require(verify(root) == m, 'Frozen identity changed while waiting for the GPU')
    cap.save(root/'byte-qualification-proof.json', completed_byte_proof(m))
    cap.require(not ready.bench.q.gpu_state()['pids'], 'Foreign GPU process before launch')
    cap.require(cap.host_memory()['MemAvailable'] >= m['budget']['required_available_bytes'],
                'Insufficient complete-history CPU headroom')
    p = cap.read(root/'profile.json')
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('VLLM_', 'KVMEM_'))}
    env.update(PYTHONPATH=str(root/'runtime/src'), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').is_file():
            env['CUDA_HOME'] = toolkit
            break
    command = [sys.executable, '-m', 'vllm_kvmem', 'serve', '--config', str(root/'profile.json'),
               '--run-dir', str(root/'server')]
    cap.save(root/'launch.json', dict(argv=command, cwd=str(root/'runtime')))
    with (root/'server.log').open('x') as log:
        started, wall = time.monotonic(), time.time()
        server = subprocess.Popen(command, cwd=root/'runtime', env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        cap.save(root/'server-owner.json', dict(pid=server.pid, create_time=psutil.Process(server.pid).create_time()))
        cap.save(root/'status.json', dict(status='loading', pid=server.pid, at=time.time()))
        async def guard():
            while True:
                state = await asyncio.to_thread(ready.bench.q.gpu_state)
                cap.require(not(set(state['pids'])-ready.bench.q.owned_pids(server)), 'Foreign GPU process appeared')
                cap.require(server.poll() is None, 'Owned server exited')
                memory = cap.host_memory()
                cap.require(memory['MemAvailable'] >= m['budget']['minimum_free_bytes'], 'Host free memory below8GiB')
                with (root/'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.monotonic(), wall_at=time.time(), **state, memory=memory))+'\n')
                path = root/'server/capture/archive.json'
                if path.exists():
                    try:
                        archive = cap.read(path)
                    except (OSError, ValueError):
                        archive = None
                    if archive is not None:
                        cap.require(archive.get('archive_backend') == 'cpu' and
                            archive.get('disk_read_bytes') == archive.get('disk_write_bytes') == 0 and
                            archive.get('repository_gc') is False, 'Complete CPU-only history contract changed')
                await asyncio.sleep(1)
        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=5), trust_env=False,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32)) as client:
                url = f"http://127.0.0.1:{p['port']}"
                for _ in range(1800):
                    try:
                        if (await client.get(url+'/health', timeout=2)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError('Server startup')
                capture = root/'server/capture'
                geometry = cap.read(capture/'cache-geometry.json')
                cap.require(geometry.get('archive_codec') == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1' and
                    geometry.get('cpu_archive_bytes_per_token') == cap.ARCHIVE_BYTES_PER_TOKEN and
                    geometry.get('select_budget') == 65536 and geometry.get('gen_budget') == 32768,
                    'Actual FP8 geometry differs from qualification budget')
                async def snapshot():
                    begin = time.monotonic()
                    response = await client.get(url+'/metrics', timeout=5)
                    response.raise_for_status()
                    row = dict(started_at=begin, finished_at=time.monotonic(), values=ready.native.metrics(response.text))
                    with (root/'native_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    return row
                before = await snapshot()
                cap.require(all(before['values'][k] == 0 for k in
                    ('vllm:num_requests_running', 'vllm:num_requests_waiting')), 'Server is not initially quiescent')
                cap.save(root/'status.json', dict(status='running', pid=server.pid, at=time.time()))
                result = await workload(client, p, cap.read(root/'fixture.json'), root/'workload', capture)
                result['native_reconciliation'] = await cap.reconcile_outputs(snapshot, before,
                    result['total_stream_output_tokens'], timeout=60.)
                result['geometry'] = geometry
                cap.save(root/'results.json', result)
                return result
        try:
            return await ready.supervise_monitored(measure(), guard(), monitor_name='GPU/memory/history guard')
        finally:
            await asyncio.to_thread(ready.bench.q.stop_owned, server)
            cap.save(root/'timing.json', dict(started_wall=wall, finished_wall=time.time(),
                elapsed_seconds=time.monotonic()-started, scope='Launch through owned process cleanup'))


async def run(args):
    import psutil
    root = args.output.resolve()
    m = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    # Before idle waits or Popen: pending or invalid byte evidence cannot launch.
    cap.save(root/'byte-qualification-proof.json', completed_byte_proof(m))
    owner = psutil.Process()
    cap.save(root/'owner.json', dict(pid=owner.pid, create_time=owner.create_time(), argv=sys.argv, at=time.time()))
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, active.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            result = await run_server(root, m)
        cap.save(root/'status.json', dict(status='completed_pending_review' if result['status'] == 'passed'
            else 'coverage_insufficient', result_status=result['status'], at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--byte-qualification', type=Path)
    parser.add_argument('--port', type=int, default=18546)
    args = parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(all((args.config, args.fixture, args.candidate, args.byte_qualification)), 'Explicit frozen inputs required')
    cap.require(1024 <= args.port <= 65535, 'Invalid dedicated server port')
    prepare(args)


if __name__ == '__main__':
    main()
