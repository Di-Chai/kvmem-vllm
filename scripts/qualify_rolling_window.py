"""Qualify rolling active KV with two complete real agent histories.

Preparation is CPU-only. Execute only the prepared frozen entry point. This
checks restore/refresh integrity with fixed output quotas, not task quality or
performance; the GPU byte oracle deliberately introduces synchronization.
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
COUNTS = (1024, 4096, 1024)
USERS = 2


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


shared = module('rolling_ready_shared', 'benchmark_ready_agents.py')
core = module('rolling_capacity_work', 'capacity_scaling_workload.py')
cap, work, bench, native = shared.cap, shared.work, shared.bench, shared.native


def qualification_profile(source, port):
    profile = copy.deepcopy(source)
    cap.require(profile.get('model_format') == 'safetensors' and
        profile.get('draft_method') == 'mtp' and profile.get('draft_tokens') == 4 and
        profile.get('execution_mode') == 'compiled_graph', 'Require compiled native NVFP4 MTP(k=4) profile')
    profile.update(port=port,max_num_seqs=USERS,max_model_len=262144,verify_session_restore=True,
                   query_replay_budget=4096)
    kv = profile['kvmem']
    cap.require(kv.get('block_tokens') == 32 and kv.get('archive_backend') == 'cpu' and
        kv.get('cpu_read_cache_bytes') == 100_000_000_000, 'Require FP8 CPU100GB history profile')
    cap.require(all(kv.get(k) is True for k in ('request_planning','async_refresh','async_restore',
        'cpu_history_coordination','session_capture_terminal','action_pressure')),
        'Require coordinated async pressure and terminal capture')
    cap.require(kv.get('session_checkpoint_policy') == 'pm' and
        not kv.get('session_capture_decode',False), 'Require terminal PM without per-block decode capture')
    kv.update(select_budget=32768,gen_budget=4096,repository_blocks=8192,
              rolling_active_window=True,step_latency_interval=1,
              active_resident_max_fraction=1e-6)
    return profile


def validate_fixture(fixture, profile):
    """Keep original IDs/messages; fixed qualification quotas are separate."""
    from vllm_kvmem.request_plan import RequestPlan
    cap.require(type(fixture.get('eos')) is int and fixture['eos'] >= 0,
                'Validated tokenizer EOS required before GPU admission')
    # Source output quotas are provenance, not the qualification quota. Do not
    # reject the source's unused longer turns against this smaller G=4096.
    plans = work.validate_agent_plans(fixture,concurrency=USERS,warmup_turns=1,
                                     epoch_turns=len(COUNTS),gen_budget=32768)
    bounds, queries = [], []
    for plan in plans:
        tokens = plan['initial_tokens']
        cap.require(len(tokens) >= 131072, 'Require complete original 131K or longer history')
        cap.require(plan.get('source_task') and type(plan.get('source_turn')) is int,
                    'Require original task and turn provenance')
        query = RequestPlan.parse(plan['query_plan'],len(tokens),32,4096,
                                  profile['kvmem'].get('guided_query_tokens',0))
        cap.require(query.query is not None and query.replay_end-query.replay_begin <=
                    profile['query_replay_budget'], 'Complete Query exceeds qualified replay budget')
        # The original semantic validator verifies hard pins against the same
        # K=32K capacity. No Query clipping or pin removal is permitted.
        from vllm_kvmem.config import KVMemConfig
        memory = KVMemConfig(select_budget=32768,gen_budget=4096,block_tokens=32,
            sink_blocks=profile['kvmem'].get('sink_blocks',0),
            recent_blocks=profile['kvmem'].get('recent_blocks',0))
        query.mandatory_fit(len(tokens),memory)
        for turn in plan['turns'][1:len(COUNTS)]:
            cap.require(turn['tool_tokens'] and turn.get('external_message_proof'),
                        'Resume requires original complete external tool messages')
        bound = len(tokens)+sum(COUNTS)+sum(len(t['tool_tokens']) for t in plan['turns'][1:len(COUNTS)])
        cap.require(bound <= profile['max_model_len'], 'Complete qualification history exceeds model ceiling')
        bounds.append(bound)
        queries.append(dict(span=list(query.query),replay_begin=query.replay_begin,replay_end=query.replay_end,
                            initial_tokens=len(tokens),source_task=plan['source_task']))
    cap.require(len({p['source_task'] for p in plans}) == USERS, 'Require independent source tasks')
    archive = sum(((n+31)//32*32+64)*cap.ARCHIVE_BYTES_PER_TOKEN for n in bounds)
    cap.require(archive <= profile['kvmem']['cpu_read_cache_bytes'], 'CPU historical capacity exceeded')
    return dict(history_limits=bounds,queries=queries,archive_bytes=archive,
        minimum_free_bytes=8*1024**3,
        required_available_bytes=archive+USERS*cap.INDEX_BYTES_PER_REQUEST+32*1024**3)


def prepare(args):
    root = args.output.resolve()
    cap.require(not root.exists(), 'Use a fresh qualification directory')
    profile = qualification_profile(cap.read(args.config),args.port)
    fixture = cap.read(args.fixture)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    eos = tokenizer.eos_token_id
    cap.require(type(eos) is int and eos >= 0, 'Tokenizer has no explicit EOS')
    cap.require('eos' not in fixture or fixture['eos'] == eos, 'Fixture EOS differs from tokenizer')
    fixture['eos'] = eos
    budget = validate_fixture(fixture,profile)
    candidate = args.candidate.resolve()
    cap.require((candidate/'src/vllm_kvmem').is_dir(), 'Candidate runtime source is missing')
    root.mkdir(parents=True)
    for folder in ('src','scripts','csrc'):
        shutil.copytree(ROOT/folder,root/'frozen'/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(candidate,root/'runtime',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    cap.save(root/'profile.json',profile)
    cap.save(root/'fixture.json',fixture)
    cap.save(root/'manifest.json',dict(kind='rolling_window_qualification_v1',at=time.time(),
        source_sha256=cap.hashes(root/'frozen'),runtime_sha256=cap.hashes(root/'runtime'),
        profile_sha256=cap.sha(root/'profile.json'),fixture_sha256=cap.sha(root/'fixture.json'),
        fixture_source=str(args.fixture.resolve()),fixture_source_sha256=cap.sha(args.fixture),
        runtime_source=str(candidate),native_source=cap.native_source_identity(),budget=budget,
        output_quotas=list(COUNTS),concurrency=USERS,idle_seconds=10,
        scope='Complete original first two agent histories and external tool messages/waits; own accepted output IDs. '
              'Fixed 1024/4096/1024 generation quotas, byte-oracle correctness qualification, not quality or throughput. '
              'A 1e-6 idle resident fraction forces cold cross-round CPU restores; each continuation requires '
              'an immutable FP8 rebuild oracle. Original tool waits are unchanged.',
        cancellation_scope='No cancellation injection; cancellation correctness requires separate acceptance.'))
    cap.save(root/'status.json',dict(status='prepared',at=time.time()))


def verify(root):
    m = cap.read(root/'manifest.json')
    cap.require(ROOT.resolve() == (root/'frozen').resolve(), 'Run the frozen entry point')
    cap.require(m.get('kind') == 'rolling_window_qualification_v1', 'Unexpected qualification kind')
    cap.require(cap.hashes(root/'frozen') == m['source_sha256'], 'Frozen source changed')
    cap.require(cap.hashes(root/'runtime') == m['runtime_sha256'], 'Candidate runtime changed')
    cap.require(cap.sha(root/'profile.json') == m['profile_sha256'] and
                cap.sha(root/'fixture.json') == m['fixture_sha256'], 'Profile or fixture changed')
    cap.require(cap.native_source_identity() == m['native_source'], 'Installed native implementation changed')
    profile = cap.read(root/'profile.json')
    cap.require(profile == qualification_profile(profile,profile['port']), 'Qualification profile lost required controls')
    cap.require(validate_fixture(cap.read(root/'fixture.json'),profile) == m['budget'], 'Historical bounds changed')
    return m


def history_evidence(rows, publications):
    proofs = []
    for row in rows:
        expected = row['input_tokens']+len(row['output_ids'])-1
        matches = [p for p in publications if core.native_matches(row,p['request_id'])]
        cap.require(matches and all(p.get('history_end') == p.get('checkpoint_tokens') == expected and
            p.get('checkpoint_kind') == 'M' for p in matches), 'Incomplete accepted CPU history publication')
        proofs.append(dict(request_id=row['request_id'],history_tokens=expected,matches=len(matches)))
    return proofs


def oracle_evidence(capture, rows):
    cap.require(not (capture/'restore_mismatch.json').exists(), 'Restore byte mismatch')
    targets = shared.records(capture/'verified_restores.jsonl')
    states = shared.records(capture/'verified_state_restores.jsonl')
    rolling = shared.records(capture/'verified_rolling_pressure.jsonl')
    resumes = [row for row in rows if row['kind'] == 'workload' and row['turn'] > 0]
    cap.require(len(resumes) == USERS*(len(COUNTS)-1), 'Repeated resumes incomplete')
    for row in resumes:
        target = [r for r in targets if core.native_matches(row,r['request_id'])]
        state = [r for r in states if core.native_matches(row,r['request_id'])]
        cap.require(target and state and all(r.get('byte_equal') is True and r.get('mtp_attention_layers',0)>0
            and r.get('working_attention_layers',0)>0 and r.get('target_kv_bytes',0)>0
            and r.get('oracle_kind')=='fp8_immutable_rebuild'
            for r in target) and all(r.get('recurrent_byte_equal') is True for r in state),
            'Each continuation requires cold immutable FP8 target/MTP and recurrent restore byte proofs')
    proven = []
    for user in range(USERS):
        user_rows = [row for row in rows if row['agent'] == user and row['kind'] == 'workload']
        matches = [r for r in rolling if any(core.native_matches(row,r['request_id']) for row in user_rows)]
        valid = []
        for record in matches:
            request = next(row for row in user_rows if core.native_matches(row,record['request_id']))
            step = record.get('previous_accepted_step') or {}
            if (record.get('replay_tokens') == 0 and record.get('cancelled') is False and
                    record.get('history_end',-1) >= request['input_tokens'] and
                    step.get('accepted',0) > 0 and
                    step.get('history_begin',-1)+step.get('accepted',0) == record['history_end'] and
                    record.get('private_forward_steps') == 0 and record.get('mtp_attention_layers',0)>0 and
                    record.get('attention_layers',0)>0 and record.get('recurrent_tensors',0)>0 and
                    record.get('target_kv_bytes',0)>0 and
                    record.get('target_kv_byte_equal') is True and record.get('recurrent_byte_equal') is True):
                valid.append(record)
        cap.require(valid and len(valid)==len(matches),
                    'Each user requires real zero-replay decode pressure with accepted-boundary byte proofs')
        proven.extend(valid)
    cap.require(any(r['previous_accepted_step']['accepted'] > 1 for r in proven),
                'No multi-token MTP acceptance immediately before a pressure boundary')
    steps = shared.records(capture/'model_step_latency.jsonl')
    concurrent = [r for r in steps if r.get('decode_requests') == USERS and
                  r.get('prefill_requests') == 0 and r.get('prefill_tokens') == 0]
    cap.require(concurrent, 'No actual two-request GPU decode batch was observed')
    return dict(target_restore_records=len(targets),recurrent_restore_records=len(states),
                verified_rolling_refreshes=proven,concurrent_decode_steps=len(concurrent),
                cold_restore_required=True,cancellation_validated=False)


async def workload(client,profile,fixture,directory,capture):
    directory.mkdir()
    recorder = work.Recorder(directory,USERS)
    plans = fixture['agent_plans'][:USERS]
    states = [dict(session=f'rolling-qualification-{i}',tokens=list(p['initial_tokens']))
              for i,p in enumerate(plans)]
    publications, proofs = [], []
    try:
        async def call(i,turn,cleanup=False):
            state, plan = states[i],plans[i]
            if turn and not cleanup:
                source = plan['turns'][turn]
                recorder.phase(i,'tool_wait',0,turn)
                await asyncio.sleep(source['tool_wait_seconds'])
                state['tokens'].extend(source['tool_tokens'])
            row = await work.stream_call(client,profile,[fixture['eos']] if cleanup else state['tokens'],
                state['session'],1 if cleanup else COUNTS[turn],recorder,agent=i,epoch=0,turn=turn,
                kind='cleanup' if cleanup else 'workload',request_id=f'rolling-{i}-{turn}-{int(cleanup)}',
                new_input_tokens=1 if cleanup else len(plan['initial_tokens']) if turn==0 else
                    len(plan['turns'][turn]['tool_tokens']),query=not cleanup,
                query_plan=None if cleanup else plan['query_plan'],
                source_metadata=dict(source_task=plan['source_task'],source_turn=plan['turns'][turn]['source_turn'],
                                     original_output_quota=plan['turns'][turn]['output_tokens']))
            if not cleanup:state['tokens'].extend(row['output_ids'])
            return row

        for turn in range(len(COUNTS)):
            # Every offered request drains normally, including when its peer
            # fails. A benchmark deadline never cancels a live model request.
            result = await asyncio.gather(*(call(i,turn) for i in range(USERS)),return_exceptions=True)
            errors = [r for r in result if isinstance(r,BaseException)]
            if errors:raise errors[0]
            state = await work.wait_retirement(capture,timeout=180.,not_before=max(r['begin'] for r in result),
                expected_views={s['session']:len(s['tokens'])-1 for s in states})
            cap.save(directory/f'resources-turn-{turn}.json',state)
            archive = cap.read(capture/'archive.json')
            cap.save(directory/f'archive-turn-{turn}.json',archive)
            publications.extend(archive.get('completed',[]))
            proofs.extend(history_evidence(result,archive.get('completed',[])))
        result = await asyncio.gather(*(call(i,len(COUNTS)-1,True) for i in range(USERS)),return_exceptions=True)
        errors = [r for r in result if isinstance(r,BaseException)]
        if errors:raise errors[0]
        final = await work.wait_retirement(capture,timeout=180.,not_before=max(r['begin'] for r in result),
            expected_views={s['session']:1 for s in states})
        cap.save(directory/'resources-after.json',final)
        oracle = oracle_evidence(capture,recorder.rows)
        return dict(status='passed',history_proofs=proofs,oracle=oracle,
            total_stream_output_tokens=sum(len(r['output_ids']) for r in recorder.rows),
            cleanup_completed=True,cancellation_validated=False)
    finally:
        cap.save(directory/'traces.json',recorder.rows)
        cap.save(directory/'history_publications.json',publications)
        recorder.progress();recorder.stream.close()


async def run_server(root,m):
    import httpx
    import psutil
    await bench.q.wait_idle(root,m['idle_seconds'])
    cap.require(not bench.q.gpu_state()['pids'], 'Foreign GPU process before launch')
    cap.require(cap.host_memory()['MemAvailable'] >= m['budget']['required_available_bytes'],
                'Insufficient complete-history CPU headroom')
    profile = cap.read(root/'profile.json')
    environment = {k:v for k,v in os.environ.items() if not k.upper().startswith(('VLLM_','KVMEM_'))}
    environment.update(PYTHONPATH=str(root/'runtime/src'),PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false',MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',VLLM_LOG_STATS_INTERVAL='1')
    for toolkit in ('/usr/local/cuda-13','/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').is_file():environment['CUDA_HOME']=toolkit;break
    command = [sys.executable,'-m','vllm_kvmem','serve','--config',str(root/'profile.json'),
               '--run-dir',str(root/'server')]
    cap.save(root/'launch.json',dict(argv=command,cwd=str(root/'runtime')))
    with (root/'server.log').open('x') as log:
        server = subprocess.Popen(command,cwd=root/'runtime',env=environment,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
        cap.save(root/'server-owner.json',dict(pid=server.pid,create_time=psutil.Process(server.pid).create_time()))
        cap.save(root/'status.json',dict(status='loading',pid=server.pid,at=time.time()))
        async def guard():
            while True:
                state = await asyncio.to_thread(bench.q.gpu_state)
                cap.require(not(set(state['pids'])-bench.q.owned_pids(server)), 'Foreign GPU process appeared')
                cap.require(server.poll() is None, 'Owned server exited')
                memory = cap.host_memory()
                cap.require(memory['MemAvailable'] >= m['budget']['minimum_free_bytes'], 'Host free memory below8GiB')
                with (root/'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(at=time.monotonic(),wall=time.time(),**state,memory=memory))+'\n')
                archive_path = root/'server/capture/archive.json'
                if archive_path.exists():
                    try:archive=cap.read(archive_path)
                    except (OSError,ValueError):archive=None
                    if archive is not None:
                        cap.require(archive.get('archive_backend')=='cpu' and
                            archive.get('disk_read_bytes')==archive.get('disk_write_bytes')==0 and
                            archive.get('repository_gc') is False, 'Complete CPU-only non-GC history contract changed')
                await asyncio.sleep(1)
        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600,connect=5),trust_env=False) as client:
                url=f"http://127.0.0.1:{profile['port']}"
                for _ in range(1800):
                    try:
                        if (await client.get(url+'/health',timeout=2)).status_code==200:break
                    except httpx.HTTPError:pass
                    await asyncio.sleep(1)
                else:raise TimeoutError('Server startup')
                capture=root/'server/capture'
                geometry=cap.read(capture/'cache-geometry.json')
                cap.require(geometry.get('archive_codec')=='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1' and
                    geometry.get('cpu_archive_bytes_per_token')==cap.ARCHIVE_BYTES_PER_TOKEN and
                    geometry.get('select_budget')==profile['kvmem']['select_budget'] and
                    geometry.get('gen_budget')==profile['kvmem']['gen_budget'],
                    'Actual FP8 geometry differs from qualification budget')
                async def snapshot():
                    started=time.monotonic();response=await client.get(url+'/metrics',timeout=5)
                    response.raise_for_status()
                    row=dict(started_at=started,finished_at=time.monotonic(),values=native.metrics(response.text))
                    with (root/'native_samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    return row
                before=await snapshot()
                cap.save(root/'status.json',dict(status='running',pid=server.pid,at=time.time()))
                result=await workload(client,profile,cap.read(root/'fixture.json'),root/'workload',capture)
                result['native_reconciliation']=await cap.reconcile_outputs(snapshot,before,
                    result['total_stream_output_tokens'],timeout=60.)
                result['geometry']=geometry
                cap.save(root/'results.json',result)
                return result
        try:
            async with asyncio.TaskGroup() as group:
                watcher=group.create_task(guard());task=group.create_task(measure())
                result=await task;watcher.cancel()
            return result
        finally:await asyncio.to_thread(bench.q.stop_owned,server)


async def run(args):
    import psutil
    root=args.output.resolve();m=verify(root)
    cap.require(cap.read(root/'status.json')['status']=='prepared', 'No automatic restart')
    owner=psutil.Process()
    cap.save(root/'owner.json',dict(pid=owner.pid,create_time=owner.create_time(),argv=sys.argv,at=time.time()))
    loop=asyncio.get_running_loop();active=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,active.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await run_server(root,m)
        cap.save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--config',type=Path);parser.add_argument('--fixture',type=Path)
    parser.add_argument('--candidate',type=Path);parser.add_argument('--port',type=int,default=18532)
    args=parser.parse_args()
    if args.run_prepared:return asyncio.run(run(args))
    cap.require(all((args.config,args.fixture,args.candidate)), 'Explicit profile, fixture and candidate required')
    cap.require(1024 <= args.port <= 65535, 'Invalid dedicated server port')
    prepare(args)


if __name__=='__main__':main()
