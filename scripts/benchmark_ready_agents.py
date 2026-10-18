"""Frozen continuous real-tool trace replay, native/base/pipelined KVMem.

Output lengths and external tool messages/waits are frozen. Accepted outputs
come from each tested model. This measures service efficiency, not task quality.
"""
import argparse
import asyncio
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/file)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


cap = module('ready_capacity', 'benchmark_capacity_scaling.py')
bench = module('ready_timed', 'benchmark_timed_agents.py')
native = module('ready_native', 'benchmark_native_agents.py')
work = module('ready_workload', 'timed_agent_workload.py')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


async def supervise_monitored(workload, monitor, *, monitor_name='telemetry'):
    """Fail the workload immediately if its required monitor stops or fails.

    Await both tasks on every exit so a failed monitor cannot leave model
    requests running. Raising the original monitor exception also preserves
    the failure that the server owner records before stopping its own process.
    """
    worker=asyncio.create_task(workload)
    observer=asyncio.create_task(monitor)
    try:
        await asyncio.wait((worker,observer),return_when=asyncio.FIRST_COMPLETED)
        if observer.done():
            await observer
            raise RuntimeError(f'{monitor_name} stopped before workload completion')
        return await worker
    finally:
        for task in (worker,observer):
            if not task.done():task.cancel()
        await asyncio.gather(worker,observer,return_exceptions=True)


def execution_profile(directory, window, gpu_rows, users):
    """Separate sampled demand from actual completed public GPU batches."""
    lo, hi = window['begin'], window['end']
    traces=cap.read(directory/'tools/traces.json')
    observations=[r for r in gpu_rows if lo<=r['at']<hi]
    low=[r for r in observations if r['utilization']<10]
    demanded=[r for r in low if any(t['begin']<=r['at']<t['end'] for t in traces)]
    capture=directory/'server/capture'
    steps=[r for r in records(capture/'model_step_latency.jsonl') if lo<=r['started_at']<hi]
    pure=[r for r in steps if r['decode_requests'] and not r['prefill_requests'] and not r['prefill_tokens']]
    histogram={str(n):sum(r['decode_requests']==n for r in pure) for n in range(1,users+1)}
    restores=[r for r in records(capture/'async_restore.jsonl') if lo<=r['started']<hi]
    return dict(sample_count=len(observations),low_utilization_samples=len(low),
        low_utilization_with_http_samples=len(demanded),pure_decode_steps=len(pure),
        pure_decode_batch_histogram=histogram,
        mean_decode_batch_per_pure_step=sum(r['decode_requests'] for r in pure)/len(pure) if pure else None,
        restore_count=len(restores),resident_restores=sum(r.get('lane')=='resident' for r in restores),
        restore_seconds_p95=work.percentile([r['finished']-r['started'] for r in restores],.95),
        maintenance_host_ms_p95=work.percentile([r['maintenance_host_ms'] for r in pure],.95),
        scope='Batch histogram counts completed GPU steps, not wall-time occupancy. '
              'NVML low-utilization with HTTP demand is not proof of GPU-ready work. '
              'Native same-step batch is unavailable here, never inferred from running gauges.')


def fixture_bounds(fixture, users, turns, ceiling=262144):
    cap.require(type(fixture.get('eos')) is int and fixture['eos']>=0,
                'Frozen fixture requires a validated EOS token before GPU admission')
    plans = work.validate_agent_plans(fixture, concurrency=users, warmup_turns=2,
                                     epoch_turns=turns, gen_budget=32768)
    bounds = [len(p['initial_tokens']) + sum(t['output_tokens'] + len(t['tool_tokens'])
              for t in p['turns'][:turns]) for p in plans]
    cap.require(all(n <= ceiling for n in bounds), 'Frozen epoch would exceed full historical ceiling')
    # History remains intact until the declared epoch finishes normally. The
    # finite trace supplies a tighter bound than ten hypothetical full256K users.
    archive = sum(((n+31)//32*32+64)*cap.ARCHIVE_BYTES_PER_TOKEN for n in bounds)
    cap.require(archive <= 100_000_000_000, 'Full CPU history would exceed configured cap')
    return dict(history_limits=bounds, archive_bytes=archive,
                mean_k_index_bytes=users*cap.INDEX_BYTES_PER_REQUEST,
                extra_bytes=32*1024**3, minimum_free_bytes=8*1024**3,
                required_available_bytes=archive+users*cap.INDEX_BYTES_PER_REQUEST+32*1024**3)


def native_sequence_limit(value, users):
    """Keep client demand fixed while optionally tuning native active slots."""
    if value is None:
        return users
    cap.require(type(value) is int and 1 <= value <= users,
                'Native active sequence limit must be between one and client concurrency')
    return value


def prepare(a):
    root = a.output.resolve()
    cap.require(not root.exists(), 'Use a new output directory')
    profile, fixture = cap.read(a.config), cap.read(a.fixture)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(profile['tokenizer'],local_files_only=True)
    eos=tokenizer.eos_token_id
    cap.require(type(eos) is int and eos>=0, 'Tokenizer requires one explicit EOS token')
    cap.require('eos' not in fixture or fixture['eos']==eos, 'Fixture EOS differs from local tokenizer')
    fixture['eos']=eos
    budget = fixture_bounds(fixture, a.users, a.turns)
    native_seqs = native_sequence_limit(getattr(a, 'native_max_num_seqs', None), a.users)
    cap.require(profile['kvmem']['select_budget']==65536 and profile['kvmem']['gen_budget']==32768
        and profile['max_model_len']==262144 and profile['kvmem']['archive_backend']=='cpu'
        and profile['kvmem']['cpu_read_cache_bytes']==100_000_000_000, 'Require K64/G32/full256K/CPU100GB')
    cap.require(not a.qualify or a.arms==['kvmem-ready'], 'Byte oracle qualification runs only the candidate')
    root.mkdir(parents=True)
    profile.update(port=a.port, max_num_seqs=a.users, verify_session_restore=a.qualify)
    for folder in ('src','scripts','csrc'):
        shutil.copytree(ROOT/folder, root/'frozen'/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    runtime_hashes = {}
    for arm, source in [('kvmem-base', a.baseline), ('kvmem-ready', a.candidate)]:
        shutil.copytree(source.resolve(), root/'runtime'/arm,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        runtime_hashes[arm] = cap.hashes(root/'runtime'/arm)
    profiles = {}
    for arm in a.arms:
        p = json.loads(json.dumps(profile))
        if arm=='native':
            p['max_num_seqs']=native_seqs
        if arm=='kvmem-ready':
            p['kvmem']['resident_restore_slots']=a.restore_slots
        cap.save(root/(arm+'.json'),p)
        profiles[arm]=cap.sha(root/(arm+'.json'))
    cap.save(root/'fixture.json',fixture)
    cap.save(root/'manifest.json',dict(kind='ready_agents_v1', created_at=time.time(),
        source_sha256=cap.hashes(root/'frozen'), runtime_sha256=runtime_hashes,
        profiles_sha256=profiles, fixture_sha256=cap.sha(root/'fixture.json'),
        fixture_source=str(a.fixture.resolve()), fixture_source_sha256=cap.sha(a.fixture),
        workload=dict(concurrency=a.users, warmup_turns=2, epoch_turns=a.turns,
                      duration_seconds=a.duration, mode='tools'),
        arms=a.arms, port=a.port, idle_seconds=10, budget=budget, qualify=a.qualify,
        native_max_num_seqs=native_seqs,
        qualification=str(a.qualification.resolve()) if a.qualification else None,
        native_source=cap.native_source_identity(),
        scope='Identical recorded external tool messages, waits and generation quotas; own accepted outputs. '
              'Native async/APC and KVMem deployment comparison, not on-policy agent quality. '
              'Warmup excluded; cold epoch replacements inside the window remain charged.'))
    cap.save(root/'status.json',dict(status='prepared',at=time.time()))


def verify(root):
    m=cap.read(root/'manifest.json')
    cap.require(ROOT.resolve()==(root/'frozen').resolve(), 'Run the frozen entry point')
    cap.require(m['kind']=='ready_agents_v1' and cap.hashes(root/'frozen')==m['source_sha256'], 'Frozen source changed')
    cap.require(cap.sha(root/'fixture.json')==m['fixture_sha256'], 'Frozen fixture changed')
    cap.require(all(cap.sha(root/(arm+'.json'))==sha for arm,sha in m['profiles_sha256'].items()), 'Profile changed')
    cap.require(all(cap.hashes(root/'runtime'/arm)==sha for arm,sha in m['runtime_sha256'].items()), 'Runtime changed')
    cap.require(cap.native_source_identity()==m['native_source'], 'Installed native implementation changed')
    return m


async def await_launch_admission(root, arm, m, *, prelaunch_check=None):
    """Finish process-based device probes before the final host admission wait."""
    verify(root)
    await bench.q.wait_idle(root,m['idle_seconds'])
    cap.require(not bench.q.gpu_state()['pids'], 'Foreign GPU process before launch')
    if prelaunch_check is not None:
        await prelaunch_check(root, arm, m)
    cap.require(cap.host_memory()['MemAvailable']>=m['budget']['required_available_bytes'], 'Insufficient host headroom')


async def run_arm(root, arm, m, *, prelaunch_check=None):
    import httpx
    await await_launch_admission(root, arm, m, prelaunch_check=prelaunch_check)
    directory=root/arm;directory.mkdir()
    profile=cap.read(root/(arm+'.json'));is_native=arm=='native'
    controls=dict(MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1')
    if is_native:
        launch=native.launch.build_launch(profile,port=m['port'],async_scheduling=True,
                                          environ={**os.environ,**controls})
        command,environment,cwd=launch['argv'],launch['environment'],directory
    else:
        cwd=root/'runtime'/arm
        environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('VLLM_','KVMEM_'))}
        environment.update(PYTHONPATH=str(cwd/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-m','vllm_kvmem','serve','--config',str(root/(arm+'.json')),
                 '--run-dir',str(directory/'server')]
    environment.update(controls,VLLM_LOG_STATS_INTERVAL='1')
    # Match the previously qualified toolchain, including any FlashInfer cache keys.
    for toolkit in ('/usr/local/cuda-13','/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').is_file():
            environment['CUDA_HOME']=toolkit;break
    cap.save(directory/'launch.json',dict(argv=command,cwd=str(cwd),native=is_native))
    gpu_rows=[];samples=[]
    with (directory/'server.log').open('x') as log:
        arm_started_wall,arm_started_monotonic=time.time(),time.monotonic()
        server=subprocess.Popen(command,cwd=cwd,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        cap.save(root/'status.json',dict(status='loading',arm=arm,pid=server.pid,at=time.time()))

        async def guard():
            while True:
                state=await asyncio.to_thread(bench.q.gpu_state)
                cap.require(not(set(state['pids'])-bench.q.owned_pids(server)), 'Foreign GPU process appeared')
                cap.require(server.poll() is None, 'Server exited; preserve logs')
                memory=cap.host_memory()
                cap.require(memory['MemAvailable']>=m['budget']['minimum_free_bytes'], 'Host available memory below8GiB')
                row=dict(at=time.monotonic(),wall_at=time.time(),**state,memory=memory)
                gpu_rows.append(row)
                with (directory/'gpu_samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                await asyncio.sleep(1)

        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600,connect=5),trust_env=False,
                    limits=httpx.Limits(max_connections=64,max_keepalive_connections=64)) as client:
                url=f"http://127.0.0.1:{m['port']}"
                for _ in range(1800):
                    try:
                        if (await client.get(url+'/health',timeout=2)).status_code==200:break
                    except httpx.HTTPError:pass
                    await asyncio.sleep(1)
                else:raise TimeoutError('Server startup')
                capture=directory/'server/capture'
                published=set()
                async def snapshot():
                    response=await client.get(url+'/metrics',timeout=5);response.raise_for_status()
                    row=dict(started_at=time.monotonic(),raw_metrics=response.text,values=native.metrics(response.text))
                    row['finished_at']=time.monotonic();samples.append(row)
                    with (directory/'native_samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    if not is_native and (capture/'scheduler_resources.json').exists():
                        try:
                            resources=cap.read(capture/'scheduler_resources.json')
                            with (directory/'resources_samples.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=time.monotonic(),resources=resources))+'\n')
                        except (OSError,ValueError):pass
                    if not is_native and (capture/'archive.json').exists():
                        try:
                            archive=cap.read(capture/'archive.json')
                        except (OSError,ValueError):
                            archive=None
                        if archive is not None:
                            cap.require(archive.get('archive_backend')=='cpu' and
                                archive.get('disk_read_bytes')==archive.get('disk_write_bytes')==0 and
                                archive.get('repository_gc') is False, 'CPU-only, non-GC history contract changed')
                            summary={k:archive.get(k) for k in ('cpu_authority_bytes','cpu_cache_used_bytes',
                                'session_checkpoint_payload_bytes','pending_downloads','active_requests')}
                            with (directory/'history_samples.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=time.monotonic(),**summary))+'\n')
                            for complete in archive.get('completed',[]):
                                key=(complete['request_id'],complete.get('checkpoint_tokens'))
                                if key not in published:
                                    with (directory/'history_publications.jsonl').open('a') as stream:
                                        stream.write(json.dumps(dict(at=time.monotonic(),**complete))+'\n')
                                    published.add(key)
                    return row
                async def quiescent(*,stage,not_before):
                    for _ in range(240):
                        row=await snapshot()
                        if row['started_at']>=not_before and all(row['values'][k]==0 for k in
                                ('vllm:num_requests_running','vllm:num_requests_waiting')):
                            return dict(stage=stage,native_metrics=row)
                        await asyncio.sleep(.25)
                    raise TimeoutError('Native drain')
                async def telemetry():
                    while True:
                        await snapshot();await asyncio.sleep(1)
                await quiescent(stage='before',not_before=time.monotonic())
                async def measured_workload():
                    cap.save(root/'status.json',dict(status='running',arm=arm,pid=server.pid,at=time.time()))
                    args=SimpleNamespace(**m['workload'],run_tag=arm,capture_directory=capture,native_quiescent=quiescent)
                    result=await work.workload(client,dict(profile,native_vllm=is_native),cap.read(root/'fixture.json'),
                                               directory/'tools',args)
                    final=await quiescent(stage='after',not_before=time.monotonic())
                    last=final['native_metrics']['values']
                    cap.require(last['vllm:generation_tokens_total']==result['total_stream_output_tokens'], 'Client/native token mismatch')
                    window=cap.read(directory/'tools/window.json')
                    result.update(gpu=bench.gpu_summary(gpu_rows,window['begin'],window['end']),
                        native_boundaries=bench.native_boundaries(samples,window['begin'],window['end']),
                        preemptions=last['vllm:num_preemptions_total'],native_client_tokens_equal=True)
                    result['execution_profile']=execution_profile(directory,window,gpu_rows,m['workload']['concurrency'])
                    if not is_native:
                        cap.require(last['vllm:num_preemptions_total']==0, 'KVMem preemption')
                        archive=cap.read(capture/'archive.json')
                        cap.require(archive['disk_read_bytes']==archive['disk_write_bytes']==0, 'CPU-only history required')
                        result['archive_final']=archive
                        if (capture/'public_work.jsonl').exists():
                            result['public_work']=bench.public_work_summary(records(capture/'public_work.jsonl'),
                                                                           window['begin'],window['end'])
                        if m['qualify']:
                            proof=capture/'verified_restores.jsonl'
                            cap.require(proof.exists() and not(capture/'restore_mismatch.json').exists(), 'Restore oracle absent/failed')
                            result['byte_oracle_records']=len(proof.read_text().splitlines())
                            resources=cap.read(capture/'scheduler_resources.json')
                            result['restore_admission_final']=resources['restore_admission']
                            cap.require(resources['restore_admission']['peak_occupied']['resident']>1,
                                        'Qualification never overlapped independent resident imports')
                    return result
                result=await supervise_monitored(measured_workload(),telemetry())
                cap.save(directory/'results.json',result)
                return result
        try:
            return await supervise_monitored(measure(),guard(),monitor_name='GPU/memory guard')
        finally:
            await asyncio.to_thread(bench.q.stop_owned,server)
            cap.save(directory/'timing.json',dict(started_wall=arm_started_wall,
                finished_wall=time.time(),elapsed_seconds=time.monotonic()-arm_started_monotonic,
                scope='Server launch through owned-process cleanup; includes loading, warmup, measurement and drain.'))


async def run(a):
    root=a.output.resolve();m=verify(root)
    cap.require(cap.read(root/'status.json')['status']=='prepared','No automatic restart')
    if not m['qualify']:
        cap.require(m.get('qualification'), 'Performance requires explicit completed GPU qualification')
        q=Path(m['qualification']);qm=cap.read(q/'manifest.json')
        cap.require(cap.read(q/'status.json')['status']=='completed_pending_review', 'GPU qualification incomplete')
        cap.require(qm['runtime_sha256']['kvmem-ready']==m['runtime_sha256']['kvmem-ready'], 'Qualified candidate differs')
        capture=q/'kvmem-ready/server/capture'
        states=records(capture/'verified_state_restores.jsonl')
        target=records(capture/'verified_restores.jsonl')
        resources=cap.read(capture/'scheduler_resources.json')
        cap.require(states and target and all(r['byte_equal'] for r in target)
            and all(r['recurrent_byte_equal'] for r in states)
            and resources['restore_admission']['peak_occupied']['resident']>1
            and not(capture/'restore_mismatch.json').exists(), 'GPU qualification lacks independent overlapping byte proofs')
        cap.save(root/'qualification-proof.json',dict(source=str(q),
            verified_target_restores=len(target),verified_state_restores=len(states),
            peak_resident_permits=resources['restore_admission']['peak_occupied']['resident'],
            result_sha256=cap.sha(q/'kvmem-ready/results.json'),at=time.time()))
    loop=asyncio.get_running_loop();active=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,active.cancel)
    run_started_wall,run_started_monotonic=time.time(),time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            results={}
            for arm in m['arms']:
                results[arm]=await run_arm(root,arm,m)
                cap.save(root/'comparison.json',{k:dict(output_tps=v['output_tokens_per_second'],
                    gpu=v['gpu'],preemptions=v['preemptions']) for k,v in results.items()})
            cap.save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)
        cap.save(root/'timing.json',dict(started_wall=run_started_wall,
            finished_wall=time.time(),elapsed_seconds=time.monotonic()-run_started_monotonic,
            scope='Execution after qualification validation, including all arms and their cleanup.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run-prepared',action='store_true')
    p.add_argument('--config',type=Path);p.add_argument('--fixture',type=Path)
    p.add_argument('--baseline',type=Path);p.add_argument('--candidate',type=Path)
    p.add_argument('--users',type=int,default=10);p.add_argument('--turns',type=int,default=16)
    p.add_argument('--native-max-num-seqs',type=int,
                   help='Native server active slot limit; all --users clients still submit and queue')
    p.add_argument('--duration',type=float,default=300);p.add_argument('--restore-slots',type=int,default=4)
    p.add_argument('--port',type=int,default=18530);p.add_argument('--qualify',action='store_true')
    p.add_argument('--qualification',type=Path)
    p.add_argument('--arms',nargs='+',choices=('native','kvmem-base','kvmem-ready'),default=['native','kvmem-base','kvmem-ready'])
    a=p.parse_args()
    if a.run_prepared:return asyncio.run(run(a))
    cap.require(all((a.config,a.fixture,a.baseline,a.candidate)), 'Explicit profile, fixture, runtimes required')
    cap.require(0<a.duration<3601 and 1<=a.restore_slots<=32 and len(set(a.arms))==len(a.arms), 'Invalid execution settings')
    prepare(a)


if __name__=='__main__':main()
