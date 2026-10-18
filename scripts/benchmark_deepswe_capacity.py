"""Frozen live DeepSWE windows comparing full GPU KV with CPU history/active KV.

The original first16 agents run real tools in fresh owned containers. Formal
windows are bounded; unfinished attempts are censored, never assigned a score.
"""
import argparse
import array
import asyncio
import fcntl
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
STAGES=('native-c4','kvmem-c4','native-c8','kvmem-c8')


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


def require(ok,message):
    if not ok:raise ValueError(message)


def build_controls(cap,workers):
    require(type(workers) is int and 1<=workers<=8,'Compile workers must be between1 and8')
    paths=[importlib.metadata.distribution(package).locate_file(relative) for package,relative in (
        ('flashinfer-python','flashinfer/jit/cpp_ext.py'),('torch','torch/utils/cpp_extension.py'))]
    return dict(environment={'MAX_JOBS':str(workers),'FLASHINFER_NVCC_THREADS':'1'},
        source_sha256={str(p.resolve()):cap.sha(p) for p in paths},
        scope='Bound first-use Ninja/CUDA build concurrency on both backends; inference parameters are unchanged')


def carry_native_window(path,cap,profile,controls,duration,reference):
    path=path.resolve();original=cap.read(path.parent/'manifest.json')
    state=cap.read(path/'run.json');review=cap.read(path/'reviewed-window.json')
    require(path.name=='native-c4' and state.get('workers')==4 and state.get('status')=='window_complete'
        and state.get('cleanup_finished_at') and state.get('cleanup_errors')==[],
        'Carry only a complete, cleaned native4 window')
    require(cap.read(path/'profile.json')==profile and original['build_controls']==controls
        and original['duration_seconds']==duration,'Carried model, budgets or initialization controls differ')
    require(review['result']['whole_window']['seconds']>=duration*.99
        and not review['result']['errors'],'Carried window has inadequate coverage or API errors')
    task_manifest=cap.read(path/'manifest.json');expected=cap.read(reference/'manifest.json')
    require(task_manifest['reference_sha256']==cap.sha(reference/'manifest.json')
        and task_manifest['tasks']==expected['tasks'] and task_manifest['sampling']==expected['sampling'],
        'Carried task set or sampling protocol differs')
    require(cap.hashes(path.parent/'frozen')==original['source_sha256'],'Carried frozen runtime changed')
    for name in ('benchmark_deepswe_first16.py','native_agent_launch.py','capacity_scaling_workload.py'):
        require(cap.sha(path.parent/'frozen/scripts'/name)==cap.sha(ROOT/'scripts'/name),
                'Carried inference/workload implementation changed: '+name)
    require(cap.sha(path.parent/'frozen/benchmarks/deepswe/claude_bridge.py')==
            cap.sha(ROOT/'benchmarks/deepswe/claude_bridge.py'),'Carried protocol transport changed')
    paths=[path.parent/'manifest.json',*(path/name for name in ('manifest.json','profile.json','run.json',
        'reviewed-window.json','server.log','resources.jsonl','resource-guard.jsonl','bridge/requests.jsonl',
        'metrics-before-formal.txt','bounded-warmup.json','capacity-startup.json'))]
    return dict(path=str(path),sha256={str(p):cap.sha(p) for p in paths},
                warmup_sha256=original['input_sha256']['warmup.i32'],
                scope='Reuse the completed native4 measurement; the later foreign-GPU interruption was in KVMem startup, before any KVMem formal task')


def controller_dependency(pid):
    if pid is None:return None
    import psutil
    p=psutil.Process(pid)
    require(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE,'Dependency controller already ended')
    return dict(pid=p.pid,create_time=p.create_time(),argv=p.cmdline(),
        scope='Wait for this exact existing controller to exit before GPU admission; never signal it')


async def wait_dependency(root,owner,cap):
    if owner is None:return
    import psutil
    while True:
        try:
            p=psutil.Process(owner['pid'])
            if p.create_time()!=owner['create_time'] or p.status()==psutil.STATUS_ZOMBIE:break
            require(p.cmdline()==owner['argv'],'Dependency controller command changed unexpectedly')
        except psutil.NoSuchProcess:break
        cap.save(root/'status.json',dict(status='waiting_for_controller',pid=owner['pid'],
            controller_create_time=owner['create_time'],at=time.time()))
        await asyncio.sleep(10)
    cap.save(root/'dependency-ended.json',dict(at=time.time(),owner=owner,
        scope='Original controller identity is no longer live; GPU idle/ownership checks still apply'))


def prepare(a):
    cap=module('deepswe_capacity_geometry','benchmark_capacity_scaling.py')
    controls=build_controls(cap,a.compile_workers)
    require(not a.output.exists(),'Use a fresh artifact; no reuse or automatic retry')
    reference=ROOT/cap.REFERENCE
    require(cap.sha(reference/'manifest.json')==cap.REFERENCE_MANIFEST_SHA,'Reference changed')
    source=cap.hashes(reference/'frozen')
    require(source==cap.read(reference/'manifest.json')['source_sha256'],'Qualified runtime changed')
    profile=cap.capacity_profile(cap.read(reference/'P1.json'),a.port)
    carried=carry_native_window(a.carry_native_c4,cap,profile,controls,a.duration_seconds,a.reference) if a.carry_native_c4 else None
    dependency=controller_dependency(a.wait_for_controller)
    stages=STAGES[1:] if carried else STAGES
    root=a.output.resolve();root.mkdir(parents=True)
    shutil.copytree(reference/'frozen',root/'frozen',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    overlays=('benchmark_capacity_scaling.py','capacity_scaling_workload.py',
              'benchmark_deepswe_first16.py','benchmark_deepswe_capacity.py')
    for name in overlays:shutil.copyfile(ROOT/'scripts'/name,root/'frozen/scripts'/name)
    shutil.copytree(ROOT/'benchmarks/deepswe',root/'frozen/benchmarks/deepswe',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Initialization only: identical immutable 66K token prefix on both engines.
    niah=cap.read(a.warmup_source)
    row=next(r for r in niah['rows'] if r['served_input_tokens']>=67584)
    token_file=a.warmup_source.parent/'inputs'/(row['id']+'.i32')
    require(cap.sha(token_file)==row['served_sha256']==row['token_sha256'],'Warmup source changed')
    (root/'warmup.i32').write_bytes(token_file.read_bytes()[:67584*4])
    if carried:require(cap.sha(root/'warmup.i32')==carried['warmup_sha256'],'Carried initialization tokens differ')
    runner=module('deepswe_window_preparation','benchmark_deepswe_first16.py')
    budgets={}
    for stage in stages:
        directory=root/stage
        directory.mkdir()
        shutil.copytree(a.protocol,directory/'protocol',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        cap.save(directory/'profile.json',profile)
        workers=4 if stage.endswith('c4') else 8
        native=stage.startswith('native-')
        args=SimpleNamespace(reference=a.reference,config=directory/'profile.json',output=directory,
            workers=workers,bash_repeat_limit=32,carry_result=[],backend_kind='native' if native else 'kvmem',
            duration_seconds=a.duration_seconds)
        runner.prepare(args)
        budgets[stage]=cap.host_budget(32*1024**3,8*1024**3,offered_concurrency=workers,native=native)
    inputs={str(p.relative_to(root)):cap.sha(p) for stage in stages
            for p in (root/stage).rglob('*') if p.is_file()}
    inputs['warmup.i32']=cap.sha(root/'warmup.i32')
    cap.save(root/'manifest.json',dict(kind='live_deepswe_capacity_v1',created_at=time.time(),
        source_sha256=cap.hashes(root/'frozen'),input_sha256=inputs,
        qualified_runtime_reference=str(reference),qualified_runtime_sha256=source,
        native_source=cap.native_source_identity(),build_controls=controls,
        stages=list(stages),comparison_stages=list(STAGES),host_budget=budgets,
        carried_windows={'native-c4':carried} if carried else {},controller_dependency=dependency,
        duration_seconds=a.duration_seconds,port=a.port,bridge_port=a.bridge_port,
        idle_seconds=a.idle_seconds,context_ceiling=262144,active_budget=65536,generation_budget=32768,
        cpu_history_cap_bytes=100_000_000_000,clients='Same live first16 tasks, 4 or 8 independent agents; finish one then admit next',
        native_policy='Unmodified native async continuous batching, max_num_seqs8, APC; no artificial 5-agent lifetime gate',
        kvmem_policy='Qualified r3 P1 runtime, synchronous native scheduler, CPU history and K64/G32 active window',
        comparison_scope='Deployment-system comparison; divergent stochastic agent trajectories do not isolate plugin overhead or quality equivalence',
        window_policy='Same fixed wall-clock window, no automatic retry; unfinished attempts censored; context ceiling recorded and worker replenished',
        warmup_scope='Identical 67584+1 initialization and one-token cleanup, excluded from formal counters; warmup prefixes are separate salts'))
    cap.save(root/'status.json',dict(status='prepared',gpu_started=False,at=time.time()))
    cap.save(root/'run-command.json',dict(argv=[sys.executable,str(root/'frozen/scripts/benchmark_deepswe_capacity.py'),
        '--output',str(root),'--run-prepared']))


def verify(root,cap):
    m=cap.read(root/'manifest.json')
    require(ROOT.resolve()==(root/'frozen').resolve(),'Execute the frozen artifact entry point')
    require(m['kind']=='live_deepswe_capacity_v1' and cap.hashes(ROOT)==m['source_sha256'],'Frozen source changed')
    require(all(cap.sha(root/p)==h for p,h in m['input_sha256'].items()),'Frozen inputs changed')
    require(all(cap.sha(Path(p))==h for p,h in m['native_source']['files'].items()),'Native implementation changed')
    require(all(cap.sha(Path(p))==h for p,h in m['build_controls']['source_sha256'].items()),'Compile control implementation changed')
    for row in m.get('carried_windows',{}).values():
        require(all(cap.sha(Path(p))==h for p,h in row['sha256'].items()),'Carried measurement changed')
    return m


async def run(a):
    cap=module('deepswe_capacity_geometry','benchmark_capacity_scaling.py')
    bench=module('deepswe_capacity_shared','benchmark_timed_agents.py')
    root=a.output.resolve();m=verify(root,cap)
    require(cap.read(root/'status.json')['status']=='prepared','No automatic reuse or retry')
    loop=asyncio.get_running_loop();active=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,active.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await wait_dependency(root,m.get('controller_dependency'),cap)
            for stage in m['stages']:
                verify(root,cap)
                await bench.q.wait_idle(root,m['idle_seconds'])
                directory=root/stage;budget=m['host_budget'][stage]
                memory=cap.host_memory()
                cap.save(directory/'host-prelaunch.json',dict(memory=memory,budget=budget))
                require(memory['MemAvailable']>=budget['required_prelaunch_available_bytes'],'Insufficient host headroom for this cohort')
                command=[sys.executable,str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                    '--config',str(directory/'profile.json'),'--output',str(directory),
                    '--workers','4' if stage.endswith('c4') else '8',
                    '--backend-kind','native' if stage.startswith('native-') else 'kvmem',
                    '--duration-seconds',str(m['duration_seconds']),'--warmup-tokens',str(root/'warmup.i32'),
                    '--bridge-port',str(m['bridge_port']),'--bash-repeat-limit','32','--use-prepared']
                environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_'))}
                environment.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
                environment.update(m['build_controls']['environment'])
                with (directory/'controller.log').open('x') as log:
                    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    import psutil
                    cap.save(directory/'owner.json',dict(pid=process.pid,create_time=psutil.Process(process.pid).create_time(),argv=command))
                    cap.save(root/'status.json',dict(status='running',stage=stage,pid=process.pid,at=time.time()))
                    try:
                        while process.poll() is None:
                            state=await asyncio.to_thread(bench.q.gpu_state)
                            memory=cap.host_memory()
                            with (directory/'resource-guard.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=time.time(),gpu=state,memory=memory))+'\n')
                            require(not(set(state['pids'])-bench.q.owned_pids(process)),'Foreign GPU process appeared; stop only the owned run')
                            require(memory['MemAvailable']>=8*1024**3,'Host headroom below8GiB')
                            await asyncio.sleep(1)
                        require(process.returncode==0,'DeepSWE controller failed; preserve all records')
                        state=cap.read(directory/'run.json')
                        require(state['status'] in ('window_complete','completed'),'Window was not completed')
                        require(not state.get('cleanup_errors'),'Owned cleanup incomplete')
                    finally:
                        if process.poll() is None:
                            # Let the controller's container/network finally run.
                            os.killpg(process.pid,signal.SIGTERM)
                            try:await asyncio.to_thread(process.wait,timeout=90)
                            except subprocess.TimeoutExpired:await asyncio.to_thread(bench.q.stop_owned,process)
                cap.save(root/'status.json',dict(status='stage_complete',stage=stage,at=time.time()))
            cap.save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='interrupted' if isinstance(error,asyncio.CancelledError) else 'failed',
            error=repr(error),at=time.time(),no_automatic_retry=True))
        raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true');mode.add_argument('--run-prepared',action='store_true')
    p.add_argument('--reference',type=Path,default=Path('/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010'))
    p.add_argument('--protocol',type=Path,default=ROOT/'artifacts/deepswe-first16-20261015-r16/protocol')
    p.add_argument('--warmup-source',type=Path,default=ROOT/'artifacts/niah256k-attention-20261014/run/manifest.json')
    p.add_argument('--duration-seconds',type=float,default=1800)
    p.add_argument('--idle-seconds',type=float,default=15)
    p.add_argument('--port',type=int,default=18520);p.add_argument('--bridge-port',type=int,default=18521)
    p.add_argument('--compile-workers',type=int,default=4)
    p.add_argument('--carry-native-c4',type=Path)
    p.add_argument('--wait-for-controller',type=int)
    a=p.parse_args()
    require(math.isfinite(a.duration_seconds) and a.duration_seconds>=60,'Window must be finite and >=60s')
    if a.prepare_only:prepare(a)
    else:asyncio.run(run(a))


if __name__=='__main__':main()
