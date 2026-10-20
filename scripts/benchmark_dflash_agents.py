"""Two frozen live-tool DeepSWE windows: optimized MTP4 versus experimental DFlash2."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as ownership
import benchmark_deepswe_first16 as agents
from analyze_deepswe_capacity import analyze_stage

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(args):
    from vllm_kvmem.serve import build_launch
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    qualification=args.qualification.resolve()
    if read(qualification/'status.json')['status']!='passed':
        raise ValueError('Require completed experimental DFlash2 planned state qualification')
    result=read(qualification/'qualification/results.json')
    if result['quality']['correct']!=result['quality']['total'] or not result['all_owned_sessions_released']:
        raise ValueError('Require all narrow synthetic answers and full release')
    frozen=root/'frozen'
    for name in ('src','scripts','benchmarks'):
        shutil.copytree(ROOT/name,frozen/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    state_sources=read(qualification/'manifest.json')['runtime_sha256']
    for name,digest in state_sources.items():
        if name.startswith('src/') and sha(frozen/name)!=digest:
            raise ValueError('Serving source differs from the new state qualification: '+name)
    base=read(args.profile)
    base.update(port=args.port,served_model_name='kvmem-dflash-agent-comparison',max_model_len=262144,
        verify_session_restore=False,query_replay_budget=16384,max_num_seqs=8)
    kv=base['kvmem']
    kv.update(select_budget=36864,gen_budget=16384,recent_blocks=64,
        session_capacity=32,session_checkpoint_bytes=4294967296,cpu_read_cache_bytes=100000000000,
        cpu_history_cache_bytes_per_request=20000000000,request_planning=True,
        async_refresh=False,async_restore=False,maintenance_max_steps=0,
        cpu_history_coordination=True,rolling_active_window=False,
        action_native_decode_commit=True,action_download_pool=True,action_fused_scoring=True,
        graph_native_propagation=True,action_native_scores='off')
    order=['mtp4','dflash7'];profiles={}
    for name in order:
        cfg=json.loads(json.dumps(base));cfg.update(
            draft_method='mtp' if name=='mtp4' else 'dflash',draft_tokens=4 if name=='mtp4' else 7,
            draft_model=None if name=='mtp4' else args.draft.resolve().as_posix(),
            execution_mode='compiled_graph' if name=='mtp4' else 'full_graph',
            experimental_dflash_query_planning=name=='dflash7')
        job=root/name;job.mkdir();shutil.copytree(args.protocol,job/'protocol')
        plan=build_launch(cfg,ROOT,job/'server');cfg=plan['profile']
        save(job/'profile.json',cfg);profiles[name]=sha(job/'profile.json')
        agents.prepare(SimpleNamespace(reference=args.reference.resolve(),config=job/'profile.json',
            output=job,workers=8,bash_repeat_limit=32,carry_result=[],backend_kind='kvmem',
            duration_seconds=args.duration_seconds))
    warmup=args.warmup.resolve();raw=warmup.read_bytes()
    if len(raw)!=67584*4:raise ValueError('Use the unchanged complete 67,584-token initialization fixture')
    (root/'warmup.i32').write_bytes(raw)
    sources={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()}
    inputs={str(p.relative_to(root)):sha(p) for name in order for p in (root/name).rglob('*') if p.is_file()}
    save(root/'manifest.json',dict(order=order,profiles=profiles,source_sha256=sources,input_sha256=inputs,
        qualification=str(qualification),qualification_sha256=sha(qualification/'manifest.json'),
        qualification_result_sha256=sha(qualification/'qualification/results.json'),
        warmup_sha256=sha(root/'warmup.i32'),duration_seconds=args.duration_seconds,
        port=args.port,bridge_port=args.bridge_port,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        scope='Live original first16 Claude Code tasks, eight workers, real tools in fresh owned containers. '
            'Fixed wall window; unfinished attempts censored, no quality score inferred. '
            'Both use all four overhead optimizations and the same V2 graph/attention policy, K/G, 2048-token recent band, '
            'CPU history budget and synchronous refresh/restore. MTP4 uses its qualified compiled mode; '
            'DFlash7 uses experimental FULL graphs and independent BF16 sliding draft state. '
            'This deployment comparison does not isolate the speculative algorithm from its execution mode. '
            'The matched recent band/sync settings differ from previous throughput experiments. No historical result is reused.'))
    save(root/'status.json',dict(status='prepared',at=time.time()))


def run(args):
    root=args.output.resolve();m=read(root/'manifest.json')
    if read(root/'status.json')['status']!='prepared':raise ValueError('No implicit retries or overwritten attempts')
    if ROOT!=root/'frozen':raise ValueError('Run the frozen entry point')
    for name,digest in m['source_sha256'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen source changed')
    for name,digest in m['input_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen input changed')
    results={}
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            for name in m['order']:
                if ownership.gpu_processes():raise RuntimeError('GPU occupied; no foreign process will be stopped')
                job=root/name;command=[sys.executable,'-B',str(ROOT/'scripts/benchmark_deepswe_first16.py'),
                    '--config',str(job/'profile.json'),'--output',str(job),'--workers','8',
                    '--backend-kind','kvmem','--duration-seconds',str(m['duration_seconds']),
                    '--warmup-tokens',str(root/'warmup.i32'),'--bridge-port',str(m['bridge_port']),
                    '--bash-repeat-limit','32','--use-prepared']
                environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_'))}
                environment.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
                    TOKENIZERS_PARALLELISM='false',MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',VLLM_LOG_STATS_INTERVAL='1')
                with ownership.cleanup_on_termination() as cleanup:
                    process=None
                    try:
                        with (root/(name+'.log')).open('x') as log:
                            process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,
                                stderr=subprocess.STDOUT,start_new_session=True)
                            save(root/'status.json',dict(status='running',job=name,pid=process.pid,at=time.time()))
                            code=process.wait()
                    finally:
                        cleanup()
                        if ownership.stop_owned_groups((process,),grace_seconds=60):
                            raise RuntimeError('Owned processes remained')
                state=read(job/'run.json')
                if code or state['status']!='window_complete' or state.get('cleanup_errors'):
                    raise RuntimeError('Agent window failed; retain first attempt: '+name)
                result=analyze_stage(job);save(job/'analysis.json',result);results[name]=result
                print(name+' completed live tool window',flush=True)
        save(root/'analysis.json',dict(stages=results,scope=m['scope'],
            limitation='One bounded live-tool window per deployment; trajectories and speculative acceptance differ. '
                'Do not claim task quality from censored attempts or causally assign all throughput differences.'))
        save(root/'status.json',dict(status='completed',at=time.time()))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_attempt_preserved',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--qualification',type=Path)
    p.add_argument('--profile',type=Path);p.add_argument('--draft',type=Path,default=Path('/home/chaidi/qw3/models/Qwen3.8-27B-DFlash2-BF16'))
    p.add_argument('--reference',type=Path,default=Path('/home/chaidi/qw3/benchmark/action_pressure_first16_warmfix_20261010'))
    p.add_argument('--protocol',type=Path,default=Path('artifacts/deepswe-first16-20261015-r16/protocol'))
    p.add_argument('--warmup',type=Path,default=Path('artifacts/deepswe-capacity-20261017-r3/warmup.i32'))
    p.add_argument('--duration-seconds',type=int,default=300);p.add_argument('--port',type=int,default=18846)
    p.add_argument('--bridge-port',type=int,default=18847);a=p.parse_args()
    if a.duration_seconds<300:p.error('Keep at least a five-minute live-tool window')
    (prepare if a.action=='prepare' else run)(a)
