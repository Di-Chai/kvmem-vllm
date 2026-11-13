"""Bounded Nsight diagnosis before deciding whether CPU batching is useful."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path[:0]=[str(Path(__file__).resolve().parent),str(REPO/'scripts')]
from audit_layerwise_gilopt import read,save_new,sha,tree,compare
from profile_prefill_nsight import nsys_command,pressure_config,post

SOURCE=REPO/'artifacts/layerwise-physical-prefill-20261105-r1/merged-host128-r2/candidate'
HELPERS=('prefill_phase_probe.py','prefill_pressure_probe.py',
         'layerwise_pressure_phase_probe.py','prefill_commit_phase_probe.py')


def prepare(root):
    root.mkdir(parents=True,exist_ok=False);candidate=root/'candidate';candidate.mkdir()
    assert read(SOURCE/'status.json')['status']=='completed'
    proof=read(SOURCE/'runtime-proof.json');assert tree(SOURCE/'runtime-v1')==proof
    shutil.copytree(SOURCE/'runtime-v1',candidate/'runtime-v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('fixture.json','runtime-proof.json'):shutil.copy2(SOURCE/name,candidate/name)
    for name in HELPERS:shutil.copy2(REPO/'scripts'/name,root/name)
    profile=read(SOURCE/'profile.json')
    assert profile['kvmem']['prefill_physical_coalescing']
    save_new(candidate/'profile.json',profile)
    save_new(candidate/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
    save_new(candidate/'status.json',dict(status='prepared'))
    (root/'probe').mkdir();shutil.copy2(__file__,root/'launch.py')
    files={str(candidate/n):sha(candidate/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')}
    files.update({str(root/n):sha(root/n) for n in (*HELPERS,'launch.py')})
    files.update({b['request']:b['sha256'] for b in read(candidate/'fixture.json')['bindings']})
    save_new(root/'plan.json',dict(baseline=str(SOURCE),files=files,
        window=pressure_config(90000,109600,15,411648),nsys='/usr/local/bin/nsys',
        profiler=dict(profiler='cuda',delay_iterations=0,max_iterations=0,detailed_trace_annotation=True),
        diagnostic_only=True,performance_eligible=False,formal_resume=False,external_grading=False,
        scope='Unmodified qualified 8K runtime and native binary; only diagnostic wrappers/worker and CUDA profiler. One complete original MAB input, bounded steady prefill plus pressure trace. No CPU batching implementation.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json')==read(root/'prepared-proof.json')['plan_sha256']
    plan=read(root/'plan.json');candidate=root/'candidate'
    for path,digest in plan['files'].items():assert sha(Path(path))==digest,path
    assert tree(candidate/'runtime-v1')==read(candidate/'runtime-proof.json')
    return plan


def memory_driver(root):
    runtime=root/'candidate/runtime-v1'
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts')]
    spec=importlib.util.spec_from_file_location('commit_memory',runtime/'scripts/run_layerwise_memory.py')
    memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    return memory


def serve(root):
    plan=verify(root);memory=memory_driver(root);candidate=root/'candidate'
    args=memory.server_command(candidate,read(candidate/'fixture.json'))[5:]
    assert args[0]=='--config'
    from vllm_kvmem.serving.serve import main as build
    output=io.StringIO()
    with contextlib.redirect_stdout(output):build([*args,'--dry-run'])
    launch=json.loads(output.getvalue());argv=launch['vllm_argv']
    assert '--worker-cls' not in argv and '--profiler-config' not in argv
    argv+=['--worker-cls','prefill_commit_phase_probe.PrefillCommitPhaseWorker',
           '--profiler-config',json.dumps(plan['profiler'])]
    for name in list(os.environ):
        if name.startswith('KVMEM_'):del os.environ[name]
    os.environ.update(launch['environment'])
    os.environ['PYTHONPATH']=str(root)+os.pathsep+os.environ.get('PYTHONPATH','')
    sys.path.insert(0,str(root))
    os.environ['PREFILL_PRESSURE_RANGE']=json.dumps(plan['window'])
    os.environ['PREFILL_PHASE_PROBE_OUTPUT']=str(root/'probe')
    if launch['profile']['model_adapter']=='qwen38_flash_next_b12x':
        from vllm_kvmem.adapters.qsa_config_compat import enable
        enable()
    if launch['profile']['draft_method']=='dflash':
        from vllm_kvmem.adapters.compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    (candidate/'run/server/capture').mkdir(parents=True,exist_ok=False)
    save_new(candidate/'run/server/launch.json',launch)
    save_new(root/'server-owned.json',dict(pid=os.getpid(),pgid=os.getpgrp()))
    from vllm.entrypoints.cli.main import main
    sys.argv=['vllm',*argv];main()


def collect(root):
    plan=verify(root);candidate=root/'candidate';memory=memory_driver(root)
    def frozen(path):
        assert path==candidate;verify(root)
        return read(candidate/'fixture.json'),read(candidate/'profile.json')
    memory.verify=frozen
    argv=nsys_command(plan['nsys'],root/'trace',
        [sys.executable,'-B',str(root/'launch.py'),'serve','--root',str(root)])
    memory.server_command=lambda *_:argv
    original_infer=memory.infer
    def infer_after_warmup(*args,**kwargs):
        # memory.run only calls inference after server health is ready. Do not
        # arm at load_model: native warmup also calls execute_model with dummies.
        post(read(candidate/'profile.json')['port'],'start_profile')
        return original_infer(*args,**kwargs)
    memory.infer=infer_after_warmup
    # Nsight can finish its bounded trace before the still-running application.
    # Keep application cleanup inside memory.run's original GPU ownership lock.
    import run_recall_comparison as recall
    import benchmark_block_graph as lifecycle
    from types import SimpleNamespace
    original_stop=recall.stop_owned
    def stop_profiled_server(server):
        if server is None:return
        owned=root/'server-owned.json';application=None
        if owned.exists():
            info=read(owned)
            if info['pgid']==os.getpgrp():raise RuntimeError('Unsafe application process group')
            try:os.kill(info['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            if info['pgid']!=server.pid:
                application=SimpleNamespace(pid=info['pgid'],poll=lambda:None)
        try:server.wait(timeout=45)
        except subprocess.TimeoutExpired:pass
        remaining=lifecycle.stop_owned_groups((server,application))
        save_new(root/'cleanup.json',dict(remaining_owned_groups=remaining,
            gpu_processes=sorted(lifecycle.gpu_processes())))
        if remaining:raise RuntimeError('Diagnostic application survived cleanup')
    recall.stop_owned=stop_profiled_server
    save_new(root/'nsys-command.json',dict(argv=argv))
    (root/'status.json').write_text(json.dumps(dict(status='running')))
    try:
        memory.run(candidate)
        save_new(root/'equivalence.json',compare(Path(plan['baseline']),candidate))
        assert (root/'trace.nsys-rep').exists()
        subprocess.run([plan['nsys'],'export','--type=sqlite','--force-overwrite=false',
            '-o',str(root/'trace.sqlite'),str(root/'trace.nsys-rep')],check=True)
        (root/'status.json').write_text(json.dumps(dict(status='completed')))
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error))))
        raise
    finally:
        recall.stop_owned=original_stop


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','collect','serve'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.root.resolve())
