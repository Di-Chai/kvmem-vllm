"""Bounded CUDA/host correlation on the current full-input pressure control.

No inference module is changed. The original compiled eight-slot server runs
one complete original cold input with unchanged sampling, budgets and recall.
Only one finite steady-prefill Nsight window is collected; never a speed test.
"""
import argparse
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

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
SOURCE=REPO/'artifacts/pressure-stage-optimization-20261106-r1/pair-batch-comparison/01-baseline'
CASE='005_Accurate_Retrieval_r005_eventqa_full_q0000'
HELPERS=('profile_prefill_nsight.py','prefill_phase_probe.py','prefill_pressure_probe.py',
    'layerwise_pressure_phase_probe.py','multiowner_publication_probe.py','current_pressure_idle_probe.py',
    'python_gc_phase_probe.py','prefill_commit_phase_probe.py',
    'analyze_prefill_cpu_commits.py','analyze_prefill_nsight.py','analyze_prefill_phases.py',
    'compare_prefill_nsight.py','nsight_prefill_shapes.py')


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def driver(root=None, *, cohort=False):
    return load('current_pressure_driver',Path(root)/'launch.py' if root else
                Path(__file__).with_name('benchmark_pressure_eight.py' if cohort else
                                         'benchmark_pressure_stage_efficiency.py'))


def check_effective_controls(source,actual,fixture,profile, *, case_ids=(CASE,), workers=1):
    """All KVMem switches must match, apart from the recording directory."""
    from copy import deepcopy
    left,right=deepcopy(source['profile']),deepcopy(actual['profile'])
    for value in (left,right):value.pop('port',None)
    if left!=right:
        raise ValueError('Diagnostic changed the effective frozen model/profile')
    def flags(launch):
        return {k:v for k,v in launch['environment'].items()
                if k.startswith('KVMEM_') and k!='KVMEM_RUNNER_PROBE_DIR'}
    if flags(source)!=flags(actual):
        raise ValueError('Diagnostic changed effective KVMem execution flags')
    if (profile['execution_mode'],profile['max_num_seqs'],profile['max_num_batched_tokens'])!=(
            'compiled_graph',8,16424):
        raise ValueError('Diagnostic changed compiled scheduler controls')
    if [profile['kvmem'][k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')]!=[
            36864,16384,32,41,0]:
        raise ValueError('Diagnostic changed frozen KV controls')
    if workers not in (1,8) or len(case_ids)!=workers or len(set(case_ids))!=workers:
        raise ValueError('Diagnostic requires one or eight declared original cold owners')
    if fixture['case_ids']!=list(case_ids) or fixture['cases']!=workers:
        raise ValueError('Diagnostic must run exactly the declared original cold samples')
    return dict(passed=True,effective_profile_equal_except_port=True,
        effective_kvmem_environment_equal_except_capture_path=True,
        compiled_server_slots=8,client_workers=workers,active=36864,gen=16384,
        source_input_and_sampling_unchanged=True,performance_eligible=False)


def prepare(root,source=SOURCE,port=19480, *, cohort=False):
    module=driver(cohort=cohort);base=module.base
    source,root=Path(source).resolve(),Path(root).resolve()
    if base.read(source/'status.json')['status']!='completed':raise ValueError('Source control incomplete')
    flags={k:'0' for k in ('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING',
        'KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION')}
    source_environment = base.read(source/'run/server/launch.json')['environment']
    for name in ('KVMEM_WORKER_STATIC_GC', 'KVMEM_LAYERWISE_PRESSURE_SELECTION_YIELD',
                 'KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION'):
        if name in source_environment:
            flags[name] = source_environment[name]
    case_ids=base.read(source/'fixture.json')['case_ids'] if cohort else [CASE]
    if cohort:
        flags['KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS']='1'
        module.prepare(root,source_arm=source,fixture_path=source/'fixture.json',
            autotune_source=source.parent/'autotune-seed',variants={'baseline':flags},order=('baseline',),port=port)
    else:
        module.prepare(root,source_arm=source,overlays=(),case_ids=(CASE,),
            autotune_source=source.parent/'autotune-seed',variants={'baseline':flags},order=('baseline',),port=port)
    for name in HELPERS:shutil.copy2(Path(__file__).with_name(name),root/name)
    shutil.copy2(__file__,root/'profile-idle.py')
    shutil.copy2(source/'run/server/launch.json',root/'source-launch.json')
    (root/'probe').mkdir()
    diag=dict(source=str(source),arm='01-baseline',case_ids=case_ids,workers=8 if cohort else 1,
        nsys='/usr/local/bin/nsys',
        window=dict(begin_tokens=96000,end_tokens=256000 if cohort else 128000,
                    max_seconds=60 if cohort else 15,observe_seconds=45 if cohort else 8),
        profiler=dict(profiler='cuda',delay_iterations=0,max_iterations=0,detailed_trace_annotation=True),
        worker='current_pressure_idle_probe.CurrentPressureIdleWorker',
        files={name:base.sha(root/name) for name in (*HELPERS,'profile-idle.py','source-launch.json')},
        base_plan_sha256=base.sha(root/'plan.json'),diagnostic_only=True,performance_eligible=False,
        scope=('Eight unchanged full cold MAB inputs and original eight clients; GPU-confirmed Query gate retained. '
               'CUDA/NVTX/OSRT window starts at96K first-owner history and stops after45s (max64 public steps). '
               'Full first outputs retained; equality is reported, not used to retry. No speed claim or formal resume.'
               if cohort else 'One unchanged full cold MAB input; compiled/8-slot server, one client. '
               'CUDA/NVTX/OSRT window starts at96K history and stops after8s at a worker boundary '
               '(max64 public steps). No inference-core patch or formal resume.'))
    base.save(root/'diagnostic-plan.json',diag)
    base.save(root/'diagnostic-proof.json',dict(plan_sha256=base.sha(root/'diagnostic-plan.json')))
    base.save(root/'diagnostic-status.json',dict(status='prepared'))
    return verify(root)[1]


def verify(root):
    module=driver(root);base=module.base;plan=module.verify(root)
    if base.sha(root/'diagnostic-plan.json')!=base.read(root/'diagnostic-proof.json')['plan_sha256']:
        raise ValueError('Diagnostic plan changed')
    diag=base.read(root/'diagnostic-plan.json')
    if base.sha(root/'plan.json')!=diag['base_plan_sha256']:raise ValueError('Underlying trial plan changed')
    for name,digest in diag['files'].items():
        if base.sha(root/name)!=digest:raise ValueError('Diagnostic helper changed: '+name)
    expected_ids=diag.get('case_ids',[diag.get('case_id',CASE)])
    workers=diag.get('workers',1)
    if (plan['changed_runtime_files'] or plan['workers']!=workers or plan['case_ids']!=expected_ids
            or workers not in (1,8) or len(expected_ids)!=workers):
        raise ValueError('Diagnostic must preserve the frozen inference runtime and original input')
    return module,diag,plan


def modules(root):
    module,diag,plan=verify(root);arm=root/diag['arm'];runtime=arm/'runtime-v1'
    sys.path[:0]=[str(root),str(runtime/'src'),str(runtime/'scripts')]
    memory=load('current_pressure_memory',runtime/'scripts/run_layerwise_memory.py')
    return module,diag,plan,arm,runtime,memory


def serve(root,dry_run=False):
    module,diag,plan,arm,runtime,memory=modules(root);base=module.base
    fixture=base.read(arm/'fixture.json')
    args=memory.server_command(arm,fixture)[5:]
    if not args or args[0]!='--config':raise ValueError('Unexpected frozen serving command')
    from vllm_kvmem import serve as serving
    original=serving.build_launch
    def diagnostic_launch(*args,**kwargs):
        launch=original(*args,**kwargs)
        if '--worker-cls' in launch['vllm_argv'] or '--profiler-config' in launch['vllm_argv']:
            raise ValueError('Source already overrides worker/profiler')
        launch['vllm_argv']+=['--worker-cls',diag['worker'],'--profiler-config',json.dumps(diag['profiler'])]
        launch['environment'].update(PREFILL_PRESSURE_RANGE=json.dumps(diag['window']),
            PREFILL_PHASE_PROBE_OUTPUT=str(root/'probe'),
            PYTHONPATH=str(root)+os.pathsep+launch['environment'].get('PYTHONPATH',''))
        return launch
    serving.build_launch=diagnostic_launch
    os.environ['QUERY_STAGE_EXECUTION_JSON']=json.dumps(plan['trials'][0]['environment'])
    if dry_run:
        os.environ.pop('QUERY_STAGE_NATIVE_BINARY_JSON',None)
        os.environ.pop('QUERY_STAGE_NATIVE_AUDIT_OUTPUT',None)
        args+=['--dry-run']
    else:
        base.save(root/'server-owned.json',dict(pid=os.getpid(),pgid=os.getpgrp()))
    shim=load('frozen_query_stage_server',runtime/'scripts/_query_stage_server.py')
    sys.argv=[str(runtime/'scripts/_query_stage_server.py'),*args]
    shim.main()


def collect(root):
    module,diag,plan,arm,runtime,memory=modules(root);base=module.base
    from profile_prefill_nsight import nsys_command,post
    import benchmark_query_fps_prebuild_live as environment_module
    import run_recall_comparison as recall
    import benchmark_block_graph as lifecycle
    base.save(root/'collector-process.json',dict(pid=os.getpid(),at=time.time()))
    def frozen(path):
        if path!=arm:raise ValueError('Unexpected diagnostic arm')
        verify(root);return base.read(arm/'fixture.json'),base.read(arm/'profile.json')
    memory.verify=frozen
    command=nsys_command(diag['nsys'],root/'trace',
        [sys.executable,'-B',str(root/'profile-idle.py'),'serve','--root',str(root)])
    memory.server_command=lambda *_:command
    old_env=environment_module.server_environment;old_stop=recall.stop_owned;old_infer=memory.infer
    native,=list((runtime/'src/vllm_kvmem').glob('_native*.so'))
    def environment(path):
        env=old_env(path)
        env.update(QUERY_STAGE_EXECUTION_JSON=json.dumps(plan['trials'][0]['environment']),
            QUERY_STAGE_NATIVE_BINARY_JSON=json.dumps(dict(path=str(native.resolve()),sha256=base.sha(native))),
            QUERY_STAGE_NATIVE_AUDIT_OUTPUT=str(arm/'run/native-runtime-identity.json'),
            VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=str(arm/'autotune'))
        return env
    environment_module.server_environment=environment
    def infer(*args,**kwargs):
        launch=base.read(arm/'run/server/launch.json')
        audit=check_effective_controls(base.read(root/'source-launch.json'),launch,
            base.read(arm/'fixture.json'),base.read(arm/'profile.json'),
            case_ids=plan['case_ids'],workers=plan['workers'])
        base.save(root/'effective-control-audit.json',audit)
        post(base.read(arm/'profile.json')['port'],'start_profile')
        return old_infer(*args,**kwargs)
    memory.infer=infer
    def stop(server):
        if server is None:return
        application=None
        if (root/'server-owned.json').exists():
            owned=base.read(root/'server-owned.json')
            if owned['pgid']==os.getpgrp():raise RuntimeError('Unsafe diagnostic process group')
            try:os.kill(owned['pid'],signal.SIGTERM)
            except ProcessLookupError:pass
            if owned['pgid']!=server.pid:application=SimpleNamespace(pid=owned['pgid'],poll=lambda:None)
        try:server.wait(timeout=45)
        except subprocess.TimeoutExpired:pass
        remaining=lifecycle.stop_owned_groups((server,application))
        base.save(root/'cleanup.json',dict(remaining_owned_groups=remaining,
            gpu_processes=sorted(lifecycle.gpu_processes())))
        if remaining:raise RuntimeError('Diagnostic GPU owner survived cleanup')
    recall.stop_owned=stop
    base.save(root/'nsys-command.json',dict(argv=command,controller_injected=False))
    base.save(root/'diagnostic-status.json',dict(status='running'),replace=True)
    try:
        memory.run(arm,workers=plan['workers'])
        from audit_layerwise_gilopt import compare
        equivalence=compare(Path(diag['source']),arm,plan['case_ids'])
        base.save(root/'equivalence.json',equivalence)
        report=root/'trace.nsys-rep'
        if not report.exists():raise RuntimeError('Missing Nsight GPU report')
        subprocess.run([diag['nsys'],'export','--type=sqlite','--force-overwrite=false',
            '-o',str(root/'trace.sqlite'),str(report)],check=True,timeout=180)
        # Inference helpers also prepend their old frozen scripts to sys.path.
        # Load the explicitly hashed diagnostic analyzer, not a shadowing copy
        # bundled with the inherited inference runtime.
        analyzer = load('current_pressure_idle_analyzer', root/'analyze_prefill_cpu_commits.py')
        base.save(root/'gpu-idle-analysis.json',analyzer.analyze(root))
        # Multiowner scheduling already lacks exact replay in the unprofiled
        # control. Preserve this diagnostic's first trajectories without
        # presenting instrumentation as an accuracy/performance qualification.
        if not equivalence['passed'] and plan['workers']==1:
            raise RuntimeError('Profiled first answer or selections differ; evidence preserved')
        base.save(root/'diagnostic-status.json',dict(status='completed',
            output_and_selection_equivalent=equivalence['passed'],performance_eligible=False),replace=True)
    except BaseException as error:
        base.save(root/'diagnostic-status.json',dict(status='failed',error=repr(error)),replace=True)
        raise
    finally:
        environment_module.server_environment=old_env;recall.stop_owned=old_stop


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify','collect','serve'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--port',type=int,default=19480)
    parser.add_argument('--cohort',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,args.source,args.port,cohort=args.cohort)
    elif args.action=='verify':verify(root)
    elif args.action=='serve':serve(root,args.dry_run)
    else:collect(root)
