"""Finite MAB qualification and continuation with first-answer preservation.

Keeps the previous 1,116-case protocol. Applies only the measured P4 execution
patch and the foreground restore-arena repair, never whole workspace modules.
This is not the older performance gate: its distinct evidence and limitations
are recorded explicitly, including intrinsic eight-owner non-reproducibility.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
PREVIOUS=REPO/'artifacts/layerwise-selected-memory-suite-20261104-r3/memoryagentbench'
QUALIFIED=PREVIOUS.parent/'memoryagentbench-qualification'
REPAIRED=REPO/'artifacts/layerwise-memory-restore-arena-20261104-r1/qualification'
P4=REPO/'artifacts/g4k-pressure-implementation-20261104-r1/post-forward-reap-serial'
BASE=REPO/'artifacts/g4k-gilopt-prefill-diagnostic-20261104-r1/cohort-1'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def tree(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
def save_new(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def verify_files(root,files):
    for n,h in files.items():
        if sha(root/n)!=h:raise ValueError('Frozen source changed: '+str(root/n))

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(root, resident_values_root=None):
    # An empty preparation directory can contain the independently checked patch.
    root.mkdir(parents=True,exist_ok=True)
    if (root/'plan.json').exists():raise ValueError('Never replace a frozen plan')
    for source in (PREVIOUS,QUALIFIED,REPAIRED,P4):
        assert tree(source/'runtime-v1')==read(source/'runtime-proof.json')
    assert read(PREVIOUS/'status.json')['status']=='failed'
    assert not read(PREVIOUS/'run/cleanup.json')['gpu_processes']
    assert read(P4/'serial-equivalence.json')['passed']
    patch=[]
    bundle=read(P4.parent/'post-forward-reap-candidate.json')
    for name in bundle['expected']:
        before=(BASE/'runtime-v1'/name).read_text()
        after=(P4/'runtime-v1'/name).read_text()
        patch.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
            fromfile='a/'+name,tofile='b/'+name,n=1))
    name='src/vllm_kvmem/layerwise_checkpoints.py'
    before=(PREVIOUS/'runtime-v1'/name).read_text();after=(REPAIRED/'runtime-v1'/name).read_text()
    patch.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
        fromfile='a/'+name,tofile='b/'+name,n=1))
    sources=dict(p4=str(P4),repair=str(REPAIRED),
        changes=[*bundle['expected'],name],eight_owner_exact_equivalence=False,
        user_request='Optimize fixed-budget assembly/prefill, then run original MAB protocol to measure layerwise recall pass rate.')
    extra_files={}
    if resident_values_root is not None:
        qualification=resident_values_root/'serial/equivalence.json'
        performance=resident_values_root/'abba/final-audit/summary.json'
        assert read(qualification)['passed'] and read(qualification)['cases']==4
        perf=read(performance)
        assert perf['observed_change']>0 and perf['candidate_runs_faster_than_both_baselines']
        assert tree(resident_values_root/'serial/runtime-v1') == read(resident_values_root/'serial/runtime-proof.json')
        sources.update(resident_values=str(resident_values_root),
                       resident_performance_change=perf['observed_change'])
        shutil.copy2(REPO/'scripts/freeze_pressure_resident_values.py',root/'freeze-resident-values.py')
        implementation=resident_values_root/'serial/runtime-v1'
        if 'self.reuse_layer_views = True' in (implementation/'src/vllm_kvmem/layerwise_pressure.py').read_text():
            shutil.copy2(REPO/'scripts/freeze_pressure_layer_views.py',root/'freeze-layer-views.py')
            extra_files[str(root/'freeze-layer-views.py')]=sha(root/'freeze-layer-views.py')
            sources['reuse_layer_views']=True
        for path in (qualification,performance,resident_values_root/'candidate.json',
                     resident_values_root/'serial/runtime-proof.json',root/'freeze-resident-values.py'):
            extra_files[str(path)]=sha(path)
    (root/'execution.patch').write_text(''.join(patch))
    shutil.copy2(__file__,root/'launch.py')
    runtime=root/'runtime-v1'
    shutil.copytree(PREVIOUS/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env['GIT_CEILING_DIRECTORIES']=str(runtime.parent)
    for check in (True,False):
        subprocess.run(['git','apply',*(['--check'] if check else []),str(root/'execution.patch')],
            cwd=runtime,env=env,check=True,capture_output=True)
    old=read(PREVIOUS/'runtime-proof.json');new=tree(runtime)
    assert {n for n in new if new[n]!=old.get(n)}==set(bundle['expected'])|{name}
    if resident_values_root is not None:
        from freeze_pressure_resident_values import resident_sources
        implementation=resident_values_root/'serial/runtime-v1'
        if sources.get('reuse_layer_views'):
            from freeze_pressure_layer_views import view_sources
            for name,after in view_sources(runtime,implementation).items():
                (runtime/name).write_text(after)
                sources['changes'].append(name)
        # Copy only qualified function deltas, never the later mutable working
        # tree's candidate, and retain MAB's own position/rolling policy.
        for name,after in resident_sources(runtime, implementation).items():
            target=runtime/name
            target.write_text(after)
            sources['changes'].append(name)
        sources['changes']=sorted(set(sources['changes']))
        patch=[]
        for name in sources['changes']:
            source=PREVIOUS/'runtime-v1'/name
            before=source.read_text() if source.exists() else ''
            after=(runtime/name).read_text()
            patch.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
                fromfile='a/'+name if before else '/dev/null',tofile='b/'+name,n=1))
        (root/'execution.patch').write_text(''.join(patch))
    save_new(root/'execution-source.json',sources)
    shutil.copy2(REPO/'scripts/audit_layerwise_gilopt.py',runtime/'scripts/audit_layerwise_gilopt.py')
    new=tree(runtime)
    save_new(root/'runtime-proof.json',new)
    profile=read(PREVIOUS/'profile.json');kv=profile['kvmem']
    assert [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')]==[36864,16384,32,41,0]
    assert kv['session_freeze_prompt'] and not kv['async_restore']
    files={str(root/n):sha(root/n) for n in ('execution.patch','execution-source.json','runtime-proof.json','launch.py')}
    files.update(extra_files)
    for source in (PREVIOUS,QUALIFIED,REPAIRED,P4):
        for name in ('fixture.json','profile.json','runtime-proof.json'):
            files[str(source/name)]=sha(source/name)
    for path in (P4/'serial-equivalence.json',REPO/'artifacts/g4k-prefill-phase-abba-20261104-r1/comparison.json'):
        files[str(path)]=sha(path)
    save_new(root/'plan.json',dict(files=files,previous=str(PREVIOUS),qualification_reference=str(QUALIFIED),
        qualified_ids=read(QUALIFIED/'fixture.json')['case_ids'],workers=8,cases=1116,
        budgets=[36864,16384],external_grading=False,automatic_retry=False,scheduled_monitor=False,
        efficiency_scope='P4 phased eight-owner historical prefill +14.73%; serial full-input exact. Not continuous-serving or eight-owner exact-output acceptance.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify_root(root):
    assert sha(root/'plan.json')==read(root/'prepared-proof.json')['plan_sha256']
    plan=read(root/'plan.json');verify_files(Path('/'),plan['files'])
    assert tree(root/'runtime-v1')==read(root/'runtime-proof.json')
    return plan


def prepare_arm(root,kind):
    plan=verify_root(root);arm=root/kind
    arm.mkdir(exist_ok=False)
    shutil.copytree(root/'runtime-v1',arm/'runtime-v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    assert tree(arm/'runtime-v1')==read(root/'runtime-proof.json')
    source=QUALIFIED if kind=='qualification' else PREVIOUS
    for name in ('fixture.json','profile.json'):
        shutil.copy2(source/name,arm/name)
    fixture=read(arm/'fixture.json')
    if kind=='qualification':
        shutil.copy2(QUALIFIED/'verification-profile.json',arm/'verification-profile.json')
    else:
        qualification=root/'qualification'
        assert read(qualification/'equivalence.json')['passed']
        assert read(qualification/'execution-audit.json')['status']=='passed'
        assert read(qualification/'status.json')['status']=='completed'
        assert read(qualification/'profile.json')==read(arm/'profile.json')
        from recover_hybrid_mab import inherit_cases
        proof=inherit_cases(PREVIOUS,arm/'recovered-answers',fixture)
        assert len(proof['recovered_ids'])==5 and proof['regenerated_answers']==0
        save_new(arm/'recovery-proof.json',proof)
    files={str(arm/n):sha(arm/n) for n in ('fixture.json','profile.json')}
    for name in ('verification-profile.json','recovery-proof.json'):
        if (arm/name).exists():files[str(arm/name)]=sha(arm/name)
    if kind=='formal':
        for name in ('equivalence.json','execution-audit.json','status.json'):
            files[str(root/'qualification'/name)]=sha(root/'qualification'/name)
    for binding in fixture['bindings']:files[binding['request']]=binding['sha256']
    save_new(arm/'runtime-proof.json',read(root/'runtime-proof.json'))
    files[str(arm/'runtime-proof.json')]=sha(arm/'runtime-proof.json')
    save_new(arm/'plan.json',dict(queue=str(root),kind=kind,files=files))
    save_new(arm/'prepared-proof.json',dict(plan_sha256=sha(arm/'plan.json')))
    save_new(arm/'status.json',dict(status='prepared',expected=fixture['cases']))


def run_arm(arm):
    plan=read(arm/'plan.json');root=Path(plan['queue']);verify_root(root)
    assert sha(arm/'plan.json')==read(arm/'prepared-proof.json')['plan_sha256']
    sys.path[:0]=[str(arm/'runtime-v1/src'),str(arm/'runtime-v1/scripts')]
    spec=importlib.util.spec_from_file_location('frozen_mab',arm/'runtime-v1/scripts/run_layerwise_memory.py')
    memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    def strict(path):
        assert path==arm
        verify_root(root);verify_files(Path('/'),plan['files'])
        assert tree(arm/'runtime-v1')==read(root/'runtime-proof.json')
        if (arm/'recovery-proof.json').exists():
            proof=read(arm/'recovery-proof.json')
            verify_files(PREVIOUS,proof['source_files']);verify_files(arm/'recovered-answers',proof['recovered_files'])
        return read(arm/'fixture.json'),read(arm/'profile.json')
    memory.verify=strict
    # The original MAB driver freezes ThreadPoolExecutor(max_workers=8) inside
    # infer(); its older run(root) API has no configurable workers parameter.
    assert read(arm/'profile.json')['max_num_seqs']==8
    memory.run(arm)


def run(root):
    verify_root(root)
    sys.path[:0]=[str(root/'runtime-v1/scripts')]
    from audit_layerwise_gilopt import compare
    if (root/'process.json').exists():raise ValueError('Preserve first attempt')
    from run_recall_comparison import termination_guard,stop_owned,gpu_lock
    import benchmark_block_graph as lifecycle
    pointer=REPO/'artifacts/value-recall-active.json'
    with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        assert not lifecycle.gpu_processes()
        previous=read(pointer)
        save_new(root/'previous-pointer.json',previous)
        write(pointer,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
            process=str(root/'process.json'),at=time.time(),experiment_type='mab_optimized_continuation'))
    save_new(root/'process.json',dict(pid=os.getpid(),at=time.time()))
    with termination_guard():
        try:
            for kind in ('qualification','formal'):
                assert not lifecycle.gpu_processes()
                assert read(pointer)['queue_root']==str(root)
                prepare_arm(root,kind);arm=root/kind
                write(root/'status.json',dict(status='running',stage=kind,at=time.time()))
                command=[sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(arm)]
                env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
                env.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONDONTWRITEBYTECODE='1')
                with (root/(kind+'.log')).open('x') as log:
                    child=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(root/'child.json',dict(pid=child.pid,command=command,at=time.time()))
                try:code=child.wait()
                except BaseException:stop_owned(child);raise
                if code:raise RuntimeError('Preserved technical failure: '+kind)
                assert not lifecycle.gpu_processes()
                if kind=='qualification':
                    report=compare(QUALIFIED,arm)
                    save_new(arm/'equivalence.json',report)
                    assert report['cases']==2 and report['passed'], 'Preserved qualification divergence'
                    warm=[read(p) for p in (arm/'run/answers/cases').glob('*/result.json') if read(p)['prefix_reused']]
                    assert len(warm)==1 and warm[0]['prefix_restore_digest']
            write(root/'status.json',dict(status='inference_completed',completed=1116,at=time.time()))
        except BaseException as error:
            write(root/'status.json',dict(status='failed',error=repr(error),at=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','arm'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--resident-values-root',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root.resolve(),a.resident_values_root.resolve() if a.resident_values_root else None)
    else:{'run':run,'arm':run_arm}[a.action](a.root.resolve())
