"""Qualify pressure-only execution changes, then run the original MAB cohort.

Preserves old first answers in their original campaign. A new execution version
gets an independent campaign; no old answer is blended into its score.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from audit_layerwise_gilopt import compare, read, save_new, sha, tree

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
PREVIOUS = REPO/'artifacts/mab-pressure-optimized-20261105-r3'
BENCH = REPO/'artifacts/multiowner-pressure-fix-bench-20261105-r2'
EXECUTION_KEYS = {
    'action_download_max_bytes','action_native_max_bytes','action_native_max_rows',
    'cpu_read_cache_bytes','prefill_action_before_mtp','prefill_native_plan_arm',
    'prefill_physical_coalescing','prefill_physical_logical_tokens',
    'prefill_physical_max_chunks','prefill_physical_multi_owner','prefill_publication_pipeline',
    'prefill_physical_fill_rows',
    'action_native_prefill_static_aligned',
}
GATE_SUFFIX = "\n# Measurement only: fixed cohort gate; no Query/decode in measured prefill.\nos.environ['KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS'] = '6'\nfrom .prefill_phase_gate import install as _install_measurement_gate\n_install_measurement_gate()\n"


def prepare(root, bench=BENCH, candidate_arm='combined1', control_source=None):
    assert not root.exists(), 'Do not replace previous evidence'
    profile = read(bench/candidate_arm/'profile.json')
    original = read(PREVIOUS/'formal/profile.json')
    assert {k:v for k,v in profile.items() if k!='kvmem'} == {k:v for k,v in original.items() if k!='kvmem'}
    kv, oldkv = profile['kvmem'], original['kvmem']
    changes = {k:dict(before=oldkv.get(k),after=kv.get(k)) for k in set(kv)|set(oldkv) if kv.get(k)!=oldkv.get(k)}
    assert set(changes) <= EXECUTION_KEYS
    assert [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] == [36864,16384,32,41,0]
    root.mkdir(parents=True)
    files = {}
    for kind in ('qualification_control','qualification_candidate','formal'):
        dest = root/kind; dest.mkdir()
        source = (control_source or bench/'control') if kind=='qualification_control' else bench/candidate_arm
        assert tree(source/'runtime-v1') == read(source/'runtime-proof.json')
        profile = read(source/'profile.json')
        assert {k:v for k,v in profile.items() if k!='kvmem'} == {k:v for k,v in original.items() if k!='kvmem'}
        kv = profile['kvmem']
        assert [kv[k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')] == [36864,16384,32,41,0]
        arm_changes = {k for k in set(kv)|set(oldkv) if kv.get(k)!=oldkv.get(k)}
        assert arm_changes <= EXECUTION_KEYS
        runtime = dest/'runtime-v1'
        shutil.copytree(source/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        path = runtime/'src/vllm_kvmem/layerwise_query.py'
        code = path.read_text(); assert code.endswith(GATE_SUFFIX)
        path.write_text(code[:-len(GATE_SUFFIX)])
        fixture_source = PREVIOUS/('formal' if kind=='formal' else 'qualification')
        fixture = read(fixture_source/'fixture.json')
        fixture['execution_kvmem_overrides'] = {**fixture.get('execution_kvmem_overrides',{}),
                                                **{k:kv[k] for k in arm_changes}}
        fixture['performance_observation'] = True
        fixture['diagnostic_only'] = kind!='formal'
        assert fixture['cases'] == (1116 if kind=='formal' else 2)
        save_new(dest/'fixture.json',fixture)
        save_new(dest/'profile.json',profile)
        if kind!='formal':
            assert fixture['restore_digest_qualification']
            save_new(dest/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
        save_new(dest/'runtime-proof.json',tree(runtime))
        save_new(dest/'status.json',dict(status='prepared'))
        files.update({str(dest/n):sha(dest/n) for n in ('fixture.json','profile.json','runtime-proof.json')})
        if kind!='formal': files[str(dest/'verification-profile.json')] = sha(dest/'verification-profile.json')
        for binding in fixture['bindings']:files[binding['request']] = binding['sha256']
    assert read(root/'formal/runtime-proof.json') == read(root/'qualification_candidate/runtime-proof.json')
    shutil.copy2(__file__,root/'launch.py'); files[str(root/'launch.py')] = sha(root/'launch.py')
    # Retain immutable raw responses/evidence, without importing them as answers
    # from this new shape-dependent numerical execution version.
    prior = PREVIOUS/'formal/run/answers'
    preserved = {str(p):sha(p) for p in prior.rglob('*') if p.is_file()}
    save_new(root/'previous-preserved.json',dict(files=preserved,
        completed=len(list((prior/'cases').glob('*/response.json'))),imported_into_this_score=0))
    files[str(root/'previous-preserved.json')] = sha(root/'previous-preserved.json')
    save_new(root/'plan.json',dict(files=files,bench=str(bench),candidate_arm=candidate_arm,execution_changes=changes,
        qualification_control_source=str(control_source or bench/'control'),
        cases=1116,workers=8,order=['qualification_control','qualification_candidate','formal'],
        original_fixture=str(PREVIOUS/'formal/fixture.json'),
        qualification='Original longest cold P prefix and next question, independent A/B with byte-digest restore verification; pressure, Query and complete output must match.',
        formal_resume=False,external_grading=False,automatic_retry=False,scheduled_monitor=False,
        scope='New independent full MAB execution version. Original requests, sampling, budget and P-prefix protocol retained; pure-prefill measurement gate removed.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json') == read(root/'prepared-proof.json')['plan_sha256']
    plan = read(root/'plan.json')
    for path,digest in plan['files'].items():assert sha(Path(path)) == digest,path
    for kind in plan['order']:
        assert tree(root/kind/'runtime-v1') == read(root/kind/'runtime-proof.json')
    return plan


def arm(root, kind):
    plan=verify(root); assert kind in plan['order']
    dest=root/kind;runtime=dest/'runtime-v1'
    sys.path[:0]=[str(runtime/'src'),str(runtime/'scripts')]
    spec=importlib.util.spec_from_file_location('frozen_memory',runtime/'scripts/run_layerwise_memory.py')
    memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
    def frozen(path):
        assert path==dest;verify(root)
        return read(dest/'fixture.json'),read(dest/'profile.json')
    memory.verify=frozen
    memory.run(dest)


def run(root):
    plan=verify(root)
    bench=Path(plan['bench'])
    assert read(bench/'status.json')['status']=='completed'
    acceptance=read(bench/'efficiency-validation.json')
    assert acceptance['passed'] and acceptance['candidate']==plan['candidate_arm']
    for name,digest in acceptance['files'].items():assert sha(Path(name))==digest,name
    save_new(root/'efficiency-accepted.json',dict(source=str(bench/'efficiency-validation.json'),
                                               sha256=sha(bench/'efficiency-validation.json')))
    save_new(root/'process.json',dict(pid=os.getpid(),at=time.time()))
    pointer=REPO/'artifacts/value-recall-active.json'
    save_new(root/'previous-pointer.json',read(pointer))
    pointer.write_text(json.dumps(dict(queue_root=str(root),root=str(root),
        status=str(root/'status.json'),process=str(root/'process.json'),at=time.time(),
        experiment_type='mab_pressure_execution_qualified'),indent=2)+'\n')
    from run_recall_comparison import termination_guard,stop_owned
    child=None
    with termination_guard():
        try:
            for kind in plan['order']:
                if kind=='formal':
                    comparison=compare(root/'qualification_control',root/'qualification_candidate')
                    save_new(root/'full-input-equivalence.json',comparison)
                    assert comparison['cases']==2 and comparison['passed'], 'Preserve first divergence; no automatic resampling'
                    for qual in ('qualification_control','qualification_candidate'):
                        audit=read(root/qual/'execution-audit.json'); assert audit['status']=='passed'
                        warm=[read(p) for p in (root/qual/'run/answers/cases').glob('*/result.json') if read(p)['prefix_reused']]
                        assert len(warm)==1 and warm[0]['prefix_restore_digest']
                    for path,digest in read(root/'previous-preserved.json')['files'].items():
                        assert sha(Path(path))==digest,path
                (root/'status.json').write_text(json.dumps(dict(status='running',stage=kind,at=time.time())))
                with (root/(kind+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B',str(root/'launch.py'),'arm',
                        '--root',str(root),'--kind',kind],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                (root/'child.json').write_text(json.dumps(dict(pid=child.pid,stage=kind,at=time.time())))
                if child.wait():raise RuntimeError('Preserved technical failure: '+kind)
            (root/'status.json').write_text(json.dumps(dict(status='inference_completed',cases=1116,at=time.time())))
        except BaseException as error:
            (root/'status.json').write_text(json.dumps(dict(status='failed',error=repr(error),at=time.time())))
            raise
        finally:stop_owned(child)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','run','arm'))
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--benchmark-root',type=Path,default=BENCH)
    p.add_argument('--candidate-arm',default='combined1')
    p.add_argument('--control-source',type=Path)
    p.add_argument('--kind',choices=('qualification_control','qualification_candidate','formal'))
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root,a.benchmark_root.resolve(),a.candidate_arm,a.control_source.resolve() if a.control_source else None)
    elif a.action=='arm':arm(root,a.kind)
    else:run(root)
