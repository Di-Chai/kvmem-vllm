"""Finite complete-input comparison of existing SIMD finalization on the exposed 14K publication path."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(REPO/'scripts'))
from audit_layerwise_gilopt import read,save_new,sha,tree
from benchmark_mab_query_handoff import insert_once
from benchmark_prefill_batch_overlap import semantic_audit

BASE=REPO/'artifacts/prefill-batch-overlap-20261105-r4/d1/candidate'
SOURCE=REPO/'artifacts/fused-record-mean-20261105-r1/a1/candidate'
ORDER=('a1','b1','b2','a2')


def prepare(root):
    assert not (root/'plan.json').exists()
    root.mkdir(parents=True,exist_ok=True)
    tests=list(ET.parse(root/'cpu-tests.xml').getroot().iter('testcase'))
    skipped=[c for c in tests if c.find('skipped') is not None]
    assert all(c.get('classname')=='tests.test_compact_cpu_indices' and
               c.get('name').startswith('test_cuda_') for c in skipped)
    assert len(tests)-len(skipped)>=50 and all(not any(c.find(x) is not None for x in ('failure','error')) for c in tests)
    assert tree(SOURCE/'runtime-v1')==read(SOURCE/'runtime-proof.json')
    assert tree(BASE/'runtime-v1')==read(BASE/'runtime-proof.json')
    shutil.copy2(SOURCE.parents[1]/'incremental.patch',root/'incremental.patch')
    shutil.copy2(__file__,root/'launch.py')
    (root/'tests').mkdir()
    for name in ('test_simd_publication_gpu.py',):
        shutil.copy2(REPO/'tests'/name,root/'tests'/name)
    common={str(p):sha(p) for p in [root/'launch.py',root/'incremental.patch',root/'cpu-tests.xml',
        *sorted((root/'tests').glob('*.py')),REPO/'scripts/benchmark_mab_direct_restore.py',
        REPO/'scripts/benchmark_prefill_batch_overlap.py',REPO/'scripts/audit_layerwise_gilopt.py']}
    first=None
    for arm in ORDER:
        ar=root/arm;c=ar/'candidate';c.mkdir(parents=True)
        runtime=c/'runtime-v1'
        shutil.copytree(SOURCE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc','build'))
        for name in ('fixture.json',):shutil.copy2(BASE/name,c/name)
        profile=read(BASE/'profile.json');profile['kvmem']['cpu_reuse_record_views']=False
        profile['kvmem']['cpu_fuse_record_mean']=False
        profile['kvmem']['cpu_mean_index_simd']=arm.startswith('b')
        save_new(c/'profile.json',profile)
        save_new(c/'verification-profile.json',dict(profile,verify_session_restore=True,verify_session_restore_digest=True))
        current=tree(runtime)
        if first is None:first=current
        assert first==current
        save_new(c/'runtime-proof.json',current);save_new(c/'status.json',dict(status='prepared'))
        files=dict(common)
        files.update({str(c/n):sha(c/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')})
        files.update({b['request']:b['sha256'] for b in read(c/'fixture.json')['bindings']})
        save_new(ar/'plan.json',dict(baseline=str(BASE),files=files,formal_resume=False,external_grading=False))
        save_new(ar/'prepared-proof.json',dict(plan_sha256=sha(ar/'plan.json')))
        save_new(ar/'status.json',dict(status='prepared'))
    save_new(root/'formal-preservation.json',read(REPO/'artifacts/prefill-batch-overlap-20261105-r4/formal-preservation.json'))
    save_new(root/'plan.json',dict(order=ORDER,baseline=str(BASE),files=common,
        runtime_proof_sha256=sha(root/'b1/candidate/runtime-proof.json'),cpu_tests=len(tests)-len(skipped),
        gpu_only_cases_excluded_from_cpu=[c.get('name') for c in skipped],
        scope='Same complete cold MAB input, single live owner, 14K physical/1K logical. Active36864 Gen16384 block32 sink41 recent0; all sampling/native batch controls unchanged. Only cpu_mean_index_simd differs. No profiler or external grading.'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def run(root):
    assert sha(root/'plan.json')==read(root/'prepared-proof.json')['plan_sha256']
    plan=read(root/'plan.json')
    for name,digest in plan['files'].items():assert sha(Path(name))==digest
    gate=read(root/'gpu-gate.json')
    assert gate['passed'] and gate['cases']==2 and gate['runtime_proof_sha256']==plan['runtime_proof_sha256']
    try:
        for arm in plan['order']:
            (root/'status.json').write_text(json.dumps(dict(status='running',arm=arm)))
            subprocess.run([sys.executable,'-B',str(root/'launch.py'),'run-arm','--root',str(root/arm)],check=True)
            ar=root/arm;c=ar/'candidate'
            assert read(ar/'equivalence.json')['passed'],'First mismatch: retain evidence and stop'
            audit=semantic_audit(BASE,c)
            progress=read(c/'run/server/capture/progress.json')
            enabled=arm.startswith('b');assert progress['cpu_mean_index_simd']==enabled
            counts=progress['cpu_mean_index_stats']
            assert counts['completed_records']>10000
            if enabled:assert counts['vectorized_records']==counts['completed_records'] and counts['fallback_records']==0
            else:assert counts['vectorized_records']==counts['fallback_records']==0
            assert progress['cpu_fused_mean_stats']['records']==0
            audit['simd_mean']=counts
            save_new(ar/'semantic-audit.json',audit)
        for p,digest in read(root/'formal-preservation.json').items():assert sha(Path(p))==digest
        (root/'status.json').write_text(json.dumps(dict(status='completed')))
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',arm=arm,error=repr(error))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','run-arm'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='run':run(root)
    else:
        from benchmark_mab_direct_restore import run as run_arm
        run_arm(root)
