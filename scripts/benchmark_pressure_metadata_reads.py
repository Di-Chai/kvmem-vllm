"""Freeze reused pressure optimizations, then a finite predeclared cold-input comparison."""
import argparse
import ast
import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
sys.path.insert(0,str(REPO/'scripts'))
from audit_layerwise_gilopt import read,save_new,sha,tree
from benchmark_mab_query_handoff import insert_once

BASE=REPO/'artifacts/layerwise-physical-prefill-20261105-r1/merged-host128-r2/candidate'
OLD=REPO/'artifacts/g4k-pressure-hotpath-20261104-r1/inline-serial/runtime-v1'
PREPARED=REPO/'artifacts/g4k-pressure-implementation-20261104-r1/paired-v4-abba/02-prepared/runtime-v1'
PARTS=('storage.py','memory_storage.py','layerwise_query.py','layerwise_pressure_efficiency.py',
       'pressure_reader.py','prepared_regions.py')
ORDER=('a1','metadata','b1','b2','a2')


def metadata(code):
    """Mirror the workspace slot/order cache without importing newer recall paths."""
    code=insert_once(code,
        '        by_begin = {r.history_begin:r for r in records}\n',
        '''        record_slots = [r.handle.slot for r in records]
        record_positions = {slot:i for i,slot in enumerate(record_slots)}
        if len(record_positions) != len(records):
            raise RuntimeError('Pressure archive contains duplicate physical slots')
        by_begin = {r.history_begin:(r, s) for r, s in zip(records, record_slots)}
''')
    code=insert_once(code,'        mandatory = tuple(r.handle.slot for r in records if r.history_begin//32 in required)\n',
        '        mandatory = tuple(s for r, s in zip(records, record_slots) if r.history_begin//32 in required)\n        mandatory_set = set(mandatory)\n')
    code=insert_once(code,'        if {by_begin[r.history_begin].history_begin//32 for r in records if r.handle.slot in mandatory} != required:\n',
        '        if {r.history_begin//32 for r, s in zip(records, record_slots) if s in mandatory_set} != required:\n')
    code=insert_once(code,'''            candidates = []
            for _, _, begin, n, _ in mapping:
                r = by_begin.get(begin)
                if r is None or r.count != n:
                    raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                candidates.append(r)
''','''            candidates, candidate_slots = [], []
            for _, _, begin, n, _ in mapping:
                entry = by_begin.get(begin)
                if entry is None or entry[0].count != n:
                    raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                candidates.append(entry[0])
                candidate_slots.append(entry[1])
''')
    code=insert_once(code,'            supplied = [(r.handle.slot, scores.get(r.history_begin, 0.)) for r in candidates]\n',
        '            supplied = [(s, scores.get(r.history_begin, 0.)) for r, s in zip(candidates, candidate_slots)]\n')
    code=insert_once(code,'            allowed = {r.handle.slot for r in candidates} | set(mandatory)\n',
        '            allowed = set(candidate_slots) | mandatory_set\n')
    code=insert_once(code,'            if not set(mandatory) <= slots or not slots <= allowed or 32*len(slots) > cfg.select_budget:\n',
        '            if not mandatory_set <= slots or not slots <= allowed or 32*len(slots) > cfg.select_budget:\n')
    code=insert_once(code,'            picked = [r for r in records if r.handle.slot in slots]\n',
        '            picked = [records[record_positions[s]] for s in sorted(slots, key=record_positions.__getitem__)]\n')
    ast.parse(code);return code


def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    proof=read(BASE/'runtime-proof.json');assert tree(BASE/'runtime-v1')==proof
    assert read(BASE/'status.json')['status']=='completed'
    assert not read(BASE/'run/cleanup.json')['gpu_processes']
    patch=[]
    for name in PARTS:
        rel='src/vllm_kvmem/'+name;old=OLD/rel;new=PREPARED/rel
        assert sha(new)==read(PREPARED.parent/'runtime-proof.json')[rel]
        before=old.read_text() if old.exists() else ''
        patch.extend(difflib.unified_diff(before.splitlines(True),new.read_text().splitlines(True),
            fromfile='a/'+rel if old.exists() else '/dev/null',tofile='b/'+rel,n=1))
    (root/'prepared-reader.patch').write_text(''.join(patch))
    shutil.copy2(__file__,root/'launch.py')
    gpu_test=REPO/'artifacts/g4k-pressure-implementation-20261104-r1/components-prepared-v4/runtime-v1/tests/test_layerwise_paired_reads_gpu.py'
    shutil.copy2(gpu_test,root/'test_prepared_gpu.py')
    common={str(root/p):sha(root/p) for p in ('launch.py','prepared-reader.patch','test_prepared_gpu.py')}
    common[str(REPO/'scripts/benchmark_mab_direct_restore.py')]=sha(REPO/'scripts/benchmark_mab_direct_restore.py')
    profiles=[]
    for arm in ORDER:
        ar=root/arm;c=ar/'candidate';c.mkdir(parents=True);runtime=c/'runtime-v1'
        shutil.copytree(BASE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('fixture.json','profile.json','verification-profile.json'):shutil.copy2(BASE/name,c/name)
        if arm in ('b1','b2'):
            env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(c))
            for operation in (['--check'],[]):
                subprocess.run(['git','apply',*operation,str(root/'prepared-reader.patch')],cwd=runtime,env=env,check=True)
        if arm in ('metadata','b1','b2'):
            p=runtime/'src/vllm_kvmem/layerwise_pressure.py';p.write_text(metadata(p.read_text()))
        # Identical low-volume stage marker in every arm repairs queue timing.
        p=runtime/'src/vllm_kvmem/layerwise_query.py'
        p.write_text(insert_once(p.read_text(),'        def assemble_with_probe(self, job, queries, ready):\n',
            "        def assemble_with_probe(self, job, queries, ready):\n            self.emit(job, 'assembly_worker_started')\n"))
        current=tree(runtime);save_new(c/'runtime-proof.json',current)
        save_new(c/'status.json',dict(status='prepared'))
        changed=sorted(k for k in current if proof.get(k)!=current[k])
        files=dict(common);files.update({str(c/n):sha(c/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')})
        files.update({b['request']:b['sha256'] for b in read(c/'fixture.json')['bindings']})
        save_new(ar/'plan.json',dict(baseline=str(BASE),files=files,changed=changed,
            formal_resume=False,external_grading=False,
            scope='Unchanged full cold MAB input and 8K physical grouping. Shared stage marker; metadata cache and prepared immutable reads only. No numerical kernel/ranking change.'))
        save_new(ar/'prepared-proof.json',dict(plan_sha256=sha(ar/'plan.json')))
        save_new(ar/'status.json',dict(status='prepared'))
        profiles.append(read(c/'profile.json'))
    assert all(p==profiles[0] for p in profiles)
    assert tree(root/'b1/candidate/runtime-v1')==tree(root/'b2/candidate/runtime-v1')
    assert tree(root/'a1/candidate/runtime-v1')==tree(root/'a2/candidate/runtime-v1')
    save_new(root/'plan.json',dict(order=ORDER,files=common,baseline=str(BASE),
        primary_comparison='pooled a1+a2 versus b1+b2; metadata is a single-stage diagnostic',
        automatic_retry=False,formal_resume=False,external_grading=False,profiler=False,
        controls='NVFP4 Active36864 Gen16384 block32 sink41 recent0, logical1024 physical8192, native batch16424/maxseq8; all input/sampling/output unchanged'))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))
    save_new(root/'status.json',dict(status='prepared'))


def verify(root):
    assert sha(root/'plan.json')==read(root/'prepared-proof.json')['plan_sha256']
    plan=read(root/'plan.json')
    for n,digest in plan['files'].items():assert sha(Path(n))==digest,n
    return plan


def run(root):
    plan=verify(root)
    gate=read(root/'gpu-byte-audit.json');assert gate['passed'] and gate['cases']==6
    assert gate['runtime_proof_sha256']==sha(root/'b1/candidate/runtime-proof.json')
    try:
        for arm in plan['order']:
            (root/'status.json').write_text(__import__('json').dumps(dict(status='running',arm=arm)))
            subprocess.run([sys.executable,'-B',str(root/'launch.py'),'run-arm','--root',str(root/arm)],check=True)
            assert read(root/arm/'equivalence.json')['passed'],'First output/selection mismatch; preserve and stop'
        (root/'status.json').write_text('{"status":"completed"}\n')
    except BaseException as error:
        (root/'status.json').write_text(__import__('json').dumps(dict(status='failed',arm=arm,error=repr(error))))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run','run-arm'));p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='run':run(root)
    else:
        from benchmark_mab_direct_restore import run as run_arm
        run_arm(root)
