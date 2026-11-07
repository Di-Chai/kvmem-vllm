"""Finite full-input diagnostic of physical merging with original logical commits."""
import argparse
import difflib
import json
import os
from pathlib import Path
import shutil
import subprocess

from audit_layerwise_gilopt import read, save_new, sha, tree
from benchmark_mab_direct_restore import run as run_arm

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())


def prepare(root, baseline):
    assert not (root/'prepared-proof.json').exists()
    assert read(baseline/'status.json')['status']=='completed'
    proof=read(baseline/'runtime-proof.json')
    assert tree(baseline/'runtime-v1')==proof
    before=read(root/'source-before.json')
    patch=[]
    for name, digest in before.items():
        old=root/'source-before'/name
        assert sha(old)==digest
        patch.extend(difflib.unified_diff(old.read_text().splitlines(True),
            (REPO/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name))
    patch_path=root/'incremental.patch'
    with patch_path.open('x') as out:out.write(''.join(patch))
    fixture=read(baseline/'fixture.json')
    assert fixture['cases']==1
    profile=read(baseline/'profile.json')
    assert profile['prefill_chunk_tokens']==1024
    assert profile['kvmem']['prefill_publication_pipeline']
    assert [profile['kvmem'][k] for k in ('select_budget','gen_budget','block_tokens','sink_blocks','recent_blocks')]==[36864,16384,32,41,0]
    # Two 16-layer physical score packets exceed the historical 64MiB host
    # download ceiling. Match this scratch limit in both arms; KV is unchanged.
    profile['kvmem']['action_download_max_bytes']=max(
        profile['kvmem'].get('action_download_max_bytes',64<<20),128<<20)
    frozen=None
    for arm in ('control','merged'):
        arm_root=root/arm;candidate=arm_root/'candidate';candidate.mkdir(parents=True)
        runtime=candidate/'runtime-v1'
        shutil.copytree(baseline/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(candidate))
        subprocess.run(['git','apply','--check',str(patch_path)],cwd=runtime,env=env,check=True)
        subprocess.run(['git','apply',str(patch_path)],cwd=runtime,env=env,check=True)
        current=tree(runtime)
        if frozen is None:frozen=current
        assert current==frozen
        cfg=json.loads(json.dumps(profile))
        cfg['kvmem'].update(prefill_native_plan_arm=True,
            prefill_physical_logical_tokens=1024,prefill_physical_max_chunks=8,
            prefill_physical_coalescing=arm=='merged')
        save_new(candidate/'fixture.json',fixture)
        save_new(candidate/'profile.json',cfg)
        save_new(candidate/'verification-profile.json',dict(cfg,verify_session_restore=True,verify_session_restore_digest=True))
        save_new(candidate/'runtime-proof.json',current)
        save_new(candidate/'status.json',dict(status='prepared'))
        reference=baseline if arm=='control' else root/'control/candidate'
        files={str(candidate/n):sha(candidate/n) for n in ('fixture.json','profile.json','verification-profile.json','runtime-proof.json')}
        files.update({b['request']:b['sha256'] for b in fixture['bindings']})
        files[str(patch_path)]=sha(patch_path)
        save_new(arm_root/'plan.json',dict(baseline=str(reference),files=files,
            formal_resume=False,external_grading=False,
            numeric_differences_authorized=True,
            matched_cpu_download_ceiling_bytes=cfg['kvmem']['action_download_max_bytes'],
            scope='Same logical 1K transactions, per-layer cumulative scores and original full cold MAB input. Only physical grouping enabled in merged arm. All sampling, output, Active/Gen and native batch ceilings unchanged.'))
        save_new(arm_root/'prepared-proof.json',dict(plan_sha256=sha(arm_root/'plan.json')))
        save_new(arm_root/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py')
    save_new(root/'prepared-proof.json',dict(launcher=sha(root/'launch.py'),
        driver=sha(REPO/'scripts/benchmark_mab_direct_restore.py'),
        changed_files=sorted(k for k in frozen if frozen[k]!=proof.get(k)),
        baseline=str(baseline),runtime_equal=True))
    save_new(root/'status.json',dict(status='prepared'))


def run(root):
    proof=read(root/'prepared-proof.json')
    assert sha(root/'launch.py')==proof['launcher']
    assert sha(REPO/'scripts/benchmark_mab_direct_restore.py')==proof['driver']
    try:
        for arm in ('control','merged'):
            (root/'status.json').write_text(json.dumps(dict(status='running',arm=arm)))
            run_arm(root/arm)
        (root/'status.json').write_text(json.dumps(dict(status='completed')))
    except BaseException as error:
        (root/'status.json').write_text(json.dumps(dict(status='failed',arm=arm,error=repr(error))))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.root.resolve(),args.baseline.resolve())
    else:run(args.root.resolve())
