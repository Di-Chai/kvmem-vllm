"""Port only this goal's host-side changes onto the qualified runtime."""
import argparse
import difflib
import os
from pathlib import Path
import shutil
import subprocess

from audit_layerwise_gilopt import read,save_new,sha,tree

REPO=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
BASE=REPO/'artifacts/prefill-batch-overlap-20261105-r4/c1/candidate'
BEFORE=REPO/'artifacts/multiowner-6000-goal-20261105-r1'


def freeze(root):
    assert not root.exists()
    assert tree(BASE/'runtime-v1')==read(BASE/'runtime-proof.json')
    root.mkdir(parents=True)
    patch=[]
    for name,h in read(BEFORE/'source-before.json').items():
        source=BEFORE/'source-before'/name;assert sha(source)==h
        patch.extend(difflib.unified_diff(source.read_text().splitlines(True),
            (REPO/name).read_text().splitlines(True),fromfile='a/'+name,tofile='b/'+name,n=3))
    (root/'incremental.patch').write_text(''.join(patch))
    runtime=root/'runtime-v1'
    shutil.copytree(BASE/'runtime-v1',runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(root))
    for check in (['--check'],[]):
        subprocess.run(['git','apply',*check,str(root/'incremental.patch')],cwd=runtime,env=env,check=True)
    shutil.copy2(REPO/'src/vllm_kvmem/multiowner_prefill.py',runtime/'src/vllm_kvmem/multiowner_prefill.py')
    save_new(root/'runtime-proof.json',tree(runtime))
    save_new(root/'source.json',dict(baseline=str(BASE),source_before=str(BEFORE),
        patch_sha256=sha(root/'incremental.patch'),default_enabled=False,status='awaiting_qualification'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    freeze(p.parse_args().root.resolve())
