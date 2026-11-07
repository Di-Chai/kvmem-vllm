"""Collect a bounded native GPU trace, separately from throughput comparisons."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

from audit_layerwise_gilopt import read,save_new,sha,tree
from benchmark_mab_direct_restore import run as run_arm
from vllm_kvmem.profiling import native_profiler_config,summarize_trace

REPO = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').exists())
SOURCE = REPO/'artifacts/mab-efficiency-20261105-r1/prefill-pipeline-qualification/candidate'


def prepare(root):
    if root.exists(): raise ValueError('Do not overwrite a diagnostic')
    assert read(SOURCE/'status.json')['status']=='completed'
    proof=read(SOURCE/'runtime-proof.json');assert tree(SOURCE/'runtime-v1')==proof
    root.mkdir(parents=True);dest=root/'candidate';dest.mkdir()
    shutil.copytree(SOURCE/'runtime-v1',dest/'runtime-v1',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    fixture=read(SOURCE/'fixture.json');assert fixture['cases']==1
    fixture.update(qualification=False,restore_digest_qualification=False)
    profile=read(SOURCE/'profile.json')
    assert profile['kvmem']['prefill_publication_pipeline'] is True
    profile['profiler_config']=native_profiler_config(root/'traces',16)
    save_new(dest/'fixture.json',fixture);save_new(dest/'profile.json',profile)
    save_new(dest/'runtime-proof.json',proof);save_new(dest/'status.json',dict(status='prepared'))
    shutil.copy2(__file__,root/'launch.py')
    files={str(dest/n):sha(dest/n) for n in ['fixture.json','profile.json','runtime-proof.json']}
    files.update({b['request']:b['sha256'] for b in fixture['bindings']})
    for p in [root/'launch.py',SOURCE/'runtime-proof.json',REPO/'scripts/benchmark_mab_direct_restore.py',REPO/'scripts/benchmark_mab_query_handoff.py']:
        files[str(p)]=sha(p)
    save_new(root/'plan.json',dict(baseline=str(SOURCE),files=files,scope=__doc__,performance_eligible=False,
        start_after_prefill_tokens=100000,formal_resume=False,external_grading=False))
    save_new(root/'prepared-proof.json',dict(plan_sha256=sha(root/'plan.json')))


def collect(root):
    if (root/'collector-process.json').exists():raise ValueError('Preserve prior collection')
    save_new(root/'collector-process.json',dict(pid=os.getpid(),at=time.time()))
    from run_recall_comparison import stop_owned,termination_guard
    child=None
    with termination_guard():
        try:
            with (root/'child.log').open('x') as out:
                child=subprocess.Popen([sys.executable,'-B',str(root/'launch.py'),'arm','--root',str(root)],
                    stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            deadline=time.monotonic()+600
            capture=root/'candidate/run/server/capture'
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError('Inference exited before profiling')
                path=capture/'public_work.json'
                try:work=read(path)
                except (FileNotFoundError,json.JSONDecodeError):time.sleep(.25);continue
                if work['completed']['public_prefill_tokens']>=100000:break
                time.sleep(.25)
            else:raise TimeoutError('No eligible history window within ten minutes')
            assert work['completed']['public_non_prefill_tokens']==0
            save_new(root/'profile-start-work.json',work)
            save_new(root/'profile-owner.json',read(root/'candidate/process.json'))
            profile=read(root/'candidate/profile.json')
            url=f"http://127.0.0.1:{profile['port']}/start_profile"
            http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            request=urllib.request.Request(url,data=b'',method='POST')
            with http.open(request,timeout=180) as response:
                assert response.status==200;response.read()
            child.wait()
            if child.returncode:raise RuntimeError('Profiled inference failed; evidence preserved')
            traces=sorted((root/'traces').rglob('*.pt.trace.json.gz'))
            if not traces:traces=sorted((root/'traces').rglob('*.pt.trace.json'))
            assert traces,'Expected finite native worker trace'
            reports=[summarize_trace(p) for p in traces]
            save_new(root/'trace-summary.json',dict(performance_eligible=False,traces=reports))
        finally:
            stop_owned(child)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','collect','arm'])
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root)
    elif args.action=='collect':collect(root)
    else:run_arm(root)
