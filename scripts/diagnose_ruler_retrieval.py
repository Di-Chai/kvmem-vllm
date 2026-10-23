"""Frozen, serial 256K retrieval diagnostics with a complete-prefix control."""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import numpy as np

import benchmark_block_graph as lifecycle
from benchmark_ruler import save, sha
import qualify_graph_niah as client


CASES = ['niah_multikey_2-'+s for s in
         ('0008','0009','0000','0013','0003','0010','0011','0012','0015','0018','0019','0005')]


def prepare(root, reference, repo):
    if root.exists():
        raise ValueError('Use a new diagnostic root')
    fixture=json.loads((reference/'fixture.json').read_text())
    byid={r['id']:r for r in fixture['rows']}
    rows=[byid[c] for c in CASES]
    for r in rows:
        p=Path(fixture['input_directory'])/(r['id']+'.i32')
        if sha(p)!=r['served_sha256']:
            raise ValueError('Original input changed')
    root.mkdir(parents=True)
    runtime=root/'runtime'
    for name in ['src/vllm_kvmem','scripts','csrc','third_party/qw3']:
        shutil.copytree(repo/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['pyproject.toml','setup.py']:
        shutil.copy2(repo/name,runtime/name)
    proof={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}
    save(root/'runtime-proof.json',dict(files=proof,head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
    save(root/'fixture.json',dict(source=str(reference),source_fixture_sha256=sha(reference/'fixture.json'),
        input_directory=fixture['input_directory'],sampling=fixture['sampling'],rows=rows,
        purpose='Diagnostic only; original inputs and seeds. Original output sampling except max_tokens=1, since only pre-generation Q/K are observed. No answers are passed to observer.',
        differences=['One serial request, max_num_seqs=1, original session checkpoint capacity, no physical coalescing.',
                     'Explicit expensive diagnostics; no performance estimate.',
                     'Full-prefix control uses K=245760/G=16384: its 262144-token active capacity fits every input before the final Query picker. Zero prefill-pressure events are required.',
                     'Both requests generate one token after diagnostic capture; inference accuracy is not scored. G stays 16384.',
                     'Both observers evaluate the same 36864-token picker; full-prefix inference itself keeps every prefix token.']))
    original=json.loads((reference/'profile.json').read_text())
    for arm,budget in [('compressed',36864),('full',245760)]:
        directory=root/arm;directory.mkdir()
        profile=deepcopy(original)
        profile.update(port=18862,max_num_seqs=1)
        profile['kvmem'].update(select_budget=budget,
            guided_query_tokens=0,graph_alpha=0.,graph_trace=False,prefill_physical_coalescing=False)
        if arm=='full':profile['kvmem']['repository_blocks']=32768
        from vllm_kvmem.serve import build_launch
        build_launch(profile,repo,directory/'preflight-server')
        save(directory/'profile.json',profile)
    save(root/'status.json',dict(status='prepared',cases=CASES,at=time.time()))


def run(root):
    runtime=root/'runtime'
    proof=json.loads((root/'runtime-proof.json').read_text())
    if any(sha(runtime/p)!=v for p,v in proof['files'].items()):
        raise ValueError('Frozen diagnostic source changed')
    fixture=json.loads((root/'fixture.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
               MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ['/usr/local/cuda-13','/usr/local/cuda']:
        if (Path(toolkit)/'bin/nvcc').exists():
            env.update(CUDA_HOME=toolkit,PATH=toolkit+'/bin'+os.pathsep+env.get('PATH',''));break
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock, lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if lifecycle.gpu_processes():
            raise RuntimeError('Another GPU owner is active')
        for arm in ['compressed','full']:
            base=root/arm
            profile=json.loads((base/'profile.json').read_text())
            capture=base/'server/capture'
            server=None
            try:
                command=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(base/'profile.json'),
                         '--run-dir',str(base/'server'),'--retrieval-diagnostics']
                with (base/'server.log').open('x') as log:
                    server=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(base/'server-process.json',dict(pid=server.pid,command=command))
                save(root/'status.json',dict(status='loading',arm=arm,pid=server.pid,at=time.time()))
                deadline=time.monotonic()+1200
                opener=client.urllib.request.build_opener(client.urllib.request.ProxyHandler({}))
                while time.monotonic()<deadline:
                    if server.poll() is not None:raise RuntimeError('Diagnostic model exited')
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/health",timeout=1) as r:
                            if r.status==200:break
                    except OSError:pass
                    time.sleep(1)
                else:raise TimeoutError('Diagnostic server startup timeout')
                results=[]
                for row in fixture['rows']:
                    directory=base/'cases'/row['id'];directory.mkdir(parents=True,exist_ok=False)
                    save(root/'status.json',dict(status='testing',arm=arm,case=row['id'],completed=len(results),expected=len(CASES),at=time.time()))
                    p=Path(fixture['input_directory'])/(row['id']+'.i32')
                    if sha(p)!=row['served_sha256']:raise ValueError('Frozen input changed')
                    ids=np.fromfile(p,dtype='<i4').tolist()
                    session='retrieval-diagnostic-'+uuid.uuid4().hex
                    sampling=dict(fixture['sampling'],seed=row['seed'],max_tokens=1)
                    save(directory/'request.json',dict(input_sha256=row['served_sha256'],query_span=row['served_query'],sampling=sampling,session=session))
                    response=client.stream(profile,session,ids,row['served_query'],sampling,3600)
                    save(directory/'response.json',response)
                    terminal=client.wait_retired(capture,response['response_id'])
                    released=client.wait_owner_released(capture,session,response,terminal)
                    matches=[p for p in (capture/'retrieval_diagnostics').iterdir() if p.name.startswith(response['response_id']+'-')]
                    if len(matches)!=1 or not (matches[0]/'archive.json').exists():
                        raise ValueError('Missing complete retrieval diagnostic')
                    (directory/'diagnostic').symlink_to(matches[0],target_is_directory=True)
                    result=dict(id=row['id'],response_id=response['response_id'],diagnostic=str(matches[0]),
                                seconds=response['seconds'],terminal=terminal,owner_release=released)
                    save(directory/'result.json',result);results.append(result)
                    save(base/'progress.json',dict(completed=len(results),expected=len(CASES),results=results,at=time.time()))
                    print(json.dumps(dict(arm=arm,completed=len(results),case=row['id'],seconds=response['seconds'])),flush=True)
                    if server.poll() is not None:raise RuntimeError('Diagnostic model exited during a case')
                save(base/'status.json',dict(status='completed',cases=len(results),at=time.time()))
            except BaseException as error:
                save(root/'status.json',dict(status='failed',arm=arm,error=repr(error),at=time.time()))
                raise
            finally:
                remaining=lifecycle.stop_owned_groups((server,))
                save(base/'cleanup.json',dict(remaining_owned_groups=remaining,gpu_processes=sorted(lifecycle.gpu_processes())))
        save(root/'status.json',dict(status='captured',cases=len(CASES),arms=2,at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.root.resolve(),a.reference.resolve(),Path(__file__).resolve().parents[1])
    else:run(a.root.resolve())
