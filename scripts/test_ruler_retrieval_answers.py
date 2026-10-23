"""Frozen 12-case answer comparison, after the Q/K diagnostic releases the GPU.

This is a failure-cohort experiment, not a complete RULER benchmark. Original
256K input, Query, seeds and full output sampling are retained. The two arms
differ only in the final public Query scorer. No answers reach model selection.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace as NS
import uuid

import numpy as np

import benchmark_block_graph as lifecycle
from benchmark_ruler import save, sha
import qualify_graph_niah as client
import ruler_protocol as protocol


def read(path):return json.loads(path.read_text())


def prepare(root, diagnostic):
    if root.exists():raise ValueError('Use a new answer experiment root')
    fixture=read(diagnostic/'fixture.json')
    original=read(Path(fixture['source'])/'fixture.json')
    root.mkdir(parents=True)
    repo=Path(__file__).resolve().parents[1]
    runtime=root/'runtime'
    for name in ['src/vllm_kvmem','scripts','csrc','third_party/qw3']:
        shutil.copytree(repo/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ['pyproject.toml','setup.py']:shutil.copy2(repo/name,runtime/name)
    proof={str(p.relative_to(runtime)):sha(p) for p in runtime.rglob('*') if p.is_file()}
    save(root/'runtime-proof.json',dict(files=proof))
    rows=fixture['rows']
    for row in rows:
        if sha(Path(fixture['input_directory'])/(row['id']+'.i32'))!=row['served_sha256']:
            raise ValueError('Original fixture changed')
    save(root/'fixture.json',dict(rows=rows,input_directory=fixture['input_directory'],
        sampling=original['sampling'],starts_in_thinking=original['starts_in_thinking'],
        upstream_path=original['upstream_path'],tokenizer_sha256=original['tokenizer_sha256'],
        source=str(fixture['source']),diagnostic_dependency=str(diagnostic),
        purpose='12-case failure/control cohort; complete generation, original 8192-token output limit. Not complete-RULER accuracy.',
        difference='Final public Query: original MeanK vs global content-token attention summed per page. No graph propagation, no neighbor expansion.',
        diagnostic_fixture_sha256=sha(diagnostic/'fixture.json')))
    base=read(diagnostic/'compressed/profile.json')
    for arm in ['mean_k','token_sum']:
        directory=root/arm;directory.mkdir()
        profile=deepcopy(base);profile['port']=18863
        from vllm_kvmem.serve import build_launch
        build_launch(profile,repo,directory/'preflight-server')
        save(directory/'profile.json',profile)
    save(root/'status.json',dict(status='prepared',cases=len(rows),arms=2,at=time.time()))


def events(capture,name,response):
    path=capture/name
    return [row for row in (json.loads(line) for line in path.open())
            if client.q.native_matches(response,row)] if path.exists() else []


def coverage(ranges,span):
    return sum(max(0,min(begin+count,span[1])-max(begin,span[0]))
               for begin,count in ranges)==span[1]-span[0]


def finish_case(root,arm,fixture,row,profile,tokens,session,response,capture):
    from tokenizers import Tokenizer
    terminal=client.wait_retired(capture,response['response_id'])
    if terminal['history_end']!=len(tokens)+len(response['output_ids'])-1 or terminal.get('checkpoint_tokens'):
        raise ValueError('Accepted input/output or independent-owner retirement differs')
    released=client.wait_owner_released(capture,session,response,terminal)
    bounded=events(capture,'bounded_refresh.jsonl',response)
    query=[e for e in bounded if e['selection']=='query']
    pressure=[e for e in bounded if e['selection']=='pressure']
    method='mean_k' if arm=='mean_k' else 'content_token_softmax_page_sum'
    if (len(query)!=1 or query[0]['query_span']!=row['served_query'] or
            query[0].get('query_scoring_method')!=method or not pressure or
            any(e['guided_query_tokens'] for e in bounded)):
        raise ValueError('Public Query or pressure contract differs')
    graph=events(capture,'block_graph.jsonl',response)
    query_graph=[e for e in graph if e['phase']=='query']
    if len(query_graph)!=1 or query_graph[0]['alpha']!=0 or query_graph[0]['seed_source']!=method:
        raise ValueError('Query scoring source or zero propagation differs')
    # Pressure remains cumulative accepted attention in both arms.
    if any(e.get('phase')=='pressure' and e.get('propagation_applied',False) for e in graph):
        raise ValueError('Unexpected pressure propagation')
    action=events(capture,'action_attention.jsonl',response)
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    expected=client.q.expected_output_samples(NS(convert_tokens_to_ids=tokenizer.token_to_id),
        profile,tokens,response['output_ids'])
    if len(action)!=1 or action[0]['samples']!=expected:
        raise ValueError('Private/rejected/thinking samples contaminated attention')
    ranges=query[0]['selected_history_ranges']
    if sum(n for _,n in ranges)>profile['kvmem']['select_budget']:
        raise ValueError('Selected window exceeds fixed budget')
    gold=next(e for e in read(Path('artifacts/ruler-failures-20261022/analysis.json'))['rows']
              if e['id']==row['id'])['references'][0]['spans'][0]
    scorer=protocol.load_scorer(Path(fixture['upstream_path']))
    answer=protocol.final_answer(response['text'],starts_in_thinking=fixture['starts_in_thinking'])
    config=scorer[0][protocol.TASK_FAMILIES[row['task']]]
    cleaned=scorer[1](answer['text'],config)
    score=config['metric_fn']([cleaned],[row['outputs']])
    return dict(id=row['id'],arm=arm,official_case_score=score,answer=answer['text'],
        thinking_complete=answer['thinking_complete'],finish_reason=response['finish_reason'],
        completion_tokens=len(response['output_ids']),seconds=response['seconds'],
        value_covered=coverage(ranges,gold['token_span']),
        record_covered=coverage(ranges,[gold['key_token_span'][0],gold['token_span'][1]]),
        target_value_span=gold['token_span'],pressure_count=len(pressure)), dict(
        bounded=bounded,graph=graph,action=action,terminal=terminal,owner_release=released)


def run(root):
    runtime=root/'runtime';proof=read(root/'runtime-proof.json')
    if any(sha(runtime/p)!=value for p,value in proof['files'].items()):raise ValueError('Frozen source changed')
    fixture=read(root/'fixture.json');dependency=Path(fixture['diagnostic_dependency'])
    save(root/'status.json',dict(status='waiting_for_diagnostic_analysis',at=time.time()))
    deadline=time.monotonic()+7200
    while True:
        status=read(dependency/'analysis-status.json')['status']
        if status=='completed':break
        if status=='failed':raise RuntimeError('Diagnostic dependency failed; no automatic retry')
        if time.monotonic()>deadline:raise TimeoutError('Diagnostic dependency timeout')
        time.sleep(10)
    env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
               MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',VLLM_LOG_STATS_INTERVAL='5')
    for toolkit in ['/usr/local/cuda-13','/usr/local/cuda']:
        if (Path(toolkit)/'bin/nvcc').exists():
            env.update(CUDA_HOME=toolkit,PATH=toolkit+'/bin'+os.pathsep+env.get('PATH',''));break
    results={}
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock,lifecycle.cleanup_on_termination():
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('GPU coordination remains held')
                time.sleep(5)
        if lifecycle.gpu_processes():raise RuntimeError('Foreign GPU owner; do not stop it')
        for arm in ['mean_k','token_sum']:
            base=root/arm;profile=read(base/'profile.json');capture=base/'server/capture';server=None
            observed=[]
            try:
                command=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(base/'profile.json'),
                         '--run-dir',str(base/'server')]
                if arm=='token_sum':command+=['--experimental-query-token-scores','sum']
                with (base/'server.log').open('x') as log:
                    server=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(base/'server-process.json',dict(pid=server.pid,command=command))
                save(root/'status.json',dict(status='loading',arm=arm,pid=server.pid,at=time.time()))
                startup=time.monotonic()+1200
                opener=client.urllib.request.build_opener(client.urllib.request.ProxyHandler({}))
                while True:
                    if server.poll() is not None:raise RuntimeError('Answer model exited')
                    try:
                        with opener.open(f"http://127.0.0.1:{profile['port']}/health",timeout=1) as r:
                            if r.status==200:break
                    except OSError:pass
                    if time.monotonic()>startup:raise TimeoutError('Answer model startup timeout')
                    time.sleep(1)
                for row in fixture['rows']:
                    case=base/'cases'/row['id'];case.mkdir(parents=True,exist_ok=False)
                    save(root/'status.json',dict(status='testing',arm=arm,case=row['id'],
                        completed=len(observed),expected=len(fixture['rows']),at=time.time()))
                    source=Path(fixture['input_directory'])/(row['id']+'.i32')
                    if sha(source)!=row['served_sha256']:raise ValueError('Input changed')
                    tokens=np.fromfile(source,dtype='<i4').tolist()
                    sampling=dict(fixture['sampling'],seed=row['seed'])
                    session='retrieval-answer-'+uuid.uuid4().hex
                    save(case/'request.json',dict(sampling=sampling,session_id=session,
                        input_sha256=row['served_sha256'],query_span=row['served_query']))
                    response=client.stream(profile,session,tokens,row['served_query'],sampling,3600)
                    save(case/'response.json',response)
                    result,evidence=finish_case(root,arm,fixture,row,profile,tokens,session,response,capture)
                    save(case/'evidence.json',evidence);save(case/'result.json',result)
                    save(case/'hashes.json',{name:sha(case/name) for name in ['response.json','result.json','evidence.json','request.json']})
                    observed.append(result)
                    save(base/'progress.json',dict(completed=len(observed),expected=len(fixture['rows']),results=observed,at=time.time()))
                    print(json.dumps(result),flush=True)
                results[arm]=observed
                save(base/'scores.json',dict(cases=len(observed),mean_case_score=sum(r['official_case_score'] for r in observed)/len(observed),
                    full_value_coverage=sum(r['value_covered'] for r in observed),results=observed,
                    scope='Selected 12-case failure/control cohort, not all-RULER accuracy'))
                save(base/'status.json',dict(status='completed',cases=len(observed),at=time.time()))
            finally:
                remaining=lifecycle.stop_owned_groups((server,))
                gpu=sorted(lifecycle.gpu_processes())
                save(base/'cleanup.json',dict(remaining_owned_groups=remaining,gpu_processes=gpu))
                if remaining or gpu:raise RuntimeError('Owned answer model cleanup incomplete')
        save(root/'summary.json',dict(cases=len(fixture['rows']),results=results,
            scope='Same-input same-seed fresh MeanK vs token page-sum answer experiment; failure-biased cohort.',at=time.time()))
        save(root/'status.json',dict(status='completed',cases=len(fixture['rows']),arms=2,at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--diagnostic',type=Path)
    a=p.parse_args()
    try:
        if a.action=='prepare':prepare(a.root.resolve(),a.diagnostic.resolve())
        else:run(a.root.resolve())
    except BaseException as e:
        if a.root.exists():save(a.root/'failure.json',dict(error=repr(e),at=time.time()))
        if a.action=='run':save(a.root/'status.json',dict(status='failed',error=repr(e),at=time.time()))
        raise
