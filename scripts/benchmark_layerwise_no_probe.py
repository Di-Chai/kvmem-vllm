"""Frozen, GPU-exclusive replay/no-probe qualification followed by ABBA timing.

Original full MAB inputs and generation controls are retained. A third control
changes only MTP's history source so its effect is separate from replay removal.
Never interrupts another GPU owner or silently retries an unsuccessful trial.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from audit_layerwise_gilopt import read, save_new, sha, tree

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO/'artifacts/mab-pressure-optimized-20261105-r3/formal'


def save(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def prepare(root, cases=4):
    if root.exists():raise ValueError('Use a fresh benchmark directory')
    fixture=read(SOURCE/'fixture.json')
    rows=[r for r in fixture['rows'] if r['id'].startswith('005_')][:cases]
    if len(rows)!=cases or cases<2:raise ValueError('Need a cold Query and at least one warm Query')
    root.mkdir(parents=True)
    runtime=root/'runtime'
    for name in ('src','scripts'):
        shutil.copytree(REPO/name,runtime/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    ids={r['id'] for r in rows}
    fixture['rows']=rows
    fixture['bindings']=[r for r in fixture['bindings'] if r['id'] in ids]
    fixture['cases']=cases
    profile=read(SOURCE/'profile.json')
    profile['port']=19164
    profile['served_model_name']='layerwise-no-probe'
    # Serial clients keep public batch composition identical for equivalence.
    # The server's existing batch capacity and all KV/model settings are kept.
    save_new(root/'fixture.json',fixture)
    save_new(root/'profile.json',profile)
    save_new(root/'runtime-proof.json',tree(runtime))
    files={str(p):sha(p) for p in (SOURCE/'fixture.json',SOURCE/'profile.json')}
    for row in rows:files[row['source']]=row['source_cache_sha256']
    for row in fixture['bindings']:files[row['request']]=row['sha256']
    save_new(root/'plan.json',dict(files=files,order=['legacy','replay','no-probe'],
        performance_order=['legacy','replay','no-probe','no-probe','replay','legacy'],
        fixture_sha256=sha(root/'fixture.json'),profile_sha256=sha(root/'profile.json'),
        scope='Full original inputs and sampling; one cold and subsequent frozen-P warm requests per trial. '
              'Serial client, original max_num_seqs. Excludes loading; no accuracy grading.'))


def verify(root):
    plan=read(root/'plan.json')
    if tree(root/'runtime') != read(root/'runtime-proof.json'):
        raise ValueError('Frozen runtime changed')
    if sha(root/'fixture.json')!=plan['fixture_sha256'] or sha(root/'profile.json')!=plan['profile_sha256']:
        raise ValueError('Frozen workload changed')
    for path,digest in plan['files'].items():
        if sha(Path(path))!=digest:raise ValueError('Input or reference changed: '+path)


def stream(profile,payload):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request=urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(dict(payload,stream=True,stream_options={'include_usage':True})).encode(),
        headers={'Content-Type':'application/json'})
    started=time.perf_counter();first=None;done=False
    result=dict(output_ids=[],text='',usage=None,response_id=None,finish_reason=None,chunks=[])
    try:
        with opener.open(request,timeout=3600) as response:
            for line in response:
                if not line.startswith(b'data:'):continue
                value=line[5:].strip()
                if value==b'[DONE]':done=True;break
                event=json.loads(value)
                if event.get('error'):raise RuntimeError(event['error'])
                result['response_id']=event.get('id') or result['response_id']
                result['usage']=event.get('usage') or result['usage']
                for choice in event.get('choices',[]):
                    ids=choice.get('token_ids') or []
                    if ids:
                        at=time.perf_counter()-started
                        if first is None:first=at
                        result['chunks'].append([at,len(ids)])
                        result['output_ids'].extend(ids)
                    result['text']+=choice.get('text') or ''
                    result['finish_reason']=choice.get('finish_reason') or result['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: '+error.read().decode()) from None
    if (not done or first is None or not result['usage'] or
            result['usage']['completion_tokens']!=len(result['output_ids']) or
            result['usage']['prompt_tokens']!=len(payload['prompt'])):
        raise RuntimeError('Incomplete output stream or token accounting mismatch')
    result.update(ttft_seconds=first,seconds=time.perf_counter()-started,started_at=started)
    return result


def metrics(profile):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"http://127.0.0.1:{profile['port']}/metrics",timeout=10) as response:
        lines=response.read().decode().splitlines()
    result={}
    for line in lines:
        if line.startswith('vllm:spec_decode_') and '_total' in line.split('{')[0]:
            key,value=line.rsplit(' ',1)
            name=key.split('{')[0]
            result[name]=result.get(name,0.)+float(value)
    return result


def events_for(path,rid):
    if not path.exists():return []
    return [row for line in path.open() if (row:=json.loads(line)).get('request_id','').startswith(rid+'-')]


def trial(root,name,mode,qualification):
    verify(root)
    from benchmark_query_fps_prebuild_live import server_environment
    from run_layerwise_memory import normalize_cuda_alias
    from run_recall_comparison import gpu_lock,stop_owned,wait_health,termination_guard
    from benchmark_block_graph import gpu_processes
    from benchmark_memory import source,wait_terminal
    from graph_v2_protocol import release_owner
    runtime=root/'runtime';dest=root/name;dest.mkdir(exist_ok=False)
    profile=read(root/'profile.json');fixture=read(root/'fixture.json')
    profile['verify_session_restore']=qualification
    profile['verify_session_restore_digest']=qualification
    save_new(dest/'profile.json',profile)
    flags=['--layerwise-query-diagnostic','sequential','--layerwise-query-optimized',
           '--layerwise-query-native-batches']
    flags += ['--'+key.replace('_','-') for key,value in fixture['execution_flags'].items() if value]
    if mode!='legacy':flags+=['--layerwise-query-mtp-selection','layer0']
    if mode=='no-probe':flags+=['--layerwise-query-no-probe']
    if qualification:flags+=['--layerwise-query-full-diagnostics']
    command=[sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(dest/'profile.json'),
             '--run-dir',str(dest/'server'),*flags]
    env=normalize_cuda_alias(server_environment(runtime))
    capture=dest/'server/capture';server=None
    with termination_guard(),gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
        if gpu_processes():raise RuntimeError('Another GPU owner is active; no trial started')
        try:
            with (dest/'server.log').open('x') as log:
                server=subprocess.Popen(command,cwd=runtime,env=env,stdout=log,
                                        stderr=subprocess.STDOUT,start_new_session=True)
            save_new(dest/'process.json',dict(pid=server.pid,command=command))
            wait_health(server,profile)
            before=metrics(profile)
            session='no-probe-'+uuid.uuid4().hex
            prefix=min(r['query_begin'] for r in fixture['rows'])//1024*1024
            bindings={b['id']:b for b in fixture['bindings']}
            inputs={row['id']:source(row)['tokens'] for row in fixture['rows']}
            results=[];started=time.perf_counter()
            for index,row in enumerate(fixture['rows']):
                directory=dest/row['id'];directory.mkdir()
                tokens=inputs[row['id']];binding=bindings[row['id']]
                payload=dict(binding['sampling'],model=profile['served_model_name'],prompt=tokens,
                    max_tokens=binding['max_tokens'],thinking_token_budget=binding['thinking_token_budget'],
                    return_token_ids=True,session_id=session,cache_salt=session,
                    kvmem_query_span=[row['query_begin'],row['query_end']],
                    kvmem_replay_span=[prefix,row['query_end']],kvmem_reselect='force')
                save_new(directory/'request.json',dict({k:v for k,v in payload.items() if k!='prompt'},
                    token_sha256=row['token_sha256'],input_tokens=len(tokens)))
                result=stream(profile,payload)
                save_new(directory/'response.json',result)
                terminal=wait_terminal(capture,result['response_id'])
                save_new(directory/'terminal.json',terminal)
                events=events_for(capture/'layerwise_query.jsonl',result['response_id'])
                refresh=events_for(capture/'bounded_refresh.jsonl',result['response_id'])
                queries=[r for r in refresh if r['selection']=='query']
                layers=[e for e in events if e['event']=='layer_selected']
                if len(queries)!=1 or [e['layer'] for e in layers]!=list(range(16)):
                    raise RuntimeError('Missing one complete Query/16-layer selection')
                if mode=='no-probe' and (queries[0]['private_forward_steps']!=0 or
                        queries[0]['global_assembly_bytes']!=0 or not queries[0].get('no_probe')):
                    raise RuntimeError('Candidate secretly ran private/global work')
                if mode!='no-probe' and queries[0]['private_forward_steps']<=0:
                    raise RuntimeError('Replay control did not execute a private Query')
                if terminal.get('checkpoint_tokens')!=prefix:
                    raise RuntimeError('Frozen P checkpoint was not retained')
                restores=[e for e in events if e['event']=='pressure_prefix_restored']
                if len(restores)!=int(index>0):raise RuntimeError('Warm request did not restore its frozen prefix')
                save_new(directory/'evidence.json',dict(layers=layers,events=events,refresh=refresh))
                if qualification and index:
                    digests=events_for(capture/'verified_restores_digest.jsonl',result['response_id'])
                    if not digests:raise RuntimeError('Missing warm restore byte verification')
                    save_new(directory/'restore-digest.json',digests)
                results.append(dict(id=row['id'],warm=bool(index),seconds=result['seconds'],
                    ttft_seconds=result['ttft_seconds'],output_tokens=len(result['output_ids'])))
                print(json.dumps(dict(trial=name,id=row['id'],**{k:results[-1][k] for k in ('seconds','ttft_seconds')})),flush=True)
            elapsed=time.perf_counter()-started
            after=metrics(profile)
            save_new(dest/'summary.json',dict(mode=mode,qualification=qualification,cases=results,
                inference_wall_seconds=elapsed,output_tokens=sum(r['output_tokens'] for r in results),
                requests_per_second=len(results)/elapsed,
                output_tokens_per_second=sum(r['output_tokens'] for r in results)/elapsed,
                mtp_metrics={k:after.get(k,0)-before.get(k,0) for k in set(before)|set(after)}))
            save_new(dest/'release.json',release_owner(profile,capture,session,tokens[-1]))
        finally:
            stop_owned(server)
            save(dest/'cleanup.json',dict(remaining_gpu_processes=sorted(gpu_processes())))


def compare(root,a,b,*,require_q=True):
    checks=[]
    for row in read(root/'fixture.json')['rows']:
        left,right=(root/n/row['id'] for n in (a,b))
        evidence=[read(p/'evidence.json') for p in (left,right)]
        replies=[read(p/'response.json') for p in (left,right)]
        query=[[e['selected_starts'] for e in x['layers']] for x in evidence]
        q=[[e.get('q_sha256') for e in x['layers']] for x in evidence]
        pressure=[[e['layers'] for e in x['events'] if e['event']=='pressure_selected'] for x in evidence]
        pressure=[[[l['selected_starts'] for l in e] for e in x] for x in pressure]
        check=dict(id=row['id'],same_pressure=pressure[0]==pressure[1],same_query=query[0]==query[1],
            same_output=replies[0]['output_ids']==replies[1]['output_ids'],
            same_text=replies[0]['text']==replies[1]['text'])
        if require_q:check['same_q']=q[0]==q[1] and bool(q[0]) and all(q[0])
        checks.append(check)
    return dict(before=a,after=b,checks=checks,
                passed=all(all(v for k,v in r.items() if k!='id') for r in checks))


def summarize(root, filename='performance.json'):
    arms={}
    for mode in ('legacy','replay','no-probe'):
        runs=[read(root/f'perf-{i+1}-{value}'/'summary.json')
              for i,value in enumerate(read(root/'plan.json')['performance_order']) if value==mode]
        warm=[case for run in runs for case in run['cases'] if case['warm']]
        cold=[case for run in runs for case in run['cases'] if not case['warm']]
        arms[mode]=dict(runs=len(runs),warm_requests=len(warm),
            warm_ttft_median_seconds=statistics.median(r['ttft_seconds'] for r in warm),
            warm_ttft_mean_seconds=statistics.mean(r['ttft_seconds'] for r in warm),
            warm_ttft_range_seconds=[min(r['ttft_seconds'] for r in warm),
                                     max(r['ttft_seconds'] for r in warm)],
            cold_ttft_mean_seconds=statistics.mean(r['ttft_seconds'] for r in cold),
            warm_request_mean_seconds=statistics.mean(r['seconds'] for r in warm),
            warm_output_tokens_per_second=sum(r['output_tokens'] for r in warm)/sum(r['seconds'] for r in warm),
            inference_wall_mean_seconds=statistics.mean(r['inference_wall_seconds'] for r in runs),
            request_wall_mean_seconds=statistics.mean(sum(c['seconds'] for c in r['cases']) for r in runs),
            output_tokens=sum(r['output_tokens'] for r in runs),
            output_tokens_per_second=sum(r['output_tokens'] for r in runs)/sum(r['inference_wall_seconds'] for r in runs),
            request_output_tokens_per_second=sum(r['output_tokens'] for r in runs)/sum(
                c['seconds'] for r in runs for c in r['cases']),
            mtp_metrics={k:sum(r['mtp_metrics'].get(k,0) for r in runs) for k in runs[0]['mtp_metrics']})
    comparisons={}
    for baseline in ('legacy','replay'):
        a,b=arms[baseline],arms['no-probe']
        comparisons[baseline]=dict(
            warm_ttft_change=b['warm_ttft_median_seconds']/a['warm_ttft_median_seconds']-1,
            total_wall_change=b['inference_wall_mean_seconds']/a['inference_wall_mean_seconds']-1,
            throughput_change=b['output_tokens_per_second']/a['output_tokens_per_second']-1,
            request_wall_change=b['request_wall_mean_seconds']/a['request_wall_mean_seconds']-1,
            request_throughput_change=b['request_output_tokens_per_second']/a['request_output_tokens_per_second']-1)
    save_new(root/filename,dict(arms=arms,comparisons=comparisons,
        scope=read(root/'plan.json')['scope'],note='Two ABBA trials per mode; exploratory serial workload.'))


def run(root):
    from benchmark_block_graph import gpu_processes
    from run_recall_comparison import stop_owned,termination_guard
    verify(root);child=None
    save_new(root/'controller.json',dict(pid=os.getpid(),at=time.time()))
    with termination_guard():
        try:
            while gpu_processes():
                save(root/'status.json',dict(status='waiting_for_gpu',at=time.time()))
                time.sleep(30)
            trials=[('qual-'+mode,mode,True) for mode in read(root/'plan.json')['order']]
            trials += [(f'perf-{i+1}-{mode}',mode,False)
                       for i,mode in enumerate(read(root/'plan.json')['performance_order'])]
            for name,mode,qualification in trials:
                if not qualification and not (root/'qualification.json').exists():
                    comparison=compare(root,'qual-replay','qual-no-probe')
                    save_new(root/'qualification.json',comparison)
                    save_new(root/'legacy-comparison.json',compare(root,'qual-legacy','qual-no-probe'))
                    if not comparison['passed']:
                        raise RuntimeError('Matched replay/no-probe equivalence failed; performance gated')
                save(root/'status.json',dict(status='running',trial=name,at=time.time()))
                with (root/(name+'.log')).open('x') as log:
                    child=subprocess.Popen([sys.executable,'-B',str(root/'runtime/scripts'/Path(__file__).name),
                        'trial','--root',str(root),'--name',name,'--mode',mode,
                        *(['--qualification'] if qualification else [])],stdout=log,stderr=subprocess.STDOUT,
                        start_new_session=True)
                if child.wait():raise RuntimeError('Trial failed: '+name)
                child=None
            comparisons=[compare(root,'perf-1-legacy',f'perf-{i+1}-{mode}',require_q=False)
                         for i,mode in enumerate(read(root/'plan.json')['performance_order']) if i]
            save_new(root/'performance-consistency.json',dict(checks=comparisons,
                passed=all(c['passed'] for c in comparisons)))
            summarize(root)
            save(root/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
            raise
        finally:stop_owned(child)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','trial'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cases',type=int,default=4)
    parser.add_argument('--name')
    parser.add_argument('--mode',choices=('legacy','replay','no-probe'))
    parser.add_argument('--qualification',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if args.action=='prepare':prepare(root,args.cases)
    elif args.action=='run':run(root)
    else:trial(root,args.name,args.mode,args.qualification)
