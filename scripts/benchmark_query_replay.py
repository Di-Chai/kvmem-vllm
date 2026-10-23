"""Freeze and queue owned Query-replay comparisons after existing GPU work.

Four staggered 64K/80K/96K/112K histories make public work available during a
neighbour's Query. These shortened inputs are performance fixtures, not RULER
accuracy cases. Every arm uses identical input IDs and fixed 128-token outputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s] if path.exists() else []


ARMS = (('baseline', 'eager', False), ('compiled', 'compiled', False), ('cobatch', 'compiled', True))


def prepare(root, source, repo):
    import numpy as np
    from benchmark_prefill_publication_pipeline import runtime_origins
    from ruler_execution import validate_execution

    root, source, repo = (p.resolve() for p in (root, source, repo))
    source_fixture = read(source/'fixture.json')
    require(sha(source/'profile.json') == source_fixture['profile_sha256'], 'Source profile changed')
    require(sha(source/'fixture.json') == read(source/'dataset-summary.json')['fixture_sha256'],
            'Source fixture changed')
    names = set(read(source/'runtime-proof.json')['files'])
    for folder in ('src/vllm_kvmem', 'scripts', 'csrc', 'third_party/qw3'):
        names.update(str(p.relative_to(repo)) for p in (repo/folder).rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and
            p.suffix in ('.py','.json','.so','.h','.hpp','.cpp','.cu','.cuh'))
    origins, inherited = runtime_origins(repo, source, names)
    hashes = {name:sha(path) for name,path in origins.items()}
    root.mkdir(parents=True, exist_ok=False)
    rows=[]
    for index, row in enumerate(source_fixture['rows'][:4]):
        path=Path(source_fixture['input_directory'])/(row['id']+'.i32')
        require(sha(path)==row['served_sha256'], 'Source input IDs changed')
        original=np.fromfile(path,dtype='<i4')
        begin,end=row['served_query'];prefix=(64+16*index)*1024
        require(prefix < begin, 'Query is inside the shortened prefix')
        # Preserve the exact original question and open-thinking suffix. No
        # reference answer or needle coordinates are used to construct an input.
        tokens=np.concatenate((original[:prefix],original[begin:]))
        target=root/'inputs'/(row['id']+'.i32');target.parent.mkdir(exist_ok=True)
        tokens.tofile(target)
        rows.append(dict(id=row['id'],question=row['question'],served_query=[prefix,prefix+end-begin],
            served_input_tokens=len(tokens),served_sha256=sha(target),input_path=str(target),
            source_input_sha256=row['served_sha256'],source_query=row['served_query'],seed=42))
    fixture=dict(rows=rows,workers=4,tokenizer_sha256=source_fixture['tokenizer_sha256'],
        sampling=dict(max_tokens=128,temperature=0.,top_p=1.,top_k=-1,ignore_eos=True,
                      thinking_token_budget=64),request_timeout_seconds=1800,
        performance_only=True,accuracy_claim=False,
        scope='Four staggered shortened histories; fixed output length, fresh owners, no score/accuracy claim.')
    for name,backend,cobatch in ARMS:
        arm=root/name;arm.mkdir()
        for relative,path in origins.items():
            target=arm/'runtime'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            require(sha(target)==hashes[relative], 'Runtime changed during freeze')
        save(arm/'runtime-proof.json',dict(files=hashes,inherited_utilities=inherited,
            head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()))
        profile=deepcopy(read(source/'profile.json'));profile['port']=18854
        profile['kvmem'].update(query_replay_backend=backend,query_replay_cobatch=cobatch,
                               query_replay_max_bytes=64*1024*1024)
        validate_execution(profile, source_fixture)
        save(arm/'profile.json',profile);save(arm/'fixture.json',fixture)
        save(arm/'manifest.json',dict(profile_sha256=sha(arm/'profile.json'),
            fixture_sha256=sha(arm/'fixture.json'),requested_backend=backend,cobatch=cobatch))
    require(all(sha(path)==hashes[name] for name,path in origins.items()), 'Source changed during freeze')
    shutil.copy2(repo/'scripts/benchmark_query_replay.py',root/'driver.py')
    save(root/'plan.json',dict(driver_sha256=sha(root/'driver.py'),arms=[x[0] for x in ARMS],
        source_campaign=str(source),source_fixture_sha256=sha(source/'fixture.json'),
        input_ids_identical_between_arms=True,sampling_identical_between_arms=True,
        repeats=1,performance_only=True,dsh_started=False,mab_resumed=False,
        production_defaults_changed=False))
    save(root/'status.json',dict(status='prepared',at=time.time()))
    print(json.dumps(dict(status='prepared',root=str(root))),flush=True)


def client(root):
    import numpy as np
    from tokenizers import Tokenizer
    from types import SimpleNamespace
    from benchmark_client_io import CaptureSlice
    import qualify_block_graph as q
    import qualify_graph_niah as old
    import guided_benchmark_protocol as policy
    from benchmark_matched_prefill import audit_native_stats

    fixture,profile=read(root/'fixture.json'),read(root/'profile.json')
    capture=root/'run/server/capture';output=root/'run/answers';output.mkdir(exist_ok=False)
    require(sha(Path(profile['tokenizer'])/'tokenizer.json')==fixture['tokenizer_sha256'],'Tokenizer changed')
    tokenizer=Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
    save(output/'launch-audit.json',old.audit_launch(profile,capture))
    require(sum(old.history_bytes(row,fixture,profile) for row in fixture['rows']) <=
            profile['kvmem']['cpu_read_cache_bytes']-16*1024**3, 'Four histories exceed CPU credits')
    started=time.monotonic()
    def infer(row):
        directory=output/row['id'];directory.mkdir()
        source=Path(row['input_path']);require(sha(source)==row['served_sha256'],'Input changed')
        tokens=np.fromfile(source,dtype='<i4').tolist()
        require(len(tokens)==row['served_input_tokens'] and
            tokenizer.decode(tokens[slice(*row['served_query'])],skip_special_tokens=False)==row['question'],
            'Query coordinates changed')
        session='replay-bench-'+uuid.uuid4().hex
        events=CaptureSlice(capture/name for name in ('bounded_refresh.jsonl','block_graph.jsonl','action_attention.jsonl'))
        request_at=time.monotonic()
        response=old.stream(profile,session,tokens,row['served_query'],dict(fixture['sampling'],seed=row['seed']),
                            fixture['request_timeout_seconds'])
        save(directory/'response.json',response)
        terminal=old.wait_retired(capture,response['response_id'])
        require(terminal['history_end']==len(tokens)+len(response['output_ids'])-1 and
                not terminal.get('checkpoint_tokens'),'Private tokens leaked into accepted history')
        release=old.wait_owner_released(capture,session,response,terminal)
        evidence=policy.audit_case(capture,response['response_id'],profile,row['served_query'],
                                   require_pressure=True,event_reader=events.read)
        actions=[r for r in events.read(capture/'action_attention.jsonl') if q.native_matches(response,r)]
        expected=q.expected_output_samples(SimpleNamespace(convert_tokens_to_ids=tokenizer.token_to_id),
                                           profile,tokens,response['output_ids'])
        require(len(actions)==1 and actions[0]['samples']==expected and
                actions[0]['prefill_samples']==evidence['query']['prefill_attention_samples'],
                'Private rows leaked into accepted attention or prefill scores')
        require(len(response['output_ids'])==128,'Output length differs between comparison arms')
        save(directory/'evidence.json',dict(**evidence,action=actions[0],terminal=terminal,release=release))
        result=dict(id=row['id'],session=session,input_tokens=len(tokens),output_tokens=128,
            output_sha256=hashlib.sha256(json.dumps(response['output_ids']).encode()).hexdigest(),
            ttft_seconds=response['ttft_seconds'],seconds=response['seconds'],
            first_token_since_start=request_at-started+response['ttft_seconds'])
        save(directory/'result.json',result)
        print(json.dumps(result),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=fixture['workers']) as pool:
        results=list(pool.map(infer,fixture['rows']))
    released=q.wait_released(capture,[r['session'] for r in results],timeout=180)
    archive=released['archive'];require(archive.get('disk_read_bytes',0)==archive.get('disk_write_bytes',0)==0,
                                      'Unexpected SSD history access')
    native=read(capture/'native_attention_stats.json');audit_native_stats(native)
    queries=[r for r in records(capture/'bounded_refresh.jsonl') if r['selection']=='query']
    require(len(queries)==len(results),'Missing semantic Query reports')
    modes={mode:sum(r.get('query_execution_counts',{}).get(mode,0) for r in queries)
           for mode in ('eager','compiled','cobatch')}
    backend=profile['kvmem']['query_replay_backend'];co=profile['kvmem']['query_replay_cobatch']
    if co:
        require(modes['cobatch']>0 and modes['eager']==0,'Requested mixed Query forward did not execute')
        snapshots=[r['query_replay_cobatch'] for r in queries]
        require(all(not r['poisoned'] and not r.get('quarantined',0) and
                    r.get('peak_bytes',0)<=profile['kvmem']['query_replay_max_bytes'] for r in snapshots),
                'Mixed Query lifecycle or memory budget failed')
    else:
        require(modes[backend]>0 and all(not v for k,v in modes.items() if k!=backend),
                'Standalone Query execution mode differs')
    require(not native['validation_failures'] and native['disabled_reason'] is None,'Native score validation failed')
    duration=time.monotonic()-started
    save(output/'completed.json',dict(status='completed',cases=len(results),seconds=duration,results=results,
        protocol_passed=True,execution_counts=modes,
        input_tokens_per_campaign_second=sum(r['input_tokens'] for r in results)/duration,
        input_tokens_per_all_first_tokens_second=sum(r['input_tokens'] for r in results)/
            max(r['first_token_since_start'] for r in results),
        accuracy_claim=False,native_stats=native))


def run(root):
    import benchmark_block_graph as lifecycle
    from benchmark_native_prefill import http

    root=root.resolve();runtime=root/'runtime';manifest=read(root/'manifest.json')
    require(Path(__file__).resolve()==runtime/'scripts/benchmark_query_replay.py','Use frozen runner')
    require(sha(root/'profile.json')==manifest['profile_sha256'] and
            sha(root/'fixture.json')==manifest['fixture_sha256'],'Frozen configuration changed')
    for name,digest in read(root/'runtime-proof.json')['files'].items():
        require(sha(runtime/name)==digest,'Frozen runtime changed: '+name)
    profile=read(root/'profile.json');target=root/'run';target.mkdir(exist_ok=False)
    server=client_process=None
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock, \
            lifecycle.cleanup_on_termination() as cleanup:
        try:
            save(root/'status.json',dict(status='waiting_for_gpu',at=time.time(),existing_test_untouched=True))
            fcntl.flock(lock,fcntl.LOCK_EX)  # No model/GPU allocations before the owning test exits.
            require(not lifecycle.gpu_processes(),'Foreign GPU process remains after lock release')
            env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
            env.update(PYTHONPATH=str(runtime/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
                CUDA_HOME='/usr/local/cuda-13',MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1')
            env['PATH']='/usr/local/cuda-13/bin:'+str(Path(sys.executable).parent)+':'+env.get('PATH','')
            started=time.monotonic()
            with (target/'server.log').open('x') as log:
                server=subprocess.Popen([sys.executable,'-B','-m','vllm_kvmem','serve','--config',str(root/'profile.json'),
                    '--run-dir',str(target/'server')],cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'status.json',dict(status='loading',at=time.time(),server_pid=server.pid))
            deadline=time.monotonic()+1200
            while time.monotonic()<deadline:
                require(server.poll() is None,'Replay model exited during startup')
                try:
                    with http().open(f"http://127.0.0.1:{profile['port']}/health",timeout=1) as response:
                        if response.status==200:break
                except OSError:pass
                time.sleep(1)
            else:raise TimeoutError('Replay startup exceeded 20 minutes')
            cold=time.monotonic()-started
            with (target/'client.log').open('x') as log:
                client_process=subprocess.Popen([sys.executable,'-B',str(Path(__file__)),
                    'client','--root',str(root)],cwd=runtime,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(root/'status.json',dict(status='testing',at=time.time(),server_pid=server.pid,
                                         client_pid=client_process.pid,cold_prepare_seconds=cold))
            with (target/'resources.jsonl').open('x') as log:
                while client_process.poll() is None:
                    require(server.poll() is None,'Replay model exited during testing')
                    require(time.monotonic()-started<3600,'Replay arm exceeded one hour')
                    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
                        '--format=csv,noheader,nounits'],text=True).strip()
                    log.write(json.dumps(dict(at=time.time(),gpu=gpu))+'\n');log.flush();time.sleep(5)
            require(client_process.returncode==0,'Replay client failed; inspect client.log')
            complete=read(target/'answers/completed.json')
            save(root/'status.json',dict(status='completed',at=time.time(),cold_prepare_seconds=cold,
                seconds=complete['seconds'],execution_counts=complete['execution_counts'],protocol_passed=True))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
            raise
        finally:
            cleanup()
            remaining=lifecycle.stop_owned_groups((client_process,server))
            save(root/'cleanup.json',dict(remaining_owned_groups=remaining,gpu_processes=sorted(lifecycle.gpu_processes())))
            require(not remaining,'Owned replay processes did not exit')


def analyze(root):
    comparisons={};canonical=None
    for name,_,_ in ARMS:
        arm=root/name;require(read(arm/'status.json')['status']=='completed','Incomplete comparison arm')
        require(read(arm/'cleanup.json')==dict(remaining_owned_groups={},gpu_processes=[]),'Uncleared GPU ownership')
        result=read(arm/'run/answers/completed.json')
        timings=[r['host_seconds']*1000 for r in records(arm/'run/server/capture/async_refresh.jsonl') if r['stage']=='query']
        outputs={r['id']:r['output_sha256'] for r in result['results']}
        selections={r['id']:read(arm/'run/answers'/r['id']/'evidence.json')['query']['selected_history_ranges']
                    for r in result['results']}
        if canonical is None:canonical=(outputs,selections)
        comparisons[name]=dict(seconds=result['seconds'],cold_prepare_seconds=read(arm/'status.json')['cold_prepare_seconds'],
            end_to_end_input_tokens_per_second=result['input_tokens_per_campaign_second'],
            all_first_tokens_input_tokens_per_second=result['input_tokens_per_all_first_tokens_second'],
            standalone_query_host_ms_median=statistics.median(timings) if timings else None,
            execution_counts=result['execution_counts'],outputs_identical_to_baseline=outputs==canonical[0],
            selected_history_identical_to_baseline=selections==canonical[1])
    base=comparisons['baseline']['seconds']
    for result in comparisons.values():result['campaign_speedup_percent']=(base/result['seconds']-1)*100
    save(root/'comparison.json',dict(arms=comparisons,repeats=1,accuracy_claim=False,
        caveat='One pass per arm, including actual preparation within requests. Mixed-forward durations contain '
               'public work and are not exclusive private cost. Numeric/batch trajectory differences are reported, '
               'not assumed equivalent. No full RULER accuracy score is computed.'))
    return comparisons


def queue(root):
    sys.path.insert(0,str(root.resolve()/'baseline/runtime/scripts'))
    import benchmark_block_graph as lifecycle
    root=root.resolve();require(sha(Path(__file__))==read(root/'plan.json')['driver_sha256'],'Queue driver changed')
    child=None
    with lifecycle.cleanup_on_termination() as cleanup:
        try:
            for name in read(root/'plan.json')['arms']:
                arm=root/name
                save(root/'status.json',dict(status='queued_or_testing',arm=name,at=time.time(),
                    existing_ruler_interrupted=False))
                command=[sys.executable,'-B',str(arm/'runtime/scripts/benchmark_query_replay.py'),'run','--root',str(arm)]
                with (arm/'controller.log').open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(arm/'controller-process.json',dict(pid=child.pid,command=command,at=time.time()))
                require(child.wait()==0,'Queued replay arm failed: '+name)
            analyze(root)
            save(root/'status.json',dict(status='completed',at=time.time(),comparison='comparison.json'))
        except BaseException as error:
            save(root/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
            raise
        finally:
            cleanup();lifecycle.stop_owned_groups((child,))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','client','run','queue','analyze'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--repo',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.root,args.source,args.repo)
    else:globals()[args.command](args.root)
