"""Prepare/qualify the standalone native InputBatch observer; never a TPS test.

Preparation is CPU-only. Run only a fresh frozen entry point under the shared
GPU lock. Two small synthetic tool sessions use native async scheduling and
append their own outputs. All model cleanup precedes the explicit seal RPC.
Host dispatch/completion spans are never used as GPU durations or batch weights.
"""
import argparse
import ast
import asyncio
import copy
import fcntl
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/qualify_native_batch_observer.py'
TEST = 'tests/test_qualify_native_batch_observer.py'
OBSERVER = 'diagnostics/native_executed_batch_observer.py'
KIND = 'native_batch_observer_qualification_v1'
COUNTS = (256,384,256)
SOURCE_FILES = ('v1/worker/worker_base.py','v1/worker/gpu_worker.py',
    'v1/worker/gpu/model_runner.py','v1/worker/gpu/async_utils.py',
    'v1/worker/gpu/states.py','v1/worker/gpu/input_batch.py')
HELPERS = ('qualify_action_metadata_upload.py','benchmark_ready_agents.py',
    'benchmark_capacity_scaling.py','benchmark_timed_agents.py',
    'benchmark_native_agents.py','native_agent_launch.py','timed_agent_workload.py',
    'qualify_coordination.py')


def module(filename):
    path=ROOT/'scripts'/filename
    spec=importlib.util.spec_from_file_location('observer_qualification_'+path.stem,path)
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value
    spec.loader.exec_module(value)
    return value


gate=module('qualify_action_metadata_upload.py')
read,save,sha,require,inventory=gate.read,gate.save,gate.sha,gate.require,gate.inventory


def loopback(host):
    try:return isinstance(host,str) and ipaddress.ip_address(host).is_loopback
    except ValueError:return False


def qualification_profile(source,port):
    p=copy.deepcopy(source)
    require(p.get('host')=='127.0.0.1', 'This streaming harness requires the literal loopback127.0.0.1')
    require(type(port) is int and 1024<=port<=65535,'Dedicated port required')
    require(p.get('model_format')=='safetensors' and p.get('draft_method')=='mtp' and
        p.get('draft_tokens')==4 and p.get('execution_mode')=='compiled_graph',
        'Require qualified native NVFP4/FP8 MTP4 graph profile')
    require(p.get('max_model_len')==262144,'Keep native full context ceiling')
    p.update(port=port,max_num_seqs=2)
    # Preserve the SOURCE native effective token budget before reducing slots.
    if p.get('query_replay_budget') is not None:
        p['max_num_batched_tokens']=source['query_replay_budget']+source['max_num_seqs']*(source['draft_tokens']+1)
        del p['query_replay_budget']
    return p


def inject_observer(launch,profile,diagnostics,config):
    """Only an explicit frozen diagnostic may enable the development RPC."""
    command=list(launch['argv']);environment=dict(launch['environment'])
    require(loopback(profile['host']), 'Diagnostic host must be loopback')
    require(command.count('--host')==1 and command[command.index('--host')+1]==profile['host'],
            'Actual native host differs from validated loopback')
    require(not any(x=='--worker-extension-cls' or x.startswith('--worker-extension-cls=') for x in command),
            'Do not override an existing worker extension')
    require('--async-scheduling' in command and '--no-async-scheduling' not in command,
            'Keep native async scheduling enabled')
    require(environment.get('VLLM_PLUGINS')=='','No KVMem/plugin injection allowed')
    diagnostics,config=Path(diagnostics),Path(config)
    require(diagnostics.is_absolute() and config.is_absolute(),'Frozen absolute diagnostic paths required')
    command+=['--worker-extension-cls','native_executed_batch_observer.Extension']
    overrides=dict(PYTHONPATH=str(diagnostics),NATIVE_BATCH_OBSERVER_CONFIG=str(config),VLLM_SERVER_DEV_MODE='1')
    environment.update(overrides)
    return dict(argv=command,environment=environment,environment_overrides=overrides,
                original_argv=launch['argv'])


def small_fixture(tokenizer):
    """Deterministic synthetic data, not cropped DeepSWE or quality evidence."""
    plans=[]
    for user,length in enumerate((4096,8192)):
        note=tokenizer.encode(f'Agent {user}: inspect local test logs, then report the next check. ',add_special_tokens=False)
        require(note and all(type(n) is int and n>=0 for n in note),'Invalid local tokenizer result')
        initial=(note*((length+len(note)-1)//len(note)))[:length]
        tools=[tokenizer.encode(f'\n<tool_response>user={user}; round={turn}; test completed, exit=0; '
            'continue checking the implementation.</tool_response>\n',add_special_tokens=False) for turn in range(3)]
        plans.append(dict(initial_tokens=initial,tool_tokens=tools,tool_wait_seconds=[0,.02+user*.03,.04+user*.03]))
    return dict(schema='native_observer_synthetic_tools_v1',eos=tokenizer.eos_token_id,agents=plans,
                output_quotas=list(COUNTS),scope='Synthetic correctness only; not performance or agent quality')


def validate_fixture(f,p):
    require(f.get('schema')=='native_observer_synthetic_tools_v1' and f.get('output_quotas')==list(COUNTS),
            'Unexpected fixture protocol')
    require(type(f.get('eos')) is int and f['eos']>=0 and len(f.get('agents',[]))==2,'Require two users and EOS')
    for index,agent in enumerate(f['agents']):
        require(len(agent['initial_tokens'])==(4096,8192)[index],'Keep small distinct initial contexts')
        require(len(agent['tool_tokens'])==len(agent['tool_wait_seconds'])==3,'Require three rounds')
        for seq in [agent['initial_tokens'],*agent['tool_tokens']]:
            require(seq and all(type(t) is int and t>=0 for t in seq),'Invalid complete token IDs')
        require(all(type(x) in (int,float) and 0<=x<=1 for x in agent['tool_wait_seconds']), 'Bounded tool waits required')
        require(len(agent['initial_tokens'])+sum(map(len,agent['tool_tokens'][1:]))+sum(COUNTS)<p['max_model_len'],
                'Full own-output history must fit without truncation')


def prepare(a):
    root=a.output.resolve();require(not root.exists(),'Use a new output directory')
    require(a.observer_manifest.name=='native-executed-batch-observer-source-r2-20261017.json',
            'Explicit reviewed observer r2 manifest required')
    provenance=read(a.observer_manifest)
    require(provenance.get('schema')=='native_executed_batch_observer_source_v2' and
            provenance.get('explicit_post_drain_rpc_required_for_complete_proof') is True,'Require explicit seal revision')
    for name,expected in provenance['sha256'].items():
        require(sha(ROOT/name)==expected,'Reviewed observer artifact changed: '+name)
    tree=ast.parse((ROOT/OBSERVER).read_text())
    source_files=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and
                      any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets))
    require(tuple(source_files)==SOURCE_FILES,'Native hook source set changed')
    identity=read(a.installed_native);source=Path(identity['source_directory'])
    require(source.is_absolute() and identity.get('files'),'Full installed native identity required')
    hooks={name:identity['files'][str(source/name)] for name in SOURCE_FILES}
    p=qualification_profile(read(a.config),a.port)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(p['tokenizer'],local_files_only=True)
    fixture=small_fixture(tokenizer);validate_fixture(fixture,p)
    root.mkdir(parents=True);(root/'frozen/scripts').mkdir(parents=True);(root/'frozen/tests').mkdir()
    for name in (*HELPERS,Path(ENTRY).name):shutil.copyfile(ROOT/'scripts'/name,root/'frozen/scripts'/name)
    shutil.copyfile(ROOT/TEST,root/'frozen'/TEST)
    (root/'diagnostics').mkdir();shutil.copyfile(ROOT/OBSERVER,root/'diagnostics'/Path(OBSERVER).name)
    shutil.copyfile(a.installed_native,root/'installed-native-identity.json')
    shutil.copyfile(a.observer_manifest,root/'observer-source-manifest.json')
    save(root/'profile.json',p);save(root/'fixture.json',fixture)
    save(root/'observer.json',dict(schema='native_batch_observer_v1',enabled=True,
        output_directory=str(root/'observation'),source_sha256=hooks))
    launch=module('native_agent_launch.py').build_launch(p,port=p['port'],async_scheduling=True,environ={})
    injected=inject_observer(launch,p,root/'diagnostics',root/'observer.json')
    save(root/'planned-launch.json',{k:v for k,v in injected.items() if k!='environment'})
    files=('profile.json','fixture.json','observer.json','installed-native-identity.json',
           'observer-source-manifest.json','planned-launch.json')
    save(root/'manifest.json',dict(kind=KIND,created_wall=time.time(),source_sha256=inventory(root/'frozen'),
        diagnostic_sha256=inventory(root/'diagnostics'),files_sha256={p:sha(root/p) for p in files},
        observer_manifest_source=str(a.observer_manifest.resolve()),observer_manifest_sha256=sha(a.observer_manifest),
        source_profile=str(a.config.resolve()),source_profile_sha256=sha(a.config),
        idle_seconds=10,server_deadline_seconds=1800,minimum_free_host_bytes=8*1024**3,
        required_available_host_bytes=32*1024**3,concurrency=2,output_quotas=list(COUNTS),
        scope='Actual completed InputBatch identity/phase/frequency qualification; never GPU time-weighted batch or TPS.',
        optional_coverage=['two simultaneously pending AsyncOutputs','out-of-order output completion','mixed prefill/decode'],
        cancellation_validated=False))
    save(root/'status.json',dict(status='prepared',gpu_started=False,at=time.time()))


def verify(root,*,frozen_entry=True):
    if frozen_entry:require(Path(__file__).resolve()==root/'frozen'/ENTRY,'Run the frozen qualification entry')
    m=read(root/'manifest.json');require(m.get('kind')==KIND,'Unexpected qualification kind')
    require(inventory(root/'frozen')==m['source_sha256'] and inventory(root/'diagnostics')==m['diagnostic_sha256'],
            'Frozen harness or diagnostic changed')
    for name,expected in m['files_sha256'].items():require(sha(root/name)==expected,'Frozen input changed: '+name)
    require(sha(Path(m['observer_manifest_source']))==m['observer_manifest_sha256'],'Observer provenance changed')
    source=Path(m['source_profile']);require(sha(source)==m['source_profile_sha256'],'Source native profile changed')
    p=read(root/'profile.json');require(p==qualification_profile(read(source),p['port']),'Unexpected profile delta')
    validate_fixture(read(root/'fixture.json'),p)
    require(m['concurrency']==2 and m['output_quotas']==list(COUNTS),'Fixture controls changed')
    provenance=read(root/'observer-source-manifest.json')
    require(m['diagnostic_sha256']=={'native_executed_batch_observer.py':provenance['sha256'][OBSERVER]},
            'Prepared module differs from reviewed observer')
    identity=read(root/'installed-native-identity.json');native=Path(identity['source_directory'])
    require(read(root/'observer.json')==dict(schema='native_batch_observer_v1',enabled=True,
        output_directory=str(root/'observation'),
        source_sha256={name:identity['files'][str(native/name)] for name in SOURCE_FILES}),
        'Observer hook identities or owned output path changed')
    return m


def observation_proof(events,rows,seal,initial,*,max_requests,window=None):
    """IDs and event ordering only. Host spans never measure GPU execution time."""
    require(initial['dispatches']==initial['completed']==initial['pending']==initial['observer_errors']==0,
            'Startup/warmup leaked into observed public steps')
    require(seal.get('explicit_seal') is True and seal.get('complete_coverage') is True,'Missing explicit complete seal')
    require(all(seal.get(k)==0 for k in ('pending','failed','unmatched','dropped','observer_errors','post_close_dispatches'))
            and seal.get('armed') is False and seal.get('pending_sample') is False and
            all(seal['writer'].get(k)==0 for k in ('dropped','io_errors','queued')) and
            seal['writer'].get('writer_alive') is False,'Seal contains outstanding work, loss or errors')
    require(rows and all(r['status']=='completed' for r in rows),'Every submitted request must drain')
    responses=[r['response_id'] for r in rows]
    require(len(set(responses))==len(rows) and all(isinstance(r,str) and r for r in responses),'Unique HTTP responses required')
    by_response={r['response_id']:r for r in rows}
    require(len({r['request_id'] for r in rows})==len(rows),'Unique client requests required')
    if window is not None:require(window['begin']<window['end'],'Invalid step-selection window')
    def owner(identity):
        match=re.fullmatch(r'(.+)-0-[0-9a-f]{8}',identity)
        require(match is not None and match[1] in by_response,'Actual worker identity absent in HTTP outputs: '+identity)
        return by_response[match[1]]
    pending={};dispatch={};completed=[];membership={r['request_id']:{'prefill':0,'decode':0} for r in rows}
    max_inflight=0;max_width=0;pure={};mixed=0;raw=0;reordered=False;selected=[];mixed_widths={}
    for record in events:
        if record['kind']=='dispatch':
            b=record['batch'];key=b['execution_id'];require(key not in dispatch,'Duplicate target dispatch')
            n=b['num_reqs'];require(0<n<=max_requests and len(b['req_ids'])==len(set(b['req_ids']))==n,'Invalid actual batch membership')
            require(all(len(b[k])==n for k in ('is_prefilling','prompt_lengths','scheduled_token_upper_counts')),'Malformed batch metadata')
            require(all(x>0 for x in b['scheduled_token_upper_counts']),'Zero scheduled actual member')
            phases=[]
            for j,identity in enumerate(b['req_ids']):
                r=owner(identity);require(b['prompt_lengths'][j]==r['input_tokens'],'Canonical input prompt length differs')
                phase='prefill' if b['is_prefilling'][j] else 'decode';membership[r['request_id']][phase]+=1;phases.append(phase)
            if window is None or window['begin']<=b['dispatch_monotonic']<window['end']:
                selected.append(key)
                if all(p=='decode' for p in phases):pure[str(n)]=pure.get(str(n),0)+1
                elif 'decode' in phases:
                    mixed+=1;label=str(phases.count('decode'))+'decode/'+str(phases.count('prefill'))+'prefill'
                    mixed_widths[label]=mixed_widths.get(label,0)+1
            dispatch[key]=b;pending[key]=b;max_inflight=max(max_inflight,len(pending));max_width=max(max_width,n)
        elif record['kind']=='completed':
            key=record['execution_id'];require(key in pending and record.get('executed') is True,'Unmatched completion')
            counts=record['raw_sampled_counts'];require(len(counts)==pending[key]['num_reqs'] and all(type(x) is int and x>=0 for x in counts),'Bad sampled counts')
            reordered|=any(k<key for k in pending);raw+=sum(counts);completed.append(key);del pending[key]
        else:raise ValueError('Failed or unknown observer event')
    require(not pending and len(dispatch)==len(completed)==seal['dispatches']==seal['completed'] and dispatch,
            'Event file and seal counts do not fully reconcile')
    require(seal['writer'].get('written')==len(events),'Writer record count differs from complete file')
    require(all(v['prefill']+v['decode']>0 for v in membership.values()),'A submitted request was not observed')
    return dict(passed=True,completed_steps=len(completed),max_actual_batch=max_width,max_inflight=max_inflight,
        two_inflight_observed=max_inflight>=2,out_of_order_completion_observed=reordered,
        pure_decode_step_histogram=pure,mixed_steps=mixed,mixed_step_histogram=mixed_widths,
        selected_completed_steps=len(selected),selection_window=window,request_membership=membership,
        raw_sampler_tokens=raw,client_accepted_tokens=sum(len(r['output_ids']) for r in rows),
        scope='Step counts and membership only; no GPU duration, utilization or time-weighted batch. Raw sampler tokens are not final accepted counts.',
        optional_coverage=dict(two_inflight='covered' if max_inflight>=2 else 'not_observed',
            out_of_order_completion='covered' if reordered else 'not_observed',mixed='covered' if mixed else 'not_observed'))


def analyze_observation(events,rows,seal,initial):
    require(len(rows)==8 and sum(r['kind']=='workload' for r in rows)==6 and
            sum(r['kind']=='cleanup' for r in rows)==2,'All six rounds and two cleanup calls must drain')
    require({(r['agent'],r['turn'],r['kind']) for r in rows}==
            {(user,turn,'cleanup' if turn==3 else 'workload') for user in range(2) for turn in range(4)},
            'Missing or duplicate independent user/round identity')
    require(all(len(r['output_ids'])==(1 if r['kind']=='cleanup' else COUNTS[r['turn']]) for r in rows),
            'Fixed correctness quotas did not complete')
    proof=observation_proof(events,rows,seal,initial,max_requests=2)
    require(proof['max_actual_batch']==2,'No real two-request InputBatch; concurrency qualification incomplete')
    require(all(proof['request_membership'][r['request_id']]['prefill'] and
                proof['request_membership'][r['request_id']]['decode'] for r in rows if r['kind']=='workload'),
            'Each work request requires real prefill and decode coverage')
    return proof


def history_proof(rows,fixture,profile):
    """Rebuild every prompt from its own accepted IDs and complete tool input."""
    validate_fixture(fixture,profile)
    require(len(rows)==8 and len({(r['agent'],r['turn']) for r in rows})==8,'Incomplete cross-round transcript')
    def digest(tokens):return hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
    by_key={(r['agent'],r['turn']):r for r in rows};proof=[];sessions=set()
    for user in range(2):
        plan=fixture['agents'][user];tokens=list(plan['initial_tokens']);session=None
        for turn in range(4):
            require((user,turn) in by_key,'Missing user continuation')
            row=by_key[user,turn];cleanup=turn==3
            if turn and not cleanup:tokens.extend(plan['tool_tokens'][turn])
            expected=[fixture['eos']] if cleanup else tokens
            count=1 if cleanup else COUNTS[turn]
            require(row['status']=='completed' and row['kind']==('cleanup' if cleanup else 'workload') and
                row.get('generation_mode')=='fixed' and row.get('declared_generation_ceiling')==count and
                row.get('finish_reason')=='length','Unexpected cross-round generation contract')
            require(row['input_tokens']==len(expected) and row.get('input_sha256')==digest(expected),
                    'Prompt omits or changes initial history, own accepted output or complete tool result')
            output=row['output_ids'];require(isinstance(output,list) and len(output)==count and
                all(type(t) is int and t>=0 for t in output) and row.get('output_sha256')==digest(output) and
                row.get('actual_output_tokens')==count and row.get('usage',{}).get('completion_tokens')==count and
                row['usage'].get('prompt_tokens')==len(expected),'Accepted outputs/hash/usage disagree')
            require(row.get('new_input_tokens')==(1 if cleanup else len(expected) if turn==0 else len(plan['tool_tokens'][turn])),
                    'Incremental tool input accounting differs')
            session=row['session'] if session is None else session
            require(isinstance(session,str) and session and row['session']==session,'Cross-round session changed')
            proof.append(dict(agent=user,turn=turn,input_tokens=len(expected),input_sha256=digest(expected),
                              accepted_tokens=count,output_sha256=digest(output)))
            if not cleanup:tokens.extend(output)
        sessions.add(session)
    require(len(sessions)==2,'Independent users share a session/cache salt')
    return dict(passed=True,complete_rounds=6,cleanup_calls=2,requests=proof,
                scope='Complete logical prompts use each own accepted output; no KVMem state or quality claim')


def completed_proof(root,observer_sha):
    """A downstream diagnostic must use the exact GPU-qualified observer revision."""
    root=Path(root).resolve();m=verify(root,frozen_entry=False)
    require(read(root/'status.json').get('status')=='completed_pending_review','Observer GPU qualification is not complete')
    gate.installed_native_proof(root)
    require(m['diagnostic_sha256']=={'native_executed_batch_observer.py':observer_sha},'Observer revision differs from GPU qualification')
    result=read(root/'results.json')
    require(result.get('status')=='passed' and result.get('cleanup_completed') is True and
            result.get('native_reconciliation',{}).get('passed') is True and
            result.get('observation',{}).get('passed') is True,'Missing completed reconciled observation proof')
    for key in ('explicit_seal','final_snapshot'):
        proof=result.get(key,{})
        require(proof.get('complete_coverage') is True and proof.get('explicit_seal') is True and
                proof.get('post_close_dispatches')==0,'Missing complete post-drain seal proof')
    logs=list((root/'observation').glob('worker-*/events.jsonl'))
    require(len(logs)==1,'Require exactly one qualified worker log')
    initial=read(root/'observer-before.json');seal=read(root/'seal-response.json')
    require(seal==result['explicit_seal'] and read(root/'observer-final.json')==result['final_snapshot'],
            'Persisted seal or final snapshot differs from qualification result')
    summary=read(logs[0].parent/'summary.json')
    require(summary==result.get('shutdown_summary') and summary.get('complete_coverage') is True and
            summary.get('explicit_seal') is True and summary.get('post_close_dispatches')==0,
            'Final worker summary changed or lost coverage')
    rows=read(root/'workload/traces.json')
    proof=analyze_observation(gate.records(logs[0]),rows,seal,initial)
    require(proof==result['observation'],'Qualification observation changed after completion')
    require(history_proof(rows,read(root/'fixture.json'),read(root/'profile.json'))==result.get('cross_round_history'),
            'Qualification logical history proof changed')
    reconciliation=result['native_reconciliation']
    require(reconciliation.get('native_tokens')==reconciliation.get('client_tokens')==proof['client_accepted_tokens']
            and reconciliation.get('preemptions')==0,'Accepted native/client counts or preemptions differ')
    return dict(passed=True,manifest_sha256=sha(root/'manifest.json'),results_sha256=sha(root/'results.json'),
        observer_sha256=observer_sha,installed_native_sha256=m['files_sha256']['installed-native-identity.json'])


async def workload(client,p,f,directory,shared):
    directory.mkdir();rec=shared.work.Recorder(directory,2)
    async def user(i):
        plan=f['agents'][i];tokens=list(plan['initial_tokens']);session=f'observer-qualification-{i}'
        for turn,count in enumerate(COUNTS):
            if turn:
                rec.phase(i,'tool_wait',0,turn);await asyncio.sleep(plan['tool_wait_seconds'][turn])
                tokens.extend(plan['tool_tokens'][turn])
            row=await shared.work.stream_call(client,p,tokens,session,count,rec,agent=i,epoch=0,turn=turn,
                kind='workload',request_id=f'observer-{i}-{turn}',new_input_tokens=len(tokens) if not turn else len(plan['tool_tokens'][turn]),query=False)
            tokens.extend(row['output_ids'])
        await shared.work.stream_call(client,p,[f['eos']],session,1,rec,agent=i,epoch=0,turn=3,
            kind='cleanup',request_id=f'observer-{i}-cleanup',new_input_tokens=1,query=False)
    try:
        result=await asyncio.gather(user(0),user(1),return_exceptions=True)
        errors=[r for r in result if isinstance(r,BaseException)]
        if errors:raise errors[0]
        return rec.rows
    finally:save(directory/'traces.json',rec.rows);rec.progress();rec.stream.close()


async def rpc(client,url,method):
    response=await client.post(url+'/collective_rpc',json=dict(method=method,args=[],kwargs={},timeout=10),timeout=15)
    response.raise_for_status();results=response.json().get('results')
    require(isinstance(results,list) and len(results)==1 and isinstance(results[0],dict),'Exactly one observed native worker required')
    return results[0]


async def run_server(root,m,shared):
    import httpx
    import psutil
    await shared.bench.q.wait_idle(root,m['idle_seconds'])
    require(verify(root)==m,'Frozen state changed while waiting');gate.installed_native_proof(root)
    require(not shared.bench.q.gpu_state()['pids'],'Foreign GPU process before launch')
    require(shared.cap.host_memory()['MemAvailable']>=m['required_available_host_bytes'],'Host headroom insufficient')
    p=read(root/'profile.json');base=shared.native.launch.build_launch(p,port=p['port'],async_scheduling=True)
    launch=inject_observer(base,p,root/'diagnostics',root/'observer.json')
    require({k:v for k,v in launch.items() if k!='environment'}==read(root/'planned-launch.json'),'Actual diagnostic launch differs')
    launch['environment'].update(MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1',PYTHONDONTWRITEBYTECODE='1',VLLM_LOG_STATS_INTERVAL='1')
    save(root/'launch.json',{k:v for k,v in launch.items() if k!='environment'})
    with (root/'server.log').open('x') as log:
        server=subprocess.Popen(launch['argv'],env=launch['environment'],cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'server-owner.json',dict(pid=server.pid,create_time=psutil.Process(server.pid).create_time()))
        save(root/'status.json',dict(status='loading',pid=server.pid,at=time.time()))
        async def guard():
            while True:
                state=await asyncio.to_thread(shared.bench.q.gpu_state)
                require(server.poll() is None,'Owned native server exited')
                require(not(set(state['pids'])-shared.bench.q.owned_pids(server)),'Foreign GPU process appeared')
                require(shared.cap.host_memory()['MemAvailable']>=m['minimum_free_host_bytes'],'Host free memory below8GiB')
                await asyncio.sleep(1)
        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(180,connect=5),trust_env=False) as client:
                url=f"http://127.0.0.1:{p['port']}"
                for _ in range(1500):
                    try:
                        response=await client.get(url+'/health',timeout=2)
                        if response.status_code==200:break
                    except httpx.HTTPError:pass
                    await asyncio.sleep(1)
                else:raise TimeoutError('Native startup')
                initial=await rpc(client,url,'native_executed_batch_snapshot');save(root/'observer-before.json',initial)
                require(initial['dispatches']==initial['completed']==initial['pending']==0,'Warmup observation leak')
                async def snapshot():
                    t=time.monotonic();response=await client.get(url+'/metrics',timeout=5);response.raise_for_status()
                    row=dict(started_at=t,finished_at=time.monotonic(),values=shared.native.metrics(response.text))
                    with (root/'native_samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    return row
                before=await snapshot();save(root/'status.json',dict(status='running',pid=server.pid,at=time.time()))
                rows=await workload(client,dict(p,native_vllm=True),read(root/'fixture.json'),root/'workload',shared)
                history=history_proof(rows,read(root/'fixture.json'),p)
                reconciled=await shared.cap.reconcile_outputs(snapshot,before,sum(len(r['output_ids']) for r in rows),timeout=60.)
                # Native gauge is supplemental; completed clients and seal's own pending guards are mandatory.
                after=await snapshot();require(all(after['values'][k]==0 for k in ('vllm:num_requests_running','vllm:num_requests_waiting')),'Native requests did not retire')
                seal=await rpc(client,url,'native_executed_batch_seal');save(root/'seal-response.json',seal)
                paths=list((root/'observation').glob('worker-*/events.jsonl'));require(len(paths)==1,'Expected one native worker log')
                events=gate.records(paths[0]);analysis=analyze_observation(events,rows,seal,initial)
                persisted=read(paths[0].parent/'summary.json');require(persisted==seal,'Atomic worker seal and HTTP proof differ')
                # No inference or EOS cleanup after seal. Only a CPU RPC verifies its final status.
                final=await rpc(client,url,'native_executed_batch_snapshot');save(root/'observer-final.json',final)
                require(final['complete_coverage'] and final['explicit_seal'] and final['post_close_dispatches']==0,'Post-seal work invalidated proof')
                return dict(status='passed',cleanup_completed=True,native_reconciliation=reconciled,
                    observation=analysis,cross_round_history=history,explicit_seal=seal,final_snapshot=final,
                    cancellation_validated=False)
        try:
            return await asyncio.wait_for(shared.supervise_monitored(measure(),guard()),m['server_deadline_seconds'])
        finally:await asyncio.to_thread(shared.bench.q.stop_owned,server)


async def run(a):
    import psutil
    root=a.output.resolve();m=verify(root)
    require(read(root/'status.json')['status']=='prepared','No restart or automatic retry')
    gate.installed_native_proof(root)
    owner=psutil.Process();save(root/'owner.json',dict(pid=owner.pid,create_time=owner.create_time(),argv=sys.argv,at=time.time()))
    shared=module('benchmark_ready_agents.py')
    loop=asyncio.get_running_loop();task=asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,task.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=await run_server(root,m,shared)
            require(verify(root)==m,'Frozen inputs changed');gate.installed_native_proof(root)
            require(not shared.bench.q.gpu_state()['pids'],'Owned cleanup incomplete or foreign GPU process remains')
            summaries=list((root/'observation').glob('worker-*/summary.json'))
            require(len(summaries)==1,'Missing worker coverage proof')
            summary=read(summaries[0])
            require(summary.get('complete_coverage') is True and summary.get('explicit_seal') is True and
                    summary.get('post_close_dispatches')==0 and summary['dispatches']==result['explicit_seal']['dispatches'],
                    'Owned shutdown or post-seal dispatch invalidated coverage')
            result['shutdown_summary']=summary
            save(root/'results.json',result)
        save(root/'status.json',dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        save(root/'status.json',dict(status='failed_no_retry',error=repr(error),at=time.time()));raise
    finally:
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare-only','verify-prepared','run-prepared'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    for name in ('config','installed-native','observer-manifest'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--port',type=int,default=18572)
    a=parser.parse_args()
    if a.run_prepared:return asyncio.run(run(a))
    if a.verify_prepared:verify(a.output.resolve());print('verified_not_launched');return
    require(all((a.config,a.installed_native,a.observer_manifest)),'Explicit native profile/identity/observer r2 required')
    prepare(a)


if __name__=='__main__':main()
