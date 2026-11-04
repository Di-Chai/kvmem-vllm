"""Finite natural-generation queue probe against an already-owned idle server.

Synthetic inputs only: no Terminal-Bench instructions, answers, scoring or
server lifecycle operations. Qualification is excluded from timed A/B arms.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx

try:
    from .qualify_thinking_gc_preflight import clean
    from .terminalbench_dashboard import parse_metrics
except ImportError:
    from qualify_thinking_gc_preflight import clean
    from terminalbench_dashboard import parse_metrics


CAPTURE_LOGS=('async_restore.jsonl','scheduler_refresh.jsonl','async_refresh.jsonl',
              'scheduler_finished.jsonl','restore_admission.jsonl')
PROMPT=('This is a synthetic systems explanation exercise, not a benchmark task. '
        'Develop a detailed design for an imaginary library lending service. Discuss identifiers, '
        'state transitions, invariants, concurrency, recovery, and several worked examples. '
        'Reason carefully and continue expanding the design until the response budget is exhausted. '
        'No real files, network access, external facts, or tools are needed.')
OBSERVATIONS=('Synthetic external observation: the design must also handle two simultaneous returns. '
              'Continue your analysis with detailed examples and invariant checks.',
              'Synthetic external observation: a simulated worker restarted during a transfer. '
              'Continue by developing recovery cases, tests and alternative schedules in detail.')


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def prepare(profile_path,output):
    from transformers import AutoTokenizer
    require(not output.exists(),'Use a new fixture path')
    profile=read(profile_path)
    tokenizer=AutoTokenizer.from_pretrained(profile.get('tokenizer',profile['model']),local_files_only=True)
    encode=lambda value:tokenizer.encode(value,add_special_tokens=False)
    head=encode('<|im_start|>user\n')
    query=encode(PROMPT)
    tail=encode('<|im_end|>\n<|im_start|>assistant\n<think>\n')
    additions=[encode('</think><|im_end|>\n<|im_start|>user\n'+value)+tail for value in OBSERVATIONS]
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
    require(isinstance(eos,int) and eos>=0,'Missing actual message EOS')
    fixture=dict(schema=1,prompt=head+query+tail,query_span=[len(head),len(head)+len(query)],
        additions=additions,eos_token=eos,items=24,turns=3,output_tokens=1024,tool_wait_seconds=20,
        qualification_owners=12,qualification_output_tokens=64,seed=1000,temperature=0,
        source='Synthetic public explanation and fixed simulated observations; no benchmark tasks or answers',
        generation='Natural unconstrained greedy tokens; ignore_eos=True; no allowed_token_ids or logit bias',
        profile_sha256=hashlib.sha256(profile_path.read_bytes()).hexdigest())
    save(output,fixture)
    print(json.dumps(dict(fixture=str(output),sha256=digest(fixture),initial_tokens=len(fixture['prompt']))))


def snapshot(capture):
    return {name:read(capture/(name+'.json')) for name in ('progress','archive','scheduler_resources')}


def compact_snapshot(capture):
    scheduler=read(capture/'scheduler_resources.json')
    progress=read(capture/'progress.json')
    work=read(capture/'public_work.json')
    return dict(scheduler_resources=scheduler,
        progress={key:progress.get(key) for key in ('historical_gc','redundancy_q_capture','pressure_q_capture')},
        public_work=dict(counter_id=work.get('counter_id'),wall_time=work.get('wall_time'),
            completed={key:value for key,value in work.get('completed',{}).items() if key!='batch_composition'}))


def native_errors(before,after,observations):
    snapshots=[before,after,*observations]
    policies=[(s.get('archive',{}).get('historical_gc') or s.get('progress',{}).get('historical_gc') or {}).get('policy_failures') for s in snapshots]
    overflow=[(s.get('progress',{}).get('redundancy_q_capture') or {}).get('overflow_packets') for s in snapshots]
    return dict(policy_failures_max=max((x for x in policies if isinstance(x,int)),default=None),
                query_overflow_packets_max=max((x for x in overflow if isinstance(x,int)),default=None))


def gauge_summary(samples,start,end,key):
    rows=[row for row in samples if start<=row['at']<=end and isinstance(row.get('metrics',{}).get(key),(int,float))]
    counts=Counter()
    for index,row in enumerate(rows):
        finish=rows[index+1]['at'] if index+1<len(rows) else end
        seconds=max(0,min(finish,row['at']+2.5)-row['at'])
        counts[row['metrics'][key]]+=seconds
    seconds=sum(counts.values())
    return dict(mean=sum(value*duration for value,duration in counts.items())/seconds if seconds else None,
        peak=max(counts,default=None),observed_seconds=seconds,seconds_by_count=dict(counts))


def admission_summary(samples):
    values=[row.get('snapshot',row).get('scheduler_resources',{}) for row in samples]
    admission=[row.get('cpu_history_admission',{}) for row in values]
    known=[row for row in admission if isinstance(row.get('capacity_records'),int)]
    violations=[row for row in known if not (0<=row.get('reserved_records',-1)<=row['capacity_records'])
                or row.get('reserved_bytes',0)>row.get('capacity_bytes',0)]
    return dict(observed=bool(known),capacity_violations=len(violations),
        capacity_records=sorted({row['capacity_records'] for row in known}),
        peak_reserved_records=max((row.get('reserved_records',0) for row in known),default=None),
        peak_reserved_bytes=max((row.get('reserved_bytes',0) for row in known),default=None),
        peak_idle_protected_records=max((row.get('idle_protected_records',0) for row in known),default=None),
        peak_session_owners=max((len(row.get('session_owners',[])) for row in values),default=None),
        uncached_owner_observed=any(owner.get('phase')=='uncached' and not
            set(owner.get('namespace',[])).intersection(sample.get('release_control_sessions',[]))
            for sample,row in zip(samples,values) for owner in row.get('session_owners',[])))


class Probe:
    def __init__(self,client,model,config,capture,output):
        self.client,self.model,self.config,self.capture,self.output=client,model,config,capture,output
        self.records=[];self.releases=[];self.errors=[];self.samples=[];self.observations=[]
        self.sessions={};self.all_sessions=set();self.releasing=set();self.stop=asyncio.Event()

    def event(self,event,**values):
        with (self.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(at=time.time(),event=event,**values))+'\n')

    async def metrics(self):
        response=await self.client.get('/metrics',timeout=3)
        response.raise_for_status()
        return parse_metrics(response.text)

    async def observe(self):
        while not self.stop.is_set():
            row=dict(at=time.time(),snapshot=compact_snapshot(self.capture),
                     release_control_sessions=sorted(self.releasing))
            try:row['metrics']=await self.metrics()
            except Exception as error:row['metrics_error']=repr(error)
            self.samples.append(row)
            with (self.output/'samples.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            try:await asyncio.wait_for(self.stop.wait(),timeout=1)
            except asyncio.TimeoutError:pass

    def new_owner(self,item):
        session='synthetic-queue-'+uuid.uuid4().hex
        self.sessions[item]=session;self.all_sessions.add(session)
        self.event('owner_opened',item=item,session=session)
        return session

    async def call(self,item,tokens,turn,count):
        session=self.sessions[item]
        payload=dict(model=self.model,session_id=session,cache_salt=session,prompt=tokens,max_tokens=count,
            temperature=self.config['temperature'],seed=self.config['seed'],ignore_eos=True,return_token_ids=True,
            kvmem_reselect='auto',kvmem_query_span=self.config['query_span'],kvmem_query_mode='agent',
            kvmem_query_id='synthetic-library-design-v1',kvmem_mandatory_parts='none',kvmem_mandatory_spans=[])
        began=time.time()
        self.event('request_started',item=item,turn=turn,input_sha256=digest(tokens),input_tokens=len(tokens))
        response=await self.client.post('/v1/completions',json=payload)
        response.raise_for_status()
        value=response.json();accepted=value['choices'][0].get('token_ids')
        require(isinstance(accepted,list) and len(accepted)==count and all(type(x) is int for x in accepted),
                'Natural generation did not return the exact accepted token budget')
        require(value.get('usage',{}).get('completion_tokens')==len(accepted),'Accepted IDs/usage mismatch')
        row=dict(item=item,turn=turn,session=session,request_id=value['id'],started_at=began,ended_at=time.time(),
            input_tokens=len(tokens),input_sha256=digest(tokens),accepted_tokens=len(accepted),
            output_sha256=digest(accepted),finish_reason=value['choices'][0].get('finish_reason'))
        save(self.output/'requests'/f'{item}-{turn}.json',dict(**row,input_ids=list(tokens),accepted_ids=accepted))
        self.records.append(row);self.event('request_finished',**row)
        return accepted

    async def release(self,item):
        session=self.sessions.get(item)
        if session is None:return
        self.releasing.add(session)
        try:
            response=await self.client.post('/v1/completions',json=dict(model=self.model,session_id=session,
                cache_salt=session,prompt=[self.config['eos_token']],max_tokens=1,return_token_ids=True,
                kv_transfer_params=dict(kvmem_release_session=True)))
            response.raise_for_status();value=response.json()
            self.releases.append(dict(item=item,passed=True,request_id=value['id'],
                accepted_tokens=value.get('usage',{}).get('completion_tokens')))
            del self.sessions[item]
            self.event('owner_released',item=item,session=session)
        except Exception as error:
            self.releases.append(dict(item=item,passed=False,error=repr(error)))
            raise
        finally:
            self.releasing.discard(session)

    async def wait_idle(self,expected,turn):
        deadline=time.monotonic()+30
        wanted={(session,session) for session in expected.values()}
        while True:
            value=snapshot(self.capture);scheduler=value['scheduler_resources']
            owners=scheduler.get('session_owners',[]);views=scheduler.get('session_views',[])
            actual={tuple(row['namespace']) for row in owners if row.get('phase')=='idle'}
            if actual==wanted and len(owners)==len(wanted):break
            require(time.monotonic()<deadline,'Owned sessions did not all become retained idle checkpoints')
            await asyncio.sleep(.1)
        require({tuple(row['namespace']) for row in views}==wanted,'Idle checkpoint view lost an owner')
        require(all(row.get('history',0)>0 for row in views),'Missing retained exact-prefix history')
        self.observations.append(value)
        save(self.output/f'qualification-idle-{turn}.json',value)
        return value

    async def qualify(self):
        count=self.config['qualification_owners'];tokens={}
        for item in range(count):self.new_owner(item);tokens[item]=list(self.config['prompt'])
        try:
            for turn in range(self.config['turns']):
                if turn:
                    for item in tokens:tokens[item].extend(self.config['additions'][turn-1])
                jobs=[asyncio.create_task(self.call(item,tokens[item],turn+1,
                    self.config['qualification_output_tokens'])) for item in tokens]
                try:outputs=await asyncio.gather(*jobs)
                except BaseException:
                    for job in jobs:job.cancel()
                    await asyncio.gather(*jobs,return_exceptions=True)
                    raise
                for item,accepted in zip(tokens,outputs):tokens[item].extend(accepted)
                await self.wait_idle(dict(self.sessions),turn+1)
                if turn+1<self.config['turns']:await asyncio.sleep(1)
        finally:
            for item in list(self.sessions):
                try:await self.release(item)
                except Exception as error:self.errors.append(dict(item=item,release_error=repr(error)))

    async def queue(self,concurrency):
        queue=asyncio.Queue()
        for item in range(self.config['items']):queue.put_nowait(item)
        async def worker():
            while not queue.empty():
                item=queue.get_nowait();self.new_owner(item);tokens=list(self.config['prompt'])
                try:
                    for turn in range(self.config['turns']):
                        if turn:tokens.extend(self.config['additions'][turn-1])
                        accepted=await self.call(item,tokens,turn+1,self.config['output_tokens'])
                        tokens.extend(accepted)
                        if turn+1<self.config['turns']:
                            self.event('tool_wait_started',item=item,turn=turn+1,seconds=self.config['tool_wait_seconds'])
                            await asyncio.sleep(self.config['tool_wait_seconds'])
                            self.event('tool_wait_finished',item=item,turn=turn+1)
                finally:
                    await self.release(item);queue.task_done()
        jobs=[asyncio.create_task(worker()) for _ in range(concurrency)]
        try:await asyncio.gather(*jobs)
        except BaseException:
            for job in jobs:job.cancel()
            await asyncio.gather(*jobs,return_exceptions=True)
            raise


async def run_phase(client,model,config,capture,output,label,concurrency=None):
    require(not output.exists(),'Use a new output directory')
    output.mkdir(parents=True)
    before=snapshot(capture);require(clean(before),'Server must begin with clean owned history and no other requests')
    save(output/'before.json',before)
    offsets={name:(capture/name).stat().st_size if (capture/name).exists() else 0 for name in CAPTURE_LOGS}
    probe=Probe(client,model,config,capture,output)
    baseline_metrics=await probe.metrics()
    require(baseline_metrics.get('running')==baseline_metrics.get('waiting')==0,'Server is not idle')
    require(isinstance(baseline_metrics.get('generation'),(int,float)),'Accepted generation counter unavailable')
    watcher=asyncio.create_task(probe.observe())
    start=time.time();began=time.perf_counter();failure=None
    try:
        if concurrency is None:await probe.qualify()
        else:await probe.queue(concurrency)
    except BaseException as error:
        failure=error;probe.errors.append(dict(type=type(error).__name__,message=str(error)))
    elapsed=time.perf_counter()-began;finish=time.time()
    deadline=time.monotonic()+30
    after=snapshot(capture)
    while not clean(after) and time.monotonic()<deadline:
        await asyncio.sleep(.2);after=snapshot(capture)
    probe.stop.set();await watcher
    final_metrics=await probe.metrics()
    expected_counter=baseline_metrics['generation']+sum(row['accepted_tokens'] for row in probe.records)+sum(
        row.get('accepted_tokens') or 0 for row in probe.releases)
    deadline=time.monotonic()+10
    while final_metrics.get('generation') is not None and final_metrics['generation']<expected_counter and time.monotonic()<deadline:
        await asyncio.sleep(.1);final_metrics=await probe.metrics()
    save(output/'after-release.json',after)
    save(output/'requests.json',probe.records);save(output/'releases.json',probe.releases)
    samples=probe.samples
    admission=admission_summary([*samples,*probe.observations])
    errors=native_errors(before,after,[s['snapshot'] for s in samples]+probe.observations)
    native_events={};capture_errors=[]
    request_ids=[row['request_id'] for row in probe.records]
    for name,offset in offsets.items():
        path=capture/name;events=[]
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(offset)
                for line in stream:
                    if line.endswith(b'\n'):
                        row=json.loads(line)
                        if row.get('error'):capture_errors.append(dict(file=name,error=row['error']))
                        identity=str(row.get('request_id') or '')
                        if any(identity==rid or identity.startswith(rid+'-') for rid in request_ids):events.append(row)
        native_events[name]=events
    save(output/'native-events.json',native_events)
    accepted=sum(row['accepted_tokens'] for row in probe.records)
    control_accepted=sum(row.get('accepted_tokens') or 0 for row in probe.releases)
    native_delta=final_metrics['generation']-baseline_metrics['generation'] if final_metrics.get('generation') is not None else None
    running=gauge_summary(samples,start,finish,'running');waiting=gauge_summary(samples,start,finish,'waiting')
    expected_owners=concurrency or config['qualification_owners']
    expected_requests=(config['items'] if concurrency else config['qualification_owners'])*config['turns']
    hit_delta=after['scheduler_resources'].get('sessions',{}).get('hits',0)-before['scheduler_resources'].get('sessions',{}).get('hits',0)
    qualified=(concurrency is not None or (len(probe.observations)==config['turns'] and
        admission['peak_session_owners']==12 and hit_delta>=12*(config['turns']-1) and
        (admission['peak_idle_protected_records'] or 0)>0))
    passed=(failure is None and clean(after) and len(probe.records)==expected_requests and
        all(row['passed'] for row in probe.releases) and not probe.sessions and
        errors['policy_failures_max']==0 and errors['query_overflow_packets_max']==0 and not capture_errors and
        admission['observed'] and admission['capacity_violations']==0 and not admission['uncached_owner_observed'] and
        admission['peak_session_owners']<=expected_owners and running['peak'] is not None and running['peak']<=8 and
        native_delta==accepted+control_accepted and qualified)
    result=dict(passed=passed,label=label,task_concurrency=concurrency,model_concurrency_limit=8,
        started_at=start,finished_at=finish,wall_seconds=elapsed,accepted_output_tokens=accepted,
        accepted_tokens_per_wall_second=accepted/elapsed,release_control_tokens=control_accepted,
        native_generation_delta=native_delta,native_client_tokens_equal=native_delta==accepted+control_accepted,
        completed_requests=len(probe.records),expected_requests=expected_requests,session_hit_delta=hit_delta,
        model_running=running,model_waiting=waiting,admission=admission,qualification_passed=qualified,
        native_errors=errors,capture_errors=capture_errors,errors=probe.errors,cleanup_passed=clean(after),
        restore_events=len(native_events['async_restore.jsonl']),fixture_sha256=digest(config),
        scope='Finite synthetic natural-generation workload; no task scoring. Wall includes prefill, tools and normal session release. '
              'Input prompts/observations match between arms; generated continuations may differ with batching. No retries.')
    save(output/'summary.json',result)
    if not passed:raise RuntimeError('Queue probe failed; inspect '+str(output/'summary.json')) from failure
    print(json.dumps(result,indent=2),flush=True)
    return result


async def run(args):
    config=read(args.fixture)
    require(config.get('schema')==1 and config.get('items')==24 and config.get('turns')==3,
            'Expected the fixed 24-item three-turn fixture')
    require(not args.output.exists(),'Use a fresh experiment directory')
    args.output.mkdir(parents=True)
    save(args.output/'fixture.json',config)
    save(args.output/'transport-policy.json',dict(automatic_retries=0,max_keepalive_connections=0,
        trust_env=False,model_server_owned_by='external controller'))
    transport=httpx.AsyncHTTPTransport(retries=0,limits=httpx.Limits(max_connections=32,max_keepalive_connections=0))
    async with httpx.AsyncClient(base_url=args.base,trust_env=False,timeout=args.timeout,
            transport=transport,headers={'Connection':'close'}) as client:
        qualification=await run_phase(client,args.model,config,args.capture,args.output/'qualification12','qualification12')
        results=[]
        for concurrency in args.order:
            results.append(await run_phase(client,args.model,config,args.capture,
                args.output/f'queue{concurrency}',f'queue{concurrency}',concurrency))
    by_queue={row['task_concurrency']:row for row in results}
    save(args.output/'summary.json',dict(passed=True,qualification=qualification,arms=results,
        throughput_change_percent=100*(by_queue[12]['accepted_tokens_per_wall_second']/by_queue[8]['accepted_tokens_per_wall_second']-1),
        scope='One fixed-workload natural-generation A/B. Qualification excluded. No task-quality inference.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--profile',type=Path,required=True);prep.add_argument('--output',type=Path,required=True)
    execute=sub.add_parser('run');execute.add_argument('--base',default='http://127.0.0.1:18882')
    execute.add_argument('--model',required=True);execute.add_argument('--fixture',type=Path,required=True)
    execute.add_argument('--capture',type=Path,required=True);execute.add_argument('--output',type=Path,required=True)
    execute.add_argument('--timeout',type=float,default=600)
    execute.add_argument('--order',type=int,nargs=2,default=[8,12],choices=[8,12])
    args=parser.parse_args()
    if args.action=='prepare':prepare(args.profile,args.output)
    else:
        require(set(args.order)=={8,12},'Run each queue size exactly once')
        asyncio.run(run(args))


if __name__=='__main__':main()
