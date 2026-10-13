"""Matched direct-engine SSE workload for native/KVMem integration overhead.

Both sides must use the same model, speculation, scheduler and input parameters.
This bypasses the elastic HTTP facade; it does not qualify session restoration.
Native metrics independently confirm the common fully-active decode window.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import uuid

import httpx

import qualify_decode as q
from qualify_elastic_long_output import metrics, corroborate


async def run(args):
    prompt = json.loads(args.input.read_text())['input_ids']
    if not prompt or any(type(t) is not int or t < 0 for t in prompt):
        raise ValueError('Expected frozen unsigned token IDs')
    args.output.mkdir(parents=True, exist_ok=False)
    origin, samples, traces = time.monotonic(), [], []
    done, start = asyncio.Event(), asyncio.Event()
    async with httpx.AsyncClient(base_url=args.backend, trust_env=False,
            timeout=args.timeout, limits=httpx.Limits(max_connections=args.concurrency+4)) as client:
        response = await client.get('/metrics'); response.raise_for_status()
        before = metrics(response.text)
        if before.get('vllm:num_requests_running') or before.get('vllm:num_requests_waiting'):
            raise RuntimeError('Require an isolated backend')

        async def sample():
            last_print = -20
            while not done.is_set():
                row = {'read_started_at': time.monotonic()-origin}
                try:
                    response = await client.get('/metrics'); response.raise_for_status()
                    row['metrics'] = metrics(response.text)
                except httpx.HTTPError as error:
                    row['error'] = type(error).__name__
                row['at'] = time.monotonic()-origin; samples.append(row)
                if row['at']-last_print >= 20:
                    progress = dict(at=round(row['at'], 1), output_tokens=[t['observed_tokens'] for t in traces],
                                    running=row.get('metrics', {}).get('vllm:num_requests_running'))
                    (args.output/'progress.json').write_text(json.dumps(progress)+'\n')
                    print(json.dumps(progress), flush=True); last_print=row['at']
                try:
                    await asyncio.wait_for(done.wait(), .25)
                except TimeoutError:
                    pass

        async def request(index):
            trace = dict(client_index=index, outcome='failure', timeline=[], observed_tokens=0,
                         saw_token_ids=False, _token_digest=hashlib.sha256())
            traces.append(trace)
            await start.wait()
            trace['started_at']=time.monotonic()-origin
            payload = dict(model=args.model, prompt=prompt, max_tokens=args.output_tokens,
                temperature=0, seed=104729, ignore_eos=True, stream=True, return_token_ids=True,
                stream_options={'include_usage':True}, cache_salt=uuid.uuid4().hex)
            try:
                async with client.stream('POST','/v1/completions',json=payload) as response:
                    response.raise_for_status()
                    async for event in q.stream_events(response):
                        if event is None:
                            break
                        q.consume_event(trace,event,time.monotonic()-origin)
                        if 'metrics' in event:
                            trace['native_metrics']=event['metrics']
                q.finish_trace(trace,args.output_tokens)
                if trace['prompt_tokens'] != len(prompt):
                    raise ValueError('Prompt token count mismatch')
                trace['outcome']='success'
            except Exception as error:
                trace['error']=type(error).__name__
            finally:
                trace['ended_at']=time.monotonic()-origin
                trace['elapsed_seconds']=trace['ended_at']-trace['started_at']
                trace['token_ids_sha256']=trace.pop('_token_digest').hexdigest()

        poller=asyncio.create_task(sample())
        tasks=[asyncio.create_task(request(i)) for i in range(args.concurrency)]
        started=time.monotonic();start.set()
        try:
            await asyncio.gather(*tasks)
        finally:
            done.set();await poller
        wall=time.monotonic()-started
        await asyncio.sleep(1.3)
        response=await client.get('/metrics');response.raise_for_status();final=metrics(response.text)
    windows=q.batch_windows(SimpleNamespace(trim_tokens=128,min_window_seconds=2,
        min_events_per_request=4,capture=None,allow_queued=False),traces,[],args.concurrency,wall)
    confirmation=corroborate(samples,windows['steady_decode_window'],args.concurrency)
    preemptions=final.get('vllm:num_preemptions_total',0)-before.get('vllm:num_preemptions_total',0)
    passed=(windows['status']=='passed' and confirmation['status']=='passed' and preemptions==0
        and final.get('vllm:num_requests_running')==final.get('vllm:num_requests_waiting')==0)
    report={**windows,'native_confirmation':confirmation,'status':'passed' if passed else 'failed',
        'configuration':dict(label=args.label,concurrency=args.concurrency,input_tokens=len(prompt),
            output_tokens=args.output_tokens,seed=104729,temperature=0,unique_cache_salts=True,
            session_checkpointing=False,scope='direct_engine_SSE',
            input_token_ids_sha256=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()),
        'preemptions':preemptions,'requests':traces,'samples':samples,
        'native_counter_deltas':{k:final[k]-v for k,v in before.items() if k.endswith('_total') and k in final}}
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    summary={k:v for k,v in report.items() if k not in ('requests','samples')}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',default='http://127.0.0.1:18381')
    parser.add_argument('--model',default='kvmem-meank')
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--label',required=True)
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--output-tokens',type=int,default=4096)
    parser.add_argument('--timeout',type=float,default=1200)
    asyncio.run(run(parser.parse_args()))
