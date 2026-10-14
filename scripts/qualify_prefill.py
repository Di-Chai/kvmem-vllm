#!/usr/bin/env python3
"""Prefill service throughput with one output token and native session maintenance."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import statistics
import time
import urllib.request
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--api-base',default='http://127.0.0.1:18381/v1')
    p.add_argument('--model',default='kvmem-meank')
    p.add_argument('--prompt-token-file',type=Path,required=True)
    p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--concurrency',type=int,nargs='+',default=[1,8])
    p.add_argument('--rounds',type=int,default=2)
    p.add_argument('--timeout',type=float,default=600)
    a=p.parse_args()
    if a.rounds<1 or any(not 1<=c<=128 for c in a.concurrency):raise ValueError('Invalid prefill concurrency/rounds')
    a.output.mkdir(parents=True,exist_ok=False)
    fixture=json.loads(a.prompt_token_file.read_text());tokens=fixture['input_ids']
    digest=hashlib.sha256(json.dumps(tokens,separators=(',',':')).encode()).hexdigest()
    (a.output/'input.json').write_text(json.dumps({'input_ids':tokens,'sha256':digest}))
    http=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results=[]
    for c in a.concurrency:
        for number in range(a.rounds):
            start=time.monotonic()
            def run(index):
                payload={'model':a.model,'prompt':tokens,'max_tokens':1,'ignore_eos':True,
                    'temperature':0,'seed':104729,'stream':True,'stream_options':{'include_usage':True},
                    'return_token_ids':True,'session_id':'prefill-'+uuid.uuid4().hex}
                began=time.monotonic();first=None;usage=None;outputs=[]
                req=urllib.request.Request(a.api_base+'/completions',data=json.dumps(payload).encode(),
                    headers={'Content-Type':'application/json'})
                with http.open(req,timeout=a.timeout) as response:
                    for raw in response:
                        if not raw.startswith(b'data:'):continue
                        data=raw[5:].strip()
                        if data==b'[DONE]':break
                        event=json.loads(data)
                        if event.get('error'):
                            raise RuntimeError(f"Prefill stream failed: {event['error']}")
                        if event.get('usage'):usage=event['usage']
                        for choice in event.get('choices',[]):
                            ids=choice.get('token_ids') or []
                            if ids:
                                if first is None:first=time.monotonic()
                                outputs.extend(ids)
                ended=time.monotonic()
                assert first is not None and len(outputs)==1
                assert usage['prompt_tokens']==len(tokens) and usage['completion_tokens']==1
                return {'client_index':index,'started':began-start,'first_token':first-start,
                    'ttft_seconds':first-began,'request_seconds':ended-began,'input_tokens':len(tokens)}
            with ThreadPoolExecutor(max_workers=c) as pool:clients=list(pool.map(run,range(c)))
            elapsed=time.monotonic()-start
            first_interval=max(r['first_token'] for r in clients)-min(r['started'] for r in clients)
            deadline=time.monotonic()+30
            while True:
                native=json.loads((a.capture/'scheduler_resources.json').read_text())
                if native['requests']==native['owned_blocks']==0:break
                if time.monotonic()>deadline:raise AssertionError('Prefill native owners did not drain')
                time.sleep(.05)
            row={'status':'passed','concurrency':c,'round':number,'input_tokens_per_request':len(tokens),
                'output_tokens_per_request':1,'aggregate_input_tokens':c*len(tokens),
                'seconds_to_all_first_tokens':first_interval,'effective_input_tokens_per_second':c*len(tokens)/first_interval,
                'wall_seconds':elapsed,'whole_input_tokens_per_second':c*len(tokens)/elapsed,
                'ttft_mean_seconds':statistics.mean(r['ttft_seconds'] for r in clients),
                'ttft_max_seconds':max(r['ttft_seconds'] for r in clients),'clients':clients,
                'final_native_requests':native['requests'],'final_owned_blocks':native['owned_blocks']}
            results.append(row)
            print(json.dumps({k:v for k,v in row.items() if k!='clients'}),flush=True)
    report={'schema':'kvmem.prefill_service_qualification.v1','input_token_ids_sha256':digest,
        'prompt_sessions':True,'batches':results,
        'scope':'Input-token service throughput with one generated token per request. Includes queueing, CPU history/checkpoints and MTP preparation; not isolated GPU kernel throughput. No sustained decode is interleaved.'}
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
