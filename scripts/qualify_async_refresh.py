"""Measure public decode progress while another request refreshes its KV view.

The same workload runs against the synchronous control and asynchronous variant.
Counts use streamed token IDs, checked against final usage. Times are host
perf_counter values shared with worker refresh traces on the same machine.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import uuid

import httpx
import numpy as np


async def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text())
    fixture=json.loads(a.input.read_text())
    prompt=fixture['input_ids'];query=fixture['query_span']
    peer_prompt=prompt[:7936]+prompt[-256:]
    started=[asyncio.Event() for _ in range(a.peers)]
    timeout=httpx.Timeout(900,connect=10)
    async with httpx.AsyncClient(timeout=timeout,trust_env=False) as client:
        async def request(index,tokens,q,count):
            session='refresh-test-'+uuid.uuid4().hex
            payload=dict(model=cfg['served_model_name'],session_id=session,cache_salt=session,
                         prompt=tokens,kvmem_query_span=q,kvmem_reselect='force',
                         max_tokens=count,temperature=0,ignore_eos=True,return_token_ids=True,
                         stream=True,stream_options={'include_usage':True})
            trace=dict(index=index,begin=time.perf_counter(),timeline=[],input_tokens=len(tokens))
            ids=[]
            async with client.stream('POST',f"http://127.0.0.1:{cfg['port']}/v1/completions",json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith('data: '):continue
                    data=line[6:]
                    if data=='[DONE]':break
                    event=json.loads(data)
                    if event.get('error'):raise RuntimeError(event['error'])
                    trace['response_id']=event['id']
                    if event.get('usage'):trace['usage']=event['usage']
                    for choice in event.get('choices',[]):
                        chunk=choice.get('token_ids') or []
                        if chunk:
                            ids.extend(chunk)
                            trace['timeline'].append([time.perf_counter(),len(chunk)])
                            if index<a.peers:started[index].set()
            trace['end']=time.perf_counter();trace['output_tokens']=len(ids)
            assert len(ids)==count==trace['usage']['completion_tokens'],trace
            trace['output_sha256']=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
            (a.output/f'{index}.json').write_text(json.dumps(trace))
            return trace
        tasks=[asyncio.create_task(request(i,peer_prompt,[7936,8183],a.output_tokens)) for i in range(a.peers)]
        waiters=asyncio.gather(*(event.wait() for event in started))
        # Surface an early peer error rather than waiting indefinitely for TTFT.
        done,_=await asyncio.wait([waiters,*tasks],return_when=asyncio.FIRST_COMPLETED,timeout=180)
        if waiters not in done:
            for task in done:task.result()
            raise RuntimeError('Peers did not enter decode together')
        await asyncio.sleep(2)
        target=await request(a.peers,prompt,query,129)
        peers=await asyncio.gather(*tasks)
    rows=[json.loads(x) for x in (a.capture/'bounded_refresh.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['request_id'].startswith(target['response_id']+'-')]
    assert rows,'No refresh was exercised'
    intervals=[]
    for row in rows:
        begin,end=row['refresh_started_at'],row['refresh_finished_at']
        measured=[]
        for trace in peers:
            events=trace['timeline'];inside=[(t,n) for t,n in events if begin<=t<=end]
            # Include gaps crossing the refresh interval's endpoints.
            gaps=[v[0]-u[0] for u,v in zip(events,events[1:]) if v[0]>=begin and u[0]<=end]
            measured.append(dict(index=trace['index'],tokens_during_refresh=sum(n for _,n in inside),
                events_during_refresh=len(inside),max_gap_seconds=max(gaps,default=0),
                p95_gap_seconds=float(np.percentile(gaps,95)) if gaps else 0,
                still_decoding_at_end=events[-1][0]>end))
        intervals.append(dict(selection=row['selection'],begin=begin,end=end,seconds=end-begin,
            guided_query_tokens=row.get('guided_query_tokens'),asynchronous=row.get('asynchronous',False),
            peers=measured,aggregate_peer_tokens=sum(x['tokens_during_refresh'] for x in measured)))
    report=dict(concurrency=a.peers+1,peer_input_tokens=len(peer_prompt),
        peer_output_tokens=a.output_tokens,target_input_tokens=len(prompt),intervals=intervals,
        target_ttft_seconds=target['timeline'][0][0]-target['begin'])
    all_traces=peers+[target]
    report['wall_seconds']=max(t['end'] for t in all_traces)-min(t['begin'] for t in all_traces)
    report['end_to_end_output_tokens_per_second']=sum(t['output_tokens'] for t in all_traces)/report['wall_seconds']
    if cfg['kvmem'].get('async_refresh'):
        assert all(x['tokens_during_refresh']>0 and x['still_decoding_at_end']
                   for interval in intervals for x in interval['peers']),report
    report['status']='passed'
    (a.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--peers',type=int,default=7)
    p.add_argument('--output-tokens',type=int,default=8192)
    asyncio.run(main(p.parse_args()))
