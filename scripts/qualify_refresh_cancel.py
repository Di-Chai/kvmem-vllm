"""Abort private Query / background assembly; require fenced native page reuse."""
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import httpx


async def main(a):
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());fixture=json.loads(a.input.read_text())
    url=f"http://127.0.0.1:{cfg['port']}/v1/completions"
    report=[]
    async with httpx.AsyncClient(timeout=300,trust_env=False) as client:
        for stage in ('query','assembly_started'):
            start=time.perf_counter();session='cancel-refresh-'+uuid.uuid4().hex
            payload=dict(model=cfg['served_model_name'],session_id=session,cache_salt=session,
                prompt=fixture['input_ids'],kvmem_query_span=fixture['query_span'],kvmem_reselect='force',
                max_tokens=4096,temperature=0,ignore_eos=True,stream=True,return_token_ids=True)
            async def consume():
                async with client.stream('POST',url,json=payload) as response:
                    response.raise_for_status()
                    async for _ in response.aiter_lines():pass
            task=asyncio.create_task(consume())
            event=None
            while time.perf_counter()-start<180:
                path=a.capture/'async_refresh.jsonl'
                if path.exists():
                    lines=path.read_text().splitlines()
                    # A worker can be midway through writing the last line.
                    for line in lines:
                        try:r=json.loads(line)
                        except json.JSONDecodeError:continue
                        if r['at']>=start and r['stage']==stage:event=r;break
                if event:break
                if task.done():task.result();raise RuntimeError('Request finished before cancellation stage')
                await asyncio.sleep(.01)
            if event is None:raise TimeoutError(stage)
            at=time.perf_counter();task.cancel()
            try:await task
            except asyncio.CancelledError:pass
            ready=None
            while time.perf_counter()-at<60:
                rows=[]
                for line in (a.capture/'async_refresh.jsonl').read_text().splitlines():
                    try:rows.append(json.loads(line))
                    except json.JSONDecodeError:pass
                ready=next((r for r in rows if r['request_id']==event['request_id'] and r['stage']=='cancelled'),None)
                resources=json.loads((a.capture/'scheduler_resources.json').read_text())
                if ready and not resources['requests'] and not resources['owned_blocks'] and not resources['refresh_pending']:break
                await asyncio.sleep(.02)
            else:raise AssertionError({'stage':stage,'cancelled':ready,'resources':resources})
            # Reuse released native request/state slots and pages immediately.
            follow=await client.post(url,json=dict(model=cfg['served_model_name'],
                prompt=[1,2,3],max_tokens=32,ignore_eos=True,temperature=0,return_token_ids=True))
            follow.raise_for_status();body=follow.json()
            assert body['usage']['completion_tokens']==len(body['choices'][0]['token_ids'])==32
            report.append(dict(stage=stage,request_id=event['request_id'],cancel_at=at,
                fenced_at=ready['at'],cleanup_seconds=time.perf_counter()-at,
                owned_blocks_after_cancel=resources['owned_blocks'],followup_tokens=32))
            await asyncio.sleep(1)
    result=dict(status='passed',results=report)
    (a.output/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','input','output'):p.add_argument('--'+name,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
