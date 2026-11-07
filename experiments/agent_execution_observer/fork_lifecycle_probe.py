"""Finite observer disconnect, parent completion and parent disconnect probes."""
import argparse,asyncio,json,time
from pathlib import Path
from uuid import uuid4
import httpx
from qualify_systemone import capture_snapshot,release_evidence


def write(path,value):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(path)


def journal(capture):
    p=capture/'systemone_fork.jsonl'
    rows=[]
    for line in p.read_text().splitlines() if p.exists() else []:
        try:rows.append(json.loads(line))
        except ValueError:pass
    return rows


async def until(predicate,timeout=30):
    deadline=time.monotonic()+timeout
    while True:
        result=predicate()
        if result:return result
        if time.monotonic()>deadline:raise TimeoutError('Expected lifecycle transition missing')
        await asyncio.sleep(.02)


async def run(args):
    args.output.mkdir(exist_ok=False,parents=True)
    result=dict(status='running',cases=[])
    save=lambda:write(args.output/'results.json',result)
    save()
    try:
        async with httpx.AsyncClient(base_url=args.base_url,timeout=180,trust_env=False) as client:
            model=(await client.get('/v1/models')).json()['data'][0]['id']
            response=await client.post('/tokenize',json=dict(model=model,messages=[dict(role='user',content='Record: the parcel was damaged during shipping.\n'*500+'The parcel arrived broken. I want a refund. Write a detailed numbered list of practical steps for handling this return. Continue adding useful steps.')],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}))
            response.raise_for_status();tokens=response.json()['tokens']
            for kind in ('observer_disconnect','parent_completion','parent_disconnect'):
                before=capture_snapshot(args.capture)
                session='fork-lifecycle-'+uuid4().hex
                row=dict(kind=kind,session=session,events=[],token_ids=[],started=time.perf_counter());result['cases'].append(row);save()
                max_tokens=64 if kind=='parent_completion' else 256
                request=dict(model=model,prompt=tokens,session_id=session,cache_salt=session,temperature=0,max_tokens=max_tokens,ignore_eos=True,seed=42,return_token_ids=True,stream=True)
                write(args.output/(kind+'-parent-request.json'),request)
                async def consume():
                    async with client.stream('POST','/v1/completions',json=request) as r:
                        r.raise_for_status()
                        with (args.output/(kind+'-sse.jsonl')).open('x') as out:
                            async for line in r.aiter_lines():
                                if not line.startswith('data: '):continue
                                event=dict(at=time.perf_counter(),line=line);out.write(json.dumps(event)+'\n');out.flush()
                                if line[6:]=='[DONE]':break
                                data=json.loads(line[6:]);assert 'error' not in data,data
                                row['request_id']=data.get('id',row.get('request_id'))
                                ids=[t for c in data.get('choices',[]) for t in c.get('token_ids',[]) or []]
                                if ids:
                                    row['token_ids'].extend(ids);row['events'].append(dict(at=event['at'],count=len(ids)))
                foreground=asyncio.create_task(consume())
                decision=None
                try:
                    await until(lambda:len(row['token_ids'])>=32,120)
                    question=dict(type='noul',instructions='Record token for segmentation: parcel. '*18+'Was the parcel damaged?')
                    payload=dict(model=model,state='',cache_salt=session,kvmem_session=dict(session_id=session,mode='fork_cobatch'),questions={str(i):question for i in range(8)})
                    write(args.output/(kind+'-observer-request.json'),payload)
                    decision=asyncio.create_task(client.post('/v1/systemone',json=payload))
                    def started():
                        forks=[e for e in journal(args.capture) if e['kind']=='fork' and e['at']>row['started'] and e['parent'].startswith(row['request_id'])]
                        if not forks:return None
                        op=forks[-1]['operation']
                        return forks[-1] if sum(e['kind']=='cobatch' and e['operation']==op for e in journal(args.capture))>=2 else None
                    fork=await until(started);row['fork']=fork;op=fork['operation'];row['action_at']=time.perf_counter();save()
                    if kind=='observer_disconnect':
                        decision.cancel();await asyncio.gather(decision,return_exceptions=True)
                    elif kind=='parent_disconnect':
                        foreground.cancel();await asyncio.gather(foreground,return_exceptions=True)
                    if kind!='observer_disconnect':
                        reply=await decision;row['observer_response']=dict(status=reply.status_code,body=reply.json())
                        assert reply.status_code==409,row['observer_response']
                    finish=await until(lambda:next((e for e in journal(args.capture) if e['kind']=='finish' and e['operation']==op),None))
                    row['finish']=finish
                    assert finish['error'] is not None,finish
                    if kind=='observer_disconnect':
                        assert finish['error']=='Decision cancelled',finish
                        at_cancel=len(row['token_ids']);await until(lambda:len(row['token_ids'])>=at_cancel+8)
                        # Reuse the same private slot after cancellation, and
                        # reject an outdated logical boundary without poisoning it.
                        short={**payload,'questions':{'check':dict(type='noul',instructions='Was the parcel damaged?')}}
                        stale={**short,'kvmem_session':{**short['kvmem_session'],'history_end':1}}
                        rejected=await client.post('/v1/systemone',json=stale);assert rejected.status_code==409
                        row['stale_history_status']=rejected.status_code
                        again=await client.post('/v1/systemone',json=short);row['retry_response']=dict(status=again.status_code,body=again.json());assert again.status_code==200,row['retry_response']
                        row['foreground_continued_after_cancel']=True
                    if kind!='parent_disconnect':
                        await foreground;assert len(row['token_ids'])==max_tokens
                finally:
                    if decision is not None and not decision.done():decision.cancel();await asyncio.gather(decision,return_exceptions=True)
                    if not foreground.done():
                        if kind=='parent_disconnect':foreground.cancel()
                        await asyncio.gather(foreground,return_exceptions=True)
                    released_at=time.monotonic()
                    release=await client.post('/v1/completions',json=dict(model=model,prompt=[tokens[0]],session_id=session,cache_salt=session,max_tokens=1,temperature=0,kv_transfer_params={'kvmem_release_session':True}))
                    release.raise_for_status()
                def released():
                    snap=capture_snapshot(args.capture);pool=json.loads((args.capture/'systemone_fork_pool.json').read_text());proof=release_evidence(snap,before,0,released_at);state=snap['scheduler_resources']
                    if state.get('free_blocks',-1)+pool['reserved_blocks']==state.get('total_blocks',0)-1:proof['unsettled']=[v for v in proof['unsettled'] if v!='scheduler.free_blocks']
                    if pool['leases'] or pool['pinned_blocks'] or pool['acquired']!=pool['released']:proof['unsettled'].append('observer.live_leases')
                    proof['passed']=not proof['unsettled'];row['release_proof']=proof;row['pool']=pool
                    return snap if proof['passed'] else None
                snapshot=await until(released,30);write(args.output/(kind+'-resources-after.json'),snapshot)
                row['status']='passed';save()
            result['status']='passed'
    except BaseException as error:
        result.update(status='failed',error=repr(error));raise
    finally:save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:18995');p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    asyncio.run(run(p.parse_args()))
