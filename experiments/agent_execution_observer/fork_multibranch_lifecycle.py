"""Cancel one lane without stopping peers, then qualify foreground retirement."""
import argparse,asyncio,json,time,copy
from pathlib import Path
from uuid import uuid4
import httpx
from qualify_systemone import capture_snapshot,release_evidence


def write(p,value):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(value,indent=2));t.replace(p)
def journal(capture):
 p=capture/'systemone_fork.jsonl'
 return [json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
async def until(predicate,timeout=60):
 end=time.monotonic()+timeout
 while True:
  value=predicate()
  if value:return value
  if time.monotonic()>end:raise TimeoutError('Missing lifecycle transition')
  await asyncio.sleep(.02)

async def run(args):
 args.output.mkdir(exist_ok=False,parents=True);result=dict(status='running',cases=[])
 save=lambda:write(args.output/'results.json',result);save()
 try:
  async with httpx.AsyncClient(base_url='http://127.0.0.1:18995',timeout=180,trust_env=False) as client:
   model=(await client.get('/v1/models')).json()['data'][0]['id']
   response=await client.post('/tokenize',json=dict(model=model,messages=[dict(role='user',content='Record: the parcel was damaged during shipping.\n'*500+'The parcel arrived broken. I want a refund. Write a detailed numbered list of practical steps for handling this return. Continue adding useful steps.')],add_generation_prompt=True,chat_template_kwargs={'enable_thinking':False}));response.raise_for_status();tokens=response.json()['tokens']
   for size,kind in [(2,'observer_disconnect'),(4,'observer_disconnect'),(4,'parent_completion'),(4,'parent_disconnect')]:
    before=capture_snapshot(args.capture);session='multi-lifecycle-'+uuid4().hex;label=f'{size}-{kind}'
    row=dict(kind=kind,branches=size,session=session,events=[],token_ids=[],started=time.perf_counter());result['cases'].append(row);save()
    maximum=64 if kind=='parent_completion' else 256
    request=dict(model=model,prompt=tokens,session_id=session,cache_salt=session,temperature=0,max_tokens=maximum,ignore_eos=True,seed=42,return_token_ids=True,stream=True)
    write(args.output/(label+'-parent.json'),request)
    async def consume():
     async with client.stream('POST','/v1/completions',json=request) as reply:
      reply.raise_for_status()
      with (args.output/(label+'-sse.jsonl')).open('x') as log:
       async for line in reply.aiter_lines():
        if not line.startswith('data: '):continue
        now=time.perf_counter();log.write(json.dumps(dict(at=now,line=line))+'\n');log.flush()
        if line[6:]=='[DONE]':break
        body=json.loads(line[6:]);assert 'error' not in body;row['request_id']=body.get('id',row.get('request_id'))
        ids=[t for c in body.get('choices',[]) for t in c.get('token_ids',[]) or []]
        if ids:row['token_ids'].extend(ids);row['events'].append(dict(at=now,count=len(ids)))
    foreground=asyncio.create_task(consume());decisions=[]
    try:
     await until(lambda:len(row['token_ids'])>=32,180)
     question=dict(type='noul',instructions='Record token for segmentation: parcel. '*18+'Was the parcel damaged?')
     payload=dict(model=model,state='',cache_salt=session,kvmem_session=dict(session_id=session,mode='fork_cobatch',fork_group=dict(id=session,size=size)),questions={str(i):question for i in range(8)})
     write(args.output/(label+'-observers.json'),payload)
     decisions=[asyncio.create_task(client.post('/v1/systemone',json=copy.deepcopy(payload))) for _ in range(size)]
     def started():
      log=journal(args.capture);forks=[e for e in log if e['kind']=='fork' and e['at']>row['started'] and e['parent'].startswith(row['request_id'])]
      if len(forks)!=size:return None
      return forks if all(sum(e['kind']=='cobatch' and e['operation']==f['operation'] for e in log)>=2 for f in forks) else None
     forks=await until(started);row['forks']=forks;assert len({f['history'] for f in forks})==1;row['action_at']=time.perf_counter();save()
     if kind=='observer_disconnect':
      decisions[0].cancel();await asyncio.gather(decisions[0],return_exceptions=True)
      def cancelled():
       return next((e for e in journal(args.capture) if e['kind']=='finish' and e['operation'] in {f['operation'] for f in forks} and e['error']=='Decision cancelled'),None)
      row['cancelled_branch']=await until(cancelled)
      after=len(row['token_ids']);await until(lambda:len(row['token_ids'])>=after+4)
      # Reuse a released lane while the surviving long observations are active.
      retry={**payload,'kvmem_session':dict(session_id=session,mode='fork_cobatch'),'questions':{'short':dict(type='noul',instructions='Was the parcel damaged?')}}
      reply=await client.post('/v1/systemone',json=retry);row['retry']=dict(status=reply.status_code,body=reply.json());assert reply.status_code==200,row['retry']
      replies=await asyncio.gather(*decisions[1:]);row['survivors']=[dict(status=x.status_code,body=x.json()) for x in replies]
      assert all(x.status_code==200 for x in replies),row['survivors']
      assert all(x.json()['kvmem']['history_end']==forks[0]['history'] for x in replies)
      row['foreground_continued']=True
     else:
      if kind=='parent_disconnect':foreground.cancel();await asyncio.gather(foreground,return_exceptions=True)
      replies=await asyncio.gather(*decisions);row['responses']=[dict(status=x.status_code,body=x.json()) for x in replies]
      assert all(x.status_code==409 for x in replies),row['responses']
     if kind!='parent_disconnect':await foreground;assert len(row['token_ids'])==maximum
    finally:
     for decision in decisions:
      if not decision.done():decision.cancel()
     await asyncio.gather(*decisions,return_exceptions=True)
     if not foreground.done():foreground.cancel()
     await asyncio.gather(foreground,return_exceptions=True)
     released_at=time.monotonic()
     released=await client.post('/v1/completions',json=dict(model=model,prompt=[tokens[0]],session_id=session,cache_salt=session,max_tokens=1,temperature=0,kv_transfer_params={'kvmem_release_session':True}));released.raise_for_status()
    def release_proof():
     snap=capture_snapshot(args.capture);pool=json.loads((args.capture/'systemone_fork_pool.json').read_text());proof=release_evidence(snap,before,0,released_at);state=snap['scheduler_resources']
     if state.get('free_blocks',-1)+pool['reserved_blocks']==state.get('total_blocks',0)-1:proof['unsettled']=[s for s in proof['unsettled'] if s!='scheduler.free_blocks']
     if pool['leases'] or pool['pinned_blocks'] or pool['acquired']!=pool['released']:proof['unsettled'].append('observer.live_leases')
     proof['passed']=not proof['unsettled'];row['release_proof']=proof;row['pool']=pool
     return snap if proof['passed'] else None
    snap=await until(release_proof);write(args.output/(label+'-resources.json'),snap);row['status']='passed';save()
   result['status']='passed'
 except BaseException as error:result.update(status='failed',error=repr(error));raise
 finally:save()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(run(p.parse_args()))
