"""Real-server correctness checks for revocable views and cancellation fences.

These checks are excluded from performance measurements. Every observation uses
an exact native request ID; a successful HTTP continuation alone is insufficient.
"""
import asyncio
import json
import re
import time
import uuid


def read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}


def rows(path):
    if not path.exists():return []
    result=[]
    lines=path.read_text().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1 or line.endswith('\n'):raise
    return result


def native_id(response_id, records):
    pattern=re.compile(re.escape(response_id)+r'-0-[0-9a-f]{8}')
    matched={r['request_id'] for r in records if pattern.fullmatch(r['request_id'])}
    if len(matched)!=1:raise AssertionError('Missing or ambiguous exact native request ID: '+response_id)
    return matched.pop()


async def wait_until(predicate, *, timeout=120.):
    started=time.monotonic()
    while time.monotonic()-started<timeout:
        value=predicate()
        if value:return value
        await asyncio.sleep(.005)
    raise TimeoutError('Lifecycle evidence did not become ready')


async def quiescent(capture):
    def settled():
        state=read(capture/'scheduler_resources.json')
        if (not state or state['requests'] or state['refresh_pending'] or state['connector_pending']
                or state['owned_blocks'] or state['reservations']):return None
        cpu=state.get('cpu_history_admission',{})
        if cpu.get('active') or cpu.get('retiring'):return None
        return state
    state=await wait_until(settled)
    validate_quiescent(state)
    return state


def validate_quiescent(state):
    if (state['requests'] or state['refresh_pending'] or state['connector_pending']
            or state['owned_blocks'] or state['reservations']):
        raise AssertionError('Request/page/fence ownership remains after quiescence')
    idle=state.get('active_residency',{}).get('idle_blocks',0)
    if state['free_blocks']+idle!=state['total_blocks']-1:
        raise AssertionError('Quiescent native pages are neither free nor owned by an idle view')
    if idle>state.get('active_residency',{}).get('max_blocks',0):
        raise AssertionError('Idle views exceeded their native-page budget')
    cpu=state['cpu_history_admission']
    if (cpu['active'] or cpu['retiring'] or cpu['waiting'] or cpu['available_bytes']<0
            or cpu['last_observed_archive_bytes']>cpu['reserved_bytes']
            or cpu['reserved_bytes']+cpu['progress_reserve_bytes']+cpu['available_bytes']!=cpu['capacity_bytes']):
        raise AssertionError('CPU history accounting exceeded admitted credits')


def exact_restore(capture, response_id, history, *, resident, cancelled):
    restores=rows(capture/'async_restore.jsonl')
    rid=native_id(response_id,restores)
    values=[r for r in restores if r['request_id']==rid]
    if len(values)!=1:raise AssertionError('Duplicate restore proof')
    row=values[0]
    if (row['history_tokens']!=history or row['cancelled'] is not cancelled
            or row['model_bytes']<=0 or row['finished']<row['started']):
        raise AssertionError('Restore history/cancellation differs from the expected checkpoint')
    if row.get('resident_active_tokens',0)!=(row['active_tokens'] if resident else 0):
        raise AssertionError('Test did not exercise its required hot/cold path')
    if resident and row['uploaded_bytes']!=0:raise AssertionError('Resident KV was uploaded again')
    if not resident and row['uploaded_bytes']<=0:raise AssertionError('Cold restore did not transfer historical KV')
    proofs={}
    for name in ('verified_state_restores.jsonl','verified_restores.jsonl'):
        found=[r for r in rows(capture/name) if r['request_id']==rid]
        if len(found)!=1:raise AssertionError('Missing complete byte oracle: '+name)
        proofs[name]=found[0]
    state,full=proofs.values()
    geometry=read(capture/'cache-geometry.json')
    archive_width=2 if geometry.get('archive_codec')=='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1' else 3
    attention=[n for g in geometry['groups'] if not g['recurrent'] for n in g['layers']]
    recurrent=[n for g in geometry['groups'] if g['recurrent'] for n in g['layers']]
    mtp=[n for n in attention if n.startswith('mtp.') or '.mtp.' in n]
    if (not attention or not recurrent or not mtp or state['pressure_restore'] is not False
            or state['history_tokens']!=history or full['history_tokens']!=history or state['checkpoint_kind']!='M'
            or state['recurrent_byte_equal'] is not True or full['byte_equal'] is not True
            or state['working_tokens']!=row['active_tokens'] or full['working_tokens']!=row['active_tokens']
            or state['recurrent_tensors']!=2*len(recurrent) or full['recurrent_tensors']!=2*len(recurrent)
            or full['working_attention_layers']!=len(attention) or full['mtp_attention_layers']!=len(mtp)
            or full['draft_layers']!=0
            or full['target_kv_bytes']!=row['active_tokens']*geometry['cpu_archive_bytes_per_token']//archive_width*2):
        raise AssertionError('Incomplete native target/MTP/recurrent oracle coverage')
    return row


def validate_report(capture, report):
    """Recheck saved lifecycle evidence against exact worker/retirement logs."""
    if report['passed'] is not True:raise AssertionError('Lifecycle suite did not pass')
    checks=report['checks']
    pressure=report.get('pressure_only',False)
    if type(pressure) is not bool:raise AssertionError('Invalid lifecycle retention policy')
    expected={'resident_cancel','cleanup'} if pressure else {'ttl','cold_cancel','resident_cancel','cleanup'}
    if set(checks)!=expected:
        raise AssertionError('Incomplete lifecycle suite')
    validate_quiescent(checks['cleanup'])
    if pressure and checks['cleanup']['active_residency'].get('retention_policy')!='pressure':
        raise AssertionError('Pressure-only lifecycle used a TTL cache')
    for mode in (('resident',) if pressure else ('ttl','cold','resident')):
        item=checks['ttl' if mode=='ttl' else mode+'_cancel']
        first=item['first']
        if len(first['output_ids'])!=64:raise AssertionError('Incomplete initial checkpoint stream')
        history=first['input_tokens']+63
        response_id=item['second']['response_id'] if mode=='ttl' else item['response_id']
        proof=exact_restore(capture,response_id,history,resident=mode=='resident',cancelled=mode=='cold')
        if proof!=item['restore']:raise AssertionError('Saved lifecycle proof differs from worker evidence')
        if mode=='ttl':
            validate_quiescent(item['before']);validate_quiescent(item['after'])
            if (len(item['second']['output_ids'])!=32 or
                    item['after']['active_residency']['expired']<=item['before']['active_residency']['expired']):
                raise AssertionError('No observed TTL expiry and complete cold continuation')
        else:
            validate_quiescent(item['resources'])
            terminals=[r for r in rows(capture/'scheduler_finished.jsonl') if r['request_id']==proof['request_id']]
            if (len(terminals)!=1 or terminals[0]!=item['terminal'] or terminals[0]['status']!='FINISHED_ABORTED'
                    or terminals[0]['lifetime_reservation_retained'] is not True):
                raise AssertionError('Cancellation did not retain its native lease through retirement')
            if mode=='cold':
                stages=[r for r in rows(capture/'async_refresh.jsonl')
                        if r['request_id']==proof['request_id'] and r['stage']=='restore_submitted']
                if (stages!=[item['stage']] or not item['stage']['at']<=item['cancelled_at']<proof['finished']):
                    raise AssertionError('Cold cancellation missed the restore interval')
            follow=item['followup'];restores=rows(capture/'async_restore.jsonl')
            rid=native_id(follow['response_id'],restores)
            resumed=[r for r in restores if r['request_id']==rid]
            if (len(follow['output_ids'])!=32 or len(resumed)!=1 or resumed[0]!=item['followup_restore']
                    or resumed[0]['history_tokens']!=history or resumed[0]['cancelled']):
                raise AssertionError('Cancelled turn lost its complete CPU checkpoint')
            exact_restore(capture,follow['response_id'],history,resident=False,cancelled=False)
    return True


async def exercise(client, profile, fixture, capture, directory, q):
    directory.mkdir()
    report={}
    async def initial(name, length):
        tokens=fixture['prompt'][:length]
        result=await q.call(client,profile,tokens,name,64)
        await quiescent(capture)
        return result,tokens+result['output_ids']+fixture['tool'][:512]
    async def cleanup(name):
        await q.call(client,profile,[fixture['eos']],name,1,query=False)
        return await quiescent(capture)
    pressure=profile['kvmem'].get('active_resident_pressure_only',False)
    ttl=profile['kvmem']['active_resident_ttl_ms']/1000

    # A single exact M view fits this production profile. An idle engine need
    # not run a wall-clock timer; the successor supplies the expiry activity.
    if not pressure:
        if ttl<=0:raise AssertionError('Finite TTL lifecycle requires an enabled TTL cache')
        name='policy-lifecycle-ttl'
        first,tokens=await initial(name,32768)
        before=read(capture/'scheduler_resources.json')
        await asyncio.sleep(ttl+.25)
        second=await q.call(client,profile,tokens,name,32)
        after=await quiescent(capture)
        proof=exact_restore(capture,second['response_id'],first['input_tokens']+63,resident=False,cancelled=False)
        if after['active_residency']['expired']<=before['active_residency']['expired']:
            raise AssertionError('TTL expiry was not exercised')
        report['ttl']=dict(first=first,second=second,restore=proof,before=before,after=after)
        await cleanup(name)

    for mode in (('resident',) if pressure else ('cold','resident')):
        name='policy-lifecycle-cancel-'+mode
        first,tokens=await initial(name,65536 if mode=='cold' else 32768)
        if mode=='cold':await asyncio.sleep(ttl+.25)
        identifier='policy-cancel-'+uuid.uuid4().hex
        response_id='cmpl-'+identifier
        ready=asyncio.Event()
        task=asyncio.create_task(q.call(client,profile,tokens,name,8192,ready=ready,request_id=identifier))
        stage=None
        try:
            if mode=='resident':
                # Cancel after a proved hot import has actually decoded; this
                # exercises the transferred native page owner and its cleanup.
                await asyncio.wait_for(ready.wait(),120)
            else:
                pattern=re.compile(re.escape(response_id)+r'-0-[0-9a-f]{8}')
                def submitted():
                    if task.done():task.result();raise AssertionError('Request ended before cancel stage')
                    return next((r for r in rows(capture/'async_refresh.jsonl')
                        if pattern.fullmatch(r['request_id']) and r['stage']=='restore_submitted'),None)
                stage=await wait_until(submitted)
            if task.done():task.result();raise AssertionError('Request completed before cancellation')
            cancelled_at=time.perf_counter()
        finally:
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass
        def aborted():
            records=rows(capture/'scheduler_finished.jsonl')
            pattern=re.compile(re.escape(response_id)+r'-0-[0-9a-f]{8}')
            return next((r for r in records if pattern.fullmatch(r['request_id']) and r['status']=='FINISHED_ABORTED'),None)
        terminal=await wait_until(aborted)
        state=await quiescent(capture)
        proof=exact_restore(capture,response_id,first['input_tokens']+63,
                            resident=mode=='resident',cancelled=mode=='cold')
        if mode=='cold' and not stage['at']<=cancelled_at<proof['finished']:
            raise AssertionError('Cold cancellation missed the in-flight restore interval')
        if terminal['lifetime_reservation_retained'] is not True:
            raise AssertionError('Cancelled request lost its page reservation before its retirement fence')
        # Same accepted input lineage must remain usable after cancellation.
        follow=await q.call(client,profile,tokens,name,32)
        await quiescent(capture)
        resumed_id=native_id(follow['response_id'],rows(capture/'async_restore.jsonl'))
        resumed=next(r for r in rows(capture/'async_restore.jsonl') if r['request_id']==resumed_id)
        if resumed['history_tokens']!=first['input_tokens']+63 or resumed['cancelled']:
            raise AssertionError('Cancelled turn failed to preserve its original complete CPU checkpoint')
        exact_restore(capture,follow['response_id'],first['input_tokens']+63,resident=False,cancelled=False)
        report[mode+'_cancel']=dict(first=first,response_id=response_id,cancelled_at=cancelled_at,
            stage=stage,terminal=terminal,resources=state,restore=proof,followup=follow,followup_restore=resumed)
        await cleanup(name)
    report['cleanup']=await quiescent(capture)
    result=dict(passed=True,checks=report,pressure_only=pressure,
        scope='Hot-import cancellation under the selected retention policy; TTL/cold cancellation additionally '
              'checked by the finite-TTL arm. Full byte oracles; excluded from throughput.')
    validate_report(capture,result)
    q.save(directory/'results.json',result)
    return report
