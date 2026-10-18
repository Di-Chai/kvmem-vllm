"""Reject impossible joint reservations before shared scheduling/model work."""
import json
import asyncio
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.http_sessions import normalize_query_plan
from vllm_kvmem.request_plan import RequestCapacityError, RequestPlan
from vllm_kvmem.session_control import SessionLedger, _Entry
from vllm_kvmem.prompt_checkpoints import token_bytes


def config():
    return KVMemConfig(block_tokens=32,select_budget=32768,gen_budget=32768,repository_blocks=8192)


def environment(monkeypatch, *, rolling=False):
    c=config()
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps(dict(
        block_tokens=c.block_tokens,select_budget=c.select_budget,gen_budget=c.gen_budget,
        sink_blocks=c.sink_blocks,recent_blocks=c.recent_blocks,
        query_replay_budget=32768,max_model_len=262144,rolling_active_window=rolling)))
    return c


@pytest.mark.parametrize('query_tokens',[32736,32000])
def test_http_rejects_joint_zero_prefix_and_keep_band_overflow(monkeypatch,query_tokens):
    environment(monkeypatch)
    payload=dict(prompt=[1]*(32768+query_tokens),max_tokens=32768,
                 kvmem_query_span=[32768,32768+query_tokens])
    with pytest.raises(RequestCapacityError,match='active capacity'):
        normalize_query_plan(payload)
    assert 'kv_transfer_params' not in payload
    # A lower output reservation fits the same complete Query; never truncate Q.
    payload['max_tokens']=4096
    assert normalize_query_plan(payload)
    assert payload['kv_transfer_params']['kvmem_request_plan']['kvmem_query_span']==[32768,32768+query_tokens]


def test_rolling_and_dense_query_do_not_need_a_fictional_historical_prefix(monkeypatch):
    environment(monkeypatch,rolling=True)
    payload=dict(prompt=[1]*65504,max_tokens=32768,kvmem_query_span=[32768,65504])
    assert normalize_query_plan(payload)
    environment(monkeypatch)
    dense=dict(prompt=[1]*32736,max_tokens=32768,kvmem_query_span=[0,32736])
    assert normalize_query_plan(dense)
    plan=replace(RequestPlan.parse(dense['kv_transfer_params']['kvmem_request_plan'],32736,32,32768),output_tokens=32768)
    assert plan.selection_budget(0,32736,65536)==32768


def session_fixture(monkeypatch):
    c=environment(monkeypatch)
    imports=[]
    ledger=SessionLedger(4,32,lambda *args:imports.append(args),lambda *args:None,
        lambda *args:None,select_budget=c.select_budget,gen_budget=c.gen_budget,
        memory_config=c,checkpoint_policy='pm')
    prompt=[1]*64769
    entry=_Entry(('tenant','session'),1,None,phase='idle',history=64768,window=8192,
        tokens=token_bytes(prompt,64768),query=(32768,64768),mandatory=((32768,64768,'query'),),
        prompt_history=32768,prompt_window=8192,prompt_tokens=token_bytes(prompt,32768))
    ledger.entries[entry.namespace]=entry
    request=NS(session_id='session',cache_salt='tenant',request_id='next',prompt_token_ids=prompt,
        num_prompt_tokens=len(prompt),sampling_params=NS(max_tokens=32768),
        kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[32768,64768],
                                                 'kvmem_replay_span':[32768,64769]}})
    return ledger,entry,request,imports


def test_unchanged_session_auto_query_is_validated_after_prefix_suppression(monkeypatch):
    ledger,entry,request,imports=session_fixture(monkeypatch)
    payload=dict(prompt=request.prompt_token_ids,max_tokens=32768,session_id='session',
                 kvmem_query_span=[32768,64768])
    assert normalize_query_plan(payload)  # Cannot decide the native M/P hit in HTTP.
    assert ledger.lookup(request,0)==64768
    assert imports==[('next',64768,8192)]
    assert 'kvmem_query_span' not in request.kv_transfer_params['kvmem_request_plan']


def test_failed_force_p_validation_preserves_m_lineage_and_never_imports(monkeypatch):
    ledger,entry,request,imports=session_fixture(monkeypatch)
    request.kv_transfer_params['kvmem_request_plan']['kvmem_reselect']='force'
    before=dict(vars(entry))
    with pytest.raises(RequestCapacityError):ledger.lookup(request,0)
    assert vars(entry)==before
    assert ledger.entries[entry.namespace] is entry and not ledger.requests and not imports
    assert not ledger.take_commands().evictions


def test_cold_session_capacity_failure_does_not_acquire_or_evict(monkeypatch):
    ledger,entry,request,imports=session_fixture(monkeypatch)
    request.session_id='new'
    with pytest.raises(RequestCapacityError):ledger.lookup(request,0)
    assert list(ledger.entries)==[entry.namespace]
    assert not ledger.requests and not imports and ledger.generation==0


def test_mixed_message_query_preserves_roles_and_atomic_bounds():
    spans=[dict(role='user',begin=35,end=69),dict(role='assistant',begin=72,end=108),
           dict(role='tool',begin=110,end=149)]
    plan=RequestPlan.parse(dict(kvmem_message_spans=spans,kvmem_query_message_range=[0,3]),
                           160,32,256)
    assert plan.query==(35,149) and plan.replay_begin==32
    assert plan.admit(32,116,160,0,512)==0
    assert plan.admit(32,117,160,0,512)==117
    assert plan.message_spans==tuple((s['begin'],s['end'],s['role']) for s in spans)
    assert RequestPlan.parse(plan.payload(),160,32,256)==replace(plan,explicit_mandatory=True)
    with pytest.raises(ValueError,match='replay capacity|headroom'):
        RequestPlan.parse(dict(kvmem_message_spans=spans,kvmem_query_message_range=[0,3]),160,32,128)


def test_elastic_joint_rejection_releases_history_and_allows_retry(tmp_path):
    httpx=pytest.importorskip('httpx')
    from test_elastic_gateway import configuration
    from vllm_kvmem.elastic_gateway import create_app
    cfg=configuration(tmp_path)
    cfg.update(context_mode='meank',request_planning=True,active_context_tokens=128,
        gen_budget_tokens=128,block_tokens=32,sink_blocks=1,recent_blocks=1,
        max_output_tokens=128,max_model_len=4096,active_bytes=1000,
        token_history_bytes=10000,max_body_bytes=10000)
    seen=[]
    def upstream(request):
        data=json.loads(request.content);seen.append(data)
        return httpx.Response(200,json=dict(choices=[dict(token_ids=[8])],
            usage=dict(completion_tokens=1,prompt_tokens=len(data['prompt']))))
    async def run():
        app=create_app(cfg)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream),base_url='http://native') as native:
            app.state.client=native
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://gateway') as client:
                headers={'Authorization':'Bearer key-a'}
                path='/v1/kvmem/sessions/s/turn'
                body=dict(append_token_ids=[1]*600,max_tokens=128,kvmem_query_span=[512,600])
                response=await client.post(path,headers=headers,json=body)
                assert response.status_code==400 and not seen
                status=(await client.get('/v1/kvmem/status',headers=headers)).json()
                assert status['sessions']==[] and status['active_reserved_bytes']==0
                body['max_tokens']=8
                response=await client.post(path,headers=headers,json=body)
                assert response.status_code==200 and len(seen)==1
                before=(await client.get('/v1/kvmem/status',headers=headers)).json()
                assert before['sessions'][0]['version']==1
                assert before['sessions'][0]['history_tokens']==601
                invalid=dict(version=1,append_token_ids=[2]*7,max_tokens=128,
                    kvmem_query_span=[512,608],kvmem_reselect='force')
                response=await client.post(path,headers=headers,json=invalid)
                assert response.status_code==400 and len(seen)==1
                after=(await client.get('/v1/kvmem/status',headers=headers)).json()
                assert after['sessions']==before['sessions']
                assert after['token_history_bytes']==before['token_history_bytes']
                assert after['active_reserved_bytes']==after['restore_reserved_bytes']==0
                assert after['active_requests']==after['queued_requests']==0
                invalid['max_tokens']=8
                response=await client.post(path,headers=headers,json=invalid)
                assert response.status_code==200 and response.json()['version']==2
                assert len(seen)==2
    asyncio.run(run())
