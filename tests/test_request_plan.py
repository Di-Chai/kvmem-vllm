import pytest
from vllm_kvmem.request_plan import RequestPlan


def test_full_query_and_partial_tail_are_one_atomic_replay():
    plan=RequestPlan.parse({'kvmem_query_span':[9001,11003], 'kvmem_reselect':'force'},12000,128,4096)
    assert plan.replay_begin==8960
    assert plan.admit(8832,4096,12000,0,12288)==128
    assert plan.admit(8960,1024,12000,0,12288)==0
    assert plan.admit(8960,4096,12000,0,12288)==2043
    assert plan.action(8960,12000,0,12288)=='query'
    with pytest.raises(RuntimeError,match='split'):
        plan.admit(9984,4096,12000,0,12288)


def test_pressure_keeps_complete_query_headroom_without_shortening_history():
    p=RequestPlan.parse({'kvmem_query_span':[25003,27103]},28001,128,4096)
    h=offset=0;events=[]
    while h<28001:
        n=p.admit(h,4096,28001,offset,12288)
        assert n>0 and h-offset+n<=12288
        action=p.action(h,28001,offset,12288)
        if h>=8192 and action:
            events.append((action,h,n));offset=h-8192
        h+=n
    assert h==28001
    queries=[e for e in events if e[0]=='query']
    assert queries==[('query',24960,2143)]
    assert any(e[0]=='pressure' for e in events)


def test_message_roles_and_complete_range_choose_only_user_query():
    p=RequestPlan.parse({'kvmem_message_spans':[
        {'role':'tool','begin':0,'end':4000}, {'role':'user','begin':4001,'end':6003}],
        'kvmem_query_message_range':[1,2]},6100,128,4096)
    assert p.query==(4001,6003)
    with pytest.raises(ValueError,match='user messages'):
        RequestPlan.parse({'kvmem_message_spans':[{'role':'tool','begin':0,'end':40}],
            'kvmem_query_message_range':[0,1]},40,128,4096)


@pytest.mark.parametrize('payload',[{'kvmem_reselect':'force'},
    {'kvmem_reselect':'off','kvmem_query_span':[1,3]},
    {'kvmem_prefill_window':'semantic_chunk','kvmem_query_span':[1,3]},
    {'kvmem_query_span':[True,3]}, {'kvmem_query_span':[1,9000]},
    {'kvmem_reselect':'truncate'}, {'unsupported':True}])
def test_invalid_plans_fail_explicitly(payload):
    with pytest.raises(ValueError): RequestPlan.parse(payload,9000,128,4096)


def test_keep_selected_exhaustion_and_off_do_not_invent_query():
    p=RequestPlan.parse({'kvmem_reselect':'off','kvmem_prefill_window':'keep_selected'},12000,128,4096)
    assert p.query is None
    assert p.action(12160,12000,0,12288)=='exhausted'
    p=RequestPlan.parse({'kvmem_reselect':'off'},25000,128,4096)
    assert p.action(12160,25000,0,12288)=='pressure'


def test_native_http_moves_plan_to_the_supported_metadata_channel(monkeypatch):
    import json
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'block_tokens':128,'gen_budget':4096,'select_budget':8192}))
    from vllm_kvmem.http_sessions import normalize_query_plan
    payload={'prompt':list(range(12000)), 'max_tokens':128, 'kvmem_query_span':[9001,11003],
        'kvmem_reselect':'force'}
    assert normalize_query_plan(payload)
    assert payload['kv_transfer_params']['kvmem_request_plan']['kvmem_query_span']==[9001,11003]
    assert 'kvmem_query_span' not in payload


def test_index_finalization_matches_original_serial_content_sum():
    import torch
    from vllm_kvmem import _native
    raw=torch.randn(3,128,2,32,generator=torch.Generator().manual_seed(9)).bfloat16()
    raw[:,0]=2**24;raw[:,1]=1;raw[:,2]=-2**24
    sums=torch.full((2,2,32),float('nan'))
    _native.finalize_mean_bf16(raw.view(torch.uint8).numpy(),sums.numpy(),128)
    expected=torch.zeros_like(sums)
    for t in range(128): expected.add_(raw[:2,t].float())
    assert torch.equal(sums,expected)
    assert torch.equal((sums/128).half(),(expected/128).half())


def test_native_http_unbounded_null_output_is_normalized_within_g(monkeypatch):
    import json
    from vllm_kvmem.http_sessions import normalize_query_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG',json.dumps({'block_tokens':128,'gen_budget':4096,'select_budget':8192}))
    payload={'prompt':[1,2,3], 'max_tokens':None}
    assert normalize_query_plan(payload) and payload['max_tokens']==16


def test_off_pressure_visits_boundary_before_native_mtp_page_allocation():
    p=RequestPlan.parse({'kvmem_reselect':'off'},65001,128,4096)
    h=offset=0;pressures=0
    while h<65001:
        n=p.admit(h,4128,65001,offset,12288)
        assert n>0 and h-offset+n+7<=12288
        if p.action(h,65001,offset,12288)=='pressure':
            pressures+=1;offset=h-8192
        h+=n
    assert h==65001 and pressures>0


def test_planning_rejects_g_without_pressure_replay_and_mtp_headroom():
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.options import RuntimeOptions
    with pytest.raises(ValueError,match='three blocks'):
        RuntimeOptions(memory=KVMemConfig(select_budget=8192,gen_budget=256),request_planning=True)
