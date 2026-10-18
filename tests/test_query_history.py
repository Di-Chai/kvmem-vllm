import math
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.query_history import QueryRelevance


def test_original_normalization_decay_and_replay_are_logical_rounds():
    history = QueryRelevance(0.5)
    history.begin('a',range(16))
    assert history.preview({0:1.,4:3.,20:100.}) == {0:.25,4:.75}
    assert history.commit()
    history.begin('b',range(32))
    expected = {0:.625,4:.375,8:.5}
    assert history.preview({0:2.,8:2.}) == expected
    assert history.preview({0:2.,8:2.}) == expected
    history.commit()
    assert not history.commit()
    history.begin('regeneration',range(32))
    assert history.preview({0:2.,8:2.}) == expected
    history.commit()
    assert history.policy.round_count == 2


def test_no_query_does_not_decay_and_aborted_divergence_does_not_commit():
    h = QueryRelevance(.5)
    h.begin('a',range(16)); h.preview({0:1}); h.commit()
    h.begin('b',range(24)); h.commit()
    assert h.policy.scores == {0:1}
    h.begin('c',list(range(8))+[100]*24)
    assert h.preview({4:1}) == {4:1}
    h.abort()
    assert h.policy.committed_scores == {0:1}
    h.begin('d',range(32)); h.preview({4:1});h.commit()
    assert h.policy.scores == {0:.5,4:1}


def test_users_and_snapshots_are_isolated():
    a,b = QueryRelevance(),QueryRelevance()
    for h in (a,b): h.begin('same-request',range(16))
    a.preview({0:1});b.preview({4:1});a.commit();b.commit()
    saved=a.policy.snapshot()
    a.begin('next',range(24));a.preview({8:1});a.commit()
    assert b.policy.scores == {4:1}
    a.policy.restore(saved)
    assert a.policy.scores == {0:1}


def test_original_extreme_values_and_invalid_scores():
    h=_native.QueryHistory();h.begin(16,0,.5)
    assert h.preview({0:1e308,4:1e308}) == {0:.5,4:.5}
    assert h.preview({0:1e-300}) == {0:1}
    for bad in (-1,math.nan,math.inf):
        with pytest.raises(ValueError): h.preview({0:bad})


def test_history_changes_actual_topk_selection(tmp_path,monkeypatch):
    from test_authority import setup,payload,drain
    from vllm_kvmem.authority import _Ready
    a,archive=setup(tmp_path)
    a.append(*payload(0,16),history_begin=0,local_rope_begin=0);drain(a)
    a.query_history=QueryRelevance(1)
    a.query_history.begin('old',range(12));a.query_history.preview({4:1});a.query_history.commit()
    a.query_history.begin('new',range(16))
    # Instantaneous relevance favours page 8. Retained evidence makes page 4 win.
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *args:torch.tensor([0.,.4,.6,0.]))
    plan=a.select_for_queries(torch.ones(2,1,2,2))
    handles={r.handle.slot:r.history_begin for r in a.records.values()}
    assert [handles[r.block_id] for r in plan.remaps] == [0,4]
    a.memory.attach_completion(_Ready());assert a.memory.commit_selection()
    del plan
    assert a.close();assert archive.close()


def test_private_batch_preserves_public_inputs_and_native_identity():
    from vllm.v1.worker.gpu.input_batch import InputBatch,InputBuffers
    from vllm_kvmem.guided_query import private_batch
    batch=InputBatch.make_dummy(2,8,InputBuffers(2,8,torch.device('cpu')))
    batch.idx_mapping[1]=7
    batch.idx_mapping_np[1]=7
    private=private_batch(batch,1,[42,43],100,torch.device('cpu'))
    assert private.req_ids == [batch.req_ids[1]] and private.idx_mapping.tolist() == [7]
    assert private.positions.tolist() == [100,101] and private.seq_lens.tolist() == [102]
    private.input_ids.zero_()
    assert batch.num_tokens == 8 and batch.positions.tolist() == [0]*8


def test_guided_and_history_options_can_be_enabled_together():
    from test_options import config
    from vllm_kvmem.options import RuntimeOptions
    payload={'kvmem':{'request_planning':True,'query_history':True,'guided_query_tokens':32,
        'session_capacity':8,'session_checkpoint_bytes':1024,'async_archive_retirement':True,
        'repository_blocks':8192}}
    options=RuntimeOptions.from_vllm(config(payload))
    assert options.query_history and options.guided_reserve == 384


def test_guided_reserve_is_accounted_before_query_admission():
    from vllm_kvmem.request_plan import RequestPlan
    p=RequestPlan.parse({'kvmem_query_span':[48000,50000]},50000,128,16384,384)
    boundary=p.prefill_boundary(45000,49152)
    assert boundary+(p.replay_end-p.replay_begin)+384+128 <= 49152
    with pytest.raises(ValueError,match='headroom'):
        RequestPlan.parse({'kvmem_query_span':[32768,48768]},48768,128,16384,384)


@pytest.mark.parametrize('restored',[False,True])
def test_query_capture_cannot_publish_unconfirmed_mtp_neighbour(restored):
    from vllm_kvmem.capture_runtime import CaptureRuntime
    from vllm_kvmem.session_control import SessionBinding
    import numpy as np
    ns=SimpleNamespace
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype=torch.float8_e4m3fn
    runtime.options=ns(query_history=True)
    runtime.queue=[];runtime.authorities={};runtime.batches=0
    if restored:
        runtime.authorities['b']=ns(query_history=ns(request_id='old',observation={0:1}))
    runtime.names=('l0','l1');runtime.row_elements=4
    runtime.scales=torch.ones(2)
    runtime.downloads=ns(pending=[],buffers=[1],download=lambda data:ns(payload=data))
    runtime.runner=ns(_kvmem_layers=runtime.names,_kvmem_sessions=ns(bindings={
        'b':SessionBinding(('u','s'),1,'b',(1,2,3,4),query_span=(0,4))}))
    calls=[]
    real_drain=runtime.drain_requests
    def drain_requests(ids, **kwargs):
        calls.append((tuple(ids),len(runtime.queue),kwargs['wait']))
        return real_drain(ids,**kwargs)
    runtime.drain_requests=drain_requests
    runtime.drain=lambda **kwargs:pytest.fail('Query capture used a global publication barrier')
    captures=[ns(name=name,positions=torch.arange(8),
        key=torch.ones(8,1,2,dtype=torch.bfloat16),value=torch.ones(8,1,2,dtype=torch.bfloat16),
        query=torch.ones(8,2,2,dtype=torch.bfloat16)) for name in runtime.names]
    batch=ns(num_reqs=2,num_tokens=8,req_ids=['a','b'],
        query_start_loc=torch.tensor([0,4,8]),query_start_loc_np=np.array([0,4,8]),
        positions=torch.arange(8),has_prefill=True,is_prefilling_np=np.array([False,True]),
        num_computed_tokens_np=np.array([100,0]),num_scheduled_tokens=np.array([4,4]))
    runtime.submit(captures,batch,torch.tensor([0,0]))
    assert calls==[(('a','b'),0,True)]  # Only this batch's older prefixes can publish.
    assert runtime.queue[0].confirmed is False
    assert runtime.queue[0].request_tokens==(4,4)
    assert len(runtime.queue)==1 and set(runtime.queue[0].query_inputs)=={'b'}
    assert runtime.queue[0].query_inputs['b'][1] is not None, 'Old observations cannot suppress regeneration capture'
    assert set(runtime.authorities)==({'b'} if restored else set())


def test_guided_allocation_uses_logical_pages_not_native_gdn_page_size(monkeypatch):
    from test_capacity import make_gate
    gate,manager,pool=make_gate()
    assert manager.block_size==16
    gate.guided_reserve=4
    gate.guided_block_tokens=4
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG','{"guided_reserve":4}')
    request=SimpleNamespace(request_id='a',num_prompt_tokens=24,max_tokens=1,
        num_computed_tokens=20,kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[20,24]}})
    calls=[]
    gate.native.allocate_slots=lambda req,n,**kwargs:calls.append(kwargs) or []
    assert gate.allocate_slots(request,4)==[]
    assert calls[0]['num_lookahead_tokens']==4
    gate.guided_reserve=8
    gate.reservations.clear()
    request.request_id='nearly-full'
    request.num_computed_tokens=52
    request.num_prompt_tokens=59
    request.kv_transfer_params={'kvmem_request_plan':{'kvmem_query_span':[52,59]}}
    assert gate.allocate_slots(request,7)==[]
    assert 'num_lookahead_tokens' not in calls[-1], 'A private rewrite cannot overrun an existing active view'
