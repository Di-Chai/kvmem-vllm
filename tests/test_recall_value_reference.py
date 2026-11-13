"""V-scale, accepted-domain, fixed-keep and no-fallback runtime contracts."""
import json

import pytest
import torch

from vllm_kvmem.serving import recall_experiment as experiment
from vllm_kvmem.serving.recall_value_reference import prepare_reference, score_value_records
from vllm_kvmem.serving.recall_value_scoring import REFERENCE_METHODS, aggregate_scores
from test_recall_experiment import ScoringAuthority, native_authority, finish, metadata
from test_recall_value_scoring import direct_oracle


@pytest.fixture(autouse=True)
def reference_environment(monkeypatch):
    monkeypatch.setenv('KVMEM_VALUE_RECALL_REFERENCE','1')
    monkeypatch.setenv('KVMEM_RECALL_DEVICE','cpu')
    monkeypatch.setenv('KVMEM_PRESSURE_QUERY_WINDOW','128')
    monkeypatch.setenv('KVMEM_RECALL_QUERY_BATCH','3')
    monkeypatch.setenv('KVMEM_RECALL_BLOCK_BATCH','2')
    monkeypatch.delenv('KVMEM_VALUE_RECALL_DIAGNOSTICS',raising=False)


def test_archived_fp8_values_are_reinterpreted_and_dequantized_per_layer():
    g=torch.Generator().manual_seed(18)
    keys=torch.randn(2,3,32,2,3,generator=g);counts=[32,9,2]
    values=torch.randn_like(keys).to(torch.float8_e4m3fn)
    scales=torch.tensor([2.5,.25]);q=torch.randn(2,5,4,3,generator=g)
    authority=ScoringAuthority(keys,counts,backend='qw3_cuda');authority.recall_value_scales=scales
    rows=list(authority.records.values());calls=[]
    def read(handles):
        calls.append(handles)
        return [(rows[h.slot].raw,values[:,h.slot,:counts[h.slot]].contiguous().view(torch.uint8)) for h in handles]
    authority.read_records=read
    iterator,stats=prepare_reference(authority,rows,q,[True]*3)
    actual,_=aggregate_scores(iterator(),layers=2,heads=4)
    expected={m:torch.zeros(3,dtype=torch.float64) for m in REFERENCE_METHODS}
    for l in range(2):
        for h in range(4):
            result,*_=direct_oracle(q[l,:,h].half().float(),keys[l,:,:,h//2],values[l,:,:,h//2].float()*scales[l],counts,[True]*3)
            for m in expected:expected[m]+=result[m].sum(0)/8
    for m in expected:torch.testing.assert_close(torch.tensor(actual[m],dtype=torch.float64),expected[m],atol=3e-6,rtol=1e-5)
    assert stats['value_scales']==scales.tolist() and stats['archive_kv_read_bytes']>0
    _,again=prepare_reference(authority,rows,q,[True]*3)
    assert len(calls)==1 and again['new_operand_blocks']==0


def test_missing_value_scale_fails_without_assuming_uint8_is_a_value():
    a=ScoringAuthority(torch.zeros(1,2,32,1,2),[32,32])
    with pytest.raises(ValueError,match='explicit.*Value scales'):
        score_value_records(a,list(a.records.values()),torch.ones(1,1,1,2),[True,True],'block_value_norm')


@pytest.mark.parametrize('method',REFERENCE_METHODS[1:])
@pytest.mark.parametrize('floating_values',[False,True])
def test_packed_operands_preserve_every_value_scale_and_final_score(monkeypatch,method,floating_values):
    g=torch.Generator().manual_seed(818)
    counts=[32,32,17,1,31]
    keys=torch.randn(2,5,32,2,7,generator=g).to(torch.float8_e4m3fn)
    values=torch.randn(2,5,32,2,7,generator=g).to(torch.float8_e4m3fn)
    a=ScoringAuthority(keys,counts,backend='qw3_cuda');a.recall_value_scales=torch.tensor([.371,1.293])
    rows=list(a.records.values());q=torch.randn(2,7,4,7,generator=g)
    def read(handles):
        return [(rows[h.slot].raw, (values[:,h.slot,:counts[h.slot]].float() if floating_values else
            values[:,h.slot,:counts[h.slot]].contiguous().view(torch.uint8))) for h in handles]
    a.read_records=read
    mask=[True,False,True,True,True]
    baseline,_=prepare_reference(a,rows,q,mask,cache_tiles=True)
    packed,stats=prepare_reference(a,rows,q,mask,cache_tiles=True,packed=True)
    for l in range(2):
        for h in range(2):
            for begin in range(0,len(rows),2):
                end=min(begin+2,len(rows))
                old=baseline.read_blocks(begin,end,l,h);new=packed.read_blocks(begin,end,l,h)
                assert all(torch.equal(x,y) for x,y in zip(old,new))
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND','streaming')
    old,_=score_value_records(a,rows,q,mask,method)
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND','streaming_packed')
    new,stats=score_value_records(a,rows,q,mask,method)
    assert old==new
    assert stats['actual_backend']=='streaming_packed' and stats['operand_backend']=='packed'
    assert stats['operand_pack_seconds']>=0 and stats['operand_transfer_bytes']>0


@pytest.mark.parametrize('method',REFERENCE_METHODS)
@pytest.mark.parametrize('event_type',['prefill_pressure','decode_pressure','query'])
def test_all_three_entrypoints_keep_method_budget_and_include_fixed_keeps(tmp_path,monkeypatch,method,event_type):
    monkeypatch.setenv('KVMEM_RECALL_METHOD',method)
    a=native_authority(count=256,budget=160,sink=1,recent=1);a.recall_value_scales=torch.ones(1)
    memory=a.memory;records=list(a.records.values())
    # Values vary independently of Keys; attention mass and Value scores need not rank alike.
    def read(handles):
        return [(a.records[(h.slot,h.generation)].raw,
            torch.full_like(a.records[(h.slot,h.generation)].raw,float(h.slot+1)).to(torch.float8_e4m3fn).view(torch.uint8)) for h in handles]
    a.read_records=read
    query=torch.tensor([[[[1.,0.]]]])
    if event_type!='query':query=query.expand(1,128,1,2)
    meta=(metadata(a) if event_type!='query' else dict(query_source='public_query',query_begin=256,query_end=257))
    try:
        plan=experiment.select_for_event(a,query,[records[2].handle],max_tokens=128,
            event_type=event_type,query_metadata=meta,request_id='new-reference',output=tmp_path)
        event=a.recall_selection_report;chosen=set(event['selected_block_ids'])
        assert {records[0].handle.slot,records[2].handle.slot,records[-1].handle.slot}<=chosen
        assert event['score_mask_count']==8  # Native Query mask excludes sink/recent; normalization includes both.
        assert event['charged_tokens']<=128 and event['actual_method']==event['configured_method']==method
        assert event['fallback_reason'] is None
        assert event['value_scales']==[1.] if method!='exact_mass' else True
        finish(memory);del plan
    finally:assert memory.close()


def test_same_state_diagnostics_use_full_selected_set_and_restore_actual_picker(tmp_path,monkeypatch):
    monkeypatch.setenv('KVMEM_RECALL_METHOD','exact_mass')
    monkeypatch.setenv('KVMEM_VALUE_RECALL_DIAGNOSTICS','query')
    a=native_authority(count=256,budget=160,sink=1,recent=1);a.recall_value_scales=torch.ones(1)
    rows=list(a.records.values());q=torch.tensor([[[[1.,0.]]]])
    try:
        plan=experiment.select_for_event(a,q,[rows[2].handle],max_tokens=128,event_type='query',
            query_metadata=dict(query_source='public_query',query_begin=256,query_end=257),
            request_id='frozen-state',output=tmp_path)
        report=json.loads((tmp_path/'value_recall_diagnostics.jsonl').read_text())
        assert set(report['methods'])==set(REFERENCE_METHODS)
        assert report['normalization_mask']==[True]*8
        assert set(a.recall_selection_report['selected_block_ids'])==set(report['methods']['exact_mass']['selected_block_ids'])
        assert report['exact_mass_reference_max_abs_difference']<1e-5
        for m in REFERENCE_METHODS:
            out=report['methods'][m]
            assert len(out['ranked_block_ids'])==8 and len(out['block_scores'])==8
            assert out['set_output_error']['mean_query_layer_head_l2']==0  # Zero V, despite different score ranks.
        finish(a.memory);del plan
    finally:assert a.memory.close()


def test_real_state_profile_restores_backend_and_compares_native_selections(monkeypatch):
    from vllm_kvmem.serving.recall_value_profile import compare_query_backends
    monkeypatch.setenv('KVMEM_VALUE_RECALL_BACKEND','streaming_packed')
    monkeypatch.setenv('KVMEM_RECALL_BLOCK_BATCH','1024')
    a=native_authority(count=256,budget=160,sink=1,recent=1)
    a.recall_value_scales=torch.ones(1)
    rows=list(a.records.values());q=torch.ones(1,3,1,2)
    try:
        report=compare_query_backends(a,rows,q,[True]*8,[rows[2].handle.slot],max_tokens=128)
        for method in REFERENCE_METHODS[1:]:
            assert report['methods'][method]['scores_close']
            assert report['methods'][method]['selected_set_equal']
            assert report['methods'][method]['query_and_value_scales_equal']
        import os
        assert os.environ['KVMEM_VALUE_RECALL_BACKEND']=='streaming_packed'
        assert os.environ['KVMEM_RECALL_BLOCK_BATCH']=='1024'
        assert report['candidate_tokens']==256 and report['normalization_mask_all']
    finally:assert a.memory.close()
