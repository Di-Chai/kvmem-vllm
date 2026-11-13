"""Independent ordered publications must not free another owner's DMA source."""
from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.capture_runtime import CapturedPacket
from vllm_kvmem.serving.multiowner_prefill import (owner_chunk_limit, capture_owner_plans,
    attach_owner_views, bind_owner_payload)
from vllm_kvmem.serving.terminal import TerminalPrefix
from vllm_kvmem.core.options import RuntimeOptions
from test_logical_capture_publication import runtime_fixture
from test_logical_publication import fixture


@contextmanager
def owners(tmp_path, *, second_chunks=2):
    with runtime_fixture(tmp_path/'first') as a, fixture(tmp_path/'second',request_id='s',chunks=second_chunks) as b:
        r=a.runtime;r.options.prefill_physical_multi_owner=True
        r.authorities['s']=b.authority;r.accepted['s']=r.rejected['s']=0;r.last_steps['s']=[]
        r._snapshot_capacity('s',b.authority)
        n=a.group.count;t=n+b.group.count
        # Deliberately different K/V in the second owner's source region.
        b.raw.fill_(3);b.values.fill_(27)
        header=torch.tensor([0,n,t,0,0,0,0,0,0],dtype=torch.int64)
        raw=torch.cat((a.raw,b.raw),dim=1);v=torch.cat((a.values,b.values),dim=1)
        data=torch.cat((header.view(torch.uint8),raw.view(torch.uint8).flatten(),v.flatten()))
        download=NS(payload=data,slot=0,query=lambda:True,event=NS(synchronize=lambda:None))
        parent=CapturedPacket(('r','s'),t,download,confirmed=False,request_tokens=(n,b.group.count),
            source_generation=17,action_inputs={'r':object(),'s':object()})
        attach_owner_views(parent,{'r':(a.group,a.parts),'s':(b.group,b.parts)})
        r.queue=[parent];r.downloads.pending={0:download};r.released=[]
        def release(d):
            assert all(c.logical_cursor.phase=='released' for c in parent.logical_owners.values())
            assert all(c.logical_payload is None for c in parent.logical_owners.values())
            r.released.append(d);del r.downloads.pending[d.slot];return True
        r.downloads.release=release
        yield r,parent,a,b


@pytest.mark.parametrize('order',[('r','s'),('s','r')])
@pytest.mark.parametrize('retained',[0,1,2048,4096])
def test_independent_commit_cancel_and_shared_release(tmp_path,order,retained):
    with owners(tmp_path) as (r,p,a,b):
        assert not r.drain_requests(set(order),wait=True)
        assert a.authority.history_end==b.authority.history_end==0
        terminals={} if retained==4096 else {'s':TerminalPrefix(retained,4096,0,'FINISHED_ABORTED')}
        r.begin_step(terminals,terminals)
        for i,rid in enumerate(order):
            assert r.drain_requests({rid},wait=True)
            assert r.publication_ready({rid})
            assert bool(r.released)==(i==1)
            if i==0:
                assert not p.source_released and r.downloads.pending
                assert not r.publication_ready({order[1]})
        assert a.authority.history_end==4096 and b.authority.history_end==retained
        assert r.accepted=={'r':4096,'s':retained}
        assert r.terminal_discarded.get('s',0)==4096-retained
        assert len(r.released)==1 and not r.queue and p.source_released
        for env,expected_raw,expected_v in ((a,a.raw,a.values),(b,b.raw,b.values)):
            env.authority.publish_sealed(lambda:None)
            for record in env.authority.records.values():
                k,v=env.authority.read_record(record.handle)
                start=record.history_begin
                assert torch.equal(k.view(torch.uint8),expected_raw[:,start:start+record.count].view(torch.uint8))
                assert torch.equal(v,expected_v[:,start:start+record.count])


def test_completed_peer_survives_later_failure_without_source_release(tmp_path,monkeypatch):
    with owners(tmp_path) as (r,p,a,b):
        r.begin_step();assert r.drain_requests({'r'},wait=True)
        def fail(*args,**kwargs):raise RuntimeError('injected child failure')
        monkeypatch.setattr(b.authority,'_prepare_append',fail)
        with pytest.raises(RuntimeError,match='child failure'):r.drain_requests({'s'},wait=True)
        assert r.accepted['r']==4096 and r.accepted['s']==0
        assert p.published=={0} and not r.released and not p.source_released
        with pytest.raises(RuntimeError,match='child failure'):r.drain_requests({'r'},wait=True)


def test_release_failure_is_quarantined_before_last_owner_ack(tmp_path):
    with owners(tmp_path) as (r,p,a,b):
        r.begin_step();assert r.drain_requests({'r'},wait=True)
        r.downloads.release=lambda _:False
        with pytest.raises(RuntimeError,match='DMA lease'):r.drain_requests({'s'},wait=True)
        assert p.published=={0} and not p.source_released and r.downloads.pending
        with pytest.raises(RuntimeError,match='DMA lease'):r.publication_ready({'s'})


def test_ready_mixed_cohort_can_publish_both_owners_in_one_claim(tmp_path):
    with owners(tmp_path) as (r,p,a,b):
        r.begin_step();assert r.drain_requests(None,wait=True)
        assert r.accepted=={'r':4096,'s':4096} and len(r.released)==1
        assert r.batched_append_requests==4


@pytest.mark.parametrize('order',[('r','s'),('s','r')])
def test_coalesced_owner_and_ordinary_peer_share_source(tmp_path,order):
    with owners(tmp_path) as (r,p,a,b):
        del p.logical_owners[1]
        p.tokens=6144;p.request_tokens=(4096,2048)
        header=torch.tensor([0,4096,6144,0,0,0,0,0,0],dtype=torch.int64)
        raw=torch.cat((a.raw,b.raw[:,:2048]),dim=1)
        values=torch.cat((a.values,b.values[:,:2048]),dim=1)
        p.download.payload=torch.cat((header.view(torch.uint8),raw.view(torch.uint8).flatten(),values.flatten()))
        p.action_inputs['s']=b.parts[0].action_packet(b.state)
        r.begin_step()
        for i,rid in enumerate(order):
            assert r.drain_requests({rid},wait=True)
            assert bool(r.released)==(i==1)
        assert r.accepted=={'r':4096,'s':2048}
        assert b.graph.accepted_through==2048 and b.state.prefill.samples==64
        assert r.batched_append_requests==3 and p.source_released


def test_shared_header_validation_precedes_any_append(tmp_path):
    with owners(tmp_path) as (r,p,a,b):
        p.download.payload[:72].view(torch.int64)[1]=4000
        r.begin_step()
        with pytest.raises(ValueError,match='admitted prefix'):r.drain_requests({'r'},wait=True)
        assert a.authority.history_end==b.authority.history_end==0 and not r.released


@pytest.mark.parametrize('count,expected',[(1,8),(2,7),(4,3),(8,1),(16,0)])
def test_cohort_sampling_stays_inside_existing_bank(count,expected):
    assert owner_chunk_limit(count,448,1024,8)==expected
    assert count*expected*32<=448


def test_capture_plan_validation_reuses_original_per_owner_checks(tmp_path):
    with owners(tmp_path) as (r,p,a,b):
        r.runner._kvmem_logical_prefill_step=7
        batch=NS(num_reqs=2,req_ids=('r','s'),is_prefilling_np=(True,True),
            num_computed_tokens_np=(0,0),query_start_loc_np=(0,4096,8192))
        inputs=NS(logical_groups={'r':a.group,'s':b.group},logical_slices={'r':a.parts,'s':b.parts})
        assert set(capture_owner_plans(r,inputs,batch))=={'r','s'}
        batch.num_computed_tokens_np=(0,32)
        with pytest.raises(RuntimeError,match='immutable'):capture_owner_plans(r,inputs,batch)


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_option_strict_boolean(value):
    with pytest.raises(ValueError,match='prefill_physical_multi_owner'):
        RuntimeOptions(prefill_physical_multi_owner=value)


def test_option_transport_preserves_original_budgets(tmp_path):
    from test_physical_prefill_options import model_profile,payload,build_launch,runtime,worker_payload
    profile=model_profile(tmp_path)
    profile['prefill_chunk_tokens']=1024
    profile['kvmem']=payload(prefill_physical_multi_owner=True,prefill_physical_logical_tokens=1024,
        prefill_physical_max_chunks=8,action_native_max_rows=448,select_budget=36864,gen_budget=16384)
    option=runtime(worker_payload(build_launch(profile,tmp_path,tmp_path/'launch')))
    assert option.prefill_physical_multi_owner
    assert option.memory.select_budget==36864 and option.memory.gen_budget==16384
