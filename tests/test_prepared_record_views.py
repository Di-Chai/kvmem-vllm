"""Staging exports must preserve bytes, generations and asynchronous lifetimes."""
import hashlib
from threading import Event

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.core.config import KVMemConfig
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.core.options import RuntimeOptions


def make_owner(dtype, enabled, *, backend='qw3_cuda', resident=4, layers=2, heads=1, dim=64, block=4):
    cfg=KVMemConfig(block_tokens=block,select_budget=block*32,gen_budget=block*8,
        repository_blocks=64,sink_blocks=1,recent_blocks=0)
    archive=MemoryArchive('',layers*block*heads*dim*(dtype.itemsize+1),64,
        cpu_direct_record_seal=True)
    archive.async_seal=True
    owner=RequestAuthority('views',cfg,archive,layers,heads,dim,raw_dtype=dtype,
        scoring_layers=layers-1,mean_k_backend=backend,resident_records=resident,
        cpu_reuse_record_views=enabled)
    return owner,archive


@pytest.mark.parametrize('dtype',[torch.float8_e4m3fn,torch.bfloat16])
@pytest.mark.parametrize('backend',['qw3_cuda','torch_reference'])
@pytest.mark.parametrize('batch',[False,True])
def test_partial_acceptance_recycled_slots_and_exact_old_records(dtype,backend,batch):
    generator=torch.Generator().manual_seed(851)
    raw=torch.randn(2,40,1,64,generator=generator).to(dtype)
    values=torch.randint(0,256,raw.shape,dtype=torch.uint8,generator=generator)
    outputs=[]
    for enabled in (False,True):
        a,archive=make_owner(dtype,enabled,backend=backend)
        snapshots=[]
        try:
            start=0
            for n,accepted in ((7,5),(8,7),(8,8),(6,4)):
                kwargs=dict(history_begin=start,local_rope_begin=start,accepted=accepted)
                r,v=raw[:,start:start+n],values[:,start:start+n]
                if batch:assert append_authority_batch([(a,r,v,kwargs)])==1
                else:assert a.append(r,v,**kwargs)
                start+=accepted
                a.publish_sealed(lambda:archive.wait_for_capacity(1))
            for record in a.records.values():
                r,v=a.read_record(record.handle)
                expected=raw[:,record.history_begin:record.history_begin+4].view(torch.uint8).numpy().tobytes()+values[:,record.history_begin:record.history_begin+4].numpy().tobytes()
                assert r.view(torch.uint8).numpy().tobytes()+v.numpy().tobytes()==expected
                assert record.digest==hashlib.sha256(expected).hexdigest()
                snapshots.append((expected,record.key_sum.numpy().tobytes(),
                                  None if record.mean_key is None else record.mean_key.numpy().tobytes()))
            if enabled:
                assert a.record_view_stats['bound_records']==6
                assert a.record_view_stats['slot_exports']==len(a._record_buffers)<6
                assert a.record_view_stats['sealed_records']==6
            outputs.append(snapshots)
        finally:
            assert a.close() and archive.close()
            assert not a._record_buffer_views and a.record_staging_bytes==0
    assert outputs[0]==outputs[1]


def test_pending_seal_retains_exports_and_prevents_reuse(monkeypatch):
    a,archive=make_owner(torch.float8_e4m3fn,True,resident=1)
    entered,release=Event(),Event();pack=_native.pack_record_pairs
    def delayed(*args):
        entered.set();assert release.wait(5)
        return pack(*args)
    monkeypatch.setattr(_native,'pack_record_pairs',delayed)
    raw=torch.full((2,4,1,64),2.).to(a.raw_dtype);values=torch.full(raw.shape,17,dtype=torch.uint8)
    try:
        assert append_authority_batch([(a,raw,values,dict(history_begin=0,local_rope_begin=0))])==1
        assert entered.wait(3)
        views=a._record_buffer_views[0]
        assert append_authority_batch([(a,raw,values,dict(history_begin=4,local_rope_begin=4))])==0
        assert a.history_end==4 and a._record_buffer_views[0] is views
        release.set();a.publish_sealed(lambda:archive.wait_for_capacity(1))
        raw2=torch.full(raw.shape,6.).to(a.raw_dtype);v2=torch.full(values.shape,51,dtype=torch.uint8)
        assert append_authority_batch([(a,raw2,v2,dict(history_begin=4,local_rope_begin=4))])==1
        a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert a._record_buffer_views[0] is views
        first=next(iter(a.records.values()));r,v=a.read_record(first.handle)
        assert torch.equal(r.view(torch.uint8),raw.view(torch.uint8)) and torch.equal(v,values)
    finally:
        release.set();a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert a.close() and archive.close()


def test_refused_archive_preserves_views_until_retry(monkeypatch):
    a,archive=make_owner(torch.bfloat16,True,resident=1)
    write=archive.write_from_views
    monkeypatch.setattr(archive,'write_from_views',lambda *args:None)
    raw=torch.full((2,4,1,64),3.).to(a.raw_dtype);values=torch.full(raw.shape,29,dtype=torch.uint8)
    try:
        assert a.append(raw,values,history_begin=0,local_rope_begin=0)
        record=next(iter(a.records.values()))
        views=record.archive_views
        assert views is not None and record.scatter_views is not None and not record.submitted
        assert not a.flush() and record.archive_views is views
        monkeypatch.setattr(archive,'write_from_views',write)
        a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert record.archive_views is None and record.scatter_views is None
        assert a.record_view_stats['sealed_records']==1
    finally:
        monkeypatch.setattr(archive,'write_from_views',write)
        assert a.close() and archive.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_worker_profile_transport(tmp_path,enabled):
    from test_graph_options import model_profile,runtime,worker_payload
    from vllm_kvmem.serving.serve import build_launch
    profile=model_profile(tmp_path);profile['kvmem']['cpu_reuse_record_views']=enabled
    assert runtime(worker_payload(build_launch(profile,tmp_path,tmp_path/'out'))).cpu_reuse_record_views is enabled
    assert RuntimeOptions().cpu_reuse_record_views is False


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_strict_option(value):
    with pytest.raises(ValueError,match='cpu_reuse_record_views'):
        RuntimeOptions(cpu_reuse_record_views=value)
