from threading import Event

import pytest
import torch

from vllm_kvmem.authority import RequestAuthority, append_authority_batch, _Ready
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive


def setup(dtype, backend='qw3_cuda', resident=2):
    cfg=KVMemConfig(block_tokens=4,select_budget=128,gen_budget=32,
        repository_blocks=64,sink_blocks=1,recent_blocks=0)
    archive=MemoryArchive('',2*4*64*(dtype.itemsize+1),64)
    archive.async_seal=True
    a=RequestAuthority('pool-owner',cfg,archive,2,1,64,raw_dtype=dtype,
        scoring_layers=1,mean_k_backend=backend,resident_records=resident)
    return a,archive


def source(dtype,value):
    raw=torch.full((2,4,1,64),float(value),dtype=torch.float32).to(dtype)
    values=torch.full(raw.shape,value,dtype=torch.uint8)
    return raw,values


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float8_e4m3fn])
@pytest.mark.parametrize('backend',['torch_reference','qw3_cuda'])
def test_many_sealed_pages_reuse_bounded_storage_without_changing_old_history(dtype,backend):
    a,archive=setup(dtype,backend)
    try:
        first=None;pointer=None
        for i in range(24):
            raw,values=source(dtype,i+1)
            assert append_authority_batch([(a,raw,values,
                dict(history_begin=4*i,local_rope_begin=4*i))])==1
            a.publish_sealed(lambda:archive.wait_for_capacity(1))
            assert len(a._record_buffers)==1 and not a._busy_record_buffers
            current=a._record_buffers[0][0].data_ptr()
            if pointer is None:pointer=current
            assert current==pointer
            if first is None:first=next(iter(a.records.values())).handle
        old_raw,old_values=a.read_record(first)
        r,v=source(dtype,1)
        assert torch.equal(old_raw.view(torch.uint8),r.view(torch.uint8))
        assert torch.equal(old_values,v)
        assert a.record_staging_bytes==a.record_bytes
        assert a.history_end==96 and archive.tier.used_slots==24
    finally:
        assert a.close() and a.record_staging_bytes==0 and archive.close()


def test_inflight_archive_source_cannot_be_reused_before_publication(monkeypatch):
    a,archive=setup(torch.float8_e4m3fn,resident=1)
    entered,release=Event(),Event();original=archive.tier.io
    def delayed(*args,**kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(archive.tier,'io',delayed)
    try:
        raw,v=source(a.raw_dtype,3)
        assert a.append(raw,v,history_begin=0,local_rope_begin=0)
        assert entered.wait(3)
        assert len(a._record_buffers)==1 and len(a._busy_record_buffers)==1
        r2,v2=source(a.raw_dtype,7)
        assert not a.append(r2,v2,history_begin=4,local_rope_begin=4)
        assert a.history_end==4 and len(a._record_buffers)==1
        release.set();a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert not a._busy_record_buffers
        assert a.append(r2,v2,history_begin=4,local_rope_begin=4)
        a.publish_sealed(lambda:archive.wait_for_capacity(1))
        first=next(r for r in a.records.values() if r.history_begin==0)
        restored,rv=a.read_record(first.handle)
        assert torch.equal(restored.view(torch.uint8),raw.view(torch.uint8)) and torch.equal(rv,v)
    finally:
        release.set()
        for job,_ in a.writes:job.result(timeout=5)
        assert a.close() and archive.close()


def test_partial_rewind_returns_slot_only_after_ordinary_reclamation():
    a,archive=setup(torch.float8_e4m3fn,resident=1)
    try:
        raw,v=source(a.raw_dtype,2)
        assert a.append(raw,v,history_begin=0,local_rope_begin=0)
        a.publish_sealed(lambda:archive.wait_for_capacity(1))
        checkpoint=a.memory.checkpoint()
        assert a.append(raw[:,:2],v[:,:2],history_begin=4,local_rope_begin=4)
        assert len(a._busy_record_buffers)==1
        assert a.begin_rewind(checkpoint) is not None
        a.attach_rewind_completion(_Ready());assert a.poll_rewind()
        assert a.history_end==4 and not a._busy_record_buffers
        raw2,v2=source(a.raw_dtype,9)
        assert a.append(raw2,v2,history_begin=4,local_rope_begin=4)
        a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert len(a._record_buffers)==1
    finally:
        del checkpoint  # The native checkpoint pins the preserved prefix.
        assert a.close() and archive.close()


def test_cancellation_keeps_partial_buffer_until_external_reader_releases_it():
    a,archive=setup(torch.float8_e4m3fn,resident=1)
    raw,v=source(a.raw_dtype,4)
    assert a.append(raw[:,:2],v[:,:2],history_begin=0,local_rope_begin=0)
    record=next(iter(a.records.values()));lease=a.memory.repository.pin(record.handle)
    try:
        assert not a.close()
        assert a.record_staging_bytes==a.record_bytes and len(a._busy_record_buffers)==1
    finally:lease.reset()
    assert a.close() and a.record_staging_bytes==0 and archive.close()
