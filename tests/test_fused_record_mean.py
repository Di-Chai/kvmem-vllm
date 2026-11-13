"""Copy/finalize fusion preserves codec bytes and FP32 token order."""
import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.core.config import KVMemConfig
from vllm_kvmem.core.memory_storage import MemoryArchive
from vllm_kvmem.core.options import RuntimeOptions


def make_owner(dtype, enabled, *, resident=4, layers=3, heads=2, dim=64, block=32):
    cfg=KVMemConfig(block_tokens=block,select_budget=block*32,gen_budget=block*8,
        repository_blocks=64,sink_blocks=1,recent_blocks=0)
    archive=MemoryArchive('',layers*block*heads*dim*(dtype.itemsize+1),64,
        cpu_direct_record_seal=True)
    archive.async_seal=True
    owner=RequestAuthority('fused',cfg,archive,layers,heads,dim,raw_dtype=dtype,
        scoring_layers=layers-1,mean_k_backend='qw3_cuda',resident_records=resident,
        cpu_fuse_record_mean=enabled)
    return owner,archive


def descriptor(dtype, offset=0, padded=False):
    g=torch.Generator().manual_seed(823)
    raw=torch.randn(3,48,2,64,generator=g).to(dtype)
    # Includes every FP8 encoding, or BF16 NaN/inf/subnormal/signed-zero codes.
    if dtype == torch.float8_e4m3fn:
        raw.view(torch.uint8).copy_(torch.arange(raw.numel()).remainder(256).to(torch.uint8).reshape(raw.shape))
    else:
        codes=torch.tensor([0,32768,1,32769,32640,65408,32704,65472,16256,49024],dtype=torch.int32).to(torch.int16)
        raw.view(torch.int16).flatten()[:len(codes)]=codes
    v=torch.randint(0,256,raw.shape,dtype=torch.uint8,generator=g)
    stride=2 if padded else 1
    destination=torch.full((3,32*stride,2,64),2.).to(dtype)[:,::stride]
    values=torch.full(destination.shape,17,dtype=torch.uint8)
    sums=torch.full((2,2,64),-311.,dtype=torch.float32)
    return (raw.view(torch.uint8).numpy(),v.numpy(),destination.view(torch.uint8).numpy(),
        values.numpy(),None,sums.numpy(),3,offset,32-offset,True)


@pytest.mark.parametrize('dtype',[torch.float8_e4m3fn,torch.bfloat16])
@pytest.mark.parametrize('offset',[0,7,31])
@pytest.mark.parametrize('padded',[False,True])
def test_full_and_partial_pages_bit_exact_including_nonfinite(dtype,offset,padded):
    old=descriptor(dtype,offset,padded);new=descriptor(dtype,offset,padded)
    assert _native.scatter_history_batch([old]) is None
    _native.finalize_mean_batch([(old[2],old[5],32,dtype==torch.float8_e4m3fn)])
    assert _native.scatter_history_batch([(*new,True)])==1
    for i in (2,3,5):
        np.testing.assert_array_equal(old[i].view(np.uint8),new[i].view(np.uint8))


@pytest.mark.parametrize('alias',['sum_raw','sum_other_values','sum_other_sum','page_other_page','page_own_values'])
def test_alias_is_rejected_before_any_write(alias):
    a=list(descriptor(torch.float8_e4m3fn));b=list(descriptor(torch.float8_e4m3fn))
    if alias=='sum_raw':a[5]=a[0].reshape(-1)[:2*2*64*4].view(np.float32).reshape(2,2,64)
    elif alias=='sum_other_values':a[5]=b[3].reshape(-1)[:2*2*64*4].view(np.float32).reshape(2,2,64)
    elif alias=='sum_other_sum':a[5]=b[5]
    elif alias=='page_other_page':b[2]=a[2]
    else:a[3]=a[2]
    before=[arr.copy() for row in (a,b) for arr in (row[2],row[3],row[5])]
    with pytest.raises(ValueError,match='alias'):_native.scatter_history_batch([(*a,True),tuple(b)])
    after=[arr for row in (a,b) for arr in (row[2],row[3],row[5])]
    for x,y in zip(before,after):np.testing.assert_array_equal(x,y)


def test_invalid_later_descriptor_does_not_partially_write():
    a=descriptor(torch.bfloat16);b=list(descriptor(torch.bfloat16));b[8]=33
    before=[a[i].copy() for i in (2,3,5)]
    with pytest.raises(ValueError):_native.scatter_history_batch([(*a,True),(*b,True)])
    for i,x in zip((2,3,5),before):np.testing.assert_array_equal(a[i],x)
    with pytest.raises(ValueError,match='bool'):_native.scatter_history_batch([(*a,1)])
    with pytest.raises(ValueError,match='copy-only'):_native.scatter_history_batch([(*a[:9],False,True)])


@pytest.mark.parametrize('dtype',[torch.float8_e4m3fn,torch.bfloat16])
def test_mixed_owners_partial_acceptance_recycled_archive_and_no_double_finalization(dtype,monkeypatch):
    raw=torch.randn(3,160,2,64,generator=torch.Generator().manual_seed(811)).to(dtype)
    values=torch.arange(raw.numel()).remainder(256).to(torch.uint8).reshape(raw.shape)
    owners=[make_owner(dtype,enabled) for enabled in (False,True)]
    try:
        for begin,candidates,accepted in [(0,64,64),(64,16,5),(69,27,27),(96,64,64)]:
            assert append_authority_batch([(a,raw[:,begin:begin+candidates],values[:,begin:begin+candidates],
                dict(history_begin=begin,local_rope_begin=begin,accepted=accepted)) for a,_ in owners])==2
            for a,archive in owners:a.publish_sealed(lambda:archive.wait_for_capacity(1))
        assert owners[0][0].fused_mean_stats==dict(records=0,batches=0)
        assert owners[1][0].fused_mean_stats==dict(records=5,batches=3)
        assert owners[0][0].mean_index_stats.snapshot()==owners[1][0].mean_index_stats.snapshot()
        for (old,new) in zip(*(a.records.values() for a,_ in owners)):
            assert old.digest==new.digest
            assert torch.equal(old.mean_key.view(torch.int16),new.mean_key.view(torch.int16))
            for i,r in enumerate((old,new)):
                k,v=owners[i][0].read_record(r.handle)
                assert torch.equal(k.view(torch.uint8),raw[:,r.history_begin:r.history_begin+32].view(torch.uint8))
                assert torch.equal(v,values[:,r.history_begin:r.history_begin+32])
    finally:
        for a,archive in owners:assert a.close() and archive.close()


@pytest.mark.parametrize('enabled',[False,True])
def test_worker_profile_transport(tmp_path,enabled):
    from test_graph_options import model_profile,runtime,worker_payload
    from vllm_kvmem.serving.serve import build_launch
    profile=model_profile(tmp_path);profile['kvmem'].update(cpu_fuse_record_mean=enabled,mean_k_backend='qw3_cuda')
    assert runtime(worker_payload(build_launch(profile,tmp_path,tmp_path/'out'))).cpu_fuse_record_mean is enabled
    assert RuntimeOptions().cpu_fuse_record_mean is False


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_strict_option(value):
    with pytest.raises(ValueError,match='cpu_fuse_record_mean'):RuntimeOptions(cpu_fuse_record_mean=value)


def test_unqualified_combinations_rejected():
    with pytest.raises(ValueError,match='SIMD'):
        RuntimeOptions(cpu_fuse_record_mean=True,cpu_mean_index_simd=True,mean_k_backend='qw3_cuda')
    with pytest.raises(ValueError,match='native'):
        RuntimeOptions(cpu_fuse_record_mean=True,mean_k_backend='torch_reference')
