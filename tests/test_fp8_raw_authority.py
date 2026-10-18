"""FP8 immutable source, accepted-prefix indexing and compact history ownership."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive


def owner(tmp_path, name='fp8', *, backend='qw3_cuda', archive=None):
    cfg = KVMemConfig(block_tokens=4, select_budget=8, gen_budget=8,
                     repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = archive or AsyncArchive(str(tmp_path), 2*4*1*64*2, 32)
    return RequestAuthority(name,cfg,archive,2,1,64,raw_dtype=torch.float8_e4m3fn,
                            scoring_layers=1,mean_k_backend=backend),archive


def publish(a):
    a.publish_sealed(lambda: None)


@pytest.mark.parametrize('batched',[False,True])
def test_compact_accepted_history_sealed_and_partial_roundtrip_and_mean_index(tmp_path,batched):
    a,archive=owner(tmp_path)
    gen=torch.Generator().manual_seed(447)
    raw=(torch.randn(2,13,1,64,generator=gen)*3).to(torch.float8_e4m3fn)
    values=torch.randint(0,256,raw.shape,dtype=torch.uint8,generator=gen)
    try:
        assert a.record_bytes==1024 and a.raw_record_bytes==a.value_record_bytes==512
        begin=0
        for candidates,accepted in [(5,3),(6,4),(4,2)]:
            r,v=raw[:,begin:begin+candidates],values[:,begin:begin+candidates]
            kwargs=dict(history_begin=begin,local_rope_begin=begin,accepted=accepted)
            if batched: assert append_authority_batch([(a,r,v,kwargs)])==1
            else: assert a.append(r,v,**kwargs)
            begin+=accepted
        publish(a)
        assert a.history_end==9 and archive.tier.used_slots==2
        for index,record in enumerate(a.records.values()):
            stored,value=a.read_record(record.handle)
            first=index*4; count=record.count
            assert torch.equal(stored.view(torch.uint8),raw[:,first:first+count].view(torch.uint8))
            assert torch.equal(value,values[:,first:first+count])
            expected=torch.zeros_like(record.key_sum)
            for token in range(count): expected+=raw[:1,first+token].float()
            assert torch.equal(record.key_sum,expected)
        _,means=a.mean_keys()
        assert means.dtype==torch.float16
        assert torch.equal(means[:,0],(a.records[(0,1)].key_sum/4).half())
        # No rejected candidate beyond the nine accepted rows reaches storage.
        assert sum(r.count for r in a.records.values())==9
    finally:
        assert a.close() and archive.close()


def test_fp8_lookup_widens_every_finite_e4m3_code_before_sequential_mean():
    codes=torch.arange(256,dtype=torch.uint8)
    codes=codes[(codes!=127)&(codes!=255)]
    widened=torch.empty(1,1,254)
    _native.finalize_mean_fp8(codes.reshape(1,1,1,254).numpy(),widened.numpy(),1)
    assert torch.equal(widened.flatten(),codes.view(torch.float8_e4m3fn).float())
    raw=codes.reshape(1,254,1,1)
    sums=torch.full((1,1,1),123.)
    _native.finalize_mean_fp8(raw.numpy(),sums.numpy(),254)
    expected=torch.zeros_like(sums)
    for value in codes.view(torch.float8_e4m3fn).float():expected+=value
    assert torch.equal(sums,expected)
    before=sums.clone()
    with pytest.raises(ValueError):_native.finalize_mean_fp8(raw.numpy(),sums.numpy(),255)
    assert torch.equal(sums,before)


@pytest.mark.parametrize('count',[1,2,3,4])
def test_fused_fp8_decode_scatter_matches_accepted_torch_sum_and_byte_copies(count):
    g=torch.Generator().manual_seed(789+count)
    raw=torch.randn(17,7,4,256,generator=g).mul(20).to(torch.float8_e4m3fn)
    values=torch.randint(0,256,raw.shape,dtype=torch.uint8,generator=g)
    target=torch.zeros(17,32,4,256).to(torch.float8_e4m3fn)
    target_values=torch.zeros(17,32,4,256,dtype=torch.uint8)
    sums=torch.randn(16,4,256,generator=g)
    expected=sums+raw[:16,1:1+count].float().sum(1)
    _native.scatter_history_batch([(raw.view(torch.uint8).numpy(),values.numpy(),
        target.view(torch.uint8).numpy(),target_values.numpy(),None,sums.numpy(),1,13,count)])
    assert torch.equal(sums,expected)
    assert torch.equal(target[:,13:13+count].view(torch.uint8),raw[:,1:1+count].view(torch.uint8))
    assert torch.equal(target_values[:,13:13+count],values[:,1:1+count])


def test_invalid_later_fp8_scatter_does_not_mutate_prior_destination():
    raw=torch.ones(1,4,1,64).to(torch.float8_e4m3fn)
    value=torch.zeros_like(raw,dtype=torch.uint8)
    target=torch.zeros_like(raw);target_v=torch.ones_like(value);sums=torch.zeros(1,1,64)
    job=(raw.view(torch.uint8).numpy(),value.numpy(),target.view(torch.uint8).numpy(),
         target_v.numpy(),None,sums.numpy(),0,0,1)
    before=target.view(torch.uint8).clone()
    with pytest.raises(ValueError):_native.scatter_history_batch([job,(*job[:8],5)])
    assert torch.equal(target.view(torch.uint8),before) and target_v.eq(1).all() and sums.eq(0).all()


def test_fp8_rewind_preserves_sealed_index_and_rejects_old_suffix(tmp_path):
    from vllm_kvmem.authority import _Ready
    a,archive=owner(tmp_path)
    raw=torch.arange(2*12*64).reshape(2,12,1,64).float().remainder(100).to(torch.float8_e4m3fn)
    values=raw.view(torch.uint8).clone()
    try:
        assert a.append(raw[:,:8],values[:,:8],history_begin=0,local_rope_begin=0)
        publish(a); checkpoint=a.memory.checkpoint()
        assert a.append(raw[:,8:],values[:,8:],history_begin=8,local_rope_begin=8)
        publish(a)
        assert a.begin_rewind(checkpoint) is not None
        a.attach_rewind_completion(_Ready());assert a.poll_rewind()
        assert a.history_end==8 and len(a.records)==2
        assert a.append(raw[:,:4],values[:,:4],history_begin=8,local_rope_begin=8)
        publish(a)
        tail=next(r for r in a.records.values() if r.history_begin==8)
        assert torch.equal(a.read_record(tail.handle)[0].view(torch.uint8),raw[:,:4].view(torch.uint8))
        del checkpoint
    finally:
        assert a.close() and archive.close()


def test_live_archive_geometry_charges_fp8_target_and_mtp_without_recurrent_history():
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm_kvmem.cache_manifest import archive_bytes_per_token
    attention=FullAttentionSpec(block_size=1616,num_kv_heads=4,head_size=256,dtype=torch.float8_e4m3fn)
    recurrent=MambaSpec(block_size=262144,shapes=((16,),),dtypes=(torch.float32,))
    cfg=NS(kv_cache_groups=[NS(kv_cache_spec=attention,layer_names=[str(i) for i in range(17)]),
                           NS(kv_cache_spec=recurrent,layer_names=['recurrent'])])
    assert archive_bytes_per_token(cfg)==34816
    assert archive_bytes_per_token(cfg)*262144==9126805504


def test_fp8_raw_cache_accounts_one_byte_per_element():
    from vllm_kvmem.gpu_raw_cache import GpuRawCache
    cache=GpuRawCache(4096,(2,4,1,64),torch.device('cpu'),2,4,dtype=torch.float8_e4m3fn)
    assert cache.record_bytes==512 and cache.index.slots==8 and cache.capacity_bytes==4096


def test_fp8_rope_widens_without_requantizing_before_destination_scale():
    from vllm_kvmem.codec import neox_rope,materialize_k,KVFormat
    raw=torch.tensor([[[.3125,1.75,3.,-2.5]]]).to(torch.float8_e4m3fn)
    cos=torch.tensor([[.73,.81]]);sin=torch.tensor([[.68,.59]])
    expected=torch.cat((raw.float()[...,:2]*cos[:,None]-raw.float()[...,2:]*sin[:,None],
                        raw.float()[...,:2]*sin[:,None]+raw.float()[...,2:]*cos[:,None]),dim=-1)
    actual=neox_rope(raw,cos,sin)
    assert actual.dtype==torch.float32 and torch.equal(actual,expected)
    fmt=KVFormat(torch.float8_e4m3fn,4,1,4)
    output=materialize_k(raw,cos,sin,fmt,torch.tensor(.031))
    assert torch.equal(output.view(torch.uint8),(expected/.031).clamp(-448,448).to(fmt.dtype).view(torch.uint8))


@pytest.mark.parametrize(('resident','pressure'),[(False,False),(True,False),(False,True)])
@pytest.mark.parametrize('corrupt',[False,True])
def test_restore_checks_cold_fp8_oracle_and_preserves_hot_native_oracle(
        tmp_path,monkeypatch,resident,pressure,corrupt):
    import json
    from vllm_kvmem.session_worker import SessionWorker
    from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool
    monkeypatch.setenv('KVMEM_VERIFY_SESSION_RESTORE','1')
    worker=SessionWorker.__new__(SessionWorker)
    # The synchronous fixture has no outstanding model-copy event; use the real
    # readiness interface without bypassing any KV/recurrent byte comparison.
    worker.pool=PromptCheckpointPool(capacity=1,max_payload_bytes=1)
    worker.runner=NS(_kvmem_probe_output=tmp_path,_kvmem_mtp_modules={'mtp':None})
    working_tokens=3 if pressure else 4
    worker.bindings={'a':NS(namespace=('user','session'),pressure_restore=pressure,history=11,window=working_tokens)}
    worker.runner._kvmem_capture_runtime=NS(heads=1,dim=2,names=('target','mtp'))
    native=torch.arange(32,dtype=torch.uint8)
    rebuilt=(native+20)[:working_tokens*8]
    worker.restore_oracles={('user','session'):(11,native)}
    calls=[]
    worker._rebuilt_target_bytes=lambda *args:(calls.append('rebuild') or rebuilt)
    actual=(native if resident else rebuilt).clone()
    if corrupt:actual[0]^=1
    def target_bytes(request_id,count):
        assert count==working_tokens
        return actual
    worker._target_bytes=target_bytes
    recurrent=torch.arange(3)
    worker.model_adapter=NS(_recurrent_views=lambda *args:{'r':recurrent},_draft_views=lambda *args:{})
    checkpoint=NS(position=11,kind='M',window=NS(window_tokens=4),
        authority=NS(raw_dtype=torch.float8_e4m3fn),
        model=NS(recurrent={'r':recurrent.clone()},draft={},draft_begin=0))
    expected_kind='native_terminal_window' if resident else 'fp8_immutable_rebuild'
    if corrupt:
        with pytest.raises(RuntimeError,match=expected_kind):
            worker._verify_restored('a',checkpoint,resident)
        assert not (tmp_path/'verified_restores.jsonl').exists()
    else:
        worker._verify_restored('a',checkpoint,resident)
        report=json.loads((tmp_path/'verified_restores.jsonl').read_text())
        assert report['oracle_kind']==expected_kind and report['byte_equal']
        assert report['working_tokens']==working_tokens
    assert calls==([] if resident else ['rebuild'])
