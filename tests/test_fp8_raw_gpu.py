"""Byte-exact compact-FP8 reconstruction, partial tails and GPU cache reuse."""
import os

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Explicit GPU qualification only')


@pytest.mark.parametrize('mode',['generic','packed','direct','split'])
@pytest.mark.parametrize('count',[32,35])
def test_fp8_authority_restores_dense_pages_and_reselection_against_cpu_oracle(tmp_path,mode,count):
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.codec import native_vllm_fp8_views,neox_rope
    from vllm_kvmem.gpu_raw_cache import GpuRawCache
    from vllm_kvmem.window_assembly import LayerWindow,NativeWindowAssembler
    cfg=KVMemConfig(block_tokens=16,select_budget=32,gen_budget=32,
                   repository_blocks=12,sink_blocks=0,recent_blocks=0)
    layers,heads,dim=2,2,64
    archive=AsyncArchive(str(tmp_path),layers*16*heads*dim*2,12)
    authority=RequestAuthority('fp8',cfg,archive,layers,heads,dim,raw_dtype=torch.float8_e4m3fn)
    g=torch.Generator().manual_seed(6671)
    raw=torch.randn(layers,count,heads,dim,generator=g).mul(3).to(torch.float8_e4m3fn)
    values=(torch.randn(layers,count,heads,dim,generator=g)/.023).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
    assert authority.append(raw,values,history_begin=0,local_rope_begin=0)
    authority.publish_sealed(lambda:None)
    checkpoint=authority.memory.checkpoint_window(allow_partial=True)
    table=torch.tensor([3,5,1,7,2,6,4,8],dtype=torch.int32,device='cuda')
    angles=torch.randn(128,16,generator=g)
    trig=torch.cat((angles.cos(),angles.sin()),-1).bfloat16()
    scale=torch.tensor(.037,device='cuda')
    native=[]
    for _ in range(layers):
        backing=torch.full((10,heads,8,dim*2),13,dtype=torch.uint8,device='cuda')
        key,value=native_vllm_fp8_views(backing,heads,dim)
        native.append(LayerWindow(key,value,trig.cuda(),scale,table))
    cache=(GpuRawCache(2*layers*16*heads*dim,(layers,16,heads,dim),torch.device('cuda:0'),1,2,
                       dtype=torch.float8_e4m3fn) if mode=='split' else None)
    packed=mode!='generic';direct=mode in ('direct','split')
    try:
        for attempt in range(3):
            if attempt==1:
                records=list(authority.records.values())
                plan=authority.memory.select({r.handle.slot:float(r.history_begin>=16) for r in records})
            else:
                if attempt==2:
                    for layer in native:
                        layer.key.view(torch.uint8).fill_(13);layer.value.view(torch.uint8).fill_(13)
                plan=authority.memory.restore_window(checkpoint)
            assembler=NativeWindowAssembler(tuple(native),16,chunk_blocks=1,packed=packed,
                direct_pinned_reads=direct,raw_cache=cache,reuse_resident_v=mode=='split',
                raw_dtype=torch.float8_e4m3fn)
            result=assembler.assemble(authority,plan);result.event.synchronize()
            if attempt in (0,2):
                assert result.rebuilt_tokens==count and result.skipped_tokens==0
                assert result.partial_tokens==count%16
            if mode=='split' and attempt==2:
                assert result.raw_cache_hit_tokens==32
                assert result.uploaded_bytes==layers*heads*dim*(32+2*(count-32))
            if mode=='split' and attempt==1 and count>32:
                assert result.resident_v_reused_tokens>0
            for layer_index,layer in enumerate(native):
                for remap in plan.remaps:
                    first=authority.records[next(k for k in authority.records if k[0]==remap.block_id)].history_begin
                    positions=torch.arange(remap.to_base,remap.to_base+remap.n_tokens)
                    source=raw[layer_index,first:first+remap.n_tokens]
                    rotated=neox_rope(source,trig[positions,:16],trig[positions,16:])
                    expected=(rotated/.037).clamp(-448,448).to(torch.float8_e4m3fn).view(torch.uint8)
                    p=table[positions.cuda()//8].long();offsets=positions.cuda()%8
                    assert torch.equal(layer.key.view(torch.uint8)[p,offsets].cpu(),expected)
                    assert torch.equal(layer.value.view(torch.uint8)[p,offsets].cpu(),values[layer_index,first:first+remap.n_tokens])
            authority.memory.attach_completion(result);assert authority.memory.commit_selection()
            assert assembler.close()
            # A pybind Remap view retains its parent plan and archive leases.
            del remap,plan,result
    finally:
        del checkpoint
        assert authority.close() and archive.close()


def test_native_capture_quantizes_raw_k_before_d2h_and_preserves_upstream_qkv(tmp_path):
    from types import SimpleNamespace as NS
    from vllm_kvmem.capture_runtime import CaptureRuntime
    import numpy as np
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype=torch.float8_e4m3fn
    runtime.names=('target','mtp');runtime.layers=[None,None];runtime.row_elements=128
    runtime.options=NS(query_history=False);runtime.queue=[];runtime.authorities={};runtime.batches=0
    runtime.scales=torch.tensor([.031,.047],device='cuda')
    runtime.downloads=NS(pending=[],buffers=[1,2],download=lambda data:NS(payload=data.cpu()))
    runtime.drain=lambda **_:True
    captures=[NS(name=name,key=torch.randn(5,1,64,device='cuda',dtype=torch.bfloat16),
                 value=torch.randn(5,1,64,device='cuda',dtype=torch.bfloat16),
                 positions=torch.arange(5,device='cuda')) for name in runtime.names]
    originals=[(c.key.clone(),c.value.clone()) for c in captures]
    batch=NS(num_reqs=1,num_tokens=5,req_ids=['a'],query_start_loc=torch.tensor([0,5],device='cuda'),
             query_start_loc_np=np.array([0,5]),positions=torch.arange(5,device='cuda'),has_prefill=False)
    runtime.submit(captures,batch,torch.tensor([1],device='cuda'))
    data=runtime.queue[0].download.payload
    assert data.numel()==40+2*5*1*64*2
    keys=data[40:40+640].view(torch.float8_e4m3fn).reshape(2,5,1,64)
    expected=torch.stack([c.key for c in captures]).float().clamp(-448,448).to(torch.float8_e4m3fn)
    assert torch.equal(keys.view(torch.uint8),expected.cpu().view(torch.uint8))
    for c,(key,value) in zip(captures,originals):
        assert torch.equal(c.key,key) and torch.equal(c.value,value)
