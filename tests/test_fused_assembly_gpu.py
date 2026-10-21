"""Explicit CUDA byte oracles for multi-layer reconstruction, not throughput."""
import math
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')


@pytest.mark.parametrize('raw_dtype,table_dtype,trig_dtype,padded', [
    (torch.float8_e4m3fn,torch.int32,torch.float32,False),
    (torch.float8_e4m3fn,torch.int64,torch.float32,True),
    (torch.float8_e4m3fn,torch.int32,torch.float16,True),
    (torch.float8_e4m3fn,torch.int64,torch.bfloat16,False),
    (torch.bfloat16,torch.int64,torch.bfloat16,True),
    (torch.bfloat16,torch.int32,torch.float32,False),
    (torch.float16,torch.int32,torch.float16,True),
    (torch.float16,torch.int64,torch.float32,False),
])
def test_fused_kernel_preserves_all_native_bytes(raw_dtype,table_dtype,trig_dtype,padded):
    from vllm_kvmem.split_assembly import (assemble_split_layer,assemble_split_layers,
                                         prepare_fused_layer_rows)
    from vllm_kvmem.window_assembly import LayerWindow
    torch.manual_seed(12791)
    records,layers,block,heads,dim,page = 5,3,32,2,80,16
    shape = (records,layers,block,heads,dim)
    if raw_dtype == torch.float8_e4m3fn:
        # All encodings include signed zero, finite extrema and both NaNs.
        raw = (torch.arange(math.prod(shape),device='cuda')%256).to(torch.uint8).view(raw_dtype).reshape(shape)
    else:
        raw = torch.randn(shape,device='cuda').to(raw_dtype)
    values = torch.randint(0,256,shape,device='cuda',dtype=torch.uint8)
    ri = torch.tensor([4,0,2],device='cuda')
    vi = torch.tensor([1,3,0],device='cuda')
    bases = torch.tensor([3,39,73],device='cuda')
    frames,backings = [[],[]],[[],[]]
    for layer in range(layers):
        capacity = 12+layer
        padding = 2*layer if padded else 0
        width = 2*dim+padding
        backing = torch.full((capacity,heads,page,width),197,device='cuda',dtype=torch.uint8)
        table = (torch.randperm(8,device='cuda')+1).to(table_dtype)
        angles = torch.randn(128,16,device='cuda')
        trig = torch.cat((angles.cos(),angles.sin(),angles[:,:padding]),dim=-1).to(trig_dtype)[:,:32]
        scale = torch.tensor(.043701171875+layer*.01025390625,device='cuda')
        for arm in range(2):
            tensor = backing.clone()
            stride = (heads*page*width,width,page*width,1)
            key = tensor.as_strided((capacity,page,heads,dim),stride,0).view(torch.float8_e4m3fn)
            value = tensor.as_strided((capacity,page,heads,dim),stride,dim).view(torch.float8_e4m3fn)
            frames[arm].append(LayerWindow(key,value,trig,scale,table))
            backings[arm].append(tensor)
    spec,reason = prepare_fused_layer_rows(frames[1])
    assert reason is None
    ptrs = torch.tensor(spec.rows,device='cuda',dtype=torch.int64)
    kwargs = dict(layers=layers,block_tokens=block,raw_record_bytes=raw[0].numel()*raw.element_size(),
                  value_record_bytes=values[0].numel(),raw_dtype=raw_dtype)
    for i,layer in enumerate(frames[0]):
        assemble_split_layer(raw.view(torch.uint8).flatten(),ri,values.flatten(),vi,bases,layer,
                             layer_index=i,**kwargs)
    assemble_split_layers(raw.view(torch.uint8).flatten(),ri,values.flatten(),vi,bases,ptrs,spec.geometry,**kwargs)
    torch.cuda.synchronize()
    assert all(torch.equal(a,b) for a,b in zip(*backings))


@pytest.mark.parametrize('tail,prefetch',[(0,False),(0,True),(7,False),(7,True)])
def test_full_assembly_matches_resident_and_restore_bytes(tmp_path,tail,prefetch):
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.window_assembly import LayerWindow,NativeWindowAssembler
    block,layers,heads,dim,page = 32,3,2,64,16
    count = 600*block+tail
    select_pages = 300+bool(tail)
    config = KVMemConfig(block_tokens=block,select_budget=select_pages*block,gen_budget=302*block,
                         repository_blocks=1024,sink_blocks=0,recent_blocks=0)
    archive = MemoryArchive(str(tmp_path),layers*block*heads*dim*2,1024,cpu_direct_record_seal=True)
    owner = RequestAuthority('fused-gpu',config,archive,layers,heads,dim,
                             raw_dtype=torch.float8_e4m3fn,resident_records=640)
    torch.manual_seed(3010971)
    raw = torch.randn(layers,count,heads,dim).to(torch.float8_e4m3fn)
    encoded = torch.randint(0,256,raw.shape,dtype=torch.uint8)
    assert owner.append(raw,encoded,history_begin=0,local_rope_begin=0)
    owner.publish_sealed(lambda: archive.wait_for_capacity(.05))
    page_count = math.ceil(count/page)
    table = (torch.randperm(page_count,device='cuda')+1).to(torch.int32)
    angles = torch.randn((page_count+1)*page,16,device='cuda')
    trig = torch.cat((angles.cos(),angles.sin()),dim=-1)
    positions = torch.arange(count,device='cuda')
    slots = table[positions//page].long()*page+positions%page
    frames = [[],[]]
    for index in range(layers):
        backing = torch.zeros((page_count+1,heads,page,2*dim),dtype=torch.uint8,device='cuda')
        key,value = native_vllm_fp8_views(backing,heads,dim)
        scale = torch.tensor(.04052734375+index*.00390625,device='cuda')
        assemble_fp8(raw[index].cuda(),encoded[index].view(torch.float8_e4m3fn).cuda(),
                     positions,slots,trig,key,value,scale)
        other_key,other_value = native_vllm_fp8_views(backing.clone(),heads,dim)
        frames[0].append(LayerWindow(key,value,trig,scale,table))
        frames[1].append(LayerWindow(other_key,other_value,trig,scale,table))
    chosen = set(range(1,600,2))|({600} if tail else set())
    scores = {r.handle.slot: float(r.history_begin//block in chosen) for r in owner.records.values()}
    plan = owner.memory.select(scores,action=False)
    selection = tuple((r.block_id,r.n_tokens,r.to_base) for r in plan.remaps)
    pools,pointers = [None,None],[None,None]
    for restoring in (False,True):
        if restoring:
            checkpoint = owner.memory.checkpoint_window(allow_partial=True)
            plan = owner.memory.restore_window(checkpoint)
            for layer in frames[0]+frames[1]:
                layer.kernel_page_table = table.flip(0).contiguous()
                layer.key.view(torch.uint8).fill_(199)
                layer.value.view(torch.uint8).fill_(199)
        results,assemblers = [],[]
        for arm in range(2):
            assembler = NativeWindowAssembler(tuple(frames[arm]),block,chunk_blocks=128,packed=True,
                direct_pinned_reads=True,reuse_resident_v=True,raw_dtype=torch.float8_e4m3fn,
                read_prefetch=prefetch,fused_layers=bool(arm),staging=pools[arm])
            if pools[arm] is None:
                pools[arm] = assembler.staging
            assembler._shared_staging = True
            result = assembler.assemble(owner,plan)
            result.event.synchronize()
            results.append(result)
            assemblers.append(assembler)
            addresses = tuple(b.data_ptr() for b in assembler.transfers.buffers+assembler.staging.buffers)
            if pointers[arm] is not None:
                assert addresses == pointers[arm]
            pointers[arm] = addresses
            assert len(set(addresses)) == 4
        assert tuple((r.block_id,r.n_tokens,r.to_base) for r in plan.remaps) == selection
        assert all(torch.equal(a.key.view(torch.uint8),b.key.view(torch.uint8)) and
                   torch.equal(a.value.view(torch.uint8),b.value.view(torch.uint8)) for a,b in zip(*frames))
        baseline,result = results
        assert baseline.rebuilt_tokens == result.rebuilt_tokens == 300*block+tail
        assert baseline.uploaded_bytes == result.uploaded_bytes
        assert baseline.chunks == result.chunks == 3+bool(tail)
        assert result.read_stats['assembly_fused_chunks'] == (0 if tail else 3)
        assert result.read_stats['assembly_fused_layer_launches'] == (0 if tail else 9)
        assert result.read_stats['assembly_fused_metadata_bytes'] == (0 if tail else layers*12*8*2)
        if tail:
            assert result.read_stats['assembly_fused_fallback:partial'] == 3
            assert result.read_stats['assembly_fused_fallback:generic'] == 1
        if prefetch:
            assert result.read_stats['assembly_prefetch_submitted'] == result.read_stats['assembly_prefetch_completed'] == 2
            assert result.read_stats['assembly_prefetch_abandoned'] == 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        owner.memory.attach_completion(result)
        assert owner.memory.commit_selection()
        assert all(assembler.close() for assembler in assemblers)
        assert not pools[1].fused_owners
        del plan,result,baseline,results
        if restoring:
            del checkpoint
    assert all(pool.close() for pool in pools)
    assert owner.close() and archive.close()
