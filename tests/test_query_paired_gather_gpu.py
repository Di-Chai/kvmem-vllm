"""Final Query CPU gather must preserve every installed and untouched KV byte."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('layer_index,tail,page',
                         [(0,32,64),(8,32,1616),(15,32,64),
                          (0,17,1616),(8,17,64),(15,17,1616)])
def test_query_native_paired_gather_keeps_full_gpu_cache(tmp_path, monkeypatch,
                                                        layer_index, tail, page):
    from vllm_kvmem import _native
    from vllm_kvmem.authority import record_key
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    from vllm_kvmem.memory import Phase
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.window_assembly import NativeWindowAssembler, LayerWindow, _AssemblyStaging

    assert callable(getattr(_native, 'gather_record_regions', None))
    torch.manual_seed(571)
    count,layers,block,heads,dim = 265,17,32,4,256
    row_bytes = block*heads*dim
    archive = MemoryArchive(str(tmp_path/'archive'), layers*row_bytes*2, count+2,
                            gather_workers=8)
    repo = _native.Repository(count+2, block)
    leases = [repo.reserve(i*block,tail if i==count-1 else block).commit() for i in range(count)]
    tensors,records,chosen = {},{},[]
    for i,lease in enumerate(leases):
        raw = torch.randn((layers,block,heads,dim)).to(torch.float8_e4m3fn)
        values = torch.randint(0,256,(layers,block,heads,dim),dtype=torch.uint8).view(torch.float8_e4m3fn)
        tensors[lease.handle.slot] = raw,values
        n = tail if i==count-1 else block
        record = NS(handle=lease.handle,history_begin=i*block,count=n,archived=n==block)
        records[record_key(lease.handle)] = record
        chosen.append(record)
        if n==block:
            archive.write_from_views('r',[lease],[raw.view(torch.uint8).numpy(),
                                                   values.view(torch.uint8).numpy()]).result(5)
    total = sum(r.count for r in chosen)
    pages = (total+page-1)//page+3
    table = torch.randperm(pages-1,device='cuda').to(torch.int32)+1
    angles = torch.randn((total,32),device='cuda')
    trig = torch.cat((angles.cos(),angles.sin()),dim=1)
    scale = torch.tensor(.04052734375,device='cuda')
    device = torch.device('cuda',torch.cuda.current_device())
    authority = NS(raw_element_bytes=1,shape=(layers,block,heads,dim),
        record_bytes=layers*row_bytes*2,records=records,archive=archive,poll=lambda:None,
        memory=NS(request_id='r',admitted_read=lambda h:(repo.pin(h),None),phase=Phase.READY),
        read_records=lambda handles:[tensors[h.slot] for h in handles])
    current = {}
    def factory(cls,runner,rid,names,block_tokens,*,chunk_blocks,arena):
        assert names==(f'layer{layer_index}',) and chunk_blocks==256 and arena=='layerwise'
        return cls((current['layer'],),block_tokens,chunk_blocks,staging=current['pool'],
                   raw_dtype=torch.float8_e4m3fn,value_capacity_bytes=total*heads*dim)
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',classmethod(factory))
    outputs,stats,pools = [],[],[]
    try:
        for enabled in (False,True):
            cache = torch.full((pages,page,heads,2,dim),73,dtype=torch.uint8,device=device)
            view = cache.view(torch.float8_e4m3fn)
            current['layer'] = LayerWindow(view[:,:,:,0],view[:,:,:,1],trig,scale,table)
            current['pool'] = _AssemblyStaging(2*row_bytes*256,device,total*heads*dim)
            pools.append(current['pool'])
            query = LayerwiseQuery.__new__(LayerwiseQuery)
            query.names = tuple(f'layer{i}' for i in range(16))
            query.runner = NS(device=device,_kvmem_capture_runtime=NS(options=NS(assembly_chunk_blocks=32)))
            query.batched_reads = query.packed_assembly = query.paired_reads = True
            query.paired_gather = enabled
            state = dict(authority=authority,request_id='r',selected_tokens=total,
                         next_layer=layer_index,span=(total,total+1))
            stats.append(query.assemble_layer_pinned(state,layer_index,chosen))
            torch.cuda.synchronize()
            outputs.append(cache.cpu())
        assert torch.equal(*outputs)
        assert all(s['uploaded_bytes']==total*heads*dim*2 and s['chunks']==2 for s in stats)
        assert all(s['inline_read_chunks']==0 and s['prepared_read_chunks']==0 for s in stats)
        assert stats[0]['native_paired_gather_chunks']==0
        assert stats[1]['native_paired_gather_chunks']>0
        assert archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        torch.cuda.synchronize()
        for pool in pools: assert pool.close()
        assert archive.close()
