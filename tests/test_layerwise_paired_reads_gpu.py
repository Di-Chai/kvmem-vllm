"""Exercise actual pinned reads, DMA and KV assembly against the old two reads."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                             reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('layer_index',[0,8,16])
@pytest.mark.parametrize('tail',[32,17])
def test_paired_pinned_layer_assembly_preserves_every_gpu_kv_byte(tmp_path,monkeypatch,layer_index,tail):
    from vllm_kvmem import _native
    from vllm_kvmem.authority import record_key
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.window_assembly import NativeWindowAssembler,LayerWindow,_AssemblyStaging
    torch.manual_seed(571)
    count,layers,block,heads,dim=265,17,32,4,256
    row_bytes=block*heads*dim;record_bytes=layers*row_bytes*2
    archive=MemoryArchive(str(tmp_path/'archive'),record_bytes,count+2,gather_workers=8)
    repo=_native.Repository(count+2,block)
    leases=[repo.reserve(i*block,tail if i==count-1 else block).commit() for i in range(count)]
    tensors={};records={};chosen=[]
    for i,lease in enumerate(leases):
        raw=torch.randn((layers,block,heads,dim)).to(torch.float8_e4m3fn)
        values=torch.randint(0,256,(layers,block,heads,dim),dtype=torch.uint8).view(torch.float8_e4m3fn)
        tensors[lease.handle.slot]=(raw,values)
        n=tail if i==count-1 else block
        record=NS(handle=lease.handle,history_begin=i*block,count=n,archived=n==block)
        records[record_key(lease.handle)]=record;chosen.append(record)
        if n==block:
            archive.write_from_views('r',[lease],[raw.view(torch.uint8).numpy(),
                                                  values.view(torch.uint8).numpy()]).result(5)
    # Preserve a skipped native remap and exercise a complete 256-record chunk
    # followed by either a short complete chunk or the original partial path.
    total=sum(r.count for r in chosen);page=64;pages=(total+page-1)//page+3
    table=torch.randperm(pages-1,device='cuda',dtype=torch.int64).to(torch.int32)+1
    angles=torch.randn((total,32),device='cuda');trig=torch.cat((angles.cos(),angles.sin()),dim=1)
    scale=torch.tensor(.04052734375,device='cuda')
    device=torch.device('cuda',torch.cuda.current_device())
    authority=NS(raw_element_bytes=1,shape=(layers,block,heads,dim),record_bytes=record_bytes,
        records=records,archive=archive,poll=lambda:None,
        memory=NS(request_id='r',admitted_read=lambda h:(repo.pin(h),None)),
        read_records=lambda handles:[tensors[h.slot] for h in handles])
    current={};staging=[];outputs=[];statistics=[]
    def factory(cls,runner,rid,names,block_tokens,*,chunk_blocks,arena):
        assert rid=='r' and names==(f'layer{layer_index}',) and block_tokens==32
        assert chunk_blocks==256 and arena=='pressure'
        return cls((current['layer'],),block_tokens,chunk_blocks,
                   staging=current['pool'],raw_dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',classmethod(factory))
    try:
        for paired in (False,True):
            cache=torch.full((pages,page,heads,2,dim),73,dtype=torch.uint8,device=device)
            view=cache.view(torch.float8_e4m3fn)
            current['layer']=LayerWindow(view[:,:,:,0],view[:,:,:,1],trig,scale,table)
            current['pool']=_AssemblyStaging(2*row_bytes*256,device)
            staging.append(current['pool'])
            query=LayerwiseQuery.__new__(LayerwiseQuery)
            query.names=tuple(f'layer{i}' for i in range(16))
            query.runner=NS(device=device,_kvmem_capture_runtime=NS(options=NS(assembly_chunk_blocks=32)))
            query.batched_reads=query.packed_assembly=True;query.paired_reads=paired
            state=dict(authority=authority,request_id='r',selected_tokens=total,assembly_arena='pressure')
            stats=query.assemble_layer_pinned(state,layer_index,chosen,
                layer_name=f'layer{layer_index}',skip_history_begins=(0,))
            torch.cuda.synchronize()
            statistics.append(stats);outputs.append(cache.cpu())
        assert torch.equal(outputs[0],outputs[1])
        assert statistics[0]['paired_read_chunks']==0 and statistics[1]['paired_read_chunks']>0
        assert statistics[0]['uploaded_bytes']==statistics[1]['uploaded_bytes']==(total-32)*heads*dim*2
        assert statistics[0]['chunks']==statistics[1]['chunks']==2
        assert archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        torch.cuda.synchronize()
        for pool in staging:assert pool.close()
        assert archive.close()
