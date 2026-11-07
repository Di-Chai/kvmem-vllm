"""Exercise actual pinned reads, DMA and KV assembly against the old two reads."""
import os
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                             reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('layer_index',[0,8,16])
@pytest.mark.parametrize('tail',[32,17])
@pytest.mark.parametrize('page',[64,1616])
@pytest.mark.parametrize('tiled',[False,True])
def test_paired_pinned_layer_assembly_preserves_every_gpu_kv_byte(tmp_path,monkeypatch,layer_index,tail,page,tiled):
    from vllm_kvmem import _native
    from vllm_kvmem.authority import record_key
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem import packed_assembly
    from vllm_kvmem import split_assembly
    from vllm_kvmem.memory import Phase
    from vllm_kvmem.pressure_reader import prepare_pressure_reader
    from vllm_kvmem.window_assembly import NativeWindowAssembler,LayerWindow,_AssemblyStaging
    gather = split_assembly.gather_resident_values
    def selected_gather(*args, **kwargs):
        kwargs['tiled'] = tiled
        return gather(*args, **kwargs)
    monkeypatch.setattr(split_assembly, 'gather_resident_values', selected_gather)
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
    total=sum(r.count for r in chosen);pages=(total+page-1)//page+3
    table=torch.randperm(pages-1,device='cuda',dtype=torch.int64).to(torch.int32)+1
    angles=torch.randn((total,32),device='cuda');trig=torch.cat((angles.cos(),angles.sin()),dim=1)
    scale=torch.tensor(.04052734375,device='cuda')
    device=torch.device('cuda',torch.cuda.current_device())
    authority=NS(raw_element_bytes=1,shape=(layers,block,heads,dim),record_bytes=record_bytes,
        records=records,archive=archive,poll=lambda:None,history_end=total,
        memory=NS(request_id='r',admitted_read=lambda h:(repo.pin(h),None),phase=Phase.ASSEMBLING,_plan=object()),
        read_records=lambda handles:[tensors[h.slot] for h in handles])
    current={};staging=[];outputs=[];statistics=[]
    def factory(cls,runner,rid,names,block_tokens,*,chunk_blocks,arena):
        assert rid=='r' and names==(f'layer{layer_index}',) and block_tokens==32
        assert chunk_blocks==256 and arena=='pressure'
        return cls((current['layer'],),block_tokens,chunk_blocks,
                   staging=current['pool'],raw_dtype=torch.float8_e4m3fn,
                   value_capacity_bytes=total*heads*dim)
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',classmethod(factory))
    read_packed = packed_assembly.read_packed_records
    inline_enabled = False
    def read(*args, **kwargs):
        kwargs['inline_cpu'] = bool(kwargs.get('inline_cpu') and inline_enabled)
        return read_packed(*args, **kwargs)
    monkeypatch.setattr(packed_assembly, 'read_packed_records', read)
    try:
        for paired, inline_enabled, prepared, borrowed, resident in (
                (False,False,False,False,False),(True,False,False,False,False),
                (True,True,False,False,False),(True,True,True,False,False),
                (True,True,False,True,False),(True,True,False,False,True),
                (True,True,False,True,True)):
            cache=torch.full((pages,page,heads,2,dim),73,dtype=torch.uint8,device=device)
            view=cache.view(torch.float8_e4m3fn)
            # Nontrivial resident source cycles cross the 256-record chunk
            # boundary. All variants start with the same original V bytes.
            permutation=[0]+list(range(2,count-1))+[1,count-1]
            mapping=[]
            for i,record in enumerate(chosen):
                base=permutation[i]*32;mapping.append((-1,-1,record.history_begin,record.count,base))
                pos=torch.arange(base,base+record.count,device=device)
                physical=table[pos//page].long()
                view[:,:,:,1].view(torch.uint8)[physical,pos%page]=tensors[record.handle.slot][1][layer_index,:record.count].view(torch.uint8).cuda()
            mapping=tuple(sorted(mapping,key=lambda r:r[4]))
            current['layer']=LayerWindow(view[:,:,:,0],view[:,:,:,1],trig,scale,table)
            current['pool']=_AssemblyStaging(2*row_bytes*256,device,total*heads*dim)
            staging.append(current['pool'])
            query=LayerwiseQuery.__new__(LayerwiseQuery)
            query.names=tuple(f'layer{i}' for i in range(16))
            query.runner=NS(device=device,_kvmem_capture_runtime=NS(options=NS(
                assembly_chunk_blocks=32,reuse_resident_v=True,memory=NS(select_budget=total))))
            query.batched_reads=query.packed_assembly=True;query.paired_reads=paired
            state=dict(authority=authority,request_id='r',selected_tokens=total,
                       assembly_arena='pressure',pressure_stage='prefill')
            if resident:
                state['pressure_value_sources']=('r',total,total,(mapping,)*16)
                state['pressure_value_metadata_owners']=[]
            parent=None
            if borrowed:
                views=[LayerWindow(view[:,:,:,0],view[:,:,:,1],trig,scale,table) for _ in range(layers)]
                parent=NativeWindowAssembler(tuple(views),32,32,raw_dtype=torch.float8_e4m3fn)
                parent._runner_binding=(query.runner,'r');parent.staging.claim(parent)
                query.runner._kvmem_pressure_assembly_staging=current['pool']
                state['pressure_parent_assembler']=parent
            reader=prepare_pressure_reader(authority,authority.memory._plan,[chosen]) if prepared else None
            with (reader if reader is not None else nullcontext()):
                if reader is not None:state['prepared_pressure_reader']=reader
                stats=query.assemble_layer_pinned(state,layer_index,chosen,
                    layer_name=f'layer{layer_index}',skip_history_begins=(0,))
            torch.cuda.synchronize()
            if parent is not None:
                parent.staging.release(parent)
                assert parent.close()
            statistics.append(stats);outputs.append(cache.cpu())
        assert all(torch.equal(outputs[0],output) for output in outputs[1:])
        assert statistics[0]['paired_read_chunks']==0
        assert statistics[1]['paired_read_chunks']==statistics[2]['paired_read_chunks']>0
        assert statistics[0]['inline_read_chunks']==statistics[1]['inline_read_chunks']==0
        assert statistics[2]['inline_read_chunks']==statistics[2]['paired_read_chunks']
        assert all(s['prepared_read_chunks']==0 for s in statistics[:3])
        assert statistics[3]['prepared_read_chunks']==statistics[3]['paired_read_chunks']>0
        assert all(s['uploaded_bytes']==(total-32)*heads*dim*2 and s['chunks']==2 for s in statistics[:5])
        assert statistics[5]['uploaded_bytes']==(total-32)*heads*dim*(1 if tail==32 else 2)
        assert statistics[5].get('resident_v_reused_tokens',0)==(total-32 if tail==32 else 0)
        assert statistics[6]['uploaded_bytes']==statistics[5]['uploaded_bytes']
        assert statistics[6].get('resident_v_reused_tokens',0)==statistics[5].get('resident_v_reused_tokens',0)
        assert archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        torch.cuda.synchronize()
        for pool in staging:assert pool.close()
        assert archive.close()
