"""Actual Query mixed-V installation versus the original full archive reads."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                            reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('layer_index,tail,page,hit_full',
    [(0,32,64,270),(8,32,1616,193),(15,32,64,270),
     (0,17,1616,193),(8,17,64,270),(15,17,1616,193)])
def test_query_resident_v_preserves_entire_cache_across_mixed_cycles_and_tails(
        tmp_path,monkeypatch,layer_index,tail,page,hit_full):
    from vllm_kvmem import _native
    from vllm_kvmem.authority import record_key
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    from vllm_kvmem.memory import Phase
    from vllm_kvmem.memory_storage import MemoryArchive
    from vllm_kvmem.query_resident_values import bind_query_resident_sources
    from vllm_kvmem.window_assembly import NativeWindowAssembler,LayerWindow,_AssemblyStaging
    from test_query_resident_values import global_assembly

    torch.manual_seed(187)
    # A restored request keeps archive namespace r while its native owner ID
    # changes. Layer 8 cases cover both full and partial restored-prefix paths.
    native_id='warm-request' if layer_index==8 else 'r'
    selected_count,layers,block,heads,dim=321,17,32,4,256
    full_old=selected_count-int(tail!=32)
    missing=full_old-hit_full
    # These four globally present V records are intentionally noncanonical.
    # Two were skipped and two copied from an older native window; the new
    # helper must fetch their canonical V from archive rather than reuse them.
    unsafe_indices=(0,1,2,3);certified_hits=hit_full-len(unsafe_indices)
    history_count=full_old+missing+int(tail!=32)
    partial_id=history_count-1 if tail!=32 else None
    global_indices=list(range(161,full_old))+list(range(161))
    chosen_indices=list(range(hit_full))+list(range(full_old,full_old+missing))
    if partial_id is not None:
        global_indices.append(partial_id);chosen_indices.append(partial_id)
    assert len(global_indices)==len(chosen_indices)==selected_count
    row_bytes=block*heads*dim
    archive=MemoryArchive(str(tmp_path/'archive'),layers*row_bytes*2,history_count+2,gather_workers=8)
    repository=_native.Repository(history_count+2,block)
    leases=[repository.reserve(i*block,tail if i==partial_id else block).commit()
            for i in range(history_count)]
    records=[];tensors={}
    for i,lease in enumerate(leases):
        raw=torch.randn((layers,block,heads,dim)).to(torch.float8_e4m3fn)
        values=torch.randint(0,256,(layers,block,heads,dim),dtype=torch.uint8).view(torch.float8_e4m3fn)
        n=tail if i==partial_id else block
        record=NS(handle=lease.handle,history_begin=i*block,count=n,archived=n==block)
        records.append(record);tensors[lease.handle.slot]=(raw,values)
        if n==block:
            archive.write_from_views('r',[lease],[raw.view(torch.uint8).numpy(),
                                                   values.view(torch.uint8).numpy()]).result(5)
    records=tuple(records)
    unsafe={records[i].handle.slot for i in unsafe_indices}
    chosen=[records[i] for i in chosen_indices]
    total=sum(r.count for r in chosen)
    history=records[-1].history_begin+records[-1].count
    authority=NS(raw_element_bytes=1,raw_dtype=torch.float8_e4m3fn,
        raw_record_bytes=layers*row_bytes,value_record_bytes=layers*row_bytes,
        shape=(layers,block,heads,dim),history_end=history,
        record_bytes=layers*row_bytes*2,records={record_key(r.handle):r for r in records},
        archive=archive,poll=lambda:None,
        memory=NS(request_id='r',phase=Phase.READY,
                  admitted_read=lambda h:(repository.pin(h),None)),
        read_records=lambda handles:[tensors[h.slot] for h in handles])
    working=[];cursor=0
    for i in global_indices:
        r=records[i]
        working.append((r.handle.slot,r.handle.generation,r.history_begin,r.count,cursor))
        cursor+=r.count
    assert cursor==total
    device=torch.device('cuda',torch.cuda.current_device())
    pages=(total+page-1)//page+3
    table=torch.randperm(pages-1,device=device).to(torch.int32)+1
    angles=torch.randn((total,32),device=device)
    trig=torch.cat((angles.cos(),angles.sin()),dim=1)
    scale=torch.tensor(.04052734375,device=device)
    current={}
    def factory(cls,runner,rid,names,block_tokens,*,chunk_blocks,arena):
        assert rid==native_id and names==(f'layer{layer_index}',) and arena=='layerwise' and chunk_blocks==256
        return cls((current['layer'],),block_tokens,chunk_blocks,staging=current['pool'],
                   raw_dtype=torch.float8_e4m3fn,value_capacity_bytes=total*heads*dim)
    monkeypatch.setattr(NativeWindowAssembler,'from_runner',classmethod(factory))
    outputs,statistics,pools,states=[],[],[],[]
    variants=((False,False),(True,False),(False,True),(True,True))
    queries=[]
    try:
        for enabled,lazy in variants:
            cache=torch.full((pages,page,heads,2,dim),73,dtype=torch.uint8,device=device)
            view=cache.view(torch.float8_e4m3fn)
            for slot,_,_,n,base in working:
                position=torch.arange(base,base+n,device=device)
                physical=table[position//page].long()
                initial=tensors[slot][1][layer_index,:n].view(torch.uint8).cuda()
                if slot in unsafe:initial=initial^25
                view[:,:,:,1].view(torch.uint8)[physical,position%page]=initial
            current['layer']=LayerWindow(view[:,:,:,0],view[:,:,:,1],trig,scale,table)
            current['pool']=_AssemblyStaging(2*row_bytes*256,device,total*heads*dim)
            pools.append(current['pool'])
            query=LayerwiseQuery.__new__(LayerwiseQuery)
            query.names=tuple(f'layer{i}' for i in range(16))
            query.runner=NS(device=device,_kvmem_request_generations={native_id:7},
                _kvmem_capture_runtime=NS(authorities={native_id:authority},
                    options=NS(assembly_chunk_blocks=32,reuse_resident_v=True)))
            query.batched_reads=query.packed_assembly=query.paired_reads=True
            query.paired_gather=query.no_probe=False
            query.resident_values=enabled;query.lazy_reader=lazy;query.mtp_selection='global'
            queries.append(query)
            state=dict(authority=authority,request_id=native_id,records=records,history=history,
                selected_tokens=total,global_slots={records[i].handle.slot for i in global_indices},
                next_layer=layer_index,span=(history,history+1))
            states.append(state)
            if enabled:
                proof=global_assembly(query,state,working,
                    noncanonical=tuple(records[i].handle.slot for i in (2,3)),
                    skip=tuple(records[i].handle.slot for i in (0,1)))
                bind_query_resident_sources(query,state,working,global_assembly=proof)
            before_leases=repository.stats['leases']
            statistics.append(query.assemble_layer_pinned(state,layer_index,chosen))
            torch.cuda.synchronize()
            outputs.append(cache.cpu())
            reader=state.get('query_region_reader')
            if lazy:
                # The legacy partial-tail group is read as a whole through
                # read_records; only preceding full groups enter the cache.
                cached_blocks=selected_count if tail==32 else 256
                assert reader is query._query_reader(state)
                assert not reader.closed and reader.source_blocks==cached_blocks
                assert reader.read_chunks==statistics[-1]['prepared_read_chunks']
                assert statistics[-1]['prepared_read_chunks']>0
                assert repository.stats['leases']==before_leases+cached_blocks
                assert archive._prepared_region_readers==1
            else:
                assert reader is None and statistics[-1]['prepared_read_chunks']==0
            # Source owners and metadata are retired through the production
            # final-Query hook only after this variant's native GPU fence.
            query._close_query_resources(state)
            if lazy:
                assert reader.closed and not reader.sources and not reader.by_handle
                assert 'query_region_reader' not in state
            assert not state.get('query_resident_metadata_owners')
            assert repository.stats['leases']==before_leases
            assert getattr(archive,'_prepared_region_readers',0)==0
            assert archive.io_usage()['archive_io_pending_bytes']==0
        assert all(torch.equal(outputs[0],result) for result in outputs[1:])
        assert statistics[0]['uploaded_bytes']==total*heads*dim*2
        assert statistics[2]['uploaded_bytes']==statistics[0]['uploaded_bytes']
        if tail==32:
            for index in (1,3):
                result=statistics[index]
                assert result['query_resident_v'] and result['query_resident_fallback_reason'] is None
                assert result['resident_v_reused_tokens']==certified_hits*32
                assert result['resident_v_missing_tokens']==(missing+len(unsafe))*32
                assert result['resident_v_snapshot_bytes']==certified_hits*row_bytes
                assert result['resident_v_avoided_archive_bytes']==certified_hits*row_bytes
                assert result['resident_v_uncertified_tokens']==len(unsafe)*32
                assert result['query_resident_source_scope']=='initial_global_archive_uploaded_only'
                assert result['uploaded_bytes']==(2*selected_count-certified_hits)*row_bytes
                assert result['chunks']==(certified_hits+255)//256+(missing+len(unsafe)+255)//256
                assert result['resident_v_raw_only_chunks']==(certified_hits+255)//256
            cached=statistics[3]['assembly_read_stats']
            assert cached['query_reader_new_blocks']==selected_count
            assert cached['query_reader_source_blocks']==selected_count
            assert cached['query_reader_reused_blocks']==0
            assert cached['logical_bytes']==statistics[3]['uploaded_bytes']
        else:
            for index in (1,3):
                assert not statistics[index]['query_resident_v']
                assert statistics[index]['query_resident_fallback_reason']=='partial_selected_record'
                assert statistics[index]['uploaded_bytes']==statistics[0]['uploaded_bytes']
        assert archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        torch.cuda.synchronize()
        for query,state in zip(queries,states):query._close_query_resources(state)
        states.clear()
        for pool in pools:assert pool.close()
        assert archive.close()
