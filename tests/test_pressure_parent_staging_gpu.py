"""Byte-exact native pressure transaction versus its original child pools."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',
                               reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('page', [64, 1616])
def test_parent_pool_preserves_all_layer_cache_bytes_and_source_leases(tmp_path,monkeypatch,page):
    from vllm_kvmem import _native
    from vllm_kvmem.serving.authority import record_key
    from vllm_kvmem.serving.layerwise_pressure_efficiency import install_single_assembly
    from vllm_kvmem.serving.layerwise_query import LayerwiseQuery
    from vllm_kvmem.core.memory import Phase
    from vllm_kvmem.core.memory_storage import MemoryArchive
    from vllm_kvmem.serving.window_assembly import LayerWindow,NativeWindowAssembler

    monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','0')
    torch.manual_seed(7921)
    count, history_count, layers, block, heads, dim = 601, 720, 17, 32, 4, 256
    total, row_bytes = count*block, block*heads*dim
    names=tuple(f'layer{i}' for i in range(layers))
    repository=_native.Repository(history_count+2,block)
    archive=MemoryArchive(str(tmp_path/'archive'),layers*row_bytes*2,history_count+2,gather_workers=8)
    records=[];leases=[]
    # Finite raw FP8 K, while V covers every byte including NaN payloads. The
    # test compares bytes rather than FP values, including untouched pages.
    raw_template=torch.randint(16,127,(layers,block,heads,dim),dtype=torch.uint8)
    value_template=torch.randint(0,256,(layers,block,heads,dim),dtype=torch.uint8)
    for i in range(history_count):
        lease=repository.reserve(i*block,block).commit();leases.append(lease)
        records.append(NS(handle=lease.handle,history_begin=i*block,count=block,archived=True))
        raw=(raw_template^(i%16)).view(torch.float8_e4m3fn)
        value=(value_template^(i%256)).view(torch.float8_e4m3fn)
        archive.write_from_views('archive-owner',[lease],
            [raw.view(torch.uint8).numpy(),value.view(torch.uint8).numpy()]).result(10)
    chosen=[records[i*7:i*7+count] for i in range(16)]
    skip={records[i].history_begin for i in (0,1,257,545)}
    plan=NS(total_window_tokens=total,remaps=[NS(block_id=r.handle.slot,to_base=i*32,
        n_tokens=32,skip=r.history_begin in skip) for i,r in enumerate(chosen[0])])
    released=[];plan.release_cpu_sources=lambda:released.append(True)
    authority=NS(raw_element_bytes=1,raw_dtype=torch.float8_e4m3fn,
        raw_record_bytes=layers*row_bytes,value_record_bytes=layers*row_bytes,
        shape=(layers,block,heads,dim),history_end=history_count*block,
        records={record_key(r.handle):r for r in records},archive=archive,poll=lambda:None,
        memory=NS(request_id='archive-owner',phase=Phase.ASSEMBLING,_plan=plan,
                  admitted_read=lambda handle:(repository.pin(handle),None)))
    device=torch.device('cuda',torch.cuda.current_device())
    pages=(total+page-1)//page+3
    table=torch.randperm(pages-1,device=device).to(torch.int32)+1
    angle=torch.randn((total,32),device=device)
    trig=torch.cat((angle.cos(),angle.sin()),dim=1)
    scale=torch.tensor(.04052734375,device=device)
    outputs=[];statistics=[];parents=[];runners=[]
    try:
        for enabled,paired,workers in ((False,False,1),(False,True,1),(True,True,1),(False,True,2)):
            monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING',str(int(enabled)))
            monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER',str(int(paired)))
            monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS',str(workers))
            caches=[torch.full((pages,page,heads,2,dim),73,dtype=torch.uint8,device=device)
                    for _ in range(layers)]
            views=[cache.view(torch.float8_e4m3fn) for cache in caches]
            windows=tuple(LayerWindow(v[:,:,:,0],v[:,:,:,1],trig,scale,table) for v in views)
            runner=NS(device=device,_kvmem_capture_runtime=NS(
                authorities={'restored-native-owner':authority},
                options=NS(assembly_chunk_blocks=32,reuse_resident_v=False)))
            runners.append(runner)
            query=LayerwiseQuery.__new__(LayerwiseQuery)
            query.names=names[:16];query.runner=runner
            query.packed_assembly=query.paired_reads=query.batched_reads=True
            query.resident_values=query.lazy_reader=False
            controller=NS(runner=runner,names=names[:16],query=query,
                          prepared_reads=True,reuse_layer_views=True,resident_values=False)
            parent=NativeWindowAssembler(windows,32,32,raw_dtype=torch.float8_e4m3fn)
            parent._runner_binding=(runner,'restored-native-owner')
            parents.append(parent)
            job=NS(command=NS(request_id='restored-native-owner',pressure_stage='prefill'),
                   layerwise_pressure_plan=(chosen,[]))
            install_single_assembly(parent,controller,job,names)
            before=repository.stats['leases']
            result=parent.assemble(authority,plan)
            result.event.synchronize()
            assert result.query() and job.layerwise_single_assembly_done
            assert repository.stats['leases']==before
            assert archive._prepared_region_readers==0
            assert archive.io_usage()['archive_io_pending_bytes']==0
            assert parent.staging.owner is None
            outputs.append([cache.cpu() for cache in caches])
            statistics.append(result)
            if paired:
                assert result.read_stats['pressure_paired_gather_chunks']==result.chunks
                assert result.read_stats['pressure_paired_fallback_chunks']==0
                assert result.read_stats['pressure_paired_copied_bytes']==result.uploaded_bytes
                assert result.read_stats['pressure_paired_copy_seconds']>0
                assert result.read_stats['layerwise_native_paired_gather_chunks']==result.chunks
                assert result.read_stats['pressure_paired_workers_requested']==workers
                assert result.read_stats['pressure_paired_gather_wall_seconds']>0
                assert result.read_stats['pressure_paired_native_timing_scope']=='sum_of_native_calls_not_elapsed_wall'
            else:
                assert 'pressure_paired_gather_requested' not in result.read_stats
            if enabled:
                assert not hasattr(runner,'_kvmem_pressure_assembly_staging')
                assert result.read_stats['layerwise_parent_staging']
                assert result.read_stats['layerwise_parent_chunk_blocks']==544
                assert parent.staging.slot_bytes==17*32*row_bytes*2
                assert result.read_stats['parent_staging_cpu_fill_seconds']>0
                assert result.read_stats['parent_staging_cache_copy_seconds']>0
            else:
                assert runner._kvmem_pressure_assembly_staging.slot_bytes==256*row_bytes*2
                assert not parent.staging.buffers
            assert parent.close()
        assert all(torch.equal(outputs[0][layer],output[layer]) for output in outputs[1:] for layer in range(layers))
        assert all(stat.uploaded_bytes==(17*count-8)*2*row_bytes for stat in statistics)
        assert statistics[0].chunks==statistics[1].chunks
        assert statistics[2].chunks < statistics[0].chunks
        assert (statistics[0].rebuilt_tokens,statistics[0].skipped_tokens)==((count-4)*32,4*32)
        assert len(released)==4
        # Every skipped layer-0/MTP destination is still the original byte73.
        for layer in (0,16):
            for i in (0,1,257,545):
                positions=torch.arange(i*32,(i+1)*32)
                physical=table.cpu()[positions//page].long()
                assert torch.all(outputs[1][layer][physical,positions%page]==73)
    finally:
        torch.cuda.synchronize()
        for parent in parents:parent.close()
        for runner in runners:
            child=getattr(runner,'_kvmem_pressure_assembly_staging',None)
            if child is not None:assert child.close()
        assert archive.close()
