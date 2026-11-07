"""Pressure-only grouping and ownership without a CUDA/model process."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.memory import Phase
from vllm_kvmem.pressure_parent_staging import prepare_parent_pressure_copy


def fixture(monkeypatch, count=1152):
    monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE', '0')
    names = tuple(f'layer{i}' for i in range(17))
    records = [NS(history_begin=i*32, count=32, archived=True) for i in range(count+16)]
    chosen = [records[i:i+count] for i in range(16)]
    plan = NS(total_window_tokens=count*32)
    authority = NS(raw_element_bytes=1, raw_dtype=torch.float8_e4m3fn,
                   shape=(17,32,4,256), memory=NS(phase=Phase.ASSEMBLING, _plan=plan))
    runner = NS(_kvmem_capture_runtime=NS(authorities={'warm-id': authority}))
    controller = NS(runner=runner, names=names[:16],
                    query=NS(packed_assembly=True, paired_reads=True, batched_reads=True))
    parent = NS(layers=tuple(object() for _ in names), block_tokens=32, chunk_blocks=32,
                raw_dtype=torch.float8_e4m3fn, _runner_binding=(runner,'warm-id'), closed=False)
    parent.staging = NS(owner=parent, slot_bytes=17*32*32768*2)
    parent._validate_borrowed_layers = lambda layers, block, lease: None
    reader = NS(authority=authority, plan=plan, validate=lambda: None)
    state = dict(authority=authority, request_id='warm-id', assembly_arena='pressure',
                 pressure_stage='prefill', selected_tokens=count*32, prepared_pressure_reader=reader)
    return parent, controller, state, plan, chosen, names


def test_existing_parent_capacity_reduces_groups_without_changing_dense_mapping(monkeypatch):
    args = fixture(monkeypatch)
    skip = (0, 32, 257*32, 545*32)
    result = prepare_parent_pressure_copy(*args, skip)
    assert result.chunk_blocks == 544 and result.row_bytes == 32768
    assert len(result.groups) == 17
    for index, groups in enumerate(result.groups):
        records = args[4][index if index<16 else 0]
        observed = [(r.history_begin, cursor+j*32) for cursor,g in groups for j,r in enumerate(g)]
        expected = [(r.history_begin, j*32) for j,r in enumerate(records)
                    if index not in (0,16) or r.history_begin not in skip]
        assert observed == expected
        assert max(map(lambda g:len(g[1]), groups)) <= 544
    assert len(result.groups[1]) == 3  # Original child capacity 256 needs five.


@pytest.mark.parametrize('change', [
    'phase','plan','owner','binding','closed','pool_owner','slot_bytes','dtype',
    'selected_extent','query_route','original_rope','resident_v','reader_owner','reader_plan',
    'borrowed_metadata','decode','query_arena'])
def test_bad_ownership_or_execution_route_fails_before_any_gpu_write(monkeypatch, change):
    parent,c,s,p,chosen,names = fixture(monkeypatch,5)
    if change=='phase':s['authority'].memory.phase=Phase.READY
    if change=='plan':s['authority'].memory._plan=object()
    if change=='owner':c.runner._kvmem_capture_runtime.authorities.clear()
    if change=='binding':parent._runner_binding=(c.runner,'foreign-id')
    if change=='closed':parent.closed=True
    if change=='pool_owner':parent.staging.owner=object()
    if change=='slot_bytes':parent.staging.slot_bytes+=1
    if change=='dtype':parent.raw_dtype=torch.bfloat16
    if change=='selected_extent':s['selected_tokens']+=32
    if change=='query_route':c.query.paired_reads=False
    if change=='original_rope':monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','1')
    if change=='resident_v':c.resident_values=True
    if change=='reader_owner':s['prepared_pressure_reader'].authority=object()
    if change=='reader_plan':s['prepared_pressure_reader'].plan=object()
    if change=='decode':s['pressure_stage']='decode'
    if change=='query_arena':s['assembly_arena']='layerwise'
    if change=='borrowed_metadata':
        def reject(*args):raise ValueError('Native page table changed')
        parent._validate_borrowed_layers=reject
    with pytest.raises(ValueError):prepare_parent_pressure_copy(parent,c,s,p,chosen,names)


@pytest.mark.parametrize('unsupported', ['partial','unpublished','reader'])
def test_unsupported_sources_explicitly_keep_original_path(monkeypatch, unsupported):
    parent,c,s,p,chosen,names = fixture(monkeypatch,5)
    if unsupported=='partial':
        # All target layers have the same partial extent, as required by the
        # original native transaction; this route must fall back before writes.
        for rows in chosen: rows[-1]=NS(history_begin=10000+len(rows), count=17, archived=False)
        s['selected_tokens']=p.total_window_tokens=4*32+17
    if unsupported=='unpublished':chosen[0][0].archived=False
    if unsupported=='reader':s.pop('prepared_pressure_reader')
    assert prepare_parent_pressure_copy(parent,c,s,p,chosen,names) is None
    assert s['pressure_parent_staging_fallback_reason'] == (
        'prepared_reader_unavailable' if unsupported=='reader' else 'partial_or_unpublished_record')


def test_decode_retains_original_rolling_path_with_prefill_candidate_enabled(monkeypatch):
    from vllm_kvmem.layerwise_pressure_efficiency import install_single_assembly
    from vllm_kvmem import window_assembly, pressure_parent_staging
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING','1')
    monkeypatch.setattr(pressure_parent_staging,'assemble_parent_pressure_layers',
                        lambda *a:pytest.fail('Prefill helper entered rolling decode'))
    record=NS(history_begin=0,count=32,handle=NS(slot=0))
    plan=NS(remaps=[NS(block_id=0,to_base=0,n_tokens=32,skip=True)],total_window_tokens=32)
    authority=NS(memory=NS(phase=Phase.ASSEMBLING,_plan=plan))
    calls=[]
    def assemble(state,index,records,**kwargs):
        calls.append((state['pressure_stage'],index,kwargs['skip_history_begins']))
        return dict(uploaded_bytes=64,chunks=1)
    controller=NS(names=('layer',),query=NS(assemble_layer_pinned=assemble),
                  runner=NS(device='cuda'),rolling_decode=True)
    job=NS(command=NS(request_id='r',pressure_stage='decode'),layerwise_pressure_plan=([[record]],[]))
    parent=NS(layers=['layer','draft'])
    monkeypatch.setattr(window_assembly,'_validate_destinations',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'Event',lambda:NS(record=lambda stream:None))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'stream')
    install_single_assembly(parent,controller,job,('layer','draft'))
    result=parent._assemble(authority,plan)
    assert calls == [('decode',0,set()),('decode',1,set())]
    assert result.rebuilt_tokens==32 and result.skipped_tokens==0
    assert 'layerwise_parent_staging' not in result.read_stats


def test_failed_submit_retains_packed_addressing_on_parent(monkeypatch):
    from vllm_kvmem import pressure_parent_staging, layerwise_pressure_efficiency
    parent,c,s,p,chosen,names=fixture(monkeypatch,2)
    parent._metadata_owners=[]
    s['packed_layer_addressing']=('key',object(),object())
    parent.staging.destination=lambda size:(torch.empty(size,dtype=torch.uint8),object(),0)
    parent.staging.consumed=lambda slot:None
    parent.transfers=NS(upload_from_reader=lambda *a,**k:NS(event=object()))
    c.runner.device='cuda'
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:NS(wait_event=lambda event:None))
    def fail(*args):raise OSError('split submit failed')
    monkeypatch.setattr(layerwise_pressure_efficiency,'packed_layer_chunk',fail)
    with pytest.raises(OSError,match='split submit failed'):
        pressure_parent_staging.assemble_parent_pressure_layers(parent,c,s,p,chosen,names)
    assert parent._metadata_owners==list(s['packed_layer_addressing'][1:])
    assert parent.staging.owner is parent  # Only the outer assembly may release.


def test_failure_after_first_chunk_uses_outer_dma_fence_and_preserves_sources(monkeypatch):
    from vllm_kvmem import pressure_parent_staging,layerwise_pressure_efficiency
    from vllm_kvmem.window_assembly import NativeWindowAssembler
    parent,c,s,p,chosen,names=fixture(monkeypatch,600)
    parent._metadata_owners=[];parent._restore_lease=None;parent.raw_cache=None
    s['packed_layer_addressing']=('key',object(),object())
    stream_waits=[];releases=[];failures=[];source_releases=[];copies=[];consumed=[]
    stream=NS(wait_event=lambda event:stream_waits.append(event))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:stream)
    failure_fence=NS(record=lambda actual:None)
    monkeypatch.setattr(torch.cuda,'Event',lambda:failure_fence)
    parent.device=c.runner.device='cuda'
    parent.staging.owner=None
    def claim(owner):
        assert parent.staging.owner is None
        parent.staging.owner=owner
    def release(owner,failed):
        assert parent.staging.owner is owner
        releases.append(failed);parent.staging.owner=None
    parent.staging.claim=claim;parent.staging.release=release
    parent.staging.destination=lambda size:(torch.empty(size,dtype=torch.uint8),object(),0)
    parent.staging.consumed=lambda slot:consumed.append(slot)
    first=NS(event=object())
    def upload(size,gpu,fill,**kwargs):
        fill(torch.empty(size,dtype=torch.uint8))
        parent.transfers.pending[0]=first
        return first
    parent.transfers=NS(upload_from_reader=upload,pending={})
    def read(group,regions,*,destination):
        copies.append(tuple(group))
        if len(copies)==2:raise OSError('Second archive copy failed')
        return destination,dict(cache_copy_seconds=.001)
    s['prepared_pressure_reader'].read=read
    s['authority'].memory.fail_transaction=lambda result:failures.append(result)
    p.release_cpu_sources=lambda:source_releases.append(True)
    monkeypatch.setattr(layerwise_pressure_efficiency,'packed_layer_chunk',lambda *args:True)
    parent._assemble=lambda authority,plan:pressure_parent_staging.assemble_parent_pressure_layers(
        parent,c,s,plan,chosen,names)
    with pytest.raises(OSError,match='Second archive copy failed'):
        NativeWindowAssembler.assemble(parent,s['authority'],p)
    assert len(copies)==2 and len(copies[0])==544 and len(copies[1])==56
    assert consumed==[0]
    assert stream_waits==[first.event,first.event]  # Normal consumer then failure fence.
    assert releases==[failure_fence] and parent.staging.owner is None
    assert parent._completion is failure_fence
    assert failures[0].event is failure_fence and failures[0].owners==(parent,p,s['authority'])
    assert source_releases==[True]
    assert parent._metadata_owners==list(s['packed_layer_addressing'][1:])
