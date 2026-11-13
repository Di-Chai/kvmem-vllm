"""Query resident V plans preserve native owners, dense order and fallback scope."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.core.memory import Phase
from vllm_kvmem.serving.query_resident_values import bind_query_resident_sources as _bind_query_resident_sources
from vllm_kvmem.serving.query_resident_values import plan_query_resident_values


def global_assembly(query,state,working,*,noncanonical=(),skip=()):
    import torch
    from vllm_kvmem.serving.window_assembly import AssemblyResult, NativeWindowAssembler
    authority=state['authority']
    assembler=NativeWindowAssembler.__new__(NativeWindowAssembler)
    assembler.closed=True;assembler._runner_binding=(query.runner,state['request_id'])
    assembler.reuse_resident_v=assembler.direct_pinned_reads=True
    assembler.raw_cache=None;assembler.raw_dtype=torch.float8_e4m3fn
    assembler.layers=tuple(range(authority.shape[0]))
    remaps=[NS(block_id=slot,n_tokens=count,to_base=base,from_base=base,
               working_k_resident=slot in noncanonical,skip=slot in skip)
            for slot,gen,begin,count,base in working]
    plan=NS(remaps=remaps,total_window_tokens=state['selected_tokens'])
    rebuilt=sum(r.n_tokens for r in remaps if not r.skip)
    skipped=sum(r.n_tokens for r in remaps if r.skip)
    reused=sum(r.n_tokens for r in remaps if not r.skip and r.working_k_resident)
    partial=sum(r.n_tokens for r in remaps if r.n_tokens!=32)
    result=AssemblyResult(NS(query=lambda:True),(assembler,plan,authority),rebuilt,skipped,1,
        uploaded_bytes=(2*rebuilt-reused)*authority.shape[0]*authority.shape[2]*authority.shape[3],
        resident_v_reused_tokens=reused,split_tokens=rebuilt-partial,partial_tokens=partial)
    return assembler,result,plan


def bind_query_resident_sources(query,state,working,*,noncanonical=(),skip=()):
    return _bind_query_resident_sources(query,state,working,
        global_assembly=global_assembly(query,state,working,noncanonical=noncanonical,skip=skip))


def fixture():
    import torch
    records = tuple(NS(handle=NS(slot=i,generation=9),history_begin=32*i,
                       count=32,archived=True) for i in range(6))
    authority = NS(records={(i,9):r for i,r in enumerate(records)},history_end=192,
                   memory=NS(request_id='r',phase=Phase.READY),shape=(17,32,4,256),
                   raw_element_bytes=1,raw_dtype=torch.float8_e4m3fn,
                   raw_record_bytes=17*32*4*256,value_record_bytes=17*32*4*256)
    query = NS(names=tuple(range(16)),no_probe=False,mtp_selection='global',
               runner=NS(_kvmem_request_generations={'r':7},
                         _kvmem_capture_runtime=NS(authorities={'r':authority})))
    state = dict(authority=authority,request_id='r',history=192,span=(192,193),
                 next_layer=0,records=records,selected_tokens=96,global_slots={4,0,2})
    working = [(i,9,32*i,32,j*32) for j,i in enumerate((4,0,2))]
    return query,state,working,records


def test_mixed_plan_uses_actual_global_positions_and_keeps_destinations():
    query,state,working,records = fixture()
    bind_query_resident_sources(query,state,working)
    chosen = [records[0],records[1],records[4]]
    result = plan_query_resident_values(query,state,0,chosen)
    assert result == (tuple(chosen),(0,2),(1,),(32,0),None)
    state['next_layer'] = 8
    assert plan_query_resident_values(query,state,8,chosen) == result


def test_native_retained_and_skipped_global_values_are_not_canonical_hits():
    query,state,working,records=fixture()
    binding=bind_query_resident_sources(query,state,working,noncanonical=(4,),skip=(2,))
    assert binding.canonical_keys==frozenset({(0,9)})
    assert binding.value_proof['canonical_blocks']==1
    assert binding.value_proof['uncertified_blocks']==2
    assert binding.value_proof['records'][(0,9)]['value_region']==(17*32768,17*32768)
    assert plan_query_resident_values(query,state,0,[records[0],records[1],records[4]])[1:]==(
        (0,),(1,2),(32,),None)
    assert plan_query_resident_values(query,state,0,[records[4],records[2],records[1]])[-1]==(
        'no_canonical_resident_values')


@pytest.mark.parametrize('fault',['missing','owner','event','closed','remap','position','bytes',
                                  'reused','owners','tokens'])
def test_global_archive_proof_rejects_unbound_or_inconsistent_results(fault):
    query,state,working,_=fixture()
    assembler,result,plan=proof=global_assembly(query,state,working)
    if fault=='missing':proof=None
    elif fault=='owner':assembler._runner_binding=(query.runner,'other')
    elif fault=='event':result.event=NS(query=lambda:False)
    elif fault=='closed':assembler.closed=False
    elif fault=='remap':plan.remaps[0].block_id=5
    elif fault=='position':plan.remaps[0].to_base=64
    elif fault=='bytes':result.uploaded_bytes-=32768
    elif fault=='reused':result.resident_v_reused_tokens=32
    elif fault=='owners':result.owners=(assembler,plan)
    else:result.rebuilt_tokens-=32
    with pytest.raises(ValueError):
        _bind_query_resident_sources(query,state,working,global_assembly=proof)


def test_unsupported_global_route_declines_before_any_resident_plan():
    query,state,working,records=fixture()
    proof=global_assembly(query,state,working)
    proof[0].direct_pinned_reads=False
    binding=_bind_query_resident_sources(query,state,working,global_assembly=proof)
    assert not binding.canonical_keys
    assert plan_query_resident_values(query,state,0,[records[0],records[1],records[4]])[-1]==(
        'unsupported_global_value_provenance')


def test_actual_prefix_publish_restore_binds_new_native_request_to_old_archive_namespace():
    from contextlib import nullcontext
    from vllm_kvmem.serving.session_worker import SessionWorker

    query,state,working,records=fixture()
    authority=state['authority'];authority.gc=None
    runtime=query.runner._kvmem_capture_runtime
    runtime.authorities.clear()  # CaptureRuntime.finish detaches the old owner.
    runtime.accepted={};runtime.rejected={};runtime.last_steps={}
    runtime._publication_lock=nullcontext
    runtime._raise_publication_error=lambda:None
    capacity=[]
    runtime._snapshot_capacity=lambda rid,a:capacity.append((rid,a))
    commits=[]
    query.runner._kvmem_window_adapter=NS(commit=lambda *args:commits.append(args))
    query.runner._kvmem_request_generations={'warm-request':8}
    worker=SessionWorker.__new__(SessionWorker)
    worker.runner=query.runner
    worker.options=NS(memory=NS(select_budget=96,gen_budget=32))
    worker.bindings={'warm-request':NS(history=192,window=96,
                                     namespace=('session','salt'),generation=1)}
    worker.restored_tokens=0;worker.completed=[]
    # Exercise the production handoff: it leaves memory.request_id unchanged
    # and installs runtime.authorities[new_native_id] under its publish lock.
    worker.publish_restore('warm-request',authority)
    assert authority.memory.request_id=='r'
    assert runtime.authorities=={'warm-request':authority}
    assert runtime.accepted=={'warm-request':192}
    assert commits==[('warm-request',192,96)] and capacity==[('warm-request',authority)]
    state['request_id']='warm-request'
    binding=bind_query_resident_sources(query,state,working)
    assert binding.request_id=='warm-request' and binding.archive_namespace=='r'
    assert binding.request_generation==8
    chosen=[records[0],records[1],records[4]]
    assert plan_query_resident_values(query,state,0,chosen)[1:]==((0,2),(1,),(32,0),None)
    runtime.authorities['warm-request']=NS(**vars(authority))
    with pytest.raises(ValueError,match='request-to-authority binding'):
        plan_query_resident_values(query,state,0,chosen)


@pytest.mark.parametrize('fault', ['pressure','phase','no_probe','mtp','request','history',
                                  'missing_generation','original_rope'])
def test_binding_rejects_foreign_scope(monkeypatch,fault):
    query,state,working,_ = fixture()
    if fault=='pressure': state['assembly_arena']='pressure'
    elif fault=='phase': state['authority'].memory.phase=Phase.ASSEMBLING
    elif fault=='no_probe': query.no_probe=True
    elif fault=='mtp': query.mtp_selection='layer0'
    elif fault=='request': state['request_id']='other'
    elif fault=='history': state['authority'].history_end+=32
    elif fault=='missing_generation': query.runner._kvmem_request_generations.clear()
    else: monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','1')
    with pytest.raises(ValueError): bind_query_resident_sources(query,state,working)


@pytest.mark.parametrize('fault',['gap','duplicate','generation','history','count','slots','twice'])
def test_native_source_map_is_validated_before_use(fault):
    query,state,working,_ = fixture()
    if fault=='gap':working[1]=(0,9,0,32,64)
    elif fault=='duplicate':working[1]=(4,9,128,32,32)
    elif fault=='generation':working[1]=(0,10,0,32,32)
    elif fault=='history':working[1]=(0,9,32,32,32)
    elif fault=='count':working[1]=(0,9,0,31,32)
    elif fault=='slots':state['global_slots']={4,0,3}
    else:bind_query_resident_sources(query,state,working)
    with pytest.raises(ValueError):bind_query_resident_sources(query,state,working)


@pytest.mark.parametrize('fault',['native_generation','authority','history','records','layer',
                                'memory_owner','record_table','repeated_layer','duplicate',
                                'foreign_record','source_mutated','missing_source_mutated',
                                'native_binding','archive_namespace'])
def test_plan_rejects_changed_request_or_sources(fault):
    query,state,working,records = fixture()
    binding=bind_query_resident_sources(query,state,working)
    chosen=[records[0],records[1],records[4]]
    if fault=='native_generation':query.runner._kvmem_request_generations['r']+=1
    elif fault=='authority':state['authority']=NS(**vars(state['authority']))
    elif fault=='memory_owner':state['authority'].memory=NS(**vars(state['authority'].memory))
    elif fault=='record_table':state['authority'].records=dict(state['authority'].records)
    elif fault=='native_binding':query.runner._kvmem_capture_runtime.authorities.clear()
    elif fault=='archive_namespace':state['authority'].memory.request_id='rebound'
    elif fault=='history':state['history']+=32;state['authority'].history_end+=32
    elif fault=='records':state['records']=tuple(list(records))
    elif fault=='layer':state['next_layer']=1
    elif fault=='repeated_layer':binding.used_layers.add(0)
    elif fault=='duplicate':chosen=[records[0],records[0],records[4]]
    elif fault=='foreign_record':chosen[0]=NS(**vars(records[0]))
    elif fault=='missing_source_mutated':records[1].history_begin=999
    else:records[0].history_begin=999
    with pytest.raises(ValueError):plan_query_resident_values(query,state,0,chosen)


def test_partial_unpublished_and_no_hit_are_explicit_fallbacks():
    query,state,working,records = fixture()
    bind_query_resident_sources(query,state,working)
    assert plan_query_resident_values(query,state,0,[records[1],records[3],records[5]])[-1]=='no_resident_values'
    records[1].archived=False
    assert plan_query_resident_values(query,state,0,[records[0],records[1],records[4]])[-1]=='unpublished_selected_record'

    query,state,_,records=fixture()
    records[0].count=17
    state['selected_tokens']=81
    working=[(4,9,128,32,0),(0,9,0,17,32),(2,9,64,32,49)]
    bind_query_resident_sources(query,state,working)
    assert plan_query_resident_values(query,state,0,[records[0],records[1],records[4]])[-1]=='partial_selected_record'


@pytest.mark.parametrize('fail_read',[False,True])
def test_mixed_reader_uses_existing_scratch_and_snapshots_before_all_writes(monkeypatch,fail_read):
    import torch
    from vllm_kvmem.serving import split_assembly
    from vllm_kvmem.serving.query_resident_values import assemble_query_resident_values
    query,state,working,records=fixture()
    query.packed_assembly=True
    query.runner.device='cpu'
    query.runner._kvmem_capture_runtime.options=NS(reuse_resident_v=True)
    authority=state['authority'];authority.raw_element_bytes=1;authority.shape=(17,32,4,256)
    bind_query_resident_sources(query,state,working)
    row_bytes=32*4*256
    timeline=[]
    class Event:
        def record(self,*args):pass
        def synchronize(self):pass
    stream=NS(wait_event=lambda e:None)
    monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda *args:stream)
    layer=NS(value=NS(shape=(4,32,4,256)),kernel_page_table=NS(numel=lambda:4),
             cos_sin=NS(shape=(96,64)))
    staging=NS(slot_bytes=2*row_bytes*256,value_capacity_bytes=96*4*256,
               value_destination=lambda n:torch.empty(n,dtype=torch.uint8),
               destination=lambda n:(torch.empty(n,dtype=torch.uint8),None,0),
               consumed=lambda slot:None)
    def metadata(values):
        tensor=torch.tensor(values,dtype=torch.int64)
        assembler._metadata_owners.append(tensor)
        return tensor
    def upload(size,gpu,fill,**kwargs):
        host=torch.empty(size,dtype=torch.uint8)
        fill(host)
        return NS(event=Event())
    assembler=NS(staging=staging,layers=(layer,),_metadata_owners=[],
                 _upload_metadata=metadata,transfers=NS(upload_from_reader=upload))
    staging.owner=assembler
    def snapshot(layers,bases,block,destination,**kwargs):
        timeline.append(('snapshot',bases.tolist(),destination.numel()))
        return destination
    def write(raw,raw_indices,values,value_indices,bases,layer,**kwargs):
        assert timeline[0][0]=='snapshot'
        timeline.append(('write',raw_indices.tolist(),value_indices.tolist(),bases.tolist()))
    monkeypatch.setattr(split_assembly,'gather_resident_values',snapshot)
    monkeypatch.setattr(split_assembly,'assemble_split_layer',write)
    def reader(authority,handles,*,destination,**kwargs):
        assert timeline[0][0]=='snapshot'
        timeline.append(('read',[h.slot for h in handles],kwargs,destination.numel()))
        if fail_read:raise OSError('injected CPU read failure')
        destination.fill_(7)
        return destination,dict(logical_bytes=destination.numel(),prepared_cpu_region_read=True,
                               inline_cpu_region_read=True,query_reader_source_blocks=2)
    selected=[records[0],records[1],records[4]]
    if fail_read:
        with pytest.raises(OSError,match='injected'):
            assemble_query_resident_values(query,state,0,selected,assembler,256,reader=reader)
        assert all(row[0]!='write' for row in timeline)
    else:
        stats=assemble_query_resident_values(query,state,0,selected,assembler,256,reader=reader)
        assert stats['uploaded_bytes']==4*row_bytes
        assert stats['resident_v_avoided_archive_bytes']==2*row_bytes
        assert stats['prepared_read_chunks']==stats['inline_read_chunks']==2
        assert stats['assembly_read_stats']['logical_bytes']==4*row_bytes
        assert stats['assembly_read_stats']['query_reader_source_blocks']==2
        assert [row for row in timeline if row[0]=='write']==[
            ('write',[0,1],[0,1],[0,64]),('write',[0],[0],[32])]
        assert [row[2] for row in timeline if row[0]=='read']==[
            {'region':(0,row_bytes)}, {'regions':((0,row_bytes),(17*row_bytes,row_bytes))}]
    assert timeline[0]==('snapshot',[32,0],2*row_bytes)
    # Addressing owners survive both success and an error until the caller's
    # original synchronization retires the Query state.
    assert state['query_resident_metadata_owners']
