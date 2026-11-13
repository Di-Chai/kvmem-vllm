from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.action_attention import window_segments
from vllm_kvmem.serving.layerwise_pressure import extend_records
from vllm_kvmem.serving.layerwise_pressure_efficiency import assembly_groups, packet_histories


@pytest.mark.parametrize('skip', [(), (0,), (32, 64), (0, 32, 64, 96)])
def test_skipped_native_records_keep_original_dense_destinations(skip):
    records = [NS(history_begin=i*32, count=32 if i < 3 else 17) for i in range(4)]
    groups, total = assembly_groups(records, 2, skip)
    observed = []
    for offset, group in groups:
        for record in group:
            observed.append((record.history_begin, offset)); offset += record.count
    assert total == 113
    assert observed == [(r.history_begin, i*32) for i,r in enumerate(records) if r.history_begin not in skip]


def test_unknown_skipped_source_is_not_silently_ignored():
    with pytest.raises(ValueError, match='absent'):
        assembly_groups([NS(history_begin=0,count=32)], 32, [64])


@pytest.mark.parametrize('tail', [16, 32])
@pytest.mark.parametrize('appended', [0, 1, 16, 32, 512, 16384])
@pytest.mark.parametrize('count', [1, 32, 512])
def test_fast_layer_metadata_matches_original_extension_at_partial_boundaries(tail, appended, count):
    origin = 256 + tail
    other = 256 if tail != 32 else 192
    layers = (((0,1,0,32,0), (1,1,256,tail,32)),
              ((0,1,0,32,0), (2,1,other,tail,32)))
    begin = origin+appended; local = 32+tail+appended
    expected = [window_segments(extend_records(r,origin,begin), begin,local,count,32)
                for r in layers]
    starts,counts,history = expected[0]
    pending=NS(prefill=(begin+count,count,starts,local),history=history)
    assert packet_histories((origin,layers),pending,starts,counts) == [x[2] for x in expected]


def test_fast_metadata_does_not_reuse_another_layers_keys():
    layers=(((0,1,0,32,0),(1,1,64,32,32)),((0,1,0,32,0),(2,1,192,32,32)))
    pending=NS(prefill=(288,32,[0,32,64],64),history=[0,64,256])
    result=packet_histories((256,layers),pending,[0,32,64],[32,32,32])
    assert result==[[0,64,256],[0,192,256]]
    result[0][0]=999
    assert result[1][0]==0 and pending.history[0]==0


@pytest.mark.parametrize('stage', ['prefill', 'decode', 'prefill-diverged'])
def test_single_assembly_visits_each_target_and_draft_once_with_original_skips(monkeypatch, stage):
    import torch
    # A session's layer zero diverged from the native frame (KVMEM_LAYERWISE_SESSIONS)
    # rebuilds layer zero and the draft fully, as rolling decode does.
    diverged = stage == 'prefill-diverged'
    stage = 'prefill' if diverged else stage
    from vllm_kvmem.core.memory import Phase
    from vllm_kvmem.serving import window_assembly
    from vllm_kvmem.serving.layerwise_pressure_efficiency import install_single_assembly
    records = [NS(history_begin=i*32,count=32,handle=NS(slot=i)) for i in range(5)]
    chosen = [[records[0],records[2]], [records[0],records[4]]]
    plan = NS(remaps=[NS(block_id=0,to_base=0,n_tokens=32,skip=True),
                     NS(block_id=2,to_base=32,n_tokens=32,skip=False)], total_window_tokens=64)
    authority=NS(memory=NS(phase=Phase.ASSEMBLING,_plan=plan))
    job=NS(command=NS(request_id='r',pressure_stage=stage),layerwise_pressure_plan=(chosen,[]),
           layerwise_layer0_diverged=diverged)
    calls=[]
    def assemble(state,index,selected,**kwargs):
        assert state['pressure_stage'] == stage
        calls.append((index,[r.history_begin for r in selected],kwargs,state['assembly_arena']))
        return dict(uploaded_bytes=64,chunks=1)
    controller=NS(names=('target0','target1'),query=NS(assemble_layer_pinned=assemble),
                  runner=NS(device='cuda'),rolling_decode=stage=='decode')
    assembler=NS(layers=['layer0','layer1','draft'])
    monkeypatch.setattr(window_assembly,'_validate_destinations',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'Event',lambda:NS(record=lambda stream:None))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'stream')
    install_single_assembly(assembler,controller,job,('target0','target1','draft'))
    result=assembler._assemble(authority,plan)
    full = stage == 'decode' or diverged
    skip = set() if full else {0}
    assert [(i,ids,k['layer_name'],k['skip_history_begins'],arena) for i,ids,k,arena in calls] == [
        (0,[0,64],'target0',skip,'pressure'),(1,[0,128],'target1',(),'pressure'),
        (2,[0,64],'draft',skip,'pressure')]
    assert (result.rebuilt_tokens,result.skipped_tokens,result.uploaded_bytes)==(
        (64,0,192) if full else (32,32,192))
    assert job.layerwise_single_assembly_done


def test_single_assembly_rejects_draft_archive_reordering():
    from vllm_kvmem.serving.layerwise_pressure_efficiency import install_single_assembly
    with pytest.raises(ValueError,match='archive layer order'):
        install_single_assembly(NS(),NS(names=('a','b')),NS(),('a','draft','b'))


@pytest.mark.parametrize('failed',[False,True])
def test_single_assembly_closes_union_reader_on_success_and_submit_failure(monkeypatch,failed):
    import torch
    from vllm_kvmem.serving import pressure_reader, window_assembly
    from vllm_kvmem.core.memory import Phase
    from vllm_kvmem.serving.layerwise_pressure_efficiency import install_single_assembly
    records=[NS(history_begin=i*32,count=32,handle=NS(slot=i)) for i in range(3)]
    chosen=[[records[0],records[1]],[records[0],records[2]]]
    plan=NS(remaps=[NS(block_id=i,to_base=i*32,n_tokens=32,skip=False) for i in (0,1)],total_window_tokens=64)
    authority=NS(memory=NS(phase=Phase.ASSEMBLING,_plan=plan))
    job=NS(command=NS(request_id='r',pressure_stage='prefill'),layerwise_pressure_plan=(chosen,[]))
    states=[];calls=[]
    class Reader:
        source_blocks=3
        def __enter__(self):calls.append('enter');return self
        def __exit__(self,*args):calls.append('close')
    reader=Reader()
    def prepare(a,p,layers):
        assert a is authority and p is plan and layers is chosen
        calls.append('prepare');return reader
    def assemble(state,index,records,**kwargs):
        assert state['prepared_pressure_reader'] is reader
        states.append(state)
        if failed and index==1:raise OSError('submit failed')
        return dict(uploaded_bytes=64,chunks=1,inline_read_chunks=1,prepared_read_chunks=1)
    monkeypatch.setattr(pressure_reader,'prepare_pressure_reader',prepare)
    monkeypatch.setattr(window_assembly,'_validate_destinations',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'Event',lambda:NS(record=lambda stream:None))
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'stream')
    controller=NS(names=('a','b'),query=NS(assemble_layer_pinned=assemble,paired_reads=True),
                  runner=NS(device='cuda'),prepared_reads=True)
    assembler=NS(layers=['a','b','draft'])
    install_single_assembly(assembler,controller,job,('a','b','draft'))
    if failed:
        with pytest.raises(OSError,match='submit failed'):assembler._assemble(authority,plan)
        assert not getattr(job,'layerwise_single_assembly_done',False)
    else:
        result=assembler._assemble(authority,plan)
        assert result.read_stats['layerwise_prepared_source_blocks']==3
        assert result.read_stats['layerwise_prepared_read_chunks']==3
    assert calls==['prepare','enter','close']
