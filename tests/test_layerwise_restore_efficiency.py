from types import SimpleNamespace as NS

import pytest
import torch

from test_layerwise_checkpoints import prefix
from vllm_kvmem.serving.layerwise_checkpoints import capture
from vllm_kvmem.serving.layerwise_restore_efficiency import install_direct_restore, finish_direct_restore
from vllm_kvmem.core.memory import Phase


def fixture(monkeypatch):
    from vllm_kvmem.serving import layerwise_restore_efficiency as module
    controller, authority = prefix()
    snapshot = capture(controller,'old',authority,288)
    names = (*controller.names,'draft')
    authority.shape = (len(names),32,1,4)
    authority.raw_dtype = torch.float8_e4m3fn
    plan = NS(remaps=[NS(block_id=r[0],n_tokens=r[3],to_base=r[4],skip=True)
                      for r in snapshot.maps[0]], total_window_tokens=96)
    authority.memory.phase = Phase.ASSEMBLING
    authority.memory._plan = plan
    worker = NS(runner=NS(_kvmem_layerwise_pressure=controller,device='cuda'),
        runtime=NS(names=names),options=NS(session_freeze_prompt=True),
        bindings={'new':NS(pressure_restore=False)})
    assembler = NS(layers=names,block_tokens=32,heads=1,dim=4,raw_dtype=authority.raw_dtype)
    checkpoint = NS(kind='P',authority=authority,layerwise_pressure=snapshot)
    controller.query.direct_prefix_restore = True
    controller.query.optimized = True
    calls = []
    def assemble(state,index,records,*,layer_name):
        assert 'new' not in controller.maps and 'new' not in controller.histories
        assert state['assembly_arena'] == 'layerwise'
        calls.append((index,layer_name,[r.history_begin for r in records]))
        return dict(uploaded_bytes=768,chunks=1)
    controller.query.assemble_layer_pinned = assemble
    controller.query.emit = lambda *args,**kwargs: calls.append(('published',kwargs))
    event = NS(ready=False)
    event.query = lambda:event.ready
    event.record = lambda stream:calls.append(('fence',stream))
    monkeypatch.setattr(torch.cuda,'Event',lambda:event)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:'consumer')
    monkeypatch.setattr(module,'_validate_destinations',lambda *args:None)
    return worker,checkpoint,assembler,plan,event,calls


def test_direct_restore_writes_each_target_and_draft_once_then_publishes(monkeypatch):
    worker,checkpoint,assembler,plan,event,calls = fixture(monkeypatch)
    controller = worker.runner._kvmem_layerwise_pressure
    assert install_direct_restore(worker,'new',checkpoint,assembler,plan)
    result = assembler._assemble(checkpoint.authority,plan)
    assert [c[2] for c in calls[:3]] == [[0,64,256],[0,128,256],[0,64,256]]
    assert len(calls) == 4 and calls[-1] == ('fence','consumer')
    assert result.uploaded_bytes == 3*768 and result.rebuilt_tokens == 96
    assert result.event is event and not result.query()
    with pytest.raises(ValueError): finish_direct_restore(worker,'new',checkpoint,assembler)
    event.ready = True
    with pytest.raises(ValueError): finish_direct_restore(worker,'new',checkpoint,assembler)
    checkpoint.authority.memory.phase = Phase.READY
    finish_direct_restore(worker,'new',checkpoint,assembler)
    assert not hasattr(assembler,'_assemble') and not hasattr(assembler,'_direct_prefix_frame')
    assert controller.maps['new'] == (288,checkpoint.layerwise_pressure.maps)
    assert calls[-1][0] == 'published'
    with pytest.raises(ValueError): finish_direct_restore(worker,'new',checkpoint,assembler)


def test_uncommitted_plan_is_validated_instead_of_previous_query_view(monkeypatch):
    worker,checkpoint,assembler,plan,event,calls = fixture(monkeypatch)
    checkpoint.authority.memory._store.working_records = lambda:()
    assert install_direct_restore(worker,'new',checkpoint,assembler,plan)
    assembler._assemble(checkpoint.authority,plan)
    event.ready = True
    checkpoint.authority.memory.phase = Phase.READY
    with pytest.raises(ValueError): finish_direct_restore(worker,'new',checkpoint,assembler)
    assert 'new' not in worker.runner._kvmem_layerwise_pressure.maps


def test_finished_restore_does_not_retain_plan_until_cyclic_gc(monkeypatch):
    import gc
    import weakref
    worker,checkpoint,assembler,old_plan,event,calls = fixture(monkeypatch)
    class Plan: pass
    plan = Plan()
    plan.__dict__.update(vars(old_plan))
    checkpoint.authority.memory._plan = plan
    ref = weakref.ref(plan)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        assert install_direct_restore(worker,'new',checkpoint,assembler,plan)
        result = assembler._assemble(checkpoint.authority,plan)
        event.ready = True
        checkpoint.authority.memory.phase = Phase.READY
        checkpoint.authority.memory._plan = None
        finish_direct_restore(worker,'new',checkpoint,assembler)
        del result,plan
        assert ref() is None, 'An idle session must release native plan pins without gc.collect()'
    finally:
        if was_enabled:gc.enable()


@pytest.mark.parametrize('fault',['disabled','controller','kind','pressure','frozen','unoptimized',
    'layer_order','phase','plan','length','position','shape','dtype','budget'])
def test_invalid_direct_restore_never_writes_gpu_or_publishes(monkeypatch,fault):
    worker,checkpoint,assembler,plan,event,calls = fixture(monkeypatch)
    controller = worker.runner._kvmem_layerwise_pressure
    if fault=='disabled':controller.query.direct_prefix_restore=False
    if fault=='controller':worker.runner._kvmem_layerwise_pressure=None
    if fault=='kind':checkpoint.kind='M'
    if fault=='pressure':worker.bindings['new'].pressure_restore=True
    if fault=='frozen':worker.options.session_freeze_prompt=False
    if fault=='unoptimized':controller.query.optimized=False
    if fault=='layer_order':worker.runtime.names=tuple(reversed(worker.runtime.names))
    if fault=='phase':checkpoint.authority.memory.phase=Phase.READY
    if fault=='plan':checkpoint.authority.memory._plan=object()
    if fault=='length':plan.remaps[0].n_tokens=16
    if fault=='position':plan.remaps[0].to_base=32
    if fault=='shape':checkpoint.authority.shape=(2,32,1,4)
    if fault=='dtype':assembler.raw_dtype=torch.bfloat16
    if fault=='budget':plan.total_window_tokens=64
    if fault in ('disabled','controller'):
        assert not install_direct_restore(worker,'new',checkpoint,assembler,plan)
    else:
        with pytest.raises(ValueError): install_direct_restore(worker,'new',checkpoint,assembler,plan)
    assert not calls and 'new' not in controller.maps
