from types import SimpleNamespace

import pytest

from vllm_kvmem.serving.layerwise_query import LayerwiseQuery, top_set


def runner(*, eager=True, sequences=1):
    return SimpleNamespace(model_config=SimpleNamespace(enforce_eager=eager),
        _kvmem_graph=False, _kvmem_layers=('layer3', 'layer7'),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=sequences)))


@pytest.mark.parametrize('eager,sequences', [(False,1),(True,2)])
def test_rejects_execution_that_cannot_run_ordered_python_hooks(monkeypatch,eager,sequences):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    with pytest.raises(ValueError):
        LayerwiseQuery(runner(eager=eager,sequences=sequences))


def test_rejects_stale_layer_order_before_touching_attention(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    engine=runner();engine._kvmem_last_batch=SimpleNamespace(req_ids=['request'])
    controller=LayerwiseQuery(engine)
    controller.active=dict(request_id='request',next_layer=0)
    with pytest.raises(RuntimeError,match='causal order'):
        controller.projection(1,None)


def test_rejects_partial_query_instead_of_looking_at_future_rows(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    engine=runner();engine._kvmem_last_batch=SimpleNamespace(req_ids=['request'],
        num_computed_tokens_np=[100],num_tokens=3)
    controller=LayerwiseQuery(engine)
    controller.active=dict(request_id='request',next_layer=0,history=100,span=(100,105))
    with pytest.raises(RuntimeError,match='complete public Query'):
        controller.projection(0,None)


def test_prohibits_later_refresh_with_a_shared_history_mapping(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    controller=LayerwiseQuery(runner())
    controller.finished.add('request')
    job=SimpleNamespace(command=SimpleNamespace(request_id='request',selection='pressure'))
    with pytest.raises(RuntimeError,match='prohibits another refresh'):
        controller.prepare(None,job,None)


@pytest.mark.parametrize('first_arena', ['main', 'layerwise'])
def test_single_layer_query_scratch_and_full_checkpoint_restore_do_not_collide(monkeypatch, first_arena):
    import torch
    from vllm_kvmem.core import native_pages; from vllm_kvmem.serving import window_assembly
    names = ('a', 'b')
    cache = torch.zeros((4,32,1,2), dtype=torch.float8_e4m3fn)
    modules = {name: SimpleNamespace(
        attn=SimpleNamespace(layer_name=name, kv_cache_dtype='fp8_e4m3',
                             kv_cache=cache, _k_scale=torch.tensor(1.)),
        num_kv_heads=1, head_dim=2,
        rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros((64,2)))) for name in names}
    engine = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: modules.items()),
        req_states=SimpleNamespace(req_id_to_index={'q':0}),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=names)]))
    monkeypatch.setattr(native_pages, 'kernel_table', lambda *args: torch.tensor([1,2]))
    monkeypatch.setattr(window_assembly, 'native_vllm_fp8_views', lambda *args: (cache, cache.clone()))
    def assemble(arena):
        return window_assembly.NativeWindowAssembler.from_runner(
            engine, 'q', names if arena == 'main' else names[:1], 32, arena=arena)
    first = assemble(first_arena)
    second = assemble('layerwise' if first_arena == 'main' else 'main')
    assert first.staging is not second.staging
    full, single = assemble('main'), assemble('layerwise')
    assert full.staging.slot_bytes == 2 * single.staging.slot_bytes
    assert full.staging is engine._kvmem_assembly_staging
    assert single.staging is engine._kvmem_layerwise_assembly_staging
    assert assemble('main').staging is full.staging
    assert assemble('layerwise').staging is single.staging


@pytest.mark.parametrize('chosen',[(),(1,2,3)])
def test_rejects_native_picker_mandatory_or_budget_violation(chosen):
    memory=SimpleNamespace(preview_selection=lambda *a,**k: chosen)
    records=[SimpleNamespace(handle=SimpleNamespace(slot=i)) for i in (1,2,3)]
    with pytest.raises(RuntimeError,match='mandatory content or exceeded'):
        top_set(memory,records,[3.,2.,1.],(1,),64)


def test_native_batch_visits_only_armed_owners_in_this_public_batch(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES','1')
    controller=LayerwiseQuery(runner(eager=False,sequences=8))
    controller.active_owners={'a':dict(request_id='a'), 'b':dict(request_id='b'),
                              'waiting':dict(request_id='waiting')}
    controller.runner._kvmem_last_batch=SimpleNamespace(req_ids=['other','b','a'])
    visited=[]
    monkeypatch.setattr(controller,'_projection',lambda state,layer,record:visited.append((state['request_id'],layer)))
    controller.projection(3,None)
    assert visited==[('a',3),('b',3)]
    assert controller.has_active


def test_batched_query_checks_its_own_token_extent_not_total_batch(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES','1')
    controller=LayerwiseQuery(runner(eager=False,sequences=8))
    controller.runner._kvmem_last_batch=SimpleNamespace(req_ids=['other','a'],
        num_computed_tokens_np=[0,100],query_start_loc_np=[0,50,53])
    state=dict(request_id='a',next_layer=0,history=100,span=(100,105))
    with pytest.raises(RuntimeError,match='complete public Query'):
        controller._projection(state,0,None)


def test_optimized_logging_flushes_layer_evidence_at_completion(monkeypatch,tmp_path):
    import json
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED','1')
    engine=runner();engine._kvmem_probe_output=tmp_path
    controller=LayerwiseQuery(engine)
    controller.emit('layer_selected',request_id='a',layer=0)
    controller.emit('layer_selected',request_id='b',layer=0)
    controller.emit('completed',request_id='a')
    rows=[json.loads(line) for line in controller.path.read_text().splitlines()]
    assert [(r['event'],r['request_id']) for r in rows]==[
        ('layer_selected','a'),('layer_selected','b'),('completed','a')]
    assert not controller.buffered_events


def test_persistent_capture_reads_the_just_computed_query_rows(monkeypatch):
    import torch
    import vllm_kvmem.serving.layerwise_query as implementation
    from vllm_kvmem.adapters.capture import CapturedLayer
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES','1')
    monkeypatch.setattr(implementation,'install_native_query_dispatch',lambda _:None)
    buffers=CapturedLayer('layer3',torch.arange(8),torch.full((8,1,2),-999.),
                          torch.zeros(8,1,2),torch.zeros(8,1,2))
    def original(qkv,positions):
        buffers.query[:len(qkv)].copy_(qkv.reshape(-1,1,2))
        return ('native-q','native-k','native-v','native-gate')
    module=SimpleNamespace(_project_qkv_gate=original,_kvmem_capture_buffers=buffers)
    engine=runner(eager=False,sequences=8);engine._kvmem_layers=('layer3',)
    engine.model=SimpleNamespace(named_modules=lambda:[('layer3',module)])
    implementation.install_layerwise_query(engine)
    controller=engine._kvmem_layerwise_query;controller.active_owners={'a':dict(request_id='a')}
    observed=[]
    monkeypatch.setattr(controller,'projection',lambda i,record:observed.append(record.query.clone()))
    for value in (3.,7.):
        result=module._project_qkv_gate(torch.full((2,2),value),torch.tensor([100,101]))
        assert result==('native-q','native-k','native-v','native-gate')
    assert [tensor.shape for tensor in observed]==[(2,1,2),(2,1,2)]
    assert torch.all(observed[0]==3.) and torch.all(observed[1]==7.)
