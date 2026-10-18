"""CPU lifecycle checks for pressure with newly installed locality evidence."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.prefill_attention import select_pressure
from test_core import Event
from test_graph_lifecycle import graph_checkpoints, complete
from test_graph_pressure_v2 import configure


def finish(pool, checkpoint, authority, archive, plan):
    complete(authority)
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    # Retained P snapshots must release their independent graph/resource owners.
    if pool.entries:
        pool.drop_live(next(iter(pool.entries.values())))
    assert authority.close() and archive.close()
    assert authority.block_graph.budget.used == 0


@pytest.mark.parametrize('edited', [False, True])
def test_pressure_restores_only_statistics_proven_by_full_m_prefix(tmp_path, edited):
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path)
    state = configure(authority, alpha=0)
    # Store distinct real policy snapshots at P and M. Future evidence must be
    # retained only after all M tokens were checked, not simply the P prefix.
    prompt.action_history = ({0:1.,4:2.},1,({0:(1.,1.)},8,1,1))
    terminal.action_history = ({0:9.,4:3.,8:50.},2,({4:(8.,2.)},12,2,2))
    tokens = list(range(24))
    if edited:
        tokens[9] = 999
    checkpoint = pool.claim(prompt.namespace, 'new', tokens, position=8)
    binding = NS(tokens=tuple(tokens), locality_spans=((0,8,'user'),(8,12,'tool')),
        locality_provenance=(('schema_version',1),('source','incoming_renderer')))
    plan = pool.prepare_claim_window(checkpoint, pressure=True, metadata_binding=binding)
    assert authority.history_end == 8
    assert graph.accepted_through == (8 if edited else 12)
    assert state.prefill.end == (8 if edited else 12)
    assert state.scores == ({0:1.,4:2.} if edited else {0:9.,4:3.,8:50.})
    assert authority.graph_selection_report['seed_source'] == 'output'
    assert authority._graph_selection_binding == graph.binding
    assert graph.message_spans == binding.locality_spans
    finish(pool, checkpoint, authority, archive, plan)


def test_partial_terminal_pressure_keeps_tail_and_uses_saved_candidates(tmp_path):
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path, partial=True)
    state = configure(authority, alpha=0)
    terminal.action_history = ({0:5.,4:1.,8:0.},1)
    checkpoint = pool.claim(terminal.namespace,'new',list(range(24)),position=10)
    assert checkpoint is terminal
    binding = NS(locality_spans=((0,8,'user'),(8,18,'assistant')),tokens=tuple(range(24)))
    plan = pool.prepare_claim_window(checkpoint, pressure=True, pressure_budget=8, metadata_binding=binding)
    selected = {row[2]:row[3] for row in authority.memory._store.working_records()}
    assert selected == {0:4,8:2}
    assert plan.total_window_tokens == 6
    assert graph.accepted_through == authority.history_end == 10
    assert graph.partial.source == 8 and graph.partial.output_samples == 1
    assert graph.message_end_known == (True,True)  # Full incoming message evidence remains valid.
    finish(pool, checkpoint, authority, archive, plan)


def test_async_prepare_passes_incoming_metadata_before_any_model_or_page_copy(tmp_path):
    from vllm_kvmem.async_restore import _step_restore
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path)
    configure(authority)
    binding = NS(namespace=prompt.namespace,generation=3,history=8,window=8,
        pressure_restore=True,tokens=tuple(range(24)),mandatory_spans=(),
        locality_spans=((0,8,'user'),(8,12,'tool')),
        locality_provenance=(('schema_version',1),('source','async_incoming')))
    class ModelPreparationReached(Exception):
        pass
    def prepare_model(snapshot, rid, *, lease):
        assert graph.message_spans == binding.locality_spans
        assert authority.graph_selection_report['locality_provenance']['source'] == 'async_incoming'
        assert authority._graph_selection_binding == graph.binding
        assert authority.history_end == 8 and graph.accepted_through == 12
        raise ModelPreparationReached  # Stop before creating any device event.
    worker = NS(bindings={'new':binding},imports={'new'},pool=pool,
                model_adapter=NS(prepare_restore=prepare_model))
    lease = NS(validate=lambda *args,**kwargs:None)
    runner = NS(_kvmem_sessions=worker,req_states=NS(req_id_to_index={'new':0}),
                _kvmem_restore_allocations={'new':lease})
    command = NS(request_id='new',generation=3,history_end=8,window_tokens=8,resident=False,block_ids=())
    job = NS(command=command,stage='prepare',checkpoint=None,future=None,done=False)
    manager = NS(runner=runner,options=NS(memory=authority.memory.config),jobs={'new':job})
    with pytest.raises(ModelPreparationReached):
        _step_restore(manager,job)
    assert job.checkpoint is prompt and job.authority is authority
    finish(pool, prompt, authority, archive, None)


def test_pressure_quota_failure_is_retryable_before_native_selection(tmp_path):
    from test_graph_lifecycle import fixture
    authority, archive, graph = fixture(tmp_path)
    state = configure(authority)
    state.scores = {20:1.}
    before = graph.binding, state.snapshot(), dict(state.mass), state.samples, authority.memory.phase
    graph.max_bytes_per_session = graph.live_bytes * 2
    with pytest.raises(BufferError):
        select_pressure(authority)
    assert before == (graph.binding,state.snapshot(),state.mass,state.samples,authority.memory.phase)
    assert graph.bytes == graph.live_bytes
    graph.max_bytes_per_session = 33554432
    plan = select_pressure(authority)
    complete(authority)
    del plan
    assert authority.close() and archive.close()


def test_pressure_trace_io_failure_leaves_ready_state_and_releases_all_graph_leases(tmp_path, monkeypatch):
    from test_graph_lifecycle import fixture
    import vllm_kvmem.graph_integration as integration
    authority, archive, graph = fixture(tmp_path)
    state = configure(authority)
    state.scores = {20:1.}
    before = authority.memory.phase, graph.binding, graph.bytes, state.snapshot()
    original = integration.emit_graph_report
    def fail(*args):
        raise OSError('test diagnostic storage failure')
    monkeypatch.setattr(integration,'emit_graph_report',fail)
    with pytest.raises(OSError,match='diagnostic storage'):
        select_pressure(authority)
    assert before == (authority.memory.phase,graph.binding,graph.bytes,state.snapshot())
    assert graph.bytes == graph.live_bytes
    monkeypatch.setattr(integration,'emit_graph_report',original)
    plan = select_pressure(authority)
    complete(authority)
    del plan
    assert authority.close() and archive.close()
