from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.action_attention import ActionHistory
from vllm_kvmem.prefill_attention import pressure_seed, select_pressure
from test_graph_lifecycle import fixture, complete, graph_checkpoints


def configure(a, *, alpha=.2, policy='graph'):
    o = a.graph_options
    o.graph_scoring_version = 'v2'; o.graph_alpha = alpha; o.pressure_policy = policy
    a.action_history = ActionHistory(NS(memory=a.memory.config, action_scope='non-thinking',
        action_interval=1, action_decay=.5, action_pressure=True, prefill_attention=True,
        prefill_attention_half_life=8192, native_prefill_commit=True), [10,11,12,13,14])
    return a.action_history


@pytest.mark.parametrize('scores,mass,samples,prefill,source', [
    ({20:1.}, {}, 0, {4:(100.,1.)}, 'output'),
    ({20:1.}, {16:2.}, 2, {4:(100.,1.)}, 'output'),
    ({0:1.}, {}, 0, {4:(100.,1.)}, 'output'),
    ({}, {}, 0, {4:(2.,4.)}, 'prefill'),
    ({}, {}, 0, {}, 'none'),
])
def test_alpha_zero_matches_native_old_pressure_without_mutating_statistics(
        tmp_path, scores, mass, samples, prefill, source):
    a, archive, graph = fixture(tmp_path, sink=1)
    state = configure(a, alpha=0)
    state.scores = dict(scores); state.mass = dict(mass); state.samples = samples
    state.prefill.values = dict(prefill)
    candidates = tuple(a.memory._store.working_records())
    raw, actual_source = pressure_seed(a, candidates)
    assert actual_source == source
    before = state.snapshot(), dict(state.mass), state.samples, graph.binding, graph.rows
    assert pressure_seed(a, candidates) == (raw, source)
    expected = a.memory.preview_selection({}, action=False, pressure_scores=raw, max_tokens=8)
    plan = select_pressure(a, max_tokens=8)
    assert tuple(r.block_id for r in plan.remaps) == expected
    assert before == (state.snapshot(), state.mass, state.samples, graph.binding, graph.rows)
    assert a.graph_selection_report['seed_source'] == source
    complete(a); del plan
    assert a.close() and archive.close()


def test_pressure_propagates_then_uses_original_picker(tmp_path):
    a, archive, graph = fixture(tmp_path)
    state = configure(a, alpha=1)
    state.scores = {20:1.}
    plan = select_pressure(a, max_tokens=8)
    chosen = {r.history_begin for r in a.records.values() if r.handle.slot in {p.block_id for p in plan.remaps}}
    assert chosen == {4,20}
    assert a.graph_selection_report['phase'] == 'pressure'
    assert a.graph_selection_report['added_blocks'] == 1
    complete(a); del plan
    assert a.close() and archive.close()


def test_query_max_pooling_does_not_change_cumulative_attention_pressure(tmp_path):
    a,archive,graph=fixture(tmp_path)
    state=configure(a,policy='attention')
    a.graph_options.graph_aggregation='max'
    state.prefill.values={4:(.1,1.),20:(1.,1.)}
    raw,source=pressure_seed(a,tuple(a.memory._store.working_records()))
    before=state.snapshot(),graph.binding
    expected=a.memory.preview_selection({},action=False,pressure_scores=raw,max_tokens=8)
    plan=select_pressure(a,max_tokens=8)
    assert tuple(r.block_id for r in plan.remaps)==expected
    assert a.graph_selection_report['propagation_applied'] is False
    assert a.graph_selection_report['seed_source']==source=='prefill'
    assert (state.snapshot(),graph.binding)==before
    complete(a);del plan
    assert a.close() and archive.close()


def test_cold_target_cannot_be_recalled_and_zero_candidates_remain(tmp_path):
    a, archive, graph = fixture(tmp_path)
    state = configure(a, alpha=1)
    p = a.memory.select({r.handle.slot:float(r.history_begin) for r in a.records.values()}, action=False)
    complete(a); del p
    assert {r[2] for r in a.memory._store.working_records()} == {12,16,20}
    state.scores={4:999.,20:1.}
    plan=select_pressure(a,max_tokens=8)
    chosen={r.history_begin for r in a.records.values() if r.handle.slot in {p.block_id for p in plan.remaps}}
    assert chosen == {16,20}  # 20->4 is filtered; zero 16 still fills natively.
    complete(a); del plan
    assert a.close() and archive.close()


@pytest.mark.parametrize('damage', ['generation','count','empty'])
def test_invalid_pressure_candidates_fail_before_assembly(tmp_path, damage):
    a, archive, graph = fixture(tmp_path)
    configure(a)
    rows=list(a.memory._store.working_records())
    if damage=='empty': rows=[]
    else:
        row=list(rows[0]); row[1 if damage=='generation' else 3]+=1; rows[0]=tuple(row)
    phase=a.memory.phase
    with pytest.raises((ValueError,RuntimeError)):
        select_pressure(a,candidate_records=rows)
    assert a.memory.phase is phase and graph.bytes==graph.live_bytes
    assert a.close() and archive.close()


def test_pressure_snapshot_staleness_rejected_at_publish(tmp_path):
    a, archive, graph=fixture(tmp_path)
    configure(a)
    plan=select_pressure(a)
    graph.invalidate()
    from test_core import Event
    a.memory.attach_completion(Event(True))
    with pytest.raises(RuntimeError,match='Stale'):
        a.memory.commit_selection()
    del plan
    assert a.close() and archive.close()


def test_pressure_budget_does_not_reserve_query_recent_band(tmp_path):
    a,archive,graph=fixture(tmp_path,sink=1,recent=2,budget=16)
    state=configure(a,alpha=0)
    state.scores={4:1.}
    plan=select_pressure(a,max_tokens=8)
    assert [r.history_begin for r in a.records.values() if r.handle.slot in {p.block_id for p in plan.remaps}]==[0,4]
    complete(a);del plan
    assert a.memory._store.runtime_select_budget==0
    assert a.close() and archive.close()


def test_native_pressure_all_fit_still_honors_explicit_working_candidates(tmp_path):
    from test_query_score_domain import fixture as small_fixture
    a,archive=small_fixture(tmp_path,budget=16,sink=1)
    records=sorted(a.records.values(),key=lambda r:r.history_begin)
    candidates=[(records[2].handle.slot,0.)]
    picked=a.memory.preview_selection({},pressure_scores=candidates,action=False)
    assert picked==(records[0].handle.slot,records[2].handle.slot)
    assert a.close() and archive.close()


def test_pressure_restore_installs_new_boundaries_before_snapshot_and_allows_verified_m(tmp_path):
    pool,prompt,terminal,a,archive,adapter,graph=graph_checkpoints(tmp_path)
    configure(a)
    checkpoint=pool.claim(prompt.namespace,'new',list(range(24)),position=8)
    assert checkpoint is prompt
    metadata=NS(message_spans=(),locality_spans=((0,8,'user'),(8,12,'tool')),
        locality_provenance=(('schema_version',1),('source','test_renderer')))
    plan=pool.prepare_claim_window(checkpoint,pressure=True,metadata_binding=metadata)
    assert graph.accepted_through==12 and a.history_end==8
    assert graph.message_spans==metadata.locality_spans
    assert a.graph_selection_report['locality_provenance']['source']=='test_renderer'
    assert a.graph_selection_report['authority_accepted_boundary']==8
    assert a._graph_selection_binding==graph.binding
    complete(a)
    from test_core import Event
    assert pool.finish_claim(checkpoint,Event(True)) is a
    del plan
    assert a.close() and archive.close()


def test_pressure_restore_metadata_conflict_precedes_rewind(tmp_path):
    pool,prompt,terminal,a,archive,adapter,graph=graph_checkpoints(tmp_path)
    configure(a)
    checkpoint=pool.claim(prompt.namespace,'new',list(range(24)),position=8)
    before=graph.binding,a.history_end
    with pytest.raises(ValueError,match='Conflicting'):
        pool.prepare_claim_window(checkpoint,pressure=True,
            metadata_binding=NS(message_spans=(),locality_spans=((0,4,'user'),)))
    assert (graph.binding,a.history_end)==before
    from test_core import Event
    assert pool.evict(checkpoint,Event(True))
    assert archive.close()
