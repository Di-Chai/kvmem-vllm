"""Native history retirement plus immutable block-graph refresh transactions."""
from dataclasses import replace

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.block_graph import BlockGraph, GraphSample
from vllm_kvmem.graph_scoring import score_graph
from vllm_kvmem.historical_graph import compact_retired, note_retired
from vllm_kvmem.memory import Phase
from vllm_kvmem.prefill_attention import select_pressure
from test_graph_lifecycle import attach, complete
from test_graph_pressure_v2 import configure
from test_historical_authority import setup, append, advance, close
from test_historical_options import settings


def fixture(tmp_path, *, mode=n.HistoricalGcMode.OFF, count=32, future=False):
    authority, archive, worker = setup(tmp_path, mode=mode)
    graph = attach(authority)
    configure(authority, alpha=1)
    authority.graph_options.graph_trace = False
    _, handles = append(authority, count=count)
    authority.poll()
    end = 32 if future else count
    samples = [GraphSample(4, (0, 4), (.8, .2)),
               GraphSample(20, (4, 20), (.9, .1)),
               GraphSample(28, (20, 28), (.8, .2))]
    graph.prepare(0, end, [row for row in samples if row.position < end]).commit()
    if future:
        authority._graph_validated_prefix = end
    authority.action_history.scores = {20: 1., 28: .2}
    return authority, archive, worker, graph, handles


def retire(authority, worker, *handles):
    for handle in handles:
        authority.memory.repository.retire_permanently(handle)
    advance(authority, worker)


@pytest.mark.parametrize('policy', ['graph', 'attention'])
def test_retired_old_gpu_pages_are_filtered_before_pressure_scoring(tmp_path, policy):
    a, archive, worker, graph, handles = fixture(tmp_path)
    plan = None
    try:
        a.graph_options.pressure_policy = policy
        retire(a, worker, handles[1], handles[2])
        assert {row[2] for row in a.memory._store.working_records()} == set(range(0, 32, 4))
        assert 4 not in {record.history_begin for record in a.records.values()}
        plan = select_pressure(a, max_tokens=12)
        chosen = {row.block_id for row in plan.remaps}
        assert not chosen.intersection({handles[1].slot, handles[2].slot})
        assert a.graph_selection_report['candidate_count'] == 6
        assert all(row.source not in (4, 8) and not set(row.targets).intersection((4, 8))
                   for row in graph.rows)
        complete(a)
        del plan
        plan = None
        advance(a, worker)
        assert not {row[2] for row in a.memory._store.working_records()}.intersection((4, 8))
    finally:
        plan = None
        close(a, archive)


def test_query_guard_stops_policy_retirement_only_until_native_admission(tmp_path, monkeypatch):
    a, archive, worker, graph, _ = fixture(tmp_path, mode=n.HistoricalGcMode.DELETE)
    plan = None
    try:
        from vllm_kvmem import retrieval
        original = retrieval.mean_key_scores
        def score(*args):
            for _ in range(16):
                worker.advance()
            assert a.gc.scope.stats['retired'] == 0
            return original(*args)
        monkeypatch.setattr(retrieval, 'mean_key_scores', score)
        plan = a.select_for_queries(torch.zeros(2, 1, 1, 2), max_tokens=12)
        binding = graph.binding
        advance(a, worker)
        assert a.gc.scope.stats['retired'] > 0
        assert graph.binding == binding  # Existing admitted assembly remains valid.
        complete(a)
        del plan
        plan = None
    finally:
        plan = None
        close(a, archive)


def test_exception_releases_selection_guard_and_leaves_request_ready(tmp_path, monkeypatch):
    a, archive, worker, graph, _ = fixture(tmp_path, mode=n.HistoricalGcMode.DELETE)
    try:
        def fail(*_):
            raise ValueError('injected scorer failure')
        monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', fail)
        with pytest.raises(ValueError, match='injected'):
            a.select_for_queries(torch.zeros(2, 1, 1, 2))
        assert a.memory.phase is Phase.READY
        assert not a.memory._strict_history_selection
        advance(a, worker)
        assert a.gc.scope.stats['retired'] > 0
        assert graph.bytes == graph.live_bytes
    finally:
        close(a, archive)


def test_failed_hold_rebuilds_the_live_index_before_scoring(tmp_path, monkeypatch):
    a, archive, worker, graph, handles = fixture(tmp_path)
    original_scope = a.gc.scope
    plan = None
    class Scope:
        calls = 0
        def __getattr__(self, name):
            return getattr(original_scope, name)
        def protect_selection(self, values):
            self.calls += 1
            if self.calls == 1:
                a.memory.repository.retire_permanently(handles[1])
                return None
            return original_scope.protect_selection(values)
    scope = a.gc.scope = Scope()
    widths = []
    try:
        def score(queries, means, eligible):
            widths.append(means.shape[1])
            return torch.ones(means.shape[1])
        monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', score)
        plan = a.select_for_queries(torch.zeros(2, 1, 1, 2), max_tokens=12)
        assert scope.calls == 2 and widths == [7]
        assert handles[1].slot not in {row.block_id for row in plan.remaps}
        complete(a)
        del plan
        plan = None
    finally:
        a.gc.scope = original_scope
        plan = None
        close(a, archive)


def test_admission_retry_recomputes_graph_scores_instead_of_repicking_old_vector(tmp_path, monkeypatch):
    a, archive, worker, graph, handles = fixture(tmp_path)
    original_store = a.memory._store
    plan = None
    class Store:
        calls = 0
        def __getattr__(self, name):
            return getattr(original_store, name)
        def select(self, selected, *args):
            self.calls += 1
            if self.calls == 1:
                # Explicit retirement can override policy protection. This
                # simulates its pre-admission race without beginning assembly.
                a.memory.repository.retire_permanently(handles[1])
                raise n.HistoryAdmissionRetry('injected retirement race')
            return original_store.select(selected, *args)
    store = a.memory._store = Store()
    widths = []
    try:
        def score(queries, means, eligible):
            widths.append(means.shape[1])
            return torch.ones(means.shape[1])
        monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', score)
        plan = a.select_for_queries(torch.zeros(2, 1, 1, 2), max_tokens=12)
        assert store.calls == 2 and widths == [8, 7]
        assert handles[1].slot not in {row.block_id for row in plan.remaps}
        complete(a)
        del plan
        plan = None
    finally:
        a.memory._store = original_store
        original_store = store = None
        plan = None
        close(a, archive)


def test_graph_cleanup_preserves_future_verified_replay_nodes(tmp_path):
    a, archive, worker, graph, handles = fixture(tmp_path, count=16, future=True)
    plan = None
    try:
        retire(a, worker, handles[1])
        plan = a.select_for_queries(torch.zeros(2, 1, 1, 2), max_tokens=12)
        assert graph.accepted_through == 32 and a.history_end == 16
        assert {row.source for row in graph.rows} == {20, 28}
        assert next(row for row in graph.rows if row.source == 28).targets == (20,)
        assert next(row for row in graph.rows if row.source == 20).targets == ()
        complete(a)
        del plan
        plan = None
    finally:
        plan = None
        close(a, archive)


def test_cleanup_defers_for_detached_acceptance_and_budget_pressure(tmp_path):
    a, archive, worker, graph, handles = fixture(tmp_path)
    old = graph.snapshot()
    update = None
    try:
        retire(a, worker, handles[1])
        binding = graph.binding
        update = graph.prepare(32, 33).detach()
        compact_retired(a)
        assert graph.binding == binding and a._graph_gc_pending
        update.abort()
        update = None
        original_limit = graph.max_bytes_per_session
        graph.max_bytes_per_session = graph.bytes
        compact_retired(a)
        assert graph.binding == binding and a._graph_gc_reconcile
        graph.max_bytes_per_session = original_limit
        compact_retired(a)
        assert graph.binding != binding and not a._graph_gc_pending
        assert any(row.source == 4 for row in old.rows)
        assert all(row.source != 4 and 4 not in row.targets for row in graph.rows)
    finally:
        if update is not None:
            update.abort()
        old.close()
        close(a, archive)


def test_retirement_notice_from_old_epoch_does_not_delete_new_branch(tmp_path):
    a, archive, worker, graph, handles = fixture(tmp_path)
    try:
        record = a.records[(handles[1].slot, handles[1].generation)]
        note_retired(a, record)
        graph.invalidate()
        before = graph.rows
        compact_retired(a)
        assert graph.rows == before
        assert not a._graph_gc_pending
    finally:
        record = None
        close(a, archive)


@pytest.mark.parametrize('mode', ['collect', 'locality'])
@pytest.mark.parametrize('gc_mode', ['off', 'shadow', 'delete'])
def test_historical_gc_and_block_graph_configuration_can_coexist(mode, gc_mode):
    options = replace(settings(gc_mode), graph_mode=mode, graph_scoring_version='v2',
                      query_history=False)
    assert options.graph_mode == mode and options.historical_gc.mode == gc_mode
    with pytest.raises(ValueError, match='repository_gc'):
        replace(options, repository_gc=True)
