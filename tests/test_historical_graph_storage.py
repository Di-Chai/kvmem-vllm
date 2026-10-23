"""Immutable graph storage maintenance without model/runtime dependencies."""
import pytest

from vllm_kvmem.block_graph import BlockGraph, GraphSample
from vllm_kvmem.graph_scoring import score_graph


def test_cow_retirement_preserves_positions_raw_weights_and_snapshot_accounting():
    graph = BlockGraph('cow', 4)
    graph.prepare(0, 12, [GraphSample(4, (0, 4), (.8, .2)),
                         GraphSample(8, (0, 4, 8), (.2, .7, .1))]).commit()
    original = graph.snapshot()
    before = graph.binding
    try:
        assert graph.retire_blocks({4})
        assert graph.binding != before
        assert graph.accepted_through == graph.completed_through == 12
        assert [row.source for row in graph.rows] == [8]
        assert graph.rows[0].targets == (0,) and graph.rows[0].weights == (.2,)
        assert [row.source for row in original.rows] == [4, 8]
        with graph.snapshot() as snapshot:
            with score_graph([.6, .4], [0, 8], [True, True], snapshot,
                    mode='locality', scoring_version='v2', alpha=1) as scores:
                assert scores.scores == [.6, .4]  # No new adjacency across the hole.
        assert graph.bytes == graph.live_bytes + original.bytes
    finally:
        original.close()
        graph.close()
    assert graph.budget.used == 0


def test_retirement_during_reserved_update_is_rejected_without_mutation():
    graph = BlockGraph('pending', 4)
    graph.prepare(0, 8, [GraphSample(4, (0, 4), (.8, .2))]).commit()
    binding, rows, charged = graph.binding, graph.rows, graph.bytes
    update = graph.prepare(8, 9).detach()
    try:
        with pytest.raises(BlockingIOError, match='reserved acceptance'):
            graph.retire_blocks({0})
        assert graph.binding == binding and graph.rows == rows
    finally:
        update.abort()
    assert graph.bytes == charged
    assert graph.retire_blocks({0})
    assert graph.rows[0].targets == ()
    graph.close()
    assert graph.budget.used == 0


def test_retirement_budget_failure_is_atomic_and_retryable():
    graph = BlockGraph('capacity', 4)
    graph.prepare(0, 8, [GraphSample(4, (0, 4), (.8, .2))]).commit()
    binding, rows, charged = graph.binding, graph.rows, graph.bytes
    limit = graph.max_bytes_per_session
    graph.max_bytes_per_session = charged
    with pytest.raises(BufferError):
        graph.retire_blocks({0})
    assert (graph.binding, graph.rows, graph.bytes) == (binding, rows, charged)
    graph.max_bytes_per_session = limit
    assert graph.retire_blocks({0})
    graph.close()
    assert graph.budget.used == 0


def test_retirement_filters_partial_without_changing_accepted_boundary_or_counts():
    graph = BlockGraph('partial', 4)
    graph.prepare(0, 9, [GraphSample(8, (0, 4, 8), (.2, .7, .1))]).commit()
    old = graph.snapshot()
    try:
        assert graph.retire_blocks({4})
        assert graph.accepted_through == 9 and graph.completed_through == 8
        assert graph.partial.targets == (0, 8)
        assert graph.partial.sums == (.2, .1) and graph.partial.samples == 1
        assert old.partial.targets == (0, 4, 8)
    finally:
        old.close()
        graph.close()
    assert graph.budget.used == 0

def test_gc_rebinding_accepts_only_contiguous_retirement_revisions():
    graph = BlockGraph('binding', 4)
    graph.prepare(0, 12, [GraphSample(4, (0, 4), (.8, .2)),
                         GraphSample(8, (0, 4, 8), (.2, .7, .1))]).commit()
    binding = graph.binding
    try:
        graph.retire_blocks({0})
        intermediate = graph.binding
        graph.retire_blocks({4})
        assert graph.rebind_after_gc(binding) == graph.binding
        assert graph.rebind_after_gc(intermediate) == graph.binding
        graph.invalidate()
        with pytest.raises(RuntimeError, match='Stale'):
            graph.rebind_after_gc(binding)
        # A later GC revision cannot hide an intervening epoch/accepted update.
        changed = graph.binding
        graph.prepare(12, 16, [GraphSample(12, (8, 12), (.8, .2))]).commit()
        graph.retire_blocks({8})
        with pytest.raises(RuntimeError, match='Stale'):
            graph.rebind_after_gc(changed)
    finally:
        graph.close()
