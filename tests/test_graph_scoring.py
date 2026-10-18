from dataclasses import replace
from types import SimpleNamespace
import math
import pytest

from vllm_kvmem.block_graph import GraphRow
from vllm_kvmem.graph_scoring import score_graph, locality_boundary


def snapshot(rows=(), spans=(), block_tokens=4):
    return SimpleNamespace(block_tokens=block_tokens, rows=tuple(rows), partial=None,
        message_spans=tuple(spans), closed=False, revision=1, epoch=1,
        accepted_through=40, completed_through=40, bytes=1024)


def edge(source, target, weight):
    return GraphRow(source, (target,), (weight,), 1)


def score(raw, rows=(), *, starts=None, eligible=None, spans=(), **kwargs):
    starts = list(range(0, len(raw) * 4, 4)) if starts is None else starts
    eligible = [True] * len(raw) if eligible is None else eligible
    return score_graph(raw, starts, eligible, snapshot(rows, spans), **kwargs)


@pytest.mark.parametrize('mode', ['off', 'collect'])
def test_disabled_mode_returns_exact_input_object(mode):
    raw = [.123456789, .876543211]
    assert score(raw, [edge(4, 0, .8)], mode=mode).scores is raw


@pytest.mark.parametrize('kwargs', [dict(mode='multi', locality_weight=0, attention_weight=0),
    dict(mode='attention'), dict(mode='locality', starts=[0, 8]),
    dict(mode='attention', rows=[edge(4, 0, .8)], eligible=[False, True])])
def test_zero_or_identity_returns_raw_object(kwargs):
    raw = [.123456789, .876543211]
    assert score(raw, **kwargs).scores is raw


@pytest.mark.parametrize('raw,eligible', [([], []), ([0., 0.], [True, True]),
    ([3.], [True]), ([1., 2.], [False, False])])
def test_empty_zero_single_identity(raw, eligible):
    assert score(raw, eligible=eligible).scores is raw


def test_causal_attention_direction_and_small_retained_mass():
    out = score([.1, .9], [edge(4, 0, .03)], mode='attention', attention_weight=1., locality_weight=0)
    assert out.scores == pytest.approx([.127, .873])
    assert out.diagnostics['normalized_output_sum'] == pytest.approx(1.)


def test_one_hop_does_not_propagate_received_mass_again():
    out = score([.1, .2, .7], [edge(8, 4, .8), edge(4, 0, .5)], mode='attention',
                locality_weight=0, attention_weight=1.)
    assert out.scores == pytest.approx([.2, .66, .14])


def test_both_graphs_read_same_initial_vector():
    raw = [.1, .2, .7]
    out = score(raw, [edge(8, 0, .8)], mode='multi', locality_weight=.1, attention_weight=.2)
    assert out.scores == pytest.approx([.217, .22, .563])
    assert math.fsum(out.scores) == pytest.approx(1.)


def test_missing_masked_edges_keep_mass_without_topk_refill_or_renormalize():
    rows = [GraphRow(8, (0, 4, 12), (.03, .07, .1), 1)]
    out = score([0., .2, .8], rows, starts=[0, 8, 16], eligible=[False, True, True],
                mode='attention', locality_weight=0, attention_weight=1)
    assert out.scores == [0., .2, .8]
    assert out.diagnostics['attention_missing_count'] == 2
    assert out.diagnostics['attention_masked_count'] == 1


def test_masked_source_never_sends_or_receives():
    raw = [0., .3, .7]
    out = score(raw, [edge(0, 8, 1.), edge(8, 0, 1.)], eligible=[False, True, True],
                mode='attention', locality_weight=0, attention_weight=1.)
    assert out.scores is raw


def test_single_neighbor_uses_half_slot_and_cross_boundary_quarter():
    out = score([1., 0.], mode='locality', locality_weight=1., attention_weight=0)
    assert out.scores == pytest.approx([.5, .5])
    boundary = score([1., 0.], spans=[(0, 4, 'user'), (4, 8, 'tool')],
                     mode='locality', locality_weight=1., attention_weight=0)
    assert boundary.scores == pytest.approx([.875, .125])


def test_locality_does_not_jump_missing_original_blocks():
    raw = [.2, .8]
    assert score(raw, starts=[0, 8], mode='locality').scores is raw


@pytest.mark.parametrize('spans,coefficient,fallback', [
    ([(0, 8, 'user')], 1., False), ([(0, 4, 'user'), (4, 8, 'tool')], .25, False),
    ([(0, 3, 'user'), (3, 8, 'tool')], .25, False), ([], 1., True),
    ([(8, 12, 'tool')], 1., True), ([(0, 2, 'user')], .25, False)])
def test_locality_boundary_precise_full_block_union(spans, coefficient, fallback):
    assert locality_boundary(0, 4, 4, spans, .25) == (coefficient, fallback)


def test_raw_mean_k_total_restored_without_final_renormalization():
    out = score([2., 8.], [edge(4, 0, .5)], mode='attention', attention_weight=.2)
    assert out.scores == pytest.approx([2.8, 7.2])
    assert out.diagnostics['raw_eligible_sum'] == 10
    assert out.diagnostics['normalized_output_sum'] == pytest.approx(1.)
    assert out.diagnostics['propagated_score_sum'] == pytest.approx(10.)


def test_requested_six_block_operator_example_fixed_budget():
    # Deliberately artificial forward edge from the user-provided mathematical
    # illustration. The production source aggregator only emits past edges.
    raw = [.01, .60, .15, .12, .07, .05]
    result = score(raw, [edge(4, 20, .8)])
    select = lambda values: sorted(sorted(range(6), key=lambda i: (-values[i], i))[:3])
    assert select(raw) == [1, 2, 3]
    assert select(result.scores) == [1, 2, 5]
    assert select(score(raw, [edge(4, 20, .8)], mode='off').scores) == [1, 2, 3]
    assert 3 * 4 == 12


def test_trace_distinguishes_self_retention_from_external_dependencies():
    result = score([.1, .9], [edge(4, 0, .03)], mode='attention', trace=True)
    first = next(item for item in result.trace if item['target'] == 0)
    assert first['self_retained'] == .1
    assert first['attention_incoming'] == pytest.approx(.0054)
    assert first['sources'][0]['source'] == 4
    assert first['sources'][0]['edge_weight'] == .03
    assert first['final_score'] == pytest.approx(.1054)


def test_trace_is_bounded():
    result = score([1.] * 512, mode='locality', trace=True)
    assert len(result.trace) <= 64
    assert all(len(item['sources']) <= 4 for item in result.trace)


@pytest.mark.parametrize('kwargs', [dict(locality_weight=float('nan')),
    dict(attention_weight=-.1), dict(locality_weight=.9, attention_weight=.2),
    dict(cross_message_weight=2), dict(mode='unknown')])
def test_invalid_weights_and_modes_fail(kwargs):
    with pytest.raises(ValueError):
        score([.2, .8], **kwargs)


@pytest.mark.parametrize('raw', [[float('nan'), 1.], [-.1, 1.1], [float('inf'), 0.]])
def test_invalid_raw_scores_rejected(raw):
    with pytest.raises(ValueError):
        score(raw)


def test_slots_and_generations_are_not_graph_identities():
    # No logical block 0 exists in this index even if its former physical slot
    # is now occupied by block 12; its edge must not land on the replacement.
    raw = [.2, .8]
    result = score(raw, [edge(8, 0, .5)], starts=[8, 12], mode='attention')
    assert result.scores is raw
    assert result.diagnostics['attention_missing_count'] == 1


def test_trace_prioritizes_newly_selected_targets_and_keeps_true_contributors():
    raw = [1.] * 512
    result = score(raw, [edge(2044, 1024, .8)], mode='multi', trace=True, trace_targets=(1024,))
    assert len(result.trace) == 64
    first = result.trace[0]
    assert first['target'] == 1024
    assert any(source['source'] == 2044 and source['kind'] == 'attention' for source in first['sources'])


def test_real_snapshot_reserves_scratch_before_large_locality_allocations():
    from vllm_kvmem.block_graph import BlockGraph, GraphBudget
    graph = BlockGraph('bounded', 32, budget=GraphBudget(4096), max_bytes_per_session=4096)
    raw, starts, mask = [1.] * 8192, list(range(0, 262144, 32)), [True] * 8192
    with graph.snapshot() as snap:
        before = graph.bytes
        with pytest.raises(BufferError, match='capacity'):
            score_graph(raw, starts, mask, snap, mode='locality')
        assert graph.bytes == before
    graph.close()
    assert graph.budget.used == 0


def test_result_lifetime_stays_charged_after_snapshot_close_until_result_release():
    from vllm_kvmem.block_graph import BlockGraph
    from vllm_kvmem.graph_scoring import scoring_scratch_bytes
    graph = BlockGraph('owner', 4)
    snap = graph.snapshot()
    before = graph.bytes
    result = score_graph([.2, .8], [0, 4], [True, True], snap, mode='locality', trace=True)
    reserved = scoring_scratch_bytes(2, 0, 0, trace=True)
    assert graph.bytes == before + reserved
    snap.close()
    assert graph.bytes == graph.live_bytes + reserved
    assert result.scores == pytest.approx([.23, .77])
    result.close()
    assert result.scores is result.diagnostics is None and result.trace == ()
    assert graph.bytes == graph.live_bytes
    graph.close()
    assert graph.budget.used == 0


def test_score_exception_returns_scratch_charge_and_identity_needs_none():
    from vllm_kvmem.block_graph import BlockGraph, GraphBudget
    graph = BlockGraph('owner', 4)
    with graph.snapshot() as snap:
        before = graph.bytes
        with pytest.raises(ValueError):
            score_graph([.1, .9], [0, 0], [True, True], snap, mode='multi')
        assert graph.bytes == before
    graph.close()
    small = BlockGraph('identity', 4, budget=GraphBudget(4096), max_bytes_per_session=4096)
    with small.snapshot() as snap:
        before = small.bytes
        raw = [.1, .9]
        with score_graph(raw, [0, 4], [True, True], snap,
                         locality_weight=0, attention_weight=0) as result:
            assert result.scores is raw and small.bytes == before
    small.close()


def test_scratch_charge_is_released_when_result_is_discarded():
    import gc
    from vllm_kvmem.block_graph import BlockGraph
    graph = BlockGraph('owner', 4)
    with graph.snapshot() as snap:
        before = graph.bytes
        for _ in range(20):
            score_graph([.2, .8], [0, 4], [True, True], snap, mode='multi')
        gc.collect()
        assert graph.bytes == before
    graph.close()
