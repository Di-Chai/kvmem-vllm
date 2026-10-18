"""V2's independent dense operator oracle and sparse propagation contracts."""
import math
import random
from types import SimpleNamespace

import pytest

from vllm_kvmem.block_graph import BlockGraph, GraphBudget, GraphPartial, GraphRow
from vllm_kvmem.graph_scoring import score_graph, scoring_scratch_bytes


def snapshot(rows=(), *, spans=(), ends_known=None, completed=128, partial=None):
    return SimpleNamespace(block_tokens=4, rows=tuple(rows), partial=partial,
        message_spans=tuple(spans), message_end_known=((True,) * len(spans)
        if ends_known is None else tuple(ends_known)), closed=False, revision=1, epoch=1,
        accepted_through=completed + (2 if partial else 0), completed_through=completed,
        bytes=1024)


def score(raw, rows=(), *, starts=None, mask=None, snap=None, **kwargs):
    starts = list(range(0, len(raw) * 4, 4)) if starts is None else starts
    mask = [True] * len(raw) if mask is None else mask
    return score_graph(raw, starts, mask, snapshot(rows) if snap is None else snap,
                       scoring_version='v2', **kwargs)


def matrix_oracle(raw, starts, mask, snap, mode, alpha):
    """Build independent row-stochastic matrices, then multiply their transpose.

    This deliberately uses a dense row/column formulation and ordinary sums;
    it never calls the production sparse transfer or boundary helper.
    """
    n = len(raw)
    identity = [[float(i == j) for j in range(n)] for i in range(n)]
    locality, attention = [row[:] for row in identity], [row[:] for row in identity]
    boundaries = {begin for begin, _end, _role in snap.message_spans}
    boundaries.update(end for (_begin, end, _role), known in
                      zip(snap.message_spans, snap.message_end_known) if known)
    rows = {row.source: row for row in snap.rows}
    for i, source in enumerate(starts):
        if not mask[i]:
            continue
        for j, target in enumerate(starts):
            if not mask[j] or abs(source - target) != snap.block_tokens:
                continue
            low, high = min(source, target), max(source, target) + snap.block_tokens
            contained = any(begin <= low and high <= end for begin, end, _ in snap.message_spans)
            crossing = any(low < boundary < high for boundary in boundaries)
            probability = .5 * (.25 if not contained and crossing else 1.)
            locality[i][j] = probability
            locality[i][i] -= probability
        row = rows.get(source)
        if row is None or row.samples <= 0 or source + snap.block_tokens > snap.completed_through:
            continue
        masses = dict(zip(row.targets, row.weights))
        valid = [masses.get(target, 0.) if mask[j] and j != i else 0.
                 for j, target in enumerate(starts)]
        total = sum(valid)
        if total:
            attention[i] = [value / total for value in valid]
    total = sum(raw[i] for i in range(n) if mask[i])
    if total == 0 or alpha == 0:
        return list(raw)
    r = [raw[i] / total if mask[i] else 0. for i in range(n)]
    mixed = [[(1 - alpha) * identity[i][j] + alpha * (
                locality[i][j] if mode == 'locality' else
                attention[i][j] if mode == 'attention' else
                (locality[i][j] + attention[i][j]) / 2)
              for j in range(n)] for i in range(n)]
    return [sum(r[i] * mixed[i][j] for i in range(n)) * total if mask[j] else raw[j]
            for j in range(n)]


def test_handoff_multi_numerical_oracle_and_diagnostics():
    raw = [.1, .2, .7]
    result = score(raw, [GraphRow(8, (0,), (.8,), 1)], trace=True)
    assert result.scores == pytest.approx([.175, .220, .605])
    d = result.diagnostics
    assert d['scoring_version'] == 'v2' and d['alpha'] == .2
    assert d['locality_share'] == d['attention_share'] == .5
    assert d['locality_weight'] == d['attention_weight'] == .1
    assert d['attention_seed_coverage'] == pytest.approx(.7)
    assert d['attention_retained_raw_mass'] == pytest.approx(.56)
    assert d['attention_effective_raw_mass'] == pytest.approx(.56)
    assert d['attention_external_mass'] == pytest.approx(.07)
    assert d['locality_external_mass'] == pytest.approx(.06)
    assert d['self_retained_mass'] == pytest.approx(.87)
    assert d['normalized_l1_change'] == pytest.approx(.19)
    assert d['raw_l1_change'] == pytest.approx(.19)
    assert d['locality_unknown_fraction'] == 1.
    target = next(item for item in result.trace if item['target'] == 0)
    contributor = next(item for item in target['sources'] if item['kind'] == 'attention')
    assert contributor['raw_edge_weight'] == .8
    assert contributor['edge_weight'] == contributor['normalized_edge_weight'] == 1.
    assert contributor['contribution'] == pytest.approx(.07)
    assert target['direct_seed_score'] == target['raw_seed_score'] == .1


def test_attention_topk_example_and_normalization_after_masking():
    row = GraphRow(16, (0, 4, 8, 12), (.03, .02, .02, .01), 1)
    raw = [0., 0., 0., 0., 1.]
    before = (row.targets, row.weights, row.samples)
    result = score(raw, [row], mode='attention', alpha=1.)
    assert result.scores == pytest.approx([.375, .25, .25, .125, 0.])
    masked = score(raw, [row], mode='attention', alpha=1., mask=[False, False, False, True, True],
                   trace=True)
    assert masked.scores == pytest.approx([0., 0., 0., 1., 0.])
    assert masked.diagnostics['attention_masked_count'] == 3
    assert masked.diagnostics['attention_retained_raw_mass'] == pytest.approx(.08)
    assert masked.diagnostics['attention_effective_raw_mass'] == pytest.approx(.01)
    assert masked.diagnostics['attention_excluded_raw_mass'] == pytest.approx(.07)
    assert masked.diagnostics['attention_identity_mass'] == pytest.approx(0.)
    assert masked.diagnostics['masked_self_mass'] == 0.
    contributor = next(item for item in masked.trace if item['target'] == 12)['sources'][0]
    assert contributor['raw_edge_weight'] == .01
    assert contributor['normalized_edge_weight'] == 1.
    assert (row.targets, row.weights, row.samples) == before
    # Candidate restrictions are temporary; the next query sees the full row.
    assert score(raw, [row], mode='attention', alpha=1.).scores == result.scores


def test_masked_nonzero_seeds_are_unchanged_and_zero_seed_targets_receive():
    raw = [13., 0., 10.]
    rows = [GraphRow(0, (4,), (.4,), 1), GraphRow(8, (0, 4), (.4, .01), 1)]
    result = score(raw, rows, mode='attention', mask=[False, True, True], alpha=.2)
    assert result.scores == pytest.approx([13., 2., 8.])
    assert result.scores[0] is raw[0]
    assert result.diagnostics['raw_eligible_sum'] == 10.
    assert result.diagnostics['propagated_score_sum'] == 23.
    assert result.diagnostics['raw_attention_external_mass'] == 2.


@pytest.mark.parametrize('weight', [1e-300, math.ulp(0.)])
def test_arbitrarily_small_positive_retained_mass_still_sends_full_share(weight):
    result = score([0., 7.], [GraphRow(4, (0,), (weight,), 1)], mode='attention', alpha=.2)
    assert result.scores == pytest.approx([1.4, 5.6])


def test_self_and_zero_mass_edges_are_removed_before_normalization():
    row = GraphRow(8, (0, 4, 8), (0., .01, .9), 1)
    result = score([0., 0., 1.], [row], mode='attention', alpha=1.)
    assert result.scores == pytest.approx([0., 1., 0.])


@pytest.mark.parametrize('kwargs', [
    dict(alpha=0.), dict(mode='off'), dict(mode='collect'),
    dict(mode='attention'), dict(mode='locality', starts=[0, 8]),
    dict(mode='attention', rows=[GraphRow(4, (0,), (.8,), 0)]),
    dict(mode='attention', rows=[GraphRow(4, (0,), (.8,), 1)], mask=[False, True]),
    dict(mode='attention', rows=[GraphRow(4, (12,), (.8,), 1)]),
    dict(mode='attention', rows=[GraphRow(4, (4,), (.8,), 1)]),
])
def test_exact_input_object_on_degenerate_operators(kwargs):
    raw = [.123456789, .876543211]
    result = score(raw, **kwargs)
    assert result.scores is raw
    assert result.diagnostics['identity']
    assert result.diagnostics['normalized_l1_change'] == 0.
    assert result.diagnostics['self_retained_mass'] == 1.


@pytest.mark.parametrize('raw,mask', [([], []), ([0., 0.], [True, True]),
    ([7.], [True]), ([1., 2.], [False, False]), ([9., 0.], [False, True])])
def test_empty_single_masked_and_zero_transport_mass_identity(raw, mask):
    assert score(raw, mask=mask).scores is raw


def test_partial_or_unsealed_attention_source_is_identity():
    raw = [.123456789, .876543211]
    snap = snapshot([GraphRow(4, (0,), (.8,), 1)], completed=4,
                    partial=GraphPartial(4, (0,), (.8,), 1))
    result = score(raw, snap=snap, mode='attention')
    assert result.scores is raw
    assert result.diagnostics['attention_seed_coverage'] == 0.
    assert result.diagnostics['partial_source'] == 4


def test_zero_seed_source_with_edges_does_not_roundtrip_other_seeds():
    raw = [.123456789, .876543211, 0.]
    assert score(raw, [GraphRow(8, (0,), (.8,), 1)], mode='attention').scores is raw


def test_multi_identity_operator_keeps_its_fixed_half_share():
    locality_only = score([0., 1.], mode='multi', alpha=.2)
    assert locality_only.scores == pytest.approx([.05, .95])
    assert score([0., 1.], mode='locality', alpha=.2).scores == pytest.approx([.1, .9])
    attention_only = score([0., 1.], [GraphRow(8, (0,), (.01,), 1)],
                           starts=[0, 8], mode='multi', alpha=.2)
    assert attention_only.scores == pytest.approx([.1, .9])
    assert attention_only.diagnostics['attention_external_mass'] == pytest.approx(.1)


def test_one_hop_only_uses_original_source_seeds():
    rows = [GraphRow(8, (4,), (.01,), 1), GraphRow(4, (0,), (.01,), 1)]
    assert score([.1, .2, .7], rows, mode='attention', alpha=1.).scores == pytest.approx([.3, .7, 0.])


def test_locality_cross_boundary_attenuation_is_not_normalized_away():
    snap = snapshot(spans=[(0, 4, 'user'), (4, 8, 'tool')])
    result = score([1., 0.], mode='locality', alpha=1., snap=snap)
    assert result.scores == pytest.approx([.875, .125])
    assert result.diagnostics['locality_cross_message_fraction'] == 1.
    assert result.diagnostics['locality_external_mass'] == .125
    assert result.diagnostics['self_retained_mass'] == .875
    known = score([1., 0.], mode='locality', alpha=1., snap=snapshot(spans=[(0, 8, 'user')]))
    assert known.scores == pytest.approx([.5, .5])
    assert known.diagnostics['locality_same_message_fraction'] == 1.


def test_unknown_message_end_does_not_create_a_boundary():
    snap = snapshot(spans=[(0, 4, 'user')], ends_known=[False])
    result = score([1., 0.], mode='locality', alpha=1., snap=snap)
    assert result.scores == pytest.approx([.5, .5])
    assert result.diagnostics['locality_unknown_fraction'] == 1.


@pytest.mark.parametrize('alpha', [-.01, 1.01, math.inf, math.nan, True, False, '0.2', None])
def test_invalid_alpha_is_rejected_even_for_identity(alpha):
    with pytest.raises(ValueError, match='alpha'):
        score([0., 0.], alpha=alpha)


@pytest.mark.parametrize('version', ['v0', 'v3', 2, None])
def test_unknown_scoring_version_rejected(version):
    with pytest.raises(ValueError, match='version'):
        score_graph([1.], [0], [True], snapshot(), scoring_version=version)


@pytest.mark.parametrize('raw', [[math.nan, 1.], [math.inf, 1.], [-.1, 1.], [1e308, 1e308]])
def test_nonfinite_negative_or_overflowing_seeds_rejected(raw):
    with pytest.raises(ValueError, match='finite'):
        score(raw, alpha=0.)


def test_v1_is_the_default_and_keeps_absolute_attention_mass():
    raw, starts, mask = [.1, .2, .7], [0, 4, 8], [True] * 3
    snap = snapshot([GraphRow(8, (0,), (.8,), 1)])
    implicit = score_graph(raw, starts, mask, snap)
    explicit = score_graph(raw, starts, mask, snap, scoring_version='v1')
    assert implicit.scores == explicit.scores == pytest.approx([.217, .220, .563])
    assert implicit.diagnostics['scoring_version'] == 'v1'
    assert implicit.diagnostics['alpha'] is None


@pytest.mark.parametrize('mode', ['locality', 'attention', 'multi'])
@pytest.mark.parametrize('alpha', [0., .2, 1.])
def test_random_small_graphs_match_independent_matrix_oracle(mode, alpha):
    rng = random.Random(90210)
    universe = list(range(0, 48, 4))
    for case in range(60):
        starts = rng.sample(universe, rng.randrange(2, 10))
        mask = [rng.random() > .3 for _ in starts]
        raw = [0. if rng.random() < .4 else rng.random() * (10 ** (case % 4)) for _ in starts]
        rows = []
        for source in universe:
            possible = [target for target in universe if target < source]
            targets = tuple(rng.sample(possible, min(len(possible), rng.randrange(5))))
            weights = tuple(rng.random() * .05 for _ in targets)
            rows.append(GraphRow(source, targets, weights, rng.randrange(3)))
        snap = snapshot(rows, spans=[(0, 12, 'user'), (12, 24, 'tool'), (28, 44, 'tool')],
                        ends_known=[True, False, True], completed=rng.choice([32, 48]))
        before = (snap.rows, snap.message_spans, snap.message_end_known, snap.completed_through)
        expected = matrix_oracle(raw, starts, mask, snap, mode, alpha)
        actual = score(raw, starts=starts, mask=mask, snap=snap, mode=mode, alpha=alpha)
        assert actual.scores == pytest.approx(expected, rel=2e-12, abs=1e-12)
        assert math.fsum(actual.scores) == pytest.approx(math.fsum(raw), rel=2e-12, abs=1e-12)
        assert all(value >= 0. and math.isfinite(value) for value in actual.scores)
        assert all(actual.scores[i] is raw[i] for i in range(len(raw)) if not mask[i])
        assert (snap.rows, snap.message_spans, snap.message_end_known, snap.completed_through) == before


def test_v2_scratch_reservation_result_lifetime_and_identity_bypass():
    graph = BlockGraph('v2-scratch', 4)
    snap = graph.snapshot()
    before = graph.bytes
    result = score([.2, .8], snap=snap, mode='locality', trace=True)
    reserved = scoring_scratch_bytes(2, 0, 0, trace=True)
    assert graph.bytes == before + reserved
    snap.close()
    assert graph.bytes == graph.live_bytes + reserved
    result.close()
    assert graph.bytes == graph.live_bytes
    graph.close()
    assert graph.budget.used == 0
    small = BlockGraph('v2-identity', 4, budget=GraphBudget(4096), max_bytes_per_session=4096)
    with small.snapshot() as snap:
        raw = [.123456789, .876543211]
        before = small.bytes
        with score(raw, snap=snap, alpha=0.) as result:
            assert result.scores is raw and small.bytes == before
        with pytest.raises(BufferError, match='capacity'):
            score(raw, snap=snap)
        assert small.bytes == before
    small.close()
    assert small.budget.used == 0


def test_v2_validation_failure_returns_reserved_scratch():
    graph = BlockGraph('v2-invalid', 4)
    with graph.snapshot() as snap:
        before = graph.bytes
        with pytest.raises(ValueError, match='distinct'):
            score([.1, .9], snap=snap, starts=[0, 0])
        assert graph.bytes == before
    graph.close()
    assert graph.budget.used == 0
