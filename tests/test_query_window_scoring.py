"""Independent token-position and scalar-attention oracles for 16/8 scoring."""
import math
import os

import pytest
import torch

from vllm_kvmem.query_window_scoring import (
    overlap_window_scores, window_layout, window_targets,
)


def fixture_inputs(starts, counts, history_end, keys, score_mask, *, mean_dtype=torch.float32):
    """Pack scorer inputs; the oracle below independently scans logical tokens."""
    layers, _, kvheads, dim = keys[0].shape
    windows = window_layout(starts, counts, history_end)
    means = torch.full((layers, 4*len(starts), kvheads, dim), -torch.inf, dtype=mean_dtype)
    for window in windows:
        pieces = [keys[i][:, offset:offset+count].float() for i, offset, count in window.parts]
        means[:, window.slot] = torch.cat(pieces, dim=1).mean(dim=1)
    block_means = torch.stack([key.float().mean(dim=1) for key in keys], dim=1).to(mean_dtype)
    targets, valid = window_targets(windows, len(starts), score_mask)
    return means, targets, valid, block_means, torch.tensor(score_mask, dtype=torch.bool)


def scalar_oracle(queries, starts, counts, history_end, keys, score_mask,
                  block_means, *, mean_dtype, scale=None, query_half=False):
    """Enumerate original positions, then loop layer, Query row and Q head.

    No layout/target helper, scatter reduction, einsum or Torch softmax is used.
    A window is present only when every original position is both available and
    allowed for scoring; holes do not shorten it or concatenate their sides.
    """
    queries = queries.cpu()
    layers, rows, heads, dim = queries.shape
    kvheads = keys[0].shape[2]
    group = heads//kvheads
    scale = dim**-.5 if scale is None else scale
    ownership = {start+offset: (i, offset) for i, (start, count)
                 in enumerate(zip(starts, counts)) for offset in range(count)}
    windows = []
    for begin in range(0, history_end, 8):
        end = min(begin+16, history_end)
        positions = list(range(begin, end))
        if any(pos not in ownership or not score_mask[ownership[pos][0]] for pos in positions):
            continue
        values = [keys[ownership[pos][0]][:, ownership[pos][1]].float() for pos in positions]
        mean = torch.stack(values, dim=1).mean(dim=1).to(mean_dtype).float()
        blocks = {ownership[pos][0] for pos in positions}
        windows.append((mean, blocks))
    result = [0.]*len(starts)
    for layer in range(layers):
        for row in range(rows):
            for head in range(heads):
                q = queries[layer, row, head]
                q = (q.half() if query_half else q).float()
                logits = []
                for block in range(len(starts)):
                    if not score_mask[block]:
                        logits.append(-math.inf)
                        continue
                    candidates = [float(torch.dot(q, mean[layer, head//group]))*scale
                                  for mean, owners in windows if block in owners]
                    if not candidates:
                        candidates = [float(torch.dot(q, block_means[layer, block, head//group].float()))*scale]
                    logits.append(max(candidates))
                finite = [value for value in logits if math.isfinite(value)]
                if not finite:
                    continue
                weights = [math.exp(value-max(finite)) for value in logits]
                total = sum(weights)
                for block, weight in enumerate(weights):
                    result[block] += weight/total/(layers*heads)
    return torch.tensor(result, dtype=torch.float32)


@pytest.mark.parametrize('starts,counts,history_end,expected', [
    ([0, 32, 64], [32, 32, 9], 73,
     [(begin, min(begin+16, 73)) for begin in range(0, 73, 8)]),
    ([0, 64], [32, 9], 73, [(0, 16), (8, 24), (16, 32), (64, 73), (72, 73)]),
    ([0, 32], [12, 32], 64, [(32, 48), (40, 56), (48, 64), (56, 64)]),
    ([32], [20], 52, [(32, 48), (40, 52), (48, 52)]),
    ([0], [7], 40, []),
    ([], [], 0, []),
])
def test_windows_follow_original_positions_and_only_global_tail_can_be_short(starts, counts, history_end, expected):
    windows = window_layout(starts, counts, history_end)
    assert [(window.begin, window.end) for window in windows] == expected
    for window in windows:
        assert window.begin % 8 == 0
        assert sum(count for _, _, count in window.parts) == window.end-window.begin
        positions = [starts[block]+offset+i for block, offset, count in window.parts for i in range(count)]
        assert positions == list(range(window.begin, window.end))


def test_cross_boundary_window_targets_both_blocks_and_forbidden_content_filters_whole_window():
    windows = window_layout([0, 32], [32, 32], 64)
    boundary = next(window for window in windows if window.begin == 24)
    assert boundary.parts == ((0, 24, 8), (1, 0, 8))
    targets, valid = window_targets(windows, 2, [True, True])
    assert bool(valid[boundary.slot]) and targets[boundary.slot].tolist() == [0, 1]
    masked_targets, masked_valid = window_targets(windows, 2, [True, False])
    assert not bool(masked_valid[boundary.slot])
    assert masked_targets[boundary.slot].tolist() == [-1, -1]
    assert masked_valid.tolist() == [True, True, True, False, False, False, False, False]


@pytest.mark.parametrize('query_half,mean_dtype,scale', [
    (False, torch.bfloat16, None), (True, torch.float16, .13),
])
@pytest.mark.parametrize('query_chunk', [1, 3, 8])
def test_gqa_fp8_keys_partial_tail_masks_scale_and_tiling_match_scalar_oracle(query_half, mean_dtype, scale, query_chunk):
    generator = torch.Generator().manual_seed(147)
    starts, counts, history_end, score_mask = [0, 32, 64], [32, 32, 9], 73, [True, False, True]
    keys = [torch.randn(2, count, 2, 3, generator=generator).to(torch.float8_e4m3fn) for count in counts]
    q = torch.randn(2, 9, 4, 3, generator=generator)
    inputs = fixture_inputs(starts, counts, history_end, keys, score_mask, mean_dtype=mean_dtype)
    observed = overlap_window_scores(q, *inputs, query_half=query_half, scale=scale, query_chunk=query_chunk)
    expected = scalar_oracle(q, starts, counts, history_end, keys, score_mask, inputs[3],
        mean_dtype=mean_dtype, query_half=query_half, scale=scale)
    torch.testing.assert_close(observed, expected, rtol=2e-6, atol=2e-6)
    assert observed[1] == 0
    assert float(observed.sum()) == pytest.approx(q.shape[1], abs=2e-6)


def test_cross_boundary_max_supplies_both_blocks_without_neighbor_postprocessing():
    keys = [torch.full((1, 32, 1, 1), -10.), torch.full((1, 32, 1, 1), -10.)]
    keys[0][:, 24:] = 10.
    keys[1][:, :8] = 10.
    inputs = fixture_inputs([0, 32], [32, 32], 64, keys, [True, True])
    q = torch.ones(1, 1, 1, 1)
    assert inputs[0][0, 3, 0, 0] == 10  # [24,40) is uniquely strongest.
    observed = overlap_window_scores(q, *inputs)
    torch.testing.assert_close(observed, torch.tensor([.5, .5]))
    targets_without_second = inputs[1].clone()
    targets_without_second[3, 1] = -1
    one_sided = overlap_window_scores(q, inputs[0], targets_without_second, *inputs[2:])
    assert one_sided[0] > .99


def test_negative_logits_do_not_acquire_zero_from_invalid_windows_or_max_initialization():
    keys = [torch.full((1, 32, 1, 1), -4.), torch.full((1, 32, 1, 1), -8.)]
    inputs = fixture_inputs([0, 32], [32, 32], 64, keys, [True, True])
    observed = overlap_window_scores(torch.ones(1, 1, 1, 1), *inputs)
    # The second block's best candidate is the cross-boundary mean -6.
    expected = torch.tensor([1/(1+math.exp(-2)), 1/(1+math.exp(2))])
    torch.testing.assert_close(observed, expected)


def test_transport_exclusion_keeps_window_valid_and_stays_in_score_softmax_domain():
    score_mask, transport_mask = [True, True], [True, False]
    keys = [torch.zeros(1, 32, 1, 1), torch.full((1, 32, 1, 1), 4.)]
    inputs = fixture_inputs([0, 32], [32, 32], 64, keys, score_mask)
    assert not transport_mask[1] and bool(inputs[2][3])
    scores = overlap_window_scores(torch.ones(1, 1, 1, 1), *inputs)
    torch.testing.assert_close(scores, torch.tensor([1/(1+math.exp(2)), 1/(1+math.exp(-2))]))
    # Applying the transport mask as a score mask would give a different seed.
    wrong_inputs = fixture_inputs([0, 32], [32, 32], 64, keys, transport_mask)
    wrong = overlap_window_scores(torch.ones(1, 1, 1, 1), *wrong_inputs)
    torch.testing.assert_close(wrong, torch.tensor([1., 0.]))


def test_only_nonempty_blocks_without_valid_windows_use_original_mean_k_fallback():
    keys = [torch.full((1, 7, 1, 1), -2.), torch.full((1, 32, 1, 1), -4.)]
    inputs = fixture_inputs([0, 32], [7, 32], 64, keys, [True, True])
    assert not bool(inputs[2][:4].any())
    # Covered blocks must not accidentally mix the fallback logits into max.
    inputs[3][:, 1] = 100.
    observed = overlap_window_scores(torch.ones(1, 1, 1, 1), *inputs)
    expected = torch.tensor([1/(1+math.exp(-2)), 1/(1+math.exp(2))])
    torch.testing.assert_close(observed, expected)


def test_max_precedes_head_layer_and_query_aggregation():
    q = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
                      [[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]]])
    means = torch.full((2, 8, 1, 2), -torch.inf)
    means[:, 0, 0] = torch.tensor([4., 0.])
    means[:, 1, 0] = torch.tensor([0., 4.])
    means[:, 4, 0] = torch.tensor([3., 3.])
    targets = torch.full((8, 2), -1, dtype=torch.long)
    targets[0, 0] = targets[1, 0] = 0
    targets[4, 0] = 1
    valid = torch.tensor([True, True, False, False, True, False, False, False])
    observed = overlap_window_scores(q, means, targets, valid, torch.zeros(2, 2, 1, 2),
        torch.tensor([True, True]), scale=1.)
    # Every head/query/layer has a different winning window but logits [4,3].
    expected = torch.tensor([2/(1+math.exp(-1)), 2/(1+math.exp(1))])
    torch.testing.assert_close(observed, expected)
    # Premature aggregation makes the two first windows score 2, below 3.
    wrong = torch.tensor([2/(1+math.exp(1)), 2/(1+math.exp(-1))])
    assert observed[0] > observed[1] and wrong[0] < wrong[1]


def test_all_score_excluded_returns_zero_without_nan_from_negative_infinity():
    keys = [torch.ones(1, 9, 1, 2)]
    inputs = fixture_inputs([0], [9], 9, keys, [False])
    scores = overlap_window_scores(torch.zeros(1, 2, 1, 2), *inputs)
    torch.testing.assert_close(scores, torch.zeros(1))


@pytest.mark.skipif(os.environ.get('KVMEM_TEST_QUERY_WINDOW_CUDA') != '1',
    reason='GPU validation requires explicit opt-in to avoid interfering with experiments')
def test_cuda_matches_independent_scalar_oracle_with_cpu_cached_means():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(84)
    keys = [torch.randn(2, count, 2, 4, generator=generator).to(torch.float8_e4m3fn) for count in (32, 32, 5)]
    q = torch.randn(2, 9, 4, 4, generator=generator)
    inputs = fixture_inputs([0, 32, 64], [32, 32, 5], 69, keys, [True, True, True], mean_dtype=torch.float16)
    expected = scalar_oracle(q, [0, 32, 64], [32, 32, 5], 69, keys, [True, True, True],
        inputs[3], mean_dtype=torch.float16, query_half=True)
    observed = overlap_window_scores(q.cuda(), *inputs, query_half=True)
    torch.testing.assert_close(observed.cpu(), expected, rtol=2e-6, atol=2e-6)
