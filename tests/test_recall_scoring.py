"""Independent scalar oracles for the five fixed-policy recall scorers."""
import math
from dataclasses import replace

import pytest
import torch

from vllm_kvmem.serving.recall_scoring import (
    METHODS, build_block_features, block_logits, concat_block_features, prepare_queries,
)


def scalar_fixed4(vectors):
    """Direct port of scalar seed/assignment rules, independent of tensor code."""
    dim = len(vectors[0])
    mean = [sum(v[d] for v in vectors) / len(vectors) for d in range(dim)]

    def cosine(left, right):
        return (sum(a * b for a, b in zip(left, right))
                / math.sqrt(max(sum(a * a for a in left), 1e-20))
                / math.sqrt(max(sum(a * a for a in right), 1e-20)))

    seeds = [max(range(len(vectors)), key=lambda i: (cosine(vectors[i], mean), -i))]
    best = [cosine(v, vectors[seeds[0]]) for v in vectors]
    assignment = [0] * len(vectors)
    for slot in range(1, 4):
        available = [i for i in range(len(vectors)) if i not in seeds]
        seed = min(available, key=lambda i: (best[i], i)) if available else seeds[0]
        seeds.append(seed)
        for i, vector in enumerate(vectors):
            value = cosine(vector, vectors[seed])
            if value > best[i]:
                best[i], assignment[i] = value, slot
    result = []
    for slot in range(4):
        members = [v for v, assigned in zip(vectors, assignment) if assigned == slot]
        result.append([sum(v[d] for v in members) / len(members) for d in range(dim)]
                      if members else vectors[seeds[0]])
    return result


def scalar_fps4(vectors):
    chosen = []
    for _ in range(min(4, len(vectors))):
        def rank(i):
            value = (min(sum((a - b) ** 2 for a, b in zip(vectors[i], vectors[j]))
                         for j in chosen) if chosen else sum(v * v for v in vectors[i]))
            return value, -i
        chosen.append(max((i for i in range(len(vectors)) if i not in chosen), key=rank))
    return [vectors[i] for i in chosen]


def scalar_logits(keys, counts, queries, method, *, result_dtype=torch.float32):
    k, q = keys.double().tolist(), queries.double().tolist()
    layers, blocks, _, kvheads, dim = keys.shape
    rows, heads = queries.shape[1:3]
    result = torch.full((layers, rows, heads, blocks), -torch.inf, dtype=result_dtype)
    for layer in range(layers):
        for row in range(rows):
            for head in range(heads):
                query = q[layer][row][head]
                kh = head // (heads // kvheads)
                for block, count in enumerate(counts):
                    if not count:
                        continue
                    vectors = [k[layer][block][token][kh] for token in range(count)]
                    if method == 'mean_k':
                        vectors = [[sum(v[d] for v in vectors) / count for d in range(dim)]]
                    elif method == 'fixed4':
                        vectors = scalar_fixed4(vectors)
                    elif method == 'fps4':
                        vectors = scalar_fps4(vectors)
                    elif method == 'min_max':
                        result[layer, row, head, block] = sum(
                            query[d] * (max(v[d] for v in vectors) if query[d] >= 0 else
                                        min(v[d] for v in vectors)) for d in range(dim)) / math.sqrt(dim)
                        continue
                    scores = [sum(a * b for a, b in zip(query, v)) / math.sqrt(dim) for v in vectors]
                    top = max(scores)
                    result[layer, row, head, block] = (top + math.log(sum(math.exp(x - top) for x in scores))
                                                       if method == 'exact_mass' else top)
    return result


def frozen_v2_slot_logits(queries, features):
    """Independent math from the frozen runtime-v2 before batched matmul.

    Retain its per-slot einsum and sequential logaddexp, so new packing,
    representative reduction and summation-order errors cannot self-confirm.
    """
    values = features.values
    layers, rows, heads, dim = queries.shape
    blocks, representatives, kvheads = values.shape[1:4]
    groups = heads // kvheads
    q = queries.float().reshape(layers, rows, kvheads, groups, dim)
    scale = dim ** -.5

    def dot(query, key):
        return torch.einsum('lqkgd,lbkd->lqkgb', query, key).reshape(layers, rows, heads, blocks)

    def mask(slot):
        valid = features.valid[:, :, slot, :].permute(0, 2, 1)
        return valid[:, None, :, None, :].expand(layers, rows, kvheads, groups, blocks).reshape(
            layers, rows, heads, blocks)

    if features.method == 'min_max':
        return ((dot(q.clamp_min(0), values[:, :, 1])
                 + dot(q.clamp_max(0), values[:, :, 0])) * scale).masked_fill(~mask(0), -torch.inf)
    logits = torch.full((layers, rows, heads, blocks), -torch.inf, dtype=torch.float32)
    for slot in range(representatives):
        score = (dot(q, values[:, :, slot]) * scale).masked_fill(~mask(slot), -torch.inf)
        logits = torch.logaddexp(logits, score) if features.method == 'exact_mass' else torch.maximum(logits, score)
    return logits


@pytest.mark.parametrize('method', METHODS)
def test_all_scorers_match_independent_gqa_partial_block_oracle(method):
    generator = torch.Generator().manual_seed(260128)
    keys = torch.randn(2, 4, 32, 2, 7, generator=generator)
    queries = torch.randn(2, 5, 6, 7, generator=generator)
    counts = [32, 9, 1, 0]
    for block, count in enumerate(counts):
        keys[:, block, count:] = torch.nan
    features = build_block_features(keys, counts, method)
    actual = block_logits(queries, features)
    expected = scalar_logits(keys, counts, queries, method)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.isneginf(actual[..., -1]).all()


def test_exact_mass_is_global_token_softmax_then_block_sum_with_eligible_mask():
    generator = torch.Generator().manual_seed(105)
    keys = torch.randn(2, 4, 7, 2, 5, generator=generator)
    queries = torch.randn(2, 11, 4, 5, generator=generator)
    counts, eligible = [7, 3, 5, 0], [True, False, True, True]
    features = build_block_features(keys, counts, 'exact_mass')
    logits = block_logits(queries, features)
    probabilities = logits.masked_fill(~torch.tensor(eligible), -torch.inf).softmax(-1)
    expected = torch.zeros_like(probabilities)
    for layer in range(2):
        for row in range(11):
            for head in range(4):
                operands = [(block, keys[layer, block, t, head // 2])
                            for block, count in enumerate(counts) if eligible[block] for t in range(count)]
                token_logits = torch.tensor([
                    sum(float(a) * float(b) for a, b in zip(queries[layer, row, head], key)) / math.sqrt(5)
                    for _, key in operands], dtype=torch.float64)
                token_probs = token_logits.softmax(0)
                for (block, _), probability in zip(operands, token_probs):
                    expected[layer, row, head, block] += probability
    torch.testing.assert_close(probabilities, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(probabilities[..., 0]))


@pytest.mark.parametrize('method', METHODS)
def test_block_tiles_and_query_tiles_preserve_full_domain_scores(method):
    generator = torch.Generator().manual_seed(64)
    keys = torch.randn(2, 5, 32, 2, 4, generator=generator)
    queries = torch.randn(2, 19, 4, 4, generator=generator)
    counts = [32, 17, 31, 4, 1]
    whole = build_block_features(keys, counts, method)
    parts = [build_block_features(keys[:, :2], counts[:2], method),
             build_block_features(keys[:, 2:], counts[2:], method)]
    joined = concat_block_features(parts)
    torch.testing.assert_close(joined.values, whole.values, rtol=0, atol=0)
    torch.testing.assert_close(joined.valid, whole.valid, rtol=0, atol=0)
    expected = block_logits(queries, whole).softmax(-1).sum((0, 1, 2)) / (2 * 4)
    actual = torch.zeros(5)
    for chunk in queries.split(3, dim=1):
        logits = torch.cat([block_logits(chunk, part) for part in parts], dim=-1)
        actual += logits.softmax(-1).sum((0, 1, 2)) / (2 * 4)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-6)
    reordered = joined.select_blocks([4, 1, 0]).to('cpu')
    torch.testing.assert_close(block_logits(queries, reordered),
                               block_logits(queries, whole)[..., [4, 1, 0]])


def test_min_max_is_an_upper_bound_on_every_true_token_dot_product():
    generator = torch.Generator().manual_seed(56)
    keys = torch.randn(2, 3, 32, 2, 8, generator=generator)
    queries = torch.randn(2, 13, 6, 8, generator=generator)
    counts = [32, 7, 1]
    upper = block_logits(queries, build_block_features(keys, counts, 'min_max'))
    for block, count in enumerate(counts):
        for token in range(count):
            raw = keys[:, block, token].repeat_interleave(3, dim=1)
            exact = (queries * raw[:, None]).sum(-1) / math.sqrt(8)
            assert (upper[..., block] >= exact - 2e-6).all()


def test_fixed4_is_synthetic_cluster_means_and_has_four_slots_per_physical_block():
    keys = torch.zeros(1, 1, 64, 1, 2)
    keys[0, 0, :, 0] = torch.tensor([[1., 0.], [3., 0.], [0., 1.], [0., 3.],
                                    [-1., 0.], [-3., 0.], [0., -1.], [0., -3.]] * 8)
    fixed = build_block_features(keys, [64], 'fixed4')
    fps = build_block_features(keys, [64], 'fps4')
    assert fixed.values.shape == fps.values.shape == (1, 1, 4, 1, 2)
    expected = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    torch.testing.assert_close(fixed.values[0, 0, :, 0], expected, rtol=0, atol=0)
    for prototype in fixed.values[0, 0, :, 0]:
        assert not any(torch.equal(prototype, key) for key in keys[0, 0, :, 0])
    for representative in fps.values[0, 0, :, 0]:
        assert any(torch.equal(representative, key) for key in keys[0, 0, :, 0])


def test_fixed4_empty_groups_duplicate_first_seed_and_zero_blocks_remain_masked():
    keys = torch.full((1, 3, 32, 1, 2), torch.nan)
    keys[0, 0, :3, 0] = torch.tensor([[2., 0.], [4., 0.], [8., 0.]])
    keys[0, 1, :2, 0] = 0.
    features = build_block_features(keys, [3, 2, 0], 'fixed4')
    torch.testing.assert_close(features.values[0, 0, :, 0],
                               torch.tensor([[14. / 3, 0.], [2., 0.], [2., 0.], [2., 0.]]))
    assert features.valid[0, :2].all()
    assert not features.valid[0, 2].any()
    assert (features.values[0, 1:] == 0).all()


def test_fps_partial_slots_masked_and_ties_choose_earliest_real_token():
    keys = torch.zeros(1, 2, 32, 1, 2)
    keys[0, 0, :4, 0] = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    features = build_block_features(keys, [4, 2], 'fps4')
    torch.testing.assert_close(features.values[0, 0], keys[0, 0, :4], rtol=0, atol=0)
    assert features.valid[0, 1, :, 0].tolist() == [True, True, False, False]


@pytest.mark.parametrize('method', METHODS)
def test_fp8_inputs_embed_exactly_in_fp32_features(method):
    generator = torch.Generator().manual_seed(99)
    keys = torch.randint(-5, 6, (1, 2, 32, 1, 4), generator=generator).to(torch.float8_e4m3fn)
    actual = build_block_features(keys, [32, 5], method)
    expected = build_block_features(keys.float(), [32, 5], method)
    torch.testing.assert_close(actual.values, expected.values, rtol=0, atol=0)
    torch.testing.assert_close(actual.valid, expected.valid, rtol=0, atol=0)


@pytest.mark.parametrize('counts', [[-1], [33], [2.5], [True], [[2]], [2, 3]])
def test_invalid_token_counts_fail_closed(counts):
    with pytest.raises(ValueError, match='counts'):
        build_block_features(torch.zeros(1, 1, 32, 1, 4), counts, 'mean_k')


@pytest.mark.parametrize('method', METHODS)
def test_valid_nonfinite_keys_and_queries_fail_closed(method):
    keys = torch.zeros(1, 1, 32, 1, 4)
    keys[0, 0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite valid Key'):
        build_block_features(keys, [1], method)
    keys.zero_()
    features = build_block_features(keys, [1], method)
    with pytest.raises(ValueError, match='Nonfinite Query'):
        block_logits(torch.full((1, 1, 1, 4), torch.inf), features)


def test_unknown_method_and_incompatible_gqa_tiles_fail_closed():
    keys = torch.zeros(1, 1, 32, 2, 4)
    with pytest.raises(ValueError, match='Unknown recall method'):
        build_block_features(keys, [32], 'attention_fallback')
    mean = build_block_features(keys, [32], 'mean_k')
    with pytest.raises(ValueError, match='geometry'):
        block_logits(torch.zeros(1, 1, 3, 4), mean)
    with pytest.raises(ValueError, match='incompatible'):
        concat_block_features([mean, build_block_features(keys, [32], 'fps4')])


@pytest.mark.parametrize('method', METHODS)
def test_zero_query_has_uniform_block_mass_except_exact_mass_counts(method):
    keys = torch.zeros(1, 3, 32, 1, 2)
    features = build_block_features(keys, [32, 8, 0], method)
    probabilities = block_logits(torch.zeros(1, 1, 1, 2), features).softmax(-1).flatten()
    expected = torch.tensor([.8, .2, 0.]) if method == 'exact_mass' else torch.tensor([.5, .5, 0.])
    torch.testing.assert_close(probabilities, expected)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('groups', [1, 2, 6])
def test_vectorized_scores_match_frozen_v2_with_fp8_keys_fp16_queries_and_gqa(method, groups):
    rng = torch.Generator().manual_seed(130032)
    keys = torch.randn(2, 5, 32, 2, 13, generator=rng).to(torch.float8_e4m3fn).float()
    # This conversion is the experiment's unchanged BF16 capture -> FP16 Q.
    queries = torch.randn(2, 9, 2 * groups, 13, generator=rng).bfloat16().half()
    counts = [32, 19, 3, 1, 0]
    for block, count in enumerate(counts):
        keys[:, block, count:] = torch.nan
    features = build_block_features(keys, counts, method)
    expected = frozen_v2_slot_logits(queries, features)
    actual = block_logits(queries, features)
    prepared = prepare_queries(queries, 2, split_sign=method == 'min_max')
    reused = torch.cat([block_logits(prepared, features.select_blocks([0, 1]), validate=False),
                        block_logits(prepared, features.select_blocks([2, 3, 4]), validate=False)], -1)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    torch.testing.assert_close(reused, expected, atol=2e-5, rtol=1e-5)
    expected_mass = expected.softmax(-1).sum((0, 1, 2), dtype=torch.float64) / (2 * 2 * groups)
    actual_mass = reused.softmax(-1).sum((0, 1, 2), dtype=torch.float64) / (2 * 2 * groups)
    torch.testing.assert_close(actual_mass, expected_mass, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual_mass.sum(), torch.tensor(9., dtype=torch.float64), atol=2e-6, rtol=0)


@pytest.mark.parametrize('method', METHODS)
def test_vectorized_small_example_matches_independent_fp64_scalar_oracle(method):
    rng = torch.Generator().manual_seed(9821)
    keys = torch.randn(1, 4, 7, 2, 5, generator=rng)
    queries = torch.randn(1, 3, 6, 5, generator=rng)
    counts = [7, 3, 1, 0]
    actual = block_logits(prepare_queries(queries, 2, split_sign=True),
                          build_block_features(keys, counts, method))
    expected = scalar_logits(keys, counts, queries, method, result_dtype=torch.float64)
    torch.testing.assert_close(actual.double(), expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize('method', METHODS)
def test_prepared_full_128_query_window_has_no_scalar_reads_or_dynamic_gathers(method):
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            assert func is not torch.ops.aten._local_scalar_dense.default, 'Scoring read a device scalar'
            assert func is not torch.ops.aten.index.Tensor, 'Scoring used a dynamic gather'
            return func(*args, **(kwargs or {}))

    rng = torch.Generator().manual_seed(128)
    keys = torch.randn(1, 3, 32, 2, 8, generator=rng)
    queries = torch.randn(1, 128, 12, 8, generator=rng)
    features = build_block_features(keys, [32, 9, 0], method)
    expected = frozen_v2_slot_logits(queries, features)
    with NoHostRead():
        prepared = prepare_queries(queries, 2, split_sign=method == 'min_max', validate=False)
        actual = block_logits(prepared, features, validate=False)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)


@pytest.mark.parametrize('method', METHODS)
def test_vectorized_ties_preserve_equal_block_scores_and_first_argmax(method):
    key = torch.tensor([1., -2., 0., 3.]).view(1, 1, 1, 1, 4)
    keys = key.expand(2, 3, 32, 2, 4).clone()
    features = build_block_features(keys, [32, 32, 32], method)
    queries = torch.tensor([1., -1., 2., -2.]).view(1, 1, 1, 4).expand(2, 5, 6, 4)
    actual = block_logits(prepare_queries(queries, 2, split_sign=True), features)
    expected = frozen_v2_slot_logits(queries, features)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(actual[..., 0], actual[..., 1])
    assert torch.equal(actual[..., 0], actual[..., 2])
    assert not actual.argmax(-1).any()


def test_minmax_prepared_signs_and_lazy_signs_have_identical_scores():
    rng = torch.Generator().manual_seed(412)
    queries = torch.randn(2, 5, 6, 8, generator=rng)
    keys = torch.randn(2, 3, 32, 2, 8, generator=rng)
    features = build_block_features(keys, [32, 11, 1], 'min_max')
    lazy = prepare_queries(queries, 2)
    signs = prepare_queries(queries, 2, split_sign=True)
    assert lazy.positive is lazy.negative is None
    torch.testing.assert_close(signs.values, signs.positive + signs.negative, rtol=0, atol=0)
    torch.testing.assert_close(block_logits(lazy, features), block_logits(signs, features), rtol=0, atol=0)


def test_prepared_query_default_validation_and_geometry_stay_strict():
    features = build_block_features(torch.zeros(1, 1, 32, 2, 4), [32], 'mean_k')
    unchecked = prepare_queries(torch.full((1, 3, 4, 4), torch.inf), 2, validate=False)
    with pytest.raises(ValueError, match='Nonfinite Query'):
        block_logits(unchecked, features)
    prepared = prepare_queries(torch.ones(1, 3, 4, 4), 2)
    for invalid in (replace(prepared, shape=(1, 4, 4, 4)),
                    replace(prepared, kv_heads=1),
                    replace(prepared, values=prepared.values.double()),
                    replace(prepared, positive=torch.ones(1))):
        with pytest.raises(ValueError, match='geometry'):
            block_logits(invalid, features, validate=False)
    with pytest.raises(ValueError, match='geometry'):
        prepare_queries(torch.ones(1, 3, 5, 4), 2, validate=False)


@pytest.mark.parametrize('method', METHODS)
def test_vectorized_default_validation_rejects_finite_input_logit_overflow(method):
    features = build_block_features(torch.ones(1, 1, 32, 1, 4), [32], method)
    # Construct finite features whose QK overflow FP32, independently of the
    # feature builder's own overflow checks during clustering/FPS.
    features = replace(features, values=features.values * 1e30)
    queries = torch.full((1, 1, 2, 4), 1e30)
    with pytest.raises(ValueError, match='logit overflow'):
        block_logits(queries, features)
