"""Independent representative/attention oracles and bounded-cache lifecycle."""
import math
import os
from threading import RLock
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_key_fps import (
    KeyFPSCache, RawKeyFPSCache, key_fps_scores, score_authority_fps,
    select_key_representatives,
)
from vllm_kvmem.query_window_scoring import WindowMeanCache


def selected_offsets_oracle(keys, counts):
    """Scalar finite candidates; no vectorized selection/gather helper."""
    layers, blocks, _, heads, _ = keys.shape
    result = torch.full((layers, blocks, 4, heads), -1, dtype=torch.long)
    values = keys.float().cpu().tolist()
    for layer in range(layers):
        for block, count in enumerate(counts):
            for head in range(heads):
                vectors = [values[layer][block][t][head] for t in range(count)]
                chosen = []
                for slot in range(min(count, 4)):
                    if not chosen:
                        rank = lambda t: sum(v*v for v in vectors[t])
                    else:
                        rank = lambda t: min(sum((a-b)**2 for a, b in zip(vectors[t], vectors[s]))
                            for s in chosen)
                    token = max((t for t in range(count) if t not in chosen),
                        key=lambda t: (rank(t), -t))
                    chosen.append(token)
                    result[layer, block, slot, head] = token
    return result


def score_oracle(queries, representatives, indices, mask, *, scale=None, query_half=False):
    layers, rows, heads, dim = queries.shape
    blocks, kvheads = representatives.shape[1], representatives.shape[3]
    q = (queries.half() if query_half else queries).float().cpu().tolist()
    k, indices = representatives.float().cpu().tolist(), indices.cpu().tolist()
    scale = dim**-.5 if scale is None else scale
    answer = [0.] * blocks
    for layer in range(layers):
        for row in range(rows):
            for head in range(heads):
                kvhead = head // (heads // kvheads)
                logits = []
                for block in range(blocks):
                    if not mask[block]:
                        logits.append(-math.inf)
                        continue
                    values = [sum(a*b for a, b in zip(q[layer][row][head], k[layer][block][slot][kvhead])) * scale
                        for slot in range(4) if indices[layer][block][slot][kvhead] >= 0]
                    logits.append(max(values))
                if not any(mask):
                    continue
                top = max(logits)
                weights = [math.exp(value-top) for value in logits]
                normalizer = sum(weights)
                for block, weight in enumerate(weights):
                    answer[block] += weight / normalizer / (layers*heads)
    return torch.tensor(answer, dtype=torch.float32)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def test_selection_matches_scalar_oracle_per_layer_head_and_keeps_real_original_keys(dtype):
    g = torch.Generator().manual_seed(91)
    keys = torch.randint(-7, 8, (3, 5, 32, 2, 6), generator=g).to(dtype)
    counts = [32, 17, 3, 1, 0]
    representatives, indices = select_key_representatives(keys, counts)
    expected = selected_offsets_oracle(keys, counts)
    assert representatives.dtype == dtype
    torch.testing.assert_close(indices, expected)
    for layer in range(3):
        for block, count in enumerate(counts):
            for head in range(2):
                offsets = indices[layer, block, :, head].tolist()
                valid = [offset for offset in offsets if offset >= 0]
                assert len(valid) == len(set(valid)) == min(count, 4)
                for slot, offset in enumerate(offsets):
                    actual = representatives[layer, block, slot, head].float()
                    target = keys[layer, block, offset, head].float() if offset >= 0 else torch.zeros(6)
                    torch.testing.assert_close(actual, target, rtol=0, atol=0)


def test_ties_choose_original_position_and_zero_keys_remain_distinct_valid_tokens():
    keys = torch.zeros(1, 2, 32, 1, 2)
    keys[0, 0, :4, 0] = torch.tensor([[2., 0.], [-2., 0.], [0., 2.], [0., -2.]])
    _, indices = select_key_representatives(keys, [4, 32])
    assert indices[0, 0, :, 0].tolist() == [0, 1, 2, 3]
    assert indices[0, 1, :, 0].tolist() == [0, 1, 2, 3]


def test_direct_differences_preserve_small_distances_between_large_keys():
    keys = torch.tensor([10000., 10001., 10002., 10003., 10004.]).reshape(1, 1, 5, 1, 1)
    representatives, indices = select_key_representatives(keys, [5])
    assert indices.flatten().tolist() == [4, 0, 2, 1]
    assert representatives.flatten().tolist() == [10004., 10000., 10002., 10001.]


def test_masked_padding_nan_does_not_become_candidate_but_valid_nan_fails():
    keys = torch.full((1, 1, 32, 1, 1), torch.nan)
    keys[:, :, :2] = torch.tensor([-5., -3.]).reshape(1, 1, 2, 1, 1)
    representatives, indices = select_key_representatives(keys, [2])
    assert indices.flatten().tolist() == [0, 1, -1, -1]
    assert representatives.flatten().tolist() == [-5., -3., 0., 0.]
    with pytest.raises(ValueError, match='Nonfinite original Key'):
        select_key_representatives(keys, [3])


@pytest.mark.parametrize('count', [-1, 33, 2.5, True])
def test_bad_counts_are_rejected(count):
    with pytest.raises(ValueError, match='counts'):
        select_key_representatives(torch.zeros(1, 1, 32, 1, 1), [count])


def test_fp32_distance_overflow_is_explicit_failure():
    keys = torch.tensor([1e20, -1e20]).reshape(1, 1, 2, 1, 1)
    with pytest.raises(ValueError, match='FP32 raw-Key norm'):
        select_key_representatives(keys, [2])


@pytest.mark.parametrize('query_half,scale', [(False, None), (True, .31)])
@pytest.mark.parametrize('query_chunk', [1, 3, 8])
def test_fp8_gqa_partial_blocks_full_domain_and_query_tiling_match_scalar_oracle(query_half, scale, query_chunk):
    g = torch.Generator().manual_seed(59)
    keys = torch.randn(2, 4, 32, 2, 5, generator=g).to(torch.float8_e4m3fn)
    representatives, indices = select_key_representatives(keys, [32, 3, 1, 0])
    representatives = representatives.half()  # Lossless storage of FP8 raw Keys.
    q = torch.randn(2, 9, 6, 5, generator=g)
    mask = torch.tensor([True, False, True, False])
    actual = key_fps_scores(q, representatives, indices, mask,
        query_half=query_half, scale=scale, query_chunk=query_chunk)
    expected = score_oracle(q, representatives, indices, mask.tolist(),
        query_half=query_half, scale=scale)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert float(actual.sum()) == pytest.approx(9., abs=2e-6)
    assert actual[1] == actual[3] == 0


def test_negative_logits_invalid_zero_slots_do_not_raise_a_partial_block_score():
    keys = torch.tensor([-4., -8.]).reshape(1, 2, 1, 1, 1)
    representatives, indices = select_key_representatives(keys, [1, 1])
    actual = key_fps_scores(torch.ones(1, 1, 1, 1), representatives, indices,
        torch.tensor([True, True]))
    expected = torch.tensor([1/(1+math.exp(-4)), 1/(1+math.exp(4))])
    torch.testing.assert_close(actual, expected)
    # Invalid values must also be masked after dot product, including NaN.
    representatives[:, :, 1:] = torch.nan
    torch.testing.assert_close(key_fps_scores(torch.ones(1, 1, 1, 1), representatives,
        indices, torch.tensor([True, True])), expected)


def test_max_precedes_layer_head_and_query_aggregation():
    q = torch.tensor([[[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]],
                      [[[0., 1.], [1., 0.]], [[1., 0.], [0., 1.]]]])
    representatives = torch.zeros(2, 2, 4, 1, 2)
    representatives[:, 0, 0, 0] = torch.tensor([4., 0.])
    representatives[:, 0, 1, 0] = torch.tensor([0., 4.])
    representatives[:, 1, 0, 0] = torch.tensor([3., 3.])
    indices = torch.full((2, 2, 4, 1), -1, dtype=torch.long)
    indices[:, 0, :2, 0] = torch.tensor([0, 1])
    indices[:, 1, 0, 0] = 0
    actual = key_fps_scores(q, representatives, indices, torch.tensor([True, True]), scale=1.)
    torch.testing.assert_close(actual, torch.tensor([2/(1+math.exp(-1)), 2/(1+math.exp(1))]))
    assert actual[0] > actual[1]  # Aggregate-before-max would reverse this rank.


def test_transport_is_not_the_normalization_domain_and_all_excluded_is_zero():
    keys = torch.tensor([0., 4.]).reshape(1, 2, 1, 1, 1)
    representatives, indices = select_key_representatives(keys, [1, 1])
    q = torch.ones(1, 1, 1, 1)
    transport = torch.tensor([True, False])
    actual = key_fps_scores(q, representatives, indices, torch.tensor([True, True]))
    wrong = key_fps_scores(q, representatives, indices, transport)
    assert actual[1] > .98 and wrong.tolist() == [1., 0.]
    torch.testing.assert_close(key_fps_scores(q, representatives, indices,
        torch.tensor([False, False])), torch.zeros(2))


def test_scoreable_block_without_any_representative_fails_explicitly():
    keys = torch.zeros(1, 1, 32, 1, 1)
    representatives, indices = select_key_representatives(keys, [0])
    with pytest.raises(ValueError, match='no valid Key representative'):
        key_fps_scores(torch.ones(1, 1, 1, 1), representatives, indices, torch.tensor([True]))


def test_scoring_scratch_limit_checked_before_layer_materialization(monkeypatch):
    import vllm_kvmem.query_key_fps as module
    monkeypatch.setattr(module, 'SCRATCH_LIMIT', 1)
    representatives, indices = select_key_representatives(torch.ones(1, 1, 1, 1, 1), [1])
    with pytest.raises(MemoryError, match='512 MiB'):
        key_fps_scores(torch.ones(1, 1, 1, 1), representatives, indices, torch.tensor([True]))


def synthetic(length=71, archive=None, value=1., *, dtype=torch.float32):
    keys = (torch.arange(length, dtype=torch.float32)[None, :, None, None]*value).to(dtype)
    records = [NS(history_begin=p, count=min(32, length-p),
        handle=NS(slot=p//32, generation=1)) for p in range(0, length, 32)]
    reads = []
    def read(handles):
        reads.append(tuple(h.slot for h in handles))
        return [(keys[:, h.slot*32:h.slot*32+records[h.slot].count], None) for h in handles]
    authority = NS(history_end=length, read_records=read, mean_k_backend='torch_reference')
    return authority, archive or NS(_lock=RLock()), records, reads, keys


def test_cache_reuses_keys_and_only_own_block_count_generation_changes_invalidate():
    a, archive, records, reads, keys = synthetic()
    cache = a.query_window_cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=20000)
    representatives, indices, stats = cache.snapshot(a, records)
    assert stats['cache_rebuilt_blocks'] == 3 and len(reads) == 1
    assert representatives[0, 0, 0, 0, 0] == 31 and indices[0, 0, 0, 0] == 31
    means, q = torch.zeros(1, 3, 1, 1), torch.ones(1, 1, 1, 1)
    first, _ = score_authority_fps(a, records, q, means, [True]*3)
    masked, stats = score_authority_fps(a, records, q, means, [True, True, False])
    assert first[2] > 0 and masked[2] == 0 and len(reads) == 1
    assert stats['method'] == 'key_fps_4' and stats['selection'] == 'raw_key_farthest_first'
    records[2].count = 3
    a.history_end = 67
    _, indices, stats = cache.snapshot(a, records)
    assert stats['cache_hit_blocks'] == 2 and stats['cache_rebuilt_blocks'] == 1
    assert indices[0, 2, :, 0].tolist()[-1] == -1
    assert stats['valid_representatives'] == 11 and stats['distinct_token_offsets_min'] == 3
    records[2].handle.generation += 1
    keys[:, 64:] = -100
    representatives, _, stats = cache.snapshot(a, records)
    assert stats['cache_rebuilt_blocks'] == 1 and representatives[0, 2, 0, 0, 0] == -100
    cache.clear()
    assert archive._query_window_cache_bytes == 0 and cache.indices is None


def test_logical_holes_never_supply_representatives_to_neighbor_blocks():
    a, archive, records, reads, _ = synthetic(96)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=20000)
    selected_records = [records[0], records[2]]
    representatives, _, stats = cache.snapshot(a, selected_records)
    assert reads == [(0, 2)] and stats['cache_rebuilt_blocks'] == 2
    assert representatives[0, :, 0, 0, 0].tolist() == [31., 95.]
    cache.clear()


def test_shrink_invalidate_slots_and_clear_rebuilds_new_history_branch():
    a, archive, records, _, keys = synthetic(96)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=20000)
    cache.snapshot(a, records)
    cache.snapshot(a, records[:2])
    assert cache.signatures[2] is None
    keys[:, 64:] = -12
    records[2].handle.generation += 1
    representatives, _, stats = cache.snapshot(a, records)
    assert stats['cache_rebuilt_blocks'] == 1 and representatives[0, 2, 0, 0, 0] == -12
    cache.clear()
    _, _, stats = cache.snapshot(a, records)
    assert stats['cache_rebuilt_blocks'] == 3
    cache.clear()


def test_owner_isolation_shared_capacity_accounts_offsets_and_prevents_active_eviction():
    a, archive, records, _, _ = synthetic(64)
    b, _, other, _, _ = synthetic(64, archive, value=-3.)
    capacity = 2*(4*4+2048+4)  # FP32 representatives, metadata, int8 offsets.
    ca = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=capacity)
    cb = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=capacity)
    with ca.use():
        first, _, _ = ca.snapshot(a, records)
        assert first[0, 1, 0, 0, 0] == 63
        with cb.use(), pytest.raises(MemoryError, match='active scoring'):
            cb.snapshot(b, other)
        assert ca.bytes == archive._query_window_cache_bytes == capacity
        del first
    with cb.use():
        second, _, _ = cb.snapshot(b, other)
        assert ca.bytes == 0 and ca.indices is None and second[0, 1, 0, 0, 0] == -189
        del second
    cb.clear()
    assert archive._query_window_cache_bytes == 0


def test_window_and_real_key_caches_share_one_pool_instead_of_double_reserving():
    a, archive, records, _, _ = synthetic(64)
    capacity = 2*(4*4+2048+4)
    window = WindowMeanCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=capacity)
    fps = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=capacity)
    with window.use():
        means, _, _ = window.snapshot(a, records)
        with fps.use(), pytest.raises(MemoryError, match='active scoring'):
            fps.snapshot(a, records)
        del means
    with fps.use():
        representatives, _, _ = fps.snapshot(a, records)
        assert window.bytes == 0 and archive._query_window_cache_bytes == fps.bytes == capacity
        del representatives
    fps.clear()


def test_one_owner_index_cannot_exceed_reserved_cache_budget():
    a, archive, records, reads, _ = synthetic(64)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=2068)
    with pytest.raises(MemoryError, match='query_window_cache_bytes'):
        cache.snapshot(a, records)
    assert not reads and cache.bytes == archive._query_window_cache_bytes == 0


def test_archive_failure_preserves_completed_batches_and_only_retries_unpublished_suffix(monkeypatch):
    a, archive, records, reads, _ = synthetic(40*32)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=200000)
    original, calls = a.read_records, 0
    def fail_second(handles):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('second batch failed')
        return original(handles)
    monkeypatch.setattr(a, 'read_records', fail_second)
    with cache.use(), pytest.raises(OSError, match='second batch'):
        cache.snapshot(a, records)
    assert cache.active == 0 and all(cache.signatures[:32])
    assert all(s is None for s in cache.signatures[32:])
    monkeypatch.setattr(a, 'read_records', original)
    _, _, stats = cache.snapshot(a, records)
    assert stats['cache_hit_blocks'] == 32 and stats['cache_rebuilt_blocks'] == 8
    assert reads == [tuple(range(32)), tuple(range(32, 40))]
    cache.clear()


def test_bad_key_never_publishes_a_half_built_cache_batch_and_retry_is_safe():
    a, archive, records, _, keys = synthetic(96)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=20000)
    original = keys[:, 88].clone()
    keys[:, 88] = torch.nan
    with pytest.raises(ValueError, match='Nonfinite original Key'):
        cache.snapshot(a, records)
    assert all(s is None for s in cache.signatures)
    keys[:, 88] = original
    _, _, stats = cache.snapshot(a, records)
    assert stats['cache_rebuilt_blocks'] == 3
    cache.clear()


def test_fp16_cache_preserves_fp8_original_values_exactly():
    a, archive, records, _, keys = synthetic(71, value=.25, dtype=torch.float8_e4m3fn)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float16, max_bytes=20000)
    representatives, indices, _ = cache.snapshot(a, records)
    assert representatives.dtype == torch.float16
    for block, record in enumerate(records):
        for slot in range(min(record.count, 4)):
            offset = int(indices[0, block, slot, 0])
            assert representatives[0, block, slot, 0, 0].float() == keys[0, record.history_begin+offset, 0, 0].float()
    cache.clear()


def test_inexact_index_dtype_conversion_fails_instead_of_changing_original_key():
    a, archive, records, _, _ = synthetic(32, value=.1)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float16, max_bytes=20000)
    with pytest.raises(ValueError, match='dtype changes an original Key'):
        cache.snapshot(a, records)
    assert all(s is None for s in cache.signatures)
    cache.clear()


def test_empty_history_cache_returns_empty_shapes_without_allocation():
    a, archive, records, reads, _ = synthetic(0)
    cache = KeyFPSCache(archive, 1, 1, 1, dtype=torch.float32, max_bytes=2068)
    representatives, indices, stats = cache.snapshot(a, records)
    assert representatives.shape == (1, 0, 4, 1, 1) and indices.shape == (1, 0, 4, 1)
    assert stats['valid_representatives'] == 0 and not reads and cache.bytes == 0
    scores = key_fps_scores(torch.ones(1, 1, 1, 1), representatives, indices,
        torch.zeros(0, dtype=torch.bool))
    assert scores.numel() == 0


@pytest.mark.skipif(os.environ.get('KVMEM_TEST_QUERY_KEY_FPS_CUDA') != '1',
    reason='GPU validation requires explicit opt-in')
def test_cuda_selection_and_gqa_scores_match_cpu_and_independent_oracle():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    g = torch.Generator().manual_seed(17)
    keys = torch.randint(-7, 8, (2, 3, 32, 2, 4), generator=g).to(torch.float8_e4m3fn)
    q = torch.randn(2, 9, 4, 4, generator=g)
    representatives, indices = select_key_representatives(keys, [32, 3, 1])
    gpu_keys, gpu_indices = select_key_representatives(keys.cuda(), [32, 3, 1])
    torch.testing.assert_close(gpu_indices.cpu(), indices)
    torch.testing.assert_close(gpu_keys.float().cpu(), representatives.float(), rtol=0, atol=0)
    mask = torch.tensor([True, False, True])
    actual = key_fps_scores(q.cuda(), representatives.half(), indices, mask, query_half=True)
    expected = score_oracle(q, representatives.half(), indices, mask.tolist(), query_half=True)
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)


@pytest.mark.skipif(os.environ.get('KVMEM_TEST_QUERY_KEY_FPS_CUDA') != '1',
    reason='GPU memory validation requires explicit opt-in')
def test_cuda_256k_geometry_scratch_stays_below_512_mib():
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    representatives = torch.ones((1, 8192, 4, 4, 256), dtype=torch.float16)
    indices = torch.arange(4, dtype=torch.int8)[None, None, :, None].expand(1, 8192, 4, 4)
    q = torch.ones(1, 8, 24, 256, device='cuda')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    scores = key_fps_scores(q, representatives, indices, torch.ones(8192, dtype=torch.bool), query_half=True)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated()-baseline <= 512*1024**2
    torch.testing.assert_close(scores.cpu(), torch.full((8192,), 8/8192))
