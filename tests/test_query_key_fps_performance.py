"""Optimization must preserve original Key choices, scores and owned lifetimes."""
from types import SimpleNamespace as NS
import os

import pytest
import torch

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.query_key_fps import (KeyFPSCache, key_fps_scores,
    select_key_representatives, select_key_representatives_optimized)


def setup(tmp_path):
    config = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=64,
        repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 3*32*2*4*2, 16, gather_workers=2)
    authority = RequestAuthority('fps', config, archive, 3, 2, 4,
        scoring_layers=2, raw_dtype=torch.float8_e4m3fn, resident_records=4,
        query_score_mode='key_fps_4', query_window_cache_bytes=1<<20)
    generator = torch.Generator().manual_seed(91)
    raw = torch.randn(3, 67, 2, 4, generator=generator).to(torch.float8_e4m3fn)
    value = torch.full((3, 67, 2, 4), 255, dtype=torch.uint8)
    assert authority.append(raw, value, history_begin=0, local_rope_begin=0)
    assert authority.flush()
    for job, _ in authority.writes:
        job.result()
    authority.poll()
    records = sorted(authority.records.values(), key=lambda r:r.history_begin)
    return authority, archive, records, raw


def test_key_only_region_excludes_v_draft_and_keeps_partial_tail_and_input_order(tmp_path, monkeypatch):
    a, archive, records, raw = setup(tmp_path)
    calls = []
    original = archive.read_region
    def read(*args, **kwargs):
        calls.append((args[2], args[3], len(args[1])))
        return original(*args, **kwargs)
    monkeypatch.setattr(archive, 'read_region', read)
    packed, stats = a.read_score_keys([records[i].handle for i in (2, 0, 1)])
    assert calls == [(0, 2*32*2*4, 2)]
    for out, old in enumerate((2, 0, 1)):
        count = records[old].count
        assert torch.equal(packed[out, :, :count].view(torch.uint8),
            raw[:2, old*32:old*32+count].view(torch.uint8))
        assert bool((packed[out, :, count:].float() == 0).all())
    assert stats['value_read_bytes'] == stats['draft_key_read_bytes'] == 0
    assert stats['key_read_bytes'] == 2*2*32*2*4
    assert stats['key_resident_bytes'] == 2*3*2*4
    assert a.close() and archive.close()
    # The output remains valid after archive ownership and all read leases end.
    assert torch.equal(packed[1].view(torch.uint8), raw[:2, :32].view(torch.uint8))


def test_key_only_queue_failure_releases_every_admitted_lease(tmp_path, monkeypatch):
    a, archive, records, _ = setup(tmp_path)
    monkeypatch.setattr(archive, 'read_region', lambda *args, **kwargs:None)
    with pytest.raises(BlockingIOError, match='queue is full'):
        a.read_score_keys([r.handle for r in records])
    assert a.close() and archive.close()


def test_key_only_rejects_duplicates_before_read_admission(tmp_path):
    a, archive, records, _ = setup(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        a.read_score_keys([records[0].handle]*2)
    with pytest.raises(ValueError, match='one to 32'):
        a.read_score_keys([records[0].handle]*33)
    assert a.close() and archive.close()


def test_optimized_cpu_cache_matches_reference_and_reports_fallback(tmp_path):
    a, archive, records, _ = setup(tmp_path)
    old = KeyFPSCache(archive, 2, 2, 4, dtype=torch.float32, max_bytes=1<<20)
    optimized = a.query_window_cache
    with old.use():
        old_keys, old_ids, _ = old.snapshot(a, records)
    with optimized.use():
        keys, ids, stats = optimized.snapshot(a, records, optimized=True, device='cpu')
    assert torch.equal(keys, old_keys) and torch.equal(ids, old_ids)
    assert stats['optimized'] and stats['fps_gpu_groups'] == 0
    assert stats['fps_cpu_fallback_groups'] == 2*3*2
    assert stats['fps_fallback_reasons']
    with optimized.use():
        _, _, hot = optimized.snapshot(a, records, optimized=True, device='cpu')
    assert hot['cache_hit_blocks'] == 3 and hot['key_read_bytes'] == 0
    assert hot['fps_cpu_fallback_groups'] == 0
    old.clear()
    assert a.close() and archive.close()


CUDA = pytest.mark.skipif(os.getenv('KVMEM_TEST_CUDA') != '1', reason='Explicit GPU component opt-in')


@CUDA
def test_gpu_optimized_fps_exact_original_indices_and_tie_fallback():
    torch.manual_seed(245)
    keys = torch.randn(2, 5, 32, 2, 256).to(torch.float8_e4m3fn)
    keys[:, 0] = 0
    counts = [32, 31, 3, 1, 0]
    expected, expected_indices = select_key_representatives(keys, counts)
    actual, indices, stats = select_key_representatives_optimized(keys, counts, device='cuda')
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(indices, expected_indices)
    assert stats['fps_gpu_groups'] == 20 and stats['fps_cpu_fallback_groups'] >= 4
    assert stats['fps_fallback_reason'] == 'near-tie FP32 reduction guard'


@CUDA
def test_score_device_packing_preserves_exact_probabilities_with_gqa_and_query_tiling():
    torch.manual_seed(801)
    keys = torch.randn(2, 9, 32, 2, 256).to(torch.float8_e4m3fn)
    reps, ids = select_key_representatives(keys, [32]*8+[3])
    reps = reps.half()
    q = torch.randn(2, 11, 6, 256, device='cuda')
    mask = torch.tensor([False]+[True]*8, device='cuda')
    old = key_fps_scores(q, reps, ids, mask, query_half=True)
    new = key_fps_scores(q, reps, ids, mask, query_half=True, optimized=True)
    assert torch.equal(old, new)


@CUDA
def test_gpu_key_only_cache_pinned_lifecycle_and_mask_independence(tmp_path):
    a, archive, records, _ = setup(tmp_path)
    cache = a.query_window_cache
    with cache.use():
        reps, ids, stats = cache.snapshot(a, records, optimized=True, device='cuda')
    q = torch.randn(2, 2, 4, 4, device='cuda')
    full = key_fps_scores(q, reps, ids, torch.ones(3, dtype=torch.bool), optimized=True)
    masked = key_fps_scores(q, reps, ids, torch.tensor([True, False, True]), optimized=True)
    assert full.sum().item() == pytest.approx(2., abs=1e-6)
    assert masked[1] == 0 and stats['value_read_bytes'] == 0
    assert stats['fps_gpu_groups'] == 12
    assert a.close() and archive.close()
