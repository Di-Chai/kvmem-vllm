"""Real-Key cache publication must remain exact, bounded and retryable."""
import os
from threading import RLock
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_key_fps import (
    KeyFPSCache, select_key_representatives, select_key_representatives_optimized,
)


def fixture(*, blocks=6, dtype=torch.float8_e4m3fn, cache_dtype=torch.float16,
            archive=None, capacity=1 << 20, seed=572):
    """Only a bounded original K slab; no real archive/model/GPU ownership."""
    layers, heads, dim = 2, 2, 8
    raw = torch.randn(layers, blocks, 32, heads, dim,
        generator=torch.Generator().manual_seed(seed)).to(dtype)
    records = [NS(history_begin=i*32, count=3 if i == blocks-1 else 32,
        handle=NS(slot=i, generation=1)) for i in range(blocks)]
    reads, slabs = [], []

    def read_records(handles):
        reads.append(tuple(h.slot for h in handles))
        return [(raw[:, h.slot, :records[h.slot].count], None) for h in handles]

    def read_score_keys(handles, *, pin_memory=False):
        reads.append(tuple(h.slot for h in handles))
        packed = torch.zeros(len(handles), layers, 32, heads, dim,
            dtype=dtype, pin_memory=pin_memory)
        for out, handle in enumerate(handles):
            count = records[handle.slot].count
            packed[out, :, :count].copy_(raw[:, handle.slot, :count])
        slabs.append(packed)
        return packed, dict(key_read_bytes=packed.numel()*packed.element_size(),
            key_read_batches=1, value_read_bytes=0, draft_key_read_bytes=0)

    archive = archive or NS(_lock=RLock())
    authority = NS(history_end=(blocks-1)*32+3, read_records=read_records,
        read_score_keys=read_score_keys)
    cache = KeyFPSCache(archive, layers, heads, dim, dtype=cache_dtype, max_bytes=capacity)
    return NS(authority=authority, archive=archive, records=records, cache=cache,
        raw=raw, reads=reads, slabs=slabs)


def reference(case):
    keys, offsets = select_key_representatives(case.raw, [r.count for r in case.records])
    return keys.to(case.cache.dtype), offsets.to(torch.int8)


def snapshot(case, *, optimized=True, device='cpu'):
    with case.cache.use():
        return case.cache.snapshot(case.authority, case.records,
            optimized=optimized, device=device)


def change(case, rows):
    for row in rows:
        case.records[row].handle.generation += 1
        case.raw[:, row] = (case.raw[:, row].float()*2+3).to(case.raw.dtype)


@pytest.mark.parametrize('cache_dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('optimized', [False, True])
def test_cpu_publication_preserves_real_keys_offsets_partial_tail_and_owned_storage(cache_dtype, optimized):
    case = fixture(cache_dtype=cache_dtype)
    try:
        expected, offsets = reference(case)
        keys, actual_offsets, stats = snapshot(case, optimized=optimized)
        assert torch.equal(keys, expected) and torch.equal(actual_offsets, offsets)
        assert actual_offsets[:, -1, -1].eq(-1).all()
        assert keys[:, -1, -1].eq(0).all()
        # Published representatives must not alias an archive/read destination.
        case.raw.zero_()
        for slab in case.slabs:
            slab.zero_()
        assert torch.equal(keys, expected)
        assert stats['cache_rebuilt_blocks'] == len(case.records)
        assert stats['fps_device_cache_conversion_batches'] == 0
        assert stats['fps_publication_strict_check_batches'] == 1
        assert stats['fps_publication_slice_batches'] == int(optimized)
        assert stats['fps_publication_index_copy_batches'] == int(not optimized)
    finally:
        case.cache.clear()
    assert case.archive._query_window_cache_bytes == 0


@pytest.mark.parametrize('raw_dtype,cache_dtype,value', [
    (torch.float32, torch.float16, .1),
    (torch.bfloat16, torch.float16, 2.**-30),
    (torch.float16, torch.bfloat16, 1.001953125),
])
@pytest.mark.parametrize('optimized', [False, True])
def test_cpu_non_fp8_lossy_conversion_fails_before_any_valid_signature(raw_dtype, cache_dtype, value, optimized):
    case = fixture(dtype=raw_dtype, cache_dtype=cache_dtype)
    case.raw.fill_(value)
    try:
        with pytest.raises(ValueError, match='dtype changes an original Key'):
            snapshot(case, optimized=optimized)
        assert all(signature is None for signature in case.cache.signatures)
        # Failure leaves reserved space bounded and does not retain an active use.
        assert case.cache.active == 0
        assert case.archive._query_window_cache_bytes == case.cache.bytes <= case.cache.max_bytes
        case.raw.zero_()
        keys, offsets, stats = snapshot(case, optimized=optimized)
        expected, expected_offsets = reference(case)
        assert torch.equal(keys, expected) and torch.equal(offsets, expected_offsets)
        assert stats['cache_rebuilt_blocks'] == len(case.records)
    finally:
        case.cache.clear()


@pytest.mark.parametrize('dirty', [(1, 2, 3), (0, 2, 5)])
def test_contiguous_and_sparse_dirty_publications_leave_other_rows_intact_and_hot_cache_unread(dirty):
    case = fixture()
    try:
        keys, offsets, _ = snapshot(case)
        old_keys, old_offsets = keys.clone(), offsets.clone()
        old_signatures = list(case.cache.signatures)
        change(case, dirty)
        actual, actual_offsets, stats = snapshot(case)
        expected, expected_offsets = reference(case)
        assert torch.equal(actual, expected) and torch.equal(actual_offsets, expected_offsets)
        assert case.reads == [tuple(range(6)), dirty]
        assert stats['cache_rebuilt_blocks'] == len(dirty)
        contiguous = dirty[-1]-dirty[0]+1 == len(dirty)
        assert stats['fps_publication_slice_batches'] == int(contiguous)
        assert stats['fps_publication_index_copy_batches'] == int(not contiguous)
        for row in set(range(6))-set(dirty):
            assert torch.equal(actual[:, row], old_keys[:, row])
            assert torch.equal(actual_offsets[:, row], old_offsets[:, row])
            assert case.cache.signatures[row] == old_signatures[row]
        pointers = actual.data_ptr(), actual_offsets.data_ptr()
        hot_keys, hot_offsets, hot = snapshot(case)
        assert pointers == (hot_keys.data_ptr(), hot_offsets.data_ptr())
        assert hot['cache_hit_blocks'] == 6 and hot['key_read_bytes'] == 0
        assert hot['cache_rebuilt_blocks'] == 0 and len(case.reads) == 2
    finally:
        case.cache.clear()


@pytest.mark.parametrize('dirty', [(1, 2, 3), (0, 2, 5)])
def test_offset_write_failure_invalidates_the_whole_batch_without_touching_clean_rows(monkeypatch, dirty):
    case = fixture()
    try:
        keys, offsets, _ = snapshot(case)
        old_keys, old_offsets = keys.clone(), offsets.clone()
        old_signatures = list(case.cache.signatures)
        change(case, dirty)
        expected, expected_offsets = reference(case)
        offset_storage = case.cache.indices.untyped_storage().data_ptr()
        original_copy, original_index = torch.Tensor.copy_, torch.Tensor.index_copy_
        failures = []

        def fail_offset_copy(tensor, *args, **kwargs):
            if tensor.untyped_storage().data_ptr() == offset_storage:
                failures.append('copy')
                raise OSError('injected offset publication failure')
            return original_copy(tensor, *args, **kwargs)

        def fail_offset_index(tensor, *args, **kwargs):
            if tensor.untyped_storage().data_ptr() == offset_storage:
                failures.append('index')
                raise OSError('injected offset publication failure')
            return original_index(tensor, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'copy_', fail_offset_copy)
            patch.setattr(torch.Tensor, 'index_copy_', fail_offset_index)
            with pytest.raises(OSError, match='injected offset publication'):
                snapshot(case)
        assert len(failures) == 1
        assert case.cache.active == 0
        for row in dirty:
            # K writes happened, but incomplete K/offset pairs are never valid.
            assert torch.equal(case.cache.tensor[:, row], expected[:, row])
            assert case.cache.signatures[row] is None
        for row in set(range(6))-set(dirty):
            assert torch.equal(case.cache.tensor[:, row], old_keys[:, row])
            assert torch.equal(case.cache.indices[:, row], old_offsets[:, row])
            assert case.cache.signatures[row] == old_signatures[row]
        actual, actual_offsets, retry = snapshot(case)
        assert torch.equal(actual, expected) and torch.equal(actual_offsets, expected_offsets)
        assert retry['cache_rebuilt_blocks'] == len(dirty)
        assert case.reads == [tuple(range(6)), dirty, dirty]
    finally:
        case.cache.clear()


def test_later_batch_failure_preserves_completed_batch_and_rebuilds_only_unpublished_rows(monkeypatch):
    case = fixture(blocks=34)
    original_copy = torch.Tensor.copy_
    offset_writes = []

    def fail_second_offset_write(tensor, *args, **kwargs):
        indices = case.cache.indices
        if indices is not None and tensor.untyped_storage().data_ptr() == indices.untyped_storage().data_ptr():
            offset_writes.append(tensor.shape[1])
            if len(offset_writes) == 2:
                raise OSError('injected second batch publication failure')
        return original_copy(tensor, *args, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, 'copy_', fail_second_offset_write)
            with pytest.raises(OSError, match='second batch publication'):
                snapshot(case)
        assert offset_writes == [32, 2]
        assert all(signature is not None for signature in case.cache.signatures[:32])
        assert all(signature is None for signature in case.cache.signatures[32:])
        assert case.cache.active == 0
        keys, offsets, stats = snapshot(case)
        expected, expected_offsets = reference(case)
        assert torch.equal(keys, expected) and torch.equal(offsets, expected_offsets)
        assert stats['cache_hit_blocks'] == 32 and stats['cache_rebuilt_blocks'] == 2
        assert case.reads == [tuple(range(32)), (32, 33), (32, 33)]
    finally:
        case.cache.clear()


def test_optimized_publication_charges_one_shared_budget_and_never_evicts_active_owner():
    row_bytes = 2*4*2*8*2 + 2048 + 2*4*2
    capacity = 4*row_bytes
    first = fixture(blocks=4, capacity=capacity)
    second = fixture(blocks=4, capacity=capacity, archive=first.archive, seed=45)
    try:
        with first.cache.use():
            first_keys, first_offsets, _ = snapshot(first)
            before = first_keys.clone(), first_offsets.clone()
            with pytest.raises(MemoryError, match='active scoring owners'):
                snapshot(second)
            assert first.archive._query_window_cache_bytes == capacity
            assert torch.equal(first_keys, before[0]) and torch.equal(first_offsets, before[1])
            assert second.cache.bytes == 0 and second.cache.active == 0
            del first_keys, first_offsets
        second_keys, second_offsets, _ = snapshot(second)
        expected, offsets = reference(second)
        assert torch.equal(second_keys, expected) and torch.equal(second_offsets, offsets)
        assert first.cache.capacity == 0 and first.cache.indices is None
        assert first.archive._query_window_cache_bytes == capacity
    finally:
        first.cache.clear()
        second.cache.clear()
    assert first.archive._query_window_cache_bytes == 0


CUDA = pytest.mark.skipif(os.getenv('KVMEM_TEST_CUDA') != '1', reason='Explicit GPU component opt-in')


@CUDA
@pytest.mark.parametrize('cache_dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_all_finite_fp8_bit_patterns_embed_exactly_in_cache_dtype(cache_dtype):
    # The publication fast path relies on this exact embedding instead of
    # expanding every batch twice on CPU. Include subnormals and signed zero.
    codes = torch.arange(256, dtype=torch.int16).to(torch.uint8)
    source = codes.view(torch.float8_e4m3fn)
    source = codes[torch.isfinite(source.float())].view(torch.float8_e4m3fn)
    expected = source.to(cache_dtype)
    actual = source.cuda().to(cache_dtype).cpu()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual.to(source.dtype).view(torch.uint8), source.view(torch.uint8))


@CUDA
@pytest.mark.parametrize('cache_dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_gpu_device_conversion_and_near_tie_fallback_preserve_all_original_keys(cache_dtype):
    raw = torch.randn(2, 5, 32, 2, 256,
        generator=torch.Generator().manual_seed(312)).to(torch.float8_e4m3fn)
    # Equal-norm, distinct directions require CPU tie handling from all Keys.
    raw[:, 0].zero_()
    for token, (coordinate, value) in enumerate(((0, 4.), (0, -4.), (1, 4.), (1, -4.))):
        raw[:, 0, token, :, coordinate] = value
    raw[:, 1].zero_()
    counts = [32, 32, 3, 1, 0]
    expected, expected_offsets = select_key_representatives(raw, counts)
    keys, offsets, stats = select_key_representatives_optimized(raw, counts,
        device='cuda', cache_dtype=cache_dtype)
    assert keys.device.type == offsets.device.type == 'cpu'
    assert keys.dtype == cache_dtype
    assert torch.equal(keys, expected.to(cache_dtype))
    assert torch.equal(offsets, expected_offsets)
    assert stats['fps_device_cache_conversion'] is True
    assert stats['fps_cpu_fallback_groups'] >= 2*2*2
    assert offsets[:, 0, :, 0].tolist() == [[0, 1, 2, 3]]*2
    assert offsets[:, 4].eq(-1).all() and keys[:, 4].eq(0).all()


@CUDA
def test_gpu_publication_keeps_sparse_updates_and_hot_cache_exact():
    case = fixture()
    try:
        snapshot(case, device='cuda')
        change(case, (0, 2, 5))
        keys, offsets, stats = snapshot(case, device='cuda')
        expected, expected_offsets = reference(case)
        assert torch.equal(keys, expected) and torch.equal(offsets, expected_offsets)
        assert stats['cache_rebuilt_blocks'] == 3
        assert stats['fps_gpu_groups'] == 2*3*2
        assert stats['fps_device_cache_conversion_batches'] == 1
        assert stats['fps_publication_strict_check_batches'] == 0
        assert stats['fps_publication_index_copy_batches'] == 1
        reads = len(case.reads)
        pointers = keys.data_ptr(), offsets.data_ptr()
        hot_keys, hot_offsets, hot = snapshot(case, device='cuda')
        assert (hot_keys.data_ptr(), hot_offsets.data_ptr()) == pointers
        assert hot['cache_hit_blocks'] == 6 and hot['cache_rebuilt_blocks'] == 0
        assert hot['fps_gpu_groups'] == 0 and len(case.reads) == reads
    finally:
        case.cache.clear()
