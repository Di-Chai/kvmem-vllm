"""Explicit CUDA byte/lifetime qualification for production full-batch packing.

Uses the deployed 17-layer, 4-KV-head, 256-dimension geometry. This is a
component qualification, not an attention or end-to-end throughput benchmark.
"""
import gc
import os
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest
import torch

from vllm_kvmem.capture_pack import PackLimits, source_signature
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.transfer import PackedDownloads

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')


def fixture(tokens=128, requests=3, *, strided=False, capacity=None):
    device = torch.device('cuda', torch.cuda.current_device())
    capacity = capacity or tokens
    scale_values = (.125, .3, 1., 1.75, 3.25, 7.)
    scales = torch.tensor([scale_values[i % 6] for i in range(17)], device=device)
    local = torch.arange(capacity, device=device, dtype=torch.int64) + 49152
    captured = []
    for i in range(17):
        # Offset and noncontiguous rows exercise opaque pointer + stride access.
        stride, offset = (2, 1) if strided else (1, 0)
        shape = (capacity*stride+offset, 4, 256)
        key = (torch.randn(shape, device=device, dtype=torch.bfloat16)*20)[offset::stride]
        value = (torch.randn(shape, device=device, dtype=torch.bfloat16)*(i+20))[offset::stride]
        captured.append(NS(key=key, value=value, positions=local))
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.scales, runtime.raw_dtype = scales, torch.float8_e4m3fn
    runtime.options = NS(capture_fused_pack=True)
    runtime._capture_pack_limits = PackLimits(17, 4, 256, 2048, 8)
    runtime._capture_pack_cache = None
    batch, rejected = batch_for(captured, tokens, requests)
    return runtime, captured, batch, rejected


def batch_for(captured, tokens, requests):
    positions = captured[0].positions
    if positions.ndim == 2:
        positions = positions[0]
    starts = np.array([i*tokens//requests for i in range(requests+1)], dtype=np.int32)
    device = positions.device
    batch = NS(num_reqs=requests, num_tokens=tokens,
        query_start_loc=torch.tensor(starts, device=device), query_start_loc_np=starts,
        positions=positions+65536)
    # Includes accepted-prefix rejection values; packing preserves them without
    # interpreting or prematurely publishing candidate MTP rows.
    rejected = torch.arange(requests, device=device, dtype=torch.int32).remainder(3)
    return batch, rejected


def reference(runtime, captured, batch, rejected, indices=None):
    oracle = CaptureRuntime.__new__(CaptureRuntime)
    oracle.scales, oracle.raw_dtype = runtime.scales, runtime.raw_dtype
    return oracle._capture_payload(captured, batch, rejected,
        tuple(range(batch.num_reqs)) if indices is None else indices)


def compare(runtime, captured, batch, rejected, indices=None, *, fused=True):
    before = runtime.__dict__.get('capture_pack_stats', {}).get('fused_batches', 0)
    indices = tuple(range(batch.num_reqs)) if indices is None else indices
    expected, expected_rows = reference(runtime, captured, batch, rejected, indices)
    actual, rows = runtime._capture_payload(captured, batch, rejected, indices)
    assert rows == expected_rows
    assert torch.equal(actual, expected), 'Production pack changed header or FP8 bytes'
    assert runtime.capture_pack_stats.get('fused_batches', 0) == before+int(fused)
    return actual


@pytest.mark.parametrize('tokens,requests,strided', [(1, 1, False), (7, 3, True),
    (128, 8, False), (1952, 3, True), (2048, 1, False)])
def test_actual_geometry_full_batch_owner_header_and_strided_rows(tokens, requests, strided):
    runtime, captured, batch, rejected = fixture(tokens, requests, strided=strided)
    result = compare(runtime, captured, batch, rejected)
    header = result[:(4*requests+1)*8].view(torch.int64).cpu()
    starts = batch.query_start_loc.cpu().long()
    assert torch.equal(header, torch.cat((starts, starts[:-1]+114688,
                                         starts[:-1]+49152, rejected.cpu().long())))
    assert runtime._capture_pack_cache.limits.metadata_bytes == 1088


def test_all_bfloat16_bit_patterns_match_signed_nan_saturation_and_scale_rounding():
    runtime, captured, batch, rejected = fixture(64, 3)
    # Exhaustive BF16 inputs include signed NaNs/zero, subnormals, infinities,
    # and every representable FP8 rounding boundary across non-power-of-two V scales.
    patterns = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    patterns = patterns.reshape(64, 4, 256).to(runtime.scales.device)
    for layer in captured:
        layer.key.copy_(patterns)
        layer.value.copy_(patterns)
    compare(runtime, captured, batch, rejected)


def test_buffer_replay_row_count_scale_value_and_source_address_changes():
    runtime, captured, batch, rejected = fixture(128, 3, strided=True, capacity=256)
    first = compare(runtime, captured, batch, rejected)
    cache = runtime._capture_pack_cache
    table_address, host_address = cache.pointers.data_ptr(), cache.host.data_ptr()
    for tokens in (7, 256, 128):
        for i, layer in enumerate(captured):
            layer.key.fill_(tokens+i)
            layer.value.fill_(-tokens-i)
        runtime.scales.mul_(1.01)  # Values must be read live, never cached on CPU.
        batch, rejected = batch_for(captured, tokens, 3)
        current = compare(runtime, captured, batch, rejected)
    assert not torch.equal(first, current)
    cache.event.synchronize()
    old_signature = cache.binding.signature
    captured[-1].key = captured[-1].key.clone()  # Draft/MTP prefix buffer may be replaced.
    captured[-1].value = captured[-1].value.clone()
    runtime.scales = runtime.scales.clone()*1.25
    compare(runtime, captured, batch, rejected)
    assert cache.binding.signature != old_signature
    assert runtime._capture_pack_cache is cache
    assert (cache.pointers.data_ptr(), cache.host.data_ptr()) == (table_address, host_address)
    cache.event.synchronize()
    # Capacity changes at an identical data pointer are part of the generation.
    captured[-1].key = captured[-1].key[:128]
    compare(runtime, captured, batch, rejected)
    assert cache.binding.signature == source_signature(captured, runtime.scales)


def test_busy_source_generation_falls_back_without_wait_and_retains_old_sources():
    runtime, captured, batch, rejected = fixture(128, 3)
    compare(runtime, captured, batch, rejected)
    cache = runtime._capture_pack_cache
    old = captured[-1].key
    old_ref = weakref.ref(old)
    captured[-1].key = old.clone()
    del old
    real_event = cache.binding.event
    cache.binding.event = NS(query=lambda: False,
        synchronize=lambda: pytest.fail('Fused pack may not wait for replacement'))
    try:
        compare(runtime, captured, batch, rejected, fused=False)
        gc.collect()
        assert old_ref() is not None
        assert runtime.capture_pack_stats['fallback:source_generation_inflight'] == 1
    finally:
        cache.binding.event = real_event
    real_event.synchronize()
    compare(runtime, captured, batch, rejected)
    gc.collect()
    assert old_ref() is None


def test_cross_stream_generation_rebind_has_one_table_and_preserves_producer_order():
    runtime, captured, batch, rejected = fixture(128, 3)
    compare(runtime, captured, batch, rejected)
    cache = runtime._capture_pack_cache
    cache.event.synchronize()
    producer = torch.cuda.Stream(device=runtime.scales.device)
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(2_000_000)
        for layer in captured:
            layer.key.add_(3)
            layer.value.sub_(7)
        compare(runtime, captured, batch, rejected)
    producer.synchronize()
    assert cache.binding.stream == producer.cuda_stream
    assert runtime._capture_pack_cache is cache


def test_busy_cross_stream_same_sources_fallback_preserves_original_completion_event():
    runtime, captured, batch, rejected = fixture(128, 3)
    expected = compare(runtime, captured, batch, rejected).cpu()
    cache = runtime._capture_pack_cache
    first, second = (torch.cuda.Stream(device=runtime.scales.device) for _ in range(2))
    first.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(first):
        torch.cuda._sleep(200_000_000)
        runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
    original_event = cache.binding.event
    assert not original_event.query()
    second.wait_stream(first)
    with torch.cuda.stream(second):
        result, _ = runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
    assert cache.binding.stream == first.cuda_stream
    assert cache.binding.event is original_event
    assert runtime.capture_pack_stats['fallback:source_generation_inflight'] == 1
    assert runtime.capture_pack_stats['fused_batches'] == 2
    second.synchronize()
    assert torch.equal(result.cpu(), expected)
    compare(runtime, captured, batch, rejected)  # Completion permits later rebind.
    assert runtime._capture_pack_cache is cache


def test_nondefault_stream_keeps_freed_sources_headers_and_pointer_table_alive():
    runtime, captured, batch, rejected = fixture(128, 3)
    compare(runtime, captured, batch, rejected)  # Allocate table on the original stream.
    oracle = reference(runtime, captured, batch, rejected)[0].cpu()
    runtime._capture_pack_cache.event.synchronize()
    producer = torch.cuda.Stream(device=runtime.scales.device)
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        torch.cuda._sleep(20_000_000)
        result, _ = runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
    # No source generation or header is kept by this test. The allocator must
    # honor record_stream while the nondefault stream is still consuming it.
    runtime._capture_pack_cache = None
    del runtime, captured, batch, rejected
    gc.collect()
    for _ in range(8):
        churn = [torch.empty(size, dtype=torch.uint8, device='cuda').fill_(0xA5)
                 for size in (16, 32, 544, 1024, 128*8, 128*4*256*2)]
        del churn
    producer.synchronize()
    assert torch.equal(result.cpu(), oracle)


@pytest.mark.parametrize('mode', ['subset', 'reordered', 'positions2d', 'inner_stride', 'raw_bf16'])
def test_unsupported_geometry_preserves_original_path_exactly(mode):
    runtime, captured, batch, rejected = fixture(128, 3)
    indices = (0, 1, 2)
    if mode == 'subset':
        indices = (0, 2)
    elif mode == 'reordered':
        indices = (2, 1, 0)
    elif mode == 'positions2d':
        captured[0].positions = captured[0].positions.expand(3, -1)
    elif mode == 'inner_stride':
        for layer in captured:
            expanded = torch.empty((128, 4, 512), device=layer.key.device, dtype=layer.key.dtype)
            expanded[..., ::2].copy_(layer.key)
            layer.key = expanded[..., ::2]
    elif mode == 'raw_bf16':
        runtime.raw_dtype = torch.bfloat16
    compare(runtime, captured, batch, rejected, indices, fused=False)
    assert runtime._capture_pack_cache is None
    assert runtime.capture_pack_stats['reference_batches'] == 1


def test_fixed_dma_slots_preserve_two_versions_while_capture_buffers_are_replayed():
    runtime, captured, batch, rejected = fixture(128, 3)
    pool = PackedDownloads(runtime._capture_pack_limits.payload_bytes, 2, runtime.scales.device)
    transfers, expected = [], []
    try:
        for marker in (1, 19):
            for layer in captured:
                layer.key.fill_(marker)
                layer.value.fill_(-marker)
            oracle, _ = reference(runtime, captured, batch, rejected)
            payload, _ = runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
            transfers.append(pool.download(payload))
            expected.append(oracle)
        assert all(transfers) and pool.download(payload) is None
        for transfer, oracle in zip(transfers, expected):
            transfer.event.synchronize()
            assert torch.equal(transfer.payload, oracle.cpu())
        addresses = tuple(buffer.data_ptr() for buffer in pool.buffers)
        first_slot = transfers[0].slot
        assert pool.release(transfers.pop(0))
        replacement = pool.download(payload)
        transfers.append(replacement)
        assert replacement.slot == first_slot
        assert tuple(buffer.data_ptr() for buffer in pool.buffers) == addresses
    finally:
        for transfer in transfers:
            if transfer is not None:
                transfer.event.synchronize()
                assert pool.release(transfer)
        assert pool.close()


def test_supported_kernel_failure_is_not_silently_retried_or_allowed_to_replace(monkeypatch):
    import vllm_kvmem.capture_pack as pack
    runtime, captured, batch, rejected = fixture(128, 3)
    compare(runtime, captured, batch, rejected)
    cache = runtime._capture_pack_cache

    class FailingLaunch:
        def __getitem__(self, _):
            def fail(*args, **kwargs):
                raise RuntimeError('injected payload launch failure')
            return fail

    with monkeypatch.context() as patched:
        patched.setattr(pack, '_payload', FailingLaunch())
        with pytest.raises(RuntimeError, match='injected payload'):
            runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
    assert cache.poisoned and cache.binding.owners
    with pytest.raises(RuntimeError, match='unresolved CUDA failure'):
        runtime._capture_payload(captured, batch, rejected, (0, 1, 2))
    assert runtime.capture_pack_stats == {'fused_batches': 1,
        'fused_payload_bytes': (4*3+1)*8+2*17*128*4*256}
    torch.cuda.current_stream().synchronize()  # Test cleanup, never production fallback.
