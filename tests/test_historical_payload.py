"""Native CPU storage owns bytes through read/DMA completion, independently of GPU leases."""
import gc
import time

import pytest

from vllm_kvmem import _native as n


def close(arena):
    deadline = time.monotonic() + 5
    while not arena.close():
        assert time.monotonic() < deadline
        time.sleep(.001)


def test_buffer_export_keeps_retired_cpu_payload_alive_without_gpu_dependency():
    repo = n.Repository(1, 32)
    gpu = repo.reserve(0, 32).commit()
    h = n.HistoricalPayloadHandle(gpu.handle.slot, gpu.handle.generation)
    arena = n.HistoricalPayloadArena(1, 32, background=False)
    try:
        arena.publish(h, bytes(range(32)))
        reader = arena.acquire(h)
        data = memoryview(reader)
        assert data.readonly and bytes(data) == bytes(range(32))
        with pytest.raises(TypeError):
            data[0] = 1
        assert repo.retire_permanently(gpu.handle) == n.RetireResult.COMMITTED
        assert arena.retire(h) == n.HistoricalPayloadRetireResult.COMMITTED
        del reader
        gc.collect()
        assert arena.advance() == 0 and arena.stats['used_bytes'] == 32
        assert bytes(data) == bytes(range(32))
        data.release()
        del data
        gc.collect()
        assert arena.stats['released_slots'] == 0
        assert arena.advance() == 1 and arena.stats['used_bytes'] == 0
        assert arena.reclaimed(h) and repo.ready_slots == 0
        assert gpu.record.token_begin == 0
    finally:
        gpu.reset()
        close(arena)


def test_old_payload_handle_cannot_read_or_retire_replacement():
    arena = n.HistoricalPayloadArena(1, 16, background=False)
    try:
        old = n.HistoricalPayloadHandle(0, 1)
        new = n.HistoricalPayloadHandle(0, 2)
        arena.publish(old, b'a' * 16)
        assert arena.retire(old) == n.HistoricalPayloadRetireResult.COMMITTED
        assert arena.advance() == 1
        arena.publish(new, b'b' * 16)
        assert not arena.acquire(old)
        assert arena.retire(old) == n.HistoricalPayloadRetireResult.STALE
        data = memoryview(arena.acquire(new))
        assert bytes(data) == b'b' * 16
        data.release()
    finally:
        close(arena)


def test_close_does_not_wait_for_exported_buffer_and_blocks_new_readers():
    arena = n.HistoricalPayloadArena(1, 16, poll_ms=1)
    h = n.HistoricalPayloadHandle(0, 1)
    arena.publish(h, b'c' * 16)
    data = memoryview(arena.acquire(h))
    assert not arena.close()
    assert not arena.acquire(h)
    assert bytes(data) == b'c' * 16
    data.release()
    close(arena)
    assert arena.stats['released_bytes'] == 16


def test_publication_validates_layout_and_rejects_live_overwrites():
    arena = n.HistoricalPayloadArena(1, 16, background=False)
    h = n.HistoricalPayloadHandle(0, 1)
    try:
        with pytest.raises(ValueError):
            arena.publish(h, b'a')
        with pytest.raises(ValueError, match='contiguous'):
            arena.publish(h, memoryview(bytes(range(32)))[::2])
        arena.publish(h, b'a' * 16)
        with pytest.raises(RuntimeError, match='not reusable'):
            arena.publish(n.HistoricalPayloadHandle(0, 2), b'b' * 16)
        assert arena.stats['used_bytes'] == 16
    finally:
        close(arena)
