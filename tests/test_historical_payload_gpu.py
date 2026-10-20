"""Explicit GPU gate for independent historical CPU/GPU payload lifetimes."""
import os
import time

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='explicit GPU opt-in required')


def test_cpu_history_frees_while_independent_gpu_window_is_still_in_flight():
    import torch
    from vllm_kvmem import _native as n

    size = 65536
    expected = bytes(range(256)) * (size // 256)
    repo = n.Repository(1, 32)
    gpu_lease = repo.reserve(0, 32).commit()
    handle = gpu_lease.handle
    arena = n.HistoricalPayloadArena(1, size, poll_ms=1)
    cpu_handle = n.HistoricalPayloadHandle(0, 1)
    arena.publish(cpu_handle, expected)
    staging = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    source = arena.acquire(cpu_handle)
    n.gather_record_region(memoryview(staging.numpy()), [source], size, 0, size, 0, 1)
    del source
    stream = torch.cuda.Stream()
    complete = torch.cuda.Event()
    with torch.cuda.stream(stream):
        working = torch.empty(size, dtype=torch.uint8, device='cuda')
        working.copy_(staging, non_blocking=True)
        # Keep a real GPU operation in flight while the CPU worker reclaims.
        # This is a lifetime gate, not a decode performance measurement.
        torch.cuda._sleep(1_000_000_000)
        result = working.clone()
        complete.record(stream)
    try:
        assert repo.retire_permanently(handle) == n.RetireResult.COMMITTED
        assert arena.retire(cpu_handle) == n.HistoricalPayloadRetireResult.COMMITTED
        deadline = time.monotonic() + 5
        while not arena.reclaimed(cpu_handle):
            assert time.monotonic() < deadline
            time.sleep(.001)
        assert arena.stats['used_bytes'] == 0
        assert not complete.query(), 'GPU delay ended before independent CPU reclamation was observed'
        assert gpu_lease.record.token_begin == 0 and repo.ready_slots == 0
        complete.synchronize()
        assert bytes(result.cpu().numpy()) == expected
    finally:
        complete.synchronize()
        gpu_lease.reset()
        deadline = time.monotonic() + 5
        while not arena.close():
            assert time.monotonic() < deadline
            time.sleep(.001)
