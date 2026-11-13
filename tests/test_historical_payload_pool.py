"""Opt-in CPU record pool: identical bytes, reuse after reclamation, explicit config."""
import os

import numpy as np
import pytest

from vllm_kvmem import _native


def arena(**kwargs):
    try:
        a = _native.HistoricalPayloadArena(64, 3 * 4096 + 17, background=False, **kwargs)
    except TypeError:
        pytest.skip('Native extension predates the CPU record pool; rebuild _native')
    if 'pool_hits' not in a.stats:
        pytest.skip('Native extension predates the CPU record pool; rebuild _native')
    return a


def publish(a, slot, generation, data):
    half = len(data) // 2
    a.publish_segments(_native.HistoricalPayloadHandle(slot, generation),
                       [memoryview(data[:half]), memoryview(data[half:])])


@pytest.mark.parametrize('pool', [{}, dict(pool_chunk_mib=2), dict(pool_chunk_mib=2, pool_ahead_records=8)])
def test_pool_publishes_identical_bytes_and_reuses_released_records(pool):
    a = arena(**pool)
    rng = np.random.default_rng(7)
    rows = [rng.integers(0, 256, 3 * 4096 + 17, dtype=np.uint8) for _ in range(40)]
    for slot, row in enumerate(rows):
        publish(a, slot, 1, row)
    for slot, row in enumerate(rows):
        read = a.acquire(_native.HistoricalPayloadHandle(slot, 1))
        assert bytes(memoryview(read)) == row.tobytes()
        del read
    for slot in range(40):
        a.retire(_native.HistoricalPayloadHandle(slot, 1))
    while a.advance():
        pass
    before = a.stats
    for slot, row in enumerate(rows[:10]):
        publish(a, slot, 2, row[::-1].copy())
        read = a.acquire(_native.HistoricalPayloadHandle(slot, 2))
        assert bytes(memoryview(read)) == row[::-1].tobytes()
        del read
    stats = a.stats
    if pool:
        assert stats['pool_reserved_bytes'] > 0
        assert stats['pool_hits'] - before['pool_hits'] == 10  # Released records are reused.
    else:
        assert stats['pool_hits'] == stats['pool_misses'] == stats['pool_reserved_bytes'] == 0
    a.close()


def test_prefault_requires_a_pool():
    with pytest.raises((ValueError, TypeError)):
        arena(pool_ahead_records=4)


def test_python_config_is_opt_in(monkeypatch):
    from vllm_kvmem.core.native_memory_storage import payload_pool_config
    monkeypatch.delenv('KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB', raising=False)
    monkeypatch.delenv('KVMEM_CPU_PAYLOAD_POOL_AHEAD', raising=False)
    assert payload_pool_config() == {}
    monkeypatch.setenv('KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB', '1024')
    assert payload_pool_config() == dict(pool_chunk_mib=1024, pool_ahead_records=0)
    monkeypatch.setenv('KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB', '0')
    monkeypatch.setenv('KVMEM_CPU_PAYLOAD_POOL_AHEAD', '4')
    with pytest.raises(ValueError, match='requires'):
        payload_pool_config()
    monkeypatch.setenv('KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB', 'x')
    with pytest.raises(ValueError, match='non-negative'):
        payload_pool_config()
