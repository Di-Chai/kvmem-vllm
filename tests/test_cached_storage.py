import threading
import pytest
from vllm_kvmem.cached_storage import CachedArchive
from test_authority import setup, payload, drain


def test_cached_archive_checks_authority_before_hits_and_preserves_mixed_read_order(tmp_path):
    size = 2*4*1*2*3
    archive = CachedArchive(str(tmp_path), size, 16, cache_bytes=2*size, namespace_bytes=size)
    a, _ = setup(tmp_path, 'a', archive)
    b, _ = setup(tmp_path, 'b', archive)
    assert a.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    assert b.append(*payload(20, 4), history_begin=0, local_rope_begin=0)
    drain(a); drain(b)
    # Force a mixed cache/disk read with deterministic resident placement.
    records = list(a.records.values())
    leases = [a.memory.repository.pin(r.handle) for r in records]
    archive.read_cache.clear()
    first, _ = archive.read('a', leases[:1]).result(timeout=5)
    ordered, stats = archive.read('a', list(reversed(leases))).result(timeout=5)
    assert stats['cache_hit_bytes'] == size and stats['bytes'] == size
    assert bytes(ordered[size:]) == bytes(first)
    hit, hit_stats = archive.read('a', leases[1:]).result(timeout=5)
    assert hit_stats['bytes'] == 0 and hit_stats['cache_hit_bytes'] == size
    assert bytes(hit) == bytes(ordered[:size])
    assert archive.read_cache.used_bytes <= 2*size
    assert archive.read_cache.by_namespace['a'] <= size
    with pytest.raises(KeyError):
        archive.read('foreign', leases)
    for lease in leases:
        lease.reset()
    assert a.close()
    assert all(key[0] != 'a' for key in archive.read_cache.entries)
    assert b.close() and archive.close()
    assert archive.read_cache.used_bytes == 0


def test_cached_payload_does_not_allow_early_cancellation_reclaim(tmp_path, monkeypatch):
    size = 2*4*1*2*3
    archive = CachedArchive(str(tmp_path), size, 8, cache_bytes=size, namespace_bytes=size)
    authority, _ = setup(tmp_path, 'a', archive)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    drain(authority)
    record = next(iter(authority.records.values()))
    lease = authority.memory.repository.pin(record.handle)
    started, proceed = threading.Event(), threading.Event()
    tier = archive.tier

    class Delayed:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            started.set()
            assert proceed.wait(5)
            return tier.io(*args)

    archive.tier = Delayed()
    job = archive.read('a', [lease])
    assert started.wait(5)
    lease.reset()
    assert not authority.close() and not archive.close()
    proceed.set()
    data, _ = job.result(timeout=5)
    assert len(data) == size
    assert authority.close() and archive.close()


def test_cached_write_is_not_readable_before_archive_write_completes_and_failure_evicts(tmp_path):
    from vllm_kvmem import _native
    archive = CachedArchive(str(tmp_path), 4096, 2, cache_bytes=8192, namespace_bytes=8192)
    repo = _native.Repository(1, 16)
    lease = repo.reserve(0, 16).commit()
    started, proceed = threading.Event(), threading.Event()
    tier = archive.tier

    class Failing:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            started.set()
            assert proceed.wait(5)
            raise OSError('injected write failure')

    archive.tier = Failing()
    job = archive.write('a', [lease], b'x' * 4096)
    assert started.wait(5) and archive.read_cache.used_bytes == 4096
    with pytest.raises(KeyError, match='not yet published'):
        archive.read('a', [lease])
    proceed.set()
    with pytest.raises(OSError, match='injected'):
        job.result(timeout=5)
    assert archive.tier.used_slots == 0 and archive.read_cache.used_bytes == 0
    handle = lease.handle
    assert repo.retire(handle)
    lease.reset()
    assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == _native.ReclaimResult.RECLAIMED
    assert archive.close()
