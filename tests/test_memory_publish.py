"""Immutable CPU publication remains atomic while copies run outside the lock."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vllm_kvmem import _native, memory_storage
from vllm_kvmem.memory_storage import MemoryArchive, MemorySpan, MemoryTier


@pytest.mark.skipif(not hasattr(_native, 'copy_record_slices'), reason='New host extension required')
def test_native_owned_records_preserve_order_gaps_and_mutable_source_independence():
    source = bytearray(b'abcd____WXYZ')
    records = _native.copy_record_slices(memoryview(source), [8, 0], 4)
    assert records == [b'WXYZ', b'abcd']
    assert all(type(record) is bytes for record in records)
    source[:] = b'?' * len(source)
    assert records == [b'WXYZ', b'abcd']
    assert _native.copy_record_slices(memoryview(source), [], 4) == []


@pytest.mark.skipif(not hasattr(_native, 'copy_record_slices'), reason='New host extension required')
@pytest.mark.parametrize('source,offsets,size', [
    (memoryview(b'abcd'), [1], 4),
    (memoryview(b'abcd'), [0], 0),
    (memoryview(b'abcd')[::2], [0], 2),
    (memoryview(b'abcd'), [0, 4], 4),
])
def test_native_copy_rejects_invalid_extent_before_exposing_any_record(source, offsets, size):
    with pytest.raises((ValueError, TypeError, BufferError)):
        _native.copy_record_slices(source, offsets, size)


def test_older_extension_fallback_keeps_independent_record_owners(monkeypatch):
    monkeypatch.delattr(_native, 'copy_record_slices', raising=False)
    source = bytearray(b'abcdWXYZ')
    records = memory_storage.copy_record_slices(memoryview(source), [4, 0], 4)
    source[:] = b'?' * len(source)
    assert records == [b'WXYZ', b'abcd']


def test_native_failure_is_not_retried_through_python(monkeypatch):
    def fail(*args):
        raise OSError('native copy failure')
    monkeypatch.setattr(_native, 'copy_record_slices', fail, raising=False)
    with pytest.raises(OSError, match='native copy failure'):
        memory_storage.copy_record_slices(memoryview(b'abcd'), [0], 4)


def blocked_copy(monkeypatch):
    started, release = Event(), Event()
    original = memory_storage.copy_record_slices

    def copy(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(memory_storage, 'copy_record_slices', copy)
    return started, release


def test_long_copy_does_not_hold_tier_lock_or_publish_partial_records(monkeypatch):
    tier = MemoryTier(4, 3)
    for slot in range(3):
        tier.allocate(slot)
    tier.io([MemorySpan(2, 0, 4)], b'live', True)
    started, release = blocked_copy(monkeypatch)
    spans = [MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        writing = pool.submit(tier.io, spans, b'abcdWXYZ', True)
        try:
            assert started.wait(5)
            # A separate thread must read an already-published record while the
            # writer is paused, proving the publication lock is not retained.
            destination = bytearray(4)
            pool.submit(tier.io, [MemorySpan(2, 0, 4)], destination, False).result(2)
            assert destination == b'live' and tier.records == {2: b'live'}
            with pytest.raises(KeyError, match='not published'):
                tier.io(spans, bytearray(8), False)
            release.set()
            stats = writing.result(5)
            assert tier.records == {0: b'abcd', 1: b'WXYZ', 2: b'live'}
            assert stats['bytes'] == 8 and stats['seal_record_copy_seconds'] >= 0
        finally:
            release.set()


def test_copy_cannot_publish_into_reallocated_generation(monkeypatch):
    tier = MemoryTier(4, 2)
    first = tier.allocate(0)
    tier.allocate(1)
    started, release = blocked_copy(monkeypatch)
    spans = [MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(tier.io, spans, b'abcdWXYZ', True)
        try:
            assert started.wait(5)
            assert tier.release(first)
            replacement = tier.allocate(0)
            assert replacement.generation != first.generation
            release.set()
            with pytest.raises(ValueError, match='allocation changed'):
                writing.result(5)
            assert tier.records == {} and tier.used_slots == 2
            tier.io(spans, b'new!okay', True)
            assert tier.records == {0: b'new!', 1: b'okay'}
        finally:
            release.set()


def test_competing_publication_cannot_overwrite_or_partly_commit(monkeypatch):
    tier = MemoryTier(4, 2)
    tier.allocate(0); tier.allocate(1)
    original = memory_storage.copy_record_slices
    started, release = blocked_copy(monkeypatch)
    spans = [MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)]
    with ThreadPoolExecutor(max_workers=1) as pool:
        writing = pool.submit(tier.io, spans, b'abcdWXYZ', True)
        try:
            assert started.wait(5)
            monkeypatch.setattr(memory_storage, 'copy_record_slices', original)
            tier.io([spans[1]], b'____wins', True)
            release.set()
            with pytest.raises(ValueError, match='overwrite'):
                writing.result(5)
            assert tier.records == {1: b'wins'}
        finally:
            release.set()


def test_failed_copy_leaves_all_records_unpublished_and_can_retry(monkeypatch):
    tier = MemoryTier(4, 2)
    tier.allocate(0); tier.allocate(1)
    spans = [MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)]
    original = memory_storage.copy_record_slices

    def fail(*args):
        original(*args)  # Even a failure after allocation/copy cannot publish.
        raise MemoryError('injected record allocation failure')

    monkeypatch.setattr(memory_storage, 'copy_record_slices', fail)
    with pytest.raises(MemoryError, match='injected'):
        tier.io(spans, b'abcdWXYZ', True)
    assert tier.records == {} and tier.used_slots == 2
    monkeypatch.setattr(memory_storage, 'copy_record_slices', original)
    tier.io(spans, b'abcdWXYZ', True)
    assert tier.records == {0: b'abcd', 1: b'WXYZ'}


def test_publication_insertion_failure_rolls_back_every_new_record():
    class FailingPublication(dict):
        def update(self, values):
            self[next(iter(values))] = next(iter(values.values()))
            raise MemoryError('injected publication failure')

    tier = MemoryTier(4, 3)
    for slot in range(3):
        tier.allocate(slot)
    tier.records = FailingPublication({2: b'live'})
    with pytest.raises(MemoryError, match='publication failure'):
        tier.io([MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)], b'abcdWXYZ', True)
    assert tier.records == {2: b'live'} and tier.used_slots == 3


def test_cpu_archive_failed_copy_releases_admission_allocations_and_leases(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 2)
    repo = _native.Repository(2, 4)
    leases = [repo.reserve(i * 4, 4).commit() for i in range(2)]
    handles = [lease.handle for lease in leases]
    started, release = Event(), Event()

    def fail(*args):
        started.set()
        assert release.wait(5)
        raise OSError('CPU record copy failed')

    monkeypatch.setattr(memory_storage, 'copy_record_slices', fail)
    job = archive.write('cpu', leases, b'abcdWXYZ')
    try:
        assert started.wait(5)
        assert not job.cancel() and not archive.close()
        with pytest.raises(KeyError, match='not yet published'):
            archive.read('cpu', leases)
        for handle, lease in zip(handles, leases):
            assert repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda record: archive.reclaim('cpu', record)) == _native.ReclaimResult.READERS
        release.set()
        with pytest.raises(OSError, match='CPU record copy failed'):
            job.result(5)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert archive.tier.used_slots == 0 and archive.tier.records == {}
        for handle in handles:
            assert repo.reclaim(handle, lambda record: archive.reclaim('cpu', record)) == _native.ReclaimResult.RECLAIMED
        assert archive.close()
    finally:
        release.set()
