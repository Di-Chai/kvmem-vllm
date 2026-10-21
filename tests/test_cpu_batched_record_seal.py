"""One native copy per immutable record, one GIL release per admitted batch."""
import hashlib
from threading import Event
import weakref

import numpy as np
import pytest

from vllm_kvmem import _native, memory_storage
from vllm_kvmem.memory_storage import MemoryArchive
from test_cpu_direct_record_seal import repository, reclaim_all


def test_native_packed_pairs_are_independent_owned_bytes():
    sources = [bytearray(v) for v in (b'ab', b'cd', b'WX', b'YZ')]
    records = _native.pack_record_pairs(sources, 4)
    assert records == [b'abcd', b'WXYZ'] and all(type(x) is bytes for x in records)
    assert records[0] is not records[1]
    for source in sources:
        source[:] = b'!!'
    assert records == [b'abcd', b'WXYZ']


@pytest.mark.parametrize('sources,size', [([], 4), ([b'a'], 1), ([b'a', b'b'], 0),
    ([b'a', b'b'], 3), ([b'abc', b'd'], 2),
    ([b'ab', b'cd', memoryview(b'1234')[::2], b'zz'], 4),
    ([b'ab', b'cd', np.ones(1, dtype=np.int32), b'zz'], 4)])
def test_native_pair_validation(sources, size):
    with pytest.raises(ValueError):
        _native.pack_record_pairs(sources, size)


def test_batch_seal_retains_sources_until_copy_then_publishes_exact_records(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 2, io_max_bytes=16, cpu_direct_record_seal=True)
    repo, leases, handles = repository(2)
    entered, release = Event(), Event()
    native, owned, calls = _native.pack_record_pairs, [], []
    def delayed(views, size):
        calls.append(len(views))
        entered.set()
        assert release.wait(5)
        result = native(views, size)
        owned.extend(result)
        return result
    def no_second_copy(*args):
        raise AssertionError('Batch seal must not allocate/copy another slab')
    monkeypatch.setattr(_native, 'pack_record_pairs', delayed)
    monkeypatch.setattr(memory_storage, 'copy_record_slices', no_second_copy)
    def submit():
        sources = [np.array([i, i+1], dtype=np.uint8) for i in range(4)]
        return archive.write_from_views('owner', leases, sources), [weakref.ref(v) for v in sources]
    job, refs = submit()
    try:
        assert entered.wait(5) and all(ref() is not None for ref in refs)
        assert archive.io_usage()['archive_io_pending_bytes'] == 16
        assert not archive.tier.records and not archive.close()
        assert archive.write_from_views('other', leases, [b'ab', b'cd', b'ef', b'gh']) is None
        with pytest.raises(KeyError):
            archive.read('owner', leases)
        release.set()
        stats = job.result(5)
        assert calls == [4] and all(ref() is None for ref in refs)
        assert owned == [b'\x00\x01\x01\x02', b'\x02\x03\x03\x04']
        assert all(record is owned[i] for i, record in enumerate(archive.tier.records.values()))
        assert stats['record_digests'] == tuple(hashlib.sha256(v).hexdigest() for v in owned)
        assert stats['seal_record_copy_seconds'] == 0
        reclaim_all(repo, leases, handles, archive)
        assert not archive.tier.records and archive.close()
    finally:
        release.set()


@pytest.mark.parametrize('failure', ['pack', 'count', 'mutable', 'hash', 'publication'])
def test_batched_failure_releases_whole_admission_and_source_exports(tmp_path, monkeypatch, failure):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_direct_record_seal=True)
    repo, leases, handles = repository(2)
    native = _native.pack_record_pairs
    def pack(views, size):
        try:
            if failure == 'pack':
                raise OSError('Injected batch pack failure')
            records = native(views, size)
            if failure == 'count':
                return records[:1]
            if failure == 'mutable':
                return [records[0], bytearray(records[1])]
            return records
        finally:
            views = None
    def failed_hash(record):
        raise OSError('Injected hash failure')
    class FailedInsert(dict):
        def update(self, records):
            self[next(iter(records))] = next(iter(records.values()))
            raise MemoryError('Injected partial insertion')
    monkeypatch.setattr(_native, 'pack_record_pairs', pack)
    if failure == 'hash':
        monkeypatch.setattr(memory_storage.hashlib, 'sha256', failed_hash)
    if failure == 'publication':
        archive.tier.records = FailedInsert()
    def submit():
        sources = [np.array([i, i+1], dtype=np.uint8) for i in range(4)]
        return archive.write_from_views('owner', leases, sources), [weakref.ref(v) for v in sources]
    job, refs = submit()
    with pytest.raises((OSError, RuntimeError, MemoryError)):
        job.result(5)
    assert all(ref() is None for ref in refs)
    assert not archive.tier.records and not archive._entries and archive.tier.used_slots == 0
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()
