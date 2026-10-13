import hashlib
from threading import Event

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.cached_storage import CachedArchive
from test_authority import setup, payload, drain


def test_native_snapshot_is_exact_and_does_not_borrow_mutable_sources():
    sources = [bytearray(b'raw'), bytearray(b'value')]
    result = _native.pack_record_views([memoryview(s) for s in sources])
    assert type(result) is bytes and result == b'rawvalue'
    sources[0][:] = b'new'
    assert result == b'rawvalue'
    for views in ([], [memoryview(bytearray(8))[::2]]):
        with pytest.raises((ValueError, TypeError, BufferError)):
            _native.pack_record_views(views)


def test_async_seal_retains_generations_and_defers_digest_publication(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path, resident=1)
    archive.async_seal = True
    started, release = Event(), Event()
    real_pack = _native.pack_record_views

    def delayed(views):
        started.set()
        assert release.wait(5)
        return real_pack(views)

    monkeypatch.setattr(_native, 'pack_record_views', delayed)
    try:
        assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        assert started.wait(5)
        record = next(iter(authority.records.values()))
        assert record.raw is None and record.digest is None and not record.archived
        assert archive.io_usage()['archive_io_pending_bytes'] == 2 * authority.record_bytes
        assert not authority.append(*payload(4, 1), history_begin=4, local_rope_begin=4)
        assert not archive.close()
        release.set()
        drain(authority)
        expected = payload(0, 4)
        encoded = expected[0].view(torch.uint8).numpy().tobytes() + expected[1].numpy().tobytes()
        assert record.digest == hashlib.sha256(encoded).hexdigest() and record.archived
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert authority.read_record(record.handle)[0].equal(expected[0])
        assert authority.close() and archive.close()
    finally:
        release.set()


def test_async_snapshot_failure_never_publishes_and_releases_admission(tmp_path, monkeypatch):
    authority, archive = setup(tmp_path)
    archive.async_seal = True
    def fail(_):
        raise OSError('snapshot failure')
    monkeypatch.setattr(_native, 'pack_record_views', fail)
    assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
    with pytest.raises(OSError, match='snapshot failure'):
        authority.writes[0][0].result(5)
    assert not next(iter(authority.records.values())).archived
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    assert archive.tier.used_slots == 0 and archive.close()


def test_async_cached_records_roundtrip_and_refused_writer_does_not_pack(tmp_path, monkeypatch):
    size = 2 * 4 * 1 * 2 * 3
    archive = CachedArchive(str(tmp_path), size, 16, cache_bytes=16*size,
                            namespace_bytes=16*size, io_max_bytes=2*size, max_pending=1)
    authority, _ = setup(tmp_path, archive=archive, resident=4)
    archive.async_seal = True
    started, release = Event(), Event()
    calls = []
    real_pack = _native.pack_record_views
    def delayed(views):
        calls.append(1); started.set(); assert release.wait(5)
        return real_pack(views)
    monkeypatch.setattr(_native, 'pack_record_views', delayed)
    try:
        assert authority.append(*payload(0, 4), history_begin=0, local_rope_begin=0)
        assert started.wait(5)
        assert authority.append(*payload(4, 4), history_begin=4, local_rope_begin=4)
        assert len(calls) == 1 and len(authority._unsubmitted) == 1
        release.set()
        authority.writes[0][0].result(5); authority.poll()
        drain(authority)
        records = list(authority.records.values())
        lease = authority.memory.repository.pin(records[0].handle)
        _, stats = archive.read('a', [lease]).result(5)
        lease.reset()
        assert stats['cache_hit_bytes'] == size and stats['bytes'] == 0
        assert authority.close() and archive.close()
    finally:
        release.set()


def test_publish_barrier_retries_a_sealed_record_refused_by_another_namespace(tmp_path, monkeypatch):
    from vllm_kvmem.packed_assembly import read_packed_records
    size = 2 * 4 * 1 * 2 * 3
    archive = CachedArchive(str(tmp_path),size,16,cache_bytes=16*size,
                            namespace_bytes=16*size,max_pending=1)
    archive.async_seal = True
    a,_ = setup(tmp_path,'a',archive)
    b,_ = setup(tmp_path,'b',archive)
    started, release = Event(), Event()
    real_pack = _native.pack_record_views
    def delayed(views):
        started.set(); assert release.wait(5)
        return real_pack(views)
    monkeypatch.setattr(_native,'pack_record_views',delayed)
    try:
        assert a.append(*payload(0,4),history_begin=0,local_rope_begin=0)
        assert started.wait(5)
        assert b.append(*payload(20,4),history_begin=0,local_rope_begin=0)
        record=next(iter(b.records.values()))
        assert not record.submitted and not b.writes
        with pytest.raises(ValueError,match='published'):
            read_packed_records(b,[record.handle])
        release.set()
        def progress():
            for job,_ in tuple(a.writes):job.result(5)
            a.poll()
        b.publish_sealed(progress)
        assert record.archived and record.digest and not b._unsubmitted
        data,_=read_packed_records(b,[record.handle])
        expected=payload(20,4)
        encoded=expected[0].view(torch.uint8).numpy().tobytes()+expected[1].numpy().tobytes()
        assert bytes(data.numpy()) == encoded
        assert a.close() and b.close() and archive.close()
    finally:
        release.set()
