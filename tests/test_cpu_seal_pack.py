"""Parallel immutable record packing preserves leases, order and rollback."""
import hashlib
import weakref
from threading import Event, Lock, Thread

import numpy as np
import pytest

from test_cpu_direct_record_seal import repository, reclaim_all
from vllm_kvmem import _native
from vllm_kvmem.memory_storage import MemoryArchive


def archive_at(path, workers=4, **kwargs):
    return MemoryArchive(str(path), 4, kwargs.pop('capacity', 17),
        cpu_direct_record_seal=True, cpu_seal_pack_workers=workers, **kwargs)


@pytest.mark.parametrize('workers,count', [(1, 17), (2, 17), (4, 17), (8, 17), (8, 3), (8, 1)])
@pytest.mark.parametrize('hash_workers', [1, 8])
def test_records_digests_order_and_independent_ownership(tmp_path, monkeypatch, workers, count, hash_workers):
    archive = archive_at(tmp_path, workers, cpu_seal_hash_workers=hash_workers)
    repo, leases, handles = repository(count)
    expected = [bytes([i, 255-i, i+1, 254-i]) for i in range(count)]
    sources = [bytearray(part) for value in expected for part in (value[:2], value[2:])]
    pack, packed, lock = _native.pack_record_pairs, [], Lock()

    def observed(views, record_bytes):
        result = pack(views, record_bytes)
        with lock:
            packed.extend(result)
        return result

    monkeypatch.setattr(_native, 'pack_record_pairs', observed)
    stats = archive.write_from_views('owner', leases, sources).result(5)
    assert stats['record_digests'] == tuple(hashlib.sha256(record).hexdigest() for record in expected)
    assert len(packed) == count and len({id(record) for record in packed}) == count
    for index, entry in enumerate(archive._entries.values()):
        value = archive.tier.records[entry.allocation.slot]
        assert value == expected[index] and any(value is record for record in packed)
    for source in sources:
        source[:] = b'!!'
    assert archive.read('owner', leases[::-1]).result(5)[0] == b''.join(expected[::-1])
    usage = archive.io_usage()
    assert usage['seal_records'] == count and usage['seal_payload_bytes'] == 4*count
    assert usage['seal_pack_seconds'] == stats['seal_pack_seconds']
    assert usage['cpu_seal_pack_workers'] == workers
    assert usage['seal_pack_parallel_batches_completed'] == (workers > 1 and count > 1)
    assert usage['seal_pack_parallel_shards_completed'] == (min(workers, count) if workers > 1 and count > 1 else 0)
    assert len(archive._gather_executor._threads) <= 8
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_out_of_order_workers_are_published_in_record_order(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 2, capacity=2)
    repo, leases, handles = repository(2)
    later_done = Event()
    pack = archive._pack_range

    def reordered(views, start, end):
        if start == 0:
            assert later_done.wait(5)
            return pack(views, start, end)
        result = pack(views, start, end)
        later_done.set()
        return result

    monkeypatch.setattr(archive, '_pack_range', reordered)
    stats = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ']).result(5)
    assert stats['record_digests'] == tuple(hashlib.sha256(record).hexdigest() for record in (b'abcd', b'WXYZ'))
    assert archive.read('owner', leases).result(5)[0] == b'abcdWXYZ'
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_worker_failure_drains_peer_before_atomic_rollback(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 2, capacity=2)
    repo, leases, handles = repository(2)
    entered, failed, release = Event(), Event(), Event()
    pack = _native.pack_record_pairs

    def failing(views, record_bytes):
        try:
            if bytes(views[0]) == b'ab':
                assert entered.wait(5)
                failed.set()
                raise OSError('pack shard failed')
            entered.set()
            assert release.wait(5)
            return pack(views, record_bytes)
        finally:
            views = None

    monkeypatch.setattr(_native, 'pack_record_pairs', failing)
    job = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
    try:
        assert failed.wait(5)
        assert not job.done() and not job.cancel() and not archive.close()
        assert archive.io_usage()['archive_io_pending_bytes'] == 16
        assert not archive.tier.records and not any(entry.ready for entry in archive._entries.values())
        with pytest.raises(KeyError, match='not yet published'):
            archive.read('owner', leases)
        for lease, handle in zip(leases, handles):
            repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda r: archive.reclaim('owner', r)) == _native.ReclaimResult.READERS
        release.set()
        with pytest.raises(OSError, match='pack shard failed'):
            job.result(5)
        assert not archive._entries and archive.tier.used_slots == 0
        usage = archive.io_usage()
        assert usage['archive_io_pending_bytes'] == usage['seal_batches'] == 0
        assert usage['seal_pack_parallel_batches_completed'] == usage['seal_pack_parallel_shards_completed'] == 0
        monkeypatch.setattr(_native, 'pack_record_pairs', pack)
        assert archive._pack_records((b'ab', b'cd', b'WX', b'YZ')) == [b'abcd', b'WXYZ']
        reclaim_all(repo, leases, handles, archive)
    finally:
        release.set()
        assert archive.close()


def test_failed_future_does_not_keep_source_exports(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 2, capacity=2)
    repo, leases, handles = repository(2)
    sources = [np.array([i, i+1], dtype=np.uint8) for i in range(4)]
    refs = [weakref.ref(source) for source in sources]

    def failing(views, record_bytes):
        views = None
        raise OSError('pack without retained inputs')

    monkeypatch.setattr(_native, 'pack_record_pairs', failing)
    job = archive.write_from_views('owner', leases, sources)
    sources.clear()
    with pytest.raises(OSError, match='without retained inputs'):
        job.result(5)
    assert all(ref() is None for ref in refs)
    assert not archive._entries and archive.io_usage()['archive_io_pending_bytes'] == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close() and job.done()


def test_completed_pack_count_does_not_claim_failed_hash_publication(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 2, capacity=2, cpu_seal_hash_workers=2)
    repo, leases, handles = repository(2)

    def failing(record):
        record = None
        raise OSError('hash after completed pack failed')

    monkeypatch.setattr(hashlib, 'sha256', failing)
    with pytest.raises(OSError, match='after completed pack'):
        archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ']).result(5)
    usage = archive.io_usage()
    assert usage['seal_pack_parallel_batches_completed'] == 1
    assert usage['seal_pack_parallel_shards_completed'] == 2
    assert usage['seal_batches'] == usage['seal_records'] == usage['seal_payload_bytes'] == 0
    assert not archive._entries and archive.tier.used_slots == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_invalid_complete_batch_is_rejected_before_any_shard(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 8, capacity=2)
    repo, leases, handles = repository(2)
    calls = []
    monkeypatch.setattr(archive, '_pack_range', lambda *args: calls.append(1))
    with pytest.raises(ValueError, match='each complete'):
        archive.write_from_views('owner', leases, [b'ab', b'cd', b'W', b'YZ'])
    assert not calls and not archive._entries and not archive._gather_executor._threads
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_two_owners_share_existing_bounded_pool_and_admission(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 8, capacity=34, gather_workers=2, max_pending=2, io_max_bytes=272)
    repo, leases, handles = repository(17)
    queued, release, lock = Event(), Event(), Lock()
    submit, pack = archive._gather_executor.submit, archive._pack_range
    jobs = []

    def blocked(*args):
        assert release.wait(5)
        return pack(*args)

    def observed(*args, **kwargs):
        future = submit(*args, **kwargs)
        with lock:
            jobs.append(future)
            if len(jobs) == 16:
                queued.set()
        return future

    monkeypatch.setattr(archive, '_pack_range', blocked)
    monkeypatch.setattr(archive._gather_executor, 'submit', observed)
    first = archive.write_from_views('owner', leases, [b'ab', b'cd']*17)
    second = archive.write_from_views('other', leases, [b'WX', b'YZ']*17)
    try:
        assert queued.wait(5)
        assert len(jobs) == 16 and len(archive._gather_executor._threads) == 2
        assert archive.io_usage()['archive_io_pending_bytes'] == 272
        assert archive.write_from_views('third', leases[:1], [b'12', b'34']) is None
        assert not first.cancel() and not second.cancel() and not archive.close()
        release.set()
        assert first.result(5)['record_digests'] == (hashlib.sha256(b'abcd').hexdigest(),)*17
        assert second.result(5)['record_digests'] == (hashlib.sha256(b'WXYZ').hexdigest(),)*17
        assert archive.read('owner', leases).result(5)[0] == b'abcd'*17
        assert archive.read('other', leases).result(5)[0] == b'WXYZ'*17
        usage = archive.io_usage()
        assert usage['seal_pack_parallel_batches_completed'] == 2
        assert usage['seal_pack_parallel_shards_completed'] == 16
        for lease, handle in zip(leases, handles):
            repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda record:
                archive.reclaim('owner', record) and archive.reclaim('other', record)) == _native.ReclaimResult.RECLAIMED
    finally:
        release.set()
        assert archive.close()


@pytest.mark.parametrize('running', [False, True])
def test_submit_failure_drains_unreturned_work_and_poisons_pool(tmp_path, monkeypatch, running):
    archive = archive_at(tmp_path, 2, capacity=2)
    repo, leases, handles = repository(2)
    entered, release, draining = Event(), Event(), Event()
    adjust, shutdown, pack, start = (archive._gather_executor._adjust_thread_count,
        archive._gather_executor.shutdown, archive._pack_range, Thread.start)
    calls = []

    def blocked(*args):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return pack(*args)

    def fail_after_start():
        adjust()
        assert entered.wait(5)
        raise OSError('pack submit failed after starting')

    def fail_start(thread):
        if thread.name.startswith('kvmem-cpu-gather'):
            raise OSError('pack worker failed to start')
        return start(thread)

    def drain(*args, **kwargs):
        assert kwargs == dict(wait=True, cancel_futures=True)
        draining.set()
        return shutdown(*args, **kwargs)

    monkeypatch.setattr(archive, '_pack_range', blocked)
    monkeypatch.setattr(archive._gather_executor, 'shutdown', drain)
    if running:
        monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', fail_after_start)
    else:
        monkeypatch.setattr(Thread, 'start', fail_start)
    job = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
    try:
        assert draining.wait(5)
        if running:
            assert not job.done() and not archive.close()
            assert archive.io_usage()['archive_io_pending_bytes'] == 16
        release.set()
        with pytest.raises(OSError, match='failed'):
            job.result(5)
        assert calls == ([1] if running else [])
        assert archive._gather_shutdown_done.is_set()
        assert not archive._entries and archive.tier.used_slots == 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.write_from_views('owner', leases[:1], [b'ab', b'cd']).result(5)
        reclaim_all(repo, leases, handles, archive)
    finally:
        release.set()
        assert archive.close()


def test_older_extension_uses_serial_fallback(tmp_path, monkeypatch):
    archive = archive_at(tmp_path, 8, capacity=2)
    repo, leases, handles = repository(2)
    monkeypatch.setattr(_native, 'pack_record_pairs', None)
    archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ']).result(5)
    assert archive.read('owner', leases).result(5)[0] == b'abcdWXYZ'
    assert archive.io_usage()['seal_pack_parallel_batches_completed'] == 0
    assert not archive._gather_executor._threads
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


@pytest.mark.parametrize('workers', [1, 8])
def test_pack_cannot_recursively_submit_to_its_worker_pool(tmp_path, workers):
    archive = archive_at(tmp_path, workers)

    def nested():
        archive._gather_worker.active = True
        try:
            with pytest.raises(RuntimeError, match='recursively pack'):
                archive._pack_records((b'ab', b'cd'))
        finally:
            archive._gather_worker.active = False

    archive._gather_executor.submit(nested).result(5)
    assert archive.close()


@pytest.mark.parametrize('value', [True, False, 0, -1, 9, None, 1.5, '8'])
def test_pack_worker_configuration_is_strict(tmp_path, value):
    with pytest.raises(ValueError, match='cpu_seal_pack_workers must be an integer'):
        archive_at(tmp_path, value)


@pytest.mark.parametrize('direct,native', [(False, False), (True, True), (False, True)])
def test_parallel_pack_rejects_ineligible_backends(tmp_path, direct, native):
    with pytest.raises(ValueError, match='requires direct CPU records without native payloads'):
        MemoryArchive(str(tmp_path), 4, 2, cpu_seal_pack_workers=8,
                      cpu_direct_record_seal=direct, native_payloads=native)
