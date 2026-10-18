from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread

import pytest

from vllm_kvmem import _native
from vllm_kvmem import memory_storage
from vllm_kvmem.memory_storage import (
    MemoryArchive, MemorySpan, MemoryTier, gather_records_native, gather_records_python,
)


@pytest.mark.parametrize('backend', [gather_records_python, gather_records_native])
@pytest.mark.parametrize('offset,length', [(0, 8), (0, 5), (5, 3)])
def test_reference_and_native_gather_preserve_order_and_disjoint_ranges(backend, offset, length):
    sources = [b'ABCDEFGH', b'12345678', b'abcdefgh']
    output = bytearray(b'_'*(3*length))
    assert backend(output, sources, 8, offset, length, 1, 3) == 2*length
    assert output == b'_'*length + b''.join(v[offset:offset+length] for v in sources[1:])
    assert backend(output, sources, 8, offset, length, 0, 1) == length
    assert output == b''.join(v[offset:offset+length] for v in sources)


@pytest.mark.parametrize('offset,length', [(0, 8), (2, 5)])
def test_missing_native_entry_points_fall_back_without_changing_result(monkeypatch, offset, length):
    monkeypatch.setattr(_native, 'gather_cached_records', None)
    monkeypatch.setattr(_native, 'gather_record_region', None)
    sources = [b'ABCDEFGH', b'abcdefgh']
    output = bytearray(2*length)
    assert gather_records_native(output, sources, 8, offset, length, 0, 2) == 2*length
    assert output == b''.join(v[offset:offset+length] for v in sources)


def test_present_native_failure_is_not_retried_as_python_fallback(monkeypatch):
    def fail(*args):
        raise OSError('native failure')
    monkeypatch.setattr(_native, 'gather_cached_records', fail)
    monkeypatch.setattr(memory_storage, 'gather_records_python', lambda *args: pytest.fail('Unexpected fallback'))
    with pytest.raises(OSError, match='native failure'):
        gather_records_native(bytearray(8), [b'abcdefgh'], 8, 0, 8, 0, 1)


@pytest.mark.parametrize('backend', [gather_records_python, gather_records_native])
def test_invalid_gather_fails_before_copying_destination(backend):
    output = bytearray(b'_'*16)
    with pytest.raises((ValueError, TypeError, BufferError)):
        backend(output, [b'abcdefgh', b'short'], 8, 0, 8, 0, 2)
    assert output == b'_'*16
    with pytest.raises((ValueError, TypeError, BufferError)):
        backend(bytes(16), [b'abcdefgh']*2, 8, 0, 8, 0, 2)


def _archive_records(tmp_path, size=96, records=3, workers=2):
    archive = MemoryArchive(str(tmp_path), size, records+2, gather_workers=workers)
    repo = _native.Repository(records+2, 4)
    leases = [repo.reserve(i*4, 4).commit() for i in range(records)]
    values = [bytes([i+1])*size for i in range(records)]
    archive.write('a', leases, b''.join(values)).result(5)
    return archive, repo, leases, values


@pytest.mark.parametrize('region', [False, True])
def test_cpu_authority_full_and_region_gather_obey_order_and_namespace(tmp_path, region):
    archive, repo, leases, values = _archive_records(tmp_path)
    order = [2, 0, 1]
    offset, length = (64, 32) if region else (0, 96)
    output = bytearray(3*length)
    try:
        job = (archive.read_region('a', [leases[i] for i in order], offset, length, destination=output)
               if region else archive.read('a', [leases[i] for i in order], destination=output))
        data, stats = job.result(5)
        assert memoryview(data).obj is output
        assert bytes(output) == b''.join(values[i][offset:offset+length] for i in order)
        assert stats['cache_gather_tasks'] == 1
        assert stats['cache_copy_seconds'] >= 0
        assert stats['syscalls'] == 0
        assert archive.io_usage()['disk_read_bytes'] == 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        with pytest.raises(KeyError):
            archive.read('wrong-namespace', leases)
    finally:
        assert archive.close()


@pytest.mark.parametrize('region', [False, True])
def test_parallel_cpu_authority_native_gather_matches_reference(tmp_path, region):
    size = 4*1024**2 if region else 2*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=4, workers=3)
    order = [3, 0, 2, 1]
    offset, length = (size//2, size//2) if region else (0, size)
    try:
        job = (archive.read_region('a', [leases[i] for i in order], offset, length) if region
               else archive.read('a', [leases[i] for i in order]))
        output, stats = job.result(5)
        expected = bytearray(4*length)
        gather_records_python(expected, [values[i] for i in order], size, offset, length, 0, 4)
        assert output == expected
        assert stats['cache_gather_tasks'] == 3
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
    finally:
        assert archive.close()


def test_full_record_gather_releases_tier_lock_during_copy_and_allows_publication(tmp_path, monkeypatch):
    archive, repo, leases, values = _archive_records(tmp_path, records=1)
    started, release = Event(), Event()
    original = memory_storage.gather_records_native

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(memory_storage, 'gather_records_native', blocked)
    job = archive.read('a', leases)
    try:
        assert started.wait(5)
        extra = repo.reserve(4, 4).commit()
        # This writer needs the same publication lock. Holding it across bulk
        # memcpy would prevent its future from completing before release.set().
        archive.write('a', [extra], b'z'*96).result(5)
        with archive.tier.lock:
            assert len(archive.tier.records) == 2
        assert not job.done()
        assert not job.cancel() and not archive.close()
        release.set()
        assert job.result(5)[0] == values[0]
        assert archive.read('a', [extra]).result(5)[0] == b'z'*96
    finally:
        release.set()
        job.result(5)
        assert archive.close()


def test_direct_tier_borrow_keeps_original_bytes_across_slot_reuse(monkeypatch):
    tier = MemoryTier(8, 1)
    first = tier.allocate(0)
    spans = [MemorySpan(0, 0, 8)]
    tier.io(spans, b'original', True)
    started, release = Event(), Event()
    original = memory_storage.gather_records_native

    def blocked(*args):
        started.set()
        assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(memory_storage, 'gather_records_native', blocked)
    output = bytearray(8)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(tier.io, spans, output, False)
        try:
            assert started.wait(5)
            assert tier.release(first)
            tier.allocate(0)
            tier.io(spans, b'replaced', True)
            release.set()
            future.result(5)
            assert output == b'original'
        finally:
            release.set()


def test_tier_keeps_general_destination_span_order_and_gaps():
    tier = MemoryTier(4, 2)
    tier.allocate(0); tier.allocate(1)
    tier.io([MemorySpan(0, 0, 4), MemorySpan(1, 4, 4)], b'abcdWXYZ', True)
    output = bytearray(b'_'*12)
    tier.io([MemorySpan(1, 8, 4), MemorySpan(0, 0, 4)], output, False)
    assert output == b'abcd____WXYZ'


@pytest.mark.parametrize('region', [False, True])
def test_failed_read_keeps_generation_lease_until_all_copy_tasks_drain(tmp_path, monkeypatch, region):
    # At most 16 MiB of source data; this checks ownership, not bandwidth.
    size = 8*1024**2 if region else 4*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=2, workers=2)
    started, release = Event(), Event()

    def injected(destination, sources, record_bytes, offset, length, begin, end):
        if begin == 0:
            assert started.wait(5)
            raise OSError('injected gather failure')
        started.set()
        assert release.wait(5)
        return gather_records_python(destination, sources, record_bytes, offset, length, begin, end)

    monkeypatch.setattr(memory_storage, 'gather_records_native', injected)
    length = size//2 if region else size
    output = bytearray(2*length)
    job = (archive.read_region('a', leases, size//2, length, destination=output) if region
           else archive.read('a', leases, destination=output))
    handle = leases[0].handle
    try:
        assert started.wait(5)
        assert not job.done() and not job.cancel() and not archive.close()
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle, lambda record: archive.reclaim('a', record)) == _native.ReclaimResult.READERS
        release.set()
        with pytest.raises(OSError, match='gather failure'):
            job.result(5)
        assert output[length:] == values[1][:length]
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert repo.reclaim(handle, lambda record: archive.reclaim('a', record)) == _native.ReclaimResult.RECLAIMED
    finally:
        release.set()
        assert archive.close()


def test_cpu_gather_worker_configuration_is_wired_into_both_paths(tmp_path):
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.storage import make_shared_archive
    archive = make_shared_archive(RuntimeOptions(archive_backend='cpu', cpu_read_cache_bytes=8192,
        cpu_gather_workers=3), tmp_path/'not-created', 4096, 8)
    try:
        assert archive.gather_workers == 3
        assert archive.tier.gather == archive._gather
        assert not (tmp_path/'not-created').exists()
    finally:
        assert archive.close()


@pytest.mark.parametrize('workers', [0, 17, True, 1.5])
def test_invalid_gather_worker_count_is_rejected_before_allocating_archive(tmp_path, workers):
    with pytest.raises(ValueError, match='gather workers'):
        MemoryArchive(str(tmp_path), 96, 3, gather_workers=workers)


def test_first_copy_thread_start_failure_cancels_unreturned_future_permanently(tmp_path, monkeypatch):
    size = 4*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=2, workers=2)
    output = bytearray(2*size)
    original_start = Thread.start
    calls = []
    original_copy = memory_storage.gather_records_native

    def copy(*args):
        calls.append(1)
        return original_copy(*args)

    def fail_start(thread):
        if thread.name.startswith('kvmem-cpu-gather'):
            raise RuntimeError('first copy thread failed to start')
        return original_start(thread)

    monkeypatch.setattr(memory_storage, 'gather_records_native', copy)
    monkeypatch.setattr(Thread, 'start', fail_start)
    try:
        with pytest.raises(RuntimeError, match='failed to start'):
            archive.read('a', leases, destination=output).result(5)
        assert archive._gather_shutdown_done.is_set()
        assert not archive._gather_executor._threads
        assert not calls and output == bytes(2*size)
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        monkeypatch.setattr(Thread, 'start', original_start)
        # Restoring thread creation cannot resurrect the task that was queued
        # before submit raised without returning a Future.
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.read('a', [leases[0]]).result(5)  # Inline small-copy path.
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.read_region('a', leases, size//2, size//2).result(5)
        with pytest.raises(RuntimeError, match='shutdown'):
            archive._gather_executor.submit(lambda: None)
        assert not calls and output == bytes(2*size)
    finally:
        assert archive.close()


def test_enqueued_unreturned_running_copy_is_joined_before_read_lease_release(tmp_path, monkeypatch):
    size = 4*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=2, workers=2)
    started, release, draining = Event(), Event(), Event()
    original_adjust = archive._gather_executor._adjust_thread_count
    original_shutdown = archive._gather_executor.shutdown
    calls = []

    def blocked_copy(*args):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return gather_records_python(*args)

    def fail_after_start():
        original_adjust()
        assert started.wait(5)
        raise OSError('adjust failed after enqueuing and starting copy')

    def shutdown(*args, **kwargs):
        assert kwargs == {'wait': True, 'cancel_futures': True}
        draining.set()
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(memory_storage, 'gather_records_native', blocked_copy)
    monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', fail_after_start)
    monkeypatch.setattr(archive._gather_executor, 'shutdown', shutdown)
    output = bytearray(2*size)
    job = archive.read('a', leases, destination=output)
    handle = leases[0].handle
    try:
        assert draining.wait(5)
        assert not job.done() and not archive.close()
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle, lambda record: archive.reclaim('a', record)) == _native.ReclaimResult.READERS
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.read('a', [leases[1]]).result(5)
        release.set()
        with pytest.raises(OSError, match='adjust failed'):
            job.result(5)
        assert archive._gather_shutdown_done.is_set()
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert repo.reclaim(handle, lambda record: archive.reclaim('a', record)) == _native.ReclaimResult.RECLAIMED
        expected = values[0]+bytes(size)
        assert output == expected
        monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', original_adjust)
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.read('a', [leases[1]]).result(5)
        assert output == expected and len(calls) == 1
    finally:
        release.set()
        assert archive.close()


def test_concurrent_gather_submission_failure_cancels_untracked_queue_and_serializes_close(tmp_path, monkeypatch):
    size = 4*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=2, workers=2)
    both_started, release, draining = Event(), Event(), Event()
    lock, copies = Lock(), []
    original_adjust = archive._gather_executor._adjust_thread_count
    original_shutdown = archive._gather_executor.shutdown

    def blocked_copy(*args):
        with lock:
            copies.append(args[-2:])
            if len(copies) == 2:
                both_started.set()
        assert release.wait(5)
        return gather_records_python(*args)

    def fail_after_enqueue():
        original_adjust()  # Both workers are busy; the new task stays queued.
        raise OSError('concurrent submit failed after enqueue')

    def shutdown(*args, **kwargs):
        draining.set()
        return original_shutdown(*args, **kwargs)

    monkeypatch.setattr(memory_storage, 'gather_records_native', blocked_copy)
    first_output, failed_output = bytearray(2*size), bytearray(2*size)
    first = archive.read('a', leases, destination=first_output)
    failed = None
    try:
        assert both_started.wait(5)
        monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', fail_after_enqueue)
        monkeypatch.setattr(archive._gather_executor, 'shutdown', shutdown)
        failed = archive.read('a', leases, destination=failed_output)
        assert draining.wait(5)
        assert not first.done() and not failed.done() and not archive.close()
        # A caller racing with shutdown must not submit another chunk, even
        # when it would normally take the inline small-copy path.
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive._gather(bytearray(size), [values[0]], 0, size)
        release.set()
        assert first.result(5)[0] == b''.join(values)
        with pytest.raises(OSError, match='concurrent submit'):
            failed.result(5)
        assert failed_output == bytes(2*size)
        assert len(copies) == 2
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', original_adjust)
        with pytest.raises(RuntimeError, match='pool unavailable'):
            archive.read('a', leases).result(5)
        assert failed_output == bytes(2*size) and len(copies) == 2
        with ThreadPoolExecutor(max_workers=2) as callers:
            assert all(f.result(5) for f in [callers.submit(archive.close), callers.submit(archive.close)])
    finally:
        release.set()
        assert archive.close()


def test_copy_worker_cannot_close_or_reenter_its_own_pool(tmp_path, monkeypatch):
    size = 4*1024**2
    archive, repo, leases, values = _archive_records(tmp_path, size=size, records=2, workers=2)

    def copy(*args):
        with pytest.raises(RuntimeError, match='cannot close its own pool'):
            archive.close()
        with pytest.raises(RuntimeError, match='cannot recursively gather'):
            archive._gather(bytearray(0), [], 0, size)
        return gather_records_python(*args)

    monkeypatch.setattr(memory_storage, 'gather_records_native', copy)
    try:
        assert archive.read('a', leases).result(5)[0] == b''.join(values)
    finally:
        assert archive.close()
