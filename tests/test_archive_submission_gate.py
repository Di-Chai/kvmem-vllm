"""Submission failures must not leave unowned I/O touching pinned destinations."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from vllm_kvmem import _native, storage
from vllm_kvmem.memory_storage import MemoryArchive


def archive_fixture(tmp_path, *, cached=False):
    if cached:
        from vllm_kvmem.cached_storage import CachedArchive
        archive = CachedArchive(str(tmp_path), 64, 4, cache_bytes=256, namespace_bytes=256)
    else:
        archive = MemoryArchive(str(tmp_path), 64, 4)
    repo = _native.Repository(4, 16)
    leases = [repo.reserve(i * 16, 16).commit() for i in range(2)]
    archive.write('owner', [leases[0]], b'a' * 64).result(3)
    # Exercise the real first-thread-start failure after seeding immutable data.
    archive.executor.shutdown(wait=True)
    archive.executor = ThreadPoolExecutor(max_workers=1)
    return archive, repo, leases


@pytest.mark.parametrize('operation', ['write', 'seal', 'read', 'region', 'cached_read'])
@pytest.mark.parametrize('failure', ['before', 'queued', 'running', 'holder'])
def test_failed_submission_never_runs_io_or_double_releases_admission(
        tmp_path, monkeypatch, operation, failure):
    archive, repo, leases = archive_fixture(tmp_path, cached=operation == 'cached_read')
    touched, hidden_entered, hidden_exited = [], Event(), Event()
    real_event = storage.Event

    class ObservedGate(real_event):
        def wait(self, *args, **kwargs):
            hidden_entered.set()
            result = super().wait(*args, **kwargs)
            hidden_exited.set()
            return result

    monkeypatch.setattr(storage, 'Event', ObservedGate)
    original_tier = archive.tier
    original_io, original_gather = original_tier.io, archive._gather_region

    def io(*args, **kwargs):
        touched.append('io')
        return original_io(*args, **kwargs)

    def gather(*args, **kwargs):
        touched.append('gather')
        return original_gather(*args, **kwargs)

    class ObservedTier:
        def __getattr__(self, name):
            return getattr(original_tier, name)

        def io(self, *args, **kwargs):
            return io(*args, **kwargs)

    archive.tier = ObservedTier()
    monkeypatch.setattr(archive, '_gather_region', gather)
    destination = bytearray(b'z' * (32 if operation == 'region' else 64))
    original_submit = archive.executor.submit
    original_adjust = archive.executor._adjust_thread_count
    original_holder = storage.ArchiveJob

    def fail_before(*args, **kwargs):
        raise RuntimeError('before enqueue')

    def fail_start():
        if failure == 'running':
            original_adjust()
            assert hidden_entered.wait(2), 'worker did not reach the submission gate'
        raise RuntimeError('thread start failed after enqueue')

    def fail_holder(future):
        assert hidden_entered.wait(2), 'worker did not wait for its completion owner'
        raise RuntimeError('completion allocation failed')

    if failure == 'before':
        monkeypatch.setattr(archive.executor, 'submit', fail_before)
    elif failure in ('queued', 'running'):
        monkeypatch.setattr(archive.executor, '_adjust_thread_count', fail_start)
    else:
        monkeypatch.setattr(storage, 'ArchiveJob', fail_holder)
    try:
        with pytest.raises(RuntimeError, match='enqueue|allocation'):
            if operation == 'write':
                archive.write('owner', [leases[1]], b'b' * 64)
            elif operation == 'seal':
                archive.write_from_views('owner', [leases[1]], [b'b' * 32, b'c' * 32])
            elif operation in ('read', 'cached_read'):
                archive.read('owner', [leases[0]], destination=destination)
            else:
                archive.read_region('owner', [leases[0]], 16, 32, destination=destination)
        assert not touched
        assert archive._pending == archive._io_bytes == 0
        assert archive.namespace_allocated_records('owner') == 1
        assert destination == b'z' * len(destination)

        # Wake the unseen queued task (or wait for the already-started one).
        # It must only exit, even after callers can reclaim the rejected lease.
        monkeypatch.setattr(archive.executor, 'submit', original_submit)
        monkeypatch.setattr(archive.executor, '_adjust_thread_count', original_adjust)
        monkeypatch.setattr(storage, 'ArchiveJob', original_holder)
        archive.executor.submit(lambda: None).result(3)
        if failure != 'before':
            assert hidden_exited.is_set()
        assert not touched
        assert archive._pending == archive._io_bytes == 0
        assert destination == b'z' * len(destination)

        # A rejected task does not poison this pool or resurrect rolled-back
        # records. The same generation remains usable through a new admission.
        archive.write('owner', [leases[1]], b'd' * 64).result(3)
        assert archive.read('owner', [leases[1]]).result(3)[0] == b'd' * 64
        assert archive._pending == archive._io_bytes == 0
        assert len(touched) == 2
        for lease in leases:
            handle = lease.handle
            assert repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda record: archive.reclaim('owner', record)) == _native.ReclaimResult.RECLAIMED
    finally:
        # Never leave an executor whose injected first-thread-start failure
        # prevents pending harmless wrappers from being removed at teardown.
        monkeypatch.setattr(archive.executor, 'submit', original_submit)
        monkeypatch.setattr(archive.executor, '_adjust_thread_count', original_adjust)
        monkeypatch.setattr(storage, 'ArchiveJob', original_holder)
        archive.executor.shutdown(wait=True, cancel_futures=True)
        assert archive.close()


def test_accepted_job_still_owns_its_leases_and_cannot_be_cancelled(tmp_path, monkeypatch):
    archive, repo, leases = archive_fixture(tmp_path)
    started, finish = Event(), Event()
    real_io = archive.tier.io

    def delayed(*args, **kwargs):
        started.set()
        assert finish.wait(3)
        return real_io(*args, **kwargs)

    monkeypatch.setattr(archive.tier, 'io', delayed)
    destination = bytearray(64)
    job = archive.read('owner', [leases[0]], destination=destination)
    try:
        assert started.wait(2)
        assert not job.cancel() and not job.done() and not archive.close()
        handle = leases[0].handle
        assert repo.retire(handle)
        leases[0].reset()
        assert repo.reclaim(handle, lambda record: archive.reclaim('owner', record)) == _native.ReclaimResult.READERS
    finally:
        finish.set()
    assert job.result(3)[0] is not None
    assert destination == b'a' * 64
    assert repo.reclaim(handle, lambda record: archive.reclaim('owner', record)) == _native.ReclaimResult.RECLAIMED
    assert archive._pending == archive._io_bytes == 0 and archive.close()
