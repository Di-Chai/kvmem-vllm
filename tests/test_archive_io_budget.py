"""Real native I/O under shared byte pressure, including cache and failure paths."""
from threading import Event

import pytest

from vllm_kvmem import _native
from vllm_kvmem.storage import AsyncArchive
from vllm_kvmem.cached_storage import CachedArchive


def archive_at(path, cached, limit=8192):
    kwargs = dict(cache_bytes=16384, namespace_bytes=16384) if cached else {}
    return (CachedArchive if cached else AsyncArchive)(
        str(path), 4096, 8, max_pending=4, io_max_bytes=limit, **kwargs)


@pytest.mark.parametrize('cached', [False, True])
def test_byte_pressure_across_namespaces_and_direct_cache_reads(tmp_path, cached):
    archive = archive_at(tmp_path, cached)
    repo = _native.Repository(3, 16)
    leases = [repo.reserve(i * 16, 16).commit() for i in range(3)]
    archive.write('a', leases[:2], b'a' * 8192).result(5)
    tier = archive.tier
    started, proceed = Event(), Event()

    class Delayed:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            started.set()
            assert proceed.wait(5)
            return tier.io(*args)

    archive.tier = Delayed()
    destination = bytearray(8192)
    reading = archive.read('a', leases[:2], destination=destination)
    try:
        assert started.wait(5)
        # One pending job leaves three count slots, but fills all byte credits.
        assert archive.write('b', leases[2:], b'b' * 4096) is None
        assert archive.read('a', leases[:1]) is None
        assert not archive.wait_for_capacity(.01, size=4096)
        assert not archive.close()
        assert archive.io_usage() == dict(archive_io_max_bytes=8192,
            archive_io_pending_bytes=8192, archive_io_peak_bytes=8192, archive_io_deferrals=2)
        assert archive.tier.used_slots == 2
    finally:
        proceed.set()
    data, stats = reading.result(5)
    assert data.obj is destination and destination == b'a' * 8192
    if cached:
        assert stats['bytes'] == 0 and stats['cache_hit_bytes'] == 8192
    assert archive.wait_for_capacity(.1, size=8192)
    archive.write('b', leases[2:], b'b' * 4096).result(5)
    assert archive.read('b', leases[2:]).result(5)[0] == b'b' * 4096
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    assert archive.close()


def test_serving_byte_quota_rejects_unserviceable_batches_before_open(tmp_path):
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.storage import make_shared_archive
    options = RuntimeOptions(archive_io_max_bytes=4096)
    with pytest.raises(ValueError, match='full serving batch'):
        make_shared_archive(options, tmp_path / 'never-opened', 4096, 4)
    assert not (tmp_path / 'never-opened').exists()


def test_capture_waits_for_detached_retirement_bytes_instead_of_false_capacity_failure(tmp_path):
    from vllm_kvmem.capture_runtime import CaptureRuntime
    archive = archive_at(tmp_path, False, limit=4096)
    repo = _native.Repository(1, 16)
    lease = repo.reserve(0, 16).commit()
    tier = archive.tier
    started, entered_wait = Event(), Event()

    class Delayed:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            started.set()
            assert entered_wait.wait(5)
            return tier.io(*args)

    archive.tier = Delayed()
    job = archive.write('retired', [lease], b'x' * 4096)
    assert started.wait(5)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.archive, runtime.authorities, runtime.retiring = archive, {}, {'retired': {}}
    original_wait = archive.wait_for_idle

    def wait(timeout):
        entered_wait.set()
        return original_wait(timeout)

    archive.wait_for_idle = wait
    runtime._wait_archive_progress()
    assert entered_wait.is_set() and archive.io_usage()['archive_io_pending_bytes'] == 0
    job.result(5)
    assert archive.close()


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('failure', ['write_io', 'read_io', 'write_submit', 'read_submit'])
def test_io_and_submission_failure_release_byte_credits_and_generation_leases(tmp_path, cached, failure):
    archive = archive_at(tmp_path, cached, limit=4096)
    repo = _native.Repository(1, 16)
    lease = repo.reserve(0, 16).commit()
    if failure.startswith('read'):
        archive.write('a', [lease], b'a' * 4096).result(5)
    tier = archive.tier

    class Failing:
        def __getattr__(self, name):
            return getattr(tier, name)

        def io(self, *args):
            raise OSError('injected I/O error')

    if failure.endswith('io'):
        archive.tier = Failing()
    else:
        archive.executor.shutdown()
    with pytest.raises((RuntimeError, OSError)):
        if failure.startswith('write'):
            archive.write('a', [lease], b'a' * 4096).result(5)
        else:
            archive.read('a', [lease]).result(5)
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    assert archive.wait_for_capacity(.1)
    handle = lease.handle
    assert repo.retire(handle)
    lease.reset()
    assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == _native.ReclaimResult.RECLAIMED
    assert archive.tier.used_slots == 0
    if cached:
        assert archive.read_cache.used_bytes == 0
    assert archive.close()


@pytest.mark.parametrize('cached', [False, True])
def test_impossible_batch_fails_before_reserving_or_snapshotting(tmp_path, cached):
    archive = archive_at(tmp_path, cached, limit=4096)
    repo = _native.Repository(2, 16)
    leases = [repo.reserve(i * 16, 16).commit() for i in range(2)]

    class Payload:
        def __len__(self):
            return 8192

        def __bytes__(self):
            raise AssertionError('Must not copy an unadmitted payload')

    with pytest.raises(ValueError, match='batch exceeds'):
        archive.write('a', leases, Payload())
    with pytest.raises(ValueError, match='batch exceeds'):
        archive.wait_for_capacity(.1, size=8192)
    assert archive.tier.used_slots == 0
    assert archive.io_usage()['archive_io_peak_bytes'] == 0
    assert archive.close()
