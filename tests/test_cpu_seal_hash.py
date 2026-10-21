"""Bounded CPU seal hashing preserves publication, ownership and rollback."""
from copy import deepcopy
from dataclasses import fields, replace
import hashlib
import json
import weakref
from threading import Event, Lock, Thread

import pytest
import numpy as np

from test_cpu_direct_record_seal import repository, reclaim_all
from test_graph_options import model_profile, runtime, worker_payload
from test_graph_v2_options import elastic_profile
from test_historical_manager import eventually
from vllm_kvmem import _native
from vllm_kvmem.elastic_serve import build_launch as elastic_launch
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from vllm_kvmem.storage import make_shared_archive


def enabled(**updates):
    return dict(archive_backend='cpu', cpu_read_cache_bytes=1024**3,
                async_seal=True, cpu_seal_hash_workers=4, **updates)


@pytest.mark.parametrize('workers', [1, 4, 8])
@pytest.mark.parametrize('mode', ['slab', 'direct', 'native'])
def test_digest_and_record_order_identical_for_all_cpu_publication_paths(tmp_path, workers, mode):
    archive = MemoryArchive(str(tmp_path), 4, 17, cpu_seal_hash_workers=workers,
                            cpu_direct_record_seal=mode == 'direct', native_payloads=mode == 'native')
    repo, leases, handles = repository(17)
    records = [bytes([i, 255-i, i+1, 254-i]) for i in range(17)]
    sources = [bytearray(part) for record in records for part in (record[:2], record[2:])]
    stats = archive.write_from_views('owner', leases, sources).result(5)
    assert stats['record_digests'] == tuple(hashlib.sha256(record).hexdigest() for record in records)
    for source in sources:
        source[:] = b'!!'
    assert archive.read('owner', leases[::-1]).result(5)[0] == b''.join(records[::-1])
    assert archive.io_usage()['seal_records'] == 17
    assert archive.io_usage()['cpu_seal_hash_workers'] == workers
    assert archive.io_usage()['seal_hash_parallel_batches_completed'] == (workers > 1)
    assert archive.io_usage()['seal_hash_parallel_shards_completed'] == (workers if workers > 1 else 0)
    assert len(archive._gather_executor._threads) <= min(workers, archive.gather_workers)
    if workers == 1:
        assert not archive._gather_executor._threads
    if mode == 'native':
        # Native payload release is asynchronous after the last reader exits.
        for lease, handle in zip(leases, handles):
            repo.retire(handle)
            lease.reset()
            eventually(lambda: repo.reclaim(handle, lambda r: archive.reclaim('owner', r))
                       == _native.ReclaimResult.RECLAIMED)
        eventually(archive.close)
    else:
        reclaim_all(repo, leases, handles, archive)
        assert archive.close()


@pytest.mark.parametrize('mode', ['slab', 'direct', 'native'])
def test_failed_parallel_hash_future_does_not_retain_original_source_exports(tmp_path, monkeypatch, mode):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_seal_hash_workers=2,
                            cpu_direct_record_seal=mode == 'direct', native_payloads=mode == 'native')
    repo, leases, handles = repository(2)
    sources = [np.array([i, i+1], dtype=np.uint8) for i in range(4)]
    refs = [weakref.ref(source) for source in sources]

    def fail_hash(record):
        # The injected failure also obeys the callback's ownership contract.
        record = None
        raise OSError('hash failed without source retention')

    monkeypatch.setattr(hashlib, 'sha256', fail_hash)
    job = archive.write_from_views('owner', leases, sources)
    sources.clear()
    with pytest.raises(OSError, match='without source retention'):
        job.result(5)
    assert all(ref() is None for ref in refs)
    assert not any(entry.ready for entry in archive._entries.values())
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    # Retain the failed Future while checking cleanup, including native deferred frees.
    for lease, handle in zip(leases, handles):
        repo.retire(handle)
        lease.reset()
        eventually(lambda: repo.reclaim(handle, lambda r: archive.reclaim('owner', r))
                   == _native.ReclaimResult.RECLAIMED)
    assert not archive._entries and archive.tier.used_slots == 0
    eventually(archive.close)
    assert job.done()


@pytest.mark.parametrize('direct', [False, True])
def test_worker_failure_drains_peers_before_atomic_rollback_and_keeps_leases(tmp_path, monkeypatch, direct):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_seal_hash_workers=2, cpu_direct_record_seal=direct)
    repo, leases, handles = repository(2)
    entered, failed, release = Event(), Event(), Event()
    sha256 = hashlib.sha256

    def hash_record(record):
        if bytes(record) == b'abcd':
            assert entered.wait(5)
            failed.set()
            raise OSError('seal hash failed')
        entered.set()
        assert release.wait(5)
        return sha256(record)

    monkeypatch.setattr(hashlib, 'sha256', hash_record)
    job = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
    try:
        assert failed.wait(5)
        assert not job.done() and not job.cancel() and not archive.close()
        assert archive.io_usage()['archive_io_pending_bytes'] == 16
        assert not any(entry.ready for entry in archive._entries.values())
        with pytest.raises(KeyError, match='not yet published'):
            archive.read('owner', leases)
        for lease, handle in zip(leases, handles):
            repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda r: archive.reclaim('owner', r)) == _native.ReclaimResult.READERS
        release.set()
        with pytest.raises(OSError, match='seal hash failed'):
            job.result(5)
        assert not archive._entries and archive.tier.used_slots == 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        assert archive.io_usage()['seal_batches'] == 0
        assert archive.io_usage()['seal_hash_parallel_batches_completed'] == 0
        assert archive.io_usage()['seal_hash_parallel_shards_completed'] == 0
        reclaim_all(repo, leases, handles, archive)
        # A hash error does not poison the pool; only submission failures do.
        monkeypatch.setattr(hashlib, 'sha256', sha256)
        assert archive._record_digests([b'abcd', b'WXYZ']) == tuple(
            sha256(value).hexdigest() for value in (b'abcd', b'WXYZ'))
    finally:
        release.set()
        assert archive.close()


def test_two_admitted_seals_have_bounded_shards_and_no_unbounded_queue(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 64, gather_workers=2, cpu_seal_hash_workers=4,
                            max_pending=2, io_max_bytes=1024)
    repo, leases, handles = repository(64)
    queued, release, lock = Event(), Event(), Lock()
    submit, hash_range = archive._gather_executor.submit, archive._hash_range
    jobs = []

    def blocked(*args):
        assert release.wait(5)
        return hash_range(*args)

    def observed_submit(*args, **kwargs):
        future = submit(*args, **kwargs)
        with lock:
            jobs.append(future)
            if len(jobs) == 8:
                queued.set()
        return future

    monkeypatch.setattr(archive, '_hash_range', blocked)
    monkeypatch.setattr(archive._gather_executor, 'submit', observed_submit)
    first = archive.write_from_views('owner', leases[:32], [b'ab', b'cd']*32)
    second = archive.write_from_views('owner', leases[32:], [b'WX', b'YZ']*32)
    try:
        assert queued.wait(5)
        assert len(jobs) == 8 and len(archive._gather_executor._threads) == 2
        assert archive.write_from_views('other', leases[:1], [b'12', b'34']) is None
        assert len(jobs) == 8 and not first.cancel() and not archive.close()
        assert archive.io_usage()['archive_io_pending_bytes'] == 512
        release.set()
        assert first.result(5)['record_digests'] == (hashlib.sha256(b'abcd').hexdigest(),)*32
        assert second.result(5)['record_digests'] == (hashlib.sha256(b'WXYZ').hexdigest(),)*32
        assert archive.io_usage()['seal_hash_parallel_batches_completed'] == 2
        assert archive.io_usage()['seal_hash_parallel_shards_completed'] == 8
        assert archive.read('owner', leases).result(5)[0] == b'abcd'*32+b'WXYZ'*32
        reclaim_all(repo, leases, handles, archive)
    finally:
        release.set()
        assert archive.close()


def test_first_hash_thread_start_failure_cancels_unreturned_queued_work(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_seal_hash_workers=2)
    repo, leases, handles = repository(2)
    start = Thread.start
    calls = []

    def fail_start(thread):
        if thread.name.startswith('kvmem-cpu-gather'):
            raise RuntimeError('seal thread failed to start')
        return start(thread)

    monkeypatch.setattr(archive, '_hash_range', lambda *args: calls.append(1))
    monkeypatch.setattr(Thread, 'start', fail_start)
    with pytest.raises(RuntimeError, match='failed to start'):
        archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ']).result(5)
    assert archive._gather_shutdown_done.is_set() and not archive._gather_executor._threads
    assert not calls and not archive._entries and archive.tier.used_slots == 0
    assert archive.io_usage()['archive_io_pending_bytes'] == 0
    monkeypatch.setattr(Thread, 'start', start)
    with pytest.raises(RuntimeError, match='pool unavailable'):
        archive.write_from_views('owner', leases[:1], [b'ab', b'cd']).result(5)
    assert not calls and not archive._entries
    reclaim_all(repo, leases, handles, archive)
    assert archive.close()


def test_unreturned_running_hash_is_drained_before_releasing_records_and_leases(tmp_path, monkeypatch):
    archive = MemoryArchive(str(tmp_path), 4, 2, cpu_seal_hash_workers=2)
    repo, leases, handles = repository(2)
    entered, release, draining = Event(), Event(), Event()
    adjust = archive._gather_executor._adjust_thread_count
    shutdown = archive._gather_executor.shutdown
    hash_range = archive._hash_range

    def blocked_hash(*args):
        entered.set()
        assert release.wait(5)
        return hash_range(*args)

    def fail_after_start():
        adjust()
        assert entered.wait(5)
        raise OSError('seal submission failed after starting')

    def drain(*args, **kwargs):
        assert kwargs == dict(wait=True, cancel_futures=True)
        draining.set()
        return shutdown(*args, **kwargs)

    monkeypatch.setattr(archive, '_hash_range', blocked_hash)
    monkeypatch.setattr(archive._gather_executor, '_adjust_thread_count', fail_after_start)
    monkeypatch.setattr(archive._gather_executor, 'shutdown', drain)
    job = archive.write_from_views('owner', leases, [b'ab', b'cd', b'WX', b'YZ'])
    try:
        assert draining.wait(5)
        assert not job.done() and not archive.close()
        assert archive.io_usage()['archive_io_pending_bytes'] == 16
        assert not any(entry.ready for entry in archive._entries.values())
        for lease, handle in zip(leases, handles):
            repo.retire(handle)
            lease.reset()
            assert repo.reclaim(handle, lambda r: archive.reclaim('owner', r)) == _native.ReclaimResult.READERS
        release.set()
        with pytest.raises(OSError, match='failed after starting'):
            job.result(5)
        assert archive._gather_shutdown_done.is_set()
        assert not archive._entries and archive.tier.used_slots == 0
        assert archive.io_usage()['archive_io_pending_bytes'] == 0
        reclaim_all(repo, leases, handles, archive)
    finally:
        release.set()
        assert archive.close()


def test_default_and_existing_positional_arguments_stay_unchanged():
    baseline = RuntimeOptions()
    assert baseline.cpu_seal_hash_workers == 1 and runtime().cpu_seal_hash_workers == 1
    names = [field.name for field in fields(RuntimeOptions)]
    assert names.index('cpu_seal_hash_workers') == names.index('prefill_capture_ahead') + 1
    assert RuntimeOptions(*[getattr(baseline, name) for name in names[:-1]]) == baseline


@pytest.mark.parametrize('value', [0, 9, True, False, 1.0, '4', None])
def test_invalid_hash_worker_configuration_fails_before_allocating(tmp_path, value):
    with pytest.raises(ValueError, match='cpu_seal_hash_workers must be an integer'):
        RuntimeOptions(cpu_seal_hash_workers=value)
    with pytest.raises(ValueError, match='cpu_seal_hash_workers must be an integer'):
        runtime(dict(enabled(), cpu_seal_hash_workers=value))
    with pytest.raises(ValueError, match='cpu_seal_hash_workers must be an integer'):
        MemoryArchive(str(tmp_path/'unused'), 4, 2, cpu_seal_hash_workers=value)
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('backend,async_seal', [('nvme', False), ('nvme', True), ('cpu', False)])
def test_parallel_hashing_requires_cpu_async_seal(backend, async_seal):
    with pytest.raises(ValueError, match='requires archive_backend=cpu and async_seal=true'):
        runtime(dict(enabled(), archive_backend=backend, async_seal=async_seal))
    assert runtime(dict(enabled(), archive_backend=backend, async_seal=async_seal,
                        cpu_seal_hash_workers=1)).cpu_seal_hash_workers == 1


def test_profile_worker_roundtrip_and_shared_archive_transport(tmp_path):
    profile = model_profile(tmp_path)
    profile['kvmem'].update(enabled())
    original = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'hash')
    for payload in (launch['profile']['kvmem'], worker_payload(launch)):
        assert runtime(payload).cpu_seal_hash_workers == 4
    assert profile == original
    restored = build_launch(json.loads(json.dumps(launch['profile'])), tmp_path, tmp_path/'reload')
    assert worker_payload(restored) == worker_payload(launch)
    options = runtime(enabled())
    with pytest.raises(ValueError, match='requires archive_backend=cpu and async_seal=true'):
        replace(options, async_seal=False)
    archive = make_shared_archive(options, tmp_path/'unused', 4, 8)
    assert archive.cpu_seal_hash_workers == 4 and archive.close()


@pytest.mark.parametrize('workers', [None, 1, 4, 8])
def test_elastic_roundtrip_preserves_explicit_hash_workers(tmp_path, workers):
    profile = elastic_profile(tmp_path)
    profile.update(archive_backend='cpu', async_seal=True)
    if workers is not None:
        profile['cpu_seal_hash_workers'] = workers
    original = deepcopy(profile)
    config, _ = elastic_launch(profile, tmp_path, tmp_path/'elastic')
    backend = config['mean_k_backend_profile']
    native = worker_payload(build_launch(backend, tmp_path, tmp_path/'native'))
    for payload in (config, backend['kvmem'], native):
        assert payload['cpu_seal_hash_workers'] == (workers or 1)
    assert profile == original


def test_elastic_full_context_rejects_ignored_hash_optimization(tmp_path):
    profile = dict(model='unused', tokenizer='unused', users_file='unused', context_mode='full',
                   archive_backend='cpu', async_seal=True, cpu_seal_hash_workers=4)
    with pytest.raises(ValueError, match='requires MeanK context management'):
        elastic_launch(profile, tmp_path, tmp_path/'invalid')
