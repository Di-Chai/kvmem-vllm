"""Physical SSD retirement completes without unrelated new reads or writes."""
import json
import os
from threading import Event
from types import SimpleNamespace as NS

import pytest

from test_tiered_storage import RECORD, make_archive, records, reclaim
from vllm_kvmem import _native as n


def spilled(path, count=1):
    archive = make_archive(path, capacity=8, resident=1, max_pending=4)
    repo, leases, payloads = records(count)
    archive.write('a', leases, b''.join(payloads)).result(5)
    archive.spill(0).result(5)
    return archive, repo, leases


def hold_close(monkeypatch, archive):
    entered, proceed = Event(), Event()
    original = os.close
    fds = {segment.fd for segment in archive._tier.disk.segments.values()}
    def close(fd):
        if fd in fds:
            entered.set()
            assert proceed.wait(5)
        return original(fd)
    monkeypatch.setattr(os, 'close', close)
    return entered, proceed


def test_idle_last_release_reaps_without_new_archive_io_and_holds_physical_charge(tmp_path, monkeypatch):
    archive, repo, leases = spilled(tmp_path)
    entered, proceed = hold_close(monkeypatch, archive)
    try:
        assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
        assert entered.wait(5)  # Callback returned while the close remains blocked.
        state = archive.io_usage()
        assert state['logical_allocated_bytes'] == state['cpu_resident_bytes'] == 0
        assert state['ssd_retired_pending_bytes'] == state['ssd_used_bytes'] == RECORD
        assert state['ssd_file_bytes'] == state['ssd_closing_bytes'] == RECORD
        assert not archive.reclamation_ready() and not archive.close()
        proceed.set()
        assert archive.wait_for_idle(5) and archive.reclamation_ready()
        state = archive.io_usage()
        assert state['ssd_retired_pending_bytes'] == state['ssd_used_bytes'] == state['ssd_file_bytes'] == 0
        assert not state['ssd_reap_pending']
    finally:
        proceed.set()
        assert archive.wait_for_idle(5) and archive.close()


def test_burst_retirements_share_one_bounded_reaper(tmp_path, monkeypatch):
    archive, repo, leases = spilled(tmp_path, 5)
    entered, proceed = hold_close(monkeypatch, archive)
    try:
        # The last segment has exactly one record, so this first retirement
        # deterministically blocks before any later callbacks can arrive.
        assert reclaim(repo, leases[-1], archive) == n.ReclaimResult.RECLAIMED
        assert entered.wait(5)
        for lease in leases[:-1]:
            assert reclaim(repo, lease, archive) == n.ReclaimResult.RECLAIMED
        assert archive._pending == 1
        assert archive.io_usage()['ssd_reap_submissions'] == 1
        proceed.set()
        assert archive.wait_for_idle(5)
        assert archive.io_usage()['ssd_used_bytes'] == 0
        assert archive.io_usage()['ssd_reap_submissions'] == 1
    finally:
        proceed.set()
        assert archive.wait_for_idle(5) and archive.close()


def test_release_between_empty_check_and_worker_completion_resubmits_once(tmp_path, monkeypatch):
    archive, repo, leases = spilled(tmp_path, 2)
    empty, proceed = Event(), Event()
    original = archive._tier._drain_disk_releases
    calls = []
    def drain():
        original()
        calls.append(1)
        if len(calls) == 1:
            empty.set()
            assert proceed.wait(5)
    monkeypatch.setattr(archive._tier, '_drain_disk_releases', drain)
    try:
        assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
        assert empty.wait(5)
        assert reclaim(repo, leases[1], archive) == n.ReclaimResult.RECLAIMED
        assert archive._pending == 1
        proceed.set()
        assert archive.wait_for_idle(5)
        assert archive.reclamation_ready()
        assert archive.io_usage()['ssd_reap_submissions'] == 2
        assert archive.io_usage()['ssd_used_bytes'] == 0
    finally:
        proceed.set()
        assert archive.wait_for_idle(5) and archive.close()


def test_reap_submission_failure_keeps_exact_extent_and_pending_balance(tmp_path, monkeypatch):
    archive, repo, leases = spilled(tmp_path)
    original = archive.executor.submit
    def fail_after_queue(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('injected post-enqueue failure')
    monkeypatch.setattr(archive.executor, 'submit', fail_after_queue)
    handle = leases[0].handle
    assert repo.retire(handle)
    leases[0].reset()
    try:
        with pytest.raises(RuntimeError, match='post-enqueue'):
            repo.reclaim(handle, lambda rec: archive.reclaim('a', rec))
        assert archive._pending == 0 and not archive._disk_reap_pending
        assert archive.io_usage()['ssd_retired_pending_bytes'] == RECORD
        assert archive.io_usage()['ssd_used_bytes'] == RECORD
        monkeypatch.setattr(archive.executor, 'submit', original)
        assert repo.reclaim(handle, lambda rec: archive.reclaim('a', rec)) == n.ReclaimResult.RECLAIMED
        assert archive.wait_for_idle(5) and archive.reclamation_ready()
        assert archive.io_usage()['ssd_used_bytes'] == 0
    finally:
        monkeypatch.setattr(archive.executor, 'submit', original)
        assert archive.wait_for_idle(5) and archive.close()


@pytest.mark.parametrize('explicit_close', [False, True])
def test_failed_close_retains_descriptor_charge_and_cannot_acknowledge_success(tmp_path, monkeypatch, explicit_close):
    archive, repo, leases = spilled(tmp_path)
    original = os.close
    fd = next(iter(archive._tier.disk.segments.values())).fd
    def fail(selected):
        if selected == fd:
            raise OSError('injected before close syscall')
        return original(selected)
    monkeypatch.setattr(os, 'close', fail)
    try:
        if explicit_close:
            with pytest.raises(OSError, match='before close'):
                archive.close()
        else:
            assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
            with pytest.raises(OSError, match='before close'):
                archive._disk_reap_job.result(5)
            assert archive.io_usage()['ssd_retired_pending_bytes'] == RECORD
        assert archive.io_usage()['ssd_used_bytes'] == archive.io_usage()['ssd_file_bytes'] == RECORD
        assert 'before close' in archive.io_usage()['ssd_reap_error']
        assert os.fstat(fd).st_size == RECORD
        with pytest.raises(RuntimeError, match='physical reclamation'):
            archive.reclamation_ready()
        with pytest.raises(RuntimeError, match='physical reclamation'):
            archive.close()
    finally:
        # The injected failure precedes the syscall, so this test knows the fd
        # was never closed. Production deliberately does not retry ambiguous close errors.
        monkeypatch.setattr(os, 'close', original)
        archive._tier.disk._close_error = None
        archive._tier._disk_release_error = None
        if not explicit_close:
            assert archive.wait_for_idle(5)
        assert archive.close()


def test_explicit_close_keeps_last_file_charged_until_close_returns(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    archive, repo, leases = spilled(tmp_path)
    entered, proceed = hold_close(monkeypatch, archive)
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(archive.close)
        try:
            assert entered.wait(5)
            assert archive.io_usage()['ssd_used_bytes'] == RECORD
            assert archive.io_usage()['ssd_file_bytes'] == RECORD
            assert not archive._tier.closed
        finally:
            proceed.set()
        assert job.result(5)
    assert archive.io_usage()['ssd_used_bytes'] == 0


@pytest.mark.parametrize('asynchronous', [False, True])
def test_terminal_ack_waits_for_reap_and_forces_final_zero_report(tmp_path, monkeypatch, asynchronous):
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    from vllm_kvmem.core.archive_telemetry import ArchiveResourceReporter
    from test_connector import pair, send, poll
    archive, repo, leases = spilled(tmp_path)
    entered, proceed = hold_close(monkeypatch, archive)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    polls = []
    runtime.archive = archive
    runtime.runner, runtime.authorities, runtime.completed = NS(), {}, []
    runtime.publication_ready = lambda _: True
    runtime.retiring = {'a': {}} if asynchronous else {}
    runtime.retirement = NS(poll=lambda rid: polls.append(rid) or {'reclaimed': True})
    reporter = ArchiveResourceReporter(tmp_path, archive, memory_sampler=lambda: {})
    runtime._write_archive_report = lambda: reporter.publish(0, 0, force=True)
    scheduler, connector = pair()
    scheduler.physical_retirement = connector.physical_retirement = True
    connector.retirement_done = runtime.retirement_done
    try:
        assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
        assert entered.wait(5)
        assert scheduler.request_finished_all_groups(NS(request_id='a'), ()) == (True, None)
        send(scheduler, connector)
        assert poll(scheduler, connector).finished_sending is None
        assert poll(scheduler, connector).finished_sending is None
        assert scheduler.has_pending_push_work()
        assert len(polls) == int(asynchronous)  # A completed result survives the physical fence.
        assert not (tmp_path/'archive_resources.json').exists()
        proceed.set(); assert archive.wait_for_idle(5)
        assert poll(scheduler, connector).finished_sending == {'a'}
        report = json.loads((tmp_path/'archive_resources.json').read_text())
        assert report['ssd_used_bytes'] == report['ssd_file_bytes'] == report['ssd_retired_pending_bytes'] == 0
        assert not scheduler.has_pending_push_work()
    finally:
        proceed.set()
        assert archive.wait_for_idle(5) and archive.close()


def test_session_eviction_keeps_namespace_until_reap_and_final_report(tmp_path, monkeypatch):
    from vllm_kvmem.serving.session_worker import SessionWorker
    from vllm_kvmem.serving.session_control import SessionEviction
    from vllm_kvmem.core.archive_telemetry import ArchiveResourceReporter
    archive, repo, leases = spilled(tmp_path)
    entered, proceed = hold_close(monkeypatch, archive)
    reporter = ArchiveResourceReporter(tmp_path, archive, memory_sampler=lambda: {})
    worker = SessionWorker.__new__(SessionWorker)
    namespace = ('salt', 'session')
    worker.runner = NS(_kvmem_capture_runtime=NS(archive=archive,
        _write_archive_report=lambda: reporter.publish(0, 0, force=True)))
    worker.pool = NS(entries={})  # Logical eviction already released the last repository record.
    worker.evictions = {namespace: SessionEviction(namespace, 1)}
    worker.identities, worker._history_capacity_sent, worker.completed = {namespace: 1}, {}, []
    try:
        assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
        assert entered.wait(5)
        worker._poll_evictions()
        assert worker.evictions and worker.identities and not worker.completed
        assert not (tmp_path/'archive_resources.json').exists()
        proceed.set(); assert archive.wait_for_idle(5)
        worker._poll_evictions()
        assert not worker.evictions and not worker.identities
        assert worker.completed[0].kind == 'evicted'
        report = json.loads((tmp_path/'archive_resources.json').read_text())
        assert report['ssd_file_bytes'] == report['ssd_retired_pending_bytes'] == 0
    finally:
        proceed.set()
        assert archive.wait_for_idle(5) and archive.close()


def test_idle_session_only_shutdown_waits_for_final_physical_release(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    from vllm_kvmem.serving.session_worker import SessionWorker
    archive, repo, leases = spilled(tmp_path)
    entered, proceed = hold_close(monkeypatch, archive)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.archive, runtime.options = archive, NS(async_archive_retirement=True)
    runtime.authorities, runtime.retiring = {}, {}
    runtime.join_cpu_scatter = lambda: None
    runtime.drain = lambda **kwargs: None
    runtime.finish = lambda ids: None
    runtime.retirement = NS(close=lambda: True)
    runtime.runner = NS(_kvmem_capture_runtime=runtime)
    worker = SessionWorker(runtime.runner, NS(session_capacity=1, session_checkpoint_bytes=4096))
    runtime.runner._kvmem_sessions = worker
    # The pool already detached its last idle owner; only its physical history
    # remains. No active/retiring request loop can accidentally provide a wait.
    worker.identities[('salt', 'session')] = 1
    assert reclaim(repo, leases[0], archive) == n.ReclaimResult.RECLAIMED
    assert entered.wait(5)
    assert not worker.close() and worker.identities
    with ThreadPoolExecutor(max_workers=1) as executor:
        job = executor.submit(runtime.shutdown_retirement)
        try:
            assert not job.done() and not archive._closed
        finally:
            proceed.set()
        job.result(5)
    assert not worker.identities
    assert archive.io_usage()['ssd_used_bytes'] == archive.io_usage()['ssd_file_bytes'] == 0


def test_write_rollback_close_error_retains_failed_extent_and_global_fence(tmp_path, monkeypatch):
    archive = make_archive(tmp_path, capacity=8, resident=1, max_pending=4)
    original_close = os.close
    def fail_write(*args):
        raise OSError('injected SSD write failure')
    def fail_close(fd):
        raise OSError('injected rollback close failure')
    monkeypatch.setattr(os, 'pwrite', fail_write)
    monkeypatch.setattr(os, 'close', fail_close)
    try:
        with pytest.raises(OSError, match='rollback close'):
            archive._tier.disk.write(bytes(RECORD))
        assert archive.io_usage()['ssd_used_bytes'] == RECORD
        assert archive.io_usage()['ssd_segments'] == 1
        assert 'rollback close' in archive.io_usage()['ssd_reap_error']
        with pytest.raises(RuntimeError, match='physical reclamation'):
            archive.reclamation_ready()
        with pytest.raises(RuntimeError, match='physical reclamation'):
            archive.spill()
        with pytest.raises(RuntimeError, match='physical reclamation'):
            archive.close()
    finally:
        monkeypatch.setattr(os, 'close', original_close)
        archive._tier.disk._close_error = None  # Injection did not invoke the syscall.
        assert archive.close()


def test_tiered_terminal_fence_enabled_without_async_retirement():
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm_kvmem.serving.connector import KVMemConnector
    config = NS(kv_transfer_config=KVTransferConfig(kv_connector='KVMemConnector', kv_role='kv_both',
        kv_connector_extra_config={'kvmem': {'archive_backend': 'tiered'}}),
        parallel_config=NS(tensor_parallel_size=1))
    connector = KVMemConnector(config, KVConnectorRole.SCHEDULER, None)
    assert connector.physical_retirement and not connector.async_retirement
    assert connector.request_finished_all_groups(NS(request_id='a'), ()) == (True, None)
    assert connector.has_pending_push_work()
