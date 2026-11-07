"""Real lazy checkpoint files, bounded CPU owners and failure/cancel fences."""
from concurrent.futures import Future, ThreadPoolExecutor
import errno
import os
from threading import Event
import time
import weakref

import pytest
import torch

from vllm_kvmem.checkpoint_storage import CheckpointStore
from vllm_kvmem.request_checkpoint import CacheRows, capture_exact, restore_exact


def eventually(call):
    deadline = time.monotonic()+5
    while True:
        value = call()
        if value:
            return value
        assert time.monotonic() < deadline
        time.sleep(.001)


def install(store, name='a', size=16):
    identity = name, 1, 1
    entry = eventually(lambda: store.reserve_capture(identity, size))
    data = torch.arange(size, dtype=torch.uint8).reshape(1, size)
    snapshot = capture_exact(identity, [CacheRows('raw', data, (0,))], max_bytes=size)
    store.install(entry, snapshot)
    assert store.capture_ready(entry)
    return entry


def cleanup(store):
    for entry in list(store.entries.values()):
        if entry.state == 'restoring':
            store.restore_complete(entry)
        eventually(lambda: store.drop(entry))
    assert store.close()


def test_lazy_spill_exact_reload_and_reclaim_without_whole_disk_preallocation(tmp_path):
    directory = tmp_path/'lazy'
    store = CheckpointStore(directory, 16, 1024**3)
    assert not directory.exists() and store.snapshot()['cpu_charged_bytes'] == 0
    entry = install(store)
    eventually(store.poll)
    assert entry.state == 'disk' and entry.snapshot is None
    assert store.snapshot()['cpu_charged_bytes'] == 0
    assert store.snapshot()['ssd_used_bytes'] == 16
    assert list(directory.iterdir()) == []  # Unlinked files remain process-local.
    fd = entry.disk.fd
    snapshot = eventually(lambda: store.acquire_restore(entry))
    target = torch.zeros((2, 16), dtype=torch.uint8)
    assert restore_exact(snapshot, entry.identity, [CacheRows('raw', target, (1,))]).query()
    assert torch.equal(target[1], torch.arange(16, dtype=torch.uint8))
    assert target[0].sum().item() == 0
    snapshot = None
    store.restore_complete(entry)
    assert entry.disk is not None and entry.snapshot is None
    assert store.snapshot()['cpu_peak_bytes'] == 16
    assert store.drop(entry) and store.close()
    with pytest.raises(OSError): os.fstat(fd)


def test_cpu_credit_and_cancel_wait_for_write_completion(tmp_path, monkeypatch):
    store = CheckpointStore(tmp_path, 16, 64)
    entered, proceed = Event(), Event()
    original = os.pwrite
    def blocked(*args):
        entered.set()
        assert proceed.wait(5)
        return original(*args)
    monkeypatch.setattr(os, 'pwrite', blocked)
    entry = install(store)
    try:
        assert entered.wait(5)
        assert store.snapshot()['cpu_resident_bytes'] == 16
        assert store.reserve_capture(('b', 1, 1), 16) is None
        assert not store.drop(entry) and not store.close()
        assert entry.snapshot is not None
    finally:
        proceed.set()
    eventually(lambda: store.drop(entry))
    assert store.snapshot()['cpu_charged_bytes'] == store.snapshot()['ssd_used_bytes'] == 0
    assert store.close()


def test_capture_event_precedes_ssd_io_and_cancellation(tmp_path):
    store = CheckpointStore(tmp_path/'no-file-yet', 16, 64)
    entry = store.reserve_capture(('a', 1, 1), 16)
    tensor = torch.arange(16, dtype=torch.uint8).reshape(1, 16)
    snapshot = capture_exact(entry.identity, [CacheRows('raw', tensor, (0,))], max_bytes=16)
    done = Event()
    snapshot.completion.query = done.is_set
    store.install(entry, snapshot)
    assert not store.capture_ready(entry) and not store.drop(entry)
    assert not store.directory.exists()
    done.set()
    assert store.drop(entry) and store.close()


def test_short_writes_complete_before_checksum_verified_restore(tmp_path, monkeypatch):
    original = os.pwrite
    monkeypatch.setattr(os, 'pwrite', lambda fd, view, offset: original(fd, view[:3], offset))
    store = CheckpointStore(tmp_path, 16, 64)
    entry = install(store)
    eventually(store.poll)
    snapshot = eventually(lambda: store.acquire_restore(entry))
    assert snapshot.payload.tolist() == list(range(16))
    snapshot = None
    cleanup(store)


@pytest.mark.parametrize('failure', ['error', 'zero'])
def test_failed_write_keeps_cpu_source_and_charge(tmp_path, monkeypatch, failure):
    def fail(*args):
        if failure == 'error': raise OSError('injected full device')
        return 0
    monkeypatch.setattr(os, 'pwrite', fail)
    store = CheckpointStore(tmp_path, 16, 64)
    entry = install(store)
    with pytest.raises(OSError): eventually(store.poll)
    assert entry.state == 'resident' and entry.disk is None
    assert entry.snapshot.payload.tolist() == list(range(16))
    assert store.snapshot()['cpu_charged_bytes'] == 16
    assert store.snapshot()['io_failures'] == 1
    assert store.drop(entry) and store.close()


@pytest.mark.parametrize('failure', ['corrupt', 'truncated', 'read_error'])
def test_failed_read_retains_ssd_authority_and_releases_staging(tmp_path, monkeypatch, failure):
    store = CheckpointStore(tmp_path, 16, 64)
    entry = install(store)
    eventually(store.poll)
    fd = entry.disk.fd
    if failure == 'corrupt': os.pwrite(fd, b'X', 0)
    elif failure == 'truncated': os.ftruncate(fd, 3)
    else:
        def fail(*args): raise OSError('injected read failure')
        monkeypatch.setattr(os, 'preadv', fail)
    with pytest.raises(OSError): eventually(lambda: store.acquire_restore(entry))
    assert entry.state == 'disk' and entry.disk.fd == fd and entry.snapshot is None
    assert store.snapshot()['cpu_charged_bytes'] == 0
    assert store.snapshot()['ssd_used_bytes'] == 16
    assert store.drop(entry) and store.close()


def test_restore_lease_prevents_reclaim_and_other_allocations_until_gpu_fence(tmp_path):
    store = CheckpointStore(tmp_path, 16, 64)
    entry = install(store)
    eventually(store.poll)
    snapshot = eventually(lambda: store.acquire_restore(entry))
    with pytest.raises(ValueError, match='in-flight'): store.drop(entry)
    assert store.reserve_capture(('b', 1, 1), 16) is None
    snapshot = None
    store.restore_complete(entry)
    peer = store.reserve_capture(('b', 1, 1), 16)
    assert peer is not None
    store.abort_capture(peer)
    cleanup(store)


def test_concurrent_checkpoint_owners_share_one_bounded_staging_pool(tmp_path):
    store = CheckpointStore(tmp_path, 32, 512)
    def owner(index):
        entry = install(store, str(index), 16)
        snapshot = eventually(lambda: store.acquire_restore(entry))
        assert snapshot.payload.tolist() == list(range(16))
        snapshot = None
        store.restore_complete(entry)
        eventually(lambda: store.drop(entry))
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(owner, range(24)))
    stats = store.snapshot()
    assert stats['cpu_peak_bytes'] <= 32 and stats['cpu_charge_peak_bytes'] <= 32
    assert stats['logical_bytes'] == stats['ssd_used_bytes'] == stats['cpu_charged_bytes'] == 0
    assert store.close()


def test_identity_size_and_quota_are_checked_before_allocating(tmp_path):
    store = CheckpointStore(tmp_path, 16, 16)
    with pytest.raises(BufferError): store.reserve_capture(('a', 1, 1), 17)
    entry = store.reserve_capture(('a', 1, 1), 16)
    assert store.snapshot()['cpu_promised_bytes'] == 16
    with pytest.raises(ValueError): store.reserve_capture(('a', 1, 1), 16)
    with pytest.raises(BufferError): store.reserve_capture(('b', 1, 1), 1)
    store.abort_capture(entry)
    with pytest.raises(ValueError): store.drop(entry)
    assert not tmp_path.joinpath('checkpoint').exists()
    assert store.close()


def test_completed_executor_work_item_cannot_retain_uncharged_cpu_tensor(tmp_path):
    class RetainingExecutor:
        """Model Future.done before the worker drops WorkItem.fn/args."""
        def __init__(self): self.completed_work_items = []
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            self.completed_work_items.append((function, args))
            return future
        def shutdown(self, wait): pass
    store = CheckpointStore(tmp_path, 16, 64)
    store._executor.shutdown(wait=True)
    executor = store._executor = RetainingExecutor()
    entry = install(store)
    source = weakref.ref(entry.snapshot.payload)
    assert store.poll() and store.snapshot()['cpu_charged_bytes'] == 0
    assert source() is None  # Even deliberately retained WorkItem.args own no tensor.
    assert store.acquire_restore(entry) is None
    staging = weakref.ref(entry.loading)
    restored = store.acquire_restore(entry)
    assert restored is not None and staging() is restored.payload
    restored = None
    store.restore_complete(entry)
    assert store.snapshot()['cpu_charged_bytes'] == 0 and staging() is None
    assert len(executor.completed_work_items) == 2
    assert store.drop(entry) and store.close()


@pytest.mark.parametrize('closed_before_error', [False, True])
def test_failed_close_never_retries_uncertain_descriptor_or_releases_credit(
        tmp_path, monkeypatch, closed_before_error):
    store = CheckpointStore(tmp_path, 16, 64)
    entry = install(store)
    eventually(store.poll)
    fd = entry.disk.fd
    original_close = os.close
    calls = []

    def fail_close(selected):
        if selected == fd:
            calls.append(selected)
            if closed_before_error:
                original_close(selected)
            raise OSError(errno.EIO, 'injected ambiguous close error')
        return original_close(selected)

    monkeypatch.setattr(os, 'close', fail_close)
    unrelated = None
    try:
        with pytest.raises(OSError, match='ambiguous close'):
            store.drop(entry)
        monkeypatch.setattr(os, 'close', original_close)
        unrelated = os.open(tmp_path/'unrelated', os.O_CREAT | os.O_RDWR, 0o600)
        if closed_before_error:
            assert unrelated == fd  # Reproduce the dangerous descriptor reuse.
        state = store.snapshot()
        assert state['entries'] == 1
        assert state['logical_bytes'] == state['ssd_used_bytes'] == 16
        assert state['ssd_close_uncertain_bytes'] == 16
        assert 'ambiguous close' in state['close_error']
        for operation in (lambda: store.drop(entry), store.close, store.poll,
                          lambda: store.acquire_restore(entry),
                          lambda: store.reserve_capture(('b', 1, 1), 16)):
            with pytest.raises(RuntimeError, match='Checkpoint SSD close failed'):
                operation()
        assert calls == [fd]
        os.fstat(unrelated)  # It must remain open after every failed retry.
        assert store.snapshot()['logical_bytes'] == store.snapshot()['ssd_used_bytes'] == 16
        assert not store.closed
    finally:
        monkeypatch.setattr(os, 'close', original_close)
        if unrelated is not None:
            original_close(unrelated)
        if not closed_before_error:
            # This injection is known to precede the syscall. Production does
            # not have that knowledge and must never retry the descriptor.
            original_close(fd)
        store._executor.shutdown(wait=True)


@pytest.mark.parametrize('closed_before_error', [False, True])
def test_failed_write_rollback_close_retains_owner_and_blocks_successful_cleanup(
        tmp_path, monkeypatch, closed_before_error):
    store = CheckpointStore(tmp_path, 16, 64)
    original_open, original_close, original_write = os.open, os.close, os.pwrite
    opened, closes = [], []

    def tracked_open(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def partial_then_fail(fd, view, offset):
        if offset:
            raise OSError(errno.ENOSPC, 'injected write failure')
        return original_write(fd, view[:3], offset)

    def failed_rollback(fd):
        if fd in opened:
            closes.append(fd)
            if closed_before_error:
                original_close(fd)
            raise OSError(errno.EIO, 'injected rollback close error')
        return original_close(fd)

    monkeypatch.setattr(os, 'open', tracked_open)
    monkeypatch.setattr(os, 'pwrite', partial_then_fail)
    monkeypatch.setattr(os, 'close', failed_rollback)
    unrelated = None
    try:
        entry = install(store)
        eventually(lambda: store._job[2].done())
        monkeypatch.setattr(os, 'open', original_open)
        monkeypatch.setattr(os, 'close', original_close)
        fd = opened[0]
        unrelated = original_open(tmp_path/'unrelated-rollback', os.O_CREAT | os.O_RDWR, 0o600)
        if closed_before_error:
            assert unrelated == fd
        else:
            assert os.fstat(fd).st_size == 3
        state = store.snapshot()
        assert state['logical_bytes'] == state['cpu_charged_bytes'] == 16
        assert state['ssd_close_uncertain_bytes'] == 16
        assert entry.snapshot is not None  # The complete source is retained.
        for operation in (store.poll, lambda: store.drop(entry), store.close,
                          lambda: store.acquire_restore(entry)):
            with pytest.raises(RuntimeError, match='Checkpoint SSD close failed'):
                operation()
        assert closes == [fd] and not store.closed
        os.fstat(unrelated)
    finally:
        monkeypatch.setattr(os, 'open', original_open)
        monkeypatch.setattr(os, 'close', original_close)
        if unrelated is not None:
            original_close(unrelated)
        if opened and not closed_before_error:
            original_close(opened[0])
        store._executor.shutdown(wait=True)
