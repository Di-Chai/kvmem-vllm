"""Worker page-table publication waits for checksum and staging admission."""
import os
from threading import Event
import time

import pytest
import torch

from vllm_kvmem.request_checkpoint import CacheRows, capture_exact, restore_exact
from vllm_kvmem.request_swap_worker import RequestSwapWorker
from test_request_swap_worker import fixture, command, usage, Adapter


def eventually(call):
    deadline = time.monotonic()+5
    while True:
        value = call()
        if value:
            return value
        assert time.monotonic() < deadline
        time.sleep(.001)


class ExactAdapter(Adapter):
    def __init__(self, calls):
        super().__init__(calls)
        self.rows = torch.arange(24, dtype=torch.uint8).reshape(2, 12)
        self.restored = torch.zeros_like(self.rows)

    def capture(self, request_id, operation_id, block_ids, max_bytes):
        self.calls.append(('capture', request_id))
        index, generation = (0, 2) if request_id == 'a' else (1, 3)
        result = capture_exact((request_id, generation, operation_id),
            [CacheRows('raw', self.rows, (index,))], max_bytes=max_bytes)
        result.completion.query = self.capture_event.query
        return result

    def restore(self, snapshot, request_id, operation_id, block_ids):
        self.calls.append(('restore', request_id))
        index, generation = (0, 2) if request_id == 'a' else (1, 3)
        restore_exact(snapshot, (request_id, generation, operation_id),
                      [CacheRows('raw', self.restored, (index,))])
        return self.restore_event


def make(tmp_path):
    original, _, state, calls = fixture()
    adapter = ExactAdapter(calls)
    worker = RequestSwapWorker(original.runner, 32, adapter=adapter,
        checkpoint_ssd_bytes=128, checkpoint_directory=tmp_path)
    return worker, adapter, state, calls


def paused_pair(worker, adapter):
    adapter.capture_event.done = True
    first = worker.execute(command())
    assert first.query()
    second = worker.execute(command(request_id='b', history_end=128, command_id=20, operation_id=2))
    eventually(second.query)  # CPU promise needs 32 bytes; spills a's 12-byte owner.
    assert worker.checkpoints['a'].stored.state == 'disk'
    return first, second


def drop_all(worker):
    for rid, held in list(worker.checkpoints.items()):
        job = worker.execute(command('drop', request_id=rid,
            operation_id=held.command.operation_id, history_end=held.command.history_end,
            command_id=held.command.command_id+100))
        eventually(job.query)
    assert worker.close()


def test_more_paused_logical_bytes_than_cpu_staging_and_exact_restore(tmp_path):
    worker, adapter, _, calls = make(tmp_path)
    paused_pair(worker, adapter)
    assert usage(worker)['live'] == 24
    metrics = worker.snapshot()['storage']
    assert metrics['cpu_resident_bytes'] == 12 and metrics['ssd_used_bytes'] == 12
    calls.clear()
    restored = worker.execute(command('restore'))
    assert calls == []  # Neither table nor GPU destination changes before read.
    eventually(lambda: (restored.query(), any(c[0] == 'restore' for c in calls))[1])
    assert torch.equal(adapter.rows[0], adapter.restored[0])
    assert worker.snapshot()['storage']['cpu_resident_bytes'] == 24
    assert not restored.query()  # H2D owner must survive the delayed event.
    adapter.restore_event.done = True
    assert restored.query()
    assert worker.snapshot()['storage']['cpu_resident_bytes'] == 12
    assert worker.checkpoints['a'].stored.disk is not None
    drop_all(worker)
    assert usage(worker)['live'] == 0
    assert worker.snapshot()['storage']['cpu_peak_bytes'] <= 32


def test_ssd_read_failure_cannot_publish_new_tables_and_can_retry_same_operation(tmp_path, monkeypatch):
    worker, adapter, _, calls = make(tmp_path)
    paused_pair(worker, adapter)
    calls.clear()
    original = os.preadv
    def fail(*args): raise OSError('injected disk read failure')
    monkeypatch.setattr(os, 'preadv', fail)
    restored = worker.execute(command('restore'))
    with pytest.raises(OSError, match='injected'): eventually(restored.query)
    assert calls == []
    assert worker.checkpoints['a'].stored.state == 'disk'
    assert usage(worker)['live'] == 24
    monkeypatch.setattr(os, 'preadv', original)
    adapter.restore_event.done = True
    eventually(restored.query)
    assert torch.equal(adapter.rows[0], adapter.restored[0])
    drop_all(worker)


def test_cancel_waits_for_spill_and_never_releases_logical_or_cpu_credit_early(tmp_path, monkeypatch):
    worker, adapter, _, calls = make(tmp_path)
    adapter.capture_event.done = True
    assert worker.execute(command()).query()
    entered, proceed = Event(), Event()
    original = os.pwrite
    def blocked(*args):
        entered.set()
        assert proceed.wait(5)
        return original(*args)
    monkeypatch.setattr(os, 'pwrite', blocked)
    peer = worker.execute(command(request_id='b', history_end=128, command_id=20, operation_id=2))
    assert not peer.query() and entered.wait(5)
    dropping = worker.execute(command('drop'))
    try:
        assert not dropping.query()
        assert usage(worker)['live'] == 12 and usage(worker)['promised'] == 32
        assert worker.snapshot()['storage']['cpu_resident_bytes'] == 12
        assert not any(c[0] == 'release' for c in calls)
    finally:
        proceed.set()
    eventually(dropping.query)
    eventually(peer.query)
    drop_all(worker)


def test_capture_over_cpu_cap_rejected_before_preparing_logical_credit(tmp_path):
    worker, adapter, _, calls = make(tmp_path)
    with pytest.raises(BufferError, match='CPU staging'):
        worker.execute(command(checkpoint_bytes=33))
    assert usage(worker)['live'] == usage(worker)['promised'] == 0
    assert calls == [] and worker.close()
