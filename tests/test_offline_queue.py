"""Offline admission must refill around slow jobs without releasing live KV."""
from threading import Event, Thread

import pytest

from vllm_kvmem.offline_queue import OfflineQueue, OfflineTask


def tasks(n, amount=40):
    return [OfflineTask(str(i), i, amount) for i in range(n)]


def test_refills_one_finished_slot_before_the_other_request_finishes():
    third_started = Event()
    states = []
    queue = OfflineQueue(tasks(3), max_inflight=2, history_capacity_bytes=80)
    def work(task):
        if task.payload == 1:
            assert third_started.wait(5), 'A slow peer blocked replacement admission'
        elif task.payload == 2:
            third_started.set()
        return task.id
    result = queue.run(work, lambda task, output: output, on_state=states.append)
    assert set(result) == {'0', '1', '2'}
    replacement = next(s for s in states if s['tasks'][2]['state'] == 'running')
    assert replacement['tasks'][1]['state'] == 'running'
    assert queue.inflight == queue.reserved_bytes == 0
    assert queue.peak_inflight == 2 and queue.peak_bytes == 80


def test_one_slow_retirement_does_not_block_other_retirements_or_refill():
    third_started = Event()
    queue = OfflineQueue(tasks(3), max_inflight=2, history_capacity_bytes=80,
                         retirement_workers=2)
    def work(task):
        if task.payload == 2:
            third_started.set()
        return task.id
    def finish(task, output):
        if task.payload == 0:
            assert third_started.wait(5), 'Retirement accidentally became a batch fence'
        return output
    assert set(queue.run(work, finish)) == {'0', '1', '2'}


def test_cpu_limit_holds_complete_history_through_retirement_and_preserves_fifo():
    queue = OfflineQueue(tasks(5, 60), max_inflight=8, history_capacity_bytes=100)
    starts, fences = [], []
    def work(task):
        assert len(starts) == len(fences)
        starts.append(task.id)
        return task.id
    def finish(task, output):
        assert queue.reserved_bytes == 60 and queue.inflight == 1
        fences.append(task.id)
        return output
    assert queue.run(work, finish) == starts == fences == ['0', '1', '2', '3', '4']
    assert queue.peak_inflight == 1 and queue.peak_bytes == 60


def test_failure_is_published_without_waiting_for_blocked_peer_or_refilling():
    peer_started, release_peer, failure_seen = Event(), Event(), Event()
    queue = OfflineQueue(tasks(3), max_inflight=2, history_capacity_bytes=80)
    errors = []
    def work(task):
        if task.payload == 0:
            assert peer_started.wait(5)
            raise RuntimeError('model failure')
        assert task.payload == 1
        peer_started.set()
        assert release_peer.wait(5)
        return task.id
    def run():
        try:
            queue.run(work, lambda task, output: output,
                      on_failure=lambda error: failure_seen.set())
        except RuntimeError as error:
            errors.append(str(error))
    thread = Thread(target=run)
    thread.start()
    try:
        assert failure_seen.wait(5)
        thread.join(1)
        assert not thread.is_alive() and errors == ['model failure']
        assert queue.states['2']['state'] == 'cancelled'
        assert queue.reserved_bytes == 80  # Failure is not a resource-release fence.
    finally:
        release_peer.set()
        thread.join(5)


def test_failed_retirement_does_not_return_memory_or_start_next_job():
    queue = OfflineQueue(tasks(2), max_inflight=1, history_capacity_bytes=40)
    failures = []
    def fail(task, output):
        raise RuntimeError('retirement fence failed')
    with pytest.raises(RuntimeError, match='retirement fence'):
        queue.run(lambda task: task.id, fail, on_failure=failures.append)
    assert len(failures) == 1
    assert queue.reserved_bytes == 40 and queue.inflight == 1
    assert queue.states['1']['state'] == 'cancelled'


@pytest.mark.parametrize('items,limit', [
    (tasks(1, 101), 100), (tasks(1, 0), 100),
    ([OfflineTask('same', 0, 1), OfflineTask('same', 1, 1)], 100),
    ([OfflineTask('', 0, 1)], 100), (tasks(1), True)])
def test_invalid_or_oversize_jobs_fail_before_submission(items, limit):
    with pytest.raises(ValueError):
        OfflineQueue(items, max_inflight=8, history_capacity_bytes=limit)


def test_finished_attempt_cannot_be_reused_or_implicitly_retried():
    queue = OfflineQueue([], max_inflight=1, history_capacity_bytes=40)
    assert queue.run(lambda task: None, lambda task, result: None) == []
    with pytest.raises(RuntimeError, match='cannot be reused'):
        queue.run(lambda task: None, lambda task, result: None)
