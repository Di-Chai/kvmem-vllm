"""KVMEM_RESTORE_FIRST: queued urgent work runs before queued normal work, FIFO within a class,
never preempting the running task; futures, exceptions and shutdown behave like ThreadPoolExecutor."""
import threading
import time

import pytest

from vllm_kvmem.priority_executor import PriorityExecutor, restore_first


def test_urgent_overtakes_queued_but_not_running_work():
    ex = PriorityExecutor('t')
    gate, order = threading.Event(), []
    first = ex.submit(lambda: (gate.wait(5), order.append('running'))[1])
    time.sleep(.05)                                     # 'running' occupies the worker
    normals = [ex.submit(order.append, f'n{i}') for i in range(3)]
    urgents = [ex.submit_urgent(order.append, f'u{i}') for i in range(2)]
    gate.set()
    for f in (first, *normals, *urgents):
        f.result(5)
    assert order == ['running', 'u0', 'u1', 'n0', 'n1', 'n2']
    assert ex.urgent_submitted == 2 and ex.urgent_overtakes == 6
    ex.shutdown()


def test_results_exceptions_and_shutdown():
    ex = PriorityExecutor('t')
    assert ex.submit(lambda a, b=1: a + b, 2, b=3).result(5) == 5
    def boom():
        raise ValueError('x')
    with pytest.raises(ValueError):
        ex.submit_urgent(boom).result(5)
    ex.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        ex.submit(print)


def test_flag(monkeypatch):
    monkeypatch.delenv('KVMEM_RESTORE_FIRST', raising=False)
    assert restore_first() is False
    monkeypatch.setenv('KVMEM_RESTORE_FIRST', 'yes')
    with pytest.raises(ValueError):
        restore_first()
