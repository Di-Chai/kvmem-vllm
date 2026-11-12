"""Opt-in KVMEM_RESTORE_FIRST=1: the refresh producer thread runs a queued checkpoint
swap-in before queued pressure-refresh assemblies.

Both still run one at a time on the same single worker and share its staging arena;
only the order of *queued* work changes (FIFO within each class; nothing preempts a
running task). A swap-in then waits for at most the assembly already in progress
instead of every assembly queued ahead of it.
"""
import os
import threading
from collections import deque
from concurrent.futures import Future


def restore_first():
    value = os.getenv('KVMEM_RESTORE_FIRST', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_RESTORE_FIRST must be 0 or 1')
    return value == '1'


class PriorityExecutor:
    """A one-thread executor with two FIFO classes: urgent before normal."""

    def __init__(self, thread_name_prefix='kvmem-priority'):
        self._urgent, self._normal = deque(), deque()
        self._condition = threading.Condition()
        self._shutdown = False
        self.urgent_submitted = self.urgent_overtakes = 0
        self._thread = threading.Thread(target=self._run, name=thread_name_prefix + '_0', daemon=True)
        self._thread.start()

    def _put(self, queue, fn, args, kwargs):
        future = Future()
        with self._condition:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            queue.append((future, fn, args, kwargs))
            self._condition.notify()
        return future

    def submit(self, fn, *args, **kwargs):
        return self._put(self._normal, fn, args, kwargs)

    def submit_urgent(self, fn, *args, **kwargs):
        with self._condition:
            self.urgent_submitted += 1
            self.urgent_overtakes += len(self._normal)
        return self._put(self._urgent, fn, args, kwargs)

    def _run(self):
        while True:
            with self._condition:
                while not self._urgent and not self._normal and not self._shutdown:
                    self._condition.wait()
                if not self._urgent and not self._normal:
                    return
                future, fn, args, kwargs = (self._urgent or self._normal).popleft()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:  # noqa: BLE001 - delivered through the future
                future.set_exception(error)
            else:
                future.set_result(result)

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._condition:
            self._shutdown = True
            if cancel_futures:
                for queue in (self._urgent, self._normal):
                    while queue:
                        queue.popleft()[0].cancel()
            self._condition.notify_all()
        if wait:
            self._thread.join()
