"""Bounded FIFO admission in front of continuous batching.

The model service keeps ownership of GPU batching. This queue admits independent
offline jobs, retains their history credits through a separate retirement fence,
and refills individual slots. It never evicts history or retries failed jobs.
"""
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import time


@dataclass(frozen=True)
class OfflineTask:
    id: str
    payload: object
    history_bytes: int


class OfflineQueue:
    """Single-owner coordinator with separate inference and retirement pools.

    ``finish(task, output)`` must confirm the task's own release fence before
    returning. Both its admission slot and CPU credits remain charged until
    then; other finished tasks can retire and refill independently. Callbacks
    run on the coordinator except work/finish. A failure calls on_failure before
    waiting for other threads, so an owning service controller can abort I/O.
    """

    def __init__(self, tasks, *, max_inflight, history_capacity_bytes,
                 retirement_workers=2, clock=time.monotonic):
        if any(type(n) is not int or n <= 0 for n in
               (max_inflight, history_capacity_bytes, retirement_workers)):
            raise ValueError('Offline queue limits must be positive integers')
        self.tasks = tuple(tasks)
        ids = [task.id for task in self.tasks]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError('Offline task IDs must be unique nonempty strings')
        if any(type(t.history_bytes) is not int or not 0 < t.history_bytes <= history_capacity_bytes
               for t in self.tasks):
            raise ValueError('Each full history must fit the CPU admission budget')
        self.max_inflight = max_inflight
        self.history_capacity_bytes = history_capacity_bytes
        self.retirement_workers = min(retirement_workers, max_inflight)
        self.clock = clock
        self.reserved_bytes = self.inflight = self.peak_inflight = self.peak_bytes = 0
        self.started = False
        now = clock()
        self.states = {t.id: dict(id=t.id, state='waiting', queued_at=now,
                                 history_bytes=t.history_bytes) for t in self.tasks}

    def snapshot(self):
        return dict(policy='continuous_fifo', counts=dict(Counter(
            row['state'] for row in self.states.values())), inflight=self.inflight,
            max_inflight=self.max_inflight, reserved_bytes=self.reserved_bytes,
            history_capacity_bytes=self.history_capacity_bytes,
            peak_inflight=self.peak_inflight, peak_reserved_bytes=self.peak_bytes,
            tasks=[dict(row) for row in self.states.values()])

    def run(self, work, finish, *, on_result=lambda result: None,
            on_failure=lambda error: None, on_state=lambda state: None):
        if self.started:
            raise RuntimeError('An offline queue attempt cannot be reused')
        self.started = True
        waiting = deque(self.tasks)
        inference, retiring, results = {}, {}, []
        model_pool = ThreadPoolExecutor(max_workers=self.max_inflight,
                                        thread_name_prefix='offline-model')
        retire_pool = ThreadPoolExecutor(max_workers=self.retirement_workers,
                                         thread_name_prefix='offline-retire')
        current = None
        try:
            on_state(self.snapshot())
            while waiting or inference or retiring:
                # Publish every available release before admitting more input.
                for future in tuple(retiring):
                    if not future.done():
                        continue
                    current = retiring.pop(future)
                    result = future.result()
                    self.reserved_bytes -= current.history_bytes
                    self.inflight -= 1
                    self.states[current.id].update(state='completed', completed_at=self.clock())
                    results.append(result)
                    on_result(result)
                    on_state(self.snapshot())
                    current = None
                for future in tuple(inference):
                    if not future.done():
                        continue
                    current = inference.pop(future)
                    output = future.result()
                    self.states[current.id].update(state='retiring', inference_done_at=self.clock())
                    retiring[retire_pool.submit(finish, current, output)] = current
                    on_state(self.snapshot())
                    current = None
                while waiting and self.inflight < self.max_inflight:
                    task = waiting[0]
                    if self.reserved_bytes + task.history_bytes > self.history_capacity_bytes:
                        break  # FIFO backpressure; no truncation/recomputation.
                    current = waiting.popleft()
                    self.reserved_bytes += current.history_bytes
                    self.inflight += 1
                    self.peak_inflight = max(self.peak_inflight, self.inflight)
                    self.peak_bytes = max(self.peak_bytes, self.reserved_bytes)
                    self.states[current.id].update(state='running', admitted_at=self.clock())
                    inference[model_pool.submit(work, current)] = current
                    on_state(self.snapshot())
                    current = None
                pending = set(inference) | set(retiring)
                if pending:
                    # A just-submitted fast retirement can refill this slot
                    # immediately; never wait for an entire wave to complete.
                    wait(pending, return_when=FIRST_COMPLETED)
                elif waiting:
                    raise RuntimeError('Offline queue made no progress without a live owner')
        except BaseException as error:
            if current is not None:
                self.states[current.id].update(state='failed', failed_at=self.clock())
            for task in waiting:
                self.states[task.id]['state'] = 'cancelled'
            for future in (*inference, *retiring):
                future.cancel()
            # Outstanding jobs still own their credits. Do not pretend that a
            # failed future means worker DMA or session history was released.
            on_failure(error)
            on_state(self.snapshot())
            raise
        finally:
            # On error, blocking network calls must be aborted by their owning
            # controller; publishing failure cannot wait on that same I/O.
            model_pool.shutdown(wait=False, cancel_futures=True)
            retire_pool.shutdown(wait=False, cancel_futures=True)
        return results
