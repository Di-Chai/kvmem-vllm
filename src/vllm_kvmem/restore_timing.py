"""Optional engine-thread restore timings; wall time is not GPU idle time."""
from contextlib import contextmanager, nullcontext
import time


class RestoreTiming:
    def __init__(self, enabled=False, *, clock=None, thread_clock=None):
        if type(enabled) is not bool:
            raise ValueError('Restore timing enabled must be boolean')
        self.enabled = enabled
        self._clock = clock if clock is not None else time.perf_counter
        self._thread_clock = thread_clock if thread_clock is not None else time.thread_time
        self._stages = []

    def stage(self, name):
        # The disabled path neither reads a clock nor creates stage records.
        if not self.enabled:
            return nullcontext()
        return self._measure(name)

    @contextmanager
    def _measure(self, name):
        started = self._clock()
        cpu_started = self._thread_clock()
        error = None
        try:
            yield
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            cpu_finished = self._thread_clock()
            finished = self._clock()
            record = dict(name=name, started_at=started, finished_at=finished,
                          wall_seconds=finished-started,
                          thread_cpu_seconds=cpu_finished-cpu_started,
                          status='error' if error is not None else 'ok')
            if error is not None:
                record['error_type'] = error
            self._stages.append(record)

    def summary(self):
        totals = {}
        for record in self._stages:
            total = totals.setdefault(record['name'], dict(calls=0, wall_seconds=0., thread_cpu_seconds=0.))
            total['calls'] += 1
            total['wall_seconds'] += record['wall_seconds']
            total['thread_cpu_seconds'] += record['thread_cpu_seconds']
        return dict(enabled=self.enabled, stages=[dict(row) for row in self._stages], totals=totals,
                    scope='Inclusive host intervals. Nested stages overlap; wall minus thread CPU '
                          'includes waiting and descheduling, not specifically GIL wait or GPU idle.')
