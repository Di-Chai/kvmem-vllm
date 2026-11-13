"""Pace packed KV uploads on the background producer, never the engine thread.

Idle GPU work is unrestricted. Busy transfers share a token bucket, including
both refresh and checkpoint restore. A copy cannot be preempted: the burst bound
is at least one submitted chunk, and an oversized chunk leaves debt for the next
copy. This is a submission-rate ceiling, not a hardware bandwidth guarantee.
"""
from threading import Condition
import time


class TransferBudget:
    def __init__(self, bytes_per_second, burst_bytes, *, clock=time.monotonic):
        if bytes_per_second <= 0 or burst_bytes <= 0:
            raise ValueError('Transfer rate and burst must be positive')
        self.rate, self.burst, self.clock = bytes_per_second, burst_bytes, clock
        self.condition = Condition()
        self.busy = False
        self.credit = float(burst_bytes)
        self.updated = clock()
        self.bytes = self.busy_bytes = self.waits = 0
        self.wait_seconds = 0.

    def _refill(self):
        now = self.clock()
        self.credit = min(self.burst, self.credit + max(0., now-self.updated)*self.rate)
        self.updated = now

    def set_busy(self, busy):
        with self.condition:
            self._refill()
            self.busy = bool(busy)
            if not self.busy:
                self.credit = float(self.burst)
                self.condition.notify_all()

    def acquire(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError('Transfer size must be a positive byte count')
        with self.condition:
            while self.busy:
                self._refill()
                need = min(size, self.burst) - self.credit
                if need <= 0:
                    self.credit -= size
                    self.busy_bytes += size
                    break
                start = self.clock()
                self.waits += 1
                self.condition.wait(min(.05, need/self.rate))
                self.wait_seconds += max(0., self.clock()-start)
            self.bytes += size

    def snapshot(self):
        with self.condition:
            return dict(rate_bytes_per_second=self.rate, burst_bytes=self.burst,
                        busy=self.busy, uploaded_bytes=self.bytes, busy_uploaded_bytes=self.busy_bytes,
                        waits=self.waits, producer_wait_seconds=self.wait_seconds)
