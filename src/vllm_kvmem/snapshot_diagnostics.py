"""Rate-limit replaceable diagnostics without sampling lifecycle transitions.

Callers retain exact counters in memory and provide inexpensive state identities.
An interval of zero preserves per-step snapshots for diagnostic comparisons.
"""
import math
import os
import time


class SnapshotCadence:
    def __init__(self, interval_seconds=None, *, clock=time.perf_counter):
        if interval_seconds is None:
            interval_seconds = float(os.getenv('KVMEM_SNAPSHOT_INTERVAL_SECONDS', '1'))
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError('Snapshot interval must be finite and nonnegative')
        self.interval, self.clock = interval_seconds, clock
        self.last_time = None
        self.last_identity = None

    def due(self, identity, *, force=False):
        now = self.clock()
        if (force or self.last_time is None or identity != self.last_identity or
                now - self.last_time >= self.interval):
            self.last_time, self.last_identity = now, identity
            return True
        return False
