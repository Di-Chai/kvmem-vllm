"""Conservative per-request credits for accepted history not yet archived.

Worker snapshots can precede the latest capture DMA. Scheduler-computed history
therefore consumes credits until the worker reports that exact accepted prefix.
Draft candidates reserve worst-case space; rejection returns it implicitly.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ArchiveCapacity:
    history_end: int
    ready_records: int

    def __post_init__(self):
        if any(type(v) is not int or v < 0 for v in (self.history_end, self.ready_records)):
            raise ValueError('Archive capacity requires nonnegative integer extents')


class ArchiveAdmission:
    def __init__(self, block_tokens, repository_records):
        if min(block_tokens, repository_records) <= 0:
            raise ValueError('Invalid repository capacity geometry')
        self.block_tokens, self.repository_records = block_tokens, repository_records
        self.snapshots = {}
        self.sequence = -1
        self.deferred = 0

    def update(self, sequence, snapshots, live_requests):
        if type(sequence) is not int or sequence <= self.sequence:
            raise ValueError('Stale archive capacity snapshot')
        for request_id, snapshot in snapshots.items():
            if not isinstance(snapshot, ArchiveCapacity) or snapshot.ready_records > self.repository_records:
                raise ValueError('Invalid worker archive capacity')
            if request_id in live_requests:
                previous = self.snapshots.get(request_id)
                if previous is not None and snapshot.history_end < previous.history_end:
                    raise ValueError('Archive accepted history moved backwards')
        self.sequence = sequence
        self.snapshots = {k: v for k, v in self.snapshots.items() if k in live_requests}
        self.snapshots.update({k: v for k, v in snapshots.items() if k in live_requests})

    def allows(self, request_id, history, query_tokens):
        if any(type(v) is not int or v < 0 for v in (history, query_tokens)):
            raise ValueError('Invalid scheduler accepted/query extent')
        snapshot = self.snapshots.get(request_id, ArchiveCapacity(0, self.repository_records))
        if history < snapshot.history_end:
            raise ValueError('Scheduler history precedes its archive')
        tail = snapshot.history_end % self.block_tokens
        pending = history - snapshot.history_end + query_tokens
        needed = (tail + pending + self.block_tokens - 1) // self.block_tokens - bool(tail)
        allowed = needed <= snapshot.ready_records
        self.deferred += int(not allowed)
        return allowed

    def forget(self, request_id):
        self.snapshots.pop(request_id, None)
