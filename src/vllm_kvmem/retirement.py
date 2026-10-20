"""Bounded CPU-only finalization; native connector acknowledgment owns release.

Only submit detached authorities whose captured GPU work has drained. One worker
owns each submitted authority through publication, optional verification and
lease-safe reclamation. An exception is never a successful completion.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time
from threading import Event

import torch


def retire_authority(authority, verify, timeout=180):
    started_at = time.perf_counter()
    authority.stop_gc()
    deadline = time.monotonic() + timeout
    retry = Event()
    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError('Authority retirement did not drain publication/read leases')
        return left

    while not authority.flush():
        # Request-owned writes may free I/O admission, but complete unpublished
        # records must still fit their lifetime archive reservation.
        for job, _ in authority.writes:
            job.result(timeout=remaining())
        if not authority.archive.wait_for_capacity(remaining()):
            raise TimeoutError('Archive retirement I/O admission timed out')
        retry.wait(min(0.001, remaining()))
    for job, _ in authority.writes:
        job.result(timeout=remaining())
    authority.poll()
    published = sum(record.archived for record in authority.records.values())
    reclaimed = authority.gc_reclaimed_records
    if published + reclaimed != authority.history_end // authority.memory.config.block_tokens:
        raise RuntimeError('Finalized authority has unpublished complete archive records')
    published_at = time.perf_counter()
    verified = 0
    if verify:
        for record in authority.records.values():
            if not record.archived:
                continue
            while True:
                try:
                    raw, values = authority.read_record(record.handle)
                    break
                except BlockingIOError:
                    if not authority.archive.wait_for_capacity(remaining()):
                        raise TimeoutError('Archive retirement read admission timed out')
            digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()
                                    + values.numpy().tobytes()).hexdigest()
            if digest != record.digest:
                raise RuntimeError('Archived raw K/V checksum differs from accepted capture')
            verified += 1
        record = None  # Verification must release its last index owner before close.
    verified_at = time.perf_counter()
    result = {"history_end": authority.history_end, "records": len(authority.records),
              "sealed_records_archived": published, "sealed_records_verified": verified,
              "gc_reclaimed_records": reclaimed}
    while not authority.close():
        # A concurrent archive operation can temporarily own its release lock;
        # readers must drop their generation lease before storage can recycle.
        retry.wait(min(0.001, remaining()))
    finished_at = time.perf_counter()
    result.update(retirement_started_at=started_at, retirement_finished_at=finished_at,
                  publication_seconds=published_at-started_at,
                  verification_seconds=verified_at-published_at,
                  reclamation_seconds=finished_at-verified_at)
    return result


class RetirementQueue:
    def __init__(self, capacity, verify):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Retirement requires a positive request bound')
        self.capacity, self.verify = capacity, verify
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='kvmem-retire')
        self.jobs = {}
        self.closed = False

    def submit(self, request_id, authority):
        if self.closed or request_id in self.jobs:
            raise RuntimeError('Closed retirement queue or duplicate request identity')
        if len(self.jobs) >= self.capacity:
            raise BufferError('Native authority admission exceeded retirement capacity')
        self.jobs[request_id] = self.executor.submit(retire_authority, authority, self.verify)

    def poll(self, request_id):
        job = self.jobs[request_id]
        if not job.done():
            return None
        result = job.result()  # A failed authority remains owned; never acknowledge it.
        del self.jobs[request_id]
        return result

    def close(self):
        if self.jobs:
            return False
        if not self.closed:
            self.executor.shutdown(wait=True)
            self.closed = True
        return True
