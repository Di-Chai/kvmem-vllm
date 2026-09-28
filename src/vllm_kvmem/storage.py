"""Bounded asynchronous NVMe archive over the original coalescing I/O engine.

Records are immutable after publication. Request namespace plus repository
generation is mandatory. No LRU eviction of authoritative data is permitted;
repository GC releases records only after all consumers have dropped leases.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock

from . import _native


def read_destination(destination, size):
    """Retain the caller's exclusive writable byte buffer until I/O completes."""
    if destination is None:
        return None
    try:
        view = memoryview(destination)
    except TypeError as error:
        raise ValueError("Read destination must export writable contiguous bytes") from error
    if view.readonly or view.ndim != 1 or view.itemsize != 1 or not view.c_contiguous or view.nbytes != size:
        raise ValueError("Read destination must have the exact writable contiguous byte extent")
    return view.cast('B')


@dataclass
class _Entry:
    allocation: object
    ready: bool = False


class ArchiveJob:
    """Once admitted, I/O drains even if its request is cancelled."""
    def __init__(self, future: Future):
        self._future = future

    def result(self, timeout=None):
        return self._future.result(timeout)

    def done(self):
        return self._future.done()

    def cancel(self):
        return False


class AsyncArchive:
    def __init__(self, directory: str, record_bytes: int, capacity: int,
                 workers: int = 2, max_pending: int = 4):
        if min(record_bytes, capacity, workers, max_pending) <= 0:
            raise ValueError("Archive capacities must be positive")
        self.record_bytes = record_bytes
        self.tier = _native.NvmeTier(directory, record_bytes * capacity, record_bytes)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kvmem-io")
        self.max_pending = max_pending
        self._pending = 0
        self._lock = Lock()
        self._entries = {}
        self._free_ids = list(reversed(range(capacity)))
        self._closed = False

    @staticmethod
    def key(namespace: str, lease):
        if not namespace or not lease:
            raise ValueError("Archive access requires a namespace and a live lease")
        h = lease.handle
        return namespace, h.slot, h.generation

    def _admit(self):
        if self._closed:
            raise RuntimeError("Archive is closed")
        if self._pending >= self.max_pending:
            return False
        self._pending += 1
        return True

    def write(self, namespace: str, leases: list, payload: bytes) -> ArchiveJob | None:
        if len(payload) != len(leases) * self.record_bytes or not leases:
            raise ValueError("Write requires complete, fixed-size records")
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate archive record in one batch")
        payload = bytes(payload)  # Snapshot mutable buffers before asynchronous IO.
        with self._lock:
            if any(key in self._entries for key in keys):
                raise ValueError("Cannot overwrite immutable archive records")
            if len(self._free_ids) < len(keys) or not self._admit():
                return None
            entries = []
            for key in keys:
                allocation = self.tier.allocate(self._free_ids.pop())
                if allocation.slot < 0:
                    raise RuntimeError("Archive allocator capacity invariant violated")
                entry = _Entry(allocation)
                self._entries[key] = entry
                entries.append(entry)
            owned = [lease.clone() for lease in leases]
        spans = [_native.IoSpan(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                 for i, e in enumerate(entries)]

        def work():
            try:
                stats = self.tier.io(spans, payload, True)
                with self._lock:
                    for entry in entries:
                        entry.ready = True
                return stats
            except BaseException:
                with self._lock:
                    for key in keys:
                        self._release(key)
                raise
            finally:
                owned.clear()
                with self._lock:
                    self._pending -= 1
        # Executor submission and close are serialized; close cannot shut down
        # the pool while an admitted task still needs to be submitted.
        with self._lock:
            return ArchiveJob(self.executor.submit(work))

    def read(self, namespace: str, leases: list, *, destination=None) -> ArchiveJob | None:
        if not leases:
            raise ValueError("Empty read")
        destination = read_destination(destination, len(leases) * self.record_bytes)
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(e is None or not e.ready for e in entries):
                raise KeyError("Archive record missing, stale, or not yet published")
            if not self._admit():
                return None
            owned = [lease.clone() for lease in leases]
        spans = [_native.IoSpan(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                 for i, e in enumerate(entries)]

        def work():
            try:
                payload = bytearray(len(entries) * self.record_bytes) if destination is None else destination
                stats = self.tier.io(spans, payload, False)
                return payload, stats
            finally:
                owned.clear()
                with self._lock:
                    self._pending -= 1
        with self._lock:
            return ArchiveJob(self.executor.submit(work))

    def _release(self, key) -> bool:
        entry = self._entries.get(key)
        if entry is None:
            return True
        if not self.tier.release(entry.allocation):
            return False
        self._free_ids.append(entry.allocation.block_id)
        del self._entries[key]
        return True

    def reclaim(self, namespace: str, record) -> bool:
        """Only call from Repository.reclaim's release callback (leases == 0)."""
        key = namespace, record.handle.slot, record.handle.generation
        if not self._lock.acquire(blocking=False):
            return False
        try:
            return self._release(key)
        finally:
            self._lock.release()

    def close(self) -> bool:
        with self._lock:
            if self._pending:
                return False
            if self._closed:
                return True
            self._closed = True
            self.executor.shutdown(wait=False)
            self._entries.clear()
            self._free_ids.clear()
            self.tier = None
            return True
