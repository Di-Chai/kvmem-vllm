"""Bounded asynchronous NVMe archive over the original coalescing I/O engine.

Records are immutable after publication. Request namespace plus repository
generation is mandatory. No LRU eviction of authoritative data is permitted;
repository GC releases records only after all consumers have dropped leases.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Lock

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


def make_shared_archive(options, directory, record_bytes, max_requests, *, max_query_tokens=0):
    """One process-wide arena; each admitted authority has a lifetime partition."""
    capacity = options.memory.repository_blocks * (max_requests + options.session_capacity)
    if options.archive_io_max_bytes:
        block = options.memory.block_tokens
        # Refuse an impossible serving configuration before opening the arena.
        # A full window read and one maximal capture flush must each fit intact.
        largest_batch = max((options.memory.select_budget + options.memory.gen_budget) // block,
                            max(16, (max_query_tokens + block - 1) // block + 1)) * record_bytes
        if options.archive_io_max_bytes < largest_batch:
            raise ValueError('Archive I/O byte capacity cannot fit one full serving batch')
    cls, kwargs = AsyncArchive, {}
    if options.cpu_read_cache_bytes:
        from .cached_storage import CachedArchive
        cls = CachedArchive
        kwargs.update(cache_bytes=options.cpu_read_cache_bytes,
                      gather_workers=options.cpu_gather_workers,
                      namespace_bytes=(options.cpu_history_cache_bytes_per_request or
                          (options.memory.select_budget + options.memory.gen_budget)
                          // options.memory.block_tokens * record_bytes))
    return cls(str(directory), record_bytes, capacity, preallocate=options.nvme_preallocate,
               max_bytes=options.nvme_max_bytes, io_max_bytes=options.archive_io_max_bytes, **kwargs)


class AsyncArchive:
    def __init__(self, directory: str, record_bytes: int, capacity: int,
                 workers: int = 2, max_pending: int = 4, *, preallocate: bool = False,
                 max_bytes: int = 0, io_max_bytes: int = 0):
        if min(record_bytes, capacity, workers, max_pending) <= 0:
            raise ValueError("Archive capacities must be positive")
        self.record_bytes = record_bytes
        if type(preallocate) is not bool or type(max_bytes) is not int or max_bytes < 0:
            raise ValueError('Invalid NVMe reservation/quota option')
        self.capacity_bytes = record_bytes * capacity
        if max_bytes and self.capacity_bytes > max_bytes:
            raise ValueError('Shared NVMe arena exceeds configured byte quota')
        if type(io_max_bytes) is not int or io_max_bytes < 0 or 0 < io_max_bytes < record_bytes:
            raise ValueError('Archive I/O byte capacity must fit at least one record')
        self.io_max_bytes = io_max_bytes
        self._io_bytes = self._io_peak_bytes = self._io_deferrals = 0
        self.preallocated = preallocate
        self.tier = (_native.NvmeTier(directory, self.capacity_bytes, record_bytes, True)
                     if preallocate else _native.NvmeTier(directory, self.capacity_bytes, record_bytes))
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kvmem-io")
        self.max_pending = max_pending
        self._pending = 0
        self._lock = Lock()
        self._capacity_changed = Condition(self._lock)
        self._entries = {}
        self._free_ids = list(reversed(range(capacity)))
        self._closed = False

    @staticmethod
    def key(namespace: str, lease):
        if not namespace or not lease:
            raise ValueError("Archive access requires a namespace and a live lease")
        h = lease.handle
        return namespace, h.slot, h.generation

    def _check_extent(self, size):
        if type(size) is not int or size <= 0:
            raise ValueError('Archive I/O extent must be positive')
        if self.io_max_bytes and size > self.io_max_bytes:
            # Retrying an impossible batch would otherwise deadlock serving.
            raise ValueError('Archive I/O batch exceeds configured byte capacity')

    def _has_capacity(self, size):
        return (self._pending < self.max_pending and
                (not self.io_max_bytes or self._io_bytes + size <= self.io_max_bytes))

    def _admit(self, size):
        if self._closed:
            raise RuntimeError("Archive is closed")
        self._check_extent(size)
        if not self._has_capacity(size):
            self._io_deferrals += 1
            return False
        self._pending += 1
        self._io_bytes += size
        self._io_peak_bytes = max(self._io_peak_bytes, self._io_bytes)
        return True

    def _complete(self, size):
        """Caller holds the archive lock; I/O and its generation leases drained."""
        self._pending -= 1
        self._io_bytes -= size
        self._capacity_changed.notify_all()

    def io_usage(self):
        with self._lock:
            return dict(archive_io_max_bytes=self.io_max_bytes,
                        archive_io_pending_bytes=self._io_bytes,
                        archive_io_peak_bytes=self._io_peak_bytes,
                        archive_io_deferrals=self._io_deferrals)

    def wait_for_capacity(self, timeout, *, size=None):
        """Host retirement may wait for I/O admission; model threads need not."""
        size = self.record_bytes if size is None else size
        self._check_extent(size)
        with self._capacity_changed:
            ready = self._capacity_changed.wait_for(
                lambda: self._closed or self._has_capacity(size), timeout)
            if self._closed:
                raise RuntimeError('Archive closed while waiting for retirement I/O')
            return ready

    def wait_for_idle(self, timeout):
        """Drain shared work, including jobs owned by detached retirement."""
        with self._capacity_changed:
            ready = self._capacity_changed.wait_for(lambda: not self._pending or self._closed, timeout)
            if self._closed:
                raise RuntimeError('Archive closed while waiting for shared I/O')
            return ready

    def write(self, namespace: str, leases: list, payload: bytes) -> ArchiveJob | None:
        if len(payload) != len(leases) * self.record_bytes or not leases:
            raise ValueError("Write requires complete, fixed-size records")
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate archive record in one batch")
        size = len(leases) * self.record_bytes
        with self._lock:
            if any(key in self._entries for key in keys):
                raise ValueError("Cannot overwrite immutable archive records")
            self._check_extent(size)
            if len(self._free_ids) < len(keys) or not self._admit(size):
                return None
            entries, owned = [], []
            try:
                # Snapshot only after byte admission. Refused writers do not
                # allocate another full payload while the queue is saturated.
                payload = bytes(payload)
                for key in keys:
                    block_id = self._free_ids.pop()
                    try:
                        allocation = self.tier.allocate(block_id)
                        if allocation.slot < 0:
                            raise RuntimeError("Archive allocator capacity invariant violated")
                    except BaseException:
                        self._free_ids.append(block_id)
                        raise
                    entry = _Entry(allocation)
                    self._entries[key] = entry
                    entries.append(entry)
                owned = [lease.clone() for lease in leases]
                self._cache_write(keys, payload)
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(size)
                raise
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
                    self._complete(size)
        # Executor submission and close are serialized; close cannot shut down
        # the pool while an admitted task still needs to be submitted.
        with self._lock:
            try:
                return ArchiveJob(self.executor.submit(work))
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(size)
                raise

    def _cache_write(self, keys, payload):
        """Optional cache admission under lock, before asynchronous publication."""

    def read(self, namespace: str, leases: list, *, destination=None) -> ArchiveJob | None:
        if not leases:
            raise ValueError("Empty read")
        destination = read_destination(destination, len(leases) * self.record_bytes)
        size = len(leases) * self.record_bytes
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(e is None or not e.ready for e in entries):
                raise KeyError("Archive record missing, stale, or not yet published")
            if not self._admit(size):
                return None
            try:
                owned = [lease.clone() for lease in leases]
            except BaseException:
                self._complete(size)
                raise
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
                    self._complete(size)
        with self._lock:
            try:
                return ArchiveJob(self.executor.submit(work))
            except BaseException:
                owned.clear()
                self._complete(size)
                raise

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
            self._capacity_changed.notify_all()
            self.executor.shutdown(wait=False)
            self._entries.clear()
            self._free_ids.clear()
            self.tier = None
            return True
