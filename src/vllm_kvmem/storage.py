"""Bounded asynchronous NVMe archive over the original coalescing I/O engine.

Records are immutable after publication. Request namespace plus repository
generation is mandatory. No LRU eviction of authoritative data is permitted;
repository GC releases records only after all consumers have dropped leases.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Condition, Lock
import hashlib
import time

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
        if options.async_seal:
            largest_batch *= 2  # Owned source tensors plus immutable snapshot.
        if options.archive_io_max_bytes < largest_batch:
            raise ValueError('Archive I/O byte capacity cannot fit one full serving batch')
    cls, kwargs = AsyncArchive, {}
    if options.cpu_read_cache_bytes:
        from .cached_storage import CachedArchive
        cls = CachedArchive
        kwargs.update(cache_bytes=options.cpu_read_cache_bytes,
                      gather_workers=options.cpu_gather_workers,
                      namespace_bytes=(min(options.cpu_history_cache_bytes_per_request,
                                           options.memory.repository_blocks * record_bytes)
                          if options.cpu_history_cache_bytes_per_request else
                          (options.memory.select_budget + options.memory.gen_budget)
                          // options.memory.block_tokens * record_bytes))
    archive = cls(str(directory), record_bytes, capacity, preallocate=options.nvme_preallocate,
               max_bytes=options.nvme_max_bytes, io_max_bytes=options.archive_io_max_bytes, **kwargs)
    archive.async_seal = options.async_seal
    return archive


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
        return self._write(namespace, leases, payload=payload)

    def write_from_views(self, namespace: str, leases: list, views) -> ArchiveJob | None:
        """Admit sealed, exclusively owned tensors before packing/hashing.

        A successful call retains buffer exports and cloned generation leases.
        The caller must never mutate sealed sources. Both source and snapshot
        bytes count against I/O admission until the background write completes.
        """
        views = tuple(memoryview(v).cast('B') for v in views)
        if (len(views) != 2 * len(leases) or not leases or
                any(not v.c_contiguous or v.ndim != 1 for v in views) or
                any(views[2*i].nbytes + views[2*i+1].nbytes != self.record_bytes
                    for i in range(len(leases)))):
            raise ValueError('Sealed views must cover each complete raw-K/V record')
        return self._write(namespace, leases, views=views)

    def _write(self, namespace, leases, *, payload=None, views=None):
        deferred = views is not None
        if deferred:
            size = len(leases) * self.record_bytes
        else:
            size = len(payload)
        if size != len(leases) * self.record_bytes or not leases:
            raise ValueError("Write requires complete, fixed-size records")
        keys = [self.key(namespace, lease) for lease in leases]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate archive record in one batch")
        admitted_bytes = size * (2 if deferred else 1)
        with self._lock:
            if any(key in self._entries for key in keys):
                raise ValueError("Cannot overwrite immutable archive records")
            self._check_extent(admitted_bytes)
            if len(self._free_ids) < len(keys) or not self._admit(admitted_bytes):
                return None
            entries, owned = [], []
            try:
                # Snapshot only after byte admission. Refused writers do not
                # allocate another full payload while the queue is saturated.
                if not deferred:
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
                if not deferred:
                    self._cache_write(keys, payload)
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(admitted_bytes)
                raise
        spans = [_native.IoSpan(e.allocation.slot, i * self.record_bytes, self.record_bytes)
                 for i, e in enumerate(entries)]

        def work():
            snapshot = payload
            try:
                preparation = {}
                if deferred:
                    started = time.perf_counter()
                    snapshot = _native.pack_record_views(views)
                    packed = time.perf_counter()
                    byte_view = memoryview(snapshot)
                    digests = tuple(hashlib.sha256(byte_view[i*self.record_bytes:(i+1)*self.record_bytes])
                                    .hexdigest() for i in range(len(keys)))
                    hashed = time.perf_counter()
                    with self._lock:
                        self._cache_write(keys, snapshot)
                    preparation = dict(record_digests=digests, seal_pack_seconds=packed-started,
                        seal_hash_seconds=hashed-packed, seal_cache_seconds=time.perf_counter()-hashed)
                stats = self.tier.io(spans, snapshot, True)
                stats.update(preparation)
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
                    self._complete(admitted_bytes)
        # Executor submission and close are serialized; close cannot shut down
        # the pool while an admitted task still needs to be submitted.
        with self._lock:
            try:
                return ArchiveJob(self.executor.submit(work))
            except BaseException:
                owned.clear()
                for key in keys:
                    self._release(key)
                self._complete(admitted_bytes)
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

    def _region_sources(self, keys):
        return [None] * len(keys)

    def _gather_region(self, destination, sources, offset, length):
        _native.gather_record_region(destination, sources, self.record_bytes,
                                     offset, length, 0, len(sources))
        return 1

    def read_region(self, namespace, leases, offset, length, *, destination=None):
        """Read only raw K or V from immutable CPU records.

        The unchanged NVMe tier reads whole records on a cache miss. Its bounded
        fallback slab is charged separately; only the requested field is copied
        to the caller. Namespace, generation, readiness and leases stay mandatory.
        """
        if (not leases or type(offset) is not int or type(length) is not int or
                offset < 0 or length <= 0 or offset + length > self.record_bytes):
            raise ValueError('Invalid archive record region')
        size = len(leases) * length
        destination = read_destination(destination, size)
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError('Archive record missing, stale, or not yet published')
            sources = self._region_sources(keys)
            misses = [i for i, source in enumerate(sources) if source is None]
            # A fallback record slab and immutable cache copies can coexist.
            admitted_bytes = size + 2 * len(misses) * self.record_bytes
            if not self._admit(admitted_bytes):
                return None
            try:
                owned = [lease.clone() for lease in leases]
            except BaseException:
                self._complete(admitted_bytes)
                raise
        spans = [_native.IoSpan(entries[i].allocation.slot, j*self.record_bytes, self.record_bytes)
                 for j, i in enumerate(misses)]
        submitted_at = time.perf_counter()

        def work():
            try:
                started = time.perf_counter()
                data = bytearray(size) if destination is None else destination
                fallback = bytearray(len(misses)*self.record_bytes)
                stats = self.tier.io(spans, fallback, False)
                read_at = time.perf_counter()
                view = memoryview(fallback)
                for j, i in enumerate(misses):
                    sources[i] = view[j*self.record_bytes:(j+1)*self.record_bytes]
                tasks = self._gather_region(memoryview(data), sources, offset, length)
                copied_at = time.perf_counter()
                if misses:
                    with self._lock:
                        for i in misses:
                            self._cache_write([keys[i]], bytes(sources[i]))
                stats.update(logical_bytes=size, cache_hit_bytes=(len(keys)-len(misses))*length,
                    region_offset=offset, region_bytes=length, cache_gather_tasks=tasks,
                    read_queue_wait_seconds=started-submitted_at,
                    native_read_call_seconds=read_at-started, cache_copy_seconds=copied_at-read_at,
                    cache_fill_seconds=time.perf_counter()-copied_at)
                return data, stats
            finally:
                owned.clear()
                with self._lock:
                    self._complete(admitted_bytes)

        with self._lock:
            try:
                return ArchiveJob(self.executor.submit(work))
            except BaseException:
                owned.clear()
                self._complete(admitted_bytes)
                raise

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
