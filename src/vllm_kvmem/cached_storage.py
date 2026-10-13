"""Optional bounded CPU read cache over the authoritative native NVMe tier.

Resident entries are not authority and never bypass namespace/generation/lease
validation. I/O admission remains bounded by the underlying archive queue.
"""
import time
from concurrent.futures import ThreadPoolExecutor

from . import _native
from .storage import AsyncArchive, ArchiveJob, read_destination
from .read_cache import RecordReadCache


class CachedArchive(AsyncArchive):
    def __init__(self, *args, cache_bytes, namespace_bytes, gather_workers=8, **kwargs):
        if type(gather_workers) is not int or not 1 <= gather_workers <= 16:
            raise ValueError('Cache gather workers must be between 1 and 16')
        self.read_cache = RecordReadCache(cache_bytes, namespace_bytes)
        super().__init__(*args, **kwargs)
        # Separate from the archive executor: an I/O task may wait for gather,
        # so sharing its own bounded executor would deadlock under concurrency.
        self.gather_workers = gather_workers
        self._gather_executor = ThreadPoolExecutor(max_workers=gather_workers,
                                                   thread_name_prefix='kvmem-gather')

    def _gather(self, destination, cached):
        """Gather immutable hits without holding the GIL during bulk copies.

        This mirrors qw3's bounded CPU slab gather. The archive admission bounds
        in-flight reads; each adds at most gather_workers tasks and no payload
        copies. Every task drains before leases or the destination may be freed.
        """
        hits = sum(hit is not None for hit in cached)
        if not hits:
            return 0
        workers = min(self.gather_workers, hits)
        if workers == 1 or hits * self.record_bytes < 8 * 1024**2:
            _native.gather_cached_records(destination, cached, self.record_bytes, 0, len(cached))
            return 1
        jobs, error = [], None
        try:
            for worker in range(workers):
                jobs.append(self._gather_executor.submit(_native.gather_cached_records,
                    destination, cached, self.record_bytes,
                    len(cached) * worker // workers, len(cached) * (worker + 1) // workers))
        except BaseException as failure:
            error = failure
        for job in jobs:
            try:
                job.result()
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error
        return len(jobs)

    def _cache_write(self, keys, payload):
        for index, key in enumerate(keys):
            # Cache admission does not publish authority. read() still requires
            # completed native I/O; write failure evicts this speculative copy.
            begin = index * self.record_bytes
            self.read_cache.put(key, payload[begin:begin+self.record_bytes])

    def _region_sources(self, keys):
        return [self.read_cache.get(key) for key in keys]

    def _gather_region(self, destination, sources, offset, length):
        workers = min(self.gather_workers, len(sources))
        if workers == 1 or len(sources)*length < 8*1024**2:
            return super()._gather_region(destination, sources, offset, length)
        jobs, error = [], None
        try:
            for worker in range(workers):
                jobs.append(self._gather_executor.submit(_native.gather_record_region,
                    destination, sources, self.record_bytes, offset, length,
                    len(sources)*worker//workers, len(sources)*(worker+1)//workers))
        except BaseException as failure:
            error = failure
        for job in jobs:
            try:
                job.result()
            except BaseException as failure:
                if error is None:
                    error = failure
        if error is not None:
            raise error
        return len(jobs)

    def read(self, namespace, leases, *, destination=None):
        if not leases:
            raise ValueError("Empty read")
        destination = read_destination(destination, len(leases) * self.record_bytes)
        size = len(leases) * self.record_bytes
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError("Archive record missing, stale, or not yet published")
            if not self._admit(size):
                return None
            owned = []
            try:
                owned = [lease.clone() for lease in leases]
                cached = [self.read_cache.get(key) for key in keys]
            except BaseException:
                owned.clear()
                self._complete(size)
                raise
        spans = [_native.IoSpan(entry.allocation.slot, i*self.record_bytes, self.record_bytes)
                 for i, (entry, hit) in enumerate(zip(entries, cached)) if hit is None]

        submitted_at = time.perf_counter()
        def work():
            try:
                started_at = time.perf_counter()
                data = bytearray(len(keys) * self.record_bytes) if destination is None else destination
                allocated_at = time.perf_counter()
                view = memoryview(data)
                gather_tasks = self._gather(view, cached)
                copied_at = time.perf_counter()
                stats = self.tier.io(spans, data, False)
                read_at = time.perf_counter()
                stats['logical_bytes'] = len(data)
                stats['cache_hit_bytes'] = sum(hit is not None for hit in cached) * self.record_bytes
                with self._lock:
                    for i, (key, hit) in enumerate(zip(keys, cached)):
                        if hit is None:
                            self.read_cache.put(key, bytes(view[i*self.record_bytes:(i+1)*self.record_bytes]))
                stats.update(read_queue_wait_seconds=started_at-submitted_at,
                    read_allocation_seconds=allocated_at-started_at,
                    cache_gather_tasks=gather_tasks,
                    cache_copy_seconds=copied_at-allocated_at,
                    native_read_call_seconds=read_at-copied_at,
                    cache_fill_seconds=time.perf_counter()-read_at)
                return data, stats
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

    def _release(self, key):
        released = super()._release(key)
        if released:
            self.read_cache.remove(key)
        return released

    def close(self):
        if not super().close():
            return False
        self._gather_executor.shutdown(wait=True)
        self.read_cache.clear()
        return True
