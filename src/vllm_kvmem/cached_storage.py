"""Optional bounded CPU read cache over the authoritative native NVMe tier.

Resident entries are not authority and never bypass namespace/generation/lease
validation. I/O admission remains bounded by the underlying archive queue.
"""
import time

from . import _native
from .storage import AsyncArchive, ArchiveJob, read_destination
from .read_cache import RecordReadCache


class CachedArchive(AsyncArchive):
    def __init__(self, *args, cache_bytes, namespace_bytes, **kwargs):
        self.read_cache = RecordReadCache(cache_bytes, namespace_bytes)
        super().__init__(*args, **kwargs)

    def write(self, namespace, leases, payload):
        payload = bytes(payload)
        keys = [self.key(namespace, lease) for lease in leases]
        job = super().write(namespace, leases, payload)
        if job is None:
            return None

        with self._lock:
            for index, key in enumerate(keys):
                if key in self._entries:
                    # Cache admission does not publish authority. read() still
                    # requires native I/O to mark the entry ready; failed writes
                    # release the entry and evict this speculative cache copy.
                    begin = index * self.record_bytes
                    self.read_cache.put(key, payload[begin:begin+self.record_bytes])
        return job

    def read(self, namespace, leases, *, destination=None):
        if not leases:
            raise ValueError("Empty read")
        destination = read_destination(destination, len(leases) * self.record_bytes)
        keys = [self.key(namespace, lease) for lease in leases]
        with self._lock:
            entries = [self._entries.get(key) for key in keys]
            if any(entry is None or not entry.ready for entry in entries):
                raise KeyError("Archive record missing, stale, or not yet published")
            if not self._admit():
                return None
            owned = [lease.clone() for lease in leases]
            cached = [self.read_cache.get(key) for key in keys]
        spans = [_native.IoSpan(entry.allocation.slot, i*self.record_bytes, self.record_bytes)
                 for i, (entry, hit) in enumerate(zip(entries, cached)) if hit is None]

        submitted_at = time.perf_counter()
        def work():
            try:
                started_at = time.perf_counter()
                data = bytearray(len(keys) * self.record_bytes) if destination is None else destination
                allocated_at = time.perf_counter()
                view = memoryview(data)
                for i, hit in enumerate(cached):
                    if hit is not None:
                        view[i*self.record_bytes:(i+1)*self.record_bytes] = hit
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
                    cache_copy_seconds=copied_at-allocated_at,
                    native_read_call_seconds=read_at-copied_at,
                    cache_fill_seconds=time.perf_counter()-read_at)
                return data, stats
            finally:
                owned.clear()
                with self._lock:
                    self._pending -= 1

        with self._lock:
            return ArchiveJob(self.executor.submit(work))

    def _release(self, key):
        released = super()._release(key)
        if released:
            self.read_cache.remove(key)
        return released

    def close(self):
        if not super().close():
            return False
        self.read_cache.clear()
        return True
