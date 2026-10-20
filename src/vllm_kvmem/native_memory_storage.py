"""Native-owned immutable CPU payloads with independently deferred reclamation.

Python holds allocation descriptors only. Large records are copied and released
by native code without the GIL; exported reads retain actual CPU ownership.
"""
import time

from . import _native
from .memory_storage import MemoryAllocation, MemoryTier, gather_records_native


class NativeMemoryTier(MemoryTier):
    def __init__(self, record_bytes, capacity):
        super().__init__(record_bytes, capacity)
        self.payloads = _native.HistoricalPayloadArena(capacity, record_bytes)
        self._publishing = set()
        self._failed_publications = set()

    @staticmethod
    def payload_handle(allocation):
        return _native.HistoricalPayloadHandle(allocation.slot, allocation.generation)

    def borrow_records(self, allocations):
        """Acquire CPU rights at I/O admission, before queuing any copy task."""
        sources = []
        with self.lock:
            for allocation in allocations:
                if (self.allocations.get(allocation.slot) != allocation or
                        allocation.slot not in self.records or
                        allocation.slot in self._failed_publications):
                    raise KeyError('CPU archive record is missing or unpublished')
                source = self.payloads.acquire(self.records[allocation.slot])
                if not source:
                    raise KeyError('CPU archive record has been retired')
                sources.append(source)
        return sources

    def publish_record_views(self, allocations, record_views):
        """Copy raw-K/V views directly into native owners, without a bytes slab."""
        start = time.perf_counter_ns()
        allocations, record_views = tuple(allocations), tuple(record_views)
        if (len(allocations) != len(record_views) or
                any(type(a) is not MemoryAllocation for a in allocations) or
                len({a.slot for a in allocations}) != len(allocations)):
            raise ValueError('Invalid native CPU record publication')
        slots = {a.slot for a in allocations}
        with self.lock:
            if any(self.allocations.get(a.slot) != a for a in allocations):
                raise ValueError('CPU archive allocation changed before publication')
            if slots.intersection(self.records) or slots.intersection(self._publishing):
                raise ValueError('Cannot overwrite immutable CPU authority')
            self._publishing.update(slots)
        published = {}
        try:
            for allocation, views in zip(allocations, record_views):
                handle = self.payload_handle(allocation)
                self.payloads.publish_segments(handle, views)
                published[allocation.slot] = handle
            with self.lock:
                self.records.update(published)
            return dict(bytes=len(allocations)*self.record_bytes, syscalls=0,
                        duration_ns=time.perf_counter_ns()-start)
        except BaseException:
            # Preserve exact owners for cleanup even if the batch failed after
            # publishing a prefix. These records never become readable.
            with self.lock:
                self.records.update(published)
                self._failed_publications.update(published)
            for handle in published.values():
                self.payloads.retire(handle)
            raise
        finally:
            with self.lock:
                self._publishing.difference_update(slots)

    def publish_records(self, allocations, records):
        records = tuple(records)
        if any(type(record) is not bytes or len(record) != self.record_bytes for record in records):
            raise ValueError('Invalid owned CPU record publication')
        return self.publish_record_views(allocations, [(record,) for record in records])

    def retire_payload(self, allocation):
        """Close CPU reads independently of repository/GPU metadata ownership."""
        with self.lock:
            if self.allocations.get(allocation.slot) != allocation:
                return _native.HistoricalPayloadRetireResult.STALE
            handle = self.records.get(allocation.slot)
            if handle is None:
                return _native.HistoricalPayloadRetireResult.RETRY
            return self.payloads.retire(handle)

    def release(self, allocation):
        with self.lock:
            if self.allocations.get(allocation.slot) != allocation or allocation.slot in self._publishing:
                return False
            handle = self.records.get(allocation.slot)
            if handle is not None:
                self.payloads.retire(handle)
                if not self.payloads.reclaimed(handle):
                    return False
            del self.allocations[allocation.slot]
            self.records.pop(allocation.slot, None)
            self._failed_publications.discard(allocation.slot)
            return True

    def read_from_sources(self, spans, destination, sources):
        start = time.perf_counter_ns()
        view = memoryview(destination).cast('B')
        packed = (view.nbytes == len(spans)*self.record_bytes and
                  all(span.offset == i*self.record_bytes for i, span in enumerate(spans)))
        if not sources:
            tasks = 0
        elif packed and self.gather is not None:
            tasks = self.gather(view, sources, 0, self.record_bytes)
        elif packed:
            gather_records_native(view, sources, self.record_bytes, 0, self.record_bytes, 0, len(sources))
            tasks = 1
        else:
            for span, source in zip(spans, sources):
                view[span.offset:span.offset+span.size] = memoryview(source)
            tasks = 1
        return dict(bytes=len(spans)*self.record_bytes, syscalls=0,
                    duration_ns=time.perf_counter_ns()-start, cache_gather_tasks=tasks,
                    cache_copy_seconds=(time.perf_counter_ns()-start)/1e9)

    def io(self, spans, buffer, write):
        spans = tuple(spans)
        view = memoryview(buffer).cast('B')
        with self.lock:
            if (len({s.slot for s in spans}) != len(spans) or
                    any(s.slot not in self.allocations or s.size != self.record_bytes or
                        s.offset < 0 or s.offset+s.size > view.nbytes for s in spans)):
                raise ValueError('Invalid CPU archive I/O extent')
            allocations = tuple(self.allocations[s.slot] for s in spans)
        if write:
            return self.publish_record_views(allocations,
                [(view[s.offset:s.offset+s.size],) for s in spans])
        if view.readonly:
            raise ValueError('CPU archive read destination is readonly')
        sources = self.borrow_records(allocations)
        return self.read_from_sources(spans, view, sources)
