"""Native-owned immutable CPU payloads with independently deferred reclamation.

Python holds allocation descriptors only. Large records are copied and released
by native code without the GIL; exported reads retain actual CPU ownership.
"""
import os
import time

from .. import _native
from .memory_storage import MemoryAllocation, MemoryTier, gather_records_native
from . import settings as _settings


def payload_pool_config():
    """Opt-in pre-faulted CPU record pool (identical bytes, different placement).

    KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB: huge-page-advised slab size; 0 disables.
    KVMEM_CPU_PAYLOAD_POOL_AHEAD: free records kept pre-faulted by a native thread.
    """
    chunk = _settings.get('KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB', '0')
    ahead = _settings.get('KVMEM_CPU_PAYLOAD_POOL_AHEAD', '0')
    if not chunk.isdigit() or not ahead.isdigit():
        raise ValueError('CPU payload pool sizes must be non-negative integers')
    chunk, ahead = int(chunk), int(ahead)
    if ahead and not chunk:
        raise ValueError('KVMEM_CPU_PAYLOAD_POOL_AHEAD requires KVMEM_CPU_PAYLOAD_POOL_CHUNK_MIB')
    if chunk > 65536 or ahead > 1 << 20:
        raise ValueError('CPU payload pool sizes are out of range')
    return dict(pool_chunk_mib=chunk, pool_ahead_records=ahead) if chunk else {}


def tiering_drop_cache():
    """KVMEM_TIERING_DROP_CACHE (default 1, session tiering only): a spill waits
    for its bytes to reach the device and drops them from the page cache, and a
    promotion drops what it read, so spilled sessions really leave RAM. 0 keeps
    buffered I/O (spills return sooner; bytes linger as dirty/cached pages)."""
    value = _settings.get('KVMEM_TIERING_DROP_CACHE', '1')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_TIERING_DROP_CACHE must be 0 or 1')
    return value == '1'


class NativeMemoryTier(MemoryTier):
    def __init__(self, record_bytes, capacity):
        super().__init__(record_bytes, capacity)
        self.payloads = _native.HistoricalPayloadArena(capacity, record_bytes, **payload_pool_config())
        self._publishing = set()
        self._failed_publications = set()
        self._spill_store = None
        self._spilled = {}          # slot -> (payload handle, SSD location)

    # Opt-in session tiering (cpu_session_ttl_ms): an idle session's records
    # move to SSD and come back with identical bytes under the same handle.
    def enable_spill(self, directory, max_bytes, segment_bytes=64 * 1024 ** 2, *, drop_cache=None):
        from .tiered_storage import _SegmentStore
        if self._spill_store is not None:
            raise RuntimeError('CPU record spill is already enabled')
        if drop_cache is None:
            drop_cache = tiering_drop_cache()
        self._spill_store = _SegmentStore(directory, self.record_bytes, max_bytes, segment_bytes,
                                          drop_cache=drop_cache)
        self.spill_stats = dict(spilled=0, promoted=0, spill_busy=0)

    def spill(self, allocations, *, written=None):
        """Write each published record to SSD, then drop its CPU bytes.

        A record with a live reader stays in RAM (its SSD copy is discarded).
        Returns the number of records spilled. With ``written`` (a list), the
        new SSD locations are appended there and the caller calls
        flush_spilled() once for a whole session (one device sync, not one
        per parallel chunk)."""
        if self._spill_store is None:
            raise RuntimeError('CPU record spill is not enabled')
        spilled = 0
        deferred = written is not None
        written = written if deferred else []
        try:
            if hasattr(self.payloads, 'write_records'):
                return self._spill_batch(allocations, written)
            for allocation in allocations:
                outcome = self._spill_one(allocation, written)
                if outcome == 'full':
                    # SSD budget full: the rest of this session simply stays in RAM.
                    self.spill_stats['spill_full'] = self.spill_stats.get('spill_full', 0) + 1
                    break
                spilled += outcome == 'spilled'
        finally:
            if not deferred:
                self.flush_spilled(written)
        return spilled

    def flush_spilled(self, written):
        """Spilled bytes reach the device and leave the page cache (drop_cache),
        so RAM is really freed."""
        self._spill_store.flush(written)

    def _spill_one(self, allocation, written):
        with self.lock:
            if (self.allocations.get(allocation.slot) != allocation or allocation.slot in self._spilled or
                    allocation.slot in self._failed_publications or allocation.slot in self._publishing):
                return 'skip'
            handle = self.records.get(allocation.slot)
            read = self.payloads.acquire(handle) if handle is not None else None
        if not read:
            return 'skip'
        try:
            location = self._spill_store.write(memoryview(read))
        except OSError as error:
            if 'quota exhausted' not in str(error):
                raise
            return 'full'
        del read                                # drop the native reader before spilling
        with self.lock:
            if (self.allocations.get(allocation.slot) == allocation and
                    self.records.get(allocation.slot) is handle and self.payloads.spill(handle)):
                self._spilled[allocation.slot] = (handle, location)
                self.spill_stats['spilled'] += 1
                written.append(location)
                return 'spilled'
            self.spill_stats['spill_busy'] += 1
        self._spill_store.release(location)
        return 'skip'

    def _spill_batch(self, allocations, written):
        """Native batched spill: the bytes move without the GIL (one release per
        call); eligibility and the CPU-byte drop stay under the tier lock."""
        store = self._spill_store
        with self.lock:
            eligible = [(a, self.records[a.slot]) for a in allocations
                        if self.allocations.get(a.slot) == a and a.slot in self.records and
                        a.slot not in self._spilled and a.slot not in self._failed_publications and
                        a.slot not in self._publishing]
        if not eligible:
            return 0
        extents = store.reserve(len(eligible))
        if len(extents) < len(eligible):
            # SSD budget full: the rest of this session simply stays in RAM.
            self.spill_stats['spill_full'] = self.spill_stats.get('spill_full', 0) + 1
            eligible = eligible[:len(extents)]
        if not extents:
            return 0
        ok, digests = self.payloads.write_records([h for _, h in eligible], [e[3] for e in extents],
                                                  [e[4] for e in extents], store.drop_cache)
        with self.lock:
            index = [i for i, (a, h) in enumerate(eligible)
                     if ok[i] and self.allocations.get(a.slot) == a and self.records.get(a.slot) is h and
                     a.slot not in self._spilled]
            dropped = self.payloads.spill_many([eligible[i][1] for i in index]) if index else []
            keep = [i for i, d in zip(index, dropped) if d]
            for i, location in zip(keep, store.commit([extents[i] for i in keep], [digests[i] for i in keep])):
                self._spilled[eligible[i][0].slot] = (eligible[i][1], location)
                written.append(location)
            self.spill_stats['spilled'] += len(keep)
            self.spill_stats['spill_busy'] += len(index) - len(keep)
        kept = set(keep)
        store.abandon([e for i, e in enumerate(extents) if i not in kept])
        return len(keep)

    def _promote_batch(self, allocations):
        """Native batched promotion: read, verify (XXH64) and restore without
        the GIL; bookkeeping under the tier lock. A record restored meanwhile
        by another reader is left to that reader's bookkeeping."""
        store = self._spill_store
        with self.lock:
            entries = [(a, self._spilled[a.slot]) for a in allocations
                       if a.slot in self._spilled and self.allocations.get(a.slot) == a]
        if not entries:
            return 0
        dups, valid, fds, offsets = store.open_targets([location for _, (_, location) in entries])
        try:
            entries = [entries[i] for i in valid]
            restored = self.payloads.read_records([handle for _, (handle, _) in entries], fds, offsets,
                                                  [location.digest for _, (_, location) in entries],
                                                  store.drop_cache) if entries else []
        finally:
            store.close_targets(dups)
        release = []
        with self.lock:
            for (allocation, entry), ok in zip(entries, restored):
                if self._spilled.get(allocation.slot) is not entry:
                    continue                    # released, or promoted by another reader
                if not ok:
                    if self.payloads.spilled(entry[0]):
                        raise RuntimeError('Spilled CPU record could not be restored')
                    continue                    # restored by a concurrent reader: its bookkeeping
                del self._spilled[allocation.slot]
                release.append(entry[1])
            self.spill_stats['promoted'] += len(release)
        for location in release:
            store.release(location)
        return len(release)

    def promote(self, allocations):
        """Bring spilled records back into RAM (checksum-verified, same handle)."""
        if hasattr(self.payloads, 'read_records'):
            return self._promote_batch(allocations)
        promoted = 0
        for allocation in allocations:
            with self.lock:
                entry = self._spilled.get(allocation.slot)
                if entry is None or self.allocations.get(allocation.slot) != allocation:
                    continue
            handle, location = entry
            buffer = bytearray(self.record_bytes)
            self._spill_store.read_into(location, memoryview(buffer))
            with self.lock:
                if self._spilled.get(allocation.slot) is not entry:
                    continue                    # promoted or released concurrently
                if not self.payloads.restore(handle, buffer):
                    raise RuntimeError('Spilled CPU record could not be restored')
                del self._spilled[allocation.slot]
                self.spill_stats['promoted'] += 1
                promoted += 1
            self._spill_store.release(location)
        return promoted

    def spilled_allocations(self, allocations):
        with self.lock:
            return [a for a in allocations if a.slot in self._spilled]

    @staticmethod
    def payload_handle(allocation):
        return _native.HistoricalPayloadHandle(allocation.slot, allocation.generation)

    def borrow_records(self, allocations):
        """Acquire CPU rights at I/O admission, before queuing any copy task."""
        if self._spilled:
            # Session tiering normally promotes a session before its turn; any
            # other reader of a spilled record promotes it synchronously here.
            pending = self.spilled_allocations(allocations)
            if pending:
                self.promote(pending)
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
            spilled = self._spilled.pop(allocation.slot, None)
            if spilled is not None:
                self._spill_store.release(spilled[1])
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
