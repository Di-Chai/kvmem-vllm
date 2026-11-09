"""One public Query's lazily admitted immutable CPU record sources.

Only source metadata and read leases are reused. Each layer still copies its
own requested bytes through the original bounded archive gather implementation.
This candidate deliberately excludes independent payload retirement and GC.
"""
from dataclasses import dataclass
from threading import RLock
import time

import torch

from .memory import Phase
from .storage import read_destination


@dataclass(frozen=True)
class _Source:
    key: tuple
    record: object
    handle: object
    history_begin: int
    lease: object
    payload: bytes


def _historical_gc(authority):
    from .historical_gc import HistoricalAuthorityGc
    return isinstance(getattr(authority, 'gc', None), HistoricalAuthorityGc)


class QueryRegionReader:
    """Explicitly closed, generation-bound source cache for one READY forward.

    The caller owns one instance until public Query completion, failure or
    retirement. ``close`` refuses an active CPU producer. Archive shutdown also
    refuses this reader, including before its first lazily selected source.
    """
    def __init__(self, authority, *, request_generation, generation_getter):
        archive, memory = authority.archive, authority.memory
        from .pressure_reader import layerwise_gc
        # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: historical GC with the native
        # payload arena. Sources are then GC read rights held by this reader.
        self.historical = (layerwise_gc() and getattr(archive, 'memory_only', False) and
                           getattr(archive, '_native_payload_arena', None) is not None and
                           _historical_gc(authority))
        if not self.historical and (not getattr(archive, 'memory_only', False) or
                getattr(archive, '_native_payload_arena', None) is not None or
                getattr(archive, '_history_worker', None) is not None or
                getattr(authority, 'gc', None) is not None or
                getattr(memory, '_history_scope', None) is not None):
            raise ValueError('Lazy Query sources require immutable CPU bytes without GC')
        self.admission_rights = []
        if type(request_generation) is not int or request_generation < 0 or not callable(generation_getter):
            raise ValueError('Lazy Query reader requires its native request generation')
        self.authority, self.archive, self.memory = authority, archive, memory
        self.namespace, self.history_end = memory.request_id, authority.history_end
        self.memory_history_end = getattr(memory, 'history_end', self.history_end)
        self.shape, self.record_bytes = tuple(authority.shape), archive.record_bytes
        self.record_table = authority.records
        self.request_generation, self.generation_getter = request_generation, generation_getter
        self._lock = RLock()
        self.sources, self.by_handle = {}, {}
        self.closed = self.failed = False
        self.active = self.read_chunks = self.reused_blocks = 0
        self.registered = False
        authority.poll()
        self.validate()
        with archive._lock:
            if archive._closed:
                raise RuntimeError('Archive is closed')
            archive._prepared_region_readers = getattr(archive, '_prepared_region_readers', 0) + 1
            self.registered = True

    @property
    def source_blocks(self):
        return len(self.sources)

    def validate(self):
        authority, memory = self.authority, self.memory
        if (self.closed or self.failed or authority.memory is not memory or
                authority.archive is not self.archive or authority.records is not self.record_table or
                memory.phase is not Phase.READY or memory.request_id != self.namespace or
                authority.history_end != self.history_end or
                getattr(memory, 'history_end', self.history_end) != self.memory_history_end or
                tuple(authority.shape) != self.shape or self.archive.record_bytes != self.record_bytes or
                self.generation_getter() != self.request_generation):
            raise ValueError('Lazy Query reader belongs to a different owner, generation or accepted prefix')

    def _record(self, handle, key):
        record = self.record_table.get(key)
        if (record is None or record.count != self.shape[1] or not record.archived or
                record.history_begin < 0 or record.history_begin + record.count > self.history_end or
                (record.handle.slot, record.handle.generation) != key):
            raise ValueError('Lazy Query source is absent, partial, unpublished or foreign')
        return record

    def _validate_source(self, source):
        record = source.record
        if (self.record_table.get(source.key) is not record or record.handle is not source.handle or
                record.history_begin != source.history_begin or record.count != self.shape[1] or
                not record.archived):
            raise ValueError('Lazy Query source changed during its public forward')

    def _ensure_sources(self, handles):
        """The reader lock covers cache publication; prepare only new handles."""
        requested, pending = [], {}
        for handle in handles:
            cached = self.by_handle.get(id(handle))
            if cached is not None and cached.handle is handle:
                self._validate_source(cached)
                requested.append(cached.key)
                continue
            key = handle.slot, handle.generation
            cached = self.sources.get(key)
            if cached is not None:
                self._validate_source(cached)
            elif key not in pending:
                pending[key] = self._record(handle, key)
            requested.append(key)
        if not pending:
            return [self.sources[key].payload for key in requested], 0

        owned, installed, payloads, prepared = [], [], [], []
        try:
            for key, record in pending.items():
                lease, payload_right = self.memory.admitted_read(record.handle)
                owned.append(lease)
                if not lease:
                    raise ValueError('Stale lazy Query source generation')
                if payload_right is not None:
                    if not self.historical:
                        raise ValueError('Lazy Query sources cannot bypass native payload admission')
                    self.admission_rights.append(payload_right)
                if (lease.handle.slot, lease.handle.generation) != key:
                    raise ValueError('Lazy Query admission returned a foreign generation')
            archive = self.archive
            with archive._lock:
                self.validate()
                if archive._closed:
                    raise RuntimeError('Archive is closed')
                keys = [(self.namespace, *key) for key in pending]
                entries = [archive._entries.get(key) for key in keys]
                if any(entry is None or not entry.ready for entry in entries):
                    raise KeyError('Lazy Query archive record is absent, stale or unpublished')
                payloads = (archive._historical_region_sources(keys) if self.historical else
                            archive._region_sources(keys))
                if (len(payloads) != len(pending) or
                        any((memoryview(payload).nbytes if self.historical else
                             type(payload) is bytes and len(payload)) != self.record_bytes
                            for payload in payloads)):
                    raise ValueError('Lazy Query sources require exact immutable record bytes')
                prepared = [_Source(key, record, record.handle, record.history_begin, lease, payload)
                    for (key, record), lease, payload in zip(pending.items(), owned, payloads)]
                # Either every new source becomes visible or all new rights
                # are discarded. Previously prepared sources remain untouched.
                for source in prepared:
                    installed.append(source)
                    self.sources[source.key] = source
                    self.by_handle[id(source.handle)] = source
            owned = []  # Cache now owns the admitted leases, without cloning.
        except BaseException:
            for source in installed:
                self.sources.pop(source.key, None)
                self.by_handle.pop(id(source.handle), None)
            raise
        finally:
            for lease in owned:
                if lease is not None:
                    lease.reset()
            # Tracebacks must not retain a second list of large CPU payloads.
            payloads.clear()
            prepared.clear()
            installed.clear()
            source = None
        return [self.sources[key].payload for key in requested], len(pending)

    def read(self, authority, handles, *, destination, region=None, regions=None,
             inline_cpu=False, paired_native=False):
        """Synchronous packed-reader callback; no fallback to a different path."""
        active = False
        failure = False
        sources = []
        try:
            if authority is not self.authority:
                raise ValueError('Lazy Query reader received a foreign authority')
            if type(inline_cpu) is not bool or paired_native is not False:
                raise ValueError('Lazy Query reads use the qualified ordinary region gather')
            handles = tuple(handles)
            if not handles or (region is None) == (regions is None):
                raise ValueError('Lazy Query reads require handles and one or paired regions')
            selected_regions = (tuple(region),) if region is not None else tuple(tuple(r) for r in regions)
            if (len(selected_regions) != (1 if region is not None else 2) or
                    any(len(r) != 2 for r in selected_regions) or
                    any(type(offset) is not int or type(length) is not int or offset < 0 or length <= 0 or
                        offset + length > self.record_bytes for offset, length in selected_regions)):
                raise ValueError('Invalid lazy Query region extent')
            row_bytes = sum(length for _, length in selected_regions)
            size = len(handles) * row_bytes
            if (not isinstance(destination, torch.Tensor) or
                    destination.device.type != 'cpu' or destination.dtype != torch.uint8 or
                    destination.ndim != 1 or not destination.is_contiguous()):
                raise ValueError('Lazy Query destination must be contiguous CPU bytes')
            target = read_destination(destination.numpy(), size)
            prepared_at = time.perf_counter()
            with self._lock:
                self.validate()
                self.active += 1
                active = True
                sources, new_blocks = self._ensure_sources(handles)
            prepare_seconds = time.perf_counter() - prepared_at
            deadline, waited = time.monotonic() + 180, 0.
            archive = self.archive
            while True:
                with self._lock:
                    self.validate()
                    with archive._lock:
                        admitted = archive._admit(size)
                if admitted:
                    break
                wait_started = time.monotonic()
                remaining = deadline - wait_started
                if remaining <= 0 or not archive.wait_for_idle(remaining):
                    raise TimeoutError('Lazy Query CPU read admission did not drain')
                waited += time.monotonic() - wait_started
            try:
                started = time.perf_counter()
                cursor = tasks = 0
                for offset, length in selected_regions:
                    end = cursor + len(sources) * length
                    tasks += archive._gather_region(target[cursor:end], sources, offset, length)
                    cursor = end
                copied_seconds = time.perf_counter() - started
            finally:
                # Native gather joins all parallel producers before returning,
                # including when a copy or worker submission fails.
                sources.clear()
                with archive._lock:
                    archive._complete(size)
            with self._lock:
                self.validate()
                self.read_chunks += 1
                reused = len(handles) - new_blocks
                self.reused_blocks += reused
                total_sources = self.source_blocks
            tensor = torch.frombuffer(target, dtype=torch.uint8)
            if tensor.data_ptr() != destination.data_ptr():
                raise ValueError('Lazy Query read changed its staging destination')
            stats = dict(bytes=0, syscalls=0, logical_bytes=size, cache_hit_bytes=size,
                cache_gather_tasks=tasks, cache_copy_seconds=copied_seconds,
                read_queue_wait_seconds=0., native_read_call_seconds=0., cache_fill_seconds=0.,
                admission_wait_seconds=waited, region_bytes=row_bytes,
                inline_cpu_region_read=True, prepared_cpu_region_read=True,
                query_lazy_region_read=True, query_reader_new_blocks=new_blocks,
                query_reader_reused_blocks=reused, query_reader_source_blocks=total_sources,
                query_reader_prepare_seconds=prepare_seconds)
            if region is not None:
                stats['region_offset'] = selected_regions[0][0]
            else:
                stats.update(regions=selected_regions, region_read_count=2)
            return tensor, stats
        except BaseException:
            failure = True
            raise
        finally:
            sources.clear()
            with self._lock:
                if active:
                    self.active -= 1
                if failure:
                    self.failed = True
                close_failed = self.failed and not self.active
            if close_failed:
                self.close()

    def close(self):
        with self._lock:
            if self.closed:
                return True
            if self.active:
                return False
            with self.archive._lock:
                for source in self.sources.values():
                    source.lease.reset()
                self.sources.clear()
                self.by_handle.clear()
                self.admission_rights.clear()
                self.closed = True
                if self.registered:
                    self.archive._prepared_region_readers -= 1
                    self.registered = False
                    self.archive._capacity_changed.notify_all()
            return True

    def __enter__(self):
        try:
            self.validate()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, kind, error, traceback):
        if not self.close():
            raise RuntimeError('Lazy Query reader still has an active CPU producer')
