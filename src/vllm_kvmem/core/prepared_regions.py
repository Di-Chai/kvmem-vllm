"""Transaction-local immutable CPU sources; no extra KV payload allocation."""
import time

from .storage import read_destination


class PreparedRegionReader:
    """Sources and generation leases survive until every synchronous copy drains.

    Construction is owned by MemoryArchive under its admission lock. The
    caller closes explicitly; archive shutdown refuses a live reader. Native
    payload/GC archives retain their existing per-read protocol, except under
    KVMEM_LAYERWISE_HISTORICAL_GC: there each source is a native read right
    (``HistoricalPayloadRead``) held until close, which delays retirement of
    exactly those payloads and nothing else.
    """
    def __init__(self, archive, namespace, keys, sources, leases):
        self.archive, self.namespace = archive, namespace
        self.keys, self.sources, self.leases = tuple(keys), tuple(sources), leases
        self.closed, self.active = False, 0
        self.extra_leases = []

    def extend(self, keys, sources, leases):
        """KVMEM_PRESSURE_READER_EXTEND: append further immutable sources and
        adopt their pins (a native LeaseSet) until close. Existing indices and
        sources are unchanged; returns the index of the first new source.
        Call under the archive lock, as construction is."""
        if self.closed:
            raise RuntimeError('Prepared CPU reader is closed')
        native = getattr(self, 'native', None)
        if native is None:
            raise RuntimeError('Prepared reader extension requires native sources')
        base = native.extend(list(sources))      # validates every source before appending
        if base != len(self.keys):
            raise RuntimeError('Prepared reader sources and keys diverged')
        self.keys += tuple(keys)
        self.sources += tuple(sources)
        self.extra_leases.append(leases)
        return base

    def read(self, indices, regions, *, destination, paired_native=False, paired_workers=1, parallel=False):
        archive = self.archive
        native = getattr(self, 'native', None) if not paired_native else None
        indices = tuple(indices)
        regions = tuple(tuple(r) for r in regions)
        if native is not None:
            native.check(list(indices))   # same rule and message, without a Python loop
        if (not indices or len(regions) != 2 or
                (native is None and any(type(i) is not int or not 0 <= i < len(self.keys) for i in indices)) or
                any(len(r) != 2 for r in regions) or
                any(type(o) is not int or type(n) is not int or o < 0 or n <= 0 or
                    o+n > archive.record_bytes for o,n in regions)):
            raise ValueError('Invalid prepared CPU region extent or source index')
        if type(paired_native) is not bool:
            raise ValueError('Prepared paired native gather flag must be boolean')
        if (type(paired_workers) is not int or paired_workers not in (1,2) or
                paired_workers != 1 and not paired_native):
            raise ValueError('Prepared paired workers require the explicit paired route')
        paired_eligible = False
        if paired_native:
            from .. import _native
            from .memory_storage import region_gather_parallel_min_bytes
            CPU_REGION_GATHER_PARALLEL_MIN_BYTES = region_gather_parallel_min_bytes()
            from ..serving.pressure_reader import layerwise_gc
            if (not archive.memory_only or
                    (getattr(archive, '_native_payload_arena', None) is not None and not layerwise_gc())):
                # KVMEM_LAYERWISE_HISTORICAL_GC: the sources are read rights
                # held by this reader, so their bytes stay immutable here.
                raise ValueError('Prepared paired gather requires no-GC CPU authority')
            # Preserve the existing gather thresholds. Full-record fields and
            # any parallel-size region continue through the original loop.
            paired_eligible = all(length != archive.record_bytes and
                len(indices)*length < CPU_REGION_GATHER_PARALLEL_MIN_BYTES for _,length in regions)
            if paired_eligible and not callable(getattr(_native, 'gather_record_regions', None)):
                raise RuntimeError('Prepared paired gather requires the updated native extension')
        row_bytes = sum(n for _,n in regions)
        size = len(indices)*row_bytes
        target = read_destination(destination, size)
        if target is None:
            raise ValueError('Prepared CPU reads require owned staging')
        with archive._lock:
            if self.closed:
                raise RuntimeError('Prepared CPU reader is closed')
            if not archive._admit(size):
                return None
            try:
                sources = [] if native is not None else [self.sources[i] for i in indices]
            except BaseException:
                archive._complete(size)
                raise
            self.active += 1
        started = time.perf_counter()
        try:
            native_stats = None
            if paired_eligible:
                native_stats = archive._gather_regions_paired(target, sources, regions,
                    **({'workers':paired_workers} if paired_workers != 1 else {}))
                tasks = native_stats.get('worker_count',1)
            elif native is not None:
                cursor = tasks = 0
                listed = list(indices)
                for offset,length in regions:
                    end = cursor+len(indices)*length
                    tasks += native.gather(listed, offset, length, target[cursor:end], parallel)
                    cursor = end
            else:
                cursor = tasks = 0
                for offset,length in regions:
                    end = cursor+len(sources)*length
                    tasks += archive._gather_region(target[cursor:end], sources, offset, length)
                    cursor = end
            stats = dict(bytes=0, syscalls=0, logical_bytes=size, cache_hit_bytes=size,
                cache_gather_tasks=tasks, cache_copy_seconds=time.perf_counter()-started,
                read_queue_wait_seconds=0., native_read_call_seconds=0., cache_fill_seconds=0.,
                regions=regions, region_bytes=row_bytes, region_read_count=2,
                inline_cpu_region_read=True, prepared_cpu_region_read=True)
            if paired_native:
                stats.update(paired_native_gather=paired_eligible,
                    paired_native_fallback_reason=None if paired_eligible else 'parallel_or_full_record_region')
                if native_stats is not None:
                    stats.update({'paired_native_'+key:value for key,value in native_stats.items()})
            return target, stats
        finally:
            # _gather_region drains its parallel producers even on failure.
            sources.clear()
            with archive._lock:
                self.active -= 1
                archive._complete(size)

    def close(self):
        with self.archive._lock:
            if self.closed:
                return True
            if self.active:
                return False
            if hasattr(self.leases, 'valid'):
                self.leases.reset()      # native LeaseSet: releases every pin
            else:
                for lease in self.leases:
                    lease.reset()
            self.leases.clear()
            for extra in getattr(self, 'extra_leases', ()):
                extra.reset()
            self.extra_leases = []
            self.sources = ()
            self.native = None      # release the native source buffer exports
            self.closed = True
            self.archive._prepared_region_readers -= 1
            self.archive._capacity_changed.notify_all()
            return True
