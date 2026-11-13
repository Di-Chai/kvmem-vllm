"""One pressure plan's deduplicated, generation-bound immutable CPU reader."""
import os
import time

import torch

from ..core.memory import Phase
from ..core import settings as _settings


# Defined in core so that the store and archive can read them (layering step 7).
from ..core.archive_flags import layerwise_gc, native_reader  # noqa: E402,F401


def layer0_parallel_gather():
    """KVMEM_PRESSURE_LAYER0_PARALLEL_GATHER: parallel copies for layer 0 of a
    promised refresh only, while the next forward is blocked at its guard."""
    value = _settings.get('KVMEM_PRESSURE_LAYER0_PARALLEL_GATHER', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_LAYER0_PARALLEL_GATHER must be 0 or 1')
    return value == '1'


def reader_adopt():
    value = _settings.get('KVMEM_PRESSURE_READER_ADOPT', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_READER_ADOPT must be 0 or 1')
    return value == '1'


def reader_extend():
    """KVMEM_PRESSURE_READER_EXTEND: with lazy layer-0-first selection the
    prepared reader covers layer 0's records; add each later layer's records
    before its upload instead of reading them through the per-chunk path."""
    value = _settings.get('KVMEM_PRESSURE_READER_EXTEND', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_READER_EXTEND must be 0 or 1')
    return value == '1'


class PressureReader:
    def __init__(self, authority, plan, layers):
        paired = _settings.get('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER', '0')
        if paired not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER must be 0 or 1')
        self.paired_gather = paired == '1'
        workers = _settings.get('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS', '1')
        if workers not in ('1','2') or workers != '1' and not self.paired_gather:
            raise ValueError('Pressure paired workers require 1 or 2 and an enabled paired route')
        self.paired_workers = int(workers)
        self.read_stats = (dict(pressure_paired_gather_requested=True,
            pressure_paired_gather_chunks=0, pressure_paired_fallback_chunks=0,
            pressure_paired_copied_bytes=0, pressure_paired_workers_requested=self.paired_workers,
            pressure_paired_inline_chunks=0, pressure_paired_parallel_chunks=0,
            pressure_paired_native_calls=0, pressure_paired_gather_wall_seconds=0.,
            pressure_paired_native_timing_scope='sum_of_native_calls_not_elapsed_wall',
            **{'pressure_paired_'+key+'_seconds':0. for key in
               ('source_prepare','copy','gil_release','gil_reacquire','total')})
            if self.paired_gather else {})
        self.authority, self.plan = authority, plan
        self.namespace = authority.memory.request_id
        self.history_end = authority.history_end
        self.closed = False
        self.records = {}
        leases, keys, seen_keys = [], [], set()
        from ..core import publication_trace
        _traced = publication_trace.enabled()
        _t0 = time.perf_counter()
        authority.poll()
        _t1 = time.perf_counter()
        visits = 0
        accepted = self.records
        block = authority.shape[1]
        try:
            if _settings.get('KVMEM_PRESSURE_READER_BATCH', '0') == '1':
                visits = sum(len(layer) for layer in layers)
                self._collect_batched(authority, layers, block, leases, keys)
            for layer in (() if self.records else layers):
                visits += len(layer)
                for record in layer:
                    # Same predicate as before, cheapest term first: layers
                    # share most records, and count/archived are plain fields.
                    if id(record) in accepted or record.count != block or not record.archived:
                        continue
                    handle = record.handle
                    key = handle.slot, handle.generation
                    if key in seen_keys or authority.records.get(key) is not record:
                        raise ValueError('Pressure sources have aliased or foreign generations')
                    lease, source = authority.memory.admitted_read(handle)
                    if not lease:
                        raise ValueError('Stale pressure source generation')
                    leases.append(lease)
                    if source is not None:
                        self._keep_admission_right(source)
                    seen_keys.add(key)
                    self.records[id(record)] = (record, handle, record.history_begin, len(keys))
                    keys.append(key)
            self.validate()
            _t2 = time.perf_counter()
            lease_set = getattr(self, 'lease_set', None)
            if lease_set is not None:
                # One native LeaseSet: adopted by the reader exactly like the list.
                self.lease_set = None
                leases = [lease_set]          # reset by finally if adoption fails
                self.reader = authority.archive.prepare_region_reader(self.namespace, lease_set,
                    keys=[(self.namespace, slot, generation) for slot, generation in keys], adopt=True)
                leases = []
            elif leases and reader_adopt():
                # Same sources and leases: pass the validated keys, transfer ownership.
                self.reader = authority.archive.prepare_region_reader(self.namespace, leases,
                    keys=[(self.namespace, slot, generation) for slot, generation in keys], adopt=True)
                leases = []      # owned by the reader now; never reset here
            else:
                self.reader = (authority.archive.prepare_region_reader(self.namespace, leases) if leases else None)
            _t3 = time.perf_counter()
            self.source_blocks = len(keys)
            self.read_chunks = 0
            self.native = False
            if self.reader is not None and native_reader() and not self.paired_gather:
                # Opt-in: export the immutable sources once; reads then copy
                # with the GIL released and validate records natively.
                from .. import _native
                self.reader.native = _native.RecordSources(list(self.reader.sources),
                                                           authority.archive.record_bytes)
                self.native = True
        finally:
            for lease in leases:
                lease.reset()
        if _traced:
            _t4 = time.perf_counter()
            publication_trace.emit('reader_prepare_sections', visits=visits, unique=len(keys),
                poll=_t1-_t0, record_loop=_t2-_t1, region_reader=_t3-_t2, lease_reset=_t4-_t3)

    def _collect_batched(self, authority, layers, block, leases, keys):
        """Same records, order, checks and leases as the per-record loop.

        First-visit order is a dict's insertion order; the qualification is a
        pure function of each record, and admission is one memory call.
        """
        table = None
        if native_reader():
            from .. import _native
            collect = getattr(_native, 'collect_pressure_sources', None)
            if collect is not None:
                # Same records, order, checks and table, without Python per-record work.
                records, handles, pairs, table = collect(layers, block, authority.records)
        if table is None:
            unique = {}
            for layer in layers:
                unique.update(zip(map(id, layer), layer))
            records = [r for r in unique.values() if r.count == block and r.archived]
        if not records:
            return
        if table is None:
            handles = [r.handle for r in records]
            pairs = [(h.slot, h.generation) for h in handles]
            get = authority.records.get
            if len(set(pairs)) != len(pairs) or any(get(k) is not r for k, r in zip(pairs, records)):
                raise ValueError('Pressure sources have aliased or foreign generations')
        if table is not None and reader_adopt():
            # Native reader + adoption: the pins stay in one native LeaseSet.
            pin_set = getattr(authority.memory, 'admitted_pin_set', None)
            lease_set = pin_set(handles) if pin_set is not None else None
            if lease_set is not None:
                if not lease_set.valid():
                    lease_set.reset()
                    raise ValueError('Stale pressure source generation')
                self.lease_set = lease_set
                self.records = table
                keys.extend(pairs)
                return
        batch = getattr(authority.memory, 'admitted_read_batch', None)
        admitted = (batch(handles) if batch is not None else
                    [authority.memory.admitted_read(handle) for handle in handles])
        leases.extend(lease for lease, _ in admitted)
        if not all(lease for lease, _ in admitted):
            raise ValueError('Stale pressure source generation')
        for _, source in admitted:
            if source is not None:
                self._keep_admission_right(source)
        self.records = table if table is not None else {id(r): (r, h, r.history_begin, i)
                        for i, (r, h) in enumerate(zip(records, handles))}
        keys.extend(pairs)

    def validate(self):
        memory = self.authority.memory
        materialization = getattr(memory, '_layerwise_materialization', None)
        original = memory.phase is Phase.ASSEMBLING and memory._plan is self.plan
        pipelined = (materialization is not None and materialization.plan is self.plan and
                     materialization.job.authority is self.authority)
        if (self.closed or not (original or pipelined) or
                memory.request_id != self.namespace or self.authority.history_end != self.history_end):
            raise ValueError('Prepared pressure reader belongs to a different plan, owner or prefix')

    def read(self, records, regions, *, destination, parallel=False):
        self.validate()
        if self.reader is None:
            raise ValueError('Pressure plan has no complete immutable sources')
        if getattr(self, 'native', False):
            from .. import _native
            indices = _native.prepared_record_indices(records, self.records, self.authority.shape[1])
        else:
            indices = []
            for record in records:
                entry = self.records.get(id(record))
                if (entry is None or entry[0] is not record or record.handle is not entry[1] or
                        record.history_begin != entry[2] or record.count != self.authority.shape[1] or
                        not record.archived):
                    raise ValueError('Prepared pressure source is absent, partial or changed')
                indices.append(entry[3])
        if (destination.device.type != 'cpu' or destination.dtype != torch.uint8 or
                destination.ndim != 1 or not destination.is_contiguous()):
            raise ValueError('Prepared pressure destination must be contiguous CPU bytes')
        target = destination.numpy()
        deadline = time.monotonic()+180
        waited = 0.
        while True:
            self.validate()
            result = self.reader.read(indices, regions, destination=target,
                **({'parallel': True} if parallel and getattr(self, 'native', False) else {}),
                **({'paired_native': True} if self.paired_gather else {}),
                **({'paired_workers': self.paired_workers} if self.paired_workers != 1 else {}))
            if result is not None:
                break
            start = time.monotonic()
            remaining = deadline-start
            if remaining <= 0 or not self.authority.archive.wait_for_idle(remaining):
                raise TimeoutError('Prepared pressure read admission did not drain')
            waited += time.monotonic()-start
        data,stats = result
        stats['admission_wait_seconds'] = waited
        tensor = torch.frombuffer(data, dtype=torch.uint8)
        if tensor.data_ptr() != destination.data_ptr():
            raise ValueError('Prepared pressure read changed its staging destination')
        self.read_chunks += 1
        if self.paired_gather:
            applied = bool(stats.get('paired_native_gather', False))
            self.read_stats['pressure_paired_gather_chunks'] += int(applied)
            self.read_stats['pressure_paired_fallback_chunks'] += int(not applied)
            self.read_stats['pressure_paired_copied_bytes'] += stats.get('paired_native_copied_bytes', 0)
            actual_workers = stats.get('paired_native_worker_count', 1) if applied else 0
            self.read_stats['pressure_paired_native_calls'] += actual_workers
            self.read_stats['pressure_paired_inline_chunks'] += int(actual_workers == 1)
            self.read_stats['pressure_paired_parallel_chunks'] += int(actual_workers > 1)
            self.read_stats['pressure_paired_gather_wall_seconds'] += stats.get('paired_native_gather_wall_ns',0)/1e9
            for key in ('source_prepare','copy','gil_release','gil_reacquire','total'):
                self.read_stats['pressure_paired_'+key+'_seconds'] += stats.get('paired_native_'+key+'_ns',0)/1e9
        return tensor,stats

    def extend(self, records):
        """Prepare a lazily selected layer's records (KVMEM_PRESSURE_READER_EXTEND).

        Same qualification, alias/foreign checks, pins, entry checks and source
        bytes as construction; existing indices are unchanged. Returns False and
        leaves the reader unchanged whenever that is not possible: the caller's
        original per-chunk read path then handles those records (and raises its
        own original errors), exactly as without this option.
        """
        if self.closed or self.reader is None or not getattr(self, 'native', False):
            return False
        self.validate()
        fresh = [record for record in records if id(record) not in self.records]
        if not fresh:
            return True
        from .. import _native
        collect = getattr(_native, 'collect_pressure_sources', None)
        pin_set = getattr(self.authority.memory, 'admitted_pin_set', None)
        extend = getattr(self.authority.archive, 'extend_region_reader', None)
        if collect is None or pin_set is None or extend is None:
            return False
        try:
            collected, handles, pairs, table = collect([fresh], self.authority.shape[1], self.authority.records)
        except ValueError:
            return False
        if not collected:
            return True
        lease_set = pin_set(handles)
        if lease_set is None:
            return False
        try:
            if not lease_set.valid():
                raise ValueError('Stale pressure source generation')
            base = extend(self.reader, self.namespace, lease_set,
                          [(self.namespace, slot, generation) for slot, generation in pairs])
        except Exception:
            lease_set.reset()
            return False
        for record, handle, begin, index in table.values():
            self.records[id(record)] = (record, handle, begin, base + index)
        self.source_blocks += len(pairs)
        self.extended_blocks = getattr(self, 'extended_blocks', 0) + len(pairs)
        return True

    def contains(self, records):
        self.validate()
        if getattr(self, 'native', False):
            from .. import _native
            return _native.prepared_contains(records, self.records)
        return all(id(record) in self.records for record in records)

    def __enter__(self):
        try:
            self.validate()
        except BaseException:
            self.close()
            raise
        return self

    def _keep_admission_right(self, source):
        # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: native payload admission hands
        # back a GC read right with each lease; hold it with the plan (the
        # prepared reader's own sources are separate read rights).
        if not layerwise_gc():
            raise ValueError('Prepared pressure reads cannot bypass native payload admission')
        self.__dict__.setdefault('admission_rights', []).append(source)

    def close(self):
        if self.reader is not None and not self.reader.close():
            return False
        self.closed = True
        self.__dict__.pop('admission_rights', None)
        return True

    def __exit__(self, kind, error, traceback):
        if not self.close():
            raise RuntimeError('Pressure source reader still has an active CPU producer')


def prepare_pressure_reader(authority, plan, layers):
    archive = authority.archive
    # Independent native CPU retirement needs the existing per-copy source
    # rights. Do not silently broaden this no-GC optimization to that backend,
    # except under KVMEM_LAYERWISE_HISTORICAL_GC, where the reader holds those
    # rights itself (one per source, for the plan). Repository GC stays excluded.
    from ..core.historical_gc import HistoricalAuthorityGc
    gc = getattr(authority, 'gc', None)
    historical = (layerwise_gc() and type(gc) is HistoricalAuthorityGc and
                  getattr(archive, '_native_payload_arena', None) is not None)
    if (not getattr(archive, 'memory_only', False) or
            (gc is not None and not historical) or
            (getattr(authority.memory, '_history_scope', None) is not None and not historical) or
            (getattr(archive, '_native_payload_arena', None) is not None and not historical) or
            not hasattr(archive, 'prepare_region_reader')):
        return None
    return PressureReader(authority, plan, layers)
