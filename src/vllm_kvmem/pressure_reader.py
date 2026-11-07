"""One pressure plan's deduplicated, generation-bound immutable CPU reader."""
import os
import time

import torch

from .memory import Phase


class PressureReader:
    def __init__(self, authority, plan, layers):
        paired = os.getenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER', '0')
        if paired not in ('0', '1'):
            raise ValueError('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER must be 0 or 1')
        self.paired_gather = paired == '1'
        workers = os.getenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS', '1')
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
        authority.poll()
        try:
            for layer in layers:
                for record in layer:
                    if (record.count != authority.shape[1] or not record.archived or
                            id(record) in self.records):
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
                        raise ValueError('Prepared pressure reads cannot bypass native payload admission')
                    seen_keys.add(key)
                    self.records[id(record)] = (record, handle, record.history_begin, len(keys))
                    keys.append(key)
            self.validate()
            self.reader = (authority.archive.prepare_region_reader(self.namespace, leases) if leases else None)
            self.source_blocks = len(keys)
            self.read_chunks = 0
        finally:
            for lease in leases:
                lease.reset()

    def validate(self):
        memory = self.authority.memory
        materialization = getattr(memory, '_layerwise_materialization', None)
        original = memory.phase is Phase.ASSEMBLING and memory._plan is self.plan
        pipelined = (materialization is not None and materialization.plan is self.plan and
                     materialization.job.authority is self.authority)
        if (self.closed or not (original or pipelined) or
                memory.request_id != self.namespace or self.authority.history_end != self.history_end):
            raise ValueError('Prepared pressure reader belongs to a different plan, owner or prefix')

    def read(self, records, regions, *, destination):
        self.validate()
        if self.reader is None:
            raise ValueError('Pressure plan has no complete immutable sources')
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

    def contains(self, records):
        self.validate()
        return all(id(record) in self.records for record in records)

    def __enter__(self):
        try:
            self.validate()
        except BaseException:
            self.close()
            raise
        return self

    def close(self):
        if self.reader is not None and not self.reader.close():
            return False
        self.closed = True
        return True

    def __exit__(self, kind, error, traceback):
        if not self.close():
            raise RuntimeError('Pressure source reader still has an active CPU producer')


def prepare_pressure_reader(authority, plan, layers):
    archive = authority.archive
    # Independent native CPU retirement needs the existing per-copy source
    # rights. Do not silently broaden this no-GC optimization to that backend.
    if (not getattr(archive, 'memory_only', False) or
            getattr(authority, 'gc', None) is not None or
            getattr(authority.memory, '_history_scope', None) is not None or
            getattr(archive, '_native_payload_arena', None) is not None or
            not hasattr(archive, 'prepare_region_reader')):
        return None
    return PressureReader(authority, plan, layers)
