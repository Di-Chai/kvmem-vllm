"""Opt-in host-side allocator observations, outside model compilation/graphs.

Only isolated diagnostic overlays install this helper. Nothing synchronizes CUDA,
resets allocator peaks, forces publication, or changes tensor expressions. Host
boundaries correlate allocation requests; they are not GPU execution timestamps.
"""
from collections import Counter
from pathlib import Path
from threading import get_ident
import json
import os
import time


FIELDS = tuple(f'{name}.all.{suffix}' for name in (
    'allocated_bytes', 'reserved_bytes', 'active_bytes', 'inactive_split_bytes',
    'requested_bytes') for suffix in ('current', 'peak')) + (
    'num_alloc_retries', 'num_ooms', 'num_sync_all_streams',
    'num_device_alloc', 'num_device_free')
PHASES = {'cold_prepare', 'warmup', 'formal', 'drain', 'cleanup', 'unknown'}


class TorchAllocator:
    def __init__(self, device):
        import torch
        self.cuda, self.device = torch.cuda, device

    def capturing(self):
        return self.cuda.is_current_stream_capturing()

    def statistics(self):
        return self.cuda.memory_stats(self.device)

    def identity(self):
        return dict(backend=self.cuda.get_allocator_backend(), device=str(self.device),
                    PYTORCH_ALLOC_CONF=os.getenv('PYTORCH_ALLOC_CONF'),
                    PYTORCH_CUDA_ALLOC_CONF=os.getenv('PYTORCH_CUDA_ALLOC_CONF'))


class BoundDiagnostic:
    def __init__(self, diagnostic, fields):
        self.diagnostic, self.fields = diagnostic, fields

    def child(self, **fields):
        return BoundDiagnostic(self.diagnostic, {**self.fields, **fields})

    def mark(self, stage, **fields):
        self.diagnostic.mark(stage, **{**self.fields, **fields})


class AllocatorDiagnostic:
    def __init__(self, output, backend, *, phase_file=None, max_records=200000,
                 buffer_records=2048, decode_interval=128, clock=time.monotonic,
                 wall_clock=time.time):
        if any(type(n) is not int or n <= 0 for n in (max_records, buffer_records, decode_interval)):
            raise ValueError('Allocator diagnostic bounds must be positive integers')
        self.output, self.backend = Path(output), backend
        self.phase_file = Path(phase_file) if phase_file is not None else None
        self.max_records, self.buffer_records, self.decode_interval = max_records, buffer_records, decode_interval
        self.clock, self.wall_clock, self.thread = clock, wall_clock, get_ident()
        self.sequence = self.written = self.retained = 0
        self.current = None
        self.buffer = []
        self.counts = Counter()
        self.coverage = Counter()
        self.stages = Counter()
        self.identity = None
        self.sampling_host_seconds = 0.

    def _engine(self):
        if get_ident() != self.thread:
            raise RuntimeError('Allocator diagnostics must stay on the engine thread')

    def _phase(self):
        if self.phase_file is None:
            return 'unknown'
        self.counts['phase_reads'] += 1
        try:
            with self.phase_file.open() as stream:
                content = stream.read(4097)
            if len(content) > 4096:
                raise ValueError('Phase marker exceeds bound')
            value = json.loads(content)
            phase = value.get('phase') if isinstance(value, dict) else None
            if phase not in PHASES:
                raise ValueError('Unknown diagnostic phase')
            return phase
        except (OSError, ValueError, TypeError):
            self.counts['phase_unknown'] += 1
            return 'unknown'

    def _append(self, row):
        if self.retained >= self.max_records:
            self.counts['record_limit_drops'] += 1
            return False
        if len(self.buffer) >= self.buffer_records:
            self.counts['buffer_limit_drops'] += 1
            return False
        self.buffer.append(row)
        self.retained += 1
        return True

    def begin_batch(self, batch, *, sequence=None, offsets=None):
        self._engine()
        # Never poll allocator/capture events or touch the filesystem in a CUDA
        # graph. This check itself only asks whether this host stream captures.
        if self.backend.capturing():
            self.counts['graph_batch_skips'] += 1
            self.current = None
            return
        self.flush()
        if self.current is not None:
            self.counts['unfinished_batches'] += 1
        self.sequence += 1
        flags = [bool(v) for v in batch.is_prefilling_np]
        kind = 'prefill' if any(flags) else 'decode'
        selected = kind == 'prefill' or (self.sequence-1) % self.decode_interval == 0
        self.coverage[kind + '_batches_seen'] += 1
        self.current = None
        if not selected:
            self.coverage[kind + '_interval_skips'] += 1
            return
        if self.identity is None:
            self.identity = self.backend.identity()
        phase = self._phase()
        self.current = dict(sequence=self.sequence, step_timing_sequence=sequence,
                            phase=phase, kind=kind)
        self.coverage[kind + '_batches_selected'] += 1
        self.coverage[phase + '/' + kind + '_batches_selected'] += 1
        ids = list(batch.req_ids)
        scheduled = [int(v) for v in batch.num_scheduled_tokens]
        history = [int(v) for v in batch.num_computed_tokens_np]
        offsets = offsets or {}
        retained = self._append(dict(event='batch', **self.current, host_monotonic=self.clock(), host_wall=self.wall_clock(),
            request_ids=ids, is_prefilling=flags, num_requests=int(batch.num_reqs),
            num_tokens=int(batch.num_tokens), num_tokens_after_padding=int(batch.num_tokens_after_padding),
            scheduled_tokens_cpu_upper_bound=scheduled, history_cpu_upper_bound=history,
            local_context_cpu_upper_bound=[h+n-offsets.get(rid, 0) for rid,h,n in zip(ids,history,scheduled)],
            token_semantics='Native batch counts/padding. CPU per-request counts may be speculative upper bounds; '
                'accepted tokens require existing publication feedback, not a new GPU readback.'))
        self.coverage[kind + ('_batch_headers_retained' if retained else '_batch_headers_dropped')] += 1

    def child(self, **fields):
        return BoundDiagnostic(self, fields)

    def mark(self, stage, **fields):
        self._engine()
        if self.current is None:
            return
        if self.backend.capturing():
            self.counts['graph_stage_skips'] += 1
            return
        self.stages[stage] += 1
        kind = self.current['kind']
        if self.retained >= self.max_records or len(self.buffer) >= self.buffer_records:
            self.counts['record_limit_drops' if self.retained >= self.max_records else 'buffer_limit_drops'] += 1
            self.coverage[kind + '_snapshots_dropped'] += 1
            self.coverage[self.current['phase'] + '/' + kind + '_snapshots_dropped'] += 1
            return
        began = self.clock()
        statistics = self.backend.statistics()  # Exactly one host allocator query.
        ended = self.clock()
        self.sampling_host_seconds += ended-began
        self._append(dict(event='allocator', **{**self.current, **fields}, stage=stage,
            host_monotonic=began, host_wall=self.wall_clock(), sampling_host_seconds=ended-began,
            memory={name: statistics.get(name) for name in FIELDS}))
        self.coverage[kind + '_snapshots_retained'] += 1

    def end_batch(self):
        self._engine()
        if self.backend.capturing():
            self.counts['graph_end_skips'] += 1
            return
        if self.current is not None:
            self.mark('public_batch.end')
            self.coverage[self.current['kind'] + '_batches_completed'] += 1
            self.current = None
        self.flush()

    def summary(self):
        return dict(event='summary', counts=dict(self.counts), coverage=dict(self.coverage),
            stages=dict(self.stages), identity=self.identity, retained=self.retained,
            written=self.written, buffered=len(self.buffer), max_records=self.max_records,
            buffer_records=self.buffer_records, decode_interval=self.decode_interval,
            sampling_host_seconds=self.sampling_host_seconds,
            semantics='Host allocation-submission boundaries only. Peaks are process lifetime high-water values '
                'and were never reset; their increments are not exclusive stage costs. PyTorch reserved '
                'differs from NVIDIA driver reserved. Some counters are unavailable for cudaMallocAsync.')

    def flush(self, *, force=False):
        self._engine()
        if self.backend.capturing():
            self.counts['graph_flush_skips'] += 1
            return
        if not force and (not self.buffer or len(self.buffer) < self.buffer_records//2):
            return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open('a') as stream:
            stream.writelines(json.dumps(row)+'\n' for row in self.buffer)
            self.written += len(self.buffer)
            self.buffer.clear()
            stream.write(json.dumps(self.summary())+'\n')


def install(output, device, *, backend=None):
    enabled = os.getenv('KVMEM_ALLOCATOR_DIAGNOSTIC', '0')
    if enabled not in ('0', '1'):
        raise ValueError('KVMEM_ALLOCATOR_DIAGNOSTIC must be 0 or 1')
    if enabled == '0':
        return None
    output = Path(output)
    phase_file = os.getenv('KVMEM_ALLOCATOR_DIAGNOSTIC_PHASE_FILE', str(output/'allocator_phase.json'))
    return AllocatorDiagnostic(output/'allocator_diagnostic.jsonl', backend or TorchAllocator(device), phase_file=phase_file)


def observe(runner, stage, **fields):
    diagnostic = getattr(runner, '_kvmem_allocator_diagnostic', None)
    if diagnostic is not None:
        diagnostic.mark(stage, **fields)
