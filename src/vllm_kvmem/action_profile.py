"""Opt-in Action diagnostics. Events are polled, never synchronized here.

Host scopes are inclusive and may overlap CUDA execution. Pending event pairs
are bounded; dropping a diagnostic never drops or delays attention scoring.
"""
from collections import deque
from contextlib import contextmanager
import json
import time


class ActionProfileContext:
    def __init__(self, owner, batch, metadata=None):
        self.owner, self.batch = owner, batch
        self.metadata = metadata or {}

    def child(self, **metadata):
        return ActionProfileContext(self.owner, self.batch, {**self.metadata, **metadata})

    def stage(self, name, *, gpu=False, **metadata):
        return self.owner.stage(self.batch, name, gpu=gpu, **{**self.metadata, **metadata})


class ActionProfiler:
    def __init__(self, output, *, interval=1, max_events=256, event_factory=None,
                 capturing=None, clock=None, thread_clock=None):
        if type(interval) is not int or interval < 1 or type(max_events) is not int or max_events < 1:
            raise ValueError('Action profiling interval and event capacity must be positive integers')
        if event_factory is None or capturing is None:
            import torch
            event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
            capturing = capturing or torch.cuda.is_current_stream_capturing
        self.output, self.interval, self.max_events = output, interval, max_events
        self.event_factory, self.capturing = event_factory, capturing
        self.clock, self.thread_clock = clock or time.perf_counter, thread_clock or time.thread_time
        self.current = None
        self.cuda_seen = False
        self.pending = deque()
        self.reserved_events = 0
        self.sequence = self.scope_sequence = 0
        self.totals = {}
        self.counters = dict(batches_seen=0, batches_sampled=0, interval_skips=0,
            graph_capture_skips=0, gpu_event_drops=0, gpu_events_completed=0)

    def begin_batch(self, batch):
        cuda = batch.input_ids.device.type == 'cuda'
        self.cuda_seen = self.cuda_seen or cuda
        if cuda and self.capturing():
            self.counters['graph_capture_skips'] += 1
            return None
        if self.current is not None and self.current['source'] is batch:
            return self.current['context']
        if self.current is not None:
            self.finish_batch(status='superseded')
        self.poll()
        self.counters['batches_seen'] += 1
        self.sequence += 1
        record = dict(batch_id=self.sequence, started_at=self.clock(), cuda=cuda,
            requests=len(batch.req_ids), host_stages=[], source=batch)
        flags = getattr(batch, 'is_prefilling_np', None)
        record['request_kinds'] = ({rid: 'prefill' if flags[i] else 'output'
            for i, rid in enumerate(batch.req_ids)} if flags is not None else {})
        selected = (self.counters['batches_seen']-1) % self.interval == 0
        record['context'] = ActionProfileContext(self, record) if selected else None
        self.current = record
        self.counters['batches_sampled' if selected else 'interval_skips'] += 1
        return record['context']

    def _total(self, row):
        key = (row.get('kind', 'batch'), row.get('layer'), row['stage'])
        return self.totals.setdefault(key, dict(kind=key[0], layer=key[1], stage=key[2],
            host_calls=0, host_wall_seconds=0., host_thread_cpu_seconds=0.,
            gpu_calls=0, gpu_elapsed_ms=0., transfer_calls=0, transfer_bytes=0))

    @contextmanager
    def stage(self, batch, name, *, gpu=False, **metadata):
        # Graph capture can be entered after prepare() began a diagnostic batch.
        # Do not allocate, record, query, or serialize CUDA events in capture.
        if batch['cuda'] and self.capturing():
            self.counters['graph_capture_skips'] += 1
            yield
            return
        self.scope_sequence += 1
        row = dict(batch_id=batch['batch_id'], scope_id=self.scope_sequence, stage=name, **metadata)
        events = None
        if gpu and batch['cuda']:
            if self.reserved_events >= self.max_events:
                self.counters['gpu_event_drops'] += 1
                row['gpu_event_dropped'] = True
            else:
                self.reserved_events += 1
                events = self.event_factory(), self.event_factory()
                events[0].record()
        row['started_at'], cpu = self.clock(), self.thread_clock()
        try:
            yield
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['host_thread_cpu_seconds'] = self.thread_clock()-cpu
            row['host_wall_seconds'] = self.clock()-row['started_at']
            row['status'] = 'error' if 'error_type' in row else 'ok'
            if events is not None:
                events[1].record()
                self.pending.append((events, row.copy()))
            batch['host_stages'].append(row)
            total = self._total(row)
            total['host_calls'] += 1
            total['host_wall_seconds'] += row['host_wall_seconds']
            total['host_thread_cpu_seconds'] += row['host_thread_cpu_seconds']
            if 'transfer_bytes' in row:
                total['transfer_calls'] += 1
                total['transfer_bytes'] += row['transfer_bytes']

    def _write(self, rows):
        if rows and self.output is not None:
            with self.output.open('a') as stream:
                stream.writelines(json.dumps(row)+'\n' for row in rows)

    def poll(self):
        if self.pending and self.capturing():
            self.counters['graph_capture_skips'] += 1
            return
        rows = []
        while self.pending and self.pending[0][0][1].query():
            (start, end), source = self.pending.popleft()
            self.reserved_events -= 1
            elapsed = start.elapsed_time(end)
            total = self._total(source)
            total['gpu_calls'] += 1
            total['gpu_elapsed_ms'] += elapsed
            self.counters['gpu_events_completed'] += 1
            rows.append(dict(event='gpu_stage', gpu_elapsed_ms=elapsed, **source))
        self._write(rows)

    def finish_batch(self, *, status='submitted'):
        batch, self.current = self.current, None
        if batch is None:
            return
        if batch['cuda'] and self.capturing():
            self.counters['graph_capture_skips'] += 1
            return
        if batch['context'] is not None:
            self._write([dict(event='batch', batch_id=batch['batch_id'],
                started_at=batch['started_at'], finished_at=self.clock(), status=status,
                requests=batch['requests'], request_kinds=batch['request_kinds'],
                stages=batch['host_stages'])])
            self.flush()
        else:
            # Unsampled batches may retire old events but must not serialize
            # the full cumulative table on every otherwise cheap decode step.
            self.poll()

    def flush(self):
        # A shutdown/round boundary may still have unfinished event pairs.
        # Report them without waiting for the GPU or dropping references early.
        if self.cuda_seen and self.capturing():
            self.counters['graph_capture_skips'] += 1
            return
        self.poll()
        self._write([self.summary()])

    def summary(self):
        return dict(event='summary', counters=dict(self.counters),
            pending_gpu_event_pairs=self.reserved_events, max_gpu_event_pairs=self.max_events,
            sample_interval=self.interval, totals=[dict(v) for v in self.totals.values()],
            scope='Inclusive host wall/thread CPU and asynchronously completed CUDA-event intervals. '
                  'Nested scopes overlap; none is GPU idle. Transfer bytes count requested payload, '
                  'not observed CUDA memcpy calls. GPU event overflow drops telemetry only. '
                  'Unresolved events remain unmeasured; diagnostics never synchronize CUDA.')
