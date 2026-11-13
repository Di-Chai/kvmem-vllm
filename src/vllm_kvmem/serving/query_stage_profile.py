"""Opt-in final-Query stage observations, retired at its existing CUDA fence.

Host spans are inclusive. CUDA event intervals include stream scheduling and
host launch gaps; they are not isolated kernel durations or GPU idle time.
This collector never synchronizes CUDA and never runs for pressure assembly.
"""
from contextlib import contextmanager, nullcontext
import json
import time


def no_stage(*args, **kwargs):
    return nullcontext()


class QueryStageProfile:
    def __init__(self, output, request_id, *, event_factory=None, capturing=None,
                 clock=None, thread_clock=None, max_events=4096, max_rows=8192):
        if type(max_events) is not int or max_events < 1 or type(max_rows) is not int or max_rows < 1:
            raise ValueError('Query stage capacities must be positive integers')
        if event_factory is None or capturing is None:
            import torch
            event_factory = event_factory or (lambda: torch.cuda.Event(enable_timing=True))
            capturing = capturing or torch.cuda.is_current_stream_capturing
        self.output, self.request_id = output, request_id
        self.event_factory, self.capturing = event_factory, capturing
        self.clock, self.thread_clock = clock or time.perf_counter, thread_clock or time.thread_time
        self.max_events, self.max_rows = max_events, max_rows
        self.rows, self.pending, self.stack = [], [], []
        self.layer = None
        self.dropped_rows = self.dropped_events = 0
        self.sequence = 0
        self.finished = False

    @contextmanager
    def stage(self, name, *, gpu=False, **metadata):
        if self.finished:
            raise RuntimeError('Query profile already completed')
        if self.capturing():
            raise RuntimeError('Final Query stage profiling cannot enter a captured graph')
        if len(self.rows) >= self.max_rows:
            self.dropped_rows += 1
            yield
            return
        self.sequence += 1
        row = dict(sequence=self.sequence, parent=self.stack[-1] if self.stack else None,
                   stage=name, layer=self.layer, **metadata)
        events = None
        if gpu:
            if len(self.pending) < self.max_events:
                events = self.event_factory(), self.event_factory()
                events[0].record()
                # Reserve before nesting, not only after this scope completes.
                self.pending.append((row, events))
            else:
                self.dropped_events += 1
                row['gpu_event_dropped'] = True
        self.rows.append(row)
        row['started_at'], cpu = self.clock(), self.thread_clock()
        self.stack.append(row['sequence'])
        try:
            yield
        except BaseException as exc:
            row['error_type'] = type(exc).__name__
            raise
        finally:
            row['thread_cpu_seconds'] = self.thread_clock() - cpu
            row['finished_at'] = self.clock()
            row['wall_seconds'] = row['finished_at'] - row['started_at']
            self.stack.pop()
            if events is not None:
                events[1].record()

    def read_packed(self, authority, handles, **kwargs):
        """Exactly the normal begin/result sequence with separate host scopes."""
        from .packed_assembly import begin_read_packed_records
        with self.stage('archive_source_prepare_and_admit', records=len(handles)):
            read = begin_read_packed_records(authority, handles, **kwargs)
        with self.stage('archive_result_wait_and_release', records=len(handles)):
            result, stats = read.result()
        self._report_archive(stats)
        return result, stats

    def read_prepared(self, reader, authority, handles, **kwargs):
        """Observe one synchronous source-cache read without changing its path.

        The cached reader combines preparation and copying. Its reported nested
        timings distinguish those components; legacy begin/result scopes would
        misrepresent that execution, so this wrapper records a single host span.
        """
        with self.stage('archive_query_prepared_read', records=len(handles)):
            result, stats = reader(authority, handles, **kwargs)
        self._report_archive(stats)
        return result, stats

    def _report_archive(self, stats):
        if len(self.rows) < self.max_rows:
            self.rows.append(dict(stage='archive_report', layer=self.layer,
                stats={k:v for k,v in stats.items() if isinstance(v, (int, float, bool, str))}))
        else:
            self.dropped_rows += 1

    def score(self, compute):
        with self.stage('score_gpu_dispatch', gpu=True):
            scores = compute()
        # The host duration includes waiting for preceding model/assembly work.
        with self.stage('score_download_wait', gpu=True):
            host = scores.cpu()
        with self.stage('score_python_materialize'):
            return host.tolist()

    def finish(self):
        """Read completed events only; caller already performed its normal fence."""
        if self.finished or self.stack:
            raise RuntimeError('Query profiling completion crossed an active scope')
        self.finished = True
        unresolved = 0
        for row, (start, end) in self.pending:
            if end.query():
                row['gpu_elapsed_ms'] = start.elapsed_time(end)
            else:
                unresolved += 1
                row['gpu_unresolved'] = True
        result = dict(request_id=self.request_id, event='query_stage_profile',
            stages=self.rows, unresolved_gpu_events=unresolved,
            dropped_rows=self.dropped_rows, dropped_gpu_events=self.dropped_events,
            scope='Final public sequential Query only; initial private replay/global selection '
                  'and prefill pressure excluded. Host scopes are inclusive and nested. '
                  'CUDA event intervals may include host launch gaps and stream contention. '
                  'Archive-reported copy time is nested within read wall, not additional wall. '
                  'The caller retained its original final synchronization; this collector never waits.')
        if self.output is not None:
            with self.output.open('a') as stream:
                stream.write(json.dumps(result, allow_nan=False) + '\n')
        return result
