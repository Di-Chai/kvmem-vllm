"""Isolated native vLLM diagnostic hooks; no KVMem imports or model changes.

The scheduler only caps chunk boundaries. The worker records nonblocking CUDA
intervals around the same runner boundaries as the KVMem StepTiming diagnostic.
Instrumentation can be disabled per request to measure event/logging overhead;
that control still has the small worker/scheduler dispatch wrappers installed.
"""
from collections import deque
import json
import math
import os
from pathlib import Path
import time
from types import MethodType


def read_plan():
    return json.loads(Path(os.environ['NATIVE_PREFILL_PROBE_PLAN']).read_text())['cases']


def case_for_request(request_id, cases):
    matches = [case for case in cases if request_id.startswith('cmpl-' + case['request_id'] + '-')]
    if len(matches) != 1:
        raise ValueError('Unknown or ambiguous native diagnostic request: ' + request_id)
    return matches[0]


def cap_at_boundary(computed, proposed, prompt_tokens, chunks):
    if computed >= prompt_tokens or chunks is None:
        return proposed
    if sum(chunks) != prompt_tokens or any(type(n) is not int or n <= 0 for n in chunks):
        raise ValueError('Diagnostic chunk schedule does not cover its full prompt')
    boundary = 0
    for size in chunks:
        boundary += size
        if computed < boundary:
            return min(proposed, boundary - computed)
    raise ValueError('Unreachable chunk position')


def geometry(batch):
    return [dict(request_id=rid, query_tokens=int(n), active_seq_len=int(h)+int(n),
                 logical_seq_len=int(h)+int(n), offset=0, prefill=bool(p),
                 prefill_terminal=bool(p) and int(done)+int(n) >= int(total))
            for rid, h, n, p, done, total in zip(batch.req_ids, batch.num_computed_tokens_np,
                batch.num_scheduled_tokens, batch.is_prefilling_np,
                batch.num_computed_prefill_tokens_np, batch.prefill_len_np)]


class NativeTiming:
    def __init__(self, output, *, event_factory=None):
        self.output = output
        self.event_factory = event_factory
        self.pending = deque()
        self.active = None
        self.sequence = self.calls = self.dropped = 0
        self.previous_completed = None

    def begin(self):
        if self.active is not None:
            raise RuntimeError('Previous native timing interval did not reach sampling')
        self.calls += 1
        self.poll()
        if len(self.pending) >= 16:
            self.dropped += 1
            return
        factory = self.event_factory
        if factory is None:
            import torch
            factory = lambda: torch.cuda.Event(enable_timing=True)
        start, model, end = [factory() for _ in range(3)]
        start.record()
        self.sequence += 1
        self.active = dict(sequence=self.sequence, call_index=self.calls, started_at=time.monotonic(),
                           start=start, model=model, end=end)

    def model_begin(self, batch):
        if self.active is None:
            return
        self.active.update(batch_geometry=geometry(batch),
            prefill_tokens=sum(int(n) for n, p in zip(batch.num_scheduled_tokens, batch.is_prefilling_np) if p),
            decode_requests=sum(not bool(p) for p in batch.is_prefilling_np),
            prefill_requests=sum(bool(p) for p in batch.is_prefilling_np))
        self.active['model'].record()
        self.active['model_submitted_at'] = time.monotonic()

    def end(self):
        if self.active is None:
            return
        self.active['end'].record()
        self.active['submitted_at'] = time.monotonic()
        self.pending.append(self.active)
        self.active = None
        self.poll()

    def poll(self):
        ready = []
        while self.pending and self.pending[0]['end'].query():
            row = self.pending.popleft()
            start, model, end = (row.pop(key) for key in ('start', 'model', 'end'))
            row.update(step_ms=start.elapsed_time(end),
                host_submit_ms=1000*(row['submitted_at']-row['started_at']))
            if 'model_submitted_at' in row:
                row.update(model_ms=model.elapsed_time(end), maintenance_ms=start.elapsed_time(model),
                    maintenance_host_ms=1000*(row['model_submitted_at']-row['started_at']))
            previous = self.previous_completed
            row['previous_sequence'] = previous[0] if previous is not None else None
            row['previous_call_index'] = previous[1] if previous is not None else None
            row['previous_end_to_model_ms'] = None
            if previous is not None and 'model_submitted_at' in row:
                gap = previous[2].elapsed_time(model)
                if not math.isfinite(gap) or gap < 0:
                    raise RuntimeError('Invalid completed-step CUDA timing gap')
                row['previous_end_to_model_ms'] = gap
            self.previous_completed = (row['sequence'], row['call_index'], end)
            ready.append(row)
        if ready and self.output is not None:
            with self.output.open('a') as stream:
                stream.writelines(json.dumps(row) + '\n' for row in ready)
        return ready


def install_timing(runner, output):
    timing = NativeTiming(output)
    originals = {name: getattr(runner, name) for name in ('execute_model', 'prepare_attn', 'sample_tokens')}

    def execute(self, scheduler_output, *args, **kwargs):
        if not kwargs.get('dummy_run', False) and not kwargs.get('is_profile', False) and scheduler_output.total_num_scheduled_tokens:
            timing.begin()
        return originals['execute_model'](scheduler_output, *args, **kwargs)

    def prepare(self, batch, *args, **kwargs):
        result = originals['prepare_attn'](batch, *args, **kwargs)
        timing.model_begin(batch)
        return result

    def sample(self, *args, **kwargs):
        result = originals['sample_tokens'](*args, **kwargs)
        timing.end()
        return result

    wrapped = {name: MethodType(method, runner) for name, method in
               (('execute_model', execute), ('prepare_attn', prepare), ('sample_tokens', sample))}
    return timing, originals, wrapped


def __getattr__(name):
    # Pure accounting helpers can be imported in CPU tests without loading vLLM.
    if name == 'NativeShapeScheduler':
        from vllm.v1.core.sched.scheduler import Scheduler

        class NativeShapeScheduler(Scheduler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._shape_cases = read_plan()
                self._shape_output = Path(os.environ['NATIVE_PREFILL_PROBE_OUTPUT']) / 'scheduler.jsonl'

            def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
                case = case_for_request(request.request_id, self._shape_cases)
                if request.num_prompt_tokens != case['input_tokens']:
                    raise ValueError('Native diagnostic prompt length changed')
                capped = cap_at_boundary(num_computed_tokens, num_new_tokens,
                                         request.num_prompt_tokens, case.get('chunks'))
                return super()._reserve_prefill_lookahead(request, num_computed_tokens, capped)

            def schedule(self, *args, **kwargs):
                result = super().schedule(*args, **kwargs)
                if len(result.num_scheduled_tokens) > 1:
                    raise ValueError('Native shape diagnostic requires serial requests')
                return result

        globals()[name] = NativeShapeScheduler
        return NativeShapeScheduler
    if name == 'NativeProbeWorker':
        from vllm.v1.worker.gpu_worker import Worker

        class NativeProbeWorker(Worker):
            def load_model(self, *args, **kwargs):
                result = super().load_model(*args, **kwargs)
                if not self.use_v2_model_runner:
                    raise ValueError('Native shape diagnostic requires the V2 runner')
                self._shape_cases = read_plan()
                output = Path(os.environ['NATIVE_PREFILL_PROBE_OUTPUT'])
                self._shape_timing, self._shape_originals, self._shape_wrapped = install_timing(
                    self.model_runner, output / 'model_step_latency.jsonl')
                self._shape_enabled = False
                self._shape_ready = False
                self._shape_modes = output / 'request-modes.jsonl'
                self._shape_seen = set()
                return result

            def compile_or_warm_up_model(self, *args, **kwargs):
                # Native startup calls Worker.execute_model with synthetic
                # multi-owner batches. Do not treat these as client probes.
                result = super().compile_or_warm_up_model(*args, **kwargs)
                self._shape_ready = True
                return result

            def execute_model(self, scheduler_output):
                if not getattr(self, '_shape_ready', False):
                    return super().execute_model(scheduler_output)
                timing = getattr(self, '_shape_timing', None)
                if timing is not None:
                    timing.poll()
                    ids = tuple(scheduler_output.num_scheduled_tokens)
                    if ids:
                        if len(ids) != 1:
                            raise ValueError('Native diagnostic observed concurrent owners')
                        case = case_for_request(ids[0], self._shape_cases)
                        enabled = case['instrumented']
                        if enabled != self._shape_enabled:
                            if timing.active is not None:
                                raise RuntimeError('Cannot switch native timing inside an unfinished step')
                            for method, value in (self._shape_wrapped if enabled else self._shape_originals).items():
                                setattr(self.model_runner, method, value)
                            self._shape_enabled = enabled
                        if ids[0] not in self._shape_seen:
                            self._shape_seen.add(ids[0])
                            with self._shape_modes.open('a') as stream:
                                stream.write(json.dumps(dict(request_id=ids[0], case_id=case['id'],
                                    instrumented=enabled, at=time.monotonic())) + '\n')
                result = super().execute_model(scheduler_output)
                if timing is not None:
                    timing.poll()
                return result

        globals()[name] = NativeProbeWorker
        return NativeProbeWorker
    raise AttributeError(name)
