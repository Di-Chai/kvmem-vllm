"""Diagnostic-only logical-token capture gate and pressure CPU scopes.

The deadline is checked at worker scheduling boundaries. No timer/background
thread calls CUDA profiler APIs. A stalled worker still relies on the existing
controller request timeout and owned-process cleanup.
"""
import json
import os
from pathlib import Path
import time
from types import MethodType
from weakref import ref


def scheduled_rows(output):
    cached = output.scheduled_cached_reqs
    new = {r.req_id: int(r.num_computed_tokens) for r in output.scheduled_new_reqs}
    computed = dict(zip(cached.req_ids, map(int, cached.num_computed_tokens)))
    computed.update(new)
    return [dict(request_id=rid, computed=computed[rid], tokens=int(count),
                 prefill=rid in new or bool(cached.is_context_phase(rid)))
            for rid, count in output.num_scheduled_tokens.items() if count]


class PressureProfiler:
    """Delegate the existing profiler; defer only its explicit API start."""
    def __init__(self, profiler, config, output, *, clock=time.perf_counter):
        self.profiler, self.config, self.output, self.clock = profiler, config, Path(output), clock
        self.armed = self.done = False
        self.started = None
        self.owner = None
        self.rows, self.events = [], []
        self.reason = None

    def __getattr__(self, name):
        return getattr(self.profiler, name)

    @property
    def is_running(self):
        return self.profiler.is_running

    def start(self):
        if self.done:
            raise RuntimeError('Pressure diagnostic captures exactly one request')
        self.armed = True

    def step(self):
        if self.is_running:
            self.profiler.step()

    def stop(self):
        self.finish('external_stop')

    def finish(self, reason):
        if not self.armed and not self.is_running:
            return
        self.reason = reason
        self.profiler.stop()
        self.armed, self.done = False, True
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'pressure-range.json').write_text(json.dumps(dict(
            diagnostic_only=True, config=self.config, reason=reason,
            request_id=self.owner, public_steps=self.rows, pressure_events=self.events,
            elapsed_seconds=None if self.started is None else self.clock()-self.started,
            deadline_kind='Checked at worker boundaries; controller timeout handles a stalled worker',
        ), indent=2) + '\n')

    def before_step(self, output):
        if not self.armed:
            return
        rows = scheduled_rows(output)
        if self.started is not None and self.clock()-self.started > self.config['max_seconds']:
            self.finish('deadline_exceeded')
            raise RuntimeError('Pressure diagnostic exceeded its scheduling-boundary deadline')
        if not rows:
            return  # Maintenance iterations must remain in the same capture.
        if len(rows) != 1 or not rows[0]['prefill']:
            self.finish('unexpected_batch')
            raise RuntimeError('Pressure capture requires one ordinary prefill owner')
        row = rows[0]
        if self.owner is not None and row['request_id'] != self.owner:
            self.finish('owner_changed')
            raise RuntimeError('Pressure capture owner changed')
        if row['computed'] >= self.config['end_tokens']:
            self.finish('logical_end')
            return
        if row['computed'] < self.config['begin_tokens']:
            return
        if self.started is None:
            self.owner = row['request_id']
            self.started = self.clock()
            self.profiler.start()
            if not self.is_running:
                self.finish('start_failed')
                raise RuntimeError('CUDA profiler did not start')
        if len(self.rows) >= 64:
            self.finish('step_limit')
            raise RuntimeError('Pressure diagnostic exceeded 64 public steps')
        self.rows.append(row)


def install_pressure_scopes(worker):
    """Wrap existing call sites only; every range remains host/nonadditive."""
    runner, scopes = worker.model_runner, worker._prefill_phase_probe
    runtime = getattr(runner, '_kvmem_capture_runtime', None)
    refresh = getattr(runner, '_kvmem_async_refresh', None)
    for owner, prefix, methods in (
        (refresh, 'refresh', ('pump', 'step', '_assemble', '_submit')),
        (runtime, 'publication', ('publication_ready', 'drain', 'advance_publication')),
        (getattr(runtime, 'archive', None), 'archive', ('read_region', '_gather_region', '_copy_range')),
    ):
        for method in methods:
            scopes.wrap(owner, method, prefix + '.' + method)
    for authority in tuple(getattr(runtime, 'authorities', {}).values()):
        scopes.wrap(authority, 'publish_sealed', 'authority.publish_sealed')
    if not getattr(runner, '_kvmem_serving', False):
        return
    from vllm_kvmem import packed_assembly, prefill_attention, split_assembly
    from vllm_kvmem.window_assembly import NativeWindowAssembler, _AssemblyStaging
    from vllm_kvmem.transfer import PackedTransfers
    for owner, prefix, methods in (
        (NativeWindowAssembler, 'assembly', ('assemble', '_assemble_split', '_upload_metadata')),
        (_AssemblyStaging, 'staging', ('destination', 'consumed')),
        (PackedTransfers, 'transfer', ('upload_from_reader',)),
        (packed_assembly, 'archive', ('read_packed_records',)),
        (prefill_attention, 'selection', ('select_pressure',)),
        (split_assembly, 'rebuild', ('gather_resident_values', 'assemble_split_layer')),
    ):
        for method in methods:
            scopes.wrap(owner, method, prefix + '.' + method)
    if refresh is None or getattr(refresh.emit, '_pressure_stage_probe', False):
        return
    original = refresh.emit
    owner_ref, function = ref(original.__self__), original.__func__

    def emit(job, stage, **fields):
        owner = owner_ref()
        if owner is None:
            raise RuntimeError('Pressure diagnostic refresh owner was retired')
        active = scopes.enabled()
        if active:
            row = dict(stage=stage, command_id=job.command.command_id,
                       request_id=job.command.request_id, history=job.history,
                       selection=job.command.selection, at=time.perf_counter())
            worker.profiler.events.append(row)
            scopes.push('prefill_pressure::' + stage + '::' + str(row['command_id']))
        try:
            return MethodType(function, owner)(job, stage, **fields)
        finally:
            if active:
                scopes.pop()

    emit._pressure_stage_probe = True
    refresh.emit = emit


class _PressureWorkerMixin:
    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        config = json.loads(os.environ['PREFILL_PRESSURE_RANGE'])
        if self.profiler is None:
            # vLLM normally creates this lazily in profile(start=True). Build
            # the wrapper now without starting it, so API start can be gated.
            from vllm.profiler.wrapper import CudaProfilerWrapper
            self.profiler = CudaProfilerWrapper(self.profiler_config)
        self._pressure_gate = PressureProfiler(self.profiler, config, os.environ['PREFILL_PHASE_PROBE_OUTPUT'])
        self.profiler = self._pressure_gate
        return result

    def execute_model(self, scheduler_output, *args, **kwargs):
        try:
            self._pressure_gate.before_step(scheduler_output)
            install_pressure_scopes(self)
            self._prefill_phase_probe.save_manifest()
            return super().execute_model(scheduler_output, *args, **kwargs)
        except BaseException:
            self._pressure_gate.finish('worker_exception')
            raise


def __getattr__(name):
    if name != 'PressurePhaseWorker':
        raise AttributeError(name)
    from prefill_phase_probe import KVMemPhaseWorker
    result = type(name, (_PressureWorkerMixin, KVMemPhaseWorker), {'__module__': __name__})
    globals()[name] = result
    return result
