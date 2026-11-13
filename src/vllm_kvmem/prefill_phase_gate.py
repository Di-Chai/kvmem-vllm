"""Opt-in finite-cohort measurement gate; never enabled in ordinary serving.

Hold final Query preparation until every owner has completed public historical
prefill. Pressure work remains runnable. This separates concurrent historical
prefill wall time from Query/decode without changing any model/request control.
The resulting rate is a phased diagnostic, not normal continuous-serving TPS.
"""
import json
import os
from pathlib import Path
import time
from . import settings as _settings


class PrefillPhaseGate:
    def __init__(self, expected, emit, *, clock=time.perf_counter, completion=None):
        if type(expected) is not int or expected <= 0:
            raise ValueError('A finite positive owner count is required')
        self.expected, self.emit, self.clock = expected, emit, clock
        self.arrivals = {}
        self.released = False
        self.completion = completion

    def _release(self):
        if self.released or len(self.arrivals) != self.expected:
            return
        fields = {}
        if self.completion is not None:
            work = self.completion()
            if work is None:
                return
            submitted, completed = work['submitted'], work['completed']
            if (not work['all_submitted_completed'] or work.get('pending_fence') is not None or
                    submitted != completed):
                raise RuntimeError('Measurement completion callback returned unfinished GPU work')
            if (completed['public_prefill_tokens'] != sum(self.arrivals.values()) or
                    completed['public_non_prefill_tokens'] or completed['decode_request_batches']):
                raise RuntimeError('Completed public GPU work differs from the cold prefill cohort')
            if not work.get('counter_id') or not work.get('completed_fence'):
                raise RuntimeError('Measurement completion lacks its GPU fence identity')
            fields.update(gpu_completion_confirmed=True, public_work=work)
        self.released = True
        self.emit(dict(event='history_prefill_released', at=self.clock(),
                       history_tokens=sum(self.arrivals.values()),
                       owners=dict(self.arrivals), expected=self.expected, **fields))

    def arrive(self, job):
        if getattr(job.command, 'selection', None) != 'query':
            return
        rid = job.command.request_id
        if rid in self.arrivals or self.released:
            raise RuntimeError('Measurement cohort contains a repeated or extra Query owner')
        if job.stage != 'prepare' or job.future is not None or job.done or job.cancelled:
            raise RuntimeError('Final Query gate must precede all Query preparation')
        self.arrivals[rid] = job.history
        self.emit(dict(event='history_prefill_arrived', at=self.clock(),
                       request_id=rid, history_tokens=job.history,
                       arrived=len(self.arrivals), expected=self.expected))
        # GPU-confirmed measurement polls only from maintenance step(), after
        # all owners arrived. Ordinary serving never installs either mode.
        if self.completion is None:
            self._release()

    def permits(self, job):
        if getattr(job.command, 'selection', None) != 'query':
            return True
        if job.cancelled:
            raise RuntimeError('Measurement owner cancelled before completing the cohort')
        if job.command.request_id not in self.arrivals:
            raise RuntimeError('Unregistered Query entered measurement cohort')
        if not self.released and (job.stage != 'prepare' or job.future is not None):
            raise RuntimeError('Query work escaped the prefill measurement gate')
        self._release()
        return self.released


def install():
    value = _settings.get('KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS')
    if value is None:
        return
    expected = int(value)
    from .async_refresh import AsyncRefresh
    if getattr(AsyncRefresh, '_prefill_phase_gate_installed', False):
        return
    enqueue, step = AsyncRefresh.enqueue, AsyncRefresh.step

    def gate(manager):
        current = getattr(manager, '_prefill_phase_gate', None)
        if current is None:
            capacity = manager.runner.vllm_config.scheduler_config.max_num_seqs
            if expected > capacity:
                raise ValueError('Measurement cohort exceeds native owner capacity')
            path = Path(manager.runner._kvmem_probe_output) / 'prefill_phase_gate.jsonl'
            def emit(row):
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            completion = None
            if _settings.get('KVMEM_DIAGNOSTIC_PREFILL_GPU_COMPLETION') == '1':
                def completion():
                    work = getattr(manager.runner, '_kvmem_public_work', None)
                    if work is None:
                        raise RuntimeError('GPU-confirmed prefill gate requires public work counters')
                    # flush records the final partial-interval CUDA fence.
                    # It writes only changed progress; no forced per-poll IO
                    # and no synchronize() blocks another stream or owner.
                    work.poll(flush=True)
                    if (work.pending is not None or
                            work.submitted['public_batches'] != work.completed['public_batches']):
                        return None
                    return work.snapshot()
            current = manager._prefill_phase_gate = PrefillPhaseGate(expected, emit, completion=completion)
        return current

    def enqueue_with_gate(manager, command):
        job = enqueue(manager, command)
        gate(manager).arrive(job)
        return job

    def step_with_gate(manager, job):
        if not gate(manager).permits(job):
            return False
        return step(manager, job)

    AsyncRefresh.enqueue, AsyncRefresh.step = enqueue_with_gate, step_with_gate
    AsyncRefresh._prefill_phase_gate_installed = True
