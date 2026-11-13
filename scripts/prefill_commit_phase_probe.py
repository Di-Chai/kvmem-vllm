"""Diagnostic-only scopes for the unchanged buffered CPU commit path."""
import json
import os
from pathlib import Path
import threading
import time


class CommitTimings:
    def __init__(self, scopes):
        self.scopes=scopes
        self.rows=[]
        self.saved=-1

    def wrap(self, owner, method, label):
        original=getattr(owner,method)
        if getattr(original,'_cpu_commit_probe',None) is self:return
        # Class-level descriptors only: shallow copies must keep operating on
        # their own staged State, never a closure bound to the live history.
        def measured(*args,**kwargs):
            if not self.scopes.enabled():return original(*args,**kwargs)
            start=time.perf_counter_ns();cpu=time.thread_time_ns()
            self.scopes.push('prefill_phase::'+label)
            try:return original(*args,**kwargs)
            finally:
                self.scopes.pop()
                self.rows.append(dict(label=label,tid=threading.get_native_id(),start_ns=start,
                    end_ns=time.perf_counter_ns(),cpu_ns=time.thread_time_ns()-cpu))
        measured._cpu_commit_probe=self
        setattr(owner,method,measured)
        self.scopes.labels.add(label)

    def save(self):
        if self.saved==len(self.rows):return
        snapshot=list(self.rows)
        p=Path(os.environ['PREFILL_PHASE_PROBE_OUTPUT'])/('cpu-commits-'+str(os.getpid())+'.json')
        p.write_text(json.dumps(dict(diagnostic_only=True,rows=snapshot,
            scope='Instrumented host wall/thread CPU intervals. Native call includes argument conversion and GIL reacquisition; wall minus CPU is not automatically lock contention.')))
        self.saved=len(snapshot)


class _CommitWorkerMixin:
    def _install_phase_probe(self):
        super()._install_phase_probe()
        scopes=getattr(self,'_prefill_phase_probe',None)
        if scopes is None:return
        if not hasattr(self,'_commit_timings'):
            from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
            from vllm_kvmem._prefill_state_native import State
            from vllm_kvmem.serving.logical_publication import LogicalPublicationCursor
            self._commit_timings=CommitTimings(scopes)
            self._commit_timings.wrap(BufferedPrefillHistory,'accept','cpu_commit.buffered_accept')
            self._commit_timings.wrap(State,'committed','cpu_commit.native_committed')
            scopes.wrap(LogicalPublicationCursor,'advance','logical_publication.advance')
        runtime=getattr(self.model_runner,'_kvmem_capture_runtime',None)
        for method in ('drain_requests','ensure_download_slot','join_cpu_scatter',
                       'publication_blocked_requests','prefill_pipeline_can_advance'):
            scopes.wrap(runtime,method,'publication.'+method)
        scopes.save_manifest()

    def execute_model(self,*args,**kwargs):
        try:return super().execute_model(*args,**kwargs)
        finally:
            if self._pressure_gate.done:self._commit_timings.save()

    def sample_tokens(self,*args,**kwargs):
        try:return super().sample_tokens(*args,**kwargs)
        finally:
            if self._pressure_gate.done:self._commit_timings.save()


def __getattr__(name):
    if name!='PrefillCommitPhaseWorker':raise AttributeError(name)
    from layerwise_pressure_phase_probe import LayerwisePressurePhaseWorker
    result=type(name,(_CommitWorkerMixin,LayerwisePressurePhaseWorker),{'__module__':__name__})
    globals()[name]=result
    return result
