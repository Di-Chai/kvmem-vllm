"""Bounded, diagnostic-only publication timings; no changes to commit order.

Only class/module functions are wrapped. In particular, copied acceptance
histories must never inherit a closure bound to the original live history.
"""
import json
import os
from pathlib import Path
import threading
import time


class PublicationTimings:
    def __init__(self, scopes):
        self.scopes = scopes
        self.rows = []

    def wrap(self, owner, method, label, *, context=None):
        original = getattr(owner, method)
        if getattr(original, '_publication_timing_probe', None) is self:
            return

        def measured(*args, **kwargs):
            if not self.scopes.enabled():
                return original(*args, **kwargs)
            extra = context(*args, **kwargs) if context is not None else {}
            tid = threading.get_native_id()
            start, cpu = time.perf_counter_ns(), time.thread_time_ns()
            self.scopes.push('prefill_phase::' + label)
            failed = False
            try:
                return original(*args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                self.scopes.pop()
                self.rows.append(dict(label=label, tid=tid, start_ns=start,
                    end_ns=time.perf_counter_ns(), cpu_ns=time.thread_time_ns()-cpu,
                    failed=failed, **extra))
        measured._publication_timing_probe = self
        setattr(owner, method, measured)
        self.scopes.labels.add(label)

    def save(self):
        path = Path(os.environ['PREFILL_PHASE_PROBE_OUTPUT']) / ('publication-'+str(os.getpid())+'.json')
        path.write_text(json.dumps(dict(diagnostic_only=True, rows=list(self.rows),
            scope='Nested host intervals, not additive. Wall minus thread CPU is not automatically GIL or lock waiting.')))


def cursor_context(cursor, *args, **kwargs):
    return dict(history_begin=cursor.group.history_begin, tokens=cursor.group.count,
                logical_index=cursor.next_index)


def packet_context(runtime, packet, *args, **kwargs):
    group = packet.logical_group
    return dict(history_begin=group.history_begin, tokens=group.count)


class _PublicationMixin:
    def _install_phase_probe(self):
        super()._install_phase_probe()
        scopes = getattr(self, '_prefill_phase_probe', None)
        if scopes is None or hasattr(self, '_publication_timings'):
            return
        from vllm_kvmem import logical_publication, _native
        from vllm_kvmem.capture_runtime import CaptureRuntime
        from vllm_kvmem.authority import RequestAuthority
        from vllm_kvmem.action_attention import ActionRuntime, _PreparedActionAcceptance
        from vllm_kvmem.block_graph import GraphUpdate
        from vllm_kvmem.buffered_prefill_attention import BufferedPrefillHistory
        from vllm_kvmem._prefill_state_native import State
        from vllm_kvmem.memory_storage import MemoryArchive
        from vllm_kvmem.logical_action import LogicalActionSlice
        from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure
        self._publication_timings = timing = PublicationTimings(scopes)
        for owner, prefix, names in (
            (logical_publication, 'logical', ('append_authority_batch',)),
            (RequestAuthority, 'append', ('_prepare_append', '_commit_prepared_append', 'flush', 'poll')),
            (ActionRuntime, 'accept', ('prepare_accept', '_wait_packet', '_prefill_accept', '_report_prefill')),
            (_PreparedActionAcceptance, 'accept', ('commit',)),
            (GraphUpdate, 'graph_transaction', ('resume', 'detach', 'commit')),
            (BufferedPrefillHistory, 'cumulative', ('accept',)),
            (State, 'cumulative_native', ('committed', 'values')),
            (LayerwisePrefillPressure, 'layers', ('prepare_accept', 'commit_accept')),
            (LogicalActionSlice, 'logical', ('action_packet',)),
            (CaptureRuntime, 'publication_detail', ('_snapshot_capacity', '_record_logical_publication',
                'drain_requests', 'prefill_pipeline_can_advance', '_publication_blocked_requests')),
            (MemoryArchive, 'seal_detail', ('_seal_views', '_pack_records', '_record_digests')),
            (_native, 'native_detail', ('scatter_history_batch', 'finalize_mean_batch',
                'prepare_block_graph', 'pack_record_views', 'copy_record_slices')),
        ):
            for name in names:
                timing.wrap(owner, name, prefix+'.'+name)
        timing.wrap(logical_publication.LogicalPublicationCursor, 'advance',
                    'logical_transaction.advance', context=cursor_context)
        timing.wrap(CaptureRuntime, '_publish_logical_claim',
                    'logical_packet.publish', context=packet_context)
        scopes.save_manifest()

    def execute_model(self, *args, **kwargs):
        try:
            return super().execute_model(*args, **kwargs)
        finally:
            if self._pressure_gate.done and not getattr(self, '_publication_saved', False):
                self._publication_timings.save()
                self._publication_saved = True


def __getattr__(name):
    if name != 'PublicationBottleneckWorker':
        raise AttributeError(name)
    from layerwise_pressure_phase_probe import LayerwisePressurePhaseWorker
    result = type(name, (_PublicationMixin, LayerwisePressurePhaseWorker), {'__module__': __name__})
    globals()[name] = result
    return result
