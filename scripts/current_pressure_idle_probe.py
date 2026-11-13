"""Finite diagnostic scopes; original compiled worker and execution retained."""
from multiowner_publication_probe import _MultiOwnerMixin
from prefill_commit_phase_probe import _CommitWorkerMixin
from layerwise_pressure_phase_probe import LayerwisePressurePhaseWorker


class CurrentPressureIdleWorker(_MultiOwnerMixin, _CommitWorkerMixin,
                                LayerwisePressurePhaseWorker):
    def _install_phase_probe(self):
        super()._install_phase_probe()
        scopes=getattr(self,'_prefill_phase_probe',None)
        if scopes is None or not getattr(self.model_runner,'_kvmem_serving',False):
            return
        if not hasattr(self, '_python_gc_phase_probe'):
            from python_gc_phase_probe import GCPhaseProbe
            self._python_gc_phase_probe = GCPhaseProbe(scopes)
        import os
        self._python_gc_phase_probe.save(os.environ.get('PREFILL_PHASE_PROBE_OUTPUT'))
        from vllm_kvmem.serving.pressure_reader import PressureReader
        from vllm_kvmem.core.prepared_regions import PreparedRegionReader
        from vllm_kvmem.core.memory_storage import MemoryArchive
        from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
        from vllm_kvmem._prefill_state_native import State
        from vllm_kvmem.serving.capture_runtime import CaptureRuntime
        from vllm_kvmem import _native
        for owner,prefix,names in (
            (PressureReader,'pressure_reader',('read',)),
            (PreparedRegionReader,'prepared_reader',('read',)),
            (MemoryArchive,'archive',('_gather_region','_gather_regions_paired','_copy_range')),
            (_native,'host_native',('gather_record_region','gather_record_regions')),
            # History descriptors must remain class-level: acceptance stages
            # shallow copies, each with its own immutable buffered state.
            (BufferedPrefillHistory,'buffered_history',('scores',)),
            (State,'buffered_state',('values','scores')),
            (CaptureRuntime,'capture',('_publish_logical_claim',)),
        ):
            for name in names:scopes.wrap(owner,name,prefix+'.'+name)
        scopes.save_manifest()
