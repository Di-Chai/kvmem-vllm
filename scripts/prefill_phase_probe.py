"""Optional bounded Nsight CPU scopes; never enable for throughput results.

The worker remains the native worker (or the existing native shape probe).
Only host call ranges are recorded. GPU work must be linked through CUDA API
correlation, not clipped to these CPU durations. Background ranges retain their
own thread identity, and nested ranges are not additive.
"""
import json
import os
from pathlib import Path
from types import MethodType
from weakref import ref


PREFIX = 'prefill_phase::'


class PhaseScopes:
    def __init__(self, worker, *, push=None, pop=None):
        self.worker = worker
        if push is None or pop is None:
            from torch.cuda import nvtx
            push, pop = nvtx.range_push, nvtx.range_pop
        self.push, self.pop = push, pop
        self.labels = set()
        self.manifest_labels = None

    def enabled(self):
        return bool(getattr(getattr(self.worker, 'profiler', None), 'is_running', False))

    def wrap(self, owner, method, label):
        if owner is None:
            return False
        original = getattr(owner, method, None)
        if not callable(original) or getattr(original, '_prefill_phase_probe', None) is self:
            return False

        instance = ref(original.__self__) if isinstance(original, MethodType) else None
        function = original.__func__ if instance is not None else original

        def scoped(*args, **kwargs):
            owner = instance() if instance is not None else None
            if instance is not None and owner is None:
                raise RuntimeError('Diagnostic method owner has already been released')
            call = MethodType(function, owner) if instance is not None else function
            if not self.enabled():
                return call(*args, **kwargs)
            self.push(PREFIX + label)
            try:
                return call(*args, **kwargs)
            finally:
                # The profiler may stop while a background call is running.
                # A pushed thread-local range must still be popped exactly once.
                self.pop()

        # Do not add __wrapped__=bound_method: that would create a self-cycle
        # delaying retired authorities and their buffers until cyclic GC.
        scoped.__name__ = getattr(original, '__name__', method)
        scoped.__doc__ = getattr(original, '__doc__', None)
        scoped._prefill_phase_probe = self
        setattr(owner, method, scoped)
        self.labels.add(label)
        return True

    def install(self, runner, *, native_module=None):
        def methods(owner, prefix, names):
            for name in names:
                self.wrap(owner, name, prefix + '.' + name)

        methods(runner, 'runner', ('prepare_inputs', 'prepare_attn', 'sample',
                                  '_kvmem_prepare_action_metadata'))
        proposer = getattr(runner, 'speculator', None)
        methods(proposer, 'mtp', ('propose', '_prefill'))
        native = getattr(runner, '_kvmem_native_attention', None)
        methods(native, 'native_stats', ('arm', 'resolve', 'score', 'enqueue_validation',
                                        'resolve_validation', 'validate'))
        runtime = getattr(runner, '_kvmem_capture_runtime', None)
        methods(runtime, 'capture', ('_capture_payload', 'submit', '_copy_source_owners',
                                    '_publish_reserved_claim', '_prepare_graph_acceptance',
                                    '_write_progress'))
        action = getattr(runtime, 'action', None)
        methods(action, 'action', ('prepare', 'capture', '_capture', 'stage_before_mtp',
                                   '_resolve_mtp_capture', 'prepare_accept', '_wait_packet',
                                   '_prefill_accept'))
        # Avoid scanning records/pages or retaining retired authorities in a
        # registry. Each wrapper is owned only by its existing instance.
        for authority in tuple(getattr(runtime, 'authorities', {}).values()):
            methods(authority, 'authority', ('_prepare_append', '_commit_prepared_append',
                                            'flush', 'poll'))
            methods(getattr(authority, 'block_graph', None), 'graph', ('prepare_accept', 'prepare'))
        # Do not wrap PrefillHistory.accept on an instance: acceptance makes a
        # shallow copy of that instance before committing. A copied closure
        # bound to the original would redirect a staged update into live state.
        archive = getattr(runtime, 'archive', None)
        methods(archive, 'archive', ('write_from_views', '_seal_views', '_record_digests'))
        if native_module is not None:
            methods(native_module, 'host_native', ('scatter_history_batch', 'finalize_mean_batch',
                'pack_record_views', 'pack_record_pairs', 'copy_record_slices',
                'commit_prefill_attention', 'prepare_block_graph', 'prepare_decode_attention'))

    def save_manifest(self):
        output = os.environ.get('PREFILL_PHASE_PROBE_OUTPUT')
        if not output or self.manifest_labels == self.labels:
            return
        path = Path(output) / ('phase-scopes-' + str(os.getpid()) + '.json')
        path.write_text(json.dumps(dict(diagnostic_only=True, pid=os.getpid(),
            labels=sorted(self.labels), gate='worker.profiler.is_running',
            scope='Host call ranges only; thread-local, nested and nonadditive. '
                  'GPU attribution requires CUDA launch correlation.'), indent=2) + '\n')
        self.manifest_labels = set(self.labels)


class _PhaseWorkerMixin:
    def _install_phase_probe(self):
        probe = getattr(self, '_prefill_phase_probe', None)
        if probe is None:
            return
        # Native mode must not import the KVMem extension or install hooks.
        module = None
        if getattr(self.model_runner, '_kvmem_serving', False):
            from vllm_kvmem import _native
            module = _native
        probe.install(self.model_runner, native_module=module)
        probe.save_manifest()

    def load_model(self, *args, **kwargs):
        result = super().load_model(*args, **kwargs)
        self._prefill_phase_probe = PhaseScopes(self)
        self._install_phase_probe()
        return result

    def execute_model(self, *args, **kwargs):
        self._install_phase_probe()
        result = super().execute_model(*args, **kwargs)
        # KVMem creates its runtime after the first public forward; native
        # shape probing may install its timing methods at a request boundary.
        # Warmup installs all such wrappers before the bounded traced request.
        self._install_phase_probe()
        return result

    def sample_tokens(self, *args, **kwargs):
        self._install_phase_probe()
        result = super().sample_tokens(*args, **kwargs)
        self._install_phase_probe()
        return result


def __getattr__(name):
    # CPU tests import this module without importing vLLM or initializing CUDA.
    if name == 'NativePhaseWorker':
        from native_prefill_shape_probe import NativeProbeWorker as base
    elif name == 'KVMemPhaseWorker':
        from vllm.v1.worker.gpu_worker import Worker as base
    else:
        raise AttributeError(name)
    result = type(name, (_PhaseWorkerMixin, base), {'__module__': __name__})
    globals()[name] = result
    return result
