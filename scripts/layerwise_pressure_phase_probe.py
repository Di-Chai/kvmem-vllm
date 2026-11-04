"""Opt-in layerwise scopes for the existing bounded Nsight pressure probe.

These ranges describe host calls, including waits inside those calls. Attribute
GPU work through CUDA launch correlation; do not add nested or overlapping
ranges. This module is never enabled in a throughput or accuracy campaign.
"""


def install_layerwise_scopes(worker, *, modules=None):
    runner, scopes = worker.model_runner, worker._prefill_phase_probe
    if not getattr(runner, '_kvmem_serving', False):
        return False
    query = getattr(runner, '_kvmem_layerwise_query', None)
    if query is None:
        return False  # Installed lazily, after the ordinary runtime is created.
    pressure = getattr(runner, '_kvmem_layerwise_pressure', None)
    if pressure is None:
        raise ValueError('Layerwise pressure diagnostic requires its actual controller')

    for owner, prefix, methods in (
        (pressure, 'layerwise_pressure', ('attach_packet', 'prepare_accept',
                                         'commit_accept', 'prepare', 'committed')),
        (query, 'layerwise_query', ('prepare', '_projection', 'assemble_layer_pinned')),
    ):
        for method in methods:
            scopes.wrap(owner, method, prefix + '.' + method)

    if modules is None:
        from vllm_kvmem import cuda_scoring, native_pages, window_assembly
        modules = dict(cuda_scoring=cuda_scoring, native_pages=native_pages,
                       window_assembly=window_assembly)
    window = modules['window_assembly']
    for owner, prefix, methods in (
        (window.NativeWindowAssembler, 'assembly', ('from_runner', '__init__')),
        (window, 'assembly', ('_validate_destinations',)),
        (modules['native_pages'], 'native_pages', ('kernel_table',)),
        (modules['cuda_scoring'], 'mean_key', ('mean_key_scores',
                                             'prepare_mean_key_index',
                                             'prepared_layer_scores')),
    ):
        for method in methods:
            scopes.wrap(owner, method, prefix + '.' + method)
    # Reuse the existing host_native.commit_prefill_attention range. Never
    # wrap a PrefillHistory instance: acceptance shallow-copies that instance
    # before committing, so a bound closure could mutate the wrong state.
    return True


class _LayerwiseScopesMixin:
    def _install_phase_probe(self):
        super()._install_phase_probe()
        if getattr(self, '_prefill_phase_probe', None) is not None:
            install_layerwise_scopes(self)
            self._prefill_phase_probe.save_manifest()


def __getattr__(name):
    if name != 'LayerwisePressurePhaseWorker':
        raise AttributeError(name)
    from prefill_pressure_probe import PressurePhaseWorker
    result = type(name, (_LayerwiseScopesMixin, PressurePhaseWorker),
                  {'__module__': __name__})
    globals()[name] = result
    return result
