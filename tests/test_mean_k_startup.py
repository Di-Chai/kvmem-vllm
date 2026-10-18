"""Prepare MeanK before serving readiness, never on the first user Query."""
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')


def startup_runner(monkeypatch, tmp_path, *, serving, backend, loader):
    import vllm_kvmem.runner as runner_module
    import vllm_kvmem.cuda_scoring as scoring
    import vllm_kvmem.draft_quantization as quantization
    import vllm_kvmem.mtp_runtime as mtp
    import vllm_kvmem.profile_scopes as scopes
    from vllm_kvmem.options import RuntimeOptions

    calls = []

    class Native:
        def load_model(self):
            calls.append('native-loaded')

    monkeypatch.setenv('KVMEM_SERVING', '1' if serving else '0')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    for flag in ('KVMEM_FULL_GRAPH', 'KVMEM_SCHEDULED_REBUILD_PROBE',
                 'KVMEM_BOUNDED_TARGET', 'KVMEM_ARCHIVE_PROBE',
                 'KVMEM_REPLAY_PROBE', 'KVMEM_COMPRESSION_PROBE',
                 'KVMEM_WINDOW_METADATA', 'KVMEM_REBUILD_PROBE'):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setattr(quantization, 'install_quantized_context', lambda _: None)
    monkeypatch.setattr(runner_module, 'install_qwen_capture', lambda _: ())
    monkeypatch.setattr(mtp, 'install_mtp_runtime', lambda _: None)
    monkeypatch.setattr(scopes, 'install_profile_scopes', lambda _: calls.append('model-ready'))
    monkeypatch.setattr(scoring, 'kernel', lambda: loader(calls))
    options = SimpleNamespace(mean_k_backend=backend, step_latency_interval=0,
        cpu_history_coordination=False, repository_gc=False, nvme_preallocate=False,
        nvme_max_bytes=0, archive_io_max_bytes=0, session_capacity=0)
    monkeypatch.setattr(RuntimeOptions, 'from_vllm', classmethod(lambda cls, _: options))
    runner = runner_module.make_probe_runner(Native)()
    runner.model = object()
    runner.model_config = SimpleNamespace(enforce_eager=True)
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    runner.speculative_config = None
    return runner, calls


@pytest.mark.parametrize('serving,backend,loads', [
    (True, 'qw3_cuda', True), (True, 'torch_reference', False),
    (False, 'qw3_cuda', False), (False, 'torch_reference', False),
])
def test_selected_serving_backend_loads_before_readiness(monkeypatch, tmp_path, serving, backend, loads):
    runner, calls = startup_runner(monkeypatch, tmp_path, serving=serving, backend=backend,
        loader=lambda events: events.append('mean-k-loaded'))
    runner.load_model()
    assert calls == ['native-loaded'] + (['mean-k-loaded'] if loads else []) + ['model-ready']


def test_compiler_failure_prevents_serving_readiness(monkeypatch, tmp_path):
    def fail(events):
        events.append('mean-k-failed')
        raise RuntimeError('MeanK build failed')

    runner, calls = startup_runner(monkeypatch, tmp_path, serving=True, backend='qw3_cuda', loader=fail)
    with pytest.raises(RuntimeError, match='MeanK build failed'):
        runner.load_model()
    assert calls == ['native-loaded', 'mean-k-failed']
