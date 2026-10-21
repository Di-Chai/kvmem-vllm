"""Host-only lifecycle qualification; never imports the native GPU worker."""
import builtins
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules():
    return SimpleNamespace(worker=load_script('prefill_inductor_worker'),
                           serve=load_script('prefill_inductor_serve'))


@pytest.fixture
def lifecycle(tmp_path, monkeypatch, modules):
    events = []
    state = SimpleNamespace(active=False, close_error=None, load_error=None, warmup_error=None)

    class Handle:
        def close(self, *, success):
            assert state.active
            events.append(('close', success))
            state.active = False
            if state.close_error is not None:
                raise state.close_error

    def install(output):
        events.append(('install', output))
        state.active = True
        return Handle()

    monkeypatch.setitem(sys.modules, 'prefill_inductor_control',
                        SimpleNamespace(install_launch_audit=install))
    monkeypatch.setenv('PREFILL_INDUCTOR_RUN_DIR', str(tmp_path))

    class Native:
        def load_model(self, *args, **kwargs):
            assert state.active
            events.append(('load', args, kwargs))
            if state.load_error is not None:
                raise state.load_error
            return 'loaded'

        def compile_or_warm_up_model(self, *args, **kwargs):
            assert state.active
            events.append(('warmup', args, kwargs))
            if state.warmup_error is not None:
                raise state.warmup_error
            return 'compilation times'

        def execute_model(self, value):
            assert not state.active
            return value

    class Worker(modules.worker.StartupAuditWorkerMixin, Native):
        pass

    return SimpleNamespace(worker=Worker(), worker_class=Worker, native=Native,
                           state=state, events=events, output=tmp_path)


def test_scripts_import_without_torch_or_vllm(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == 'torch' or name.startswith(('torch.', 'vllm.', 'vllm_kvmem.')) or name in ('vllm', 'vllm_kvmem'):
            raise AssertionError('GPU/production import during diagnostic module import: ' + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    load_script('prefill_inductor_worker')
    load_script('prefill_inductor_serve')


def test_startup_only_order_and_native_return_values(lifecycle):
    x = lifecycle
    assert x.worker.load_model('model', policy='same') == 'loaded'
    # The audit stays active for native memory profiling between these calls.
    assert x.state.active
    assert x.worker.compile_or_warm_up_model(17, capture=True) == 'compilation times'
    assert x.events == [
        ('install', x.output / f'inductor-startup-{os.getpid()}.json'),
        ('load', ('model',), {'policy': 'same'}),
        ('warmup', (17,), {'capture': True}), ('close', True)]
    assert x.worker._inductor_startup_audit_handle is None
    assert x.worker_class.execute_model is x.native.execute_model
    assert x.worker.execute_model('normal dispatch') == 'normal dispatch'


@pytest.mark.parametrize('phase', ['load', 'warmup'])
@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_native_startup_failure_closes_false(lifecycle, phase, error_type):
    x = lifecycle
    error = error_type('native initialization failed')
    setattr(x.state, phase + '_error', error)
    if phase == 'warmup':
        x.worker.load_model()
    with pytest.raises(error_type) as caught:
        (x.worker.load_model if phase == 'load' else x.worker.compile_or_warm_up_model)()
    assert caught.value is error
    assert x.events[-1] == ('close', False)
    assert not x.state.active
    assert x.worker._inductor_startup_audit_handle is None


def test_cleanup_failure_preserves_original_exception(lifecycle):
    x = lifecycle
    original = RuntimeError('model load failed')
    x.state.load_error = original
    x.state.close_error = ValueError('audit write failed')
    with pytest.raises(RuntimeError) as caught:
        x.worker.load_model()
    assert caught.value is original
    assert 'audit write failed' in original.__notes__[0]
    assert not x.state.active


def test_success_receipt_failure_cannot_return_ready(lifecycle):
    x = lifecycle
    x.worker.load_model()
    x.state.close_error = RuntimeError('startup audit coverage is incomplete')
    with pytest.raises(RuntimeError, match='coverage'):
        x.worker.compile_or_warm_up_model()
    assert x.events[-1] == ('close', True)
    assert not x.state.active
    assert x.worker._inductor_startup_audit_handle is None


def test_repeated_or_unordered_initialization_is_rejected(lifecycle):
    x = lifecycle
    with pytest.raises(RuntimeError, match='active'):
        x.worker.compile_or_warm_up_model()
    x.worker.load_model()
    with pytest.raises(RuntimeError, match='restarted'):
        x.worker.load_model()
    x.worker.compile_or_warm_up_model()
    with pytest.raises(RuntimeError, match='active'):
        x.worker.compile_or_warm_up_model()
    assert sum(event[0] == 'close' for event in x.events) == 1


@pytest.mark.parametrize('value', [None, '', 'relative', '/missing-kvmem-inductor-run', '\x00'])
def test_invalid_run_directory_does_not_install(modules, monkeypatch, value):
    if value is None:
        monkeypatch.delenv('PREFILL_INDUCTOR_RUN_DIR', raising=False)
    elif '\x00' not in value:
        monkeypatch.setenv('PREFILL_INDUCTOR_RUN_DIR', value)
    else:
        monkeypatch.setattr(modules.worker.os, 'environ', {'PREFILL_INDUCTOR_RUN_DIR': value})
    with pytest.raises(ValueError, match='existing absolute'):
        modules.worker.startup_audit_path()


def test_worker_class_import_is_lazy_and_preserves_native_base(modules, monkeypatch):
    native_module = ModuleType('vllm.v1.worker.gpu_worker')
    native_module.Worker = type('NativeWorker', (), {})
    monkeypatch.setitem(sys.modules, native_module.__name__, native_module)
    cls = modules.worker.SeededAuditWorker
    assert issubclass(cls, native_module.Worker)
    assert modules.worker.SeededAuditWorker is cls
    with pytest.raises(AttributeError):
        getattr(modules.worker, 'unknown_worker')


def test_seeded_probe_keeps_probe_readiness_and_removes_startup_observer(lifecycle, modules, monkeypatch):
    x = lifecycle
    class Probe(x.native):
        def load_model(self, *args, **kwargs):
            result = super().load_model(*args, **kwargs)
            self._shape_ready = False
            return result

        def compile_or_warm_up_model(self, *args, **kwargs):
            assert not self._shape_ready
            result = super().compile_or_warm_up_model(*args, **kwargs)
            self._shape_ready = True
            return result

        def execute_model(self, output):
            assert self._shape_ready
            return super().execute_model(output)

    monkeypatch.setitem(sys.modules, 'native_prefill_shape_probe', SimpleNamespace(NativeProbeWorker=Probe))
    cls = modules.worker.SeededNativeProbeWorker
    worker = cls()
    assert cls is modules.worker.SeededNativeProbeWorker
    assert cls.__mro__[1:3] == (modules.worker.StartupAuditWorkerMixin, Probe)
    assert worker.load_model() == 'loaded'
    assert worker.compile_or_warm_up_model() == 'compilation times'
    assert worker._shape_ready and not x.state.active
    assert cls.execute_model is Probe.execute_model
    assert worker.execute_model('probe timing remains') == 'probe timing remains'


def test_plan_preserves_original_arguments_and_environment(modules, tmp_path):
    plan = {'profile': {'validated': True}, 'scope': 'original qualification',
            'vllm_argv': ['serve', 'checkpoint', '--no-enable-prefix-caching'],
            'environment': {'KVMEM_SERVING': '1', 'VLLM_CACHE_ROOT': 'isolated'}}
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return plan

    run_dir = tmp_path / 'run' / 'server'
    result = modules.serve.audited_build_launch(original)('profile', 'base', run_dir, check_paths=False)
    assert calls == [(('profile', 'base', run_dir), {'check_paths': False})]
    assert result['profile'] is plan['profile']
    assert result['scope'] == plan['scope']
    assert result['vllm_argv'] == plan['vllm_argv'] + ['--worker-cls', modules.serve.WORKER_CLASS]
    assert result['environment'] == {**plan['environment'], 'PREFILL_INDUCTOR_RUN_DIR': str(run_dir.parent)}
    assert 'PREFILL_INDUCTOR_RUN_DIR' not in plan['environment']
    assert '--worker-cls' not in plan['vllm_argv']


@pytest.mark.parametrize('flag', [['--worker-cls', 'another.Worker'], ['--worker-cls=another.Worker']])
def test_explicit_worker_is_not_silently_replaced(modules, tmp_path, flag):
    original = lambda *args, **kwargs: {'vllm_argv': ['serve', 'model', *flag], 'environment': {}}
    with pytest.raises(ValueError, match='explicit worker'):
        modules.serve.audited_build_launch(original)({}, tmp_path, tmp_path / 'server')


def test_original_profile_validation_remains_authoritative(modules, tmp_path):
    error = ValueError('unsupported original profile')

    def original(*args, **kwargs):
        raise error

    with pytest.raises(ValueError) as caught:
        modules.serve.audited_build_launch(original)({}, tmp_path, tmp_path / 'server')
    assert caught.value is error


@pytest.mark.parametrize('fail', [False, True])
def test_main_delegates_and_restores_original_launcher(modules, monkeypatch, tmp_path, fail):
    original = lambda *args, **kwargs: {'vllm_argv': ['serve', 'model'], 'environment': {}}
    serve = SimpleNamespace(build_launch=original)
    argv = ['--config', 'unchanged.json', '--run-dir', str(tmp_path / 'server')]

    def native_main(received):
        assert received is argv
        plan = serve.build_launch({}, tmp_path, tmp_path / 'server')
        assert plan['vllm_argv'][-2:] == ['--worker-cls', modules.serve.WORKER_CLASS]
        if fail:
            raise RuntimeError('native launcher failed')
        return 'original result'

    serve.main = native_main
    package = ModuleType('vllm_kvmem')
    package.serve = serve
    monkeypatch.setitem(sys.modules, 'vllm_kvmem', package)
    if fail:
        with pytest.raises(RuntimeError, match='native launcher failed'):
            modules.serve.main(argv)
    else:
        assert modules.serve.main(argv) == 'original result'
    assert serve.build_launch is original


def test_real_serve_main_records_final_plan_and_keeps_compatibility_setup(modules, monkeypatch, tmp_path):
    from vllm_kvmem import serve

    source = tmp_path / 'profile.json'
    source.write_text('{}')
    run_dir = tmp_path / 'run' / 'server'
    events = []
    original = lambda *args, **kwargs: {
        'profile': {}, 'scope': 'unchanged launcher',
        'vllm_argv': ['serve', 'same-model'],
        'environment': {'KVMEM_SERVING': '1'}}
    monkeypatch.setattr(serve, 'build_launch', original)
    # Isolate all of the real launcher's environment filtering from this test
    # process. Its ordinary compatibility and CLI entry points remain visible.
    monkeypatch.setattr(serve.os, 'environ', {'KVMEM_STALE': 'discard', 'UNCHANGED': 'yes'})
    monkeypatch.setattr(sys, 'argv', ['fixture'])
    compatibility = ModuleType('vllm_kvmem.compat')
    compatibility.enable_draft_load_config_fix = lambda: events.append('compatibility')
    monkeypatch.setitem(sys.modules, compatibility.__name__, compatibility)
    cli = ModuleType('vllm.entrypoints.cli.main')

    def native_main():
        events.append('native main')
        assert os.environ['KVMEM_SERVING'] == '1'
        assert 'KVMEM_STALE' not in os.environ
        assert os.environ['UNCHANGED'] == 'yes'
        assert os.environ['PREFILL_INDUCTOR_RUN_DIR'] == str(run_dir.parent)
        assert sys.argv == ['vllm', 'serve', 'same-model', '--worker-cls', modules.serve.WORKER_CLASS]

    cli.main = native_main
    monkeypatch.setitem(sys.modules, cli.__name__, cli)
    modules.serve.main(['--config', str(source), '--run-dir', str(run_dir)])
    plan = json.loads((run_dir / 'launch.json').read_text())
    assert plan['vllm_argv'] == sys.argv[1:]
    assert plan['environment']['PREFILL_INDUCTOR_RUN_DIR'] == str(run_dir.parent)
    assert (run_dir / 'capture').is_dir()
    assert events == ['compatibility', 'native main']
    assert serve.build_launch is original
