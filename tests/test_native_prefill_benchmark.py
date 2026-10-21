"""Keep the native control free of KVMem hooks without changing model policy."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('benchmark_native_prefill')


def launch():
    return dict(vllm_argv=[
        'serve', '/model', '--port', '18853', '--max-num-batched-tokens', '16424',
        '--kv-cache-dtype', 'fp8_e4m3', '--no-enable-prefix-caching',
        '--no-async-scheduling', '--long-prefill-token-threshold', '0',
        '--scheduler-cls', 'vllm_kvmem.scheduler.KVMemScheduler',
        '--kv-transfer-config', json.dumps({'kv_connector': 'KVMemConnector'}),
        '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 4}),
        '--middleware', 'vllm_kvmem.http_sessions.validate_prompt_session'])


def test_native_removes_only_extension_pairs_and_changes_port(benchmark):
    original = launch()
    result = benchmark.native_argv(original, 18854)
    assert result[:4] == [sys.executable, '-B', '-m', 'vllm.entrypoints.cli.main']
    assert result[4:] == [
        'serve', '/model', '--port', '18854', '--max-num-batched-tokens', '16424',
        '--kv-cache-dtype', 'fp8_e4m3', '--no-enable-prefix-caching',
        '--no-async-scheduling', '--long-prefill-token-threshold', '0',
        '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': 4})]
    assert original == launch()


@pytest.mark.parametrize('worker', [
    'prefill_inductor_worker.SeededAuditWorker',
    'prefill_inductor_worker.SeededNativeProbeWorker',
    'native_prefill_shape_probe.NativeProbeWorker'])
@pytest.mark.parametrize('equals', [False, True])
def test_native_strips_known_startup_worker_and_only_opaque_compile_key(benchmark, worker, equals):
    saved = launch()
    extra = dict(kvmem_capture_opaque_source='a' * 64, unrelated={'keep': [3, True]})
    saved['vllm_argv'] += ([f'--worker-cls={worker}', '--additional-config=' + json.dumps(extra)] if equals else
        ['--worker-cls', worker, '--additional-config', json.dumps(extra)])
    result = benchmark.native_argv(saved, 18854)
    index = result.index('--additional-config')
    assert json.loads(result[index + 1]) == {'unrelated': {'keep': [3, True]}}
    del result[index:index + 2]
    assert result == benchmark.native_argv(launch(), 18854)
    assert extra['kvmem_capture_opaque_source'] in str(saved)


def test_native_omits_empty_opaque_config_and_preserves_unrelated_json_text(benchmark):
    saved = launch()
    saved['vllm_argv'] += ['--additional-config', '{"kvmem_capture_opaque_source":"abc"}']
    assert benchmark.native_argv(saved, 18854) == benchmark.native_argv(launch(), 18854)
    saved['vllm_argv'][-1] = '{ "unrelated" : 7 }'
    assert benchmark.native_argv(saved, 18854)[-1] == '{ "unrelated" : 7 }'


@pytest.mark.parametrize('extra', [
    ['--worker-cls', 'arbitrary.Worker'], ['--worker-cls'], ['--worker-cls='],
    ['--worker-cls', 'prefill_inductor_worker.SeededAuditWorker', '--worker-cls=arbitrary.Worker'],
    ['--additional-config', '{}', '--additional-config={}'],
    ['--additional-config', '[]'], ['--additional-config', '{"kvmem_capture_opaque_source":1,"kvmem_capture_opaque_source":2}'],
    ['--additional-config', '{"nested":{"x":1,"x":2}}']])
def test_native_unknown_worker_or_ambiguous_additional_config_fails_closed(benchmark, extra):
    saved = launch()
    saved['vllm_argv'] += extra
    with pytest.raises(ValueError):
        benchmark.native_argv(saved, 18854)


def test_native_seed_helpers_use_arm_as_run_and_record_only_safe_environment(benchmark, tmp_path, monkeypatch):
    calls = []
    def installer(kind):
        def install(arm, run, env):
            calls.append((kind, arm, run))
            env['VLLM_CACHE_ROOT' if kind == 'inductor' else 'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR'] = str(run / kind)
            return {'kind': kind}
        return install
    for kind, module in [('flashinfer', 'prefill_autotune_control'), ('inductor', 'prefill_inductor_control')]:
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(install_seed=installer(kind),
            audit_seed=lambda arm, kind=kind: {'kind': kind, 'arm': str(arm)}))
    env = {'SECRET_KEY': 'must not be recorded'}
    receipt = benchmark.install_native_seeds(tmp_path, env)
    assert calls == [('flashinfer', tmp_path, tmp_path), ('inductor', tmp_path, tmp_path)]
    assert receipt['inductor'] == {'kind': 'inductor'}
    assert env['PREFILL_INDUCTOR_RUN_DIR'] == str(tmp_path)
    assert 'SECRET_KEY' not in benchmark.native_environment_receipt(env)
    for phase in ('ready', 'final'):
        report = benchmark.audit_native_seeds(tmp_path, phase)
        assert benchmark.read(tmp_path / f'autotune-{phase}-audit.json') == report


def test_native_optional_seed_noop_keeps_legacy_environment(benchmark, tmp_path):
    env = {'UNCHANGED': 'yes'}
    assert benchmark.install_native_seeds(tmp_path, env) == {'flashinfer': None, 'inductor': None}
    assert benchmark.audit_native_seeds(tmp_path, 'ready') == {'flashinfer': None, 'inductor': None}
    assert env == {'UNCHANGED': 'yes'}
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'surviving', 'prefix', 'chunk'])
def test_native_rejects_uncontrolled_launch(benchmark, change):
    saved = launch()
    argv = saved['vllm_argv']
    if change == 'missing':
        del argv[-2:]
    elif change == 'duplicate':
        argv.extend(argv[-2:])
    elif change == 'surviving':
        argv.extend(['--worker-cls', 'vllm_kvmem.Worker'])
    elif change == 'prefix':
        argv.remove('--no-enable-prefix-caching')
    else:
        argv[argv.index('--long-prefill-token-threshold') + 1] = '1024'
    with pytest.raises(ValueError):
        benchmark.native_argv(saved, 18854)


def test_native_environment_cannot_inherit_plugin_or_import_path(benchmark):
    from native_agent_launch import _environment
    env, _ = _environment({'KVMEM_ENABLED': '1', 'VLLM_PLUGINS': 'kvmem',
        'PYTHONPATH': '/frozen/kvmem/src', 'PYTHONHOME': '/unrelated',
        'HTTPS_PROXY': 'http://proxy'}, allow_long=True)
    assert env['VLLM_PLUGINS'] == ''
    assert env['VLLM_USE_V2_MODEL_RUNNER'] == '1'
    assert env['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] == '1'
    assert not any(k.startswith('KVMEM_') for k in env)
    assert not any(k in env for k in ('PYTHONPATH', 'PYTHONHOME', 'HTTPS_PROXY'))


def test_aligned_native_diagnostic_changes_only_batch_limit(benchmark):
    original = benchmark.native_argv(launch(), 18854)
    aligned = benchmark.native_argv(launch(), 18854, batch_tokens=16384)
    index = original.index('--max-num-batched-tokens') + 1
    assert original[index] == '16424' and aligned[index] == '16384'
    aligned[index] = original[index]
    assert aligned == original
    with pytest.raises(ValueError):
        benchmark.native_argv(launch(), 18854, batch_tokens=65536)


def test_idle_requires_complete_zero_running_and_waiting_counters(benchmark):
    assert not benchmark.idle([])
    running = 'vllm:num_requests_running{engine="0"} '
    waiting = 'vllm:num_requests_waiting{engine="0"} '
    assert not benchmark.idle([running + '0'])
    assert not benchmark.idle([running + '0', waiting + '1'])
    assert benchmark.idle([running + '0', waiting + '0'])


def test_chunk_diagnostic_changes_only_native_threshold(benchmark):
    original = benchmark.native_argv(launch(), 18854, batch_tokens=16384)
    chunked = benchmark.native_argv(launch(), 18854, batch_tokens=16384, prefill_chunk=2048)
    index = original.index('--long-prefill-token-threshold') + 1
    assert chunked[index] == '2048'
    chunked[index] = original[index]
    assert chunked == original
    with pytest.raises(ValueError, match='chunk diagnostic'):
        benchmark.native_argv(launch(), 18854, prefill_chunk=4096)


def test_no_existing_service_cannot_accidentally_restore_dsh(benchmark, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='no-restore-service'):
        benchmark.validate_service_policy(restore_service=True, existing_service=False)
    benchmark.validate_service_policy(restore_service=False, existing_service=False)
    monkeypatch.setitem(sys.modules, 'reference_service', SimpleNamespace(
        restore=lambda root: pytest.fail('DSH must remain stopped')))
    benchmark.finish_service(tmp_path, restore_service=False)
    assert benchmark.read(tmp_path / 'service-restore-status.json')['status'] == 'not_started'


def test_suite_preserves_order_and_rejects_changed_requests(benchmark, tmp_path):
    cases = []
    for name, warmup in [('warmup', True), ('32k-before', False), ('256k', False), ('32k-after', False)]:
        target = tmp_path / 'requests' / name
        target.mkdir(parents=True)
        benchmark.save(target / 'case.json', dict(input_path=name))
        cases.append(dict(id=name, warmup=warmup, case_sha256=benchmark.sha(target / 'case.json')))
    benchmark.save(tmp_path / 'suite.json', dict(cases=cases))
    result = benchmark.request_cases(tmp_path)
    assert [(name, warmup) for name, warmup, _ in result] == [(c['id'], c['warmup']) for c in cases]
    benchmark.save(tmp_path / 'requests/256k/case.json', dict(input_path='changed'))
    with pytest.raises(ValueError, match='Frozen suite case changed'):
        benchmark.request_cases(tmp_path)


def test_suite_rejects_escape_and_warmup_only(benchmark, tmp_path):
    benchmark.save(tmp_path / 'suite.json', dict(cases=[dict(id='../escape', warmup=False)]))
    with pytest.raises(ValueError, match='case ID'):
        benchmark.request_cases(tmp_path)
    benchmark.save(tmp_path / 'suite.json', dict(cases=[]))
    with pytest.raises(ValueError, match='measured request'):
        benchmark.request_cases(tmp_path)


def test_default_restores_but_user_opt_out_leaves_backend_stopped(benchmark, tmp_path, monkeypatch):
    restored = []
    monkeypatch.setitem(sys.modules, 'reference_service', SimpleNamespace(restore=restored.append))
    benchmark.finish_service(tmp_path, restore_service=True)
    assert restored == [tmp_path]
    benchmark.save(tmp_path / 'service-suspended.json', dict(pid=123))
    service, status = tmp_path / 'service.json', tmp_path / 'production-status.json'
    benchmark.save(tmp_path / 'restore-plan.json', dict(service_json=str(service), status_json=str(status),
        previous_service=dict(pid=123, current_native='old.json', bridge_pid=321),
        previous_status=dict(status='running', backend_pid=123), previous_model=dict(pid=123)))
    benchmark.finish_service(tmp_path, restore_service=False)
    assert restored == [tmp_path]
    assert benchmark.read(service)['pid'] is None
    assert benchmark.read(service)['bridge_pid'] == 321
    assert benchmark.read(status)['status'] == 'backend_stopped_by_user_request'
    assert benchmark.read(tmp_path / 'service-restore-status.json')['restored'] is False
