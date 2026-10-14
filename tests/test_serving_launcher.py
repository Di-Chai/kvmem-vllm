import json
import subprocess
import sys

import pytest

from vllm_kvmem.serve import build_launch


def profile(tmp_path):
    model = tmp_path/'model.gguf'
    mm = tmp_path/'mm.gguf'
    model.touch(); mm.touch()
    config = tmp_path/'config'
    draft = tmp_path/'draft'
    config.mkdir(); draft.mkdir()
    return {'model': model.name, 'mm_proj': mm.name, 'tokenizer': config.name,
            'hf_config_path': config.name, 'draft_model': draft.name,
            'kvmem': {'async_archive_retirement': True, 'session_capacity': 4,
                      'session_checkpoint_bytes': 1024**3}}


def test_launcher_preserves_mainline_precisions_and_native_execution(tmp_path):
    plan = build_launch(profile(tmp_path), tmp_path, tmp_path/'run')
    argv, env = plan['vllm_argv'], plan['environment']
    assert argv[:2] == ['serve', str(tmp_path/'model.gguf')]
    assert argv[argv.index('--kv-cache-dtype')+1] == 'fp8_e4m3'
    spec = json.loads(argv[argv.index('--speculative-config')+1])
    assert spec['num_speculative_tokens'] == 7 and spec['kv_cache_dtype'] == 'bfloat16'
    transfer = json.loads(argv[argv.index('--kv-transfer-config')+1])
    assert transfer['kv_connector_extra_config']['kvmem']['session_capacity'] == 4
    assert '--no-enable-prefix-caching' in argv and '--no-async-scheduling' in argv
    assert argv[argv.index('--middleware')+1] == 'vllm_kvmem.http_sessions.validate_prompt_session'
    assert env['KVMEM_SERVING'] == '1' and 'KVMEM_Q8_TENSORCORE' not in env
    assert env['VLLM_PLUGINS'] == 'gguf,kvmem'
    assert json.loads(argv[argv.index('--model-loader-extra-config')+1]) == {
        'mm_proj': str(tmp_path/'mm.gguf')}
    assert '--load-format' not in argv and plan['profile']['model_format'] == 'gguf'
    assert '--kernel-config' not in argv
    assert '--profiler-config' not in argv and plan['profile']['profiler_config'] is None
    assert not (tmp_path/'run').exists()


def test_full_graph_launch_keeps_original_selector_scheduler_and_archive(tmp_path):
    cfg = nvfp4_profile(tmp_path)
    cfg['execution_mode'] = 'full_graph'
    plan = build_launch(cfg, tmp_path, tmp_path / 'graph')
    argv = plan['vllm_argv']
    assert '--enforce-eager' not in argv
    assert json.loads(argv[argv.index('--compilation-config') + 1]) == {
        'mode': 0, 'cudagraph_mode': 'FULL'}
    assert argv[argv.index('--scheduler-cls') + 1] == 'vllm_kvmem.scheduler.KVMemScheduler'
    assert plan['environment']['KVMEM_FULL_GRAPH'] == '1'
    assert plan['environment']['KVMEM_BOUNDED_TARGET'] == '1'


def test_fp8_draft_weights_preserve_target_and_draft_kv_precisions(tmp_path):
    cfg = nvfp4_profile(tmp_path)
    cfg.update(draft_tokens=3, draft_quantization='fp8')
    plan = build_launch(cfg, tmp_path, tmp_path / 'fp8-draft')
    argv = plan['vllm_argv']
    spec = json.loads(argv[argv.index('--speculative-config') + 1])
    assert spec['quantization'] == 'fp8' and spec['num_speculative_tokens'] == 3
    assert spec['kv_cache_dtype'] == 'bfloat16'
    assert argv[argv.index('--kv-cache-dtype') + 1] == 'fp8_e4m3'
    assert '--quantization' not in argv  # keep target checkpoint metadata authoritative


@pytest.mark.parametrize('depth', [2, 3])
def test_meank_mtp_launch_uses_checkpoint_head_and_fp8_without_external_draft(tmp_path, depth):
    cfg = nvfp4_profile(tmp_path)
    cfg.pop('draft_model')
    cfg.update(draft_method='mtp', draft_tokens=depth, execution_mode='full_graph')
    plan = build_launch(cfg, tmp_path, tmp_path/'mtp')
    argv, env = plan['vllm_argv'], plan['environment']
    spec = json.loads(argv[argv.index('--speculative-config')+1])
    assert spec == {'method': 'mtp', 'num_speculative_tokens': depth}
    assert argv[argv.index('--kv-cache-dtype')+1] == 'fp8_e4m3'
    assert env['KVMEM_BOUNDED_MTP'] == '1' and 'KVMEM_BOUNDED_DFLASH' not in env
    assert '--kv-transfer-config' in argv and '--no-enable-prefix-caching' in argv


@pytest.mark.parametrize('changes', [
    {'unknown': True}, {'port': True}, {'max_model_len': 262145},
    {'max_model_len': 131072}, {'gpu_memory_utilization': float('nan')},
    {'gpu_memory_utilization': 1}, {'q8_tensorcore': 1},
    {'long_prefill_token_threshold': 1000}, {'model': 'absent.gguf'},
    {'kvmem': {'session_capacity': 4}}, {'kvmem': {'kv_dtype': 'bfloat16'}},
])
def test_invalid_or_unqualified_launch_refused_before_vllm_import(tmp_path, changes):
    with pytest.raises(ValueError):
        build_launch({**profile(tmp_path), **changes}, tmp_path, tmp_path/'run')


def test_dry_run_never_imports_engine_changes_env_or_creates_run_directory(tmp_path):
    config = tmp_path/'profile.json'
    config.write_text(json.dumps(profile(tmp_path)))
    code = '''import os,sys
from vllm_kvmem.serve import main
before=dict(os.environ)
main(sys.argv[1:])
assert dict(os.environ)==before
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, '--config', str(config),
                             '--run-dir', str(tmp_path/'run'), '--dry-run'],
                             check=True, capture_output=True, text=True)
    assert json.loads(result.stdout)['vllm_argv'][0] == 'serve'
    assert not (tmp_path/'run').exists()


def test_existing_run_directory_is_not_reused(tmp_path):
    from vllm_kvmem.serve import main
    config = tmp_path/'profile.json'
    config.write_text(json.dumps(profile(tmp_path)))
    run = tmp_path/'owned'; run.mkdir()
    marker = run/'keep'; marker.write_text('owned by another service')
    with pytest.raises(FileExistsError):
        main(['--config', str(config), '--run-dir', str(run)])
    assert marker.read_text() == 'owned by another service'


def test_multi_user_launch_keeps_keys_out_of_manifest_and_limits_api_workers(tmp_path):
    users = tmp_path/'users.json'
    users.write_text(json.dumps({'users': [{'id': 'alice', 'api_key': 'secret-test-token'}]}))
    cfg = profile(tmp_path)
    cfg['multi_user'] = {'users_file': users.name, 'max_inflight': 4}
    plan = build_launch(cfg, tmp_path, tmp_path/'run')
    argv = plan['vllm_argv']
    assert argv[argv.index('--middleware')+1] == 'vllm_kvmem.multi_user.MultiUserMiddleware'
    assert argv[argv.index('--api-server-count')+1] == '1'
    resolved = json.loads(plan['environment']['KVMEM_MULTI_USER_CONFIG'])
    assert resolved['users_file'] == str(users)
    assert resolved['max_inflight'] == cfg.get('max_num_seqs', 4)
    assert 'secret-test-token' not in json.dumps(plan)
    assert cfg['multi_user']['users_file'] == users.name


@pytest.mark.parametrize('multi_user', [None, {}, {'users_file': 'absent.json'},
    {'users_file': 'users.json', 'max_inflight': 5}])
def test_multi_user_launch_rejects_incomplete_auth_and_excess_admission(tmp_path, multi_user):
    (tmp_path/'users.json').write_text(json.dumps({'users': [{'id': 'alice', 'api_key': 'secret-test-token'}]}))
    cfg = profile(tmp_path)
    cfg['multi_user'] = multi_user
    with pytest.raises(ValueError):
        build_launch(cfg, tmp_path, tmp_path/'run')


def nvfp4_profile(tmp_path, quant=None):
    cfg = profile(tmp_path)
    model = tmp_path/'nvfp4'
    model.mkdir()
    if quant is None:
        quant = {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION',
                 'quantized_layers': {'model.layers.0.mlp.gate_proj': {'quant_algo': 'NVFP4'},
                                      'model.layers.0.self_attn.q_proj': {'quant_algo': 'FP8'}}}
    (model/'config.json').write_text(json.dumps({'quantization_config': quant}))
    cfg.update(model=model.name, hf_config_path=model.name, model_format='safetensors')
    del cfg['mm_proj']
    return cfg


@pytest.mark.parametrize('quant', [
    {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'},
    {'quant_method': 'modelopt', 'quantization': {'quant_algo': 'NVFP4'}},
    None,
])
def test_nvfp4_launch_preserves_checkpoint_quantization_and_bf16_draft(tmp_path, quant):
    cfg = nvfp4_profile(tmp_path, quant)
    plan = build_launch(cfg, tmp_path, tmp_path/'run')
    argv, env = plan['vllm_argv'], plan['environment']
    assert argv[:2] == ['serve', str(tmp_path/'nvfp4')]
    assert argv[argv.index('--hf-config-path')+1] == str(tmp_path/'nvfp4')
    assert argv[argv.index('--load-format')+1] == 'safetensors'
    assert '--quantization' not in argv and '--model-loader-extra-config' not in argv
    assert '--kernel-config' not in argv
    assert argv[argv.index('--dtype')+1] == 'bfloat16'
    assert argv[argv.index('--kv-cache-dtype')+1] == 'fp8_e4m3'
    assert env['VLLM_PLUGINS'] == 'kvmem' and 'KVMEM_Q8_TENSORCORE' not in env
    spec = json.loads(argv[argv.index('--speculative-config')+1])
    assert spec['model'] == str(tmp_path/'draft')
    assert spec['draft_load_config'] == {'load_format': 'safetensors'}
    assert spec['kv_cache_dtype'] == 'bfloat16' and 'quantization' not in spec


@pytest.mark.parametrize('changes,match', [
    ({'model_format': 'auto'}, 'model_format'),
    ({'q8_tensorcore': True}, 'q8_tensorcore'),
    ({'mm_proj': 'mm.gguf'}, 'mm_proj'),
    ({'mm_proj': None}, 'mm_proj'),
    ({'hf_config_path': 'config'}, 'must match model'),
    ({'model': 'model.gguf', 'hf_config_path': 'model.gguf'}, 'Missing local model'),
])
def test_nvfp4_launch_rejects_mixed_format_and_bf16_config(tmp_path, changes, match):
    cfg = nvfp4_profile(tmp_path)
    with pytest.raises(ValueError, match=match):
        build_launch({**cfg, **changes}, tmp_path, tmp_path/'run')


@pytest.mark.parametrize('quant', [
    {}, {'quant_method': 'fp8'}, {'quant_method': 'modelopt', 'quant_algo': 'FP8'},
    {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION'},
    {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION', 'quantized_layers': []},
    {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION',
     'quantized_layers': {'model.layers.0.mlp': {'quant_algo': 'FP8'}}},
    {'quant_method': 'modelopt', 'quant_algo': 'MIXED_PRECISION',
     'quantized_layers': {'model.layers.0.mlp': None}},
    {'quant_method': 'modelopt', 'quantization': []},
])
def test_nvfp4_launch_refuses_missing_or_wrong_quantization_metadata(tmp_path, quant):
    with pytest.raises(ValueError):
        build_launch(nvfp4_profile(tmp_path, quant), tmp_path, tmp_path/'run')


@pytest.mark.parametrize('config_text', [None, '{invalid json', '[]'])
def test_nvfp4_launch_refuses_unreadable_or_non_object_config(tmp_path, config_text):
    cfg = nvfp4_profile(tmp_path)
    path = tmp_path/'nvfp4'/'config.json'
    if config_text is None:
        path.unlink()
    else:
        path.write_text(config_text)
    with pytest.raises(ValueError):
        build_launch(cfg, tmp_path, tmp_path/'run')


def test_nvfp4_dry_run_is_cpu_only_and_keeps_configuration_environment_unchanged(tmp_path):
    config = tmp_path/'profile.json'
    cfg = nvfp4_profile(tmp_path)
    cfg['linear_backend_per_quant'] = {'nvfp4_w4a4': 'flashinfer_b12x'}
    config.write_text(json.dumps(cfg))
    code = '''import os,sys
from vllm_kvmem.serve import main
before=dict(os.environ)
main(sys.argv[1:])
assert dict(os.environ)==before
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, '--config', str(config),
                             '--run-dir', str(tmp_path/'run'), '--dry-run'],
                             check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    assert plan['environment']['VLLM_PLUGINS'] == 'kvmem'
    argv = plan['vllm_argv']
    assert json.loads(argv[argv.index('--kernel-config')+1]) == {
        'linear_backend_per_quant': cfg['linear_backend_per_quant']}
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('mapping', [
    {'nvfp4_w4a4': 'flashinfer_b12x'},
    {'nvfp4_w4a4': 'b12x', 'fp8_w8a8': 'torch'},
    {'nvfp4_w4a4': 'cutlass', 'fp8_w8a8': 'flashinfer_cutlass'},
    {'nvfp4_w4a4': 'flashinfer_cutlass', 'fp8_w8a8': 'b12x'},
    {'nvfp4_w4a4': 'auto', 'fp8_w8a8': 'cutlass'},
    {'fp8_w8a8': 'auto'},
])
def test_nvfp4_backend_overrides_are_explicit_and_do_not_change_precision_or_draft(tmp_path, mapping):
    cfg = nvfp4_profile(tmp_path)
    baseline = build_launch(cfg, tmp_path, tmp_path/'baseline')
    cfg['linear_backend_per_quant'] = dict(mapping)
    plan = build_launch(cfg, tmp_path, tmp_path/'run')
    argv = plan['vllm_argv']
    assert argv[:-2] == baseline['vllm_argv']
    assert argv[-2] == '--kernel-config'
    assert json.loads(argv[-1]) == {'linear_backend_per_quant': mapping}
    assert plan['environment'] == {
        **baseline['environment'], 'KVMEM_RUNNER_PROBE_DIR': str(tmp_path/'run'/'capture')}
    cfg['linear_backend_per_quant'].clear()
    assert plan['profile']['linear_backend_per_quant'] == mapping


@pytest.mark.parametrize('mapping', [
    None, {}, [], 'flashinfer_b12x', True,
    {'nvfp4': 'flashinfer_b12x'}, {'unknown': 'auto'},
    {'nvfp4_w4a4': None}, {'nvfp4_w4a4': True}, {'nvfp4_w4a4': []},
    {'nvfp4_w4a4': 'FLASHINFER_B12X'}, {'nvfp4_w4a4': 'flashinfer_b12x '},
    {'nvfp4_w4a4': 'marlin'}, {'nvfp4_w4a4': 'emulation'},
    {'fp8_w8a8': 'flashinfer_b12x'}, {'fp8_w8a8': 'marlin'},
    {'nvfp4_w4a4': 'cutlass', 'fp8_w8a8': 'typo'},
])
def test_backend_overrides_reject_invalid_schemes_values_and_precision_changes(tmp_path, mapping):
    cfg = nvfp4_profile(tmp_path)
    cfg['linear_backend_per_quant'] = mapping
    with pytest.raises(ValueError, match='linear'):
        build_launch(cfg, tmp_path, tmp_path/'run')


@pytest.mark.parametrize('mapping', [None, {}, {'nvfp4_w4a4': 'flashinfer_b12x'}])
def test_gguf_rejects_safetensors_backend_override_even_if_empty(tmp_path, mapping):
    cfg = profile(tmp_path)
    cfg['linear_backend_per_quant'] = mapping
    with pytest.raises(ValueError, match='requires a safetensors target'):
        build_launch(cfg, tmp_path, tmp_path/'run')


def test_profiler_config_is_opt_in_native_passthrough_and_cpu_only(tmp_path):
    cfg = nvfp4_profile(tmp_path)
    profiler = {'profiler': 'torch', 'torch_profiler_dir': 'traces',
                'ignore_frontend': True, 'torch_profiler_with_stack': False,
                'max_iterations': 100, 'delay_iterations': 2}
    cfg['profiler_config'] = profiler
    source = tmp_path / 'profile.json'
    source.write_text(json.dumps(cfg))
    code = '''import os,sys
from vllm_kvmem.serve import main
before=dict(os.environ)
main(sys.argv[1:])
assert dict(os.environ)==before
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, '--config', str(source),
        '--run-dir', str(tmp_path / 'run'), '--dry-run'],
        check=True, capture_output=True, text=True)
    plan = json.loads(result.stdout)
    argv = plan['vllm_argv']
    expected = {**profiler, 'torch_profiler_dir': str(tmp_path / 'traces')}
    assert json.loads(argv[argv.index('--profiler-config') + 1]) == expected
    assert plan['profile']['profiler_config'] == expected
    assert profiler['torch_profiler_dir'] == 'traces'
    assert not (tmp_path / 'traces').exists() and not (tmp_path / 'run').exists()


@pytest.mark.parametrize('profiler', [
    [], {}, 'torch', {'profiler': 'cuda'}, {'profiler': 'torch'},
    {'profiler': 'torch', 'torch_profiler_dir': ''},
    {'profiler': 'torch', 'torch_profiler_dir': 's3://bucket/traces'},
    {'profiler': 'torch', 'torch_profiler_dir': 'bad\x00path'},
    {'profiler': 'torch', 'torch_profiler_dir': 'traces', 'max_iterations': float('nan')},
])
def test_invalid_profiler_structure_rejected_without_engine(tmp_path, profiler):
    cfg = profile(tmp_path)
    cfg['profiler_config'] = profiler
    with pytest.raises(ValueError, match='profiler_config'):
        build_launch(cfg, tmp_path, tmp_path / 'run')


@pytest.mark.parametrize('mode', ['full', 'meank'])
@pytest.mark.parametrize('enabled', [False, True])
def test_elastic_profiler_passthrough_preserves_execution_mode(tmp_path, mode, enabled):
    from vllm_kvmem.elastic_serve import build_launch as build_elastic_launch
    model = nvfp4_profile(tmp_path)
    users = tmp_path / 'users.json'
    users.write_text(json.dumps({'users': [{'id': 'alice', 'api_key': 'test-profile-key'}]}))
    cfg = {key: model[key] for key in ('model', 'tokenizer')}
    cfg.update(users_file=users.name, context_mode=mode)
    if mode == 'meank':
        cfg.update(draft_model=model['draft_model'], active_context_tokens=32768)
    if enabled:
        cfg['profiler_config'] = {'profiler': 'torch', 'torch_profiler_dir': 'trace',
            'ignore_frontend': True, 'max_iterations': 100}
    resolved, argv = build_elastic_launch(cfg, tmp_path, tmp_path / 'run')
    if mode == 'meank':
        backend = resolved['mean_k_backend_profile']
        assert backend['profiler_config'] == resolved['profiler_config']
        argv = build_launch(backend, tmp_path, tmp_path / 'backend')['vllm_argv']
    assert ('--profiler-config' in argv) is enabled
    if enabled:
        expected = {**cfg['profiler_config'], 'torch_profiler_dir': str(tmp_path / 'trace')}
        assert json.loads(argv[argv.index('--profiler-config') + 1]) == expected
    assert json.loads(argv[argv.index('--speculative-config') + 1])['method'] == (
        'dflash' if mode == 'meank' else 'mtp')


def test_compiled_mtp_capture_uses_native_compiler_without_changing_memory_policy(tmp_path):
    cfg = nvfp4_profile(tmp_path)
    cfg.update(draft_method='mtp', draft_tokens=3, execution_mode='compiled_graph')
    plan = build_launch(cfg, tmp_path, tmp_path/'compiled')
    argv = plan['vllm_argv']
    assert json.loads(argv[argv.index('--compilation-config')+1]) == {
        'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
    assert '--no-async-scheduling' in argv and '--no-enable-prefix-caching' in argv
    assert plan['environment']['KVMEM_FULL_GRAPH'] == '1'
    assert '--enforce-eager' not in argv
    cfg['draft_method'] = 'dflash'
    with pytest.raises(ValueError, match='native safetensors Qwen MTP'):
        build_launch(cfg, tmp_path, tmp_path/'bad')
