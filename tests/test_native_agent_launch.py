import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    'native_agent_launch', Path(__file__).parents[1] / 'scripts/native_agent_launch.py')
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def profile():
    result = json.loads((Path(__file__).parents[1] /
        'examples/qwen-nvfp4-offline-agent-k128-g128.json').read_text())
    result.update(max_model_len=524288, max_num_seqs=8, gpu_memory_utilization=.81,
                  draft_tokens=4, query_replay_budget=16384, prefill_chunk_tokens=2048)
    return result


def option(argv, flag):
    return argv[argv.index(flag) + 1]


def test_direct_native_cli_keeps_effective_engine_shape_without_kvmem_hooks():
    original = profile()
    before = copy.deepcopy(original)
    plan = launch.build_launch(original, port=18901, async_scheduling=False, environ={})
    assert original == before
    argv = plan['argv']
    assert argv[:4] == [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve']
    for forbidden in ('--scheduler-cls', '--kv-transfer-config', '--middleware', '--no-enable-prefix-caching'):
        assert forbidden not in argv
    assert not any('vllm_kvmem' in value for value in argv)
    assert option(argv, '--max-num-batched-tokens') == '16424'
    assert option(argv, '--long-prefill-token-threshold') == '2048'
    assert option(argv, '--max-num-seqs') == '8'
    assert option(argv, '--max-model-len') == '524288'
    assert option(argv, '--gpu-memory-utilization') == '0.81'
    assert option(argv, '--kv-cache-dtype') == 'fp8_e4m3'
    assert option(argv, '--generation-config') == 'vllm'
    assert '--enable-prefix-caching' in argv and '--enable-chunked-prefill' in argv
    assert '--mamba-cache-mode' not in argv
    assert plan['native_profile']['hybrid_prefix_cache_policy'] == 'native_default_align'
    assert json.loads(option(argv, '--speculative-config')) == {'method': 'mtp', 'num_speculative_tokens': 4}
    assert json.loads(option(argv, '--compilation-config')) == {'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
    assert json.loads(option(argv, '--kernel-config')) == {
        'linear_backend_per_quant': {'nvfp4_w4a4': 'auto', 'fp8_w8a8': 'auto'}}
    assert json.loads(option(argv, '--reasoning-config')) == {'reasoning_end_str': original['reasoning_end_str']}
    assert 'kvmem' not in plan['native_profile']


def test_async_native_arms_differ_only_in_explicit_async_switch():
    p = profile()
    sync = launch.build_launch(p, port=18901, async_scheduling=False, environ={})
    asynchronous = launch.build_launch(p, port=18901, async_scheduling=True, environ={})
    left, right = copy.deepcopy(sync), copy.deepcopy(asynchronous)
    left['argv'].remove('--no-async-scheduling')
    right['argv'].remove('--async-scheduling')
    assert left['native_profile'].pop('async_scheduling') is False
    assert right['native_profile'].pop('async_scheduling') is True
    assert left == right


def test_native_environment_scrubs_parent_hooks_proxies_and_vllm_auth_without_logging_secrets():
    secret = 'do-not-write-this-in-artifacts'
    source = {'PATH': '/usr/bin', 'KVMEM_SERVING': '1', 'KVMEM_REQUEST_PLAN_CONFIG': secret,
        'kvmem_lower_case': secret, 'VLLM_API_KEY': secret, 'VLLM_PLUGINS': 'kvmem',
        'VLLM_USE_V2_MODEL_RUNNER': '0', 'VLLM_CUDAGRAPH_GC': secret,
        'PYTHONPATH': '/tmp/kvmem-hooks', 'PYTHONHOME': '/wrong/python',
        'HTTP_PROXY': secret, 'https_proxy': secret, 'ALL_PROXY': secret,
        'no_proxy': secret, 'FTP_PROXY': secret, 'CUDA_HOME': '/nonexistent/cuda',
        'DEEPSEEK_API_KEY': secret, 'HOME': '/test/home'}
    before = dict(source)
    plan = launch.build_launch(profile(), port=18901, async_scheduling=False, environ=source)
    assert source == before
    environment = plan['environment']
    assert not any(key.upper().startswith('KVMEM_') for key in environment)
    assert not any(key.upper().endswith('_PROXY') for key in environment)
    assert 'PYTHONPATH' not in environment and 'PYTHONHOME' not in environment
    assert 'VLLM_API_KEY' not in environment and 'VLLM_CUDAGRAPH_GC' not in environment
    assert environment['VLLM_PLUGINS'] == ''
    assert environment['VLLM_USE_V2_MODEL_RUNNER'] == '1'
    assert environment['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] == '1'
    assert environment['HF_HUB_OFFLINE'] == environment['TRANSFORMERS_OFFLINE'] == '1'
    assert environment['TOKENIZERS_PARALLELISM'] == 'false'
    assert environment['PATH'].split(':')[0] == str(Path(sys.executable).parent)
    assert environment['HOME'] == source['HOME']
    # Non-vLLM credentials can remain in the private subprocess environment;
    # never serialize that map. The artifact projection contains controlled
    # overrides and serving fields only, not inherited environment values.
    safe = {key: value for key, value in plan.items() if key != 'environment'}
    assert secret not in json.dumps(safe)
    assert 'PATH' not in plan['environment_overrides']


def test_valid_cuda_toolkit_is_preserved_privately(tmp_path):
    toolkit = tmp_path / 'cuda'
    (toolkit / 'bin').mkdir(parents=True)
    (toolkit / 'bin/nvcc').touch()
    plan = launch.build_launch(profile(), port=18901, async_scheduling=False,
                               environ={'CUDA_HOME': str(toolkit), 'PATH': '/usr/bin'})
    assert plan['environment']['CUDA_HOME'] == str(toolkit)
    assert str(toolkit / 'bin') in plan['environment']['PATH'].split(':')
    assert 'CUDA_HOME' not in plan['environment_overrides']


def test_short_context_does_not_inherit_long_context_override():
    p = profile()
    p['max_model_len'] = 262144
    plan = launch.build_launch(p, port=18901, async_scheduling=False,
                               environ={'VLLM_ALLOW_LONG_MAX_MODEL_LEN': '1'})
    assert 'VLLM_ALLOW_LONG_MAX_MODEL_LEN' not in plan['environment']


@pytest.mark.parametrize(('name', 'value'), [
    ('model_format', 'gguf'), ('draft_method', 'dflash'), ('draft_quantization', 'fp8'),
    ('execution_mode', 'eager'), ('draft_tokens', 0), ('draft_tokens', True),
    ('query_replay_budget', 0), ('max_num_seqs', False), ('max_model_len', -1),
    ('prefill_chunk_tokens', 999999), ('gpu_memory_utilization', float('nan')),
    ('model', 'https://example.com/model'), ('hf_config_path', '/different/model'),
    ('tokenizer', 'relative/tokenizer'), ('reasoning_end_str', ''),
    ('linear_backend_per_quant', {'nvfp4_w4a4': 'cutlass', 'fp8_w8a8': 'auto'}),
])
def test_rejects_unqualified_profile_changes(name, value):
    p = profile()
    p[name] = value
    with pytest.raises(ValueError):
        launch.build_launch(p, port=18901, async_scheduling=False, environ={})


@pytest.mark.parametrize(('port', 'async_scheduling'), [(0, False), (65536, False), (True, False), (18901, 1)])
def test_rejects_invalid_launch_controls(port, async_scheduling):
    with pytest.raises(ValueError):
        launch.build_launch(profile(), port=port, async_scheduling=async_scheduling, environ={})
