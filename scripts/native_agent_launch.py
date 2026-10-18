"""Build a direct, unmodified vLLM CLI launch for the agent comparison.

This module deliberately uses only the standard library. It must not import the
KVMem launcher, compatibility patches, connector, or vLLM itself. The returned
``environment`` is for subprocess execution only; persist only ``argv``,
``environment_overrides`` and ``native_profile`` in experiment artifacts.
"""
import copy
import json
import math
import os
from pathlib import Path
import sys


_PROXY_NAMES = frozenset((
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY', 'FTP_PROXY',
))
_ENVIRONMENT_OVERRIDES = {
    'VLLM_PLUGINS': '',
    'VLLM_USE_V2_MODEL_RUNNER': '1',
    'HF_HUB_OFFLINE': '1',
    'TRANSFORMERS_OFFLINE': '1',
    'HF_DATASETS_OFFLINE': '1',
    'TOKENIZERS_PARALLELISM': 'false',
}


def _integer(profile, name, *, minimum=1):
    value = profile.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _local_path(profile, name):
    value = profile.get(name)
    if (not isinstance(value, str) or not value or '\x00' in value or
            '://' in value or not Path(value).expanduser().is_absolute()):
        raise ValueError(f'{name} must be an absolute local path')
    return str(Path(value).expanduser())


def _text(profile, name):
    value = profile.get(name)
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError(f'{name} must be nonempty text')
    return value


def _environment(environ, *, allow_long):
    inherited = dict(os.environ if environ is None else environ)
    # A parent benchmark process normally imports KVMem. Its overrides and
    # import path must never turn this native subprocess into a KVMem server.
    environment = {
        name: value for name, value in inherited.items()
        if not name.upper().startswith(('KVMEM_', 'VLLM_'))
        and name.upper() not in _PROXY_NAMES
        and name.upper() not in ('PYTHONPATH', 'PYTHONHOME')
    }
    overrides = dict(_ENVIRONMENT_OVERRIDES)
    if allow_long:
        overrides['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'
    # Preserve an actual CUDA toolkit, never a stale CUDA_HOME inherited from a
    # caller. Keep this out of the safe artifact projection when it is custom.
    candidates = (inherited.get('CUDA_HOME'), '/usr/local/cuda-13', '/usr/local/cuda')
    cuda_home = next((value for value in candidates
                      if value and (Path(value) / 'bin/nvcc').is_file()), None)
    prefix = [str(Path(sys.executable).parent)]
    if cuda_home:
        environment['CUDA_HOME'] = cuda_home
        prefix.append(str(Path(cuda_home) / 'bin'))
        if cuda_home in ('/usr/local/cuda-13', '/usr/local/cuda'):
            overrides['CUDA_HOME'] = cuda_home
    else:
        environment.pop('CUDA_HOME', None)
    inherited_path = environment.get('PATH', os.defpath)
    environment['PATH'] = os.pathsep.join((*prefix, inherited_path))
    environment.update(overrides)
    return environment, overrides


def build_launch(profile, *, port, async_scheduling, environ=None):
    """Translate the qualified NVFP4 agent profile to a pure-native CLI.

    The effective token budget matches the KVMem launcher's Query allowance
    plus speculative slots, without importing that launcher. Prefix caching
    uses native vLLM's hybrid alignment policy. The two native arms differ only
    in the explicit async-scheduling switch.
    """
    if not isinstance(profile, dict):
        raise ValueError('profile must be an object')
    if type(async_scheduling) is not bool:
        raise ValueError('async_scheduling must be boolean')
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError('port must be an integer between 1 and 65535')
    if (profile.get('model_format') != 'safetensors' or
            profile.get('draft_method') != 'mtp' or
            profile.get('draft_quantization') is not None or
            profile.get('execution_mode') != 'compiled_graph'):
        raise ValueError('Native agent comparison requires safetensors, native MTP and compiled_graph')
    model = _local_path(profile, 'model')
    tokenizer = _local_path(profile, 'tokenizer')
    hf_config_path = _local_path(profile, 'hf_config_path')
    if model != hf_config_path:
        raise ValueError('hf_config_path must match the quantized model')
    host, served_model_name = (_text(profile, name) for name in ('host', 'served_model_name'))
    max_model_len = _integer(profile, 'max_model_len')
    max_num_seqs = _integer(profile, 'max_num_seqs')
    draft_tokens = _integer(profile, 'draft_tokens')
    if draft_tokens not in (1, 2, 3, 4, 7):
        raise ValueError('Unsupported native MTP speculative depth')
    # This source profile stores 4224, but its qualified KVMem launch expands
    # that to 16384 + 8 * (4 + 1) = 16424. Compare the effective engine budgets.
    replay = profile.get('query_replay_budget')
    if replay is not None:
        replay = _integer(profile, 'query_replay_budget')
        max_num_batched_tokens = replay + max_num_seqs * (draft_tokens + 1)
    else:
        max_num_batched_tokens = _integer(profile, 'max_num_batched_tokens')
    threshold = profile.get('prefill_chunk_tokens') or profile.get('long_prefill_token_threshold')
    if type(threshold) is not int or not 1 <= threshold <= max_num_batched_tokens:
        raise ValueError('Ordinary prefill threshold must fit the native batch budget')
    utilization = profile.get('gpu_memory_utilization')
    if (type(utilization) not in (int, float) or not math.isfinite(utilization)
            or not 0 < utilization < 1):
        raise ValueError('gpu_memory_utilization must be between zero and one')
    backends = profile.get('linear_backend_per_quant')
    if (not isinstance(backends, dict) or set(backends) != {'nvfp4_w4a4', 'fp8_w8a8'}
            or any(value != 'auto' for value in backends.values())):
        raise ValueError('Native comparison requires both qualified linear backends set to auto')
    reasoning_end_str = profile.get('reasoning_end_str')
    if reasoning_end_str is not None and (
            not isinstance(reasoning_end_str, str) or not reasoning_end_str or '\x00' in reasoning_end_str):
        raise ValueError('reasoning_end_str must be nonempty text or null')
    compilation = {'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
    speculation = {'method': 'mtp', 'num_speculative_tokens': draft_tokens}
    native_profile = dict(
        model_format='safetensors', model=model, tokenizer=tokenizer,
        hf_config_path=hf_config_path, host=host, port=port,
        served_model_name=served_model_name, dtype='bfloat16',
        kv_cache_dtype='fp8_e4m3', max_model_len=max_model_len,
        max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
        long_prefill_token_threshold=threshold, gpu_memory_utilization=utilization,
        enable_chunked_prefill=True, enable_prefix_caching=True,
        hybrid_prefix_cache_policy='native_default_align',
        async_scheduling=async_scheduling, generation_config='vllm',
        speculative_config=speculation, compilation_config=compilation,
        kernel_config={'linear_backend_per_quant': copy.deepcopy(backends)},
        reasoning_end_str=reasoning_end_str,
        enable_auto_tool_choice=True, tool_call_parser='qwen3_xml', reasoning_parser='qwen3')
    argv = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', model,
        '--served-model-name', served_model_name, '--host', host, '--port', str(port),
        '--tokenizer', tokenizer, '--hf-config-path', hf_config_path,
        '--config-format', 'hf', '--dtype', 'bfloat16', '--language-model-only',
        '--kv-cache-dtype', 'fp8_e4m3', '--max-model-len', str(max_model_len),
        '--max-num-seqs', str(max_num_seqs), '--max-num-batched-tokens', str(max_num_batched_tokens),
        '--gpu-memory-utilization', str(utilization), '--enable-prefix-caching',
        '--enable-chunked-prefill', '--generation-config', 'vllm',
        '--long-prefill-token-threshold', str(threshold),
        '--async-scheduling' if async_scheduling else '--no-async-scheduling',
        '--speculative-config', json.dumps(speculation),
        '--compilation-config', json.dumps(compilation), '--load-format', 'safetensors',
        '--kernel-config', json.dumps(native_profile['kernel_config']),
        '--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3']
    if reasoning_end_str is not None:
        argv += ['--reasoning-config', json.dumps({'reasoning_end_str': reasoning_end_str})]
    environment, overrides = _environment(environ, allow_long=max_model_len > 262144)
    return dict(argv=argv, environment=environment,
                environment_overrides=overrides, native_profile=native_profile)
