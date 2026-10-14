"""Artifact-independent launcher for the explicitly experimental serving path.

The profile supplies model locations and public connector options. Native vLLM
still owns the API, workers and shutdown; this launcher never edits its installed
files or borrows a benchmark's successful result as qualification evidence.
"""
import argparse
import json
import math
import os
import shutil
from pathlib import Path
import sys
from types import SimpleNamespace

from .options import RuntimeOptions


DEFAULTS = {
    'host': '127.0.0.1', 'port': 8000, 'served_model_name': 'kvmem',
    'model_format': 'gguf',
    'execution_mode': 'eager',
    'verify_session_restore': False,
    'draft_tokens': 7,
    'draft_method': 'dflash',
    'draft_quantization': None,
    'profiler_config': None,
    'max_num_seqs': 4, 'max_num_batched_tokens': 4224,
    'long_prefill_token_threshold': 1024, 'gpu_memory_utilization': .72,
    'max_model_len': 262144, 'q8_tensorcore': False,
}
PATHS = ('model', 'tokenizer', 'hf_config_path', 'draft_model', 'mm_proj')
# Explicit experiment controls for the two schemes in the NVFP4 checkpoint.
# Exclude weight-only/emulation backends: they change the activation precision.
LINEAR_BACKENDS = {
    'nvfp4_w4a4': frozenset(('auto', 'cutlass', 'flashinfer_cutlass', 'flashinfer_b12x', 'b12x')),
    'fp8_w8a8': frozenset(('auto', 'cutlass', 'flashinfer_cutlass', 'b12x', 'torch')),
}


def _validate_profiler_config(value, base):
    """Resolve opt-in local tracing without importing the engine in dry runs.

    Native ProfilerConfig remains responsible for validating its detailed
    options; the launcher only constrains the profiler kind and output path.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or value.get('profiler') != 'torch':
        raise ValueError('profiler_config must be a torch profiler object or null')
    directory = value.get('torch_profiler_dir')
    if (not isinstance(directory, str) or not directory.strip()
            or '\x00' in directory or '://' in directory):
        raise ValueError('profiler_config requires a local torch_profiler_dir')
    try:
        # Copy the JSON payload so callers cannot alter the launch plan later.
        config = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError('profiler_config must contain JSON values') from exc
    path = Path(directory).expanduser()
    config['torch_profiler_dir'] = str((base / path if not path.is_absolute() else path).resolve())
    return config


def _validate_nvfp4_config(model):
    """Keep checkpoint quantization metadata authoritative without importing vLLM."""
    try:
        config = json.loads((Path(model) / 'config.json').read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('NVFP4 safetensors model requires a readable config.json') from exc
    quant = config.get('quantization_config') if isinstance(config, dict) else None
    if not isinstance(quant, dict) or quant.get('quant_method') not in (
            'modelopt', 'modelopt_fp4', 'modelopt_mixed'):
        raise ValueError('Safetensors target requires ModelOpt NVFP4 quantization metadata')
    settings = quant.get('quantization', quant)
    if not isinstance(settings, dict):
        raise ValueError('Invalid ModelOpt quantization metadata')
    algorithm = settings.get('quant_algo')
    if algorithm not in ('NVFP4', 'MIXED_PRECISION'):
        raise ValueError('Safetensors target requires NVFP4 or MIXED_PRECISION quantization')
    if algorithm == 'MIXED_PRECISION':
        layers = settings.get('quantized_layers')
        if (not isinstance(layers, dict) or not layers or
                any(not isinstance(layer, dict) for layer in layers.values()) or
                not any(layer.get('quant_algo') == 'NVFP4' for layer in layers.values())):
            raise ValueError('MIXED_PRECISION target requires a quantized_layers mapping with NVFP4 layers')


def build_launch(profile, base, run_dir, *, check_paths=True):
    if not isinstance(profile, dict) or set(profile) - set(DEFAULTS) - set(PATHS) - {
            'kvmem', 'multi_user', 'linear_backend_per_quant'}:
        raise ValueError('Serving profile contains unknown fields or is not an object')
    cfg = {**DEFAULTS, **profile}
    cfg['profiler_config'] = _validate_profiler_config(cfg['profiler_config'], base)
    if cfg['execution_mode'] not in ('eager', 'full_graph', 'compiled_graph'):
        raise ValueError('execution_mode must be eager, full_graph or compiled_graph')
    if type(cfg['verify_session_restore']) is not bool:
        raise ValueError('verify_session_restore must be boolean')
    if cfg['draft_method'] not in ('dflash', 'mtp'):
        raise ValueError('draft_method must be dflash or mtp')
    allowed_depths = (1, 3, 7) if cfg['draft_method'] == 'dflash' else (1, 2, 3, 7)
    if type(cfg['draft_tokens']) is not int or cfg['draft_tokens'] not in allowed_depths:
        raise ValueError(f'draft_tokens must be one of {allowed_depths} for {cfg["draft_method"]}')
    if cfg['draft_quantization'] not in (None, 'fp8'):
        raise ValueError('draft_quantization must be null or fp8')
    if cfg['model_format'] not in ('gguf', 'safetensors'):
        raise ValueError('model_format must be gguf or safetensors')
    is_gguf = cfg['model_format'] == 'gguf'
    if cfg['execution_mode'] == 'compiled_graph' and (is_gguf or cfg['draft_method'] != 'mtp'):
        raise ValueError('Compiled capture currently requires native safetensors Qwen MTP')
    if cfg['draft_method'] == 'mtp' and (is_gguf or cfg['draft_quantization'] is not None):
        raise ValueError('Bounded MTP requires native safetensors quantization metadata')
    if not is_gguf and 'mm_proj' in cfg:
        raise ValueError('mm_proj is only supported for a GGUF target')
    if 'linear_backend_per_quant' in cfg:
        if is_gguf:
            raise ValueError('linear_backend_per_quant requires a safetensors target')
        backends = cfg['linear_backend_per_quant']
        if not isinstance(backends, dict) or not backends:
            raise ValueError('linear_backend_per_quant must be a non-empty object')
        for scheme, backend in backends.items():
            if scheme not in LINEAR_BACKENDS:
                raise ValueError(f'Unsupported linear_backend_per_quant scheme: {scheme}')
            if not isinstance(backend, str) or backend not in LINEAR_BACKENDS[scheme]:
                raise ValueError(f'Unsupported linear backend for {scheme}: {backend!r}')
        cfg['linear_backend_per_quant'] = dict(backends)
    for name in PATHS:
        if name == 'draft_model' and cfg['draft_method'] == 'mtp':
            continue
        if name == 'mm_proj' and not is_gguf:
            continue
        value = cfg.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Serving profile requires local {name}')
        path = Path(value).expanduser()
        path = (base / path if not path.is_absolute() else path).resolve()
        is_file = name == 'mm_proj' or (name == 'model' and is_gguf)
        if check_paths and not (path.is_file() if is_file else path.is_dir()):
            raise ValueError(f'Missing local {name}: {path}')
        cfg[name] = str(path)
    if is_gguf and Path(cfg['model']).suffix.lower() != '.gguf':
        raise ValueError('This serving profile currently qualifies the local GGUF target path')
    if not is_gguf:
        if cfg['hf_config_path'] != cfg['model']:
            raise ValueError('Safetensors hf_config_path must match model to preserve checkpoint quantization metadata')
        if check_paths:
            _validate_nvfp4_config(cfg['model'])
    for name in ('max_num_seqs', 'max_num_batched_tokens', 'long_prefill_token_threshold', 'max_model_len', 'port'):
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if cfg['port'] > 65535 or cfg['max_model_len'] > 262144 or cfg['max_num_batched_tokens'] < 8:
        raise ValueError('Port, draft logical-position range or verifier capacity is unsupported')
    utilization = cfg['gpu_memory_utilization']
    if type(utilization) not in (int, float) or not math.isfinite(utilization) or not 0 < utilization < 1:
        raise ValueError('gpu_memory_utilization must be strictly between zero and one')
    if type(cfg['q8_tensorcore']) is not bool:
        raise ValueError('q8_tensorcore must be boolean')
    if cfg['q8_tensorcore'] and not is_gguf:
        raise ValueError('q8_tensorcore is only supported for a GGUF target')
    for name in ('host', 'served_model_name'):
        if not isinstance(cfg[name], str) or not cfg[name] or '\x00' in cfg[name]:
            raise ValueError(f'Invalid {name}')
    payload = profile.get('kvmem', {})
    if not isinstance(payload, dict):
        raise ValueError('kvmem must contain public connector options')
    transfer = {'kv_connector': 'KVMemConnector', 'kv_role': 'kv_both',
                'kv_connector_module_path': 'vllm_kvmem.connector',
                'kv_connector_extra_config': {'kvmem': payload}}
    options = RuntimeOptions.from_vllm(SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=transfer['kv_connector_extra_config']),
        cache_config=SimpleNamespace(cache_dtype='fp8_e4m3')))
    if not options.memory.speculation:
        raise ValueError('The mainline serving profile requires the DFlash2 transaction contract')
    if options.request_planning:
        if cfg['draft_method'] != 'mtp':
            raise ValueError('Explicit Query planning currently requires native Qwen MTP')
        if options.session_capacity and (options.session_capture_terminal or not options.session_capture_decode):
            raise ValueError('Explicit Query sessions require sealed decode checkpoints, not terminal checkpoints')
        cfg['long_prefill_token_threshold'] = 0
        cfg['max_num_batched_tokens'] = options.memory.gen_budget + cfg['max_num_seqs'] * (cfg['draft_tokens'] + 1)
    if (cfg['max_model_len'] < options.memory.select_budget + options.memory.gen_budget or
            cfg['long_prefill_token_threshold'] % options.memory.block_tokens):
        raise ValueError('Model context must cover K+G and prefill chunks must align to authority records')
    loader_args = (['--model-loader-extra-config', json.dumps({'mm_proj': cfg['mm_proj']})]
                   if is_gguf else [])
    # MTP inherits the target FP8 CacheConfig, including its engine-resolved
    # layout. An explicit same-dtype override makes a pre-resolution copy in
    # this pinned vLLM and loses the subsequent layout resolution.
    speculation = ({'method': 'mtp', 'num_speculative_tokens': cfg['draft_tokens']}
                   if cfg['draft_method'] == 'mtp' else
                   {'method': 'dflash', 'model': cfg['draft_model'],
                    'num_speculative_tokens': cfg['draft_tokens'], 'draft_load_config': {'load_format': 'safetensors'},
                    'kv_cache_dtype': 'bfloat16', **({'quantization': cfg['draft_quantization']}
                        if cfg['draft_quantization'] else {})})
    argv = ['serve', cfg['model'], '--served-model-name', cfg['served_model_name'],
        '--host', cfg['host'], '--port', str(cfg['port']), '--tokenizer', cfg['tokenizer'],
        '--hf-config-path', cfg['hf_config_path'], '--config-format', 'hf', '--dtype', 'bfloat16',
        '--language-model-only', *loader_args,
        '--kv-cache-dtype', 'fp8_e4m3', '--max-model-len', str(cfg['max_model_len']),
        '--max-num-seqs', str(cfg['max_num_seqs']), '--max-num-batched-tokens', str(cfg['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(utilization), '--no-enable-prefix-caching',
        '--generation-config', 'vllm',
        '--long-prefill-token-threshold', str(cfg['long_prefill_token_threshold']), '--no-async-scheduling',
        '--scheduler-cls', 'vllm_kvmem.scheduler.KVMemScheduler', '--kv-transfer-config', json.dumps(transfer),
        '--speculative-config', json.dumps(speculation),
        '--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
        '--middleware', 'vllm_kvmem.http_sessions.validate_prompt_session',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3']
    env = {'KVMEM_SERVING': '1', 'KVMEM_RUNNER_PROBE_DIR': str(run_dir / 'capture'),
        'KVMEM_ARCHIVE_PROBE': '1', 'KVMEM_LIFECYCLE_TRACE': '1',
        'KVMEM_BOUNDED_TARGET': '1', 'KVMEM_WINDOW_METADATA': '1',
        'KVMEM_DRAFT_LOAD_COMPAT': '1', 'VLLM_USE_V2_MODEL_RUNNER': '1',
        'VLLM_PLUGINS': 'gguf,kvmem' if is_gguf else 'kvmem'}
    env['KVMEM_BOUNDED_MTP' if cfg['draft_method'] == 'mtp' else 'KVMEM_BOUNDED_DFLASH'] = '1'
    if cfg['execution_mode'] in ('full_graph', 'compiled_graph'):
        compilation = ({'mode': 3, 'cudagraph_mode': 'FULL_AND_PIECEWISE'}
                       if cfg['execution_mode'] == 'compiled_graph' else
                       {'mode': 0, 'cudagraph_mode': 'FULL'})
        argv += ['--compilation-config', json.dumps(compilation)]
        env['KVMEM_FULL_GRAPH'] = '1'
    else:
        argv += ['--enforce-eager']
    if cfg['verify_session_restore']:
        env['KVMEM_VERIFY_SESSION_RESTORE'] = '1'
    if not is_gguf:
        # vLLM selects modelopt_fp4 or modelopt_mixed from the checkpoint config.
        # Forcing one here would discard the mixed checkpoint's per-layer scheme.
        argv += ['--load-format', 'safetensors']
    if 'linear_backend_per_quant' in cfg:
        argv += ['--kernel-config', json.dumps({
            'linear_backend_per_quant': cfg['linear_backend_per_quant']})]
    if cfg['q8_tensorcore']:
        env['KVMEM_Q8_TENSORCORE'] = '1'
    if cfg['profiler_config'] is not None:
        argv += ['--profiler-config', json.dumps(cfg['profiler_config'])]
    if 'multi_user' in profile:
        from .multi_user import validate_config
        settings = profile['multi_user']
        if not isinstance(settings, dict):
            raise ValueError('multi_user must contain admission settings and a users_file')
        settings = dict(settings)
        users_file = settings.get('users_file')
        if not isinstance(users_file, str) or not users_file:
            raise ValueError('multi_user requires a users_file')
        users_path = Path(users_file).expanduser()
        settings['users_file'] = str((base / users_path).resolve())
        settings = validate_config(settings, check_users=check_paths)
        if settings['max_inflight'] > cfg['max_num_seqs']:
            raise ValueError('multi_user.max_inflight must not exceed max_num_seqs')
        cfg['multi_user'] = settings
        argv[argv.index('--middleware') + 1] = 'vllm_kvmem.multi_user.MultiUserMiddleware'
        # Admission counters and tenant queues have a single API process owner.
        argv += ['--api-server-count', '1']
        env['KVMEM_MULTI_USER_CONFIG'] = json.dumps(settings)
    if options.request_planning:
        env['KVMEM_REQUEST_PLAN_CONFIG'] = json.dumps({'block_tokens': options.memory.block_tokens,
            'gen_budget': options.memory.gen_budget, 'select_budget': options.memory.select_budget})
    return {'scope': 'experimental mainline serving; see support matrix for qualification limits',
            'profile': cfg, 'vllm_argv': argv, 'environment': env}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='New owned telemetry/archive directory')
    parser.add_argument('--dry-run', action='store_true', help='Validate local inputs and print the exact launch without importing vLLM')
    args = parser.parse_args(argv)
    source = args.config.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    plan = build_launch(json.loads(source.read_text()), source.parent, run_dir)
    if 'multi_user' in plan['profile'] and os.environ.get('VLLM_RUST_FRONTEND_PATH'):
        raise ValueError('Multi-user admission requires the Python API frontend')
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    # A new directory prevents stale armed flags, cross-server trace replacement
    # and accidental reuse of another live worker's authoritative arena.
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'capture').mkdir()
    (run_dir / 'launch.json').write_text(json.dumps(plan, indent=2) + '\n')
    for name in list(os.environ):
        if name.startswith('KVMEM_'):
            del os.environ[name]
    os.environ.update(plan['environment'])
    os.environ['PATH'] = str(Path(sys.executable).parent) + ':' + os.environ.get('PATH', '')
    if shutil.which('nvcc') is None:
        for cuda_dir in (os.environ.get('CUDA_HOME'), '/usr/local/cuda', '/usr/local/cuda-13'):
            if cuda_dir and (Path(cuda_dir) / 'bin/nvcc').is_file():
                os.environ['CUDA_HOME'] = cuda_dir
                os.environ['PATH'] = str(Path(cuda_dir) / 'bin') + ':' + os.environ['PATH']
                break
    if 'multi_user' in plan['profile']:
        # The tenant middleware owns authentication; an inherited shared key
        # would reject valid tenant keys again inside the native API stack.
        os.environ.pop('VLLM_API_KEY', None)
    from .compat import enable_draft_load_config_fix
    enable_draft_load_config_fix()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ['vllm', *plan['vllm_argv']]
    vllm_main()
