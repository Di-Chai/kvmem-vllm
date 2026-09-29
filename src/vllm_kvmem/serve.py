"""Artifact-independent launcher for the explicitly experimental serving path.

The profile supplies model locations and public connector options. Native vLLM
still owns the API, workers and shutdown; this launcher never edits its installed
files or borrows a benchmark's successful result as qualification evidence.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from .options import RuntimeOptions


DEFAULTS = {
    'host': '127.0.0.1', 'port': 8000, 'served_model_name': 'kvmem',
    'max_num_seqs': 4, 'max_num_batched_tokens': 4224,
    'long_prefill_token_threshold': 1024, 'gpu_memory_utilization': .72,
    'max_model_len': 262144, 'q8_tensorcore': False,
}
PATHS = ('model', 'tokenizer', 'hf_config_path', 'draft_model', 'mm_proj')


def build_launch(profile, base, run_dir, *, check_paths=True):
    if not isinstance(profile, dict) or set(profile) - set(DEFAULTS) - set(PATHS) - {'kvmem'}:
        raise ValueError('Serving profile contains unknown fields or is not an object')
    cfg = {**DEFAULTS, **profile}
    for name in PATHS:
        value = cfg.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f'Serving profile requires local {name}')
        path = Path(value).expanduser()
        path = (base / path if not path.is_absolute() else path).resolve()
        if check_paths and not (path.is_file() if name in ('model', 'mm_proj') else path.is_dir()):
            raise ValueError(f'Missing local {name}: {path}')
        cfg[name] = str(path)
    if Path(cfg['model']).suffix.lower() != '.gguf':
        raise ValueError('This serving profile currently qualifies the local GGUF target path')
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
    if (cfg['max_model_len'] < options.memory.select_budget + options.memory.gen_budget or
            cfg['long_prefill_token_threshold'] % options.memory.block_tokens):
        raise ValueError('Model context must cover K+G and prefill chunks must align to authority records')
    argv = ['serve', cfg['model'], '--served-model-name', cfg['served_model_name'],
        '--host', cfg['host'], '--port', str(cfg['port']), '--tokenizer', cfg['tokenizer'],
        '--hf-config-path', cfg['hf_config_path'], '--config-format', 'hf', '--dtype', 'bfloat16',
        '--language-model-only', '--model-loader-extra-config', json.dumps({'mm_proj': cfg['mm_proj']}),
        '--kv-cache-dtype', 'fp8_e4m3', '--max-model-len', str(cfg['max_model_len']),
        '--max-num-seqs', str(cfg['max_num_seqs']), '--max-num-batched-tokens', str(cfg['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(utilization), '--no-enable-prefix-caching',
        '--generation-config', 'vllm', '--enforce-eager',
        '--long-prefill-token-threshold', str(cfg['long_prefill_token_threshold']), '--no-async-scheduling',
        '--scheduler-cls', 'vllm_kvmem.scheduler.KVMemScheduler', '--kv-transfer-config', json.dumps(transfer),
        '--speculative-config', json.dumps({'method': 'dflash', 'model': cfg['draft_model'],
            'num_speculative_tokens': 7, 'draft_load_config': {'load_format': 'safetensors'}, 'kv_cache_dtype': 'bfloat16'}),
        '--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
        '--middleware', 'vllm_kvmem.http_sessions.validate_prompt_session',
        '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml', '--reasoning-parser', 'qwen3']
    env = {'KVMEM_SERVING': '1', 'KVMEM_RUNNER_PROBE_DIR': str(run_dir / 'capture'),
        'KVMEM_ARCHIVE_PROBE': '1', 'KVMEM_LIFECYCLE_TRACE': '1', 'KVMEM_BOUNDED_DFLASH': '1',
        'KVMEM_BOUNDED_TARGET': '1', 'KVMEM_WINDOW_METADATA': '1',
        'KVMEM_DRAFT_LOAD_COMPAT': '1', 'VLLM_USE_V2_MODEL_RUNNER': '1', 'VLLM_PLUGINS': 'gguf,kvmem'}
    if cfg['q8_tensorcore']:
        env['KVMEM_Q8_TENSORCORE'] = '1'
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
    from .compat import enable_draft_load_config_fix
    enable_draft_load_config_fix()
    from vllm.entrypoints.cli.main import main as vllm_main
    sys.argv = ['vllm', *plan['vllm_argv']]
    vllm_main()
