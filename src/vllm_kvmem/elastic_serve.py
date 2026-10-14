"""Launch a native graph/MTP offload backend and authenticated turn gateway."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import secrets
import signal

from .multi_user import _load_users
from .serve import _validate_nvfp4_config, _validate_profiler_config


def build_launch(profile, base, run_dir):
    required = {'model', 'tokenizer', 'users_file'}
    defaults = {'host': '127.0.0.1', 'port': 18382, 'backend_port': 18381,
        'served_model_name': 'kvmem-elastic', 'max_model_len': 262144,
        'max_num_seqs': 32, 'max_num_batched_tokens': 8192,
        'gpu_memory_utilization': .85, 'mtp_k': 3, 'cpu_kv_bytes': 16 * 1024**3,
        'active_bytes': 32 * 1024**3, 'token_history_bytes': 64 * 1024**2,
        'max_sessions': 256, 'max_inflight': 8, 'max_queue': 128,
        'session_ttl_seconds': 3600, 'queue_timeout_seconds': 120,
        'turn_timeout_seconds': 600, 'max_body_bytes': 8 * 1024**2, 'max_output_tokens': 8192,
        'active_context_tokens': 0, 'restore_bytes': 8 * 1024**3, 'max_restore_inflight': 4,
        'token_history_bytes_per_user': 4 * 1024**2, 'max_sessions_per_user': 32,
        'context_mode': 'full', 'draft_model': None, 'gen_budget_tokens': 16384,
        'session_checkpoint_bytes': 2 * 1024**3, 'verify_session_restore': False, 'draft_tokens': 7,
        'linear_backend_per_quant': None, 'draft_quantization': None, 'profiler_config': None,
        'cpu_gather_workers': 8, 'async_seal': False, 'draft_method': 'dflash',
        'gpu_raw_k_cache_bytes': 0, 'reuse_resident_v': False,'async_cpu_scatter':False,
        'cpu_scatter_min_tokens':128, 'execution_mode': 'full_graph', 'batched_cpu_append': False,
        'request_planning': False, 'mean_k_backend': 'torch_reference'}
    if not isinstance(profile, dict) or required - set(profile) or set(profile) - required - set(defaults):
        raise ValueError('Invalid elastic serving profile')
    cfg = {**defaults, **profile}
    if type(cfg['request_planning']) is not bool or cfg['mean_k_backend'] not in ('torch_reference','qw3_cuda'):
        raise ValueError('Invalid Query planning or MeanK backend')
    if cfg['request_planning'] and (cfg['context_mode'] != 'meank' or cfg['draft_method'] != 'mtp'):
        raise ValueError('Explicit Query planning requires MeanK/native MTP')
    cfg['profiler_config'] = _validate_profiler_config(cfg['profiler_config'], base)
    if type(cfg['cpu_gather_workers']) is not int or not 1 <= cfg['cpu_gather_workers'] <= 16:
        raise ValueError('cpu_gather_workers must be an integer in [1, 16]')
    if cfg['execution_mode'] not in ('full_graph', 'compiled_graph'):
        raise ValueError('execution_mode must be full_graph or compiled_graph')
    if cfg['context_mode'] not in ('full', 'meank'):
        raise ValueError('context_mode must be full or meank')
    if type(cfg['verify_session_restore']) is not bool:
        raise ValueError('verify_session_restore must be boolean')
    if type(cfg['async_seal']) is not bool:
        raise ValueError('async_seal must be boolean')
    if type(cfg['reuse_resident_v']) is not bool:
        raise ValueError('reuse_resident_v must be boolean')
    if type(cfg['batched_cpu_append']) is not bool:
        raise ValueError('batched_cpu_append must be boolean')
    if type(cfg['async_cpu_scatter']) is not bool:
        raise ValueError('async_cpu_scatter must be boolean')
    if type(cfg['gpu_raw_k_cache_bytes']) is not int or cfg['gpu_raw_k_cache_bytes'] < 0:
        raise ValueError('gpu_raw_k_cache_bytes must be a nonnegative byte count')
    for name in ('host', 'served_model_name'):
        if not isinstance(cfg[name], str) or not cfg[name] or '\x00' in cfg[name]:
            raise ValueError(f'{name} must be a nonempty string')
    for name in required:
        path = Path(cfg[name]).expanduser()
        cfg[name] = str((base / path if not path.is_absolute() else path).resolve())
    _validate_nvfp4_config(cfg['model'])
    if not Path(cfg['tokenizer']).is_dir():
        raise ValueError('Missing tokenizer directory')
    _load_users(cfg['users_file'])
    for name in set(defaults) - {'host', 'served_model_name', 'gpu_memory_utilization', 'active_context_tokens',
                               'context_mode', 'draft_model', 'verify_session_restore', 'linear_backend_per_quant',
                               'draft_quantization', 'profiler_config', 'async_seal', 'draft_method',
                               'reuse_resident_v', 'gpu_raw_k_cache_bytes','async_cpu_scatter', 'execution_mode', 'batched_cpu_append',
                               'request_planning', 'mean_k_backend'}:
        if type(cfg[name]) is not int or cfg[name] <= 0:
            raise ValueError(f'{name} must be a positive integer')
    if type(cfg['active_context_tokens']) is not int or not 0 <= cfg['active_context_tokens'] <= cfg['max_model_len']:
        raise ValueError('active_context_tokens must be zero or a bounded positive token count')
    utilization = cfg['gpu_memory_utilization']
    if cfg['mtp_k'] not in (2, 3) or type(utilization) not in (int, float) or not 0 < utilization < 1:
        raise ValueError('Unsupported MTP depth or GPU allocation')
    if cfg['port'] == cfg['backend_port'] or max(cfg['port'], cfg['backend_port']) > 65535:
        raise ValueError('Gateway and backend require distinct valid ports')
    if not 1 <= cfg['max_inflight'] <= cfg['max_num_seqs']:
        raise ValueError('Active count must fit native maximum requests')
    if cfg['restore_bytes'] > cfg['active_bytes'] or cfg['max_restore_inflight'] > cfg['max_inflight']:
        raise ValueError('Restore reservations must fit the active budget')
    if cfg['token_history_bytes_per_user'] > cfg['token_history_bytes'] or cfg['max_sessions_per_user'] > cfg['max_sessions']:
        raise ValueError('User history limits must fit the global budget')
    run_dir = run_dir.resolve()
    cfg['geometry_file'] = str(run_dir / 'cache-geometry.json')
    cfg['backend_url'] = f"http://127.0.0.1:{cfg['backend_port']}"
    if cfg['context_mode'] == 'meank':
        from .serve import build_launch as build_mean_k_launch
        if cfg['draft_method'] not in ('dflash', 'mtp'):
            raise ValueError('MeanK draft_method must be dflash or mtp')
        if ((cfg['draft_method'] == 'dflash' and
                (not isinstance(cfg['draft_model'], str) or not cfg['draft_model'])) or
                not cfg['active_context_tokens'] or cfg['max_output_tokens'] > cfg['gen_budget_tokens']):
            raise ValueError('MeanK requires a local DFlash2 draft, active context and output within G')
        if cfg['draft_method'] == 'dflash':
            draft = Path(cfg['draft_model']).expanduser()
            cfg['draft_model'] = str((base / draft if not draft.is_absolute() else draft).resolve())
        backend = {'model_format': 'safetensors', 'model': cfg['model'], 'tokenizer': cfg['tokenizer'],
            'hf_config_path': cfg['model'], 'draft_model': cfg['draft_model'], 'execution_mode': cfg['execution_mode'],
            'verify_session_restore': cfg['verify_session_restore'],
            'draft_tokens': cfg['draft_tokens'],
            'draft_method': cfg['draft_method'],
            'draft_quantization': cfg['draft_quantization'],
            'profiler_config': cfg['profiler_config'],
            'host': '127.0.0.1', 'port': cfg['backend_port'], 'served_model_name': cfg['served_model_name'],
            'max_model_len': cfg['max_model_len'], 'max_num_seqs': cfg['max_num_seqs'],
            'max_num_batched_tokens': cfg['max_num_batched_tokens'], 'long_prefill_token_threshold': 1024,
            'gpu_memory_utilization': cfg['gpu_memory_utilization'],
            'kvmem': {'select_budget': cfg['active_context_tokens'], 'gen_budget': cfg['gen_budget_tokens'],
                'repository_blocks': (cfg['max_model_len'] + 127) // 128,
                'cpu_read_cache_bytes': cfg['cpu_kv_bytes'],
                'cpu_gather_workers': cfg['cpu_gather_workers'],
                'async_seal': cfg['async_seal'],
                'gpu_raw_k_cache_bytes': cfg['gpu_raw_k_cache_bytes'],
                'reuse_resident_v': cfg['reuse_resident_v'],
                'async_cpu_scatter':cfg['async_cpu_scatter'],
                'batched_cpu_append':cfg['batched_cpu_append'],
                'cpu_scatter_min_tokens':cfg['cpu_scatter_min_tokens'],
                # The archive clips this to its actual historical record
                # geometry. MTP adds a layer; do not retain the target-only
                # 12 GiB constant and leave global CPU capacity unused.
                'cpu_history_cache_bytes_per_request': cfg['cpu_kv_bytes'],
                'packed_assembly': True, 'direct_pinned_reads': True, 'prefill_cadence': 1,
                'async_archive_retirement': True, 'verify_archive_on_finish': False,
                'session_capacity': cfg['max_sessions'], 'session_checkpoint_bytes': cfg['session_checkpoint_bytes'],
                'session_capture_terminal': not cfg['request_planning'],
                'session_capture_decode': cfg['request_planning'],
                'request_planning': cfg['request_planning'], 'mean_k_backend': cfg['mean_k_backend']}}
        if cfg['linear_backend_per_quant'] is not None:
            backend['linear_backend_per_quant'] = cfg['linear_backend_per_quant']
        # Validate the concrete native arguments before creating any resources.
        build_mean_k_launch(backend, run_dir, run_dir / 'meank')
        cfg['geometry_file'] = str(run_dir / 'meank/capture/cache-geometry.json')
        cfg['mean_k_backend_profile'] = backend
        cfg['mtp_k'] = cfg['draft_tokens'] if cfg['draft_method'] == 'mtp' else None
        return cfg, [sys.executable, '-m', 'vllm_kvmem', 'serve',
            '--config', str(run_dir / 'backend-profile.json'), '--run-dir', str(run_dir / 'meank')]
    if cfg['linear_backend_per_quant'] is not None:
        raise ValueError('Explicit linear backends currently require the MeanK profile')
    cfg['draft_method'] = 'mtp'
    transfer = {'kv_connector': 'KVMemElasticConnector', 'kv_role': 'kv_both',
        'kv_connector_module_path': 'vllm_kvmem.elastic_connector',
        'kv_connector_extra_config': {'cpu_bytes_to_use': cfg['cpu_kv_bytes'],
            'blocks_per_chunk': 1, 'offload_prompt_only': False, 'kvmem_report_dir': str(run_dir)}}
    argv = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', cfg['model'],
        '--tokenizer', cfg['tokenizer'], '--served-model-name', cfg['served_model_name'],
        '--host', '127.0.0.1', '--port', str(cfg['backend_port']), '--dtype', 'bfloat16',
        '--load-format', 'safetensors', '--language-model-only', '--kv-cache-dtype', 'fp8_e4m3',
        '--max-model-len', str(cfg['max_model_len']), '--max-num-seqs', str(cfg['max_num_seqs']),
        '--max-num-batched-tokens', str(cfg['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(cfg['gpu_memory_utilization']), '--generation-config', 'vllm',
        '--enable-prefix-caching', '--mamba-cache-mode', 'align', '--async-scheduling',
        '--long-prefill-token-threshold', '0',
        '--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': cfg['mtp_k']}),
        '--compilation-config', json.dumps({'cudagraph_mode': 'FULL_AND_PIECEWISE'}),
        '--kv-transfer-config', json.dumps(transfer), '--enable-per-request-metrics',
        '--per-request-spec-decode-metrics', 'summary',
        '--middleware', 'vllm_kvmem.elastic_control.ElasticControlMiddleware']
    if cfg['profiler_config'] is not None:
        argv += ['--profiler-config', json.dumps(cfg['profiler_config'])]
    return cfg, argv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    path, run_dir = Path(args.config).resolve(), Path(args.run_dir).resolve()
    cfg, command = build_launch(json.loads(path.read_text()), path.parent, run_dir)
    if args.dry_run:
        print(json.dumps({'config': cfg, 'argv': command}, indent=2))
        return 0
    run_dir.mkdir(parents=True, exist_ok=False)
    if cfg['context_mode'] == 'meank':
        (run_dir / 'backend-profile.json').write_text(json.dumps(cfg['mean_k_backend_profile'], indent=2) + '\n')
    (run_dir / 'launch.json').write_text(json.dumps({'config': cfg, 'argv': command}, indent=2) + '\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('KVMEM_', 'VLLM_'))}
    env.update(VLLM_PLUGINS='', VLLM_USE_V2_MODEL_RUNNER='1', VLLM_LOG_STATS_INTERVAL='1')
    admin_file = run_dir / 'admin.key'
    fd = os.open(admin_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as secret:
        secret.write(secrets.token_urlsafe(32))
    env['KVMEM_ELASTIC_ADMIN_FILE'] = str(admin_file)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    env['PATH'] = str(Path(sys.executable).parent) + ':' + env.get('PATH', '')
    if shutil.which('nvcc', path=env['PATH']) is None:
        for cuda_dir in (env.get('CUDA_HOME'), '/usr/local/cuda', '/usr/local/cuda-13'):
            if cuda_dir and (Path(cuda_dir) / 'bin/nvcc').is_file():
                env['CUDA_HOME'] = cuda_dir
                env['PATH'] = str(Path(cuda_dir) / 'bin') + ':' + env['PATH']
                break
    with (run_dir / 'backend.log').open('w') as log:
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run_dir / 'backend-api-pid').write_text(str(server.pid))
    # uvicorn restores and re-raises SIGTERM after graceful gateway shutdown.
    # A Python handler lets finally release the owned GPU backend as well.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    old_sigterm = signal.signal(signal.SIGTERM, terminate)
    try:
        import httpx
        deadline = time.monotonic() + 900
        with httpx.Client(trust_env=False, timeout=2) as client:
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Native backend failed; inspect backend.log')
                try:
                    if (client.get(cfg['backend_url'] + '/health').status_code == 200
                            and Path(cfg['geometry_file']).is_file()):
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError('Native backend startup timed out')
                time.sleep(1)
        from .elastic_gateway import create_app
        import uvicorn
        uvicorn.run(create_app(cfg), host=cfg['host'], port=cfg['port'], workers=1)
    finally:
        server.terminate()
        try:
            server.wait(timeout=90)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        signal.signal(signal.SIGTERM, old_sigterm)
    return 0
