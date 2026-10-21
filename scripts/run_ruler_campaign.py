"""Own the RULER GPU attempt and restore the recorded DSH service afterward."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import benchmark_block_graph as lifecycle
from benchmark_ruler import save, sha
from guided_benchmark_protocol import require, validate_profile
import reference_service


def audit_execution(root, count):
    capture = root / 'run/server/capture'
    config = json.loads((capture / 'action_config.json').read_text())
    require(config['native_scores'] == 'off' and all(config[k] is True for k in
        ('native_decode_commit', 'download_pool', 'fused_scoring')), 'Optimization mode drift')
    actions = [json.loads(line) for line in (capture / 'action_attention.jsonl').open()]
    require(len(actions) == count, 'Missing terminal accepted-attention reports')
    require(all(r.get('download_pool_total') and not r['scoring_backend_counts'].get('reference', 0)
                for r in actions), 'Reference scoring fallback occurred')
    require(all(r.get('decode_commit_python_calls', 0) == 0 for r in actions), 'Python acceptance path ran')
    pools = [r['download_pool_total'] for r in actions]
    require(all(not p['exhaustions'] and not p['quarantined_slots'] for p in pools), 'Download pool failure')
    fused = max(r['scoring_backend_counts'].get('fused', 0) for r in actions)
    commits = max(r.get('decode_commit_native_calls', 0) for r in actions)
    require(fused > 0, 'No fused scores were computed')
    graphs = [json.loads(line) for line in (capture / 'block_graph.jsonl').open()]
    queries = [r for r in graphs if r['phase'] == 'query']
    require(len(queries) == count and all(r['mode'] == 'multi' and r['alpha'] == .2 for r in queries),
            'Wrong Query policy or missing Query reports')
    require(all(r.get('backend') == 'cpp_v2' or r.get('native_fallback') == 'identity' for r in queries),
            'Unexpected graph fallback')
    require(all(not r.get('propagation_applied') for r in graphs if r['phase'] == 'pressure'),
            'Pressure selection unexpectedly used graph propagation')
    save(root / 'run/execution-audit.json', dict(passed=True, query_count=count, action_config=config,
        fused_calls=fused, native_decode_calls=commits, reference_score_calls=0,
        native_graph_calls=sum(r.get('backend') == 'cpp_v2' for r in queries),
        graph_identity_fallbacks=sum(r.get('native_fallback') == 'identity' for r in queries),
        download_pool_peak_bytes=max(p['allocated_pinned_bytes'] for p in pools)))


def run(root, diagnostic_cases=None, diagnostic_gen_budget=16384, diagnostic_chunk=1024,
        *, existing_service=True, restore_service=True):
    from benchmark_native_prefill import validate_service_policy, finish_service
    validate_service_policy(existing_service=existing_service, restore_service=restore_service)
    root = root.resolve()
    runtime = root / 'runtime'
    require(Path(__file__).resolve() == runtime / 'scripts/run_ruler_campaign.py', 'Use the frozen controller')
    proof = json.loads((root / 'runtime-proof.json').read_text())
    for name, digest in proof['files'].items():
        require(sha(runtime / name) == digest, 'Frozen runtime changed: ' + name)
    fixture_proof = json.loads((root / 'dataset-summary.json').read_text())
    require(sha(root / 'fixture.json') == fixture_proof['fixture_sha256'], 'Frozen fixture changed')
    fixture = json.loads((root / 'fixture.json').read_text())
    profile = json.loads((root / 'profile.json').read_text())
    require(sha(root / 'profile.json') == fixture['profile_sha256'], 'Frozen profile changed')
    require((diagnostic_gen_budget == 16384 and diagnostic_chunk == 1024) or diagnostic_cases is not None,
            'Changing budget or chunk is restricted to an explicit diagnostic cohort')
    validate_profile(profile, 'off', 'niah', expected_gen_budget=diagnostic_gen_budget,
                     expected_chunk=diagnostic_chunk)
    require(fixture['cases'] == 260 and fixture['samples_per_task'] == 20 and
            fixture['target_length'] == 262144, 'Requested 13 x 20 at 256K differs')
    count = 260 if diagnostic_cases is None else diagnostic_cases
    require(type(count) is int and 1 <= count <= 260, 'Invalid diagnostic cohort size')
    diagnostic = diagnostic_cases is not None
    require(not profile.get('profiler_config') or diagnostic,
            'Instrumented run must be labelled diagnostic')
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime / 'src'), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit) / 'bin/nvcc').exists():
            env['CUDA_HOME'] = toolkit
            break
    target = root / 'run'
    target.mkdir(exist_ok=False)
    status = root / 'status.json'
    server = client = None
    with (root / '.campaign.lock').open('a+') as campaign_lock, \
            open('/tmp/kvmem-coordination-qualification.lock', 'a+') as gpu_lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        for lock in (campaign_lock, gpu_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # No new requests are submitted to DSH by this check.
            if existing_service:
                save(status, dict(status='waiting_for_idle_service', cases=count, diagnostic_only=diagnostic, at=time.time()))
                while True:
                    with reference_service.http().open('http://127.0.0.1:18846/metrics', timeout=5) as response:
                        metrics = response.read().decode().splitlines()
                    requests = [float(line.rsplit(' ', 1)[1]) for line in metrics if line.startswith(
                        ('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
                    require(len(requests) >= 2, 'Missing DSH idle ownership evidence')
                    if not any(requests):
                        break
                    time.sleep(5)
                reference_service.suspend(root)
            require(not lifecycle.gpu_processes(), 'Foreign GPU process is active')
            command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve',
                '--config', str(root / 'profile.json'), '--run-dir', str(target / 'server')]
            started = time.monotonic()
            with (target / 'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(target / 'server-process.json', dict(pid=server.pid, command=command, started_at=time.time()))
            save(status, dict(status='loading', cases=count, diagnostic_only=diagnostic, server_pid=server.pid, at=time.time()))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                require(server.poll() is None, 'RULER model exited while loading')
                try:
                    with reference_service.http().open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('RULER model loading exceeded 20 minutes')
            prepared = time.monotonic()
            if profile.get('profiler_config'):
                from vllm_kvmem.profiling import control_profiler
                control_profiler(f"http://127.0.0.1:{profile['port']}", 'start')
            command = [sys.executable, '-B', str(runtime / 'scripts/benchmark_ruler.py'),
                       'run', '--campaign', str(root)]
            if diagnostic:
                command += ['--diagnostic-cases', str(count),
                            '--diagnostic-gen-budget', str(diagnostic_gen_budget),
                            '--diagnostic-chunk', str(diagnostic_chunk)]
            with (target / 'client.log').open('x') as log:
                client = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(target / 'client-process.json', dict(pid=client.pid, command=command, started_at=time.time()))
            save(status, dict(status='testing', cases=count, diagnostic_only=diagnostic, server_pid=server.pid, client_pid=client.pid,
                at=time.time(), cold_prepare_seconds=prepared - started))
            with (target / 'resources.jsonl').open('x') as telemetry:
                while client.poll() is None:
                    require(server.poll() is None, 'RULER model exited during inference')
                    require(not (target / 'answers/failure.json').exists(), 'Client reported a protocol or inference failure')
                    require(time.monotonic() - prepared < 12 * 3600, 'RULER inference exceeded 12 hours')
                    memory = {line.split(':')[0]:int(line.split()[1]) * 1024
                        for line in Path('/proc/meminfo').read_text().splitlines()
                        if line.startswith(('MemAvailable:', 'MemTotal:'))}
                    gpu = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True).strip()
                    with reference_service.http().open(f"http://127.0.0.1:{profile['port']}/metrics", timeout=5) as response:
                        metrics = [line for line in response.read().decode().splitlines()
                            if line.startswith('vllm:') and any(s in line for s in
                            ('spec_decode', 'num_requests_', 'prompt_tokens_total', 'generation_tokens_total'))]
                    telemetry.write(json.dumps(dict(at=time.time(), cpu=memory, gpu=gpu, native_metrics=metrics)) + '\n')
                    telemetry.flush()
                    time.sleep(5)
            require(client.returncode == 0, 'RULER client failed; inspect client.log')
            complete = json.loads((target / 'answers/completed.json').read_text())
            require(complete['status'] == 'completed' and complete['cases'] == count, 'Incomplete RULER cohort')
            audit_execution(root, count)
            save(status, dict(status='completed', at=time.time(), cases=count, diagnostic_only=diagnostic, score=complete['score'],
                cold_prepare_seconds=prepared - started, inference_seconds=time.monotonic() - prepared))
        except BaseException as error:
            save(status, dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining = lifecycle.stop_owned_groups((client, server))
            save(target / 'cleanup.json', dict(remaining_owned_groups=remaining,
                                               gpu_processes=sorted(lifecycle.gpu_processes())))
            require(not remaining, 'Owned RULER process survived cleanup')
            try:
                finish_service(root, restore_service=restore_service)
            except BaseException as error:
                save(root / 'service-restore-status.json', dict(status='failed', error=repr(error), at=time.time()))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--diagnostic-cases', type=int)
    parser.add_argument('--diagnostic-gen-budget', type=int, choices=(16384, 65536), default=16384)
    parser.add_argument('--diagnostic-chunk', type=int, choices=(1024, 2048), default=1024)
    parser.add_argument('--no-existing-service', action='store_true')
    parser.add_argument('--no-restore-service', action='store_true')
    args = parser.parse_args()
    run(args.campaign, args.diagnostic_cases, args.diagnostic_gen_budget, args.diagnostic_chunk,
        existing_service=not args.no_existing_service, restore_service=not args.no_restore_service)
