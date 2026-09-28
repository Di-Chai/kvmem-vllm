"""Keep one frozen qualified backend recipe alive for real agent attempts.

Owns only the launched API process group and shared GPU test lock. Native Q8
and experimental Tensor Core runs use separate, explicit qualified recipes.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

from qualify_engine import freeze_worker_package, stop_owned_server


def server_command(recipe, python, port):
    if recipe.get('status') != 'passed' or not recipe.get('performance_eligible'):
        raise ValueError('Agent backend requires a completed uninstrumented qualification recipe')
    command = list(recipe['command'])
    if command[1:6] != ['-m', 'vllm_kvmem', '--draft-load-compat', 'vllm', 'serve']:
        raise ValueError('Unrecognized qualification server command')
    if '--profiler-config' in command:
        raise ValueError('Profiler recipes cannot be used for agent quality/performance')
    command[0] = str(python)
    command[command.index('--port') + 1] = str(port)
    command[command.index('--host') + 1] = '127.0.0.1'
    command += ['--enable-per-request-metrics', '--per-request-spec-decode-metrics', 'summary',
                '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_xml',
                '--reasoning-parser', 'qwen3']
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True, help='Completed qualification results.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    recipe = json.loads(args.recipe.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stopped = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    report = {'scope': 'shared API for executed agent tasks; no persistent-session claim',
              'manager_pid': os.getpid(), 'recipe': str(args.recipe.resolve()),
              'recipe_sha256': hashlib.sha256(args.recipe.read_bytes()).hexdigest(), 'state': 'starting'}

    def save():
        temporary = out / 'server_state.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'server_state.json')

    with open('/tmp/kvmem-vllm-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if active:
            raise RuntimeError('GPU is occupied; existing processes are never stopped: ' + active)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        command = server_command(recipe, sys.executable, port)
        files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
        manifest = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files if name and (root / name).is_file()}
        (out / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
        worker, worker_hash = freeze_worker_package(root, out, manifest)
        env = {k: v for k, v in os.environ.items() if not k.startswith('KVMEM_')}
        env.update(recipe['plugin_options'])
        env.update(KVMEM_RUNNER_PROBE_DIR=str(out / 'capture'), HF_HUB_OFFLINE='1',
                   VLLM_USE_V2_MODEL_RUNNER='1', VLLM_PLUGINS='gguf,kvmem', MAX_JOBS='2',
                   CUDA_HOME='/usr/local/cuda-13', PYTHONPATH=str(worker))
        env['PATH'] = f'{Path(sys.executable).parent}:/usr/local/cuda-13/bin:' + env['PATH']
        report.update(command=command, port=port, api_base=f'http://127.0.0.1:{port}/v1',
                      source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                      worker_manifest_sha256=worker_hash,
                      plugin_options={k: v for k, v in env.items() if k.startswith('KVMEM_')})
        save()
        process = monitor = None
        try:
            with (out / 'server.log').open('wb') as log:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                report['server_pid'] = process.pid
                save()
                from vllm_kvmem.telemetry import ResourceMonitor
                monitor = ResourceMonitor(out / 'resources.jsonl', process.pid).start()
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 900
                while process.poll() is None and not stopped.is_set():
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2):
                            report['state'] = 'ready'
                        save()
                        print(json.dumps({'state': 'ready', 'api_base': report['api_base'], 'evidence': str(out / 'server_state.json')}), flush=True)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError('Owned agent backend did not become ready')
                        stopped.wait(1)
                while process.poll() is None and not stopped.wait(1):
                    pass
                if not stopped.is_set():
                    raise RuntimeError(f'Agent backend exited unexpectedly: {process.returncode}')
        except BaseException as error:
            report['failure'] = repr(error)
            raise
        finally:
            stop_owned_server(process)
            if monitor:
                report['resources'] = monitor.stop()
            report.update(state='failed' if 'failure' in report else 'stopped',
                          exit_code=process.returncode if process else None)
            save()


if __name__ == '__main__':
    main()
