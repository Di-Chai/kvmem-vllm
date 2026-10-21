"""Restore the owned DSH service after GPU tests, then assemble complete scores."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import benchmark_block_graph as lifecycle
from benchmark_niah import save
from qualify_block_graph import require


def healthy(url, process, timeout=1200):
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        require(process.poll() is None, 'Owned service exited: '+str(process.pid))
        try:
            with http.open(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError('Service health deadline exceeded')


def start_model(root, runtime, profile, name):
    target = root/name
    target.mkdir(exist_ok=False)
    command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve',
               '--config', str(profile), '--run-dir', str(target/'server')]
    env = {k:v for k,v in os.environ.items()
           if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').exists():
            env['CUDA_HOME'] = toolkit
            break
    with (target/'server.log').open('x') as log:
        process = subprocess.Popen(command, cwd=runtime, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    info = dict(pid=process.pid, command=command, runtime=str(runtime), profile=str(profile),
                started_at=time.time(), log=str(target/'server.log'))
    save(target/'process.json', info)
    try:
        healthy('http://127.0.0.1:18846/health', process)
    except BaseException:
        lifecycle.stop_owned_groups((process,))
        raise
    return process, info, env, target


def restore(args, success):
    root = args.restore.resolve()
    plan = json.loads((root/'restore-plan.json').read_text())
    candidate = root/'runtime'
    qualified = False
    with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(not lifecycle.gpu_processes(), 'GPU is occupied; do not stop another worker')
        if success:
            process = None
            try:
                process, _, env, target = start_model(root, candidate, root/'qualification.json', 'qualification')
                command = [sys.executable, '-B', str(candidate/'scripts/qualify_guided_refresh.py'),
                    '--config', str(root/'qualification.json'), '--capture', str(target/'server/capture'),
                    '--output', str(target/'answers')]
                with (target/'client.log').open('x') as log:
                    result = subprocess.run(command, env=env, cwd=candidate,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=3600)
                qualified = result.returncode == 0 and json.loads((target/'answers/results.json').read_text())['status'] == 'passed'
                require(qualified, 'Full-budget guided lifecycle gate failed')
            except Exception as error:
                save(root/'qualification-failure.json', dict(error=repr(error), at=time.time()))
            finally:
                lifecycle.stop_owned_groups((process,))
        runtime = candidate if qualified else Path(plan['previous_runtime'])
        profile = root/('production.json' if qualified else 'previous-profile.json')
        process, info, env, target = start_model(root, runtime, profile, 'production')
        if qualified:
            bridge = plan['bridge']
            pid = bridge['pid']
            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            expected = [s.encode() for s in bridge['command']]
            require(cmdline[:-1] == expected and os.getpgid(pid) == pid,
                    'DSH bridge owner changed; leave it intact and record manual integration needed')
            os.killpg(pid, signal.SIGTERM)
            deadline = time.monotonic()+30
            while lifecycle.live_group_members(pid) and time.monotonic() < deadline:
                time.sleep(.2)
            require(not lifecycle.live_group_members(pid), 'Old bridge did not stop')
            command = list(bridge['command'])
            previous_output = command[command.index('--output')+1]
            command[command.index('--output')+1] = str(root/'bridge')
            command += ['--restore-from', previous_output]
            with (root/'bridge.log').open('x') as log:
                replacement = subprocess.Popen(command, cwd=candidate,
                    env=dict(env, PYTHONPATH=str(candidate/'src')+':'+str(candidate)),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            bridge = dict(pid=replacement.pid, command=command, cwd=str(candidate),
                          log=str(root/'bridge.log'), started_at=time.time())
            save(root/'bridge-process.json', bridge)
            healthy('http://127.0.0.1:18385/health', replacement, 60)
        else:
            bridge = plan['bridge']
        service_path = Path(plan['service_json'])
        service = json.loads(service_path.read_text())
        service.update(pid=process.pid, current_native=str(target/'process.json'), profile=str(profile),
                       runtime=str(runtime/'src'), continuation_bridge=bridge)
        service['guided_refresh_lifecycle'] = dict(qualified=qualified, evidence=str(root/'qualification/answers/results.json'))
        save(service_path, service)
        status_path = service_path.with_name('status.json')
        status = json.loads(status_path.read_text())
        status.update(status='running_guided_epoch_qualified' if qualified else 'running_previous_profile_after_test',
                      backend_pid=process.pid, backend_profile=str(profile), bridge_pid=bridge['pid'])
        save(status_path, status)
        save(root/'status.json', dict(status='running', qualified_new_lifecycle=qualified,
                                     model=info, bridge=bridge, at=time.time()))


def run(args):
    root = args.campaign.resolve()
    while True:
        status = json.loads((root/'status.json').read_text())
        if status['status'] in ('failed', 'inference-completed'):
            break
        time.sleep(20)
    try:
        restore(args, status['status'] == 'inference-completed')
        if status['status'] == 'failed':
            save(root/'completion-status.json', dict(status='inference-failed-service-restored'))
            return
        while True:
            grading = json.loads((root/'grading-status.json').read_text())
            if grading['status'] == 'completed':
                break
            pid = json.loads((root/'grading-process.json').read_text())['pid']
            process = Path(f'/proc/{pid}')
            require(process.exists() and
                    (process/'stat').read_text().rsplit(')', 1)[1].split()[0] not in ('Z', 'X')
                    and b'analyze_guided.py' in (process/'cmdline').read_bytes(),
                    'Grader exited before completing both arms')
            time.sleep(20)
        command = [sys.executable, '-B', str(root/'runtime/scripts/summarize_guided.py'),
                   '--campaign', str(root)]
        with (root/'comparison.log').open('x') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env=dict(os.environ, PYTHONPATH=str(root/'runtime/src')))
        args.credential_file.unlink(missing_ok=True)
        save(root/'completion-status.json', dict(status='completed', at=time.time(),
                                                comparison=str(root/'comparison.json')))
    except BaseException as error:
        save(root/'completion-status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--restore', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    run(parser.parse_args())
