"""Own one GPU worker at a time for the frozen guided off/on quality pairs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import benchmark_block_graph as lifecycle
from guided_benchmark_protocol import sha, require, validate_pair
from qualify_block_graph import save

ORDER = [('niah', 'off'), ('niah', 'guided'),
         ('longmemeval_s', 'guided'), ('longmemeval_s', 'off'),
         ('memoryagentbench', 'off'), ('memoryagentbench', 'guided')]


def run(root):
    root = root.resolve()
    runtime = root/'runtime'
    manifest = json.loads((root/'manifest.json').read_text())
    proof = json.loads((root/'runtime-proof.json').read_text())
    require(Path(__file__).resolve() == runtime/'scripts/run_guided_campaign.py',
            'Run the frozen controller')
    for name, digest in proof['files'].items():
        require(sha(runtime/name) == digest, 'Frozen runtime changed: '+name)
    for name, digest in manifest['profile_hashes'].items():
        require(sha(root/name) == digest, 'Frozen profile changed: '+name)
    for suite in {suite for suite, _ in ORDER}:
        validate_pair(*(json.loads((root/f'{suite}-{arm}.json').read_text())
                        for arm in ('off', 'guided')), suite)
    env = {k:v for k,v in os.environ.items()
           if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime/'src'), PYTHONDONTWRITEBYTECODE='1',
               TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1',
               VLLM_LOG_STATS_INTERVAL='1')
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit)/'bin/nvcc').exists():
            env['CUDA_HOME'] = toolkit
            break
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with (root/'.campaign.lock').open('a+') as campaign_lock, \
            open('/tmp/kvmem-coordination-qualification.lock', 'a+') as gpu_lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        for lock in (campaign_lock, gpu_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not lifecycle.gpu_processes(), 'GPU occupied; leave unrelated processes intact')
        save(root/'status.json', dict(status='running', order=ORDER, started=time.time()))
        try:
            for suite, arm in ORDER:
                name = f'{suite}-{arm}'
                target = root/name
                status = target/'status.json'
                if status.exists():
                    require(json.loads(status.read_text())['status'] == 'completed',
                            'Retain failed/interrupted attempts: '+name)
                    continue
                target.mkdir(exist_ok=True)
                profile = root/(name+'.json')
                cfg = json.loads(profile.read_text())
                command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve',
                           '--config', str(profile), '--run-dir', str(target/'server')]
                server = client = None
                started = time.monotonic()
                try:
                    with (target/'server.log').open('x') as log:
                        server = subprocess.Popen(command, cwd=runtime, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    save(status, dict(status='loading', server_pid=server.pid, at=time.time()))
                    for _ in range(1200):
                        require(server.poll() is None, 'Server exited while loading '+name)
                        try:
                            with opener.open(f"http://127.0.0.1:{cfg['port']}/health", timeout=1) as response:
                                if response.status == 200:
                                    break
                        except OSError:
                            pass
                        time.sleep(1)
                    else:
                        raise TimeoutError('Model startup exceeded 20 minutes')
                    prepared = time.monotonic()
                    command = [sys.executable, '-B', str(runtime/'scripts/benchmark_guided.py'),
                               'run', '--campaign', str(root), '--suite', suite, '--arm', arm]
                    with (target/'client.log').open('x') as log:
                        client = subprocess.Popen(command, cwd=runtime, env=env,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    save(status, dict(status='testing', server_pid=server.pid, client_pid=client.pid,
                        at=time.time(), cold_prepare_seconds=prepared-started))
                    print(json.dumps(dict(job=name, status='testing')), flush=True)
                    with (target/'resources.jsonl').open('x') as telemetry:
                        while client.poll() is None:
                            require(server.poll() is None, 'Server exited while testing '+name)
                            require(time.monotonic()-prepared < 12*3600, 'Suite exceeded 12 hours')
                            memory = {line.split(':')[0]:int(line.split()[1])*1024
                                      for line in Path('/proc/meminfo').read_text().splitlines()
                                      if line.startswith(('MemAvailable:', 'MemTotal:'))}
                            gpu = subprocess.check_output(['nvidia-smi',
                                '--query-gpu=memory.used,memory.free,utilization.gpu',
                                '--format=csv,noheader,nounits'], text=True).strip()
                            telemetry.write(json.dumps(dict(at=time.time(), cpu=memory, gpu=gpu))+'\n')
                            telemetry.flush()
                            time.sleep(5)
                    require(client.returncode == 0, 'Client failed; inspect '+str(target/'client.log'))
                    complete = json.loads((target/'answers/completed.json').read_text())
                    require(complete['status'] == 'completed', 'Missing completion proof')
                    save(status, dict(status='completed', at=time.time(), cases=complete['cases'],
                        cold_prepare_seconds=prepared-started, inference_seconds=time.monotonic()-prepared))
                    print(json.dumps(dict(job=name, status='completed')), flush=True)
                except BaseException as error:
                    save(status, dict(status='failed', error=repr(error), at=time.time()))
                    raise
                finally:
                    remaining = lifecycle.stop_owned_groups((client, server))
                    save(target/'cleanup.json', dict(remaining_owned_groups=remaining,
                                                    gpu_processes=sorted(lifecycle.gpu_processes())))
                    require(not remaining, 'Owned processes survived cleanup')
            save(root/'status.json', dict(status='inference-completed', order=ORDER, at=time.time()))
        except BaseException as error:
            save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    run(parser.parse_args().campaign)
