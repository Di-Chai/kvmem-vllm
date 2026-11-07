"""Run the remaining owned experiments after the paired worker exits.

The service is loopback-only; the original service is restored separately after
inspection. A failed stage keeps its raw evidence and stops this coordinator.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import build_opener, ProxyHandler


ROOT = Path.cwd()
CODE = ROOT / 'experiments/graph_cleanup'
PY = sys.executable


def save(name, data):
    (ROOT/name).write_text(json.dumps(data, indent=2)+'\n')


def stage(name, command):
    save('pipeline-status.json', dict(stage=name, started=time.time(), command=command))
    tag=name;attempt=0
    while (ROOT/f'{tag}.log').exists():
        attempt+=1;tag=f'{name}.retry-{attempt}'
    with (ROOT/f'{tag}.log').open('x') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        save(f'{tag}.process.json', dict(pid=proc.pid, command=command))
        code = proc.wait()
    if code:
        raise RuntimeError(f'{name} exited {code}')


def main():
    save('pipeline-status.json', dict(stage='waiting_for_paired', started=time.time()))
    while True:
        state = json.loads((ROOT/'paired-r1/status.json').read_text())
        failure = ROOT/'paired-r1/failure.json'
        if failure.exists():
            raise RuntimeError(f'Paired run failed: {failure.read_text()}')
        if state['phase'] == 'completed':
            break
        time.sleep(10)
    # The paired worker explicitly shuts its engine down in finally. Wait for
    # that specific PID to exit before allocating another model.
    pid = int((ROOT/'paired-r1.pid').read_text())
    for _ in range(60):
        if not Path(f'/proc/{pid}').exists():
            break
        stat = Path(f'/proc/{pid}/stat').read_text().split()
        if stat[2] == 'Z':
            break
        time.sleep(2)
    else:
        raise RuntimeError('Completed paired process did not exit')
    if not (ROOT/'dose-r1/status.json').exists():
        stage('dose-r1', [PY, '-u', str(CODE/'dose_study.py'),
              '--inference', str(ROOT/'paired-r1'), '--output', str(ROOT/'dose-r1')])
    else:
        assert json.loads((ROOT/'dose-r1/status.json').read_text())['status']=='completed'
    if not (ROOT/'recovery-r1/results.json').exists():
        stage('recovery-r1', [PY, '-u', str(CODE/'recovery_study.py'),
              '--archive', str(ROOT/'paired-r1/recovery/attention-kv.pt'),
              '--output', str(ROOT/'recovery-r1')])
    else:
        recovery=json.loads((ROOT/'recovery-r1/results.json').read_text())
        assert len(recovery)==4 and all(r['raw_copy_verified'] for r in recovery)
    command = [PY, '-u', '-m', 'vllm.entrypoints.openai.api_server',
        '--model', '/root/data/kvmem-vllm-gc/models/Qwen3.8-27B-GPTQ-Int4',
        '--tokenizer', '/root/data/recall-comparison-20261028-r1/assets/tokenizer',
        '--served-model-name', 'cleanup-4090', '--host', '127.0.0.1', '--port', '18996',
        '--dtype', 'bfloat16', '--quantization', 'gptq_marlin',
        '--max-model-len', '34816', '--max-num-seqs', '16', '--max-num-batched-tokens', '1024',
        '--gpu-memory-utilization', '.90', '--kv-cache-dtype', 'fp8_e4m3',
        '--language-model-only', '--enforce-eager', '--no-enable-prefix-caching',
        '--generation-config', 'vllm', '--enable-per-request-metrics',
        '--attention-config', '{"backend":"FLASHINFER"}']
    save('pipeline-status.json', dict(stage='starting_api', started=time.time()))
    attempt=1
    while (ROOT/f'api-r{attempt}.log').exists():
        attempt+=1
    with (ROOT/f'api-r{attempt}.log').open('x') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    save(f'api-r{attempt}.process.json', dict(pid=server.pid, command=command,
          environment={k:os.environ.get(k) for k in ('VLLM_PLUGINS','PYTHONNOUSERSITE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
    try:
        opener = build_opener(ProxyHandler({}))
        # This host stores code/model files on Ceph; cold metadata reads can
        # exceed several minutes. Startup is outside all measured workloads.
        for _ in range(600):
            if server.poll() is not None:
                raise RuntimeError('API server exited during startup')
            try:
                with opener.open('http://127.0.0.1:18996/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise RuntimeError('API server health timeout')
        # Keep GPU monitoring separate from request timings; no trace content.
        with (ROOT/f'gpu-telemetry-api-r{attempt}.csv').open('x') as log:
            monitor = subprocess.Popen(['nvidia-smi',
                '--query-gpu=timestamp,memory.used,utilization.gpu,utilization.memory,power.draw,temperature.gpu',
                '--format=csv', '-l', '1'], stdout=log, stderr=subprocess.STDOUT)
        try:
            if not (ROOT/'judge-r1/graph-observer/cpu-costs.json').exists():
                stage('judge-r1', [PY, '-u', str(CODE/'service_study.py'), 'judge',
                      '--inference', str(ROOT/'paired-r1'), '--output', str(ROOT/'judge-r1')])
            if not (ROOT/'judge-r2/status.json').exists():
                diagnostic=['--continue-recorded-controls'] if (ROOT/'judge-r2/controls.json').exists() else []
                stage('judge-r2', [PY, '-u', str(CODE/'judge_revision.py'),
                      '--inference', str(ROOT/'paired-r1'), '--output', str(ROOT/'judge-r2'), *diagnostic])
            resume=['--resume'] if (ROOT/'serving-r1').exists() else []
            stage('serving-r1', [PY, '-u', str(CODE/'service_study.py'), 'matrix',
                  '--inference', str(ROOT/'paired-r1'), '--output', str(ROOT/'serving-r1'), *resume])
        finally:
            monitor.terminate()
            monitor.wait(timeout=10)
        save('pipeline-status.json', dict(stage='completed', ended=time.time()))
    finally:
        # Only this coordinator's isolated API process group is stopped.
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=45)
            except subprocess.TimeoutExpired:
                save('api-shutdown-pending.json', dict(pid=server.pid))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        save('pipeline-failure.json', dict(error=repr(exc), time=time.time()))
        raise
