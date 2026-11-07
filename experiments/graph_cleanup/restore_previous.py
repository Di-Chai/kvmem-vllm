"""Restore the previously idle owned service after the isolated run finishes."""
import json
from pathlib import Path
import shlex
import subprocess
import time
from urllib.request import build_opener,ProxyHandler


ROOT=Path.cwd()
RUNTIME=Path('/root/data/systemone-multifork-4090-20261105-r1/runtime')
opener=build_opener(ProxyHandler({}))


def models(port):
    try:
        with opener.open(f'http://127.0.0.1:{port}/v1/models',timeout=2) as response:
            return json.load(response)
    except Exception:
        return None


def main():
    # Never stop a service from this helper. The experiment coordinator owns
    # its API shutdown; this helper only starts the saved prior configuration.
    if models(18996):raise RuntimeError('Experiment API still active')
    existing=models(18995)
    if existing:
        if not any(m['id']=='systemone-fork-4090' for m in existing['data']):
            raise RuntimeError('A different service now occupies the prior port')
        (ROOT/'restored-service.json').write_text(json.dumps(dict(restored=True,already_present=True,models=existing,time=time.time()),indent=2))
        return
    previous=json.loads((ROOT/'previous-server.json').read_text())
    argv=list(previous['argv'])
    assert argv[0]=='/root/data/agent-execution-observer-20261104-r1/venv/bin/python'
    assert argv[argv.index('--config')+1]=='/root/data/systemone-multifork-4090-20261105-r1/profile-4-timing.json'
    argv[argv.index('--run-dir')+1]='/dev/shm/systemone-restored-after-graph-cleanup-20261105'
    if Path(argv[argv.index('--run-dir')+1]).exists():raise RuntimeError('Restore run directory already exists; inspect it first')
    shell=('source /root/data/kvmem-vllm-gc/tools/cuda130-env.sh && '
           'export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 && '
           'export PYTHONPATH='+shlex.quote(str(RUNTIME/'src')+':'+str(RUNTIME/'scripts'))+' && exec '+shlex.join(argv))
    with (ROOT/'restored-service.log').open('x') as log:
        process=subprocess.Popen(['bash','-c',shell],cwd=RUNTIME,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'restored-service.process.json').write_text(json.dumps(dict(pid=process.pid,argv=argv,cwd=str(RUNTIME)),indent=2))
    for _ in range(600):
        if process.poll() is not None:raise RuntimeError('Restored service exited during startup')
        state=models(18995)
        if state and any(m['id']=='systemone-fork-4090' for m in state['data']):
            with opener.open('http://127.0.0.1:18995/metrics',timeout=5) as response:
                metrics=response.read().decode()
            (ROOT/'restored-service.json').write_text(json.dumps(dict(restored=True,pid=process.pid,models=state,
                request_metrics=[line for line in metrics.splitlines() if line.startswith(('vllm:num_requests_running','vllm:num_requests_waiting'))],
                time=time.time()),indent=2))
            print('Previous System One service restored and healthy',flush=True)
            return
        time.sleep(2)
    raise RuntimeError('Restored service health timeout; process retained for inspection')


if __name__=='__main__':main()
