"""Finite dependency queue: finish the owned trace, then four measured runs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_layerwise_gilopt import read, save_new
from run_recall_comparison import stop_owned, termination_guard

REPO = Path(__file__).resolve().parents[1]
TASK = REPO/'artifacts/multiowner-pressure-fix-20261105-r1'
TRACE = REPO/'artifacts/multiowner-6000-baseline-profile-20261105-r1'
BENCH = REPO/'artifacts/multiowner-pressure-fix-bench-20261105-r2'


def status(**value):
    (TASK/'status.json').write_text(json.dumps(dict(at=time.time(),**value),indent=2)+'\n')


def main():
    assert read(TRACE/'status.json')['status'] in ('running','completed')
    assert read(BENCH/'status.json')['status']=='prepared'
    save_new(TASK/'process.json',dict(pid=os.getpid(),at=time.time()))
    pointer=REPO/'artifacts/value-recall-active.json'
    save_new(TASK/'previous-pointer.json',read(pointer))
    pointer.write_text(json.dumps(dict(queue_root=str(TASK),root=str(TASK),
        status=str(TASK/'status.json'),process=str(TASK/'process.json'),
        experiment_type='finite_multiowner_pressure_optimization',at=time.time()),indent=2)+'\n')
    child=None
    with termination_guard():
        try:
            status(status='running',stage='baseline_trace',active_root=str(TRACE))
            deadline=time.monotonic()+1800
            while True:
                observed=read(TRACE/'status.json')['status']
                if observed=='completed':break
                if observed!='running' or time.monotonic()>deadline:
                    raise RuntimeError('Trace did not complete; do not start measurements')
                time.sleep(2)
            assert not read(TRACE/'cleanup.json')['gpu_processes']
            status(status='running',stage='unprofiled_full_input',active_root=str(BENCH))
            with (TASK/'measurement-driver.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-B',str(BENCH/'launch.py'),
                    'run','--root',str(BENCH)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_new(TASK/'child.json',dict(pid=child.pid,root=str(BENCH),at=time.time()))
            if child.wait():raise RuntimeError('Measurement stopped; first evidence preserved')
            status(status='measurements_completed',stage='awaiting_model_qualification',active_root=str(BENCH))
        except BaseException as error:
            status(status='failed',error=repr(error));raise
        finally:stop_owned(child)


if __name__=='__main__':main()
