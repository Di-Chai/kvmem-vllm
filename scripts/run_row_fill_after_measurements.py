"""Finite GPU qualification and measurement dependency; no periodic notifications."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_layerwise_gilopt import read,save_new,sha
from run_recall_comparison import gpu_lock,stop_owned,termination_guard
import benchmark_block_graph as lifecycle

REPO=Path(__file__).resolve().parents[1]
TASK=REPO/'artifacts/multiowner-pressure-fix-20261105-r1'
PRIOR=REPO/'artifacts/multiowner-pressure-fix-bench-20261105-r2'
ROOT=REPO/'artifacts/multiowner-row-fill-20261105-r1'


def status(**value):
    (TASK/'status.json').write_text(json.dumps(dict(at=time.time(),**value),indent=2)+'\n')


def main():
    save_new(TASK/'row-fill-process.json',dict(pid=os.getpid(),at=time.time()))
    child=None
    with termination_guard():
        try:
            deadline=time.monotonic()+1800
            while read(TASK/'status.json')['status']!='measurements_completed':
                if read(PRIOR/'status.json')['status']=='failed' or time.monotonic()>deadline:
                    raise RuntimeError('Prior measurement did not complete; preserve and stop')
                time.sleep(2)
            assert read(PRIOR/'status.json')['status']=='completed'
            status(status='running',stage='row_fill_gpu_qualification',active_root=str(ROOT))
            pointer=REPO/'artifacts/value-recall-active.json';p=read(pointer)
            assert p['queue_root']==str(TASK)
            p['process']=str(TASK/'row-fill-process.json');pointer.write_text(json.dumps(p,indent=2)+'\n')
            assert '94 passed' in (ROOT/'cpu-corrected-fixture-tests.log').read_text()
            for name,digest in read(ROOT/'components/proof.json').items():
                assert sha(ROOT/'components'/name)==digest
            with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
                assert not lifecycle.gpu_processes(), 'Another GPU owner arrived'
                env=dict(os.environ,PYTHONPATH=str(ROOT/'b1/runtime-v1/src')+os.pathsep+str(REPO/'scripts'),KVMEM_RUN_GPU_TESTS='1')
                with (ROOT/'gpu-tests.log').open('x') as log:
                    child=subprocess.Popen([sys.executable,'-m','pytest','-q',str(ROOT/'components/test_multiowner_prefill_gpu.py')],
                        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                if child.wait():raise RuntimeError('Preserved GPU component failure')
                assert '8 passed' in (ROOT/'gpu-tests.log').read_text()
                assert not lifecycle.gpu_processes()
            save_new(ROOT/'qualification.json',dict(passed=True,cpu_passed=94,gpu_passed=8,
                runtime_proof_sha256=sha(ROOT/'b1/runtime-proof.json'),
                files={str(ROOT/n):sha(ROOT/n) for n in ('cpu-corrected-fixture-tests.log','gpu-tests.log','components/proof.json')},
                scope='Full448 native sample rows across six uneven and eight mixed owners; original FA2 output byte equality, original FP32 oracle tolerance, full17layer shared DMA bytes.'))
            status(status='running',stage='row_fill_unprofiled_full_input',active_root=str(ROOT))
            with (TASK/'row-fill-driver.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-B',str(ROOT/'launch.py'),'run','--root',str(ROOT)],
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save_new(TASK/'row-fill-child.json',dict(pid=child.pid,root=str(ROOT),at=time.time()))
            if child.wait():raise RuntimeError('Preserved row-fill measurement failure')
            status(status='measurements_completed',stage='awaiting_efficiency_review',active_root=str(ROOT))
        except BaseException as error:
            status(status='failed',error=repr(error));raise
        finally:stop_owned(child)


if __name__=='__main__':main()
