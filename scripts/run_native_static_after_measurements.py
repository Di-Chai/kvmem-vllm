"""Finite component qualification and full-input measurement dependency."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_layerwise_gilopt import read, save_new, sha
from run_recall_comparison import gpu_lock, stop_owned, termination_guard
import benchmark_block_graph as lifecycle

REPO = Path(__file__).resolve().parents[1]
TASK = REPO/'artifacts/multiowner-pressure-fix-20261105-r1'
PRIOR = REPO/'artifacts/multiowner-row-fill-20261105-r1'
ROOT = REPO/'artifacts/multiowner-static-native-20261105-r3'
PREFIX = 'native-static-v3'


def status(**value):
    (TASK/'status.json').write_text(json.dumps(dict(at=time.time(), **value), indent=2)+'\n')


def main():
    save_new(TASK/(PREFIX+'-process.json'), dict(pid=os.getpid(), at=time.time()))
    child = None
    with termination_guard():
        try:
            deadline = time.monotonic()+1800
            while read(PRIOR/'status.json')['status'] != 'completed':
                if read(PRIOR/'status.json')['status'] == 'failed' or time.monotonic() > deadline:
                    raise RuntimeError('Prior measurement did not complete; preserve and stop')
                time.sleep(2)
            status(status='running', stage='native_static_qualification', active_root=str(ROOT))
            pointer = REPO/'artifacts/value-recall-active.json'; previous = read(pointer)
            assert previous['queue_root'] == str(TASK)
            previous['process'] = str(TASK/(PREFIX+'-process.json'))
            pointer.write_text(json.dumps(previous, indent=2)+'\n')
            for name, digest in read(ROOT/'components/proof.json').items():
                assert sha(ROOT/'components'/name) == digest
            with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
                assert not lifecycle.gpu_processes(), 'Another GPU owner arrived'
                env = dict(os.environ, KVMEM_RUN_GPU_TESTS='1',
                    PYTHONPATH=str(ROOT/'b1/runtime-v1/src')+os.pathsep+str(ROOT/'components'))
                tests = ['test_native_prefill_static_aligned.py', 'test_native_prefill_static_aligned_gpu.py',
                         'test_multiowner_prefill_gpu.py']
                with (ROOT/'component-tests.log').open('x') as log:
                    child = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        *[str(ROOT/'components'/n) for n in tests]], env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                if child.wait(): raise RuntimeError('Preserved native static component failure')
                assert '71 passed' in (ROOT/'component-tests.log').read_text()
                assert not lifecycle.gpu_processes()
            save_new(ROOT/'qualification.json', dict(passed=True, tests=71,
                runtime_proof_sha256=sha(ROOT/'b1/runtime-proof.json'),
                files={str(ROOT/n):sha(ROOT/n) for n in ('component-tests.log', 'components/proof.json')},
                scope='Static versus generic full LSE, scores and attention bytes; six/eight owner logical statistics and FA2 output; shared KV DMA bytes. No throughput claim.'))
            status(status='running', stage='native_static_unprofiled_full_input', active_root=str(ROOT))
            with (TASK/(PREFIX+'-driver.log')).open('x') as log:
                child = subprocess.Popen([sys.executable, '-B', str(ROOT/'launch.py'), 'run', '--root', str(ROOT)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save_new(TASK/(PREFIX+'-child.json'), dict(pid=child.pid, root=str(ROOT), at=time.time()))
            if child.wait(): raise RuntimeError('Preserved native static measurement failure')
            status(status='measurements_completed', stage='awaiting_efficiency_review', active_root=str(ROOT))
        except BaseException as error:
            status(status='failed', error=repr(error)); raise
        finally: stop_owned(child)


if __name__ == '__main__': main()
