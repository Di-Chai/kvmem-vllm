import importlib.util
from pathlib import Path
import subprocess
import sys


spec=importlib.util.spec_from_file_location('deepswe_supervision',Path(__file__).resolve().parents[1]/'benchmarks/deepswe/supervision.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_adoption_tracks_identity_and_does_not_invent_exit_status():
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
    try:
        adopted=module.AdoptedProcess(child.pid)
        assert adopted.poll() is None
        adopted.start-=1
        assert adopted.poll()==0  # A reused PID must never be treated as owned.
        adopted.start+=1
        child.terminate();child.wait(timeout=5)
        assert adopted.poll()==0
        assert adopted.wait(timeout=1) is None
        assert adopted.returncode is None
    finally:
        if child.poll() is None:child.kill();child.wait()
