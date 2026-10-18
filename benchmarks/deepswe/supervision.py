"""Transfer supervision without restarting the owned benchmark workloads."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def identity(pid):
    tail=(Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()
    return {'state':tail[0],'parent':int(tail[1]),'start':int(tail[19])}


class AdoptedProcess:
    # An adopted shell's exit status cannot be waited for by its new owner.
    # The frozen cell's durable agent/verifier exit statuses remain authoritative.
    adopted=True

    def __init__(self,pid):
        self.pid=pid;self.start=identity(pid)['start'];self.returncode=None
        assert os.getpgid(pid)==pid, 'Only separately owned process groups can be adopted'

    def poll(self):
        try:current=identity(self.pid)
        except FileNotFoundError:return 0
        return 0 if current['start']!=self.start or current['state']=='Z' else None

    def wait(self,timeout=None):
        deadline=time.monotonic()+timeout if timeout is not None else float('inf')
        while self.poll() is None:
            if time.monotonic()>=deadline:raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(.1)
        return None


def inventory(controller,root,output):
    proc=Path('/proc')/str(controller)
    args=(proc/'cmdline').read_bytes().split(b'\0')
    assert str((proc/'cwd').resolve())==str(root.resolve())
    assert any(Path(os.fsdecode(arg)).name=='benchmark_deepswe_first16.py' for arg in args if arg)
    assert b'--output' in args
    assert (root/os.fsdecode(args[args.index(b'--output')+1])).resolve()==output.resolve()
    state=json.loads((output/'run.json').read_text())
    assert state['status']=='running'
    found={'controller':controller,'controller_start':identity(controller)['start'],'cells':{}}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:
            if identity(int(entry.name))['parent']!=controller:continue
            child=[os.fsdecode(v) for v in (entry/'cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,ProcessLookupError):continue
        pid=int(entry.name)
        if str(output/'setup/run_deepswe_cell.sh') in child:
            at=child.index(str(output/'setup/run_deepswe_cell.sh'));task=child[at+1]
            assert child[at+2]==str(output/'cases'/task)
            assert task in state['tasks']
            found['cells'][task]=pid
        elif str(root/'benchmarks/deepswe/claude_bridge.py') in child:
            assert child[child.index('--output')+1]==str(output/'bridge')
            found['bridge']=pid
        elif 'vllm_kvmem' in child and 'serve' in child:
            assert child[child.index('--run-dir')+1]==str(output/'server')
            found['server']=pid
    assert 'bridge' in found and 'server' in found, 'Owned native and bridge processes must both remain alive'
    assert set(found['cells'])=={task for task,value in state['tasks'].items() if value['state']=='running'}
    return found


def adopt(controller,root,output):
    before=inventory(controller,root,output)
    os.kill(controller,signal.SIGSTOP)
    try:
        found=inventory(controller,root,output)
        assert before['controller_start']==found['controller_start']
        network=json.loads(subprocess.check_output(['docker','network','inspect',json.loads((output/'run.json').read_text())['network']]))[0]
        assert network['Labels']['com.kvmem.deepswe.owner']==json.loads((output/'run.json').read_text())['id']
        for task in found['cells']:
            cid_path=output/'cases'/task/'agent.container.cid'
            if not cid_path.exists():continue
            container=json.loads(subprocess.check_output(['docker','inspect',cid_path.read_text().strip()]))[0]
            labels=container['Config']['Labels']
            assert labels['com.qw3.deepswe.run-owner']==hashlib.sha256(str(output/'cases'/task).encode()).hexdigest()[:16]
        children={name:AdoptedProcess(found[name]) for name in ('server','bridge')}
        cells={task:AdoptedProcess(pid) for task,pid in found['cells'].items()}
        env=(Path('/proc')/str(found['bridge'])/'environ').read_bytes().split(b'\0')
        token=next(v.split(b'=',1)[1].decode() for v in env if v.startswith(b'DEEPSWE_CONTROL_TOKEN='))
        # State, native processes, live HTTP streams and containers continue.
        # The old supervisor alone is stopped; its finally block must not kill
        # these transferred process groups.
        os.kill(controller,signal.SIGKILL)
        return children,cells,token,found
    except BaseException:
        os.kill(controller,signal.SIGCONT)
        raise
