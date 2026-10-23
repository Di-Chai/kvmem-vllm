"""Run max, Query-only and max+guided serially after the native baseline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):return json.loads(path.read_text())


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def prerequisite_ready(root):
    state=read(root/'status.json')
    if state['status']=='failed':raise RuntimeError('Native baseline failed; inspect it before continuing')
    if state['status']!='completed':return False
    cleanup=root/'run/cleanup.json'
    return cleanup.exists() and read(cleanup)==dict(remaining_owned_groups={},gpu_processes=[])


def run(root):
    root=root.resolve();plan=read(root/'queue.json');child=None;completed=[]
    def stop(*_):raise KeyboardInterrupt('Pooling queue stopped')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with (root/'.queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while not prerequisite_ready(Path(plan['prerequisite'])):
                save(root/'status.json',dict(status='waiting_for_native_baseline',completed=completed,at=time.time()))
                time.sleep(10)
            from ruler_pooling_protocol import ARMS
            if plan['order']!=list(ARMS):raise ValueError('Queue order changed')
            for row in plan['campaigns']:
                campaign=Path(row['path'])
                from benchmark_ruler import sha
                if sha(campaign/'manifest.json')!=row['manifest_sha256']:
                    raise ValueError('Campaign manifest changed')
                if (campaign/'status.json').exists():
                    state=read(campaign/'status.json')
                    if row['arm'] not in plan.get('reuse_completed_controls', ()):
                        raise RuntimeError('Campaign already attempted; preserve it and explicitly plan recovery')
                    audit=read(campaign/'run/execution-audit.json')
                    cleanup=read(campaign/'run/cleanup.json')
                    scores=read(campaign/'run/answers/scores.json')
                    if not (state['status']=='completed' and audit['passed'] and
                            scores['cases']==260 and audit['query_count']==260 and
                            cleanup==dict(remaining_owned_groups={},gpu_processes=[])):
                        raise RuntimeError('Reused control lacks completed audits and cleanup')
                    completed.append(dict(arm=row['arm'],score=state['score'],path=str(campaign),
                                          reused_control=True,at=time.time()))
                    continue
                qualification=plan.get('guided_qualification')
                if qualification:
                    qualified=Path(qualification)
                    qstate=read(qualified/'status.json')
                    qaudit=read(qualified/'run/execution-audit.json')['optimized_execution']['guided']
                    if not (qstate['status']=='completed' and qstate['diagnostic_only']
                            and qaudit['published']>0 and qaudit['rollback_verified']==qaudit['attempts']
                            and read(qualified/'run/cleanup.json')==dict(remaining_owned_groups={},gpu_processes=[])):
                        raise RuntimeError('Guided-pressure GPU qualification has not passed')
                env={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
                env.update(PYTHONPATH=str(root/'runtime/src'),PYTHONDONTWRITEBYTECODE='1',
                    TOKENIZERS_PARALLELISM='false',MAX_JOBS='4',FLASHINFER_NVCC_THREADS='1')
                command=[sys.executable,'-B',str(campaign/'runtime/scripts/run_ruler_campaign.py'),
                    '--campaign',str(campaign),'--no-existing-service','--no-restore-service']
                with (campaign/'controller.log').open('x') as log:
                    child=subprocess.Popen(command,cwd=campaign/'runtime',env=env,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                save(root/'status.json',dict(status='running',arm=row['arm'],controller_pid=child.pid,
                    completed=completed,remaining=plan['order'][len(completed)+1:],at=time.time()))
                save(campaign/'controller-process.json',dict(pid=child.pid,command=command,at=time.time()))
                while child.poll() is None:time.sleep(5)
                if child.returncode!=0:raise RuntimeError('Campaign failed: '+row['arm'])
                state=read(campaign/'status.json');cleanup=read(campaign/'run/cleanup.json')
                if state['status']!='completed' or cleanup!=dict(remaining_owned_groups={},gpu_processes=[]):
                    raise RuntimeError('Campaign not fully completed and released: '+row['arm'])
                completed.append(dict(arm=row['arm'],score=state['score'],path=str(campaign),at=time.time()))
                child=None
            save(root/'status.json',dict(status='completed',completed=completed,at=time.time()))
        except BaseException as error:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait(timeout=180)
            save(root/'status.json',dict(status='failed',error=repr(error),completed=completed,at=time.time()))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
