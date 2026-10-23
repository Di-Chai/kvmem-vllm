"""One-shot postprocessing dependency, not a recurring monitor or wakeup task."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):return json.loads(path.read_text())
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def finish(root,timeout):
    state_path=root/'analysis-status.json'
    with open('/tmp/kvmem-retrieval-analysis-'+root.name+'.lock','a+') as once:
        fcntl.flock(once,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+timeout
        save(state_path,dict(status='waiting_for_capture',pid=os.getpid(),at=time.time()))
        while read(root/'status.json')['status']!='captured':
            if read(root/'status.json')['status']=='failed':raise RuntimeError('Capture failed; no automatic retry')
            if time.monotonic()>deadline:raise TimeoutError('Capture dependency timeout')
            time.sleep(10)
        import benchmark_block_graph as lifecycle
        with open('/tmp/kvmem-coordination-qualification.lock','a+') as gpu_lock:
            # Capture publishes its terminal status just before releasing this lock.
            while True:
                try:fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()>deadline:raise TimeoutError('GPU coordination lock remains held')
                    time.sleep(5)
            if lifecycle.gpu_processes():raise RuntimeError('Foreign GPU owner; do not interrupt it')
            fixture=read(root/'fixture.json');answers={}
            for arm in ['compressed','full']:
                if read(root/arm/'status.json')['status']!='completed':
                    raise ValueError('Incomplete capture arm')
                cleanup=read(root/arm/'cleanup.json')
                if cleanup['remaining_owned_groups'] or cleanup['gpu_processes']:raise ValueError('Capture cleanup incomplete')
                for row in fixture['rows']:
                    directory=root/arm/'cases'/row['id']
                    request=read(directory/'request.json')
                    if request['input_sha256']!=row['served_sha256'] or request['query_span']!=row['served_query']:
                        raise ValueError('Original input or Query changed')
                    if request['sampling']!=dict(fixture['sampling'],seed=row['seed'],max_tokens=1):
                        raise ValueError('Sampling changed')
                    if digest(Path(fixture['input_directory'])/(row['id']+'.i32'))!=row['served_sha256']:
                        raise ValueError('Fixture bytes changed')
                    for name in ['response.json','result.json','request.json']:
                        path=directory/name;answers[str(path.relative_to(root))]=digest(path)
            save(root/'analysis-input-hashes.json',answers)
            save(state_path,dict(status='analyzing',pid=os.getpid(),at=time.time()))
            command=[sys.executable,'-B',str(Path(__file__).with_name('analyze_ruler_retrieval_diagnostic.py')),
                     '--root',str(root)]
            with (root/'analysis.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            if any(digest(root/p)!=h for p,h in answers.items()):raise ValueError('Captured result bytes changed')
            summary=read(root/'analysis/summary.json')
            if summary['cases']!=len(fixture['rows']):raise ValueError('Missing paired analysis')
            for row in summary['rows']:
                for native in row['native_attention'].values():
                    if not (.99999<native['normalized_row_mass_min']<=native['normalized_row_mass_max']<1.00001):
                        raise ValueError('Native attention denominator failed')
            if lifecycle.gpu_processes():raise RuntimeError('Analysis left a GPU process')
            save(root/'analysis-audit.json',dict(status='passed',cases=summary['cases'],capture_arms=2,
                completed_capture_cases=2*summary['cases'],capture_result_hashes_unchanged=True,
                normalized_native_attention=True,remaining_gpu_processes=[],
                source_proof=str(root/'analysis-source-proof.json'),at=time.time()))
            save(state_path,dict(status='completed',cases=summary['cases'],at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--timeout',type=float,default=7200);a=p.parse_args()
    try:finish(a.root.resolve(),a.timeout)
    except BaseException as e:
        save(a.root/'analysis-status.json',dict(status='failed',error=repr(e),at=time.time()));raise
