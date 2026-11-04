"""Finite authorized efficiency/quality queue; never resample completed answers."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import recall_comparison_protocol as p


def run(root):
    plan=p.read(root/'queue-plan.json');digest=p.sha(root/'queue-plan.json')
    if p.sha(root/'runtime-proof.json')!=plan['runtime_proof_sha256']:
        raise ValueError('Frozen queue runtime proof changed')
    for name,expected in p.read(root/'runtime-proof.json').items():
        if p.sha(root/'runtime-v1'/name)!=expected:raise ValueError('Frozen queue source changed: '+name)
    if plan.get('grading_python'):
        from run_hybrid_suite_queue import grading_python
        interpreter=grading_python(plan)
        if any(s.get('grading') and s['grading']['command'][0]!=interpreter for s in plan['stages']):
            raise ValueError('A grading stage does not use the frozen evaluator environment')
    graders=[]
    with (root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for stage in plan['stages']:
            if p.sha(root/'queue-plan.json')!=digest:raise ValueError('Frozen suite queue changed')
            campaign=Path(stage['root']);state=p.read(campaign/'status.json')
            if p.sha(campaign/stage['proof_file'])!=stage['proof_sha256']:
                raise ValueError('Prepared campaign changed: '+stage['id'])
            p.save(root/'status.json',dict(status='running',stage=stage['id'],active_root=str(campaign),at=time.time()))
            pointer=dict(root=str(campaign),campaign=str(campaign),queue_root=str(root),
                status=str(campaign/'status.json'),process=str(campaign/'process.json'),
                experiment_type=stage['id'],at=time.time())
            for name in ('value-recall-active.json','layerwise-query-active.json'):
                p.save(root.parent/name,pointer)
            if stage.get('wait'):
                while state['status'] not in ('completed','failed'):
                    process=p.read(campaign/'controller-process.json')
                    try:proc=(Path('/proc')/str(process['pid'])/'stat').read_text().rsplit(')',1)[1].split()[0]
                    except FileNotFoundError:raise ValueError('Prerequisite controller disappeared')
                    if proc in ('Z','X'):raise ValueError('Prerequisite controller exited without completing')
                    time.sleep(5);state=p.read(campaign/'status.json')
            elif state['status']!='completed':
                proof_file=campaign/stage['proof_file']
                if p.sha(proof_file)!=stage['proof_sha256']:raise ValueError('Prepared campaign changed')
                pointer=dict(root=str(campaign),campaign=str(campaign),queue_root=str(root),
                    status=str(campaign/'status.json'),process=str(campaign/'process.json'),
                    experiment_type=stage['id'],at=time.time())
                for name in ('value-recall-active.json','layerwise-query-active.json'):
                    p.save(root.parent/name,pointer)
                command=stage['command']
                with (root/(stage['id']+'.log')).open('a') as log:
                    process=subprocess.Popen(command,cwd=campaign/'runtime-v1',
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                p.save(root/'child.json',dict(pid=process.pid,stage=stage['id'],root=str(campaign),command=command,at=time.time()))
                p.save(campaign/'controller-process.json',dict(pid=process.pid,command=command,at=time.time()))
                code=process.wait();state=p.read(campaign/'status.json')
                if code:raise RuntimeError(f"{stage['id']} exited {code}; preserve answers and diagnose")
            if state['status']!='completed':raise ValueError('Stage failed: '+stage['id'])
            if stage.get('equivalence_gate'):
                from analyze_layerwise_efficiency import analyze
                analyze(campaign,Path(stage['equivalence_gate']))
            p.save(root/(stage['id']+'-completed.json'),dict(status='completed',root=str(campaign),at=time.time()))
            if stage.get('grading'):
                if plan.get('grading_python'):
                    grading_python(plan)
                grade=stage['grading'];output=Path(grade['output'])
                if output.exists():raise ValueError('Preserve existing grading attempts before recovery')
                with (root/(stage['id']+'-grading.log')).open('x') as log:
                    grader=subprocess.Popen(grade['command'],cwd=campaign/'runtime-v1',
                        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                p.save(root/(stage['id']+'-grader.json'),dict(pid=grader.pid,command=grade['command'],output=str(output),at=time.time()))
                graders.append((stage['id'],grader,output))
        if graders:
            p.save(root/'status.json',dict(status='grading',stage='grading',at=time.time()))
        for name,grader,output in graders:
            if grader.wait() or p.read(output/'status.json')['status']!='completed':
                raise ValueError('Original-rubric grading failed: '+name)
        p.save(root/'status.json',dict(status='completed',stages=len(plan['stages']),at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.root.resolve())
    except BaseException as error:
        old=p.read(args.root/'status.json') if (args.root/'status.json').exists() else {}
        p.save(args.root/'status.json',dict(old,status='failed',error=repr(error),at=time.time()));raise
