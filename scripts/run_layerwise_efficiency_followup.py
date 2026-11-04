"""Finite dependency queue after the running serial qualification.

Preserve every first attempt. Component, full-input and throughput stages each
keep their own frozen evidence. Any technical or equality failure stops here.
"""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import recall_comparison_protocol as p

REPO=Path(__file__).resolve().parents[1]


def choose_source(queue):
    state=p.read(queue/'status.json')
    if state['status'] not in ('completed','failed'):
        raise ValueError('Serial qualification has not finished')
    compiled=queue/'compiled'
    if p.read(compiled/'status.json')['status']!='completed':
        raise ValueError('Compilation stage had a technical failure; diagnose before continuing')
    report=p.read(compiled/'equivalence.json')
    native=p.read(compiled/'native-reuse-audit.json')
    if not native['passed']:
        raise ValueError('Compilation stage changed native score execution')
    if report['passed'] and report['cases']==8:
        return compiled,'Compiled execution passed all exact full-input checks.'
    # Numerical differences under compilation cannot be accepted as an
    # execution optimization. Continue only from an independently accepted
    # eager stage, and record this decision rather than hiding the rejection.
    eager=queue/'fast_metadata';accepted=p.read(eager/'equivalence.json')
    if (report['cases']!=8 or not accepted['passed'] or accepted['cases']!=8 or
            not p.read(eager/'native-reuse-audit.json')['passed']):
        raise ValueError('No fully accepted source for the next execution candidates')
    return eager,'Compiled execution changed exact results; retain its evidence and use the accepted eager source.'


def process_live(pid):
    stat=Path('/proc')/str(pid)/'stat'
    try:return stat.read_text().rsplit(')',1)[1].split()[0] not in ('Z','X')
    except FileNotFoundError:return False


def prepare(root, previous, components, qualification, performance):
    if root.exists() or qualification.exists() or performance.exists():
        raise ValueError('Use fresh output roots; preserve all existing attempts')
    from run_layerwise_query_efficiency import verify_components
    verify_components(components)
    root.mkdir(parents=True)
    shutil.copy2(__file__,root/'launch.py')
    # The preparers copy a few driver files from the workspace. Bind those
    # sources and their local imports before waiting, so later edits cannot
    # silently enter a supposedly frozen candidate.
    files={str(path):p.sha(path) for folder in (REPO/'scripts',REPO/'src/vllm_kvmem')
           for path in folder.glob('*.py')}
    p.save(root/'plan.json',dict(repo=str(REPO),previous=str(previous),components=str(components),
        qualification=str(qualification),performance=str(performance),
        source_queue_plan_sha256=p.sha(previous/'plan.json'),source_pid=p.read(previous/'process.json')['pid'],
        component_plan_sha256=p.sha(components/'plan.json'),workspace_files=files,
        launch_sha256=p.sha(root/'launch.py'),scheduled_monitor=False,no_automatic_retry=True,
        scope='GPU component equality, three exact full-input execution candidates, then matched eight-owner performance. '
              'Formal Memory remains gated by efficiency review.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))


def verify_inputs(root,plan):
    if p.sha(root/'launch.py')!=plan['launch_sha256']:
        raise ValueError('Frozen follow-up launcher changed')
    if p.sha(Path(plan['previous'])/'plan.json')!=plan['source_queue_plan_sha256']:
        raise ValueError('Running qualification controls changed')
    if p.sha(Path(plan['components'])/'plan.json')!=plan['component_plan_sha256']:
        raise ValueError('Frozen GPU component plan changed')
    for name,digest in plan['workspace_files'].items():
        if p.sha(Path(name))!=digest:
            raise ValueError('Prepared workflow source changed; create a fresh version: '+name)


def run(root):
    from run_recall_comparison import termination_guard,stop_owned
    plan=p.read(root/'plan.json');verify_inputs(root,plan)
    previous=Path(plan['previous']);components=Path(plan['components'])
    qualification=Path(plan['qualification']);performance=Path(plan['performance']);repo=Path(plan['repo'])
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'process.json').exists():
            raise ValueError('Never automatically repeat a follow-up attempt')
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        p.save(root/'status.json',dict(status='waiting_for_serial_qualification',source_pid=plan['source_pid'],at=time.time()))
        while True:
            if p.read(previous.parent/'value-recall-active.json').get('queue_root')!=str(previous):
                raise RuntimeError('Another experiment acquired GPU scheduling')
            state=p.read(previous/'status.json')['status'];live=process_live(plan['source_pid'])
            if state in ('completed','failed') and not live:break
            if not live:raise RuntimeError('Previous controller disappeared without a terminal result')
            command=(Path('/proc')/str(plan['source_pid'])/'cmdline').read_bytes().split(b'\0')
            if str(previous/'launch.py').encode() not in command:
                raise RuntimeError('Previous controller PID belongs to another process')
            time.sleep(15)
        verify_inputs(root,plan)
        source,reason=choose_source(previous)
        p.save(root/'source-selection.json',dict(source=str(source),reason=reason,at=time.time()))
        py=sys.executable
        query_script=repo/'scripts/run_layerwise_query_efficiency.py'
        matched_script=repo/'scripts/run_layerwise_matched_performance.py'
        steps=[('gpu_components',[py,'-B',str(query_script),'components','--root',str(components),'--source',str(previous)]),
            ('prepare_query_candidates',[py,'-B',str(query_script),'prepare','--root',str(qualification),
                '--source',str(source),'--components',str(components)]),
            ('query_candidates',[py,'-B',str(qualification/'launch.py'),'run','--root',str(qualification)]),
            ('prepare_matched_performance',[py,'-B',str(matched_script),'prepare','--root',str(performance),
                '--source',str(qualification/'prepared_index')]),
            ('matched_performance',[py,'-B',str(performance/'launch.py'),'run','--root',str(performance)])]
        environment=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src')+os.pathsep+str(repo/'scripts'))
        for name,command in steps:
            verify_inputs(root,plan)
            p.save(root/'status.json',dict(status='running',stage=name,at=time.time()))
            with (root/(name+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=repo,env=environment,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            p.save(root/'child.json',dict(pid=child.pid,stage=name,command=command,at=time.time()))
            try:code=child.wait()
            except BaseException:
                stop_owned(child);raise
            if code:
                raise RuntimeError('Follow-up stage failed; preserve evidence and diagnose: '+name)
            p.save(root/(name+'-completed.json'),dict(status='completed',at=time.time()))
        report=p.read(performance/'performance-comparison.json')
        p.save(root/'status.json',dict(status='completed',performance_target_met=report['goal_met'],
            next_action='Review actual throughput, phases and GPU attribution before formal Memory qualification.',
            performance=str(performance),at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    for name in ('previous','components','qualification','performance'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.previous.resolve(),args.components.resolve(),args.qualification.resolve(),args.performance.resolve())
        else:run(root)
    except BaseException as error:
        if args.action=='run' and root.exists():p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
