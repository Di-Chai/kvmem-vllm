"""One finite performance follow-up after an exact full-input qualification.

Wait for the identified owner to finish, reject any failed qualification, then
reuse the matched eight-owner driver. No retries, scheduled monitors or Memory
inference are started here.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import recall_comparison_protocol as p

REPO=Path(__file__).resolve().parents[1]


def identity(pid):
    directory=Path('/proc')/str(pid)
    try:
        fields=(directory/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0] in ('Z','X'):return None
        return dict(pid=pid,start_ticks=fields[19],
            command_sha256=hashlib.sha256((directory/'cmdline').read_bytes()).hexdigest())
    except FileNotFoundError:return None


def source_ready(state,live,expected):
    if live is not None:
        if live!=expected:raise RuntimeError('Qualification PID was reused by another process')
        return False
    if state!='completed':
        raise RuntimeError('Qualification exited without a successful terminal result')
    return True


def qualified(previous):
    acceptance=p.read(previous/'acceptance.json')
    if not acceptance['passed'] or acceptance['cases']!=8:
        raise ValueError('All eight exact full-input checks must pass')
    plan=p.read(previous/'plan.json')
    if plan.get('kind')=='cpu_private_pack':
        from run_layerwise_private_pack import qualified as private_qualified
        return private_qualified(previous)
    if plan.get('kind')=='cpu_direct_seal':
        from run_layerwise_direct_seal import qualified as direct_qualified
        return direct_qualified(previous)
    if plan.get('kind')=='cpu_region_gather':
        from run_layerwise_region_gather import qualified as region_qualified
        return region_qualified(previous)
    if plan.get('kind')=='cpu_archive_execution':
        from run_layerwise_cpu_execution import qualified as cpu_qualified
        return cpu_qualified(previous)
    if 'components' in plan:
        qualified_buffered(previous,plan)
        return previous/'candidate'
    for kind,count in (('cpu',83),('gpu',6)):
        audit=p.read(previous/(kind+'-audit.json'))
        if (not audit['passed'] or audit['returncode']!=0 or
                audit['counts']!=dict(tests=count,failures=0,errors=0,skipped=0) or
                audit['test_sha256']!=p.sha(previous/(kind+'-tests.xml'))):
            raise ValueError('Paired read component qualification is incomplete')
    return previous/'candidate'


def qualified_buffered(previous,plan):
    """Bind native state tests and actual execution, before timing this version."""
    from run_layerwise_matched_performance import buffered_backend_audit
    from run_layerwise_query_efficiency import source_audit
    components=Path(plan['components']);source=previous/'candidate'
    audit=p.read(components/'component-audit.json')
    if (p.sha(components/'component-audit.json')!=plan['component_audit_sha256'] or
            audit['status']!='passed' or
            p.sha(components/'source-proof.json')!=audit['source_proof_sha256']):
        raise ValueError('Buffered component qualification changed')
    for name,digest in p.read(components/'source-proof.json').items():
        if p.sha(components/name)!=digest:
            raise ValueError('Buffered component source changed: '+name)
    for name,count,key in [('tests.xml',28,'tests_sha256'),
                           ('integration-tests.xml',99,'integration_tests_sha256')]:
        tests=list(ET.parse(components/name).getroot().iter('testcase'))
        if (p.sha(components/name)!=audit[key] or len(tests)!=count or
                any(t.find(k) is not None for t in tests for k in ('failure','error','skipped'))):
            raise ValueError('Buffered component tests are incomplete')
    fixture=p.read(source/'fixture.json')
    if not fixture.get('layerwise_buffered_history'):
        raise ValueError('Buffered qualification did not enable native history')
    # Re-read the saved evidence; an acceptance flag alone is insufficient.
    source_audit(source)
    for row in fixture['rows']:
        buffered_backend_audit(fixture,p.read(source/'formal/sequential/cases'/row['id']/'evidence.json'))


def prepare(root,previous,performance):
    if root.exists() or performance.exists():
        raise ValueError('Use new roots; preserve previous attempts')
    pid=p.read(previous/'process.json')['pid'];owner=identity(pid)
    if owner is None:raise ValueError('Expected a live full-input qualification to follow')
    root.mkdir(parents=True);shutil.copy2(__file__,root/'launch.py')
    files={str(path):p.sha(path) for folder in (REPO/'scripts',REPO/'src/vllm_kvmem')
           for path in folder.glob('*.py')}
    p.save(root/'plan.json',dict(repo=str(REPO),previous=str(previous),performance=str(performance),
        source_owner=owner,source_plan_sha256=p.sha(previous/'plan.json'),
        workspace_files=files,launch_sha256=p.sha(root/'launch.py'),
        scheduled_monitor=False,no_automatic_retry=True,
        scope='Only matched eight-owner throughput after exact full-input qualification; no Memory inference.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))


def verify(root,plan):
    if (p.sha(root/'launch.py')!=plan['launch_sha256'] or
            p.sha(Path(plan['previous'])/'plan.json')!=plan['source_plan_sha256']):
        raise ValueError('Frozen follow-up or source plan changed')
    for name,digest in plan['workspace_files'].items():
        if p.sha(Path(name))!=digest:raise ValueError('Workflow source changed: '+name)


def run(root):
    from run_recall_comparison import termination_guard,stop_owned
    plan=p.read(root/'plan.json');verify(root,plan)
    previous=Path(plan['previous']);performance=Path(plan['performance']);repo=Path(plan['repo'])
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'process.json').exists():raise ValueError('No automatic repeat of a follow-up attempt')
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        p.save(root/'status.json',dict(status='waiting_for_qualification',source_owner=plan['source_owner'],at=time.time()))
        deadline=time.monotonic()+7200
        while True:
            if p.read(previous.parent/'value-recall-active.json').get('queue_root')!=str(previous):
                raise RuntimeError('Another experiment acquired GPU scheduling')
            state=p.read(previous/'status.json')['status']
            if source_ready(state,identity(plan['source_owner']['pid']),plan['source_owner']):break
            if time.monotonic()>=deadline:raise TimeoutError('Owner still active; leave it untouched')
            time.sleep(15)
        verify(root,plan);source=qualified(previous)
        commands=[('prepare',[sys.executable,'-B',str(repo/'scripts/run_layerwise_matched_performance.py'),
            'prepare','--root',str(performance),'--source',str(source)]),
            ('performance',[sys.executable,'-B',str(performance/'launch.py'),'run','--root',str(performance)])]
        environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src')+os.pathsep+str(repo/'scripts'))
        for stage,command in commands:
            verify(root,plan)
            p.save(root/'status.json',dict(status='running',stage=stage,at=time.time()))
            with (root/(stage+'.log')).open('x') as log:
                child=subprocess.Popen(command,cwd=repo,env=environment,stdout=log,
                    stderr=subprocess.STDOUT,start_new_session=True)
            p.save(root/'child.json',dict(pid=child.pid,stage=stage,command=command,at=time.time()))
            try:code=child.wait()
            except BaseException:
                stop_owned(child);raise
            if code:raise RuntimeError('Follow-up failed; preserve every answer and measurement: '+stage)
        report=p.read(performance/'performance-comparison.json')
        p.save(root/'status.json',dict(status='completed',performance=str(performance),
            performance_target_met=report['goal_met'],memory_started=False,at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    for name in ('previous','performance'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.previous.resolve(),args.performance.resolve())
        else:run(root)
    except BaseException as error:
        if args.action=='run' and root.exists():p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
