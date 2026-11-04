"""Finite Memory continuation after accepted, matched layerwise efficiency.

No scheduled monitor or automatic technical retry. Every inference and grading
stage has a fresh directory; failed stages retain all first answers. Final
cross-method interpretation remains a separate completion audit.
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
from run_layerwise_matched_followup import identity, source_ready

REPO=Path(__file__).resolve().parents[1]
SUITES=(('longmemeval_s',2,102),('memoryagentbench',3,1116))


def prepare(root,performance,user_request=None,continue_from=None):
    if root.exists():raise ValueError('Use a fresh Memory continuation root')
    owner=identity(p.read(performance/'process.json')['pid'])
    if user_request:
        if owner is not None or p.read(performance/'status.json')['status']!='completed':
            raise ValueError('User-selected performance measurement must be completed and released')
    elif owner is None:raise ValueError('Expected the identified performance run to be live')
    source=Path(p.read(performance/'plan.json')['source'])
    from run_layerwise_query_efficiency import source_audit
    source_audit(source)
    old=REPO/'artifacts/layerwise-suite-20261102-r7'
    grading=p.read(old/'queue-plan.json')
    from run_hybrid_suite_queue import grading_python
    grading_python(grading)
    policy=p.read(old/'grading-policy-audit.json')
    root.mkdir(parents=True);shutil.copy2(__file__,root/'launch.py')
    authorization=None
    handoff=performance
    if user_request:
        handoff=Path(p.read(performance.parent/'value-recall-active.json')['queue_root'])
        authorization=root/'user-authorization.json'
        p.save(authorization,dict(policy='user_selected_qualified_runtime',user_request=user_request,
            pressure_runtime=str(source),performance=str(performance),
            performance_sha256=p.sha(performance/'performance-comparison.json'),
            preserve_original_memory_controls=True,at=time.time()))
        from layerwise_efficiency_gate import create
        create(source,performance,authorization)
    scripts={str(path):p.sha(path) for folder in (REPO/'scripts',REPO/'src/vllm_kvmem')
             for path in folder.glob('*.py')}
    # Reuse the actual original command, changing only campaign/output paths.
    grades={stage['id']:stage['grading'] for stage in grading['stages'] if stage.get('grading')}
    stages=[dict(suite=suite,qualification=q,cases=small if q else count,
                 root=str(root/(suite+('-qualification' if q else ''))))
            for q in (True,False) for suite,small,count in SUITES]
    continuation_files={}
    if continue_from is not None:
        if not user_request or continue_from!=handoff or p.read(continue_from/'status.json')['status']!='failed':
            raise ValueError('Explicit continuation requires the failed current queue')
        from run_layerwise_memory import completed_memory_qualification,supplemental_qualification,REFERENCE
        lme=continue_from/'longmemeval_s-qualification'
        _,_,qualified=completed_memory_qualification(lme,'longmemeval_s')
        pending,supplement=supplemental_qualification(continue_from/'memoryagentbench-qualification',
            p.read(REFERENCE/'memoryagentbench-fixture.json')['rows'])
        if any((continue_from/suite).exists() for suite,_,_ in SUITES):
            raise ValueError('Formal answers require separate explicit recovery; never regenerate them')
        continuation_files={**qualified['files'],**supplement['files']}
        continuation_files[str(continue_from/'status.json')]=p.sha(continue_from/'status.json')
        # A failure of MAB-only diagnostics must not hold up qualified LME.
        stages=[dict(suite='longmemeval_s',qualification=False,cases=102,
                     root=str(root/'longmemeval_s'),qualification_source=str(lme)),
                dict(suite='memoryagentbench',qualification=True,cases=len(pending),
                     root=str(root/'memoryagentbench-qualification'),
                     qualification_after=str(continue_from/'memoryagentbench-qualification')),
                dict(suite='memoryagentbench',qualification=False,cases=1116,
                     root=str(root/'memoryagentbench'))]
    p.save(root/'plan.json',dict(repo=str(REPO),performance=str(performance),source=str(source),
        source_owner=owner,performance_plan_sha256=p.sha(performance/'plan.json'),
        handoff_root=str(handoff),authorization_file=str(authorization) if authorization else None,
        authorization_sha256=p.sha(authorization) if authorization else None,
        workspace_files=scripts,launch_sha256=p.sha(root/'launch.py'),
        grading_python=grading['grading_python'],grading_templates=grades,
        grading_policy=str(old/'grading-policy-audit.json'),
        grading_policy_sha256=p.sha(old/'grading-policy-audit.json'),
        grading_policy_files=policy['frozen_evaluation_files'],
        continuation_files=continuation_files,
        scheduled_monitor=False,no_automatic_retry=True,
        stages=stages,
        scope='After the same-runtime efficiency gate, original LME2/MAB3 model qualification, '
              'then original LME102/MAB1116 and original grading. Never resume stopped experiments.'))
    p.save(root/'status.json',dict(status='prepared',at=time.time()))
    verify(root,p.read(root/'plan.json'))


def verify(root,plan):
    if (p.sha(root/'launch.py')!=plan['launch_sha256'] or
            p.sha(Path(plan['performance'])/'plan.json')!=plan['performance_plan_sha256'] or
            p.sha(Path(plan['grading_policy']))!=plan['grading_policy_sha256']):
        raise ValueError('Frozen Memory continuation dependency changed')
    if plan.get('authorization_file'):
        authorization=Path(plan['authorization_file'])
        if p.sha(authorization)!=plan['authorization_sha256']:
            raise ValueError('Frozen user selection changed')
        from layerwise_efficiency_gate import user_selection
        user_selection(authorization,plan['source'],plan['performance'])
    for name,digest in {**plan['workspace_files'],**plan['grading_policy_files'],
                        **plan.get('continuation_files',{})}.items():
        if p.sha(Path(name))!=digest:raise ValueError('Frozen Memory source changed: '+name)
    from run_hybrid_suite_queue import grading_python
    grading_python(plan)


def grading_command(template,campaign,output):
    command=list(template['command'])
    for option,value in (('--campaign',campaign),('--output',output)):
        if command.count(option)!=1:raise ValueError('Unknown original grading command')
        command[command.index(option)+1]=str(value)
    old=Path(template['command'][2]);new=campaign/'runtime-v1/scripts'/old.name
    if p.sha(old)!=p.sha(new):raise ValueError('Original grading implementation changed')
    command[2]=str(new)
    return command


def launch(root,name,command,cwd,environment):
    from run_recall_comparison import stop_owned
    # Exclusive creation happens before spawning; no overwritten attempts.
    with (root/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=cwd,env=environment,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    p.save(root/'child.json',dict(pid=child.pid,stage=name,command=command,at=time.time()))
    try:code=child.wait()
    except BaseException:
        stop_owned(child);raise
    if code:raise RuntimeError('Preserve failed first attempt; explicit recovery required: '+name)


def check_stage(campaign,stage):
    from run_layerwise_memory import verify as memory_verify,completed_memory_qualification
    fixture,_=memory_verify(campaign);status=p.read(campaign/'status.json')
    audit=p.read(campaign/'execution-audit.json');cleanup=p.read(campaign/'run/cleanup.json')
    if (status['status']!='completed' or fixture['suite']!=stage['suite'] or
            fixture['qualification']!=stage['qualification'] or fixture['cases']!=stage['cases'] or
            audit['status']!='passed' or audit['cases']!=stage['cases'] or
            audit['regenerated_completed_answers'] or cleanup['gpu_processes'] or cleanup['remaining_owned_groups']):
        raise ValueError('Memory stage did not complete its original cohort and execution audit')
    if stage['qualification']:completed_memory_qualification(campaign,stage['suite'])


def run(root):
    from run_recall_comparison import termination_guard,gpu_lock,stop_owned
    import benchmark_block_graph as lifecycle
    from layerwise_efficiency_gate import create as efficiency_acceptance
    plan=p.read(root/'plan.json');verify(root,plan)
    previous=Path(plan['performance']);source=Path(plan['source']);repo=Path(plan['repo'])
    pointer=previous.parent/'value-recall-active.json';graders=[]
    with termination_guard(),(root/'queue.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'process.json').exists():raise ValueError('Preserve the first continuation attempt')
        p.save(root/'process.json',dict(pid=os.getpid(),at=time.time()))
        authorization=plan.get('authorization_file')
        if authorization:
            from run_layerwise_query_efficiency import released
            handoff=Path(plan['handoff_root'])
            if p.read(pointer)['queue_root']!=str(handoff):raise RuntimeError('Another experiment acquired GPU scheduling')
            released(handoff);released(previous)
        else:
            p.save(root/'status.json',dict(status='waiting_for_efficiency',source_owner=plan['source_owner'],at=time.time()))
            deadline=time.monotonic()+10800
            while True:
                if p.read(pointer)['queue_root']!=str(previous):raise RuntimeError('Another experiment acquired GPU scheduling')
                if source_ready(p.read(previous/'status.json')['status'],identity(plan['source_owner']['pid']),plan['source_owner']):break
                if time.monotonic()>=deadline:raise TimeoutError('Performance still active; leave it untouched')
                time.sleep(15)
        verify(root,plan)
        report=p.read(previous/'performance-comparison.json')
        if not report['goal_met'] and not authorization:
            p.save(root/'status.json',dict(status='not_run',memory_started=False,
                reason='Matched efficiency target not met; continue optimization before Memory.',
                performance_sha256=p.sha(previous/'performance-comparison.json'),at=time.time()))
            return
        gate=(efficiency_acceptance(source,previous,authorization) if authorization else
              efficiency_acceptance(source,previous));p.save(root/'efficiency-gate.json',gate)
        with gpu_lock(Path('/tmp/kvmem-coordination-qualification.lock')):
            if lifecycle.gpu_processes():raise RuntimeError('GPU has another owner')
            if identity(p.read(previous/'child.json')['pid']) is not None:
                raise RuntimeError('Previous inference driver has not exited')
            p.save(pointer,dict(queue_root=str(root),root=str(root),status=str(root/'status.json'),
                process=str(root/'process.json'),experiment_type='qualified-layerwise-memory-suite',at=time.time()))
        environment={k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_','VLLM_','DEEPSEEK_'))}
        environment.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(repo/'src')+os.pathsep+str(repo/'scripts'))
        environment['PATH']=str(Path(sys.executable).parent)+os.pathsep+environment.get('PATH','')
        try:
            for stage in plan['stages']:
                verify(root,plan)
                if p.read(pointer)['queue_root']!=str(root):raise RuntimeError('Another experiment acquired GPU scheduling')
                campaign=Path(stage['root']);name=campaign.name
                if campaign.exists():raise ValueError('Preserve existing Memory stage: '+name)
                p.save(root/'status.json',dict(status='running',stage='prepare_'+name,active_root=str(campaign),at=time.time()))
                command=[sys.executable,'-B',str(repo/'scripts/run_layerwise_memory.py'),'prepare',
                    '--root',str(campaign),'--suite',stage['suite'],'--pressure-runtime',str(source),
                    '--efficiency-source',str(previous)]
                if authorization:command+=['--efficiency-authorization',authorization]
                command+=['--qualification'] if stage['qualification'] else [
                    '--qualification-source',stage.get('qualification_source',str(root/(stage['suite']+'-qualification')))]
                if stage.get('qualification_after'):
                    command+=['--qualification-after',stage['qualification_after']]
                launch(root,'prepare_'+name,command,repo,environment)
                p.save(root/'status.json',dict(status='running',stage=name,active_root=str(campaign),at=time.time()))
                p.save(pointer,dict(queue_root=str(root),root=str(campaign),status=str(campaign/'status.json'),
                    process=str(campaign/'process.json'),experiment_type=name,at=time.time()))
                runtime=campaign/'runtime-v1'
                runtime_environment=dict(environment,PYTHONPATH=str(runtime/'src')+os.pathsep+str(runtime/'scripts'))
                launch(root,name,[sys.executable,'-B',str(runtime/'scripts/run_layerwise_memory.py'),
                    'run','--root',str(campaign)],runtime,runtime_environment)
                check_stage(campaign,stage)
                p.save(root/(name+'-completed.json'),dict(status='completed',cases=stage['cases'],at=time.time()))
                if not stage['qualification']:
                    output=root/('grading-'+stage['suite'])
                    if output.exists():raise ValueError('Preserve original grading attempt')
                    command=grading_command(plan['grading_templates'][stage['suite']],campaign,output)
                    with (root/(name+'-grading.log')).open('x') as log:
                        grader=subprocess.Popen(command,cwd=runtime,env=runtime_environment,stdout=log,
                            stderr=subprocess.STDOUT,start_new_session=True)
                    graders.append((stage['suite'],grader,output))
                    p.save(root/(name+'-grader.json'),dict(pid=grader.pid,command=command,output=str(output),at=time.time()))
            p.save(root/'status.json',dict(status='grading',at=time.time()))
            for name,grader,output in graders:
                if grader.wait() or p.read(output/'status.json')['status']!='completed':
                    raise RuntimeError('Preserve original grading evidence: '+name)
            p.save(root/'status.json',dict(status='completed',formal_cases=1218,
                needs_final_comparison_audit=True,at=time.time()))
        except BaseException:
            for _,grader,_ in graders:
                if grader.poll() is None:stop_owned(grader)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--performance',type=Path)
    parser.add_argument('--user-request',help='Explicit user instruction to run the selected qualified version now')
    parser.add_argument('--continue-from',type=Path,help='Preserve a failed MAB qualification and reuse completed LME qualification')
    args=parser.parse_args();root=args.root.resolve()
    try:
        if args.action=='prepare':prepare(root,args.performance.resolve(),args.user_request,
                                         args.continue_from.resolve() if args.continue_from else None)
        else:run(root)
    except BaseException as error:
        if args.action=='run' and root.exists():p.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time()))
        raise
