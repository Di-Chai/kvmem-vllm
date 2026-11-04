"""Finish reporting after one audited cleanup failure; never rerun a trial.

All original rewards and exceptions remain benchmark outcomes. This checks
record integrity, not model quality, and does not label runtime errors benign.
"""
import argparse
from collections import Counter
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

import psutil

try:
    from . import recover_terminalbench_network_audit as base
except ImportError:
    import recover_terminalbench_network_audit as base

read,save,sha,require=base.read,base.save,base.sha,base.require


def original_outcome(raw):
    error=raw.get('exception_info');reward=(raw.get('verifier_result') or {}).get('rewards')
    require(error is None or (isinstance(error,dict) and isinstance(error.get('exception_type'),str)
            and error['exception_type']),'Malformed original exception')
    require(reward is not None or error is not None,'Unscored attempt lacks an original exception')
    if reward is not None:
        require(isinstance(reward,dict) and type(reward.get('reward')) in (int,float)
                and reward['reward'] in (0,1),'Nonbinary or malformed verifier reward')
    kind=error['exception_type'] if error else None
    require(kind not in ('CancelledError','AgentTimeoutError'),'Unapproved cancellation or restored Agent deadline')
    return dict(task=raw['task_name'],trial=raw['trial_name'],rewards=reward,exception=kind,
        status='scored' if reward is not None else 'error',passed=bool(reward and reward['reward']==1))


def network_proof(arm,receipt_path):
    require(receipt_path.resolve().is_relative_to(arm.resolve()),'Network receipt leaves arm directory')
    receipt=read(receipt_path)
    require(receipt.get('passed') is True and receipt.get('kind')=='independent-pre-removal-network-audit-reconciliation'
            and receipt.get('original_network_error_retained') is True
            and receipt.get('no_task_or_verifier_execution') is True,'Missing independent cleanup audit')
    trial,cid,nid=receipt['trial'],receipt['container_id'],receipt['network_id']
    require(receipt.get('source_sha256')==sha(arm/'runtime/scripts/terminalbench_network.py')==
            read(arm/'manifest.json')['frozen']['runtime']['scripts/terminalbench_network.py'],
            'Cleanup audit source differs from frozen runtime')
    raw_path=arm/'jobs/formal'/trial/'result.json';raw=read(raw_path)
    require(sha(raw_path)==receipt['original_result_sha256'] and original_outcome(raw)['rewards'] is not None
            and raw['verifier_result']['rewards']['reward']==receipt['reward_preserved'],
            'Original scored task changed during cleanup audit')
    directory=receipt_path.parent/'independent-cleanup'
    checked_path=directory/(nid+'.json');removed_path=directory/(nid+'.removed.json')
    checked,removed=read(checked_path),read(removed_path)
    require(checked.get('status')=='checked' and removed.get('status')=='removed'
            and removed.get('retained_filesystem_diffs_unchanged') is True
            and removed.get('network')==checked.get('network') and removed['network'].get('Id')==nid
            and removed['network'].get('Containers')=={}
            and removed.get('project')==checked.get('project')==trial.lower()+'__env'
            and removed.get('retained_containers')==checked.get('retained_containers')
            and set(removed['retained_containers'])=={cid}
            and removed['retained_containers'][cid].get('state')=='exited'
            and removed.get('environment_dir')==checked.get('environment_dir')==
                str(arm/'dataset/tasks'/raw['task_name'].removeprefix('terminal-bench/')/'environment')
            and receipt.get('records')==[removed], 'Independent cleanup evidence differs')
    error=dict(trial=trial,error=f"TimeoutExpired(['docker', 'diff', '{cid}'], 60)")
    paths=[receipt_path,checked_path,removed_path,raw_path,arm/'runtime/scripts/terminalbench_network.py']
    return dict(error=error,source_hashes={str(p.relative_to(arm)):sha(p) for p in paths})


def audit_results(arm,plan):
    status=read(arm/'formal-events/status.json');manifest=read(arm/'manifest.json');job=read(arm/'job.json')
    require(status.get('expected')==status.get('completed')==status.get('released')==89
            and status.get('active')==status.get('release_errors')==status.get('verifier_cache_errors')==[]
            and status.get('network_errors')==[plan['network']['error']], 'Incomplete lifecycle or additional cleanup error')
    require(job.get('n_attempts')==1 and job.get('retry',{}).get('max_retries')==0
            and not manifest.get('preserved_scored') and not (arm/'preserved-outcomes.json').exists(),
            'Rerun or inherited task outcomes are not allowed')
    names=manifest['launched_task_names'];cohort=base.unique(status['outcomes'],'trial')
    require(len(names)==len(set(names))==manifest.get('tasks')==len(cohort)==89
            and {r['task'].removeprefix('terminal-bench/') for r in cohort.values()}==set(names), 'Original cohort differs')
    harbor=read(arm/'formal-events/harbor-result.json');stats=harbor.get('stats',{})
    require(harbor.get('finished_at') and harbor.get('n_total_trials')==stats.get('n_completed_trials')==89
            and all(stats.get(k)==0 for k in ('n_running_trials','n_pending_trials','n_cancelled_trials','n_retries')),
            'Original Harbor job did not finish all first attempts naturally')
    upstream=base.unique(harbor['trial_results'],'trial_name')
    require(set(upstream)==set(cohort) and read(arm/'jobs/formal/result.json')==
            {k:v for k,v in harbor.items() if k!='trial_results'}, 'Original Harbor aggregates differ')
    for filename in ('config.json','result.json'):
        require({p.parent.name for p in (arm/'jobs/formal').glob('*/'+filename)}==set(cohort), 'Extra/missing attempt files')
    events=base.events(arm);sources=['manifest.json','job.json','formal-events/status.json','formal-events/events.jsonl',
        'formal-events/harbor-result.json','jobs/formal/result.json','bridge/requests.jsonl']
    agents=set();verifiers=set();unscored=[];exception_count=0
    for trial,outcome in cohort.items():
        raw_name=f'jobs/formal/{trial}/result.json';event_name=f'formal-events/{trial}.json';config_name=f'jobs/formal/{trial}/config.json'
        raw=read(arm/raw_name)
        require(raw==read(arm/event_name)==upstream[trial] and original_outcome(raw)==outcome,
                'Raw result, original exception, or classification differs: '+trial)
        require(base.saved_config_matches(read(arm/config_name),raw.get('config')), 'Saved trial config differs')
        require(raw.get('finished_at'),'Original attempt did not end')
        if raw.get('agent_execution') is not None:agents.add(trial)
        if raw.get('verifier') is not None:verifiers.add(trial)
        exception_count+=int(raw.get('exception_info') is not None)
        if outcome['rewards'] is None:
            unscored.append(dict(task=raw['task_name'],trial=trial,reward=None,exception_info=raw['exception_info'],
                causal_classification='unassigned; original runtime/verifier exception retained, not presumed policy-independent'))
        sources.extend((raw_name,event_name,config_name))
    require(stats.get('n_errored_trials')==exception_count,'Original Harbor error count differs')
    for kind,expected in [('start',set(cohort)),('end',set(cohort)),('kv_released',set(cohort)),
                          ('agent-start',agents),('agent-end',agents),('verification-start',verifiers)]:
        selected=[e for e in events if e.get('event')==kind]
        require(Counter(e.get('trial') for e in selected)==Counter({t:1 for t in expected}), 'Lifecycle differs: '+kind)
        if kind=='kv_released':require(all(e.get('released') is True for e in selected),'Missing KV release')
    failures=[e for e in events if e.get('event')=='network_reclamation_failed']
    require(len(failures)==1 and {k:failures[0].get(k) for k in ('trial','error')}==plan['network']['error'],
            'Original network error differs')
    summary=dict(expected=89,outcomes=list(cohort.values()),scored=89-len(unscored),passed=sum(r['passed'] for r in cohort.values()),
                 task_timeouts=0,errors=len(unscored),retries=0)
    require(summary['passed']==status['passed'],'Pass total differs')
    return dict(passed=True,checked_at=time.time(),scope='Integrity of all 89 first-attempt outcomes; no exception is waived or assigned a reward.',
        original_exceptions_preserved=unscored,summary=summary,source_hashes={name:sha(arm/name) for name in sources})


def prepare(pair,recovery,receipt):
    require(not recovery.exists() and read(pair/'pair-status.json').get('status')=='running'
            and read(pair/'pair-status.json').get('arm')=='lru','Prepare a fresh reconciliation while LRU is running')
    network=network_proof(pair/'lru',receipt)
    require(read(pair/'lru/formal-events/status.json').get('network_errors')==[network['error']], 'Only the independently reconciled cleanup error is covered')
    paths=['lru/manifest.json','lru/profile.json','lru/job.json','lru/agent-time-policy.json','compare_terminalbench_policies.py',
           'thinking/results.json','thinking/execution-audit.json','thinking/formal-events/summary.json','thinking/user-interruption.json',
           'setup-audit-continuation-code/compare_terminalbench_scored_pairs.py']
    paths+=['lru/'+name for name in network['source_hashes']]
    for row in read(pair/'lru/formal-events/status.json')['outcomes']:
        trial=row['trial'];paths.extend((f'lru/jobs/formal/{trial}/result.json',
            f'lru/jobs/formal/{trial}/config.json',f'lru/formal-events/{trial}.json'))
    expires=time.time()+7*86400
    monitor=read(pair/'monitor-current.json') if (pair/'monitor-current.json').exists() else {}
    if monitor.get('state'):
        expires=min(expires,read(Path(monitor['state']))['expires_at'])
    require(expires>time.time(),'Existing monitor/reconciliation lifetime expired')
    plan=dict(pair=str(pair),created_at=time.time(),expires_at=expires,
        original_controller=read(pair/'pair-process.json'),original_suite=read(pair/'lru/queue-process.json'),
        network=network,receipt=str(receipt),immutable={name:sha(pair/name) for name in paths},task_retries_allowed=False,
        interpretation='Unscored original task exceptions count as unsuccessful operational outcomes. Scored-only exclusions may depend on policy and cannot establish complete-task accuracy.')
    recovery.mkdir();save(recovery/'plan.json',plan,exclusive=True);return plan


def finish(pair,recovery,plan):
    for name,digest in plan['immutable'].items():require(sha(base.confined(pair,name))==digest,'Pinned evidence changed: '+name)
    require(read(pair/'pair-process.json')==plan['original_controller'] and
            read(pair/'lru/queue-process.json')==plan['original_suite'],'Original controller identities changed')
    for record in (plan['original_controller'],plan['original_suite']):require(not base.process_alive(record),'Original controller remains active')
    arm=pair/'lru';cleanup=read(arm/'cleanup.json')
    require(cleanup.get('remaining_processes')==cleanup.get('remaining_owned_gpu_pids')==[]
            and not any(base.process_alive(p) for p in cleanup.get('owned',[])), 'Original cleanup incomplete')
    require(read(arm/'status.json').get('error')=="RuntimeError('formal exited 1; preserve logs, no retry')"
            and read(pair/'pair-status.json').get('error')=="ValueError('LRU ended unsuccessfully; preserve all attempts, no retry')",
            'Original job did not stop at the audited cleanup guard')
    network_proof(arm,Path(plan['receipt']))
    require(not (arm/'formal-events/summary.json').exists(),'Never replace an existing summary')
    result=audit_results(arm,plan);original=recovery/'original';original.mkdir()
    for name in ['pair-status.json','pair-process.json','lru/status.json','lru/failure.json','lru/cleanup.json',
                 'lru/formal-events/status.json','lru/formal-events/harbor-result.json']:
        dest=original/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pair/name,dest)
    save(recovery/'audit.json',result,exclusive=True)
    owner=psutil.Process();metadata=dict(kind='audited_lru_cleanup_reconciliation',directory=str(recovery),no_task_retries=True)
    save(pair/'pair-process.json',dict(pid=owner.pid,created=owner.create_time(),argv=owner.cmdline(),cwd=owner.cwd(),at=time.time(),recovery=metadata))
    started=read(original/'pair-status.json')['started_at']
    save(pair/'pair-status.json',dict(status='running',arm='comparison',at=time.time(),started_at=started,recovery=metadata))
    save(arm/'formal-events/summary.json',dict(result['summary'],derived=dict(kind='independent_network_cleanup_reconciliation',
        original_failure_preserved=True,original_task_exceptions_preserved=True,no_reward_imputed=True,
        audit=str(recovery/'audit.json'),audit_sha256=sha(recovery/'audit.json'))),exclusive=True)
    save(arm/'status.json',dict(status='needs_final_review',at=time.time(),recovery=metadata))
    base.call_frozen_finalizer(arm,recovery)
    for name,digest in result['source_hashes'].items():require(sha(arm/name)==digest,'Finalizer changed original evidence: '+name)
    save(pair/'lru-completed.json',dict(at=time.time(),status=read(arm/'status.json'),recovery=metadata),exclusive=True)
    analyzer=pair/'setup-audit-continuation-code/compare_terminalbench_scored_pairs.py'
    spec=importlib.util.spec_from_file_location('_final_scored_pairs',analyzer);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    comparison=module.compare(arm,pair/'thinking');comparison['user_interruption']=read(pair/'thinking/user-interruption.json')
    comparison['network_reconciliation']=dict(receipt=plan['receipt'],receipt_sha256=sha(Path(plan['receipt'])))
    comparison['caveats'].append('Three candidate tasks were ended by explicit user request; the baseline was allowed to finish. All original runtime/verifier errors remain unsuccessful full-cohort outcomes. Scored-only selection and asymmetric stopping prevent attributing total time or operational differences solely to GC policy.')
    output=pair/'accuracy-comparison-with-user-stop.json';save(output,comparison,exclusive=True)
    save(pair/'pair-status.json',dict(status='completed',started_at=started,at=time.time(),seconds=time.time()-started,
        result=str(output),recovery=metadata,quality_coverage='partial_with_original_errors_and_user_stopping_asymmetry'))


def run(pair,recovery):
    plan=read(recovery/'plan.json')
    with (recovery/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (recovery/'run-started.json').exists(),'Reconciliation was already attempted; manual audit required')
        save(recovery/'run-started.json',dict(at=time.time(),pid=os.getpid()),exclusive=True)
        try:
            while base.process_alive(plan['original_controller']) or base.process_alive(plan['original_suite']):
                require(time.time()<plan['expires_at'],'Finite wait expired; no task is cancelled')
                save(recovery/'status.json',dict(status='waiting_for_original_completion',at=time.time(),no_task_cancellation=True))
                time.sleep(30)
            require(time.time()<plan['expires_at'],'Reconciliation expired')
            finish(pair,recovery,plan);save(recovery/'status.json',dict(status='completed',at=time.time()))
        except BaseException as error:
            save(recovery/'status.json',dict(status='failed',at=time.time(),error=repr(error)))
            if read(pair/'pair-process.json').get('pid')==os.getpid():
                previous=read(pair/'pair-status.json');save(pair/'pair-status.json',dict(previous,status='failed',at=time.time(),error=repr(error)))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'))
    p.add_argument('--pair-root',type=Path,required=True);p.add_argument('--recovery',type=Path,required=True)
    p.add_argument('--network-receipt',type=Path);a=p.parse_args();pair=a.pair_root.resolve();recovery=a.recovery.resolve()
    if a.action=='prepare':
        require(a.network_receipt is not None,'A successful independent cleanup audit is required')
        plan=prepare(pair,recovery,a.network_receipt.resolve());print(json.dumps(dict(prepared=True,expires_at=plan['expires_at'])))
    else:run(pair,recovery)
