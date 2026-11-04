"""Audit an explicitly authorized stop of three tails, then run untouched LRU."""
import argparse
from collections import Counter
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time

import psutil

try:
    from . import recover_terminalbench_query_audit as known
except ImportError:
    import recover_terminalbench_query_audit as known

base = known.base
read, save, sha, require = base.read, base.save, base.sha, base.require
AUTHORIZED = frozenset(('feal-linear-cryptanalysis__VVfDvgH',
                       'make-mips-interpreter__8K2wqST', 'path-tracing__2hJgihD'))
AUTHORIZATION = '结束剩余 3 题，保留已有成绩并切换 LRU'


def audit(pair, transition, live_probe=base.process_alive):
    authorization = read(transition/'authorization.json')
    require(authorization.get('human_authorization') == AUTHORIZATION
            and set(authorization.get('active_at_stop', [])) == set(authorization.get('authorized_trials', [])) == AUTHORIZED
            and authorization.get('task_retries_allowed') is False
            and authorization.get('completed_at_stop') == 86 and authorization.get('scored_at_stop') == 82
            and authorization.get('passed_at_stop') == 53, 'Explicit user stop authorization differs')
    signal = read(transition/'stop-signal.json')
    require(signal.get('signal') == 'SIGINT' and signal.get('identity') == authorization['formal_identity']
            and set(signal.get('only_remaining_trials', [])) == AUTHORIZED
            and signal['at'] >= authorization['at'], 'Stop signal lacks the authorized process identity')
    for name, digest in authorization['completed_evidence_sha256'].items():
        require(sha(base.confined(pair, name)) == digest, 'Pre-stop completed evidence changed: '+name)
    for field, name in (('original_pair','pair-process.json'), ('original_suite','thinking/queue-process.json')):
        require(read(pair/name) == authorization[field] and not live_probe(authorization[field]),
                'Original controller/suite must exit before switching')
    arm = pair/'thinking'
    for name in ('formal-process.json','model/server-process.json','bridge-process.json','gpu-monitor-process.json'):
        require(not live_probe(read(arm/name)), 'Original process remains alive: '+name)
    cleanup = read(arm/'cleanup.json')
    require(cleanup.get('remaining_processes') == cleanup.get('remaining_owned_gpu_pids') == [], 'Original cleanup incomplete')
    require(read(arm/'status.json').get('error') == "RuntimeError('formal exited -2; preserve logs, no retry')"
            and read(pair/'pair-status.json').get('error') ==
            "RuntimeError('thinking suite exited 1; preserve all attempts, no task retries')",
            'Original exit was not the authorized interrupt')
    base.unattempted_lru(pair)
    require(not (arm/'formal-events/summary.json').exists() and not (arm/'formal-events/harbor-result.json').exists(),
            'Never replace an existing summary or claim Harbor finished its interrupted aggregate')
    status = read(arm/'formal-events/status.json'); manifest = read(arm/'manifest.json'); job = read(arm/'job.json')
    require(status.get('expected') == status.get('completed') == status.get('released') == 89
            and status.get('active') == status.get('release_errors') == status.get('verifier_cache_errors') == []
            and status.get('network_errors') == [base.ALLOWED_ERROR], 'Post-stop lifecycle incomplete or has unknown errors')
    require(manifest.get('tasks') == len(set(manifest['launched_task_names'])) == 89
            and job.get('n_attempts') == 1 and job.get('retry',{}).get('max_retries') == 0,
            'Original full one-attempt cohort differs')
    cohort = base.unique(status['outcomes'], 'trial')
    require(len(cohort) == 89 and AUTHORIZED <= set(cohort)
            and {r['task'].removeprefix('terminal-bench/') for r in cohort.values()} == set(manifest['launched_task_names']),
            'Original task/attempt cohort differs')
    lifecycle = base.events(arm)
    known_proofs = known.validate_setup_errors(arm, lifecycle)
    base.validate_network_receipt(arm, base.ALLOWED_ERROR, lifecycle)
    sources = ['manifest.json','job.json','formal-events/status.json','formal-events/events.jsonl',
               'jobs/formal/result.json','jobs/formal/config.json','bridge/requests.jsonl']
    cancelled = []
    for filename in ('result.json','config.json'):
        require({p.parent.name for p in (arm/'jobs/formal').glob('*/'+filename)} == set(cohort), 'Attempt files differ')
    for trial, outcome in cohort.items():
        raw_name = f'jobs/formal/{trial}/result.json'; event_name = f'formal-events/{trial}.json'
        raw = read(arm/raw_name)
        require(raw == read(arm/event_name), 'Raw and event result disagree: '+trial)
        config_name = f'jobs/formal/{trial}/config.json'
        require(base.saved_config_matches(read(arm/config_name), raw.get('config')), 'Saved trial config differs')
        if trial in AUTHORIZED:
            error = raw.get('exception_info') or {}
            require(error.get('exception_type') == 'CancelledError' and raw.get('verifier_result') is None
                    and raw.get('verifier') is None and raw.get('finished_at')
                    and datetime.fromisoformat(raw['finished_at'].replace('Z','+00:00')).timestamp() >= authorization['at'],
                    'Authorized tail was not retained as an original unscored cancellation')
            expected = dict(task=raw['task_name'],trial=trial,rewards=None,exception='CancelledError',status='error',passed=False)
            own = [r for r in lifecycle if r.get('trial') == trial]
            require(sum(r.get('event') == 'cancel' for r in own) == 1
                    and not any(r.get('event') == 'verification-start' for r in own), 'Cancellation lifecycle differs')
            cleaned = [r for r in own if r.get('event') == 'empty_networks_reclaimed']
            proofs = list((arm/'formal-events/network-cleanup'/trial).glob('*.removed.json'))
            require(len(cleaned) == len(proofs) == 1 and cleaned[0].get('count') == 1,
                    'Cancelled container network cleanup is incomplete')
            proof = read(proofs[0])
            require(proof.get('status') == 'removed' and proof.get('retained_filesystem_diffs_unchanged') is True,
                    'Cancelled container filesystem retention audit failed')
            sources.append(str(proofs[0].relative_to(arm)))
            cancelled.append(dict(task=raw['task_name'],trial=trial,reason='user_requested_switch_to_lru',reward=None,
                                  exception_info=error,raw_path=raw_name,raw_sha256=sha(arm/raw_name)))
        else:
            expected = known.classify(raw)
        require(expected == outcome, 'Original outcome differs: '+trial)
        sources.extend((raw_name,event_name,config_name))
    require(len(cancelled) == 3 and sum(r['status'] == 'scored' for r in cohort.values()) == 82
            and sum(r['passed'] for r in cohort.values()) == status['passed'] == 53, 'Pre-stop scores or cancellation coverage changed')
    unscored = known.UNSCORED_TRIALS | AUTHORIZED
    for kind in ('start','agent-start','agent-end','verification-start','end','kv_released'):
        selected = [r for r in lifecycle if r.get('event') == kind]
        expected = set(cohort)-unscored if kind == 'verification-start' else (
            set(cohort)-known.SETUP_TRIALS if kind in ('agent-start','agent-end') else set(cohort))
        require(Counter(r.get('trial') for r in selected) == Counter({t:1 for t in expected}), 'Lifecycle differs: '+kind)
        if kind == 'kv_released': require(all(r.get('released') is True for r in selected), 'Missing KV release')
    aggregate = read(arm/'jobs/formal/result.json'); stats = aggregate.get('stats',{})
    require(aggregate.get('finished_at') is None and aggregate.get('n_total_trials') == stats.get('n_completed_trials') == 89
            and stats.get('n_errored_trials') == 7 and stats.get('n_cancelled_trials') == 3
            and all(stats.get(k) == 0 for k in ('n_running_trials','n_pending_trials','n_retries')),
            'Interrupted Harbor aggregate no longer agrees with the raw results')
    hashes = {name:sha(arm/name) for name in sources}
    for proof in known_proofs: hashes.update(proof['pinned_source_hashes'])
    return dict(passed=True,checked_at=time.time(),scope='Evidence integrity of 89 terminal first attempts; three user cancellations are not scored completions.',
        original_completed_evidence_preserved=True,missing_rewards_imputed=False,task_retries=0,
        user_cancelled=cancelled,known_unscored_errors=known_proofs,source_hashes=hashes,
        summary=dict(expected=89,outcomes=list(cohort.values()),scored=82,passed=53,task_timeouts=0,errors=7,retries=0,
                     cancelled_by_user=3,coverage=dict(attempts=89,scored=82,unscored=7,user_cancelled=3,fully_scored=False)))


def finalize_candidate(pair, transition):
    result = audit(pair,transition)
    post = transition/'after-stop'; post.mkdir(exist_ok=False)
    for name in ('pair-status.json','pair-process.json','pair-child.json','thinking/status.json','thinking/failure.json','thinking/cleanup.json'):
        target=post/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(pair/name,target)
    save(transition/'audit.json',result,exclusive=True)
    arm=pair/'thinking'
    receipt=dict(at=time.time(),reason='user_requested_switch_to_lru',cancelled_trials=sorted(AUTHORIZED),
                 cancelled_by_user=3,audit=str(transition/'audit.json'),audit_sha256=sha(transition/'audit.json'),
                 authorization_sha256=sha(transition/'authorization.json'),original_scores_preserved=True)
    save(arm/'user-interruption.json',receipt,exclusive=True)
    save(arm/'formal-events/summary.json',dict(result['summary'],derived=dict(kind='audited_user_stop',
        audit=str(transition/'audit.json'),audit_sha256=sha(transition/'audit.json'),original_harbor_job_interrupted=True,
        original_job_failure_preserved=True,no_reward_imputed=True)),exclusive=True)
    base.call_frozen_finalizer(arm,transition)
    for name,digest in result['source_hashes'].items():require(sha(arm/name)==digest,'Finalizer changed raw evidence: '+name)
    require(read(arm/'results.json')['scored']==82 and read(arm/'results.json')['errors']==7,'Finalizer changed coverage')
    state=read(arm/'status.json');save(arm/'status.json',dict(state,user_cancelled=3,completion_reason='user_requested_switch_to_lru'))
    return receipt


def run(pair,transition):
    require(not (transition/'controller-started.json').exists(),'Switch controller already attempted')
    save(transition/'controller-started.json',dict(at=time.time(),pid=psutil.Process().pid),exclusive=True)
    receipt=finalize_candidate(pair,transition)
    started=read(transition/'before/pair-status.json')['started_at'];owner=psutil.Process()
    metadata=dict(kind='user_authorized_switch',directory=str(transition),thinking_scored=82,thinking_user_cancelled=3,
                  original_scores_preserved=True,no_task_retries=True)
    save(pair/'pair-process.json',dict(pid=owner.pid,created=owner.create_time(),argv=owner.cmdline(),cwd=owner.cwd(),at=time.time(),recovery=metadata))
    save(pair/'pair-status.json',dict(status='running',arm='lru',started_at=started,at=time.time(),recovery=metadata))
    save(pair/'thinking-completed.json',dict(at=time.time(),status=read(pair/'thinking/status.json'),recovery=metadata),exclusive=True)
    base.unattempted_lru(pair)
    lru=pair/'lru';runtime=lru/'runtime'
    with (pair/'lru-driver.log').open('xb') as log:
        child=subprocess.Popen([base.PYTHON,'-B',str(runtime/'scripts/run_terminalbench_suite.py'),'run',str(lru)],
            cwd=runtime,env=base.runtime_env(runtime),stdout=log,stderr=subprocess.STDOUT)
        save(pair/'pair-child.json',dict(arm='lru',pid=child.pid,created=psutil.Process(child.pid).create_time(),at=time.time(),recovery=metadata))
        save(transition/'lru-launched.json',read(pair/'pair-child.json'),exclusive=True)
        code=child.wait()
    require(code==0 and read(lru/'status.json').get('status')=='completed','LRU ended unsuccessfully; preserve every attempt, no retry')
    save(pair/'lru-completed.json',dict(at=time.time(),status=read(lru/'status.json'),recovery=metadata),exclusive=True)
    analyzer=pair/'setup-audit-continuation-code/compare_terminalbench_scored_pairs.py'
    require(sha(analyzer)==known.ANALYSIS_SHA,'Qualified comparison wrapper changed')
    spec=importlib.util.spec_from_file_location('_switch_scored_comparator',analyzer)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    comparison=module.compare(lru,pair/'thinking')
    comparison['user_interruption']=receipt
    comparison['caveats'].append('The human explicitly ended three candidate tasks before verification. They remain unscored and unsuccessful in the 89-task operational outcome; scored-only quality excludes them. This is an asymmetric stopping rule, so timing and full-cohort success differences cannot be attributed solely to the GC policy.')
    output=pair/'accuracy-comparison-with-user-stop.json';save(output,comparison,exclusive=True)
    save(pair/'pair-status.json',dict(status='completed',at=time.time(),started_at=started,seconds=time.time()-started,
        result=str(output),recovery=metadata,quality_coverage='partial_with_three_user_cancelled_candidate_trials'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('audit','run'))
    p.add_argument('--pair-root',type=Path,required=True);p.add_argument('--transition',type=Path,required=True)
    a=p.parse_args();pair=a.pair_root.resolve();transition=a.transition.resolve()
    if a.action=='audit':
        result=audit(pair,transition);print(json.dumps({k:v for k,v in result.items() if k not in ('source_hashes','summary','known_unscored_errors','user_cancelled')}))
    else:
        try:run(pair,transition)
        except BaseException as error:
            save(transition/'controller-error.json',dict(at=time.time(),error=repr(error)))
            record=read(pair/'pair-process.json')
            if record.get('pid')==psutil.Process().pid:
                previous=read(pair/'pair-status.json');save(pair/'pair-status.json',dict(status='failed',at=time.time(),
                    started_at=previous.get('started_at'),error=repr(error),recovery=record.get('recovery')))
            raise
