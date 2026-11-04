"""Finite, read-only paired Terminal Bench monitoring with quiet queued notices.

Only monitor state is written. Never stop/restart tasks, modify their limits,
repair experiments, or interrupt an active chat turn.
"""
import argparse
import asyncio
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time
import uuid

import psutil

try:
    from .monitor_deepswe_delivery import wake
    from .compare_terminalbench_policies import outcome_map, outcome_identity, exact_mcnemar
    from .compare_terminalbench_scored_pairs import SCHEMA, ANALYSIS_SCOPE, quality_analysis, score_eligibility
except ImportError:
    from monitor_deepswe_delivery import wake
    from compare_terminalbench_policies import outcome_map, outcome_identity, exact_mcnemar
    from compare_terminalbench_scored_pairs import SCHEMA, ANALYSIS_SCOPE, quality_analysis, score_eligibility


ARMS = ('thinking', 'lru')
PREFIX = '【Terminal Bench 双策略巡检】'
EVIDENCE_FILES = ('manifest.json', 'profile.json', 'job.json', 'agent-time-policy.json',
                  'results.json', 'execution-audit.json', 'formal-events/summary.json')
DEAD = {'missing_or_exited', 'exited', 'identity_changed'}


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def confined(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Evidence path outside experiment')
    return path


def owned_process(record):
    """PID alone is not ownership. Check creation time and optional argv/cwd."""
    if type(record.get('pid')) is not int or record['pid'] <= 0 or 'created' not in record:
        return dict(state='unverifiable_identity')
    try:
        proc = psutil.Process(record['pid'])
        if proc.create_time() != record['created']:
            return dict(state='identity_changed', pid=proc.pid)
        if proc.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            return dict(state='exited', pid=proc.pid)
        if record.get('argv') is not None and proc.cmdline() != record['argv']:
            return dict(state='identity_changed', pid=proc.pid)
        if record.get('cwd') is not None and Path(proc.cwd()).resolve() != Path(record['cwd']).resolve():
            return dict(state='identity_changed', pid=proc.pid)
        return dict(state='alive', pid=proc.pid)
    except psutil.NoSuchProcess:
        return dict(state='missing_or_exited', pid=record['pid'])
    except psutil.AccessDenied:
        return dict(state='visibility_denied', pid=record['pid'])


def resources(root):
    return dict(memory_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(root).free,
                system_disk_free_bytes=shutil.disk_usage('/').free)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def acknowledged_user_cancellation(directory, row):
    """Only a hash-bound, audited human stop is an expected unscored event."""
    if row.get('exception') != 'CancelledError' or row.get('rewards') is not None:
        return False
    marker=read(directory/'user-interruption.json',{})
    if marker.get('reason') != 'user_requested_switch_to_lru' or row.get('trial') not in marker.get('cancelled_trials',[]):
        return False
    try:
        path=confined(directory.parent,marker['audit'])
        require(sha(path)==marker['audit_sha256'],'Interruption audit changed')
        require(sha(path.parent/'authorization.json')==marker['authorization_sha256'],'Human authorization changed')
        audit=read(path,{})
        require(audit.get('passed') is True and audit.get('original_completed_evidence_preserved') is True,
                'Missing successful preservation audit')
        entries=[r for r in audit.get('user_cancelled',[]) if r.get('trial')==row['trial']]
        require(len(entries)==1,'Missing unique audited cancellation')
        raw_path=confined(directory,entries[0]['raw_path'])
        require(sha(raw_path)==entries[0]['raw_sha256'],'Original cancellation changed')
        raw=read(raw_path,{})
        require(raw.get('trial_name')==row['trial'] and raw.get('verifier_result') is None
                and (raw.get('exception_info') or {}).get('exception_type')=='CancelledError',
                'Original cancellation outcome differs')
        return True
    except (OSError,ValueError,KeyError,TypeError):
        return False


def completion_proof(root, pair):
    """Prove 89 first attempts per arm without inventing unavailable scores."""
    comparison_path = confined(root, pair.get('result', 'accuracy-comparison.json'))
    comparison = read(comparison_path, {})
    require(comparison.get('passed') is True and comparison.get('validation', {}).get('passed') is True,
            'Pair comparison/validation is missing or unsuccessful')
    wrapped = comparison.get('schema') == SCHEMA
    if wrapped:
        require(comparison.get('attempts_complete') is True and comparison.get('analysis_scope') == ANALYSIS_SCOPE,
                'Wrapper does not declare complete first attempts and jointly scored quality')
        implementation = comparison.get('analysis_implementation', {})
        for label in ('wrapper', 'comparator'):
            source = implementation.get(label, {})
            require(source.get('path') and source.get('sha256') == sha(confined(root, source['path'])),
                    'Analysis implementation hash changed: ' + label)
        comparator = confined(root, implementation['comparator']['path'])
        require(comparator == (root/'compare_terminalbench_policies.py').resolve(),
                'Wrapper did not use the original paired comparator')
        prelaunch = read(root/'prelaunch-audit.json', {})
        require(prelaunch.get('comparator_sha256') == sha(comparator), 'Frozen comparator proof differs')
    cohorts = {}; report_hashes = {}; cohort_outcomes = {}
    for arm, label in (('thinking', 'candidate'), ('lru', 'baseline')):
        directory = root / arm
        side = comparison.get(label, {})
        require(Path(side.get('root', '')).resolve() == directory.resolve(), 'Comparison arm identity differs')
        binding = comparison.get('evidence_sha256', {}).get(label, {})
        for name in EVIDENCE_FILES:
            require(binding.get(name) == sha(directory / name), 'Comparison evidence changed: ' + arm + '/' + name)
        state = read(directory / 'status.json', {})
        audit = read(directory / 'execution-audit.json', {})
        formal = read(directory / 'formal-events/summary.json', {})
        results = read(directory / 'results.json', {})
        manifest = read(directory / 'manifest.json', {})
        review = read(directory / 'final-review.json', {})
        cleanup = read(directory / 'cleanup.json', {})
        require(state.get('status') == review.get('status') == 'completed', arm + ': final review incomplete')
        require(audit.get('passed') is True and audit.get('tasks') == audit.get('attempts') == 89
                and audit.get('retries') == 0, arm + ': audit does not cover 89 first attempts')
        require(cleanup.get('remaining_processes') == cleanup.get('remaining_owned_gpu_pids') == []
                and audit.get('cleanup') == cleanup, arm + ': cleanup evidence incomplete')
        names = manifest.get('launched_task_names', [])
        require(manifest.get('tasks') == len(names) == len(set(names)) == 89,
                arm + ': full unique task set required')
        require(not manifest.get('preserved_scored') and not read(directory / 'preserved-outcomes.json', []),
                arm + ': inherited attempts are not a fresh paired run')
        job = read(directory / 'job.json', {})
        require(job.get('n_attempts') == 1 and job.get('retry', {}).get('max_retries') == 0,
                arm + ': job is not restricted to first attempts')
        outcomes = outcome_map(formal.get('outcomes', []), arm)
        require(set(outcomes) == set(names), arm + ': task identities differ')
        ordinary = all(score_eligibility(row)['eligible'] for row in outcomes.values())
        require(wrapped or ordinary, arm + ': unavailable/nonordinary scores require the scored-pairs wrapper')
        for result_set in ('outcomes', 'tasks'):
            actual = outcome_map(results.get(result_set, []), arm + '/' + result_set)
            require({k: outcome_identity(v) for k, v in actual.items()} ==
                    {k: outcome_identity(v) for k, v in outcomes.items()}, arm + ': results disagree')
        require(results.get('measurement') == audit, arm + ': results/audit disagree')
        passed = sum(row['passed'] for row in outcomes.values())
        scored = sum(row['status'] == 'scored' for row in outcomes.values())
        errors = sum(row['status'] == 'error' for row in outcomes.values())
        timeouts = sum(row['status'] == 'agent_timeout' for row in outcomes.values())
        require(scored + errors + timeouts == 89, arm + ': incomplete outcome classification')
        for value in (formal, results, side):
            require(value.get('scored') == scored and value.get('passed') == passed and value.get('errors') == errors
                    and value.get('task_timeouts', 0) == timeouts, arm + ': outcome totals disagree')
        require(formal.get('expected') == review.get('expected') == 89 and review.get('scored') == scored
                and review.get('task_timeouts', 0) == timeouts
                and review.get('infrastructure_or_unscored_errors', 0) == errors
                and formal.get('retries') == results.get('retries') == 0, arm + ': first-attempt count differs')
        if wrapped:
            require(side.get('attempts') == 89 and side.get('unscored') == 89-scored,
                    arm + ': wrapper attempts/unscored count differs')
            require(set(comparison.get('raw_outcome_evidence', {}).get(label, {})) == set(outcomes),
                    arm + ': wrapper raw evidence cohort differs')
        report = confined(directory, review.get('report', 'report.txt'))
        require(sha(report) == review.get('report_sha256'), arm + ': report changed')
        evidence_path = directory / 'evidence-hashes.json'
        require(sha(evidence_path) == review.get('evidence_sha256'), arm + ': evidence index changed')
        evidence = read(evidence_path, {})
        trial_paths = list((directory / 'jobs/formal').glob('*/result.json'))
        require({p.parent.name for p in trial_paths} == {row['trial'] for row in outcomes.values()},
                arm + ': extra or missing first-attempt result')
        configs = list((directory / 'jobs/formal').glob('*/config.json'))
        require({p.parent.name for p in configs} == {row['trial'] for row in outcomes.values()},
                arm + ': extra or missing attempted trial')
        for row in outcomes.values():
            relative = 'jobs/formal/' + row['trial'] + '/result.json'
            require(evidence.get(relative) == sha(confined(directory, relative)), arm + ': raw result changed')
            raw = read(directory / relative, {})
            exception = raw.get('exception_info')
            require(exception is None or (isinstance(exception, dict) and isinstance(exception.get('exception_type'), str)
                                          and exception['exception_type']), arm + ': malformed raw exception')
            error_type = exception.get('exception_type') if exception else None
            require(raw.get('trial_name') == row['trial'] and
                    raw.get('task_name', '').removeprefix('terminal-bench/') == row['task'].removeprefix('terminal-bench/') and
                    (raw.get('verifier_result') or {}).get('rewards') == row['rewards'] and
                    error_type == row.get('exception'), arm + ': raw outcome identity differs')
            require(row['status'] == 'scored' or exception is not None,
                    arm + ': unscored attempt has no original exception')
            rewards = (raw.get('verifier_result') or {}).get('rewards')
            expected_status = 'scored' if rewards is not None else 'agent_timeout' if error_type == 'AgentTimeoutError' else 'error'
            require(row['status'] == expected_status and row.get('passed') is bool(rewards and rewards.get('reward') == 1),
                    arm + ': raw reward/status/pass classification differs')
            if wrapped:
                name = row['task'].removeprefix('terminal-bench/')
                proof = comparison.get('raw_outcome_evidence', {}).get(label, {}).get(name, {})
                event_path = 'formal-events/' + row['trial'] + '.json'
                require(proof.get('trial') == row['trial'] and proof.get('path') == relative
                        and proof.get('sha256') == sha(directory/relative) and proof.get('exception_info') == exception
                        and proof.get('event_path') == event_path and proof.get('event_sha256') == sha(directory/event_path)
                        and read(directory/event_path) == raw, arm + ': wrapper raw exception/result binding differs')
        cohorts[arm] = dict(scored=scored, unscored=89-scored, passed=passed, attempts=89,
                            errors=errors, task_timeouts=timeouts)
        cohort_outcomes[label] = outcomes
        report_hashes[arm] = sha(report)
    require(set(read(root / 'thinking/manifest.json')['launched_task_names']) ==
            set(read(root / 'lru/manifest.json')['launched_task_names']), 'Paired task sets differ')
    quality, excluded = quality_analysis(cohort_outcomes['baseline'], cohort_outcomes['candidate'], exact_mcnemar)
    scoring_complete = quality['task_count'] == 89
    if wrapped:
        require(comparison.get('quality') == quality and comparison.get('excluded_tasks') == excluded
                and comparison.get('scoring_complete') is scoring_complete,
                'Jointly scored analysis/coverage/exclusions differ from original outcomes')
        for arm, label in (('thinking', 'candidate'), ('lru', 'baseline')):
            operational = comparison.get('operational', {}).get(label, {})
            counts = cohorts[arm]
            require(all(operational.get(key) == counts[key] for key in
                        ('attempts', 'scored', 'unscored', 'passed', 'errors', 'task_timeouts'))
                    and operational.get('success_fraction_all_attempts') == counts['passed']/89,
                    arm + ': operational all-attempt totals differ')
            unscored = sorted(name for name, row in cohort_outcomes[label].items() if row['status'] != 'scored')
            require(comparison[label].get('unscored_tasks') == unscored
                    and operational.get('unscored_tasks') == [dict(task=name,
                        outcome=outcome_identity(cohort_outcomes[label][name]),
                        exception_info=comparison['raw_outcome_evidence'][label][name]['exception_info']) for name in unscored],
                    arm + ': explicit unscored outcomes differ')
        require(comparison.get('gc_triggered_subsets', {}).get('status') == 'disabled',
                'Legacy mixed-score GC quality subsets must not be used')
    return dict(passed=True, comparison=str(comparison_path), comparison_sha256=sha(comparison_path),
                attempts_complete=True, scoring_complete=scoring_complete, jointly_scored_tasks=quality['task_count'],
                cohorts=cohorts, report_sha256=report_hashes)


def arm_completion_proof(root, arm):
    """Verify one finished arm without waiting for the paired comparison."""
    directory=root/arm
    state=read(directory/'status.json',{});review=read(directory/'final-review.json',{})
    audit=read(directory/'execution-audit.json',{});results=read(directory/'results.json',{})
    formal=read(directory/'formal-events/summary.json',{});manifest=read(directory/'manifest.json',{})
    cleanup=read(directory/'cleanup.json',{});job=read(directory/'job.json',{})
    require(state.get('status')==review.get('status')=='completed' and audit.get('passed') is True,
            arm+': final review incomplete')
    require(audit.get('tasks')==audit.get('attempts')==89 and audit.get('retries')==0
            and job.get('n_attempts')==1 and job.get('retry',{}).get('max_retries')==0,
            arm+': not 89 original first attempts')
    require(cleanup.get('remaining_processes')==cleanup.get('remaining_owned_gpu_pids')==[]
            and audit.get('cleanup')==cleanup and results.get('measurement')==audit,
            arm+': cleanup or execution audit differs')
    names=manifest.get('launched_task_names',[]);outcomes=outcome_map(formal.get('outcomes',[]),arm)
    require(len(names)==len(set(names))==manifest.get('tasks')==89 and set(outcomes)==set(names),
            arm+': original task cohort differs')
    require(not manifest.get('preserved_scored') and not read(directory/'preserved-outcomes.json',[]),
            arm+': inherited scores are not allowed')
    require({p.parent.name for p in (directory/'jobs/formal').glob('*/result.json')}==
            {p.parent.name for p in (directory/'jobs/formal').glob('*/config.json')}==
            {r['trial'] for r in outcomes.values()},arm+': extra or missing original attempt files')
    evidence_path=directory/'evidence-hashes.json';evidence=read(evidence_path,{})
    require(sha(evidence_path)==review.get('evidence_sha256')
            and sha(confined(directory,review.get('report','report.txt')))==review.get('report_sha256'),
            arm+': final report or evidence index changed')
    for name,row in outcomes.items():
        relative='jobs/formal/'+row['trial']+'/result.json';path=confined(directory,relative)
        require(evidence.get(relative)==sha(path),arm+': original result changed')
        raw=read(path,{});error=raw.get('exception_info');reward=(raw.get('verifier_result') or {}).get('rewards')
        require(raw.get('trial_name')==row['trial'] and raw.get('task_name','').removeprefix('terminal-bench/')==name
                and reward==row['rewards'] and (error or {}).get('exception_type')==row.get('exception'),
                arm+': original outcome differs')
        require(row['status']==('scored' if reward is not None else 'agent_timeout'
            if (error or {}).get('exception_type')=='AgentTimeoutError' else 'error')
            and row.get('passed') is bool(reward and reward.get('reward')==1)
            and (reward is not None or error is not None),arm+': missing reward was imputed or lacks an exception')
    totals=dict(scored=sum(r['status']=='scored' for r in outcomes.values()),
                passed=sum(r['passed'] for r in outcomes.values()),errors=sum(r['status']=='error' for r in outcomes.values()),
                task_timeouts=sum(r['status']=='agent_timeout' for r in outcomes.values()))
    require(all(formal.get(k,0)==results.get(k,0)==v for k,v in totals.items()),arm+': totals disagree')
    return dict(verified=True,arm=arm,attempts=89,unscored=89-totals['scored'],**totals,
                report=str(confined(directory,review.get('report','report.txt'))))


def inspect(root, *, now=None, probe=owned_process, resource_probe=resources, selected_arms=ARMS):
    root = Path(root).resolve(); now = time.time() if now is None else now
    require(selected_arms and all(arm in ARMS for arm in selected_arms),'Unknown monitored arm')
    pair = read(root / 'pair-status.json', {})
    status = pair.get('status', 'prepared'); issues = []; arms = {}; processes = {}
    def issue(code, message):
        issues.append(dict(code=code, message=message))
    def check_process(label, path, required=False):
        record = read(path, {})
        if not record:
            if required:
                issue('identity:' + label, label + ' lacks an owned process identity record')
            return
        value = processes[label] = probe(record)
        if value['state'] in DEAD:
            issue('process:' + label, label + ': owned process ' + value['state'])
        elif value['state'] != 'alive':
            issue('identity:' + label, label + ': ownership/visibility cannot be verified')
    if status == 'failed':
        issue('pair_failed', 'Paired controller failed: ' + str(pair.get('error', 'inspect controller logs')))
    active = pair.get('arm')
    if status == 'running':
        # Older pair-child.json contains only a PID. Never use it as ownership.
        check_process('pair', root / 'pair-process.json')
    for arm in selected_arms:
        directory = root / arm
        state = read(directory / 'status.json', {})
        phase = state.get('status', 'prepared')
        formal = read(directory / 'formal-events/status.json', {})
        rows = formal.get('outcomes', [])
        arms[arm] = dict(status=phase, completed=len(rows),
                         scored=sum(row.get('status') == 'scored' for row in rows),
                         passed=sum(row.get('passed') is True for row in rows),
                         active=formal.get('active', []))
        expected_stops={row.get('trial') for row in rows if acknowledged_user_cancellation(directory,row)}
        arms[arm]['user_cancelled']=len(expected_stops)
        if phase == 'failed':
            issue(arm + ':failed', arm + ' failed: ' + str(state.get('error', 'inspect failure.json')))
        for key in ('release_errors', 'network_errors', 'verifier_cache_errors'):
            if formal.get(key):
                issue(arm + ':' + key, arm + ': ' + key + ' ' + str(formal[key])[:1200])
        for row in rows:
            if row.get('status') in ('error', 'agent_timeout'):
                if row.get('trial') in expected_stops:
                    continue
                issue(arm + ':task:' + str(row.get('trial')), arm + ': unscored task ' +
                      str(row.get('task')) + ' / ' + str(row.get('exception')))
        if status != 'running' or active != arm or phase in ('completed', 'failed', 'prepared'):
            continue
        # Final audit follows service shutdown; dead services are expected then.
        check_process(arm + '/queue', directory / 'queue-process.json', required=True)
        all_classified = (len(rows) == 89 and len({row.get('trial') for row in rows}) == 89)
        if phase in ('smoke', 'formal') and not (phase == 'formal' and all_classified):
            for name, filename in (('model', 'model/server-process.json'), ('bridge', 'bridge-process.json'),
                                   ('gpu-monitor', 'gpu-monitor-process.json'), (phase, phase + '-process.json')):
                if (directory / filename).exists() or now - state.get('at', now) > 120:
                    check_process(arm + '/' + name, directory / filename, required=True)
    res = resource_probe(root)
    if status not in ('completed', 'failed'):
        for key, limit in (('memory_available_bytes', 8 * 2**30), ('disk_free_bytes', 30 * 2**30),
                           ('system_disk_free_bytes', 2 * 2**30)):
            if res.get(key) is not None and res[key] < limit:
                issue('resource:' + key, 'Resource pressure: ' + key + ' below ' + str(limit))
    proof = None
    if status == 'completed':
        try:
            proof = completion_proof(root, pair)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            issue('completion_invalid', 'Controller finished but paired first-attempt audit is incomplete: ' + str(error))
    selected_proof=None
    if len(selected_arms)==1 and arms[selected_arms[0]]['status']=='completed':
        try:selected_proof=arm_completion_proof(root,selected_arms[0])
        except (OSError,ValueError,KeyError,TypeError) as error:
            issue(selected_arms[0]+':completion_invalid','Selected arm final audit is incomplete: '+str(error))
    result=dict(at=now, root=str(root), pair_status=status, active_arm=active,
                arms=arms, processes=processes, resources=res, issues=issues,
                complete=proof is not None, completion=proof)
    if len(selected_arms)==1:
        result.update(scope=selected_arms[0],selected_arm_complete=selected_proof is not None,
                      selected_arm_completion=selected_proof)
    return result


def findings(snapshot, previous):
    if not previous:
        return []  # First inspection establishes a silent baseline.
    old = {item['code']: item['message'] for item in previous.get('issues', [])}
    events = [item for item in snapshot['issues'] if old.get(item['code']) != item['message']]
    if snapshot.get('selected_arm_complete') and not previous.get('selected_arm_complete') and not snapshot['complete']:
        proof=snapshot['selected_arm_completion']
        events.append(dict(code='selected_arm_complete',message=snapshot['scope']+
            ': 89 original attempts and final audit complete; passed '+str(proof['passed'])+
            ', scored '+str(proof['scored'])+', unscored '+str(proof['unscored'])+
            '. Paired comparison is pending. Report: '+proof['report']))
    if snapshot['complete'] and not previous.get('complete'):
        proof = snapshot.get('completion') or {}
        cohorts = proof.get('cohorts', {})
        unscored = {arm: value.get('unscored') for arm, value in cohorts.items()}
        events.append(dict(code='complete', message=('LRU test and paired comparison are complete; 89 original first attempts; '
            if snapshot.get('scope')=='lru' else 'Both policies completed 89 original first attempts; ')+
            'unscored counts: ' + json.dumps(unscored, sort_keys=True) +
            '; jointly scored quality pairs: ' + str(proof.get('jointly_scored_tasks')) +
            '; user-cancelled (unscored, retained): ' + json.dumps({arm:value.get('user_cancelled',0)
                for arm,value in snapshot.get('arms',{}).items()},sort_keys=True) + '. Comparison and audits are complete.'))
    return events


def notification_prefix(snapshot):
    return '【Terminal Bench LRU 巡检】' if snapshot.get('scope')=='lru' else PREFIX


def event_message(snapshot, events):
    return (notification_prefix(snapshot) + '\n只读巡检发现以下明确变化，请核对已有证据后处理。'
            '不要因耗时或无进展自动取消任务，不要恢复 Agent 总时限，不要重试或替换已有成绩。'
            '当前正在工作时，此通知只应排队。\n发现：\n' +
            json.dumps(events, ensure_ascii=False) + '\n快照：\n' + json.dumps(snapshot, ensure_ascii=False))


def step(args, state, *, now=None, inspector=inspect, notifier=wake):
    """Persist before delivery, so uncertain RPC responses retain the client ID."""
    now = time.time() if now is None else now
    path = args.state_dir / 'state.json'
    binding = dict(root=str(args.root.resolve()), thread_id=args.thread_id)
    if getattr(args,'arm',None):binding['arm']=args.arm
    if state:
        require(state.get('binding') == binding, 'Monitor state belongs to another root/thread')
    else:
        state = dict(binding=binding, started_at=now, expires_at=now + args.expires_after,
                     previous=None, outbox=[], delivered=[])
    if now >= state['expires_at']:
        state['status'] = 'expired'; save(path, state); return state
    try:
        snapshot = inspector(args.root, now=now)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        snapshot = dict(at=now, root=binding['root'], pair_status='inspection_error', arms={},
                        issues=[dict(code='inspection_error', message='Read-only inspection failed: ' + str(error))],
                        complete=False)
    events = findings(snapshot, state['previous'])
    if events:
        fingerprint = hashlib.sha256(json.dumps(events, sort_keys=True).encode()).hexdigest()
        if fingerprint not in state['delivered'] and not any(x['fingerprint'] == fingerprint for x in state['outbox']):
            key = str(uuid.uuid5(uuid.NAMESPACE_URL, binding['root'] + '\0' + args.thread_id + '\0' + fingerprint))
            state['outbox'].append(dict(key=key, fingerprint=fingerprint, events=events,
                                        message=event_message(snapshot, events)))
    state.update(previous=snapshot, checked_at=now, status='monitoring', delivery_error=None)
    save(path, state)
    if state['outbox']:
        pending = state['outbox'][0]
        try:
            receipt = asyncio.run(notifier(args.thread_id, args.socket, pending['message'], pending['key'],
                message_prefix=pending['message'].split('\n',1)[0]))
            state['last_delivery'] = dict(at=now, client_message_id=pending['key'], receipt=receipt)
            state['delivered'].append(pending['fingerprint']); state['outbox'].pop(0)
        except Exception as error:
            state['delivery_error'] = repr(error)
    if snapshot['complete'] and not state['outbox']:
        state['status'] = 'completed'
    save(path, state)
    return state


def run(args):
    args.state_dir.mkdir(parents=True, exist_ok=True)
    with (args.state_dir / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stopped = False
        def stop(*_):
            nonlocal stopped
            stopped = True
        signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
        state = read(args.state_dir / 'state.json', {})
        while not stopped and not (args.state_dir / 'STOP').exists():
            selected=(args.arm,) if args.arm else ARMS
            state = step(args, state,inspector=lambda root,**kwargs:inspect(root,selected_arms=selected,**kwargs))
            if args.once or state['status'] in ('completed', 'expired'):
                return
            deadline = min(time.time() + args.interval, state['expires_at'])
            while not stopped and not (args.state_dir / 'STOP').exists() and time.time() < deadline:
                time.sleep(max(0, min(5, deadline - time.time())))
        state.update(status='stopped', stopped_at=time.time()); save(args.state_dir / 'state.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'state-dir', 'socket'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--thread-id', required=True)
    parser.add_argument('--interval', type=int, default=1200)
    parser.add_argument('--expires-after', type=int, default=7 * 86400)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--inspect-only', action='store_true')
    parser.add_argument('--arm',choices=ARMS,help='Monitor one active arm and its completion; keep final paired-result validation')
    args = parser.parse_args()
    if args.interval < 60 or not 1 <= args.expires_after <= 31 * 86400:
        parser.error('Interval must be >=60 seconds and finite lifetime must be 1 second to 31 days')
    args.root = args.root.resolve(); args.state_dir = args.state_dir.resolve()
    if args.inspect_only:
        print(json.dumps(inspect(args.root,selected_arms=(args.arm,) if args.arm else ARMS), ensure_ascii=False, indent=2))
    else:
        run(args)
