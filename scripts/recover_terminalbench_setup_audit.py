"""Continue this pinned first-attempt cohort while preserving one setup error.

This is a separate successor to the network-only reconciler. It never changes
the deployed v1, frozen runtime, raw score, exception, or attempt count. The
missing dna-insert score stays missing; it is never assigned reward zero.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil

try:
    from . import recover_terminalbench_network_audit as base
except ImportError:
    import recover_terminalbench_network_audit as base

read, save, sha, require = base.read, base.save, base.sha, base.require
confined, events, unique = base.confined, base.events, base.unique
SETUP_TRIAL = 'dna-insert__GKooYyr'
SETUP_TASK = 'terminal-bench/dna-insert'
RAW_SHA = 'a125391c0af1b444feb4e8d938a78f7105832682d8d7b2df25ed151fe7eb6b25'
EVENT_RESULT_SHA = '512b38c788da8db232a9fa5d9020e42ad860e6b2e4a151ef73a62cb306ff58ae'
SETUP_NETWORK = '6e96885cf49a67e4ecf4ea2a663d9e2afcec68d77cd01bdf1ee7a82fe7fa192e'


def validate_setup_error(arm, lifecycle):
    raw_name = f'jobs/formal/{SETUP_TRIAL}/result.json'
    event_name = f'formal-events/{SETUP_TRIAL}.json'
    require(sha(arm/raw_name) == RAW_SHA and sha(arm/event_name) == EVENT_RESULT_SHA,
            'Pinned setup-error raw evidence changed')
    raw = read(arm/raw_name)
    require(raw == read(arm/event_name), 'Setup result copies disagree')
    error = raw.get('exception_info') or {}
    require(raw.get('trial_name') == SETUP_TRIAL and raw.get('task_name') == SETUP_TASK and
            raw.get('agent_execution') is None and raw.get('agent_result') is None and
            raw.get('verifier_result') is None and error.get('exception_type') == 'RuntimeError' and
            error.get('exception_message') == 'Command timed out after 120 seconds',
            'Setup incident is not the original unscored pre-Agent timeout')
    setup = raw.get('agent_setup') or {}
    require(setup.get('started_at') and setup.get('finished_at') and raw.get('finished_at'),
            'Missing completed setup-failure timing')
    trace = error.get('exception_traceback', '')
    require('_install_recording_tools' in trace and '_setup_agent' in trace,
            'Exception is not the pinned recording-tools setup failure')
    own = [row for row in lifecycle if row.get('trial') == SETUP_TRIAL]
    for kind in ('start', 'environment-start', 'end', 'kv_released', 'empty_networks_reclaimed'):
        require(sum(row.get('event') == kind for row in own) == 1, 'Setup lifecycle lacks unique '+kind)
    require(not any(row.get('event') in ('agent-start', 'agent-end', 'verification-start') for row in own),
            'Unscored setup attempt unexpectedly reached Agent or verifier execution')
    released = next(row for row in own if row['event'] == 'kv_released')
    require(released.get('released') is True and released.get('owners') == [],
            'Setup failure acquired or failed to release a model owner')
    bridge_path = arm/'bridge/requests.jsonl'
    bridge = [json.loads(line) for line in bridge_path.read_text().splitlines() if line]
    require(not any(row.get('session_id') == SETUP_TRIAL+'__agent' for row in bridge),
            'Setup failure unexpectedly has bridge/model requests')
    directory = arm/'formal-events/network-cleanup'/SETUP_TRIAL
    removed_path = directory/(SETUP_NETWORK+'.removed.json')
    checked_path = directory/(SETUP_NETWORK+'.json')
    removed, checked = read(removed_path), read(checked_path)
    require(removed.get('status') == 'removed' and checked.get('status') == 'checked' and
            removed.get('retained_filesystem_diffs_unchanged') is True and
            removed.get('project') == checked.get('project') == SETUP_TRIAL.lower()+'__env' and
            removed.get('network') == checked.get('network') and
            removed.get('network', {}).get('Id') == SETUP_NETWORK and
            removed.get('network', {}).get('Containers') == {} and
            removed.get('retained_containers') == checked.get('retained_containers') and
            removed.get('environment_dir') == checked.get('environment_dir') ==
            str(arm/'dataset/tasks/dna-insert/environment'), 'Setup task network cleanup evidence differs')
    ended = next(row['at'] for row in own if row['event'] == 'end')
    cleaned = next(row for row in own if row['event'] == 'empty_networks_reclaimed')
    require(cleaned.get('count') == 1 and ended <= checked['at'] <= removed['finished_at'] <= cleaned['at'],
            'Setup cleanup lifecycle order differs')
    names = (raw_name, event_name, str(removed_path.relative_to(arm)), str(checked_path.relative_to(arm)))
    return dict(passed=True, task=SETUP_TASK, trial=SETUP_TRIAL, classification='infrastructure_setup_error',
        reward=None, exception_type='RuntimeError', no_agent_execution=True, no_verifier_execution=True,
        no_model_requests=True, original_outcome_preserved=True,
        pinned_source_hashes={name: sha(arm/name) for name in names},
        bridge_journal_sha256=sha(bridge_path), scope='Attempt is complete but unscored; not a model solving failure.')


def classify(raw):
    if raw.get('trial_name') != SETUP_TRIAL:
        return base.classify(raw)
    return dict(task=SETUP_TASK, trial=SETUP_TRIAL, rewards=None, exception='RuntimeError', status='error', passed=False)


def validate_results(arm):
    manifest, status = read(arm/'manifest.json'), read(arm/'formal-events/status.json')
    names = manifest['launched_task_names']
    require(manifest.get('tasks') == len(names) == len(set(names)) == base.TASKS,
            'Need the original full 89-attempt cohort')
    require(not manifest.get('preserved_scored') and not (arm/'preserved-outcomes.json').exists(),
            'Inherited scores are not allowed')
    job = read(arm/'job.json')
    require(job.get('n_attempts') == 1 and job.get('retry', {}).get('max_retries') == 0, 'No task retries allowed')
    require(status.get('expected') == status.get('completed') == status.get('released') == base.TASKS and
            status.get('active') == [] and status.get('release_errors') == [] and
            status.get('verifier_cache_errors') == [] and status.get('network_errors') == [base.ALLOWED_ERROR],
            'Original lifecycle is incomplete or has another cleanup error')
    cohort = unique(status['outcomes'], 'trial')
    require(len(cohort) == base.TASKS and SETUP_TRIAL in cohort and
            {row['task'].removeprefix('terminal-bench/') for row in cohort.values()} == set(names),
            'Original attempt cohort differs')
    harbor = read(arm/'formal-events/harbor-result.json'); stats = harbor.get('stats', {})
    require(harbor.get('finished_at') and harbor.get('n_total_trials') == stats.get('n_completed_trials') == base.TASKS
            and stats.get('n_errored_trials') == 1 and all(stats.get(key) == 0 for key in
                ('n_running_trials', 'n_pending_trials', 'n_cancelled_trials', 'n_retries')),
            'Harbor must end with exactly one original setup error and no unfinished/retried attempts')
    upstream = unique(harbor['trial_results'], 'trial_name')
    require(set(upstream) == set(cohort), 'Harbor attempt cohort differs')
    require(read(arm/'jobs/formal/result.json') == {k: v for k, v in harbor.items() if k != 'trial_results'},
            'Harbor aggregate files disagree')
    for filename in ('config.json', 'result.json'):
        require({p.parent.name for p in (arm/'jobs/formal').glob('*/'+filename)} == set(cohort),
                'Extra or missing original attempt '+filename)
    lifecycle = events(arm)
    setup_proof = validate_setup_error(arm, lifecycle)
    scored_trials = set(cohort)-{SETUP_TRIAL}
    for kind in ('start', 'agent-start', 'agent-end', 'verification-start', 'end', 'kv_released'):
        selected = [row for row in lifecycle if row.get('event') == kind]
        expected = scored_trials if kind in ('agent-start', 'agent-end', 'verification-start') else set(cohort)
        require(Counter(row.get('trial') for row in selected) == Counter({trial: 1 for trial in expected}),
                'Missing, extra, or repeated '+kind+' event')
        if kind == 'kv_released':
            require(all(row.get('released') is True for row in selected), 'Missing native KV release acknowledgment')
    sources = ['manifest.json', 'job.json', 'formal-events/status.json', 'formal-events/events.jsonl',
               'formal-events/harbor-result.json', 'jobs/formal/result.json', 'jobs/formal/config.json', 'bridge/requests.jsonl']
    for trial, expected in cohort.items():
        raw_path, event_path = f'jobs/formal/{trial}/result.json', f'formal-events/{trial}.json'
        raw = read(confined(arm, raw_path))
        require(raw == read(confined(arm, event_path)) == upstream[trial], 'Three original results differ: '+trial)
        require(classify(raw) == expected, 'Original classification differs: '+trial)
        config_path = f'jobs/formal/{trial}/config.json'; config = read(confined(arm, config_path))
        require(config.get('trial_name') == trial and Path(config['task']['path']).name ==
                expected['task'].removeprefix('terminal-bench/') and base.saved_config_matches(config, raw.get('config')),
                'Original saved trial config differs')
        sources.extend((raw_path, event_path, config_path))
    passed = sum(row['passed'] for row in cohort.values())
    require(status.get('passed') == passed, 'Status pass count differs')
    receipt = base.validate_network_receipt(arm, base.ALLOWED_ERROR, lifecycle)
    coverage = dict(attempts=89, scored=88, unscored=1, fully_scored=False,
        unscored_tasks=[dict(task=SETUP_TASK, trial=SETUP_TRIAL, category='infrastructure_setup_error', reward=None)])
    summary = dict(expected=89, outcomes=list(cohort.values()), scored=88, passed=passed,
                   task_timeouts=0, errors=1, retries=0, coverage=coverage)
    bindings = {name: sha(arm/name) for name in sources}
    bindings.update(receipt['source_hashes']); bindings.update(setup_proof['pinned_source_hashes'])
    return dict(passed=True, attempt_cohort_complete=True, fully_scored=False, summary=summary,
                setup_error_preservation=setup_proof, network_reconciliation=receipt, source_hashes=bindings)


def prepare(pair, recovery, analysis_script):
    pair, recovery, analysis_script = pair.resolve(), recovery.resolve(), analysis_script.resolve()
    require(analysis_script.is_file() and analysis_script.is_relative_to(pair) and
            analysis_script != pair/'compare_terminalbench_policies.py',
            'Use a separately frozen coverage-aware analysis wrapper inside the campaign')
    setup = validate_setup_error(pair/'thinking', events(pair/'thinking'))
    plan = base.prepare(pair, recovery)
    plan.update(schema=2, known_unscored_attempt=setup, analysis_script=str(analysis_script),
                analysis_sha256=sha(analysis_script), analysis_output='accuracy-comparison-with-setup-error.json',
                expected_scored_thinking=88, expected_unscored_thinking=1,
                completion_semantics='89 original completed attempts; one pinned setup error remains unscored')
    plan['immutable_inputs'].update({'thinking/'+name: digest for name, digest in setup['pinned_source_hashes'].items()})
    plan['immutable_inputs'][str(analysis_script.relative_to(pair))] = sha(analysis_script)
    save(recovery/'plan.json', plan)
    return plan


def audit(pair, recovery, plan, live_probe=base.process_alive):
    require(plan.get('schema') == 2 and str(pair.resolve()) == plan.get('pair') and
            plan.get('allowed_error') == base.ALLOWED_ERROR and plan.get('expected_scored_thinking') == 88
            and plan.get('expected_unscored_thinking') == 1, 'Setup-aware plan identity differs')
    for name, expected in plan['immutable_inputs'].items():
        require(sha(confined(pair, name)) == expected, 'Pinned recovery input changed: '+name)
    require(read(pair/'pair-process.json') == plan['original_pair'] and
            read(pair/'thinking/queue-process.json') == plan['original_suite'], 'Original controller identity changed')
    for record in (plan['original_pair'], plan['original_suite']):
        require(not live_probe(record), 'Original controller/suite must finish naturally first')
    arm = pair/'thinking'
    for name in ('queue-process.json', 'formal-process.json', 'model/server-process.json',
                 'bridge-process.json', 'gpu-monitor-process.json'):
        require(not live_probe(read(arm/name)), 'An original owned process remains alive: '+name)
    cleanup = read(arm/'cleanup.json')
    require(cleanup.get('remaining_processes') == cleanup.get('remaining_owned_gpu_pids') == [] and
            not any(live_probe(record) for record in cleanup.get('owned', [])), 'Original cleanup is incomplete')
    pair_state, state = read(pair/'pair-status.json'), read(arm/'status.json')
    require(pair_state.get('status') == 'failed' and pair_state.get('error') ==
            "RuntimeError('thinking suite exited 1; preserve all attempts, no task retries')",
            'Original pair did not end at the expected network-cleanup guard')
    expected = "RuntimeError('formal exited 1; preserve logs, no retry')"
    require(state.get('status') == 'failed' and state.get('error') == expected and
            read(arm/'failure.json').get('error') == expected, 'Original suite failed for another reason')
    base.unattempted_lru(pair)
    require(not (arm/'formal-events/summary.json').exists(), 'Never replace an existing summary')
    result = validate_results(arm)
    result.update(pair=str(pair.resolve()), checked_at=time.time(), plan_sha256=sha(recovery/'plan.json'),
                  original_job_failure_retained=True, no_task_or_verifier_execution=True,
                  missing_score_is_not_model_failure=True)
    return result


def recover(pair, recovery, plan, finalizer=base.call_frozen_finalizer, live_probe=base.process_alive):
    result = audit(pair, recovery, plan, live_probe)
    original = recovery/'original'; original.mkdir(exist_ok=False)
    files = ['pair-status.json', 'pair-process.json', 'pair-child.json', 'thinking/status.json',
             'thinking/failure.json', 'thinking/cleanup.json', 'thinking/formal-events/status.json',
             'thinking/formal-events/events.jsonl', 'thinking/formal-events/harbor-result.json']
    for name in files:
        target = original/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(confined(pair, name), target)
    result['archived_originals'] = {name: sha(original/name) for name in files}
    save(recovery/'audit.json', result, exclusive=True)
    arm = pair/'thinking'; receipt = arm/'formal-events/setup-and-network-audit-reconciliation.json'
    save(receipt, dict(**result, recovery_directory=str(recovery.resolve()), audit_sha256=sha(recovery/'audit.json')), exclusive=True)
    summary = dict(result['summary'], derived=dict(kind='network-audit-reconciliation-with-original-setup-error',
        original_failure_preserved=True, original_network_errors_preserved=True, original_setup_error_preserved=True,
        no_reward_imputed=True, receipt=str(receipt.relative_to(arm)), receipt_sha256=sha(receipt)))
    save(arm/'formal-events/summary.json', summary, exclusive=True)
    finalizer(arm, recovery)
    for name, expected in result['source_hashes'].items():
        require(sha(arm/name) == expected, 'Finalizer changed original evidence: '+name)
    require(read(arm/'status.json').get('status') == 'completed', 'Frozen final review did not complete')
    require(read(arm/'results.json').get('scored') == 88 and read(arm/'results.json').get('errors') == 1,
            'Finalizer did not preserve missing-score coverage')
    save(recovery/'recovered.json', dict(at=time.time(), passed=True, fully_scored=False,
        audit_sha256=sha(recovery/'audit.json'), derived_summary_sha256=sha(arm/'formal-events/summary.json'),
        original_files_preserved=True, original_setup_error_preserved=True), exclusive=True)
    return result


def continue_pair(pair, recovery, plan):
    recover(pair, recovery, plan)
    started = read(recovery/'original/pair-status.json')['started_at']
    base.unattempted_lru(pair)
    owner = psutil.Process()
    metadata = dict(kind='audited-continuation-with-original-setup-error', directory=str(recovery.resolve()),
        original_failure=str(recovery/'original/pair-status.json'), audit_sha256=sha(recovery/'audit.json'),
        thinking_attempts=89, thinking_scored=88, thinking_unscored=1, no_task_retries=True)
    save(pair/'pair-process.json', dict(pid=owner.pid, created=owner.create_time(), argv=owner.cmdline(),
        cwd=owner.cwd(), at=time.time(), recovery=metadata))
    save(pair/'pair-status.json', dict(status='running', arm='lru', started_at=started, at=time.time(),
        recovery_started_at=time.time(), recovery=metadata))
    save(pair/'thinking-completed.json', dict(at=time.time(), status=read(pair/'thinking/status.json'), recovery=metadata), exclusive=True)
    lru = pair/'lru'; runtime = lru/'runtime'
    with (pair/'lru-driver.log').open('xb') as log:
        process = subprocess.Popen([base.PYTHON, '-B', str(runtime/'scripts/run_terminalbench_suite.py'), 'run', str(lru)],
            cwd=runtime, env=base.runtime_env(runtime), stdout=log, stderr=subprocess.STDOUT)
        child = dict(arm='lru', pid=process.pid, created=psutil.Process(process.pid).create_time(), at=time.time(), recovery=metadata)
        save(recovery/'lru-launched.json', child, exclusive=True); save(pair/'pair-child.json', child)
        code = process.wait()
    require(code == 0 and read(lru/'status.json').get('status') == 'completed',
            'Untouched LRU ended unsuccessfully; retain every attempt and do not retry')
    save(pair/'lru-completed.json', dict(at=time.time(), status=read(lru/'status.json'), recovery=metadata), exclusive=True)
    analyzer = Path(plan['analysis_script'])
    require(analyzer.resolve().is_relative_to(pair.resolve()) and sha(analyzer) == plan['analysis_sha256'],
            'Frozen coverage-aware analysis wrapper changed')
    output = confined(pair, plan['analysis_output'])
    require(not output.exists(), 'Never replace an existing analysis')
    with (pair/'coverage-comparison.log').open('xb') as log:
        subprocess.run([base.PYTHON, str(analyzer), '--baseline', str(lru), '--candidate', str(pair/'thinking'),
            '--output', str(output)], check=True, stdout=log, stderr=subprocess.STDOUT)
    save(pair/'pair-status.json', dict(status='completed', at=time.time(), started_at=started,
        seconds=time.time()-started, result=str(output), recovery=metadata,
        quality_coverage='partial_due_to_preserved_setup_error'))


def run(pair, recovery):
    plan = read(recovery/'plan.json')
    with (recovery/'.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not (recovery/'run-started.json').exists(), 'Continuation already attempted; manual audit required')
        require(time.time() < plan['expires_at'], 'Finite recovery plan already expired')
        save(recovery/'run-started.json', dict(at=time.time(), pid=os.getpid()), exclusive=True)
        try:
            while base.process_alive(plan['original_pair']) or base.process_alive(plan['original_suite']):
                require(time.time() < plan['expires_at'], 'Finite recovery wait expired; original work remains intact')
                save(recovery/'status.json', dict(status='waiting_for_original_completion', at=time.time(), no_task_cancellation=True,
                    expected_scored_thinking=88, preserved_setup_errors=1))
                time.sleep(30)
            require(time.time() < plan['expires_at'], 'Finite recovery wait expired; no successor started')
            continue_pair(pair, recovery, plan)
            save(recovery/'status.json', dict(status='completed', at=time.time(), fully_scored=False))
        except BaseException as error:
            save(recovery/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            record = read(pair/'pair-process.json')
            if record.get('pid') == os.getpid() and record.get('recovery'):
                previous = read(pair/'pair-status.json')
                save(pair/'pair-status.json', dict(status='failed', error=repr(error), at=time.time(),
                    started_at=previous.get('started_at'), recovery=record['recovery']))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'audit', 'run'))
    parser.add_argument('--pair-root', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--analysis-script', type=Path)
    args = parser.parse_args(); pair = args.pair_root.resolve(); recovery = args.recovery.resolve()
    if args.action == 'prepare':
        require(args.analysis_script is not None, 'prepare requires a separate coverage-aware analysis wrapper')
        plan = prepare(pair, recovery, args.analysis_script)
        print(json.dumps(dict(prepared=True, plan=str(recovery/'plan.json'), expires_at=plan['expires_at'])))
    elif args.action == 'audit':
        print(json.dumps(audit(pair, recovery, read(recovery/'plan.json')), indent=2))
    else:
        run(pair, recovery)
