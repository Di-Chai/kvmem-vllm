"""Recovery must preserve first-attempt scores and fail closed on other faults."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import recover_terminalbench_network_audit as recovery


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    pair = tmp_path / 'pair'
    arm = pair / 'thinking'
    lru = pair / 'lru'
    names = [f'task-{index:03}' for index in range(89)]
    names[0] = 'count-dataset-tokens'
    trials = [name + '__first' for name in names]
    trials[0] = recovery.TRIAL
    results = []
    outcomes = []
    events = []
    for index, (name, trial) in enumerate(zip(names, trials)):
        config = dict(task=dict(path=str(arm / 'dataset/tasks' / name)), trial_name=trial,
                      trials_dir=str(arm / 'jobs/formal'), source_trial=None)
        result = dict(task_name='terminal-bench/' + name, trial_name=trial,
                      config=config, verifier_result=dict(rewards=dict(reward=index % 2)),
                      exception_info=None, started_at='2026-10-29T10:00:00Z',
                      finished_at='2026-10-29T10:00:02Z',
                      agent_execution=dict(started_at='2026-10-29T10:00:00Z',
                                           finished_at='2026-10-29T10:00:01Z'))
        results.append(result)
        outcomes.append(dict(task=result['task_name'], trial=trial, rewards=dict(reward=index % 2),
                             exception=None, status='scored', passed=bool(index % 2)))
        write(arm / 'jobs/formal' / trial / 'result.json', result)
        write(arm / 'jobs/formal' / trial / 'config.json', config)
        write(arm / 'formal-events' / (trial + '.json'), result)
        for kind, at in [('start', 10), ('agent-start', 11), ('agent-end', 12),
                         ('kv_released', 13), ('verification-start', 14), ('end', 15)]:
            events.append(dict(event=kind, at=at, trial=trial, task=result['task_name'],
                               **(dict(released=True) if kind == 'kv_released' else {})))
    cid, nid = recovery.CID, recovery.NETWORK
    error = dict(trial=trials[0], error=f"TimeoutExpired(['docker', 'diff', '{cid}'], 60)")
    events.append(dict(event='network_reclamation_failed', at=20, **error))
    (arm / 'formal-events/events.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in events))
    stats = dict(n_completed_trials=89, n_errored_trials=0, n_running_trials=0,
                 n_pending_trials=0, n_cancelled_trials=0, n_retries=0)
    harbor = dict(id='job', started_at='2026-10-29T10:00:00Z',
                  finished_at='2026-10-29T11:00:00Z', n_total_trials=89,
                  stats=stats, trial_results=results)
    write(arm / 'formal-events/harbor-result.json', harbor)
    write(arm / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    write(arm / 'formal-events/status.json', dict(at=21, expected=89, completed=89,
          passed=44, active=[], released=89, release_errors=[], network_errors=[error],
          verifier_cache_errors=[], outcomes=outcomes))
    checked_rel = f'formal-events/network-cleanup/{trials[0]}/{nid}.json'
    diff = 'A /answer.txt\n'
    diff_sha = hashlib.sha256(diff.rstrip('\n').encode()).hexdigest()
    project = trials[0].lower() + '__env'
    checked = dict(at=16, network=dict(Id=nid, Driver='bridge', Containers={},
          Labels={'com.docker.compose.project': project, 'com.docker.compose.network': 'default'}),
          retained_containers={cid: dict(image='sha256:image', state='exited', diff_sha256=diff_sha)},
          project=project, environment_dir=str(arm / 'dataset/tasks' / names[0] / 'environment'),
          status='checked')
    write(arm / checked_rel, checked)
    monkeypatch.setattr(recovery, 'CHECKED_SHA', sha(arm / checked_rel))
    receipt_dir = arm / 'network-audit-recovery' / trials[0]
    receipt_dir.mkdir(parents=True)
    (receipt_dir / (cid + '.diff.txt')).write_text(diff)
    receipt = dict(started_at=30, finished_at=31, kind='post-removal-filesystem-audit-only',
          original_checked_file=checked_rel, original_checked_sha256=sha(arm / checked_rel),
          original_error_retained=True, no_task_execution=True, no_score_changes=True,
          passed=True, network_absent=True, containers={cid: dict(image='sha256:image',
          state=dict(Status='exited', Running=False, Paused=False, Restarting=False,
                     OOMKilled=False, ExitCode=137), diff_sha256=diff_sha,
          diff_seconds=1, diff_lines=1, matches_original=True)})
    write(receipt_dir / 'result.json', receipt)
    for folder in (arm, lru):
        manifest = dict(tasks=89, attempts=1, launched_task_names=names, frozen={}, files={})
        write(folder / 'manifest.json', manifest)
        write(folder / 'job.json', dict(n_attempts=1, n_concurrent_trials=12,
                                       retry=dict(max_retries=0), job_name='formal'))
        write(folder / 'agent-time-policy.json', dict(agent_time_limit='none'))
        write(folder / 'profile.json', dict(max_num_seqs=8, kvmem=dict(session_capacity=12)))
        (folder / 'runtime/scripts').mkdir(parents=True)
        (folder / 'runtime/scripts/finalize_terminalbench.py').write_text('# frozen finalizer fixture\n')
        (folder / 'runtime/scripts/run_terminalbench_suite.py').write_text('# frozen suite fixture\n')
        manifest['frozen']['runtime'] = {str(p.relative_to(folder / 'runtime')): sha(p)
                for p in (folder / 'runtime').rglob('*') if p.is_file()}
        write(folder / 'manifest.json', manifest)
    (arm / 'started').write_text('1')
    (pair / 'pair-started').write_text('1')
    write(arm / 'status.json', dict(status='failed', at=22,
          error="RuntimeError('formal exited 1; preserve logs, no retry')"))
    write(arm / 'failure.json', dict(at=22,
          error="RuntimeError('formal exited 1; preserve logs, no retry')"))
    write(arm / 'cleanup.json', dict(remaining_processes=[], remaining_owned_gpu_pids=[], owned=[]))
    write(lru / 'status.json', dict(status='prepared'))
    write(arm / 'jobs/formal/config.json', read(arm / 'job.json'))
    write(pair / 'pair-status.json', dict(status='failed', at=23, started_at=1,
          error="RuntimeError('thinking suite exited 1; preserve all attempts, no task retries')"))
    for p, pid in [(pair / 'pair-process.json', 101), (arm / 'queue-process.json', 102),
                   (arm / 'formal-process.json', 103), (arm / 'model/server-process.json', 104),
                   (arm / 'bridge-process.json', 105), (arm / 'gpu-monitor-process.json', 106)]:
        write(p, dict(pid=pid, created=1.0))
    write(pair / 'pair-child.json', dict(arm='thinking', pid=102, at=1))
    (pair / 'run_pair.py').write_text('# original controller\n')
    (pair / 'compare_terminalbench_policies.py').write_text('# original comparator\n')
    write(pair / 'prelaunch-audit.json', dict(passed=True,
          controller_sha256=sha(pair / 'run_pair.py'),
          comparator_sha256=sha(pair / 'compare_terminalbench_policies.py')))
    return dict(pair=pair, arm=arm, lru=lru, recovery=tmp_path / 'recovery',
                names=names, trials=trials, results=results, outcomes=outcomes,
                error=error, events=events, receipt=receipt_dir / 'result.json',
                checked=arm / checked_rel)


def test_network_receipt_accepts_stopped_container_and_original_error(campaign):
    c = campaign
    result = recovery.validate_network_receipt(c['arm'], c['error'], c['events'])
    assert result
    assert read(c['arm'] / 'formal-events/status.json')['network_errors'] == [c['error']]


@pytest.mark.parametrize('damage', ['network_present', 'receipt_hash', 'diff_bytes',
                                    'live_container', 'unknown_error', 'missing_failure_event'])
def test_network_receipt_rejects_unproven_reconciliation(campaign, damage):
    c = campaign
    error = deepcopy(c['error'])
    events = deepcopy(c['events'])
    receipt = read(c['receipt'])
    cid = next(iter(receipt['containers']))
    if damage == 'network_present': receipt['network_absent'] = False
    if damage == 'receipt_hash': receipt['original_checked_sha256'] = '0' * 64
    if damage == 'diff_bytes': (c['receipt'].parent / (cid + '.diff.txt')).write_text('D /answer.txt\n')
    if damage == 'live_container': receipt['containers'][cid]['state']['Running'] = True
    if damage == 'unknown_error': error['error'] = 'PermissionError(13)'
    if damage == 'missing_failure_event': events = [e for e in events if e['event'] != 'network_reclamation_failed']
    write(c['receipt'], receipt)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.validate_network_receipt(c['arm'], error, events)


def test_all_first_attempt_results_reconcile(campaign):
    result = recovery.validate_results(campaign['arm'])
    assert result['summary']['expected'] == 89
    assert result['summary']['passed'] == 44
    assert result['summary']['retries'] == 0
    assert len(result['summary']['outcomes']) == 89
    assert result['source_hashes']


@pytest.mark.parametrize('damage', ['missing_result', 'duplicate_outcome', 'different_score',
                                    'harbor_mismatch', 'cancelled_trial', 'unknown_task',
                                    'missing_release', 'release_error', 'cache_error', 'trial_config'])
def test_result_recovery_rejects_non_housekeeping_failures(campaign, damage):
    c = campaign
    status_path = c['arm'] / 'formal-events/status.json'
    status = read(status_path)
    harbor_path = c['arm'] / 'formal-events/harbor-result.json'
    harbor = read(harbor_path)
    if damage == 'missing_result': (c['arm'] / 'jobs/formal' / c['trials'][0] / 'result.json').unlink()
    if damage == 'duplicate_outcome': status['outcomes'][-1] = deepcopy(status['outcomes'][0])
    if damage == 'different_score':
        path = c['arm'] / 'formal-events' / (c['trials'][0] + '.json')
        raw = read(path); raw['verifier_result']['rewards']['reward'] = 1; write(path, raw)
    if damage == 'harbor_mismatch': harbor['trial_results'][0]['finished_at'] = '2026-10-29T10:00:03Z'
    if damage == 'cancelled_trial': harbor['stats']['n_cancelled_trials'] = 1
    if damage == 'unknown_task': status['outcomes'][0]['task'] = 'terminal-bench/foreign'
    if damage == 'missing_release':
        path = c['arm'] / 'formal-events/events.jsonl'
        events = [e for e in c['events'] if not (e['event'] == 'kv_released' and e['trial'] == c['trials'][0])]
        path.write_text(''.join(json.dumps(e) + '\n' for e in events))
    if damage == 'release_error': status['release_errors'] = [dict(trial=c['trials'][0], error='release failed')]
    if damage == 'cache_error': status['verifier_cache_errors'] = [dict(trial=c['trials'][0], error='cache failed')]
    if damage == 'trial_config':
        path = c['arm'] / 'jobs/formal' / c['trials'][0] / 'config.json'
        value = read(path); value['task']['path'] = '/different/' + c['names'][0]; write(path, value)
    write(status_path, status); write(harbor_path, harbor)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.validate_results(c['arm'])


def test_saved_trial_configuration_may_omit_an_expanded_default(campaign):
    c = campaign
    path = c['arm'] / 'jobs/formal' / c['trials'][0] / 'config.json'
    value = read(path); del value['source_trial']; write(path, value)
    assert recovery.validate_results(c['arm'])['passed'] is True


def prepared(c):
    path = c['pair'] / 'pair-status.json'
    failed = read(path)
    write(path, dict(status='running', arm='thinking', at=1, started_at=1))
    plan = recovery.prepare(c['pair'], c['recovery'])
    write(path, failed)
    return plan


def test_prepare_binds_read_only_reconciliation_without_changing_scores(campaign):
    c = campaign
    originals = {str(p): sha(p) for p in (c['arm'] / 'jobs/formal').glob('*/result.json')}
    plan = prepared(c)
    assert plan['allowed_error'] == recovery.ALLOWED_ERROR
    assert plan['task_retries_allowed'] is False
    assert plan['verifier_retries_allowed'] is False
    assert read(c['recovery'] / 'plan.json') == plan
    assert {name: sha(Path(name)) for name in originals} == originals
    assert not (c['arm'] / 'formal-events/summary.json').exists()


@pytest.mark.parametrize('damage', ['active_pair', 'active_model', 'cleanup_leftover',
                                    'lru_started', 'receipt_changed', 'foreign_failure',
                                    'pair_identity_replaced', 'suite_identity_replaced'])
def test_audit_refuses_to_cross_lifecycle_or_identity_boundaries(campaign, damage):
    c = campaign
    plan = prepared(c)
    probe = lambda _: False
    if damage == 'active_pair': probe = lambda row: row['pid'] == 101
    if damage == 'active_model': probe = lambda row: row['pid'] == 104
    if damage == 'cleanup_leftover':
        path = c['arm'] / 'cleanup.json'; value = read(path)
        value['remaining_processes'] = [dict(pid=107, created=1)]; write(path, value)
    if damage == 'lru_started': (c['lru'] / 'started').write_text('already attempted')
    if damage == 'receipt_changed':
        value = read(c['receipt']); value['no_score_changes'] = False; write(c['receipt'], value)
    if damage == 'foreign_failure':
        path = c['arm'] / 'failure.json'; value = read(path)
        value['error'] = "RuntimeError('Owned model exited')"; write(path, value)
    if damage in ('pair_identity_replaced', 'suite_identity_replaced'):
        path = c['pair'] / 'pair-process.json' if damage == 'pair_identity_replaced' else c['arm'] / 'queue-process.json'
        value = read(path); value['created'] = 2; write(path, value)
    with pytest.raises((ValueError, RuntimeError)):
        recovery.audit(c['pair'], c['recovery'], plan, live_probe=probe)
    assert not (c['arm'] / 'formal-events/summary.json').exists()


def test_recover_only_derives_missing_summary_and_preserves_original_errors(campaign):
    c = campaign
    plan = prepared(c)
    status_path = c['arm'] / 'formal-events/status.json'
    raw_paths = [status_path, c['arm'] / 'formal-events/events.jsonl',
                 *(c['arm'] / 'jobs/formal').glob('*/result.json')]
    original_hashes = {str(path): sha(path) for path in raw_paths}
    calls = []

    def finalize(arm, directory):
        calls.append((arm, directory))
        summary = read(arm / 'formal-events/summary.json')
        assert summary['outcomes'] == c['outcomes']
        assert summary['derived']['original_network_errors_preserved'] is True
        assert read(directory / 'original/thinking/formal-events/status.json')['network_errors'] == [c['error']]
        write(arm / 'status.json', dict(status='completed'))

    result = recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalize, live_probe=lambda _: False)
    assert result['passed'] is True
    assert len(calls) == 1
    assert {name: sha(Path(name)) for name in original_hashes} == original_hashes
    assert read(status_path)['network_errors'] == [c['error']]
    assert read(c['recovery'] / 'recovered.json')['original_files_preserved'] is True
    assert not (c['lru'] / 'started').exists()
    with pytest.raises((ValueError, RuntimeError)):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalize, live_probe=lambda _: False)
    assert len(calls) == 1


def test_recovery_detects_a_finalizer_that_rewrites_an_original_result(campaign):
    c = campaign
    plan = prepared(c)

    def bad_finalizer(arm, _):
        path = arm / 'jobs/formal' / c['trials'][0] / 'result.json'
        raw = read(path); raw['verifier_result']['rewards']['reward'] = 1; write(path, raw)
        write(arm / 'status.json', dict(status='completed'))

    with pytest.raises(ValueError, match='changed original evidence'):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=bad_finalizer, live_probe=lambda _: False)
    assert not (c['recovery'] / 'recovered.json').exists()
