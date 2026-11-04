"""A pinned setup failure remains an unscored first attempt, never a reward."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import recover_terminalbench_setup_audit as recovery
from test_terminalbench_network_audit_recovery import campaign as network_campaign
from test_terminalbench_network_audit_recovery import read, sha, write


@pytest.fixture
def campaign(network_campaign, monkeypatch):
    c = network_campaign
    arm = c['arm']
    original_trial = c['trials'][1]
    trial = 'dna-insert__GKooYyr'
    name = 'dna-insert'
    original_raw = arm / 'jobs/formal' / original_trial / 'result.json'
    raw = read(original_raw)
    (arm / 'jobs/formal' / original_trial).rename(arm / 'jobs/formal' / trial)
    (arm / 'formal-events' / (original_trial + '.json')).unlink()
    raw.update(task_name='terminal-bench/' + name, trial_name=trial,
               agent_execution=None, agent_result=None, verifier_result=None,
               finished_at='2026-10-29T10:02:01Z',
               agent_setup=dict(started_at='2026-10-29T10:00:00Z',
                                finished_at='2026-10-29T10:02:00Z'),
               exception_info=dict(exception_type='RuntimeError',
                    exception_message='Command timed out after 120 seconds',
                    exception_traceback='RuntimeError: Command timed out after 120 seconds\n'
                                        '_install_recording_tools during _setup_agent'))
    raw['config']['trial_name'] = trial
    raw['config']['task']['path'] = str(arm / 'dataset/tasks' / name)
    raw_path = arm / 'jobs/formal' / trial / 'result.json'
    event_path = arm / 'formal-events' / (trial + '.json')
    write(raw_path, raw); write(event_path, raw)
    write(arm / 'jobs/formal' / trial / 'config.json', raw['config'])
    c['names'][1] = name; c['trials'][1] = trial; c['results'][1] = raw
    c['outcomes'][1] = dict(task='terminal-bench/' + name, trial=trial, rewards=None,
                            exception='RuntimeError', status='error', passed=False)
    status_path = arm / 'formal-events/status.json'
    status = read(status_path); status.update(outcomes=c['outcomes'], passed=43)
    write(status_path, status)
    harbor_path = arm / 'formal-events/harbor-result.json'
    harbor = read(harbor_path); harbor['trial_results'][1] = raw; harbor['stats']['n_errored_trials'] = 1
    write(harbor_path, harbor)
    write(arm / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    for folder in (arm, c['lru']):
        manifest = read(folder / 'manifest.json'); manifest['launched_task_names'] = c['names']
        write(folder / 'manifest.json', manifest)
    lifecycle = []
    for event in c['events']:
        event = deepcopy(event)
        if event['trial'] == original_trial:
            if event['event'] in ('agent-start', 'agent-end', 'verification-start'):
                continue
            event.update(trial=trial, task='terminal-bench/' + name)
            if event['event'] == 'kv_released': event['owners'] = []
        lifecycle.append(event)
    lifecycle.extend([dict(event='environment-start', at=10.5, trial=trial, task='terminal-bench/' + name),
                      dict(event='empty_networks_reclaimed', at=21, trial=trial, count=1)])
    c['events'] = lifecycle
    (arm / 'formal-events/events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in lifecycle))
    (arm / 'bridge').mkdir()
    (arm / 'bridge/requests.jsonl').write_text('')
    network = recovery.SETUP_NETWORK
    proof = dict(at=20, finished_at=21, status='removed',
                 retained_filesystem_diffs_unchanged=True, project=trial.lower() + '__env',
                 network=dict(Id=network, Containers={},
                              Labels={'com.docker.compose.project': trial.lower() + '__env'}),
                 environment_dir=str(arm / 'dataset/tasks/dna-insert/environment'),
                 retained_containers={'e' * 64: dict(image='sha256:setup-image', state='exited', diff_sha256='f' * 64)})
    write(arm / 'formal-events/network-cleanup' / trial / (network + '.removed.json'), proof)
    checked = dict(proof, status='checked')
    checked.pop('finished_at'); checked.pop('retained_filesystem_diffs_unchanged')
    write(arm / 'formal-events/network-cleanup' / trial / (network + '.json'), checked)
    analysis = c['pair'] / 'analysis-wrapper.py'
    analysis.write_text('# Bound analysis wrapper fixture; never executed in unit tests.\n')
    c.update(setup_trial=trial, setup_raw=raw_path, setup_event=event_path, analysis=analysis)
    # Exact production incident pins remain hard-coded; only fixture byte pins
    # are substituted, never the permitted trial or failure category.
    monkeypatch.setattr(recovery, 'RAW_SHA', sha(raw_path))
    monkeypatch.setattr(recovery, 'EVENT_RESULT_SHA', sha(event_path))
    return c


def test_pinned_setup_failure_is_retained_without_synthesizing_a_score(campaign):
    c = campaign
    proof = recovery.validate_setup_error(c['arm'], c['events'])
    assert proof
    result = recovery.validate_results(c['arm'])
    summary = result['summary']
    assert summary['expected'] == 89
    assert summary['scored'] == 88
    assert summary['errors'] == 1
    assert summary['passed'] == 43
    row = next(r for r in summary['outcomes'] if r['trial'] == c['setup_trial'])
    assert row['rewards'] is None and row['exception'] == 'RuntimeError'
    assert row['status'] == 'error' and row['passed'] is False


@pytest.mark.parametrize('damage', ['raw_pin', 'event_pin', 'agent_event', 'verifier_event',
                                    'bridge_owner', 'nonempty_release', 'missing_network_cleanup'])
def test_setup_exception_requires_exact_identity_and_no_agent_execution(campaign, damage):
    c = campaign
    lifecycle = deepcopy(c['events'])
    if damage in ('raw_pin', 'event_pin'):
        path = c['setup_raw'] if damage == 'raw_pin' else c['setup_event']
        path.write_text(path.read_text() + ' ')
    if damage in ('agent_event', 'verifier_event'):
        lifecycle.append(dict(event='agent-start' if damage == 'agent_event' else 'verification-start',
                              trial=c['setup_trial'], at=12))
    if damage == 'bridge_owner':
        (c['arm'] / 'bridge/requests.jsonl').write_text(json.dumps(dict(event='session_opened',
              owner='forbidden-owner', session_id=c['setup_trial'] + '__agent')) + '\n')
    if damage == 'nonempty_release':
        next(e for e in lifecycle if e['trial'] == c['setup_trial'] and e['event'] == 'kv_released')['owners'] = ['forbidden-owner']
    if damage == 'missing_network_cleanup':
        lifecycle = [e for e in lifecycle if not (e['trial'] == c['setup_trial'] and e['event'] == 'empty_networks_reclaimed')]
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.validate_setup_error(c['arm'], lifecycle)


@pytest.mark.parametrize('damage', ['second_error', 'raw_disagreement', 'scored_missing_agent',
                                    'harbor_error_count', 'rewritten_reward'])
def test_no_other_attempt_can_be_relaxed_by_the_setup_exception(campaign, damage):
    c = campaign
    if damage == 'second_error':
        path = c['arm'] / 'formal-events/status.json'; status = read(path)
        status['outcomes'][2].update(rewards=None, exception='RuntimeError', status='error', passed=False)
        write(path, status)
    if damage == 'raw_disagreement':
        path = c['arm'] / 'formal-events/harbor-result.json'; harbor = read(path)
        harbor['trial_results'][1]['exception_info']['exception_message'] = 'different failure'
        write(path, harbor)
        write(c['arm'] / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    if damage == 'scored_missing_agent':
        lifecycle = [e for e in c['events'] if not (e['trial'] == c['trials'][2] and e['event'] == 'agent-start')]
        (c['arm'] / 'formal-events/events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in lifecycle))
    if damage == 'harbor_error_count':
        path = c['arm'] / 'formal-events/harbor-result.json'; harbor = read(path)
        harbor['stats']['n_errored_trials'] = 0
        write(path, harbor)
        write(c['arm'] / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    if damage == 'rewritten_reward':
        path = c['arm'] / 'formal-events/status.json'; status = read(path)
        status['outcomes'][1].update(rewards={'reward': 0}, exception=None, status='scored')
        write(path, status)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.validate_results(c['arm'])


def prepared(c):
    path = c['pair'] / 'pair-status.json'; failed = read(path)
    write(path, dict(status='running', arm='thinking', at=1, started_at=1))
    plan = recovery.prepare(c['pair'], c['recovery'], c['analysis'])
    write(path, failed)
    return plan


def test_recovery_keeps_the_unscored_result_and_bound_analysis(campaign):
    c = campaign
    plan = prepared(c)
    original = {str(path): sha(path) for path in [c['setup_raw'], c['setup_event'],
                c['arm'] / 'formal-events/status.json', c['arm'] / 'formal-events/events.jsonl']}

    def finalize(arm, _):
        summary = read(arm / 'formal-events/summary.json')
        assert summary['scored'] == 88 and summary['errors'] == 1
        assert next(r for r in summary['outcomes'] if r['trial'] == c['setup_trial'])['rewards'] is None
        write(arm / 'status.json', dict(status='completed'))
        write(arm / 'results.json', dict(scored=88, errors=1))

    recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalize, live_probe=lambda _: False)
    assert {name: sha(Path(name)) for name in original} == original
    assert not (c['lru'] / 'started').exists()
    with pytest.raises((ValueError, RuntimeError)):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalize, live_probe=lambda _: False)


def test_analysis_wrapper_change_blocks_the_continuation(campaign):
    c = campaign
    plan = prepared(c)
    c['analysis'].write_text('# changed after qualification\n')
    with pytest.raises((ValueError, RuntimeError)):
        recovery.audit(c['pair'], c['recovery'], plan, live_probe=lambda _: False)
    assert not (c['arm'] / 'formal-events/summary.json').exists()


def test_finalizer_cannot_claim_the_missing_score_was_recovered(campaign):
    c = campaign
    plan = prepared(c)

    def misleading_finalizer(arm, _):
        write(arm / 'status.json', dict(status='completed'))
        write(arm / 'results.json', dict(scored=89, errors=0))

    with pytest.raises(ValueError, match='missing-score coverage'):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=misleading_finalizer,
                         live_probe=lambda _: False)
    assert not (c['recovery'] / 'recovered.json').exists()


@pytest.mark.parametrize('location', ['outside_campaign', 'original_comparator'])
def test_prepare_requires_a_separate_campaign_local_analysis_wrapper(campaign, location):
    c = campaign
    if location == 'outside_campaign':
        analysis = c['pair'].parent / 'foreign-wrapper.py'; analysis.write_text('# separate file\n')
    else:
        analysis = c['pair'] / 'compare_terminalbench_policies.py'
    path = c['pair'] / 'pair-status.json'
    write(path, dict(status='running', arm='thinking', started_at=1))
    with pytest.raises(ValueError, match='separately frozen'):
        recovery.prepare(c['pair'], c['recovery'], analysis)
    assert not c['recovery'].exists()
