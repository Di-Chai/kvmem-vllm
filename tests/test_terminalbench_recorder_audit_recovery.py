"""Pinned recorder setup failures stay missing scores in the original cohort."""
from copy import deepcopy
from datetime import datetime
import json

import pytest

from scripts import recover_terminalbench_recorder_audit as recovery
from test_terminalbench_network_audit_recovery import campaign as network_campaign
from test_terminalbench_network_audit_recovery import read, sha, write
from test_terminalbench_setup_audit_recovery import campaign as setup_campaign


QEMU = (
    ('qemu-alpine-ssh__GCUsTPV', 'e509396e94759f480ca217c80c48f896c2a1a4bb95286d84b74b55b594ae610f'),
    ('qemu-startup__oL6uCsv', '3e3e8522b6b05e769b45720fc9014bd855cb1cd1991e7b9c0357812858d46b11'),
)


def replace_with_qemu(c, index, trial, network, adapter_digest):
    arm = c['arm']
    name = trial.split('__')[0]
    previous = c['trials'][index]
    raw = deepcopy(c['results'][index])
    (arm / 'jobs/formal' / previous).rename(arm / 'jobs/formal' / trial)
    (arm / 'formal-events' / (previous + '.json')).unlink()
    remote = next(row['remote'] for row in recovery.INCIDENTS if row['trial'] == trial)
    probe = dict(stage='uploaded_adapter', timeout_sec=30, elapsed_seconds=0.2,
                 return_code=1, stderr=None,
                 command="printf '%s  %s\\n' " + adapter_digest + ' ' + remote +
                         ' | sha256sum -c - && /usr/bin/python3 ' + remote + ' --version',
                 stdout=remote + ': OK\nTraceback (most recent call last):\n'
                        '    from asciinema import recorder\n'
                        "ImportError: cannot import name 'recorder' from 'asciinema' "
                        '(/usr/lib/python3/dist-packages/asciinema/__init__.py)\n')
    raw.update(task_name='terminal-bench/' + name, trial_name=trial,
               agent_execution=None, agent_result=None, verifier_result=None,
               finished_at='2026-10-29T10:02:01Z',
               agent_setup=dict(started_at='2026-10-29T10:00:00Z',
                                finished_at='2026-10-29T10:02:00Z'),
               exception_info=dict(exception_type='RuntimeError',
                    exception_message='Recorder qualification failed: ' + json.dumps(probe),
                    exception_traceback='_setup_agent calls prepare_recorder and probe_command\n'
                        "ImportError: cannot import name 'recorder' from 'asciinema'"))
    raw['config']['trial_name'] = trial
    raw['config']['task']['path'] = str(arm / 'dataset/tasks' / name)
    write(arm / 'jobs/formal' / trial / 'result.json', raw)
    write(arm / 'jobs/formal' / trial / 'config.json', raw['config'])
    write(arm / 'formal-events' / (trial + '.json'), raw)
    c['names'][index] = name
    c['trials'][index] = trial
    c['results'][index] = raw
    c['outcomes'][index] = dict(task='terminal-bench/' + name, trial=trial,
                               rewards=None, exception='RuntimeError', status='error', passed=False)
    lifecycle = []
    for event in c['events']:
        event = deepcopy(event)
        if event['trial'] == previous:
            if event['event'] in ('agent-start', 'agent-end', 'verification-start'):
                continue
            event.update(trial=trial, task='terminal-bench/' + name)
            if event['event'] == 'kv_released':
                event['owners'] = []
        lifecycle.append(event)
    lifecycle.extend([dict(event='environment-start', at=10.5, trial=trial, task='terminal-bench/' + name),
                      dict(event='empty_networks_reclaimed', at=21, trial=trial, count=1)])
    c['events'] = lifecycle
    proof = dict(at=20, finished_at=21, status='removed', retained_filesystem_diffs_unchanged=True,
                 project=trial.lower() + '__env',
                 network=dict(Id=network, Containers={},
                              Labels={'com.docker.compose.project': trial.lower() + '__env'}),
                 environment_dir=str(arm / 'dataset/tasks' / name / 'environment'),
                 retained_containers={str(index) * 64: dict(image='sha256:setup-image',
                                                          state='exited', diff_sha256='f' * 64)})
    directory = arm / 'formal-events/network-cleanup' / trial
    write(directory / (network + '.removed.json'), proof)
    checked = dict(proof, status='checked')
    checked.pop('finished_at')
    checked.pop('retained_filesystem_diffs_unchanged')
    write(directory / (network + '.json'), checked)
    return dict(probe, event='recorder_probe', at=datetime.fromisoformat('2026-10-29T10:00:01+00:00').timestamp())


@pytest.fixture
def campaign(setup_campaign, monkeypatch):
    c = setup_campaign
    arm = c['arm']
    adapter = arm / 'runtime/scripts/terminalbench_pty.py'
    adapter.write_text('# A frozen uploaded adapter fixture.\n')
    digest = sha(adapter)
    monkeypatch.setattr(recovery, 'ADAPTER_SHA', digest)
    probes = [replace_with_qemu(c, index, trial, network, digest)
              for index, (trial, network) in enumerate(QEMU, 2)]
    (arm / 'formal-harness.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in probes))
    (arm / 'formal-events/events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in c['events']))
    status = read(arm / 'formal-events/status.json')
    status.update(outcomes=c['outcomes'], passed=42)
    write(arm / 'formal-events/status.json', status)
    harbor = read(arm / 'formal-events/harbor-result.json')
    harbor['trial_results'] = c['results']
    harbor['stats']['n_errored_trials'] = 3
    write(arm / 'formal-events/harbor-result.json', harbor)
    write(arm / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    for folder in (arm, c['lru']):
        manifest = read(folder / 'manifest.json')
        manifest['launched_task_names'] = c['names']
        if folder == arm:
            manifest['frozen']['runtime']['scripts/terminalbench_pty.py'] = digest
        write(folder / 'manifest.json', manifest)
    incidents = deepcopy(recovery.INCIDENTS)
    # The production trial/category/upload paths stay fixed; only fixture byte
    # hashes are substituted for test files with the same evidence structure.
    for incident in incidents:
        trial = incident['trial']
        incident['raw_sha256'] = sha(arm / 'jobs/formal' / trial / 'result.json')
        incident['event_sha256'] = sha(arm / 'formal-events' / (trial + '.json'))
    monkeypatch.setattr(recovery, 'INCIDENTS', incidents)
    monkeypatch.setattr(recovery, 'ANALYSIS_SHA', sha(c['analysis']))
    c.update(adapter=adapter, probes=probes)
    return c


def prepared(c):
    path = c['pair'] / 'pair-status.json'
    failed = read(path)
    write(path, dict(status='running', arm='thinking', at=1, started_at=1))
    plan = recovery.prepare(c['pair'], c['recovery'], c['analysis'])
    write(path, failed)
    return plan


def test_three_setup_failures_remain_unscored_original_attempts(campaign):
    c = campaign
    proofs = recovery.validate_setup_errors(c['arm'], c['events'])
    assert len(proofs) == 3
    assert {row['trial'] for row in proofs} == recovery.SETUP_TRIALS
    for proof in proofs:
        assert proof['reward'] is None
        assert proof['no_agent_execution'] and proof['no_verifier_execution'] and proof['no_model_requests']
    summary = recovery.validate_results(c['arm'])['summary']
    assert (summary['expected'], summary['scored'], summary['errors'], summary['passed']) == (89, 86, 3, 42)
    assert len(summary['outcomes']) == 89
    missing = [row for row in summary['outcomes'] if row['trial'] in recovery.SETUP_TRIALS]
    assert len(missing) == 3
    assert all(row['rewards'] is None and row['exception'] == 'RuntimeError' and row['status'] == 'error'
               for row in missing)
    assert summary['coverage']['fully_scored'] is False


@pytest.mark.parametrize('damage', ['raw_pin', 'event_pin', 'adapter_bytes', 'adapter_manifest',
    'missing_probe', 'duplicate_probe', 'probe_mismatch', 'probe_outside_setup',
    'agent_event', 'verifier_event', 'bridge_session', 'release_owner', 'network_proof'])
def test_recorder_incident_requires_uploaded_adapter_and_no_model_execution(campaign, damage):
    c = campaign
    arm = c['arm']
    incident = recovery.INCIDENTS[1]
    trial = incident['trial']
    lifecycle = deepcopy(c['events'])
    if damage in ('raw_pin', 'event_pin'):
        path = arm / ('jobs/formal/' + trial + '/result.json' if damage == 'raw_pin'
                      else 'formal-events/' + trial + '.json')
        path.write_text(path.read_text() + ' ')
    if damage == 'adapter_bytes':
        c['adapter'].write_text('# Uploaded code changed\n')
    if damage == 'adapter_manifest':
        manifest = read(arm / 'manifest.json')
        manifest['frozen']['runtime']['scripts/terminalbench_pty.py'] = '0' * 64
        write(arm / 'manifest.json', manifest)
    if damage in ('missing_probe', 'duplicate_probe', 'probe_mismatch', 'probe_outside_setup'):
        probes = deepcopy(c['probes'])
        if damage == 'missing_probe': probes.pop(0)
        if damage == 'duplicate_probe': probes.append(deepcopy(probes[0]))
        if damage == 'probe_mismatch': probes[0]['return_code'] = 0
        if damage == 'probe_outside_setup': probes[0]['at'] = 0
        (arm / 'formal-harness.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in probes))
    if damage in ('agent_event', 'verifier_event'):
        lifecycle.append(dict(event='agent-start' if damage == 'agent_event' else 'verification-start',
                              trial=trial, at=11))
    if damage == 'bridge_session':
        (arm / 'bridge/requests.jsonl').write_text(json.dumps(dict(session_id=trial + '__agent')) + '\n')
    if damage == 'release_owner':
        next(row for row in lifecycle if row['trial'] == trial and row['event'] == 'kv_released')['owners'] = ['owner']
    if damage == 'network_proof':
        path = arm / 'formal-events/network-cleanup' / trial / (incident['network'] + '.removed.json')
        proof = read(path)
        proof['retained_filesystem_diffs_unchanged'] = False
        write(path, proof)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.validate_setup_error(arm, lifecycle, incident)


@pytest.mark.parametrize('damage', ['upload_hash', 'upload_not_ok', 'other_import', 'wrong_stage', 'return_code'])
def test_probe_validator_rejects_different_recorder_failures(campaign, damage):
    c = campaign
    incident = recovery.INCIDENTS[1]
    raw = read(c['arm'] / 'jobs/formal' / incident['trial'] / 'result.json')
    prefix = 'Recorder qualification failed: '
    detail = json.loads(raw['exception_info']['exception_message'][len(prefix):])
    if damage == 'upload_hash': detail['command'] = detail['command'].replace(recovery.ADAPTER_SHA, '0' * 64)
    if damage == 'upload_not_ok': detail['stdout'] = detail['stdout'].replace(': OK\n', ': FAILED\n')
    if damage == 'other_import': detail['stdout'] = detail['stdout'].replace('from asciinema import recorder', 'from asciinema import command')
    if damage == 'wrong_stage': detail['stage'] = 'recording_tools'
    if damage == 'return_code': detail['return_code'] = 2
    raw['exception_info']['exception_message'] = prefix + json.dumps(detail)
    with pytest.raises(ValueError):
        recovery.validate_recorder_probe(c['arm'], incident, raw)


@pytest.mark.parametrize('damage', ['fourth_error', 'raw_disagreement', 'scored_missing_agent',
    'harbor_error_count', 'imputed_reward', 'duplicate_attempt'])
def test_setup_exception_does_not_relax_other_original_attempts(campaign, damage):
    c = campaign
    arm = c['arm']
    if damage in ('fourth_error', 'imputed_reward', 'duplicate_attempt'):
        path = arm / 'formal-events/status.json'
        status = read(path)
        if damage == 'fourth_error':
            status['outcomes'][4].update(rewards=None, exception='RuntimeError', status='error', passed=False)
        if damage == 'imputed_reward':
            status['outcomes'][2].update(rewards={'reward': 0}, exception=None, status='scored')
        if damage == 'duplicate_attempt':
            status['outcomes'][-1] = deepcopy(status['outcomes'][0])
        write(path, status)
    if damage in ('raw_disagreement', 'harbor_error_count'):
        path = arm / 'formal-events/harbor-result.json'
        harbor = read(path)
        if damage == 'raw_disagreement': harbor['trial_results'][2]['exception_info']['exception_message'] = 'Other error'
        if damage == 'harbor_error_count': harbor['stats']['n_errored_trials'] = 2
        write(path, harbor)
        write(arm / 'jobs/formal/result.json', {k: v for k, v in harbor.items() if k != 'trial_results'})
    if damage == 'scored_missing_agent':
        lifecycle = [row for row in c['events'] if not (row['trial'] == c['trials'][4] and row['event'] == 'agent-start')]
        (arm / 'formal-events/events.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in lifecycle))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.validate_results(arm)


def test_recovery_preserves_every_original_score_and_setup_exception(campaign):
    c = campaign
    plan = prepared(c)
    audit = recovery.audit(c['pair'], c['recovery'], plan, live_probe=lambda _: False)
    originals = {name: sha(c['arm'] / name) for name in audit['source_hashes']}

    def finalizer(arm, _):
        summary = read(arm / 'formal-events/summary.json')
        assert (summary['scored'], summary['errors']) == (86, 3)
        assert summary['derived']['no_reward_imputed'] is True
        assert read(arm / 'formal-events/status.json')['network_errors'] == [c['error']]
        write(arm / 'status.json', dict(status='completed'))
        write(arm / 'results.json', dict(scored=86, errors=3))

    recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalizer, live_probe=lambda _: False)
    assert {name: sha(c['arm'] / name) for name in originals} == originals
    assert read(c['recovery'] / 'original/pair-status.json')['status'] == 'failed'
    assert read(c['pair'] / 'pair-status.json')['status'] == 'failed'
    assert not (c['lru'] / 'started').exists()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalizer, live_probe=lambda _: False)


@pytest.mark.parametrize('damage', ['active_process', 'cleanup_leftover', 'lru_started', 'analysis_changed',
                                    'adapter_changed', 'existing_summary'])
def test_audit_waits_for_original_cleanup_and_protects_bound_inputs(campaign, damage):
    c = campaign
    plan = prepared(c)
    if damage == 'cleanup_leftover':
        write(c['arm'] / 'cleanup.json', dict(remaining_processes=[123], remaining_owned_gpu_pids=[], owned=[]))
    if damage == 'lru_started': (c['lru'] / 'started').write_text('1')
    if damage == 'analysis_changed': c['analysis'].write_text('# Changed analysis\n')
    if damage == 'adapter_changed': c['adapter'].write_text('# Changed adapter\n')
    if damage == 'existing_summary': write(c['arm'] / 'formal-events/summary.json', dict(scored=86))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        recovery.audit(c['pair'], c['recovery'], plan, live_probe=lambda _: damage == 'active_process')
    assert not (c['recovery'] / 'original').exists()


def test_finalizer_cannot_impute_missing_scores(campaign):
    c = campaign
    plan = prepared(c)

    def finalizer(arm, _):
        write(arm / 'status.json', dict(status='completed'))
        write(arm / 'results.json', dict(scored=89, errors=0))

    with pytest.raises(ValueError, match='missing-score coverage'):
        recovery.recover(c['pair'], c['recovery'], plan, finalizer=finalizer, live_probe=lambda _: False)
    assert not (c['recovery'] / 'recovered.json').exists()


def test_prepare_requires_the_qualified_analysis_bytes(campaign):
    c = campaign
    c['analysis'].write_text('# Similar name but unreviewed analyzer\n')
    write(c['pair'] / 'pair-status.json', dict(status='running', arm='thinking', started_at=1))
    with pytest.raises(ValueError, match='separately frozen'):
        recovery.prepare(c['pair'], c['recovery'], c['analysis'])
    assert not c['recovery'].exists()

