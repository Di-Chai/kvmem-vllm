import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('agent_qualification', Path(__file__).parents[1] / 'scripts/qualify_agent.py')
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def result(root, value, exception=None, name='trial'):
    path = root / 'pier_jobs/job' / name / 'result.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'verifier_result': {'rewards': {'reward': value}}, 'exception_info': exception}))


def test_zero_reward_and_official_timeout_are_not_retried_or_hidden(tmp_path):
    result(tmp_path, 0, {'exception_type': 'AgentTimeoutError'})
    actual = agent.scored_result(tmp_path)
    assert actual['status'] == 'scored' and actual['reward'] == 0
    cmd = agent.make_command(Path('/pinned/bin/pier'), 'task', Path('/task'), tmp_path,
                             'http://172.17.0.1:443/v1', 'qualification')
    for flag in ('--n-attempts', '--n-concurrent'):
        assert cmd[cmd.index(flag) + 1] == '1'
    assert cmd[cmd.index('--max-retries') + 1] == '0'
    assert cmd[cmd.index('--agent-timeout-multiplier') + 1] == '1.0'


@pytest.mark.parametrize('reward', [None, True, float('nan'), float('inf'), '1'])
def test_missing_or_invalid_scores_are_not_quality_evidence(tmp_path, reward):
    result(tmp_path, reward)
    assert agent.scored_result(tmp_path)['status'] == 'unscored'


def test_infrastructure_exception_is_preserved_even_with_numeric_reward(tmp_path):
    result(tmp_path, 1, {'exception_type': 'VerifierInfrastructureError'})
    assert agent.scored_result(tmp_path)['status'] == 'scored_with_unclassified_exception'


def test_extra_attempt_invalidates_one_attempt_qualification(tmp_path):
    result(tmp_path, 1)
    result(tmp_path, None, name='retry')
    assert agent.scored_result(tmp_path)['status'] == 'unscored'


def test_snapshots_are_detached_and_external_symlinks_are_rejected(tmp_path):
    source, copied = tmp_path / 'source', tmp_path / 'copy'
    source.mkdir()
    (source / 'task.toml').write_text('locked task')
    hashes = agent.snapshot_task(source, copied)
    (source / 'task.toml').write_text('later source edit')
    assert (copied / 'task.toml').read_text() == 'locked task'
    assert hashes['task.toml'] == agent.digest(copied / 'task.toml')
    (source / 'external').symlink_to(tmp_path / 'copy')
    with pytest.raises(ValueError, match='symlinks'):
        agent.snapshot_task(source, tmp_path / 'second')


def test_session_agent_gate_requires_same_worker_long_restore_and_cancel(tmp_path):
    engine = {'manager_pid': 123, 'api_base': 'local', 'worker_manifest_sha256': 'frozen'}
    long = tmp_path/'long.json'
    concurrent = tmp_path/'concurrent.json'
    long.write_text(json.dumps({'engine': engine, 'status': 'passed',
                                'prompt_tokens': 134067, 'imported_tokens': 133120}))
    data = {'engine': engine, 'status': 'passed', 'concurrency': 4,
            'phases': {'same_session_overlap': {}},
            'cancellation': {'terminal': {'terminal_prefix': {'status': 'FINISHED_ABORTED'}}}}
    concurrent.write_text(json.dumps(data))
    assert len(agent.validate_session_gates([long, concurrent], engine, 4)) == 2
    with pytest.raises(ValueError, match='concurrent cancellation'):
        agent.validate_session_gates([long], engine, 2)
    with pytest.raises(ValueError, match='same frozen backend'):
        agent.validate_session_gates([long, concurrent], {**engine, 'manager_pid': 124}, 4)
    data['status'] = 'failed'
    concurrent.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='same frozen backend'):
        agent.validate_session_gates([long, concurrent], engine, 4)


def test_reused_relay_journal_excludes_preflight_and_other_run_requests():
    record = {'task_id': 'task-a', 'path': '/v1/chat/completions', 'submitted_unix': 12.5}
    assert agent.belongs_to_run(record, {'task-a'}, 10, 20)
    for changes in ({'submitted_unix': 9}, {'submitted_unix': 21}, {'submitted_unix': None},
                    {'task_id': 'preflight'}, {'path': '/v1/models'}):
        assert not agent.belongs_to_run({**record, **changes}, {'task-a'}, 10, 20)
