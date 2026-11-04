"""Per-scope audit uses final native policy counters, not shutdown release."""
import json

from test_historical_manager import actual_archive, append, eventually, steps


def test_scope_audit_keeps_policy_retirement_separate_from_terminal_cleanup(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    path = tmp_path/'scopes.jsonl'
    archive.history_scope_metrics_path = path
    session = ['bridge-owner', 'cache-salt']
    archive.set_history_scope_context('a', session, 7)
    session[0] = 'mutated-caller-copy'
    leases, _ = append(archive, repo, scope)
    try:
        steps(archive, worker, scope)
        assert scope.stats['retired'] == scope.stats['cpu_released'] == 4
        assert not path.exists()
    finally:
        leases.clear()
        eventually(archive.close)
    final = dict(scope.stats)
    assert final['retired'] == 4
    assert final['cpu_released'] == final['slots_released'] == 8
    archive.unregister_history_scope('a', scope)
    archive.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row['event'] == 'history_scope_closed' and row['namespace'] == 'a'
    assert row['policy'] == 'lru' and row['stats'] == final
    assert row['session_namespace'] == ['bridge-owner', 'cache-salt']
    assert row['session_generation'] == 7
    assert not archive._history_scope_context
    history = archive.history_stats()
    assert history['retired'] == 4 and history['cpu_released'] == 8
    assert history['scope_metrics']['records'] == 1
    assert history['scope_metrics']['errors'] == 0


def test_scope_audit_io_failure_is_visible_and_cannot_leak_closed_history(tmp_path):
    archive, worker, repo, scope = actual_archive(tmp_path)
    # Opening a directory as the JSONL file fails on the real filesystem.
    archive.history_scope_metrics_path = tmp_path
    archive.set_history_scope_context('a', ('owner', 'salt'), 1)
    leases, _ = append(archive, repo, scope)
    leases.clear()
    eventually(archive.close)
    archive.unregister_history_scope('a', scope)
    history = archive.history_stats()
    assert history['active_scopes'] == 0 and not archive._history_scope_context
    assert archive.io_usage()['cpu_authority_bytes'] == 0
    assert archive.tier is None
    assert history['scope_metrics']['records'] == 0
    assert history['scope_metrics']['errors'] == 1
    assert 'IsADirectoryError' in history['scope_metrics']['last_error']


def test_scope_audit_disabled_does_not_retain_context_or_write_files(tmp_path):
    archive, _, _, _ = actual_archive(tmp_path)
    archive.set_history_scope_context('a', ('owner', 'salt'), 1)
    eventually(archive.close)
    assert not archive._history_scope_context
    assert 'scope_metrics' not in archive.history_stats()
    assert not list(tmp_path.glob('*.jsonl'))
