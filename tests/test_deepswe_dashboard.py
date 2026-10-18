import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).resolve().parents[1] / 'scripts/deepswe_dashboard.py')
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


def sample(at, prompt, cached, output, phase='running'):
    return {'at': at, 'run_phase': phase, 'counters': {
        dashboard.PROMPT: prompt, dashboard.CACHED: cached, dashboard.OUTPUT: output}}


def test_excludes_preflight_and_cached_history_from_throughput():
    rows = [sample(90, 5000, 1000, 2000, 'starting'), sample(100, 7000, 2000, 2100),
            sample(110, 18000, 12000, 2400)]
    series = dashboard.throughput_series(rows, 99)
    assert len(series) == 2
    assert series[0]['prefill'] is None
    assert series[-1]['prefill'] == 100  # 11K prompt, but 10K restored history.
    assert series[-1]['decode'] == 30


def test_preflight_rate_is_separate_and_cannot_cross_formal_boundary():
    rows = [sample(90, 5000, 1000, 2000, 'starting'),
            sample(100, 7000, 2000, 2100, 'starting'), sample(110, 18000, 12000, 2400)]
    preflight = dashboard.throughput_series(rows, 80, 'starting')
    assert len(preflight) == 2 and preflight[-1]['decode'] == 10
    assert dashboard.throughput_series(rows, 105)[0]['decode'] is None


def test_reset_missing_metrics_and_collection_gaps_are_not_fake_zeroes():
    rows = [sample(100, 100, 0, 50), sample(110, 10, 0, 1)]
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None
    rows.append({'at': 120, 'error': 'offline'})
    rows.append(sample(130, 20, 0, 2))
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None
    rows.append(sample(200, 30, 0, 3))
    assert dashboard.throughput_series(rows, 99)[-1]['prefill'] is None
    rows.append({'at': 210, 'run_phase': 'running', 'counters': {}})
    assert dashboard.throughput_series(rows, 99)[-1]['decode'] is None


def test_partial_json_line_is_consumed_once_when_complete(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_bytes(b'{"at":1}\n{"at":')
    reader = dashboard.JsonLines(path)
    assert reader.update() == [{'at': 1}]
    assert reader.update() == []
    with path.open('ab') as stream:
        stream.write(b'2}\n')
    assert reader.update() == [{'at': 2}]
    assert list(reader.rows) == [{'at': 1}, {'at': 2}]


def test_all_manifest_tasks_visible_and_ended_run_not_reported_as_active(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'tasks': [{'task_id': 'a'}, {'task_id': 'b'}]}))
    (tmp_path / 'run.json').write_text(json.dumps({'status': 'needs_attention', 'tasks': {
        'a': {'state': 'running', 'started_at': 100}}, 'model_ready_at': 100, 'ended_at': 130}))
    state = dashboard.Dashboard(tmp_path).snapshot()
    assert [t['state'] for t in state['tasks']] == ['interrupted', 'queued']
    assert state['elapsed_seconds'] == 30
    assert state['latest'] == {}
