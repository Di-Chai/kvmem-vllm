import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('query_stage_analyzer',
    Path(__file__).resolve().parents[1]/'scripts/analyze_query_stage_profile.py')
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


def write_report(tmp_path, **overrides):
    rows = [dict(sequence=1, parent=None, stage='fill', started_at=0., finished_at=5.,
                 wall_seconds=5., thread_cpu_seconds=2.),
            dict(sequence=2, parent=1, stage='read', started_at=1., finished_at=4.,
                 wall_seconds=3., thread_cpu_seconds=1.),
            dict(stage='archive_report', stats={'cache_copy_seconds': 1.5})]
    report = dict(request_id='r', stages=rows, unresolved_gpu_events=0,
                  dropped_rows=0, dropped_gpu_events=0)
    report.update(overrides)
    path = tmp_path/'query_stage_profile.jsonl'
    path.write_text(json.dumps(report)+'\n')
    return path


def test_nested_read_is_not_double_counted(tmp_path):
    request = analyzer.summarize(write_report(tmp_path))['requests'][0]
    assert request['stages']['fill']['host_exclusive_seconds'] == 2.
    assert request['stages']['read']['host_exclusive_seconds'] == 3.
    assert request['archive_report_sums']['cache_copy_seconds'] == 1.5


@pytest.mark.parametrize('key', ['unresolved_gpu_events', 'dropped_rows', 'dropped_gpu_events'])
def test_incomplete_observation_cannot_pass(tmp_path, key):
    with pytest.raises(ValueError, match='Incomplete'):
        analyzer.summarize(write_report(tmp_path, **{key: 1}))
