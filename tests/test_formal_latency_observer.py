import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('formal_latency',
    Path(__file__).resolve().parents[1]/'scripts/observe_formal_latency.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def metrics(counts, total):
    h = m.HISTOGRAM
    return m.parse_metrics('\n'.join(
        [f'{h}_bucket{{engine="0",le="{b}",model_name="test"}} {n}'
         for b, n in zip(('0.01', '0.1', '+Inf'), counts)] +
        [f'{h}_sum{{engine="0"}} {total}', f'{h}_count{{engine="0"}} {counts[-1]}',
         'vllm:generation_tokens_total{engine="0"} 1000',
         'vllm:num_preemptions_total{engine="0"} 0']))


def test_subtracts_preflight_before_quantiles():
    result = m.delta_histogram(metrics([100, 100, 100], 1), metrics([100, 120, 120], 2))
    assert result['events'] == 20
    assert result['quantile_bucket_bounds']['p95'] == dict(lower_seconds=.01, upper_seconds=.1)
    assert result['mean_seconds'] == .05


def test_does_not_invent_tail_beyond_largest_bucket():
    result = m.delta_histogram(metrics([0, 0, 0], 0), metrics([0, 9, 10], 2))
    assert result['quantile_bucket_bounds']['p95'] == dict(lower_seconds=.1, upper_seconds=None)


def test_no_events_has_no_quantile():
    value = metrics([0, 0, 0], 0)
    assert m.delta_histogram(value, value)['quantile_bucket_bounds']['p95'] is None


def test_rejects_counter_reset():
    with pytest.raises(ValueError, match='Counter reset'):
        m.delta_histogram(metrics([10, 10, 10], 1), metrics([0, 1, 1], .1))


def test_rejects_changed_buckets():
    before, after = metrics([0, 0, 0], 0), metrics([0, 1, 1], .1)
    after['buckets']['0.2'] = after['buckets'].pop('0.1')
    with pytest.raises(ValueError, match='boundaries changed'):
        m.delta_histogram(before, after)


def test_rejects_missing_and_multiple_engines():
    with pytest.raises(ValueError, match='Incomplete'):
        m.parse_metrics('')
    with pytest.raises(ValueError, match='unique buckets'):
        m.parse_metrics(f'{m.HISTOGRAM}_bucket{{engine="0",le="+Inf"}} 1\n'
                        f'{m.HISTOGRAM}_bucket{{engine="1",le="+Inf"}} 1')


def test_completed_arm_cannot_capture_next_server(tmp_path, monkeypatch):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='failed', stage='deepswe-P1')))
    for arm in ('A0', 'P1'):
        root = tmp_path/('deepswe-'+arm)
        root.mkdir()
        for name, payload in [('run.json', dict(status='completed')),
                              ('profile.json', dict(port=18482))] + [
                (name, dict(passed=True)) for name in ('atomic-query-preflight.json',
                'short-tail-preflight.json', 'generation-preflight.json', 'cleanup-preflight.json')]:
            (root/name).write_text(json.dumps(payload))
    value = metrics([0, 1, 1], .1)
    monkeypatch.setattr(m, 'parse_metrics', lambda body: value)
    calls = []
    def fetch(url, timeout):
        calls.append(url)
        return io.BytesIO(b'')
    monkeypatch.setattr(m.urllib.request, 'build_opener', lambda *args: SimpleNamespace(open=fetch))
    m.observe(tmp_path, 2)
    assert len(calls) == 1
    assert not (tmp_path/'deepswe-A0/latency-metrics.jsonl').exists()
    assert (tmp_path/'deepswe-P1/latency-metrics.jsonl').exists()
