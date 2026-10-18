import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPT))
spec = importlib.util.spec_from_file_location('formal_latency_analysis', SCRIPT/'analyze_formal_latency.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.pop(0)


def sample(begin, end, count):
    h = 'vllm:inter_token_latency_seconds'
    return dict(started_at=begin, finished_at=end, preflight_marker_mtime=1,
        observer_sha256='fixed', interval_seconds=2,
        metrics=dict(buckets={'0.1': count, '+Inf': count}, values={h+'_count': count,
            h+'_sum': count*.05, 'vllm:generation_tokens_total': count*4,
            'vllm:num_preemptions_total': 0}))


def fixtures():
    run = dict(status='completed', tasks={str(i):dict(state='completed') for i in range(16)})
    events = [dict(event='start', task='0', turn=1, request_started_at=5, at=6),
              dict(event='finish', task='0', turn=1, at=10, metrics=dict(time_to_first_token_ms=1000,
                    generation_time_ms=2000), usage=dict(completion_tokens=101)),
              dict(event='history_released', at=11)]
    return run, events, [sample(2, 3, 100), sample(12, 13, 120)]


def test_complete_boundary_excludes_preflight_counts_and_separates_tpot():
    result = m.analyze(*fixtures())
    assert result['complete_formal_coverage']
    assert result['native_output_intervals']['events'] == 20
    assert result['request_metrics']['mean_decode_seconds_per_output_token']['p95'] == .02


def test_snapshot_overlapping_first_request_cannot_be_baseline():
    run, events, samples = fixtures()
    samples[0]['finished_at'] = 5.5
    result = m.analyze(run, events, samples)
    assert not result['complete_formal_coverage']
    assert 'native_output_intervals' not in result


def test_snapshot_before_preflight_completion_cannot_be_baseline():
    run, events, samples = fixtures()
    samples[0]['preflight_marker_mtime'] = 2.5
    assert not m.analyze(run, events, samples)['complete_formal_coverage']


def test_final_response_and_cleanup_must_be_covered():
    run, events, samples = fixtures()
    samples[-1] = sample(10.5, 11.5, 120)
    assert not m.analyze(run, events, samples)['complete_formal_coverage']


def test_running_tasks_are_not_complete_even_with_late_snapshot():
    run, events, samples = fixtures()
    run['status'] = 'running'
    assert not m.analyze(run, events, samples)['complete_formal_coverage']


def test_unclosed_request_invalidates_claim_even_with_completed_task_state():
    run, events, samples = fixtures()
    events.insert(-1, dict(event='start', task='0', turn=2, request_started_at=10.5, at=10.6))
    result = m.analyze(run, events, samples)
    assert not result['complete_formal_coverage']
    assert result['pending_model_requests'] == 1


def test_rejects_changed_observer_and_counter_resets():
    run, events, samples = fixtures()
    samples[-1]['observer_sha256'] = 'changed'
    with pytest.raises(ValueError, match='Observer'):
        m.analyze(run, events, samples)
    samples[-1] = sample(12, 13, 90)
    with pytest.raises(ValueError, match='Counter reset'):
        m.analyze(run, events, samples)


def test_no_formal_requests_has_no_latency_claim():
    run, events, samples = fixtures()
    result = m.analyze(run, [], samples)
    assert not result['complete_formal_coverage']
    assert 'native_output_intervals' not in result
