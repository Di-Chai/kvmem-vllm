"""Offline synthetic cohort evidence: no server, model, or GPU requests."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('decode_cohorts', scripts / 'analyze_decode_cohorts.py')
analysis = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(scripts))
try:
    spec.loader.exec_module(analysis)
finally:
    sys.path.remove(str(scripts))


def trace(index, offset=0):
    # No token arrivals during seconds 3 and 4; the stall must remain counted.
    times = [0, 1, 2, 5, 6, 7, 8, 9]
    return {'client_index': index, 'outcome': 'success', 'prompt_tokens': 100,
            'completion_tokens': 32, 'observed_tokens': 32, 'finish_reason': 'length',
            'finish_at': offset + 9, 'first_token_at': offset, 'last_token_at': offset + 9,
            'ttft_seconds': offset, 'elapsed_seconds': offset + 9, 'tpot_seconds': 9 / 28,
            'timeline': [{'at': offset + at, 'tokens': (i + 1) * 4, 'delta_tokens': 4,
                          'token_ids': [i * 4 + j for j in range(4)]} for i, at in enumerate(times)]}


def sample(at, indices=(0, 1), offset=0, scheduler_requests=None):
    tokens = analysis.count_at(trace(0, offset), at)
    return {'at': at, 'prefill': 20, 'decode': at * 4, 'prefill_raced': False,
            'progress': {'accepted_history': {f'client_{index}': 100 + tokens for index in indices},
                         'retiring_requests': 0},
            'scheduler': {'requests': len(indices) if scheduler_requests is None else scheduler_requests,
                          'reservations': [f'native-request-{index}' for index in indices],
                          'refresh_pending': 0, 'connector_pending': 0}}


def batch():
    return {'concurrency': 2, 'wall_seconds': 10, 'requests': [trace(0), trace(1)],
            'native_samples': [sample(at) for at in range(10)]}


def test_exact_sse_count_and_native_count_keep_zero_progress_intervals():
    value = analysis.analyze_batch(batch(), 4)
    assert value['http_service']['successful_output_tokens_per_second'] == 6.4
    assert value['decode_evidence_status'] == 'available'
    cohort = value['cohorts'][0]
    assert cohort['status'] == 'sufficient_evidence'
    assert cohort['stable_decode_owner_count'] == 2
    assert cohort['interval_count'] == 7 and cohort['seconds'] == 7
    assert cohort['zero_native_progress_intervals'] == 2
    assert cohort['zero_sse_progress_intervals'] == 2
    assert cohort['sse_output_tokens'] == 40
    assert cohort['native_accepted_input_tokens'] == 40
    assert cohort['sse_output_tokens_per_second'] == pytest.approx(40 / 7)
    assert len(cohort['segments']) == 1
    assert value['excluded_by_reason']['stream_terminal_boundary']['seconds'] == 1


def test_32_queued_clients_are_four_waves_of_eight_decode_owners():
    value = {'concurrency': 32, 'wall_seconds': 80,
             'requests': [trace(index, index // 8 * 20) for index in range(32)], 'native_samples': []}
    for at in range(80):
        wave = at // 20
        indices = range(wave * 8, (wave + 1) * 8) if at % 20 < 10 else ()
        # The engine retains all unfinished waiting requests in its total.
        value['native_samples'].append(sample(at, indices, wave * 20, 32 - wave * 8))
    report = analysis.analyze_batch(value, 4)
    assert report['http_service']['success'] == 32
    assert report['http_service']['successful_output_tokens_per_second'] == 1024 / 80
    assert report['full_requested_client_cohort']['status'] == 'not_applicable'
    assert [cohort['stable_decode_owner_count'] for cohort in report['cohorts']] == [8]
    cohort = report['cohorts'][0]
    assert len(cohort['segments']) == 4
    assert len(cohort['per_client']) == 32 and cohort['all_observed_owners_made_progress']
    assert any(interval['scheduler_requests_without_captured_owner'] == [24, 24, 24]
               for interval in cohort['intervals'])
    assert cohort['zero_sse_progress_intervals'] == 8
    assert 'owner_transition' in report['excluded_by_reason']


def test_paused_partial_prefill_is_not_inflated_into_decode_concurrency():
    value = batch()
    for item in value['native_samples']:
        item['progress']['accepted_history']['client_1'] = 99
    report = analysis.analyze_batch(value, 4)
    assert not report['cohorts']
    assert report['decode_evidence_status'] == 'insufficient_evidence'
    assert report['excluded_by_reason']['mixed_or_prefill']['interval_count'] == 8
    assert report['http_service']['status'] == 'passed'


def test_engine_waiting_requests_do_not_increase_decode_count_or_hide_prefill():
    traces = [trace(0), trace(1)]
    group = [sample(at, scheduler_requests=32) for at in (5, 6, 7)]
    interval, reason = analysis.classify_interval(group, traces, 4)
    assert reason is None and interval['stable_decode_owner_count'] == 2
    assert interval['scheduler_requests_without_captured_owner'] == [30, 30, 30]
    group[-1]['prefill'] += 1
    assert analysis.classify_interval(group, traces, 4)[1] == 'prefill_activity_or_trailing_commit'
    group[-1]['prefill'] -= 1
    traces.append(trace(2))
    for item in group:
        item['progress']['accepted_history']['client_2'] = 99
        item['scheduler']['reservations'].append('native-request-2')
    assert analysis.classify_interval(group, traces, 4)[1] == 'mixed_or_prefill'


def test_uncaptured_inflight_prefill_excluded_even_with_flat_prefill_counter():
    value = batch()
    # Both known decode streams remain open, while one already admitted long
    # prefill has not produced its first captured history. Every native counter
    # is stable, so the old counter-only guard incorrectly included this time.
    for item in value['native_samples']:
        item['scheduler']['requests'] = 32
        item['scheduler']['reservations'].append('uncaptured-prefill')
    report = analysis.analyze_batch(value, 4)
    assert not report['cohorts']
    assert report['excluded_by_reason']['admitted_owner_mismatch']['interval_count'] == 8
    assert report['excluded_by_reason']['admitted_owner_mismatch']['seconds'] == 8
    assert report['http_service']['status'] == 'passed'


def test_reservation_identity_change_excluded_even_when_count_is_unchanged():
    traces = [trace(0), trace(1)]
    group = [sample(at, scheduler_requests=32) for at in (5, 6, 7)]
    group[-1]['scheduler']['reservations'][0] = 'replacement-prefill'
    interval, reason = analysis.classify_interval(group, traces, 4)
    assert interval is None and reason == 'admission_reservation_transition'


@pytest.mark.parametrize('reservations,reason', [
    (None, 'missing_admission_reservations'),
    ('ambiguous', 'invalid_admission_reservations'),
    (['same', 'same'], 'invalid_admission_reservations'),
])
def test_missing_or_ambiguous_admission_evidence_cannot_prove_decode(reservations, reason):
    group = [sample(at) for at in (5, 6, 7)]
    group[1]['scheduler']['reservations'] = reservations
    assert analysis.classify_interval(group, [trace(0), trace(1)], 4)[1] == reason


@pytest.mark.parametrize('mutation,reason', [
    ('prefill', 'prefill_activity_or_trailing_commit'),
    ('raced', 'prefill_raced_read'),
    ('refresh', 'refresh_activity'),
    ('connector', 'connector_activity'),
    ('retiring', 'retiring_activity'),
    ('error', 'sample_read_error'),
    ('foreign', 'unknown_or_ambiguous_owner'),
    ('regressed', 'native_counter_or_timing_inconsistent'),
])
def test_contamination_is_excluded_and_reported_with_elapsed_time(mutation, reason):
    traces = [trace(0), trace(1)]
    group = [sample(at) for at in (5, 6, 7)]
    if mutation == 'prefill':
        group[-1]['prefill'] += 1
    elif mutation == 'raced':
        group[1]['prefill_raced'] = True
    elif mutation in ('refresh', 'connector'):
        group[1]['scheduler'][mutation + '_pending'] = 1
    elif mutation == 'retiring':
        group[1]['progress']['retiring_requests'] = 1
    elif mutation == 'error':
        group[1] = {'at': 6, 'error': 'JSONDecodeError'}
    elif mutation == 'foreign':
        for item in group:
            item['progress']['accepted_history'] = {'client_999': 200}
            item['scheduler']['reservations'] = ['foreign-native-owner']
    elif mutation == 'regressed':
        group[-1]['progress']['accepted_history']['client_0'] -= 8
    interval, exclusion = analysis.classify_interval(group, traces, 4)
    assert interval is None and exclusion == reason
    report = analysis.analyze_batch({'concurrency': 2, 'requests': traces, 'native_samples': group,
                                     'wall_seconds': 10}, 4)
    assert report['excluded_by_reason'][reason]['seconds'] == 1


def test_boundary_spanning_snapshot_and_terminal_tokens_are_excluded():
    traces = [trace(0), trace(1)]
    group = [sample(at) for at in (0, 1, 2)]
    group[1]['read_started_at'] = -.01
    assert analysis.classify_interval(group, traces, 4)[1] == 'stream_prefill_boundary'
    group = [sample(at) for at in (7, 8, 9)]
    assert analysis.classify_interval(group, traces, 4)[1] == 'stream_terminal_boundary'
    group = [sample(at) for at in (6, 7, 8)]
    assert analysis.classify_interval(group, traces, 8)[1] == 'stream_trim_boundary'


def test_failed_request_or_absent_token_ids_cannot_be_decode_evidence():
    value = batch()
    value['requests'][0]['outcome'] = 'failure'
    report = analysis.analyze_batch(value, 4)
    assert report['http_service']['status'] == 'failed'
    assert not report['cohorts']
    value = batch()
    del value['requests'][0]['timeline'][0]['token_ids']
    report = analysis.analyze_batch(value, 4)
    assert not report['cohorts']
    assert report['request_evidence_errors'][0]['error'] == 'missing_or_invalid_token_ids'


def test_evidence_threshold_never_removes_stalls_or_changes_throughput():
    a = analysis.analyze_batch(batch(), 4, min_seconds=2)
    b = analysis.analyze_batch(batch(), 4, min_seconds=1000)
    assert a['cohorts'][0]['status'] == 'sufficient_evidence'
    assert b['cohorts'][0]['status'] == 'insufficient_evidence'
    for field in ('seconds', 'interval_count', 'native_accepted_input_tokens', 'sse_output_tokens',
                  'sse_output_tokens_per_second', 'zero_sse_progress_intervals'):
        assert a['cohorts'][0][field] == b['cohorts'][0][field]


def test_owner_that_never_progresses_is_retained_but_insufficient_evidence():
    value = batch()
    for item in value['native_samples']:
        item['progress']['accepted_history']['client_1'] = 104
    report = analysis.analyze_batch(value, 4)
    cohort = report['cohorts'][0]
    assert cohort['seconds'] == 7 and cohort['interval_count'] == 7
    assert cohort['status'] == 'insufficient_evidence'
    assert not cohort['all_observed_owners_made_progress']


def test_cli_preserves_raw_source_and_records_sha(tmp_path):
    source, target = tmp_path / 'raw.json', tmp_path / 'analysis.json'
    raw = json.dumps({'configuration': {'trim_tokens': 4}, 'harness_sha256': 'fixture-hash',
                      'batches': [batch()]}).encode()
    source.write_bytes(raw)
    assert analysis.main(['--results', str(source), '--output', str(target)]) == 0
    report = json.loads(target.read_text())
    assert source.read_bytes() == raw
    assert report['source_results_sha256'] == hashlib.sha256(raw).hexdigest()
    assert report['source_harness_sha256'] == 'fixture-hash'
    with pytest.raises(FileExistsError):
        analysis.main(['--results', str(source), '--output', str(target)])
