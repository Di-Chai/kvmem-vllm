import pytest

from benchmark_no_probe_concurrency import client_peak, percentile, phase_summary


def response(start, seconds, count, ttft):
    return dict(started_at=start, seconds=seconds, output_ids=[1]*count, ttft_seconds=ttft)


def test_concurrent_throughput_uses_wall_span_including_between_wave_gaps():
    rows = [response(10, 4, 12, 1), response(10.1, 3, 8, 2),
            response(15, 2, 10, .5), response(15.1, 2.9, 10, 1.5)]
    result = phase_summary(rows, 2)
    assert result['wall_seconds'] == 8
    assert result['output_tokens_per_second'] == 5
    assert result['requests_per_second'] == .5
    assert result['observed_client_peak'] == 2
    assert result['ttft_p50_seconds'] == 1.25
    assert result['ttft_p95_seconds'] == pytest.approx(1.925)


def test_adjacent_requests_do_not_count_as_overlapping():
    assert client_peak([response(0, 1, 1, .1), response(1, 2, 1, .1)]) == 1
    assert client_peak([response(0, 3, 1, .1), response(1, 2, 1, .1),
                        response(2, 1, 1, .1)]) == 3


def test_percentiles_keep_single_and_extreme_samples():
    assert percentile([7], .95) == 7
    assert percentile([3, 1, 10], 0) == 1
    assert percentile([3, 1, 10], 1) == 10
