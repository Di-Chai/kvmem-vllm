import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location('block_graph_host_benchmark',
    Path(__file__).parents[1] / 'scripts' / 'benchmark_block_graph_host.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_cpu_probe_builds_real_rows_and_releases_all_ownership():
    report = benchmark.run(history_tokens=256, block_tokens=16, repeats=2)
    assert report['status'] == 'passed'
    assert report['global_bytes_after_release'] == 0
    first = report['memory_phases'][0]
    assert len(first['sessions']) == 2
    assert all(session['observed_sources'] == 16 for session in first['sessions'])
    assert all(session['prefill_samples'] == 32 for session in first['sessions'])
    assert set(report['measurements']) >= {'score_multi', 'score_attention', 'score_locality',
        'accepted_sample_aggregation_and_topk', 'restore_M_with_exact_P_preservation'}
    assert report['gpu_used'] is False and report['quality_claim'] is False


def test_insufficient_budget_is_reported_and_released_without_changing_limits():
    report = benchmark.run(history_tokens=256, block_tokens=16, repeats=2,
                           session_bytes=2000, total_bytes=4000)
    assert report['status'] == 'capacity_rejected'
    assert report['config']['max_bytes_per_session'] == 2000
    assert report['global_bytes_after_release'] == 0
