"""Resource telemetry correctness; no subprocess, inference or GPU access."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_prefill_publication_pipeline as pipeline


@pytest.mark.parametrize('resources', [[], [{'at': 1, 'gpu': '100, 200, 0'}],
    [{'processes': [], 'rss_sum_bytes': 0}],
    [{'processes': [{'pid': 10, 'VmHWM_bytes': 8192}], 'rss_sum_bytes': 0}]])
def test_missing_or_unreadable_rss_is_unknown_not_zero(resources):
    report = pipeline.sampled_process_memory(resources)
    assert report['max_sampled_process_tree_rss_bytes'] is None
    assert report['process_tree_rss_samples'] == 0
    assert report['process_tree_rss_missing_samples'] == len(resources)


def test_rss_peak_uses_simultaneous_sum_not_sum_of_individual_high_watermarks():
    resources = [dict(at=1, processes=[dict(pid=10, VmRSS_bytes=1024, VmHWM_bytes=8192),
        dict(pid=20, VmRSS_bytes=2048, VmHWM_bytes=16384)], rss_sum_bytes=3072),
        dict(at=2, processes=[dict(pid=10, VmRSS_bytes=4096, VmHWM_bytes=8192)], rss_sum_bytes=4096),
        dict(at=3, gpu='100, 200, 0')]
    report = pipeline.sampled_process_memory(resources)
    assert report['max_sampled_process_tree_rss_bytes'] == 4096
    assert report['process_tree_rss_samples'] == 2
    assert report['process_tree_rss_missing_samples'] == 1
    assert 'shared pages' in report['process_tree_memory_scope']
    assert 'PSS' in report['process_tree_memory_scope']


def test_process_tree_keeps_pid_details_and_skips_exited_or_inaccessible_children(tmp_path, monkeypatch):
    def process(pid, status, children=''):
        base = tmp_path/'proc'/str(pid)
        (base/'task'/str(pid)).mkdir(parents=True)
        (base/'status').write_text(status)
        (base/'task'/str(pid)/'children').write_text(children)
    process(10, 'Name:\tserver\nVmRSS:\t2 kB\nVmHWM:\t9 kB\n', '20 20 30 40 10')
    process(20, 'VmRSS:\t3 kB\nVmHWM:\t8 kB\n')
    process(40, 'VmRSS:\t7 kB\n')
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path == tmp_path/'proc/40/status':
            raise PermissionError('unreadable child')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(pipeline, 'Path', lambda path: tmp_path/str(path).lstrip('/'))
    observed = pipeline.owned_process_memory(10)
    assert observed == dict(processes=[dict(pid=10, VmRSS_bytes=2048, VmHWM_bytes=9216),
        dict(pid=20, VmRSS_bytes=3072, VmHWM_bytes=8192)], rss_sum_bytes=5120)
    assert pipeline.sampled_process_memory([observed])['max_sampled_process_tree_rss_bytes'] == 5120
