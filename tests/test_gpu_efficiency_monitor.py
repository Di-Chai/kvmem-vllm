import subprocess
from types import SimpleNamespace

import pytest

from scripts.gpu_efficiency_monitor import number, snapshot, summarize


def fake_runner(devices, processes):
    def run(argv, **kwargs):
        assert kwargs['timeout'] == 5
        text = processes if '--query-compute-apps=' in argv[1] else devices
        if isinstance(text, Exception):
            raise text
        return SimpleNamespace(stdout=text)
    return run


def test_device_utilization_does_not_claim_process_attribution():
    row = snapshot({10}, fake_runner(
        '0, GPU-A, 83, 12, 70000, 96000, 230.5\n1, GPU-B, 0, 0, 5, 24000, [N/A]\n',
        'GPU-A, 10, 69000\nGPU-A, 20, 1000\n'))
    assert row['utilization_scope'] == 'whole-device'
    assert row['external_pids'] == [20]
    assert row['devices'][0]['gpu_utilization_percent'] == 83
    assert row['devices'][0]['owned_memory_mib'] == 69000
    assert row['devices'][1]['power_watts'] is None
    assert row['devices'][1]['gpu_utilization_percent'] == 0
    assert row['errors'] == {}


def test_failures_and_unsupported_values_are_not_zero():
    row = snapshot({10}, fake_runner(
        '0, GPU-A, [N/A], [Not Supported], 70000, 96000, N/A\n',
        subprocess.TimeoutExpired(['nvidia-smi'], 5)))
    assert row['devices'][0]['gpu_utilization_percent'] is None
    assert row['devices'][0]['owned_memory_mib'] is None
    assert row['external_pids'] is None
    assert 'TimeoutExpired' in row['errors']['processes']
    failed = snapshot({10}, fake_runner('malformed, output', ''))
    assert failed['devices'] == []
    assert 'devices' in failed['errors']


def sample(at, value, errors=None, external=()):
    devices = [] if value is None else [dict(uuid='GPU-A', index=0,
        gpu_utilization_percent=value, memory_utilization_percent=None)]
    return dict(at=at, devices=devices, errors=errors or {}, external_pids=list(external))


def test_summary_clips_exact_interval_and_weights_irregular_samples():
    result = summarize([sample(0, 20), sample(1, 80), sample(3, 50)], .5, 4)
    metric = result['devices']['GPU-A']['metrics']['gpu_utilization_percent']
    assert metric['coverage_seconds'] == 3.5
    assert metric['coverage_fraction'] == 1
    assert metric['mean'] == pytest.approx((.5*20 + 2*80 + 50)/3.5)
    assert 'memory_utilization_percent' not in result['devices']['GPU-A']['metrics']


def test_summary_retains_failed_gaps_and_caps_stale_sample_hold():
    rows = [sample(0, 100), sample(1, None, {'devices': 'failed'}), sample(5, 0, external=[20])]
    result = summarize(rows, 0, 10, max_hold_seconds=2)
    metric = result['devices']['GPU-A']['metrics']['gpu_utilization_percent']
    assert metric['coverage_seconds'] == 3
    assert metric['coverage_fraction'] == .3
    assert metric['mean'] == pytest.approx(100/3)
    assert result['errors'] == [dict(at=1, errors={'devices': 'failed'})]
    assert result['external_pids'] == [20]


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf'])
def test_nonfinite_metric_is_rejected(value):
    with pytest.raises(ValueError, match='Non-finite'):
        number(value)
