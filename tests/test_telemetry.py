import json
from types import SimpleNamespace as NS

from vllm_kvmem import telemetry


def test_gpu_sample_separates_foreign_work_before_aggregating_memory(monkeypatch):
    monkeypatch.setattr(telemetry.subprocess, 'run', lambda *a, **kw: NS(stdout='11, 2048\n22, 4096\n'))
    monkeypatch.setattr(telemetry.os, 'getpgid', lambda pid: 10 if pid == 11 else 20)
    monkeypatch.setattr(telemetry.Path, 'read_text', lambda path: 'VmRSS: 100 kB\nVmHWM: 200 kB\n')
    sample = telemetry.gpu_process_sample(10)
    assert sample['owned'] == [{'pid': 11, 'gpu_memory_mib': 2048}]
    assert sample['foreign'] == [{'pid': 22, 'gpu_memory_mib': 4096}]
    monitor = telemetry.ResourceMonitor('/unused', 10)
    monitor._observe(sample)
    assert monitor.max_gpu_mib == 2048 and monitor.foreign_pids == {22}
    assert monitor.max_sum_rss_kib == 200


def test_sampling_failure_is_preserved_and_does_not_report_a_zero_memory_pass(tmp_path, monkeypatch):
    def failing(group):
        raise TimeoutError('injected NVML timeout')
    monkeypatch.setattr(telemetry, 'gpu_process_sample', failing)
    monitor = telemetry.ResourceMonitor(tmp_path / 'resources.jsonl', 10)
    # One iteration deterministically, with the same thread lifecycle as serving.
    original = monitor.stop_event.wait
    def finish_after_sample(interval):
        monitor.stop_event.set()
        return original(0)
    monkeypatch.setattr(monitor.stop_event, 'wait', finish_after_sample)
    monitor.start()
    monitor.thread.join(5)
    result = monitor.stop()
    assert result['sampling_errors'] == 1 and not result['exclusive_gpu_observed']
    assert result['sampled_max_owned_gpu_memory_mib'] is None
    assert 'NVML timeout' in json.loads(monitor.path.read_text())['error']
