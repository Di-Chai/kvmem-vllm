import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('host_calibrated_fp8_test',
    ROOT/'scripts/benchmark_aligned_fp8_agents.py')
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)
host = bench.host
G = host.GIB


def measurements():
    budget = dict(history_limits=[262144]*8, archive_bytes=68*G,
        mean_k_index_bytes=6*G, extra_bytes=32*G, minimum_free_bytes=8*G,
        required_available_bytes=106*G, scope='original')
    admission = dict(passed=True, samples=[dict(at=10000+2*i,
        available_bytes=106*G+G//8, consecutive_qualifying_samples=i+1) for i in range(3)])
    rows = [dict(at=float(i), wall_at=10005.+i,
        memory=dict(MemTotal=128*G, MemAvailable=(20 if i==500 else 50)*G)) for i in range(601)]
    timing = dict(started_wall=10005., finished_wall=10610.)
    return budget, admission, rows, timing


def test_lower_only_extra_allowance_preserves_complete_history_and_live_guard():
    inputs = measurements()
    report = host.derive(*inputs)
    original = copy.deepcopy(inputs[0])
    budget = host.calibrated_budget(original, report)
    assert report['reduction_bytes'] == 4*G
    assert budget['extra_bytes'] == 28*G and budget['required_available_bytes'] == 102*G
    for key in ('history_limits','archive_bytes','mean_k_index_bytes','minimum_free_bytes'):
        assert budget[key] == original[key]
    assert inputs[0] == original
    assert report['projected_available_at_new_threshold_bytes'] >= 12*G
    assert report['observations']['coverage_seconds'] == 600


@pytest.mark.parametrize('failure', ['short','gap','order','lifetime','delayed_start',
    'guard','invalid_memory','unstable','changed_guard','changed_extra','formula','no_spare'])
def test_incomplete_or_unsafe_measurements_cannot_lower_admission(failure):
    budget, admission, rows, timing = measurements()
    if failure == 'short': rows = rows[:200]
    if failure == 'gap': rows[100]['at'] += 10
    if failure == 'order': rows[100]['at'] = rows[99]['at']
    if failure == 'lifetime': timing['started_wall'] -= 20
    if failure == 'delayed_start': admission['samples'][-1]['at'] -= 100
    if failure == 'guard': rows[500]['memory']['MemAvailable'] = 7*G
    if failure == 'invalid_memory': rows[500]['memory']['MemAvailable'] = True
    if failure == 'unstable': admission['samples'][-1]['consecutive_qualifying_samples'] = 0
    if failure == 'changed_guard': budget['minimum_free_bytes'] = G
    if failure == 'changed_extra': budget.update(extra_bytes=31*G, required_available_bytes=105*G)
    if failure == 'formula': budget['required_available_bytes'] -= 1
    if failure == 'no_spare': rows[500]['memory']['MemAvailable'] = 12*G
    with pytest.raises(ValueError):
        host.derive(budget, admission, rows, timing)


def test_no_more_than_whole_gib_reduction_justified_by_measured_peak():
    budget, admission, rows, timing = measurements()
    rows[500]['memory']['MemAvailable'] = 14*G + G//2
    report = host.derive(budget, admission, rows, timing)
    assert report['reduction_bytes'] == 2*G
    assert report['extra_bytes'] == 30*G


@pytest.mark.parametrize('failure', [None,'hash','observations','profile','workload','full_history'])
def test_host_contract_binds_completed_source_and_exact_inference_protocol(tmp_path, monkeypatch, failure):
    budget, admission, rows, timing = measurements()
    report = host.derive(budget, admission, rows, timing)
    report['source_run'] = str(tmp_path/'source')
    source_profile = dict(port=18608, model='/fp8', kvmem={'select_budget':65536})
    (tmp_path/'source').mkdir()
    bench.cap.save(tmp_path/'source'/f'{bench.ARM}.json', source_profile)
    path = tmp_path/'calibration.json'; bench.cap.save(path, report)
    binding = dict(path=str(path),sha256=bench.cap.sha(path))
    workload = dict(concurrency=8,duration_seconds=300.)
    original = dict(budget=budget,fixture_sha256='fixture',device_memory_identity='device')
    source = dict(original,workload=copy.deepcopy(workload),port=18608)
    monkeypatch.setattr(bench,'baseline_proof',lambda _: (source,{'bound':'proof'}))
    monkeypatch.setattr(host,'source_report',lambda *_:copy.deepcopy(report))
    profile = copy.deepcopy(source_profile); profile['port'] = 18616
    if failure == 'hash': binding['sha256'] = 'bad'
    if failure == 'observations':
        changed = dict(report, extra_bytes=27*G); bench.cap.save(path, changed)
        binding['sha256'] = bench.cap.sha(path)
    if failure == 'profile': profile['model'] = '/another-model'
    if failure == 'workload': workload['concurrency'] = 4
    if failure == 'full_history': original = dict(original,budget=dict(budget,history_limits=[131072]*8))
    if failure is None:
        assert bench.host_contract(original,profile,workload,binding)['extra_bytes'] == 28*G
    else:
        with pytest.raises(ValueError):bench.host_contract(original,profile,workload,binding)


def test_without_calibration_legacy_launch_rule_remains_exact():
    original = {'budget':measurements()[0]}
    result = bench.host_contract(original,{}, {},None)
    assert result == original['budget'] and result is not original['budget']


def test_external_qualified_runtime_cannot_be_rerun_as_a_fresh_qualification(tmp_path, monkeypatch):
    m = dict(diagnostic_steps=0,fp8_runtime_qualification={'path':'/original/qualification'})
    monkeypatch.setattr(bench,'verify',lambda *_,**__:m)
    with pytest.raises(ValueError,match='reuses a completed exact-runtime qualification'):
        asyncio.run(bench.run_qualification(type('Args',(),{'output':tmp_path})()))


def test_measured_envelope_preserves_live_guard_and_actual_memory_values():
    original, admission, rows, timing = measurements()
    budget = host.calibrated_budget(original, host.derive(original,admission,rows,timing))
    result = host.observed_envelope(budget,admission,rows)
    assert result['live_guard_bytes'] == 8*G
    assert result['sampled_minimum_available_bytes'] == 20*G
    assert result['sampled_minimum_above_guard_bytes'] == 12*G
    rows[500]['memory']['MemAvailable'] = 7*G
    with pytest.raises(ValueError,match='unchanged live guard'):
        host.observed_envelope(budget,admission,rows)
