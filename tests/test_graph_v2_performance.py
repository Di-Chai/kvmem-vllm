"""CPU-only raw-K A/B protocol and measurement evidence tests."""
from copy import deepcopy
import fcntl
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    control = importlib.import_module('benchmark_graph_v2_performance')
    client = importlib.import_module('qualify_graph_v2_performance')
finally:
    sys.path.pop(0)


def base():
    return dict(model_format='safetensors', model='model', hf_config_path='model', tokenizer='tokenizer',
        execution_mode='compiled_graph', draft_method='mtp', draft_tokens=4,
        kvmem=dict(mean_k_backend='qw3_cuda', archive_backend='cpu', request_planning=True,
            async_archive_retirement=True, async_refresh=True, direct_pinned_reads=True, async_seal=True))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_raw_cache_is_one_variable_and_four_gib_is_total_not_per_arena():
    profiles = control.profiles(base(), 1234)
    assert profiles['raw4g']['kvmem']['gpu_raw_k_cache_bytes'] == 4*1024**3
    for profile in profiles.values():
        assert profile['kvmem']['graph_mode'] == 'multi'
        assert profile['kvmem']['pressure_policy'] == 'graph'
    configs = deepcopy(profiles)
    configs['raw4g']['kvmem']['cpu_gather_workers'] = 16
    with pytest.raises(ValueError, match='more than one'):
        control.validate_profiles(configs)
    for wrong in (2*1024**3, 8*1024**3):
        configs = deepcopy(profiles)
        configs['raw4g']['kvmem']['gpu_raw_k_cache_bytes'] = wrong
        with pytest.raises(ValueError): control.validate_profiles(configs)


def test_window_capacity_pair_is_separate_and_preserves_native_kv_budget():
    source = base()
    source['gpu_memory_utilization'] = .72
    candidate = 'raw-k-cache-window'
    profiles = control.profiles(source, 1234, candidate)
    assert set(profiles) == {'raw0', 'raw10g'}
    assert profiles['raw10g']['kvmem']['gpu_raw_k_cache_bytes'] == 10*1024**3
    assert control.CANDIDATES['raw-k-cache']['variants'] == {'raw0': 0, 'raw4g': 4*1024**3}
    for profile in profiles.values():
        assert profile['gpu_memory_utilization'] == .72
        assert profile['max_num_seqs'] == 8
        assert profile['kvmem']['select_budget'] == 36864
        client.validate_profile(profile, candidate)
    common = deepcopy(profiles['raw10g'])
    common['kvmem']['gpu_raw_k_cache_bytes'] = 0
    assert common == profiles['raw0']
    # Two separate arenas must each cover all eight immutable FP8 raw-K windows.
    required = 2 * 8 * 36864 * 17 * 4 * 256
    assert required == 10267656192
    assert 9*1024**3 < required <= profiles['raw10g']['kvmem']['gpu_raw_k_cache_bytes']
    with pytest.raises(ValueError, match='Unexpected performance profile'):
        client.validate_profile(profiles['raw10g'], 'raw-k-cache')
    with pytest.raises(ValueError, match='Unexpected performance profile'):
        client.validate_profile(control.profiles(source, 1234)['raw4g'], candidate)


@pytest.mark.parametrize('field', ['async_refresh', 'direct_pinned_reads'])
def test_budget_split_requires_the_actual_async_direct_assembly_contract(field):
    profile = base()
    profile['kvmem'][field] = False
    with pytest.raises(ValueError): control.profiles(profile, 1234)


@pytest.mark.parametrize('candidate', ['cpu-direct-record-seal', 'async-action-metadata'])
def test_each_new_pair_changes_only_its_declared_boolean(candidate):
    source = base()
    source['kvmem']['native_action_metadata'] = False
    source['kvmem'].update(gpu_raw_k_cache_bytes=10*1024**3,
                          cpu_direct_record_seal=True, async_action_metadata=True)
    profiles = control.profiles(source, 1234, candidate)
    control.validate_profiles(profiles, candidate)
    contract = control.CANDIDATES[candidate]
    for name, value in contract['variants'].items():
        kv = profiles[name]['kvmem']
        assert kv[contract['field']] is value
        assert kv['gpu_raw_k_cache_bytes'] == 0
        assert kv['native_action_metadata'] is False
        assert kv['graph_mode'] == 'multi' and kv['pressure_policy'] == 'graph'
    changed = next(reversed(profiles.values()))
    changed['kvmem']['native_action_metadata'] = True
    with pytest.raises(ValueError, match='more than one'):
        control.validate_profiles(profiles, candidate)


@pytest.mark.parametrize('candidate', list(control.CANDIDATES))
def test_real_launcher_transports_only_the_declared_candidate(tmp_path, candidate):
    from vllm_kvmem.serve import build_launch
    source = base()
    source['kvmem']['packed_assembly'] = True
    source['kvmem']['native_action_metadata'] = False
    contract = control.CANDIDATES[candidate]
    for name, profile in control.profiles(source, 1234, candidate).items():
        launch = build_launch(profile, tmp_path, tmp_path/name, check_paths=False)
        argv = launch['vllm_argv']
        worker = json.loads(argv[argv.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
        assert worker[contract['field']] == contract['variants'][name]
        assert worker['native_action_metadata'] is False


def test_sealing_candidate_requires_async_seal_and_keeps_cache_disabled():
    source = base()
    source['kvmem']['async_seal'] = False
    with pytest.raises(ValueError, match='async_seal=true'):
        control.profiles(source, 1234, 'cpu-direct-record-seal')
    profiles = control.profiles(base(), 1234, 'cpu-direct-record-seal')
    profiles['seal-direct']['kvmem']['gpu_raw_k_cache_bytes'] = 4*1024**3
    with pytest.raises(ValueError, match='raw-K disabled'):
        control.validate_profiles(profiles, 'cpu-direct-record-seal')


def test_successful_seal_counters_keep_missing_distinct_from_measured_zero():
    absent = client.seal_evidence({}, True)
    assert not absent['observed'] and not absent['measurement_complete']
    assert absent['seconds']['seal_record_copy_seconds'] is None
    archive = dict(seal_batches=2, seal_records=10, seal_payload_bytes=1000,
        seal_direct_records=10, seal_direct_bytes=1000, seal_pack_seconds=.2,
        seal_hash_seconds=.1, seal_cache_seconds=0., seal_record_copy_seconds=0.,
        seal_record_publish_seconds=.01)
    report = client.seal_evidence(archive, True)
    assert report['observed'] and report['measurement_complete'] and report['direct_path_observed']
    assert report['seconds']['seal_record_copy_seconds'] == 0.
    with pytest.raises(ValueError, match='Disabled direct sealing'):
        client.seal_evidence(archive, False)


def test_action_metadata_snapshots_are_cumulative_and_native_policy_is_preserved(tmp_path):
    capture = tmp_path/'capture'
    capture.mkdir()
    profile = control.profiles(base(), 1234, 'async-action-metadata')['metadata-async']
    save(capture/'action_config.json', dict(async_action_metadata=True, native_action_metadata=False))
    rows = [dict(metadata_upload_total=dict(submissions=n, completed=n-1, uploaded_bytes=n*100,
        packing_host_seconds=.1*n, copy_submit_host_seconds=.01*n)) for n in (2, 5)]
    (capture/'action_attention.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    report = client.metadata_evidence(capture, profile)
    assert report['async_path_observed'] and report['snapshots'] == 2
    assert report['latest_cumulative']['uploaded_bytes'] == 500
    save(capture/'action_config.json', dict(async_action_metadata=True, native_action_metadata=True))
    with pytest.raises(ValueError, match='native_action_metadata'):
        client.metadata_evidence(capture, profile)


def test_overlapping_host_service_time_is_union_and_never_labeled_gpu_idle():
    report = client.interval_evidence([dict(a=1., b=5.), dict(a=3., b=7.)], 'a', 'b')
    assert report['inclusive_sum_seconds'] == 8 and report['union_seconds'] == 6
    assert 'neither sum nor union is GPU idle' in report['scope']
    with pytest.raises(ValueError): client.interval_evidence([dict(a=2., b=1.)], 'a', 'b')


def metrics_fixture(tmp_path, hits=0):
    output, capture = tmp_path/'qualification', tmp_path/'capture'
    rows = []
    for index in range(8):
        name = str(index)
        rows.append(dict(id=name, status='completed', input_sha256=name, seed=1000, completion_tokens=2,
            input_tokens=262144, evaluation=dict(normalized_coverage=1), ttft_seconds=10, seconds=20))
        save(output/'cases'/name/'response.json', dict(output_ids=[10, 20]))
        save(output/'cases'/name/'evidence.json', dict(query=dict(selected_history_ranges=[[0, 32]]),
            pressure=[dict(selected_history_ranges=[[32, 32]])]))
    save(output/'results.json', dict(status='passed', results=rows, wall_seconds=100,
        measurement_started_at=20., measurement_finished_at=120.,
        actual_concurrency=dict(executed_request_peak=7),
        step_summary={}, macro_coverage=1., exact=8))
    capture.mkdir()
    refresh = dict(assembly_uploaded_bytes=4096, assembly_host_prepare_seconds=2,
        raw_cache_hit_tokens=hits, refresh_started_at=1, refresh_finished_at=5)
    (capture/'bounded_refresh.jsonl').write_text(json.dumps(refresh)+'\n')
    return capture, output


def test_async_telemetry_reports_missing_gather_without_inventing_an_isolated_time(tmp_path):
    capture, output = metrics_fixture(tmp_path, hits=128)
    report = client.performance(capture, output, control.profiles(base(), 1234)['raw4g'])
    assert report['raw_cache_hit_tokens'] == 128 and report['raw_cache_exercised']
    assert report['configured_per_arena_cap_bytes'] == 2*1024**3
    assert report['refresh']['h2d_payload_bytes']['value'] == 4096
    assert report['restore']['events'] == 0
    assert report['cpu_gather']['observed'] is False
    assert report['actual_allocation_observed'] is False
    assert report['accepted_output_tokens_per_wall_second'] == .16
    assert report['measurement_started_at'] == 20. and report['measurement_finished_at'] == 120.
    assert report['gpu_idle_seconds'] is None and report['gpu_idle_inferred'] is False


def test_repeated_cache_capacity_gauge_is_never_summed_as_allocated_memory():
    capacity = 2*1024**3
    rows = [dict(raw_cache_capacity_bytes=capacity) for _ in range(100)]
    report = client.cache_capacity_evidence(rows)
    assert report['observed_max_bytes'] == capacity and report['observed_rows'] == 100
    assert report['per_arena_max_bytes'] is report['total_allocated_bytes'] is None
    rows[0]['raw_cache_arena'] = 'refresh'
    assert client.cache_capacity_evidence(rows)['per_arena_max_bytes'] is None
    for row in rows:
        row['raw_cache_arena'] = 'refresh'
    rows.append(dict(raw_cache_arena='main', raw_cache_capacity_bytes=1024))
    assert client.cache_capacity_evidence(rows)['per_arena_max_bytes'] == dict(refresh=capacity, main=1024)
    assert client.cache_capacity_evidence([])['observed_max_bytes'] is None


@pytest.mark.parametrize('damage', ['failed', 'incomplete_case', 'output_count', 'missing_clock', 'wrong_clock', 'duplicate_case'])
def test_missing_failed_or_mismatched_performance_evidence_cannot_generate_rates(tmp_path, damage):
    capture, output = metrics_fixture(tmp_path)
    data = json.loads((output/'results.json').read_text())
    if damage == 'failed': data['status'] = 'failed'
    elif damage == 'incomplete_case': data['results'][0]['status'] = 'failed'
    elif damage == 'output_count': data['results'][0]['completion_tokens'] = 10
    elif damage == 'missing_clock': data.pop('measurement_started_at')
    elif damage == 'wrong_clock': data['measurement_finished_at'] += 10
    else: data['results'].append(dict(data['results'][0]))
    save(output/'results.json', data)
    with pytest.raises(ValueError):
        client.performance(capture, output, control.profiles(base(), 1234)['raw0'])


def test_disabled_cache_cannot_report_hits_and_required_metrics_cannot_be_absent(tmp_path):
    capture, output = metrics_fixture(tmp_path, hits=1)
    with pytest.raises(ValueError, match='Disabled'):
        client.performance(capture, output, control.profiles(base(), 1234)['raw0'])
    row = json.loads((capture/'bounded_refresh.jsonl').read_text())
    row.pop('assembly_uploaded_bytes')
    (capture/'bounded_refresh.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='Missing actual'):
        client.performance(capture, output, control.profiles(base(), 1234)['raw4g'])


def test_comparison_retains_output_drift_and_rejects_changed_input_or_seed(tmp_path):
    capture, output = metrics_fixture(tmp_path)
    metrics = client.performance(capture, output, control.profiles(base(), 1234)['raw0'])
    root = tmp_path/'comparison'
    for name in control.VARIANTS:
        save(root/name/'qualification/performance.json', metrics)
        save(root/name/'nvml-summary.json', dict(samples=100))
    report = control.comparison(root)
    assert report['selection_sequences_identical'] and report['accepted_outputs_identical']
    assert report['candidate_screen_status'] == 'inconclusive'
    assert 'candidate_path_observed' in report['inconclusive_reasons']
    assert 'native_eight_owner_batches_observed' in report['inconclusive_reasons']
    assert report['gpu_idle_inferred'] is False
    metrics['cases']['0']['output_ids_sha256'] = 'different accepted output'
    save(root/'raw4g/qualification/performance.json', metrics)
    assert control.comparison(root)['accepted_outputs_identical'] is False
    metrics['cases']['0']['seed'] = 2000
    save(root/'raw4g/qualification/performance.json', metrics)
    with pytest.raises(ValueError, match='input or sampling seed'):
        control.comparison(root)


@pytest.mark.parametrize('drift', ['none', 'output_eos', 'selection', 'unexercised'])
@pytest.mark.parametrize('candidate_name', ['raw-k-cache', 'raw-k-cache-window'])
def test_raw_wall_ratio_is_not_a_qualified_speedup_when_work_or_candidate_differs(tmp_path, drift, candidate_name):
    capture, output = metrics_fixture(tmp_path)
    profiles = control.profiles(base(), 1234, candidate_name)
    baseline = client.performance(capture, output, profiles['raw0'], candidate_name)
    baseline['actual_concurrency']['executed_request_peak'] = 8
    candidate = deepcopy(baseline)
    candidate['raw_cache_exercised'] = drift != 'unexercised'
    candidate['wall_seconds'] = 50
    candidate['useful_input_tokens_per_wall_second'] *= 2
    candidate['accepted_output_tokens_per_wall_second'] *= 2
    if drift == 'output_eos':
        for row in candidate['cases'].values():
            row['output_tokens'] = 1
            row['output_ids_sha256'] = 'earlier EOS'
        candidate['accepted_output_tokens_per_wall_second'] /= 2
    elif drift == 'selection':
        candidate['cases']['0']['selection_sha256'] = 'different pages'
    changed_name = next(reversed(profiles))
    for name, metrics in (('raw0', baseline), (changed_name, candidate)):
        save(tmp_path/name/'qualification/performance.json', metrics)
        save(tmp_path/name/'nvml-summary.json', dict(samples=100))
    report = control.comparison(tmp_path, candidate_name)
    assert report['raw_wall_ratio_baseline_over_candidate'] == 2
    assert report['useful_input_rate_ratio_candidate_over_baseline'] == 2
    assert 'wall_speed_ratio_baseline_over_candidate' not in report
    assert not report['performance_gain_established']
    assert report['candidate_screen_status'] == ('comparable_cold_smoke' if drift == 'none' else 'inconclusive')
    if drift == 'output_eos':
        assert report['accepted_output_rate_ratio_candidate_over_baseline'] == 1
        assert 'identical_accepted_outputs' in report['inconclusive_reasons']


def test_exercised_direct_seal_with_incomplete_counters_is_still_inconclusive(tmp_path):
    capture, output = metrics_fixture(tmp_path)
    candidate_name = 'cpu-direct-record-seal'
    profiles = control.profiles(base(), 1234, candidate_name)
    for name, profile in profiles.items():
        metrics = client.performance(capture, output, profile, candidate_name)
        metrics['actual_concurrency']['executed_request_peak'] = 8
        metrics['seal'] = client.seal_evidence(
            dict(seal_direct_records=10, seal_direct_bytes=1000) if name == 'seal-direct' else {},
            name == 'seal-direct')
        save(tmp_path/name/'qualification/performance.json', metrics)
        save(tmp_path/name/'nvml-summary.json', dict(samples=100))
    report = control.comparison(tmp_path, candidate_name)
    assert report['candidate_path_observed']
    assert report['candidate_screen_status'] == 'inconclusive'
    assert report['inconclusive_reasons'] == ['complete_seal_measurements']
    assert not report['performance_gain_established']


def frozen_campaign(tmp_path, candidate='raw-k-cache'):
    root = tmp_path/'campaign'
    root.mkdir()
    save(root/'status.json', dict(status='prepared'))
    (root/'source.diff').write_text('frozen patch')
    (root/'sample.i32').write_bytes(b'tokens')
    save(root/'fixture.json', dict(schema=control.protocol.SCHEMA))
    (root/'frozen').mkdir()
    (root/'frozen/benchmark_graph_v2_performance.py').write_text('# controller')
    jobs = []
    for name, profile in control.profiles(base(), 1234, candidate).items():
        target = root/name
        save(target/'profile.json', profile)
        save(target/'status.json', dict(status='prepared'))
        (target/'fixture.json').write_bytes((root/'fixture.json').read_bytes())
        native = target/'runtime/src/vllm_kvmem/_native.test.so'
        native.parent.mkdir(parents=True)
        native.write_bytes(b'current native')
        hashes = {'src/vllm_kvmem/_native.test.so':control.sha(native)}
        save(target/'manifest.json', dict(runtime_sha256=hashes, profile_sha256=control.sha(target/'profile.json'),
            fixture_sha256=control.sha(target/'fixture.json')))
        jobs.append(dict(id=name, manifest_sha256=control.sha(target/'manifest.json')))
    save(root/'manifest.json', dict(schema=control.SCHEMA, candidate=candidate,
        variants=control.CANDIDATES[candidate]['variants'], jobs=jobs,
        source_diff_sha256=control.sha(root/'source.diff'), fixture_sha256=control.sha(root/'fixture.json'),
        frozen_sha256={'benchmark_graph_v2_performance.py':control.sha(root/'frozen/benchmark_graph_v2_performance.py')},
        input_sha256={'sample.i32':control.sha(root/'sample.i32')}, runtime_sha256=hashes,
        model_identity=dict(config_sha256={}, tokenizer_sha256={}, weight_stat_identity={})))
    return root


@pytest.mark.parametrize('target', ['sample.i32', 'raw4g/profile.json',
    'raw0/runtime/src/vllm_kvmem/_native.test.so', 'frozen/benchmark_graph_v2_performance.py'])
def test_frozen_runtime_samples_and_profiles_cannot_change(tmp_path, target):
    root = frozen_campaign(tmp_path)
    control.validate(root)
    with (root/target).open('ab') as stream: stream.write(b' ')
    with pytest.raises(ValueError): control.validate(root)


@pytest.mark.parametrize('status', ['failed', 'running', 'completed'])
def test_failed_interrupted_completed_attempts_cannot_be_retried(tmp_path, monkeypatch, status):
    root = frozen_campaign(tmp_path)
    save(root/'status.json', dict(status=status))
    monkeypatch.setattr(control, '__file__', str(root/'frozen/benchmark_graph_v2_performance.py'))
    with pytest.raises(ValueError, match='cannot be retried'):
        control.run(SimpleNamespace(output=root, variants=['raw0']))


def test_performance_campaign_cannot_be_started_by_two_controllers(tmp_path):
    root = frozen_campaign(tmp_path)
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            control.run(SimpleNamespace(output=root, variants=['raw0']))
    assert json.loads((root/'status.json').read_text())['status'] == 'prepared'


@pytest.mark.parametrize('candidate', list(control.CANDIDATES))
def test_frozen_validation_binds_the_candidate_and_exact_pair(tmp_path, candidate):
    root = frozen_campaign(tmp_path, candidate)
    assert control.validate(root)['candidate'] == candidate
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['candidate'] = next(name for name in control.CANDIDATES if name != candidate)
    save(root/'manifest.json', manifest)
    with pytest.raises(ValueError, match='Unknown performance contract'):
        control.validate(root)
