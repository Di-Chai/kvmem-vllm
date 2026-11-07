"""Evidence validation needs no torch/vLLM, HTTP server or GPU imports."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_dynamic_memory import (
    PhaseSampler, RequestFailure, analyze, analyze_directory, arm_profile,
    client_peak, input_digest, request, resource_sample, save, semantic_profile, summarize,
)


def response(owner=0, *, start=10, seconds=5, tokens=None):
    tokens = [owner+1, owner+2] if tokens is None else tokens
    return dict(status='completed', started=start, seconds=seconds, ttft_seconds=1,
        input_digest=input_digest({'prompt': [1000+owner]*128}), output_ids=tokens,
        usage=dict(prompt_tokens=128, completion_tokens=len(tokens)))


def campaign(backends=('cpu', 'tiered', 'tiered', 'cpu')):
    config = dict(port=1234, kv_cache_memory_bytes=1024,
                  kvmem=dict(select_budget=2048, gen_budget=2048, pressure_policy='attention'))
    arms = []
    for backend in backends:
        profile = arm_profile(config, backend, disk_bytes=4096, checkpoint_bytes=2048, port=1234)
        phases = []
        for count, warmup in ((1, True), (2, False)):
            responses = [response(owner, start=10+owner) for owner in range(count)]
            phases.append(dict(concurrency=count, warmup=warmup, status='completed', responses=responses,
                summary=summarize(responses), resources=dict(failure=None,
                    observed_phase_peaks=dict(simultaneous_gpu_owners=2, gpu_owned_pages=20,
                                              cpu_resident_bytes=4096))))
        arms.append(dict(backend=backend, status='completed', phases=phases, profile=profile))
    return arms, dict(arms=list(backends), levels=[2], prompt_tokens=128, output_tokens=2)


def test_public_throughput_uses_end_to_end_wall_span_and_real_tokens():
    rows = [response(start=10, seconds=4), response(1, start=12, seconds=5)]
    result = summarize(rows)
    assert result['wall_seconds'] == 7
    assert result['output_tokens'] == 4
    assert result['output_tokens_per_second'] == pytest.approx(4/7)
    assert result['requests_per_second'] == pytest.approx(2/7)
    assert result['observed_client_peak'] == 2
    assert result['latency_p95_seconds'] == pytest.approx(4.95)


def test_adjacent_client_requests_are_not_simultaneous_gpu_owners():
    assert client_peak([response(start=0, seconds=1), response(start=1, seconds=1)]) == 1
    row = resource_sample(dict(gpu_capacity=dict(requests={'a':dict(owned_pages=3),
        'b':dict(owned_pages=0), 'c':dict(owned_pages=2)}, request_owned_pages=5)),
        dict(cpu_authority_bytes=8192), 0)
    assert row['simultaneous_gpu_owners'] == 2
    assert row['gpu_owned_pages'] == 5
    assert row['cpu_resident_bytes'] is None  # Charged slots are not payload.
    assert row['cpu_authority_charged_bytes'] == 8192


def test_sampler_retains_bounded_series_without_losing_peak_between_kept_samples(tmp_path):
    sampler = PhaseSampler(tmp_path, max_samples=8, clock=lambda: 0)
    for index in range(1000):
        amount = 99999 if index == 777 else index
        sampler.observe(dict(gpu_capacity=dict(request_owned_pages=amount,
            requests={'a':dict(owned_pages=1)})), dict(cpu_resident_bytes=amount,
                cpu_peak_bytes=200000, disk_write_bytes=index*10),  )
    result = sampler.snapshot()
    assert result['total_samples'] == 1000
    assert result['retained_samples'] <= 8
    assert result['stride'] > 1
    assert result['observed_phase_peaks']['cpu_resident_bytes'] == 99999
    assert result['observed_phase_peaks']['cpu_reported_process_peak_bytes'] == 200000
    assert result['disk_counter_deltas']['disk_write_bytes'] == 9990
    assert not any(row['cpu_resident_bytes'] == 99999 for row in result['samples'])


def test_sampler_reports_missing_sources_and_collects_exact_payload_and_swap_counters(tmp_path):
    sampler = PhaseSampler(tmp_path, clock=lambda: 4)
    sampler.sample()
    assert sampler.snapshot()['read_errors'] == dict(scheduler=1, archive=1)
    save(tmp_path/'scheduler_resources.json', dict(at=4, gpu_capacity=dict(
        request_owned_pages=3, requests={'a':dict(owned_pages=3)}),
        request_swaps=dict(pauses=2, restores=1, restore_deferrals=0,
                           requests={'b':dict(phase='paused')})))
    save(tmp_path/'archive.json', dict(cpu_resident_bytes=100, cpu_authority_bytes=200,
                                    cpu_peak_bytes=150))
    sampler.sample()
    result = sampler.snapshot()
    assert result['source_updates'] == dict(scheduler=1, archive=1)
    assert result['observed_phase_peaks']['cpu_resident_bytes'] == 100
    assert result['observed_phase_peaks']['paused_requests'] == 1
    assert result['observed_phase_peaks']['swap_pauses'] == 2


def test_host_ram_metrics_remain_separate_from_archive_payload_and_lifetime_peak(tmp_path):
    sampler = PhaseSampler(tmp_path, clock=lambda: 0)
    for rss, cgroup, payload in ((1000, 2000, 100), (1500, 2500, 50), (1200, 2300, 25)):
        sampler.observe({}, dict(cpu_resident_bytes=payload, cpu_peak_bytes=500,
            process_memory=dict(worker=dict(rss_bytes=rss, lifetime_peak_rss_bytes=3000,
                locked_bytes=10), cgroup=dict(current_bytes=cgroup, file_bytes=800))))
    peaks = sampler.snapshot()['observed_phase_peaks']
    assert peaks['cpu_resident_bytes'] == 100
    assert peaks['cpu_reported_process_peak_bytes'] == 500
    assert peaks['worker_rss_bytes'] == 1500
    assert peaks['worker_lifetime_peak_rss_bytes'] == 3000
    assert peaks['cgroup_current_bytes'] == 2500
    assert peaks['cgroup_file_bytes'] == 800
    assert peaks['worker_locked_bytes'] == 10
    assert 'cgroup_max_bytes' not in peaks  # Unlimited/unavailable is not zero.
    assert 'must not be added' in sampler.snapshot()['scope']


def test_sampler_prefers_live_compact_snapshot_over_stale_retirement_report(tmp_path):
    save(tmp_path/'scheduler_resources.json', {})
    save(tmp_path/'archive.json', dict(cpu_resident_bytes=0))
    save(tmp_path/'archive_resources.json', dict(at=2, cpu_resident_bytes=1024,
                                              cpu_peak_bytes=2048))
    sampler = PhaseSampler(tmp_path, clock=lambda: 3)
    sampler.sample()
    result = sampler.snapshot()
    assert result['source_files']['archive'] == 'archive_resources.json'
    assert result['samples'][0]['archive_at'] == 2
    assert result['observed_phase_peaks']['cpu_resident_bytes'] == 1024
    assert result['observed_phase_peaks']['cpu_archive_lifetime_peak_bytes'] == 2048


def test_zero_cpu_residency_does_not_qualify_as_phase_peak_evidence():
    arms, plan = campaign()
    arms[0]['phases'][1]['resources']['observed_phase_peaks']['cpu_resident_bytes'] = 0
    result = analyze(arms, plan)
    assert result['status'] == 'needs_review'
    assert not result['resource_evidence']['complete']
    assert not result['resource_evidence']['all_cpu_phase_peaks_observed']
    assert result['resource_evidence']['phases'][0]['cpu_evidence'] == 'missing_nonzero_payload'


def test_positive_lifetime_cpu_peak_is_preserved_but_never_called_a_phase_peak():
    arms, plan = campaign()
    peaks = arms[0]['phases'][1]['resources']['observed_phase_peaks']
    peaks['cpu_resident_bytes'] = 0
    peaks['cpu_archive_lifetime_peak_bytes'] = 4096
    result = analyze(arms, plan)
    assert result['status'] == 'passed'
    assert result['resource_evidence']['complete']
    assert not result['resource_evidence']['all_cpu_phase_peaks_observed']
    phase = result['resource_evidence']['phases'][0]
    assert phase['cpu_evidence'] == 'archive_lifetime_peak_only'
    assert not phase['cpu_phase_peak_observed']


@pytest.mark.parametrize('change', [dict(seconds=0), dict(seconds=float('nan')),
                                  dict(status='failed'), dict(output_ids=[True, 2]),
                                  dict(usage=dict(completion_tokens=100))])
def test_summary_rejects_incomplete_or_invalid_evidence(change):
    value = response()
    value.update(change)
    with pytest.raises(ValueError):
        summarize([value])


def test_stable_abba_compares_every_owner_and_excludes_warmup_from_performance():
    arms, plan = campaign()
    result = analyze(arms, plan)
    assert result['status'] == 'passed'
    assert result['output_comparison']['repeatability'] == dict(cpu='stable', tiered='stable')
    assert len(result['output_comparison']['cross_backend']) == 4
    assert len(result['output_comparison']['within_backend']) == 2
    assert all(row['repeats'] == 2 and row['concurrency'] == 2 for row in result['performance'])
    assert result['resource_evidence']['complete']


def test_stable_cross_backend_mismatch_is_not_reported_as_success():
    arms, plan = campaign()
    for arm in arms:
        if arm['backend'] == 'tiered':
            arm['phases'][1]['responses'][1]['output_ids'][0] = 99
    result = analyze(arms, plan)
    assert result['status'] == 'output_mismatch'
    assert result['output_comparison']['status'] == 'mismatch_with_stable_repeats'
    assert result['output_comparison']['cross_backend'][0]['mismatches'][0]['owner'] == 1
    assert result['output_comparison']['cross_backend'][0]['mismatches'][0]['first_different_token'] == 0


def test_aa_instability_is_explicitly_inconclusive_not_storage_equivalence():
    arms, plan = campaign()
    arms[3]['phases'][1]['responses'][0]['output_ids'][0] = 99
    result = analyze(arms, plan)
    assert result['status'] == 'needs_review'
    assert result['output_comparison']['repeatability']['cpu'] == 'unstable'
    assert result['output_comparison']['status'] == 'inconclusive_within_backend_variation'
    assert len(result['performance']) == 2  # Measurements remain available.


def test_no_repeat_and_no_sampling_cannot_claim_full_pass():
    arms, plan = campaign(('cpu', 'tiered'))
    for arm in arms:
        for phase in arm['phases']:
            phase.pop('resources')
    result = analyze(arms, plan)
    assert result['status'] == 'needs_review'
    assert result['output_comparison']['status'] == 'exact'
    assert result['output_comparison']['repeatability']['cpu'] == 'unmeasured'
    assert not result['resource_evidence']['complete']


@pytest.mark.parametrize('mutation', ['failed_arm', 'missing_response', 'input_drift', 'kg_drift', 'pool_drift', 'summary_drift'])
def test_analysis_rejects_errors_or_changed_workload_and_semantics(mutation):
    arms, plan = campaign()
    if mutation == 'failed_arm':
        arms[1]['status'] = 'failed'
    elif mutation == 'missing_response':
        arms[1]['phases'][1]['responses'].pop()
    elif mutation == 'input_drift':
        arms[1]['phases'][1]['responses'][1]['input_digest'] = 'foreign'
    elif mutation == 'kg_drift':
        arms[1]['profile']['kvmem']['gen_budget'] = 1024
    elif mutation == 'pool_drift':
        arms[1]['profile']['kv_cache_memory_bytes'] = 2048
    else:
        arms[1]['phases'][1]['summary']['wall_seconds'] = 1
    result = analyze(arms, plan)
    assert result['status'] == 'failed' and result['issues']


def test_incremental_profiles_keep_kg_picker_refresh_and_fixed_pool():
    original = dict(kv_cache_memory_bytes=123456, kvmem=dict(select_budget=36, gen_budget=16,
        pressure_policy='attention', refresh_quantum_tokens=32, request_swap_min_resident_steps=64))
    frozen = deepcopy(original)
    static = arm_profile(original, 'tiered', disk_bytes=500, checkpoint_bytes=200, port=123)
    dynamic = arm_profile(original, 'incremental', disk_bytes=500, checkpoint_bytes=200, port=123)
    assert original == frozen
    assert semantic_profile(static) == semantic_profile(dynamic)
    assert static['kvmem']['gpu_memory_admission'] == 'static'
    assert dynamic['kvmem']['gpu_memory_admission'] == 'incremental'
    assert dynamic['kvmem']['archive_backend'] == 'tiered'
    assert dynamic['kvmem']['request_checkpoint_bytes'] == 200
    assert dynamic['kvmem']['request_swap_min_resident_steps'] == 64
    assert dynamic['kv_cache_memory_bytes'] == static['kv_cache_memory_bytes'] == 123456
    arms, plan = campaign(('tiered', 'incremental', 'incremental', 'tiered'))
    assert analyze(arms, plan)['status'] == 'passed'


def test_baseline_and_ssd_profiles_keep_semantics_and_older_parser_compatibility():
    config = dict(kvmem=dict(select_budget=2048, gen_budget=2048,
        request_checkpoint_ssd_bytes=10000, pressure_policy='attention'))
    baseline = arm_profile(config, 'baseline', disk_bytes=1024, checkpoint_bytes=256, port=12)
    current = arm_profile(config, 'ssd', disk_bytes=1024, checkpoint_bytes=256, port=12,
                          checkpoint_ssd_bytes=4096)
    assert 'request_checkpoint_ssd_bytes' not in baseline['kvmem']
    assert current['kvmem']['request_checkpoint_ssd_bytes'] == 4096
    assert current['kvmem']['request_checkpoint_bytes'] == 256
    assert semantic_profile(current) == semantic_profile(baseline)


def test_checkpoint_disk_metrics_do_not_replace_history_metrics():
    sample = resource_sample({}, dict(cpu_resident_bytes=4, disk_write_bytes=8,
        request_checkpoints=dict(storage=dict(cpu_resident_bytes=16, disk_write_bytes=32,
            cpu_charge_peak_bytes=64))), 0)
    assert sample['cpu_resident_bytes'] == 4 and sample['disk_write_bytes'] == 8
    assert sample['checkpoint_cpu_resident_bytes'] == 16
    assert sample['checkpoint_disk_write_bytes'] == 32
    assert sample['checkpoint_cpu_charge_peak_bytes'] == 64


def test_directory_analysis_does_not_rewrite_original_responses(tmp_path):
    arms, plan = campaign()
    save(tmp_path/'plan.json', plan)
    save(tmp_path/'results.json', dict(arms=arms))
    before = (tmp_path/'results.json').read_bytes()
    for index, arm in enumerate(arms):
        save(tmp_path/f'{index+1:02d}-{arm["backend"]}'/'profile.json', arm['profile'])
    result = analyze_directory(tmp_path)
    assert result['status'] == 'passed'
    assert (tmp_path/'results.json').read_bytes() == before
    assert json.loads((tmp_path/'analysis.json').read_text()) == result


class FakeStream:
    def __init__(self, lines):
        self.lines = lines
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        return iter(self.lines)


def stream_line(value):
    return b'data: '+json.dumps(value).encode()+b'\n'


@pytest.mark.parametrize('failure', [None, 'error', 'missing_done', 'usage'])
def test_stream_preserves_partial_outputs_and_fails_on_error_or_bad_accounting(monkeypatch, failure):
    import benchmark_dynamic_memory as module
    lines = [stream_line(dict(choices=[dict(token_ids=[4], text='first')]))]
    if failure == 'error':
        lines.append(stream_line(dict(error=dict(message='injected'))))
    else:
        lines.append(stream_line(dict(choices=[dict(token_ids=[5], text=' second', finish_reason='length')],
                                      usage=dict(prompt_tokens=128, completion_tokens=99 if failure == 'usage' else 2))))
        if failure != 'missing_done':
            lines.append(b'data: [DONE]\n')
    monkeypatch.setattr(module.urllib.request, 'build_opener', lambda *_:SimpleNamespace(
        open=lambda *a, **k:FakeStream(lines)))
    payload = dict(prompt=[1000]*128, max_tokens=2)
    barrier = SimpleNamespace(wait=lambda **_:None)
    if failure is None:
        value = request('http://unused', payload, barrier)
        assert value['status'] == 'completed' and value['output_ids'] == [4, 5]
        assert value['output_text'] == 'first second'
    else:
        with pytest.raises(RequestFailure) as raised:
            request('http://unused', payload, barrier)
        value = raised.value.response
        assert value['status'] == 'failed'
        assert value['output_ids'] == ([4] if failure == 'error' else [4, 5])
        assert value['output_text'].startswith('first')
