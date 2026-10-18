"""CPU-only contract tests; no model, CUDA events or server is started."""
import asyncio
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('prefill_wave_benchmark',
    Path(__file__).parents[1]/'scripts/benchmark_prefill_wave.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
wave = probe.wave


def arguments(**changes):
    values = dict(resident_requests=2, seed_requests=1, concurrency=3, context_tokens=8,
        output_tokens=3, seed_context_tokens=12, seed_output_tokens=2, duration_seconds=.02,
        seed_offset_seconds=.005, run_tag='test', capture_directory=Path('/unused'),
        mode='saturated', modes=['saturated'], warmup_turns=1, pairs=1, idle_seconds=0.,
        prefill_cap_tokens=32)
    values.update(changes)
    return SimpleNamespace(**values)


def profile():
    return dict(port=1, served_model_name='fake', kvmem=dict(gen_budget=128, prefill_batch_tokens=32))


def fixture(args=None):
    return wave.prepare_fixture(dict(prompt=list(range(100, 140)), eos=2), args or arguments(), profile())


def test_frozen_ids_and_quotas_do_not_depend_on_any_arm_output():
    args = arguments(seed_requests=2, concurrency=4)
    source = dict(prompt=list(range(100, 140)), eos=2)
    original = copy.deepcopy(source)
    result = wave.prepare_fixture(source, args, profile())
    assert source == original
    assert result['residents'][0]['tokens'] == source['prompt'][:8]
    assert result['residents'][0]['suffix_token'] == source['prompt'][8]
    assert result['seeds'][0] == result['seeds'][1]
    assert result['seeds'][0]['tokens'] is not result['seeds'][1]['tokens']
    corrupted = copy.deepcopy(result)
    corrupted['seeds'][0]['tokens'][0] += 1
    with pytest.raises(ValueError, match='fingerprint'):
        wave.validate_fixture(corrupted, args, profile())


def test_overlap_merges_seed_intervals_and_keeps_half_open_token_boundaries():
    residents = [dict(timeline=[[1., 1], [2., 3], [5., 4], [8., 7]], end=12.),
                 dict(timeline=[[2., 1], [4., 2], [8., 3]], end=10.)]
    seeds = [dict(begin=0., timeline=[[5., 1]]), dict(begin=4., timeline=[[8., 1]])]
    result = wave.overlap_summary(residents, seeds, 0., 11.)
    assert result['intervals'] == [[2., 8.]]
    assert result['seconds'] == 6.
    assert result['resident_output_tokens'] == 10  # Includes t=2; excludes t=8.
    assert result['common_decode_seconds'] == 8.


def test_prefill_progress_never_uses_optimistic_inflight_or_future_cursors():
    seed = dict(response_id='cmpl-seed', input_tokens=100, timeline=[[12., 1]])
    def sample(at, phase, remaining, reason=None):
        return dict(at=at, candidates=[dict(request_id='cmpl-seed-0-1234abcd',
            phase=phase, remaining_tokens=remaining, reason=reason)])
    rows = [sample(2., 'prefill', 80), sample(3., 'in_flight', 0, 'native_step_fence'),
            sample(10., 'decode', 0), sample(11., 'decode', 0)]
    partial = wave.seed_progress(seed, rows, 10.)
    assert partial['completed_tokens_lower_bound'] == 20
    assert partial['completed_tokens_upper_bound'] == 100
    assert not partial['exact']
    # A first token exactly at T1 belongs outside the half-open observation.
    assert wave.seed_progress(seed, [], 12.)['completed_tokens_lower_bound'] == 0
    assert wave.seed_progress(seed, [], 12.01)['exact']


def test_mixed_step_cap_check_exempts_atomic_query_and_never_infers_unsampled_work():
    common = dict(decode_requests=5, prefill_requests=3, prefill_tokens=4096)
    rows = [dict(common, started_at=1., ordinary_prefill_only=True, ordinary_prefill_tokens=2048),
            dict(common, started_at=2., ordinary_prefill_only=False, atomic_prefill_tokens=4096),
            dict(common, started_at=3., ordinary_prefill_only=True, ordinary_prefill_tokens=4096)]
    result = wave.mixed_steps(rows, 1., 3., 2048)
    assert result['mixed_steps'] == 2
    assert result['ordinary_steps_exceeding_cap'] == 0
    assert result['atomic_query_steps'] == 1


def test_hot_restore_requires_full_exact_prefix_and_unambiguous_native_identity():
    data = fixture(arguments(resident_requests=1, concurrency=2))
    resident = dict(agent=0, request_id='formal', response_id='cmpl-formal')
    restore = dict(request_id='cmpl-formal-0-deadbeef', history_tokens=8, active_tokens=8,
                   resident_active_tokens=8, uploaded_bytes=0, cancelled=False)
    assert wave.restore_evidence([resident], [restore], data)['passed']
    for changes in (dict(history_tokens=7), dict(resident_active_tokens=7),
                    dict(uploaded_bytes=1), dict(cancelled=True)):
        assert not wave.restore_evidence([resident], [dict(restore, **changes)], data)['passed']
    assert not wave.restore_evidence([resident], [restore, restore], data)['passed']


class Response:
    def __init__(self, payload, failure=False):
        self.payload, self.failure = payload, failure

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        p = self.payload
        if self.failure:
            yield 'data: '+json.dumps(dict(error=dict(message='injected failure')))
            return
        seed = '-2-1-workload' in p['request_id']
        await asyncio.sleep(.03 if seed else .001)
        yield 'data: '+json.dumps(dict(id='cmpl-'+p['request_id'], choices=[dict(index=0, token_ids=[9000])]))
        if p['max_tokens'] > 1:
            await asyncio.sleep(.005 if seed else .08)
            yield 'data: '+json.dumps(dict(id='cmpl-'+p['request_id'],
                choices=[dict(index=0, token_ids=[9001]*(p['max_tokens']-1))]))
        yield 'data: '+json.dumps(dict(id='cmpl-'+p['request_id'], choices=[],
                                     usage=dict(completion_tokens=p['max_tokens'])))
        yield 'data: [DONE]'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Client:
    def __init__(self, fail=False):
        self.payloads, self.fail = [], fail

    def stream(self, method, url, *, json):
        self.payloads.append(copy.deepcopy(json))
        return Response(json, failure=self.fail and '-0-1-workload' in json['request_id'])


def stubs(monkeypatch, client):
    retirements, public = [], []
    async def retired(capture, **kwargs):
        retirements.append(kwargs)
        return dict(retired=True)
    async def counters(capture, expected, **kwargs):
        public.append(expected)
        return dict(passed=True, completed_tokens=expected, expected_tokens=expected)
    def records(path):
        if path.name != 'async_restore.jsonl':
            return []
        return [dict(request_id='cmpl-'+p['request_id']+'-0-deadbeef', history_tokens=8,
            active_tokens=8, resident_active_tokens=8, uploaded_bytes=0, cancelled=False)
            for p in client.payloads if '-1-workload' in p['request_id'] and len(p['prompt']) == 9]
    monkeypatch.setattr(wave.timed, 'wait_retirement', retired)
    monkeypatch.setattr(wave, 'wait_public_complete', counters)
    monkeypatch.setattr(wave, 'read_records', records)
    return retirements, public


def test_t1_does_not_cancel_and_all_fixed_work_plus_retirement_is_counted(tmp_path, monkeypatch):
    client = Client()
    retired, public = stubs(monkeypatch, client)
    data = fixture()
    result = asyncio.run(wave.workload(client, profile(), data, tmp_path, arguments()))
    rows = json.loads((tmp_path/'traces.json').read_text())
    window = json.loads((tmp_path/'window.json').read_text())
    formal = [r for r in rows if r['kind'] == 'workload']
    assert len(formal) == 3 and all(r['status'] == 'completed' for r in formal)
    assert result['fixed_work_output_tokens'] == 8
    assert result['total_stream_output_tokens'] == 13  # 2 warm + 8 fixed + 3 cleanup.
    assert result['full_work_makespan_seconds'] > result['duration_seconds']
    assert result['work_drain_after_t1_seconds'] > 0
    assert result['coverage_valid'] and result['cleanup_complete']
    assert result['seeds'][0]['ttft_right_censored_at_least_seconds'] > 0
    assert result['cancelled_requests'] == []
    assert public == [30]  # 2*(8 warm + 1 suffix) + 12 seed.
    assert len(retired) == 3
    for p in client.payloads:
        assert p['ignore_eos'] and p['seed'] == 1000
        if len(p['prompt']) == 9:
            assert p['prompt'] == data['residents'][0]['tokens']+[data['residents'][0]['suffix_token']]
            assert 9000 not in p['prompt']  # Warmup output must not change either arm's input.
    seed = next(r for r in formal if r['source_metadata']['role'] == 'seed')
    assert seed['begin'] >= window['begin']+.005
    assert seed['source_metadata']['planned_dispatch'] == window['begin']+.005
    assert min(r['begin'] for r in rows if r['kind'] == 'cleanup') >= window['end']


def test_failure_cancels_owned_clients_preserves_trace_and_never_claims_cleanup(tmp_path, monkeypatch):
    client = Client(fail=True)
    stubs(monkeypatch, client)
    with pytest.raises(RuntimeError, match='injected failure'):
        asyncio.run(wave.workload(client, profile(), fixture(), tmp_path, arguments()))
    result = json.loads((tmp_path/'results.json').read_text())
    assert result['status'] == 'failed' and not result['drained'] and not result['cleanup_complete']
    assert (tmp_path/'traces.json').exists()
    assert not any('-2-cleanup' in p['request_id'] for p in client.payloads)


def result(**updates):
    row = dict(status='completed', drained=True, cleanup_complete=True, cancelled_requests=[],
        native_client_total_output_reconciliation=dict(passed=True), work_identity='same',
        fixed_work_input_tokens=100, fixed_work_output_tokens=100, coverage_valid=True,
        full_work_makespan_seconds=20., full_work_retired_makespan_seconds=21., output_tokens_per_second=5.,
        full_public_prefill=dict(passed=True, completed_tokens=100),
        prefill_cap_tokens=32, public_mixed_steps=dict(mixed_with_multiple_prefills=1,
            maximum_ordinary_tokens=64, ordinary_only_mixed_steps=1, ordinary_steps_exceeding_cap=0),
        seeds=[dict(full_ttft_seconds=10.)], seed_completed_tokens_lower_bound_at_t1=50,
        seed_completed_tokens_upper_bound_at_t1=100, decode_seed_overlap={}, t0_resident_offsets=[])
    row.update(updates)
    return row


def test_better_window_output_with_slower_complete_work_is_not_a_win():
    rows = [dict(pair=0, arm='A0', results={'saturated':result()}),
            dict(pair=0, arm='P1', results={'saturated':result(
                output_tokens_per_second=8., full_work_makespan_seconds=25.)})]
    compared = probe.compare(rows)['pairs'][0]
    assert compared['window_output_only_not_an_offline_win']
    assert not compared['faster_complete_fixed_work']
    rows[1]['results']['saturated']['fixed_work_output_tokens'] = 99
    with pytest.raises(ValueError, match='identical'):
        probe.compare(rows)


def test_full_public_counter_rejects_recomputation_not_hides_it(tmp_path):
    (tmp_path/'public_work.json').write_text(json.dumps(dict(
        submitted=dict(public_prefill_tokens=31), completed=dict(public_prefill_tokens=30))))
    with pytest.raises(RuntimeError, match='re-prefill'):
        asyncio.run(wave.wait_public_complete(tmp_path, 30, timeout=.01))


def test_real_launcher_profiles_and_manifest_preserve_exactly_one_cap_difference(tmp_path):
    from test_coordination_qualification import native_profile
    from vllm_kvmem.serve import build_launch
    candidate = json.loads(native_profile(tmp_path).read_text())
    candidate['kvmem'].update(block_tokens=32, repository_blocks=32768,
        gen_budget=131072, select_budget=131072, cpu_read_cache_bytes=100_000_000_000,
        cpu_history_coordination=True, async_restore=True, maintenance_max_steps=8,
        active_resident_pressure_only=True)
    profiles = probe.cap_profiles(candidate, 2048)
    manifest = dict(comparison_kind=probe.KIND, workload=dict(prefill_cap_tokens=2048), profile_sha256={})
    for arm, p in profiles.items():
        expected = 2048 if arm == 'P1' else 0
        plan = build_launch(p, tmp_path, tmp_path/('launch-'+arm))
        argv = plan['vllm_argv']
        transfer = json.loads(argv[argv.index('--kv-transfer-config')+1])
        assert transfer['kv_connector_extra_config']['kvmem']['prefill_batch_tokens'] == expected
        path = tmp_path/(arm+'.json')
        path.write_text(json.dumps(p))
        manifest['profile_sha256'][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert probe.validate_prepared(tmp_path)['comparison_kind'] == probe.KIND
    altered = profiles['P1']
    altered['kvmem']['active_resident_max_fraction'] = .25
    path = tmp_path/'P1.json'
    path.write_text(json.dumps(altered))
    manifest['profile_sha256']['P1'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Only prefill_batch_tokens'):
        probe.validate_prepared(tmp_path)
