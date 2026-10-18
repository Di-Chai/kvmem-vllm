import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec = importlib.util.spec_from_file_location('capacity_profile',
    Path(__file__).resolve().parents[1]/'scripts/profile_deepswe_capacity.py')
profile = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(Path(spec.origin).parent))
try:
    spec.loader.exec_module(profile)
finally:
    sys.path.pop(0)


def test_low_gpu_utilization_does_not_imply_no_model_demand():
    samples = [dict(at=t, gpu='100, 0', raw_metrics='vllm:num_requests_running 1')
               for t in (1, 2, 3)]
    report = profile.gpu_demand(samples, [dict(at=1.5)], [dict(at=2.5)], 3)
    groups = {r['http_state']: r for r in report['groups']}
    assert groups['outstanding_model_http']['samples'] == 1
    assert groups['no_outstanding_model_http']['samples'] == 2


def test_cancelled_http_closes_demand_but_not_native_capacity_envelope(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps(dict(status='window_complete',
        model_ready_at=0,window_deadline=4,workers=1)))
    (tmp_path/'profile.json').write_text(json.dumps(dict(kvmem=dict(
        select_budget=65536,gen_budget=32768,block_tokens=32))))
    samples=[dict(at=t,gpu='1, 0',raw_metrics='vllm:num_requests_running 0') for t in (1,2,3)]
    (tmp_path/'resources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
    bridge=tmp_path/'bridge';bridge.mkdir()
    rows=[dict(event='start',task='a',turn=1,at=.5,prompt_tokens=200000),
          dict(event='cancelled',task='a',turn=1,at=1.5)]
    (bridge/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    pool=dict(block_size=1616,num_gpu_blocks=100,kv_cache_size_tokens=161600)
    result=profile.profile_stage(tmp_path,pool)
    assert result['closed_unsuccessful_http']==1
    assert result['capacity_overlap']['peak_http_demand']['users']==1
    groups={r['http_state']:r for r in result['gpu_demand']['groups']}
    assert groups['outstanding_model_http']['samples']==1
    assert groups['no_outstanding_model_http']['samples']==2


def test_work_bounds_exclude_unconfirmed_warmup_and_keep_inflight_uncertainty():
    fields = ('public_batches', 'public_prefill_tokens', 'public_non_prefill_tokens',
              'prefill_request_batches', 'decode_request_batches')
    def row(at, submitted, completed, instance='same'):
        return dict(wall_time=at, counter_id=instance,
            submitted={f: submitted for f in fields}, completed={f: completed for f in fields})
    result = profile.confirmed_public_work([row(9, 100, 90), row(19, 300, 280)], 10, 20)
    assert result['bounds']['public_prefill_tokens'] == dict(lower=180, upper=210)
    assert not profile.confirmed_public_work([row(19, 300, 280)], 10, 20)['available']
    with pytest.raises(ValueError, match='instance changed'):
        profile.confirmed_public_work([row(9, 100, 90), row(19, 300, 280, 'new')], 10, 20)


def test_step_groups_separate_batch_size_and_context():
    rows = [dict(prefill_tokens=p, decode_requests=d, decode_context_bucket=c, step_ms=m)
            for p, d, c, m in [(0, 4, 65536, 10), (0, 4, 65536, 20),
                                (0, 4, 131072, 30), (128, 4, 65536, 40)]]
    groups = profile.step_groups(rows)
    assert len(groups) == 3
    short = next(g for g in groups if g['kind'] == 'decode' and g['maximum_decode_context_bucket'] == 65536)
    assert short['steps'] == 2 and short['durations']['step_ms']['mean'] == 15
    assert short['durations']['model_ms']['missing_count'] == 2


def test_restore_cancelled_work_is_not_reported_as_completed():
    rows = [dict(started=1, finished=2, lane='resident', uploaded_bytes=0, model_bytes=10),
            dict(started=2, finished=3, lane='assembly', uploaded_bytes=100, cancelled=True)]
    report = profile.restore_groups(rows)
    assert len(report) == 1 and report[0]['recurrent_model_bytes'] == 10


def test_checkpoint_stage_map_keeps_missing_stages_and_sync_cost_separate():
    rows = [dict(status='submitted', host_stage_seconds={'allocation': .2},
                 host_total_seconds=.3, host_sync_wait_seconds=.01),
            dict(status='submitted', host_stage_seconds={}, host_total_seconds=.1)]
    report = profile.checkpoint_profile(rows)
    assert report['host_stages']['allocation']['mean'] == .2
    assert report['host_stages']['allocation']['missing_count'] == 1
    assert report['durations']['host_sync_wait_seconds']['mean'] == .01


def test_history_rate_is_per_decoder_and_excludes_first_and_unknown_timing():
    rows = [dict(prompt_tokens=n, usage={'completion_tokens': out}, metrics=timing)
            for n, out, timing in [(200000, 101, {'generation_time_ms': 1000}),
                                   (200000, 201, {'generation_time_ms': 1000}),
                                   (200000, 10000, None),
                                   (32000, 11, {'generation_time_ms': 1000})]]
    report = profile.history_matched_requests(rows)
    long = next(g for g in report['buckets'] if g['prompt_length_bucket'] == '192K-256K')
    assert long['weighted_per_decoder_tokens_per_second'] == 150
    assert long['summed_decoder_seconds'] == 2
    assert long['completed_model_turns'] == 3


def test_output_headroom_counts_request_maximum_not_observed_generation(tmp_path):
    (tmp_path/'profile.json').write_text(json.dumps(dict(kvmem=dict(
        select_budget=65536, gen_budget=32768, block_tokens=32))))
    directory = tmp_path/'bridge/task/00001'
    directory.mkdir(parents=True)
    (directory/'plan.json').write_text(json.dumps(dict(max_tokens=32000)))
    report = profile.output_reservations(tmp_path, [dict(task='task', turn=i) for i in (1, 2)])
    assert report['missing_plan_files'] == 1
    assert report['requested_output_tokens'] == [dict(tokens=32000, requests=1,
        available_above_select_budget=736)]


def test_native_geometry_uses_actual_pool_and_rejects_ambiguous_series():
    line = 'vllm:cache_config_info{block_size="16",num_gpu_blocks="4",kv_cache_size_tokens="48",cache_dtype="fp8_e4m3",mamba_cache_mode="align"} 1'
    result = profile.native_geometry(line)
    assert result['num_gpu_blocks'] == 4 and result['block_size'] == 16
    assert profile.native_geometry(line+'\n'+line) is None
    assert profile.native_geometry(line.replace('num_gpu_blocks="4"', 'num_gpu_blocks="None"')) is None


def test_capacity_proof_requires_actual_pure_decode_rows_and_distinct_envelopes():
    starts = [dict(task=t, turn=1, at=0, prompt_tokens=48) for t in ('a', 'b')]
    finishes = [dict(task=t, turn=1, at=10) for t in ('a', 'b')]
    geometry = dict(block_size=16, num_gpu_blocks=4, kv_cache_size_tokens=48)
    steps = [dict(started_at=105, sequence=1, decode_requests=2, prefill_tokens=0,
                  prefill_requests=0, decode_context_bucket=16),
             dict(started_at=106, sequence=2, decode_requests=2, prefill_tokens=1)]
    result = profile.capacity_overlap(starts, finishes, [dict(at=5)], steps, 100, 2, geometry, 10)
    assert result['peak_http_demand']['full_prompt_tokens'] == 96
    assert result['matched_target_pure_decode_steps'] == 1
    assert result['steps_attention_alone_exceeds_native_physical_pool'] == 1
    assert result['best_matched_cohort']['native_attention_pages_lower_bound'] == 6
    result = profile.capacity_overlap(starts, finishes, [dict(at=5)], [], 100, 2, geometry, 10)
    assert result['matched_target_pure_decode_steps'] == 0  # HTTP overlap is insufficient.
    starts[1]['task'] = 'a'; starts[1]['turn'] = 2
    assert profile.capacity_overlap(starts, [], [], steps, 100, 2, geometry, 10)[
        'matched_target_pure_decode_steps'] == 0


def test_reported_token_equivalent_is_not_a_physical_capacity_proof():
    starts = [dict(task=t, turn=1, at=0, prompt_tokens=48) for t in ('a', 'b')]
    steps = [dict(started_at=5, sequence=1, decode_requests=2, prefill_tokens=0)]
    geometry = dict(block_size=16, num_gpu_blocks=8, kv_cache_size_tokens=48)
    result = profile.capacity_overlap(starts, [], [], steps, 0, 2, geometry, 10)
    assert result['steps_above_reported_fullcontext_equivalent'] == 1
    assert result['steps_attention_alone_exceeds_native_physical_pool'] == 0


def test_unfinished_old_request_cannot_replace_a_real_near_boundary_member():
    starts = [dict(task='old', turn=1, at=-2, prompt_tokens=1000),
              dict(task='real-a', turn=1, at=0, prompt_tokens=16),
              dict(task='real-b', turn=1, at=-1, prompt_tokens=16)]
    step = dict(started_at=.005, sequence=1, decode_requests=2, prefill_tokens=0)
    geometry = dict(block_size=16, num_gpu_blocks=4, kv_cache_size_tokens=48)
    result = profile.capacity_overlap(starts, [], [], [step], 0, 2, geometry, 10)
    assert result['matched_target_pure_decode_steps'] == 0
    assert result['steps_attention_alone_exceeds_native_physical_pool'] == 0
