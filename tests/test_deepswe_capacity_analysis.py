import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('deepswe_capacity_analysis',
    Path(__file__).parents[1]/'scripts/analyze_deepswe_capacity.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def test_prefill_uses_computed_source_and_does_not_count_cached_tokens_twice():
    row=analysis.metrics('vllm:prompt_tokens_by_source_total{engine="0",source="local_compute"} 100\n'
                         'vllm:prompt_tokens_by_source_total{engine="0",source="local_cache_hit"} 900\n'
                         'vllm:prompt_tokens_total{engine="0"} 1000\n'
                         'vllm:prompt_tokens_cached_total{engine="0"} 900\n')
    assert analysis.prefill_count(row)==100
    assert analysis.prefill_count({'vllm:prompt_tokens_total':1000,'vllm:prompt_tokens_cached_total':900})==100
    with pytest.raises(ValueError):analysis.prefill_count({'vllm:prompt_tokens_total':1000})


def test_full_window_includes_idle_and_rejects_counter_reset():
    def sample(at,prefill,output):
        return dict(at=at,raw_metrics=f'vllm:prompt_tokens_total  {prefill}\n'
                    f'vllm:prompt_tokens_cached_total 0\nvllm:generation_tokens_total {output}\n')
    rows=[sample(0,0,0),sample(10,100,100),sample(20,100,100)]
    result=analysis.segment(rows,0,20)
    assert result['generation_tokens_per_second']==5
    assert result['prefill_tokens_per_second']==5
    assert result['seconds']==20
    with pytest.raises(ValueError):analysis.segment([sample(0,100,100),sample(20,0,0)],0,20)


def test_native_running_gauge_alone_cannot_prove_decode_batch(tmp_path):
    assert analysis.step_evidence(tmp_path,0,10,{'wall_at':0,'perf_counter_at':0},8)['available'] is False


def test_live_window_end_uses_observed_data_not_future_deadline(tmp_path):
    (tmp_path/'run.json').write_text(json.dumps(dict(status='running',model_ready_at=100,
        window_deadline=1900,workers=8,tasks={})))
    (tmp_path/'resources.jsonl').write_text(json.dumps(dict(at=110))+'\n')
    report=analysis.analyze_stage(tmp_path)
    assert report['observed_end']==110
    assert report['whole_window']['available'] is False


def test_unavailable_request_timing_is_reported_as_missing_not_a_latency():
    report=analysis.distribution([None,10,20])
    assert report['count']==2 and report['missing_count']==1 and report['mean']==15
    with pytest.raises(ValueError):analysis.distribution([float('nan')])


def test_certified_decode_interior_survives_all_possible_transport_splits():
    starts=[dict(task='a',turn=1,at=100)]
    finishes=[dict(task='a',turn=1,at=115,usage={'completion_tokens':101},
        metrics=dict(queue_time_ms=1000,time_to_first_token_ms=2000,generation_time_ms=10000))]
    report=analysis.decode_interiors(starts,finishes);interval=report['intervals'][0]
    for ingress in (0,1,2):
        assert 103+ingress<=interval['begin']<interval['end']<=113+ingress
    assert report['weighted_per_decoder_tokens_per_second']==10


def test_no_public_prefill_proxy_retains_zero_output_stalls_and_acquisition_bounds():
    def sample(at,out,prefill=0):
        return dict(at=at,gpu='0, 0',raw_metrics=f'vllm:prompt_tokens_total {prefill}\n'
            f'vllm:prompt_tokens_cached_total 0\nvllm:generation_tokens_total {out}\n')
    rows=[sample(1,10),sample(2.1,30),sample(3.2,30),sample(4.3,50)]
    starts=[dict(task='a',turn=1,at=0)]
    finishes=[dict(task='a',turn=1,at=10,usage={'completion_tokens':101},
        metrics=dict(queue_time_ms=0,time_to_first_token_ms=0,generation_time_ms=10000))]
    result=analysis.decode_accounting(rows,starts,finishes,4)
    phase=result['no_public_prefill_windows']['1']
    assert phase['eligible_sample_intervals']==2
    assert phase['generation_tokens']==20 and phase['seconds']==pytest.approx(2.2)
    assert result['no_public_prefill_windows']['4']['eligible_sample_intervals']==0
    assert analysis.overlap_windows([dict(begin=2.1,end=3.2)],1)==[[2.1,3.2]]
    rows[2]=sample(3.2,30,100)
    result=analysis.decode_accounting(rows,starts,finishes,4)
    assert result['no_public_prefill_windows']['1']['seconds']==pytest.approx(1.1)


def test_overlap_requires_simultaneous_completed_request_interiors():
    intervals=[dict(begin=0,end=3),dict(begin=2,end=5),dict(begin=3,end=4)]
    assert analysis.overlap_windows(intervals,2)==[[2,4]]


def test_zero_prompt_counter_cannot_hide_an_unfinished_long_prefill():
    def sample(at,out):
        return dict(at=at,gpu='0, 0',raw_metrics='vllm:prompt_tokens_total 0\n'
            f'vllm:prompt_tokens_cached_total 0\nvllm:generation_tokens_total {out}\n')
    samples=[sample(1,10),sample(2.1,30),sample(3.2,50)]
    starts=[dict(task='decode',turn=1,at=0),dict(task='still_prefilling',turn=1,at=.5)]
    finishes=[dict(task='decode',turn=1,at=10,usage={'completion_tokens':101},
        metrics=dict(queue_time_ms=0,time_to_first_token_ms=0,generation_time_ms=10000))]
    result=analysis.decode_accounting(samples,starts,finishes,1)
    assert result['no_public_prefill_windows']['1']['eligible_sample_intervals']==0


def test_native_step_bound_removes_prompt_credits_and_uses_separated_scrapes():
    def sample(at,count,b16,b32):
        return dict(at=at,raw_metrics=f'vllm:iteration_tokens_total_count{{engine="0"}} {count}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="16.0"}} {b16}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="32.0"}} {b32}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {count}\n'
            'vllm:num_preemptions_total 0\n')
    rows=[sample(0,2,1,1),sample(10,10200,8990,10090),sample(11,10220,9000,10100)]
    bound=analysis.native_batch_lower_bound(rows,100,4,4,True)
    assert bound['available'] and bound['maximum_observed_request_cohort_lower_bound']==4
    assert bound['bounds'][0]['new_high_iteration_count_lower_bound']==1199
    assert bound['bounds'][0]['high_generation_only_iteration_count_lower_bound']==1097
    contaminated=analysis.native_batch_lower_bound(rows,1500,4,4,True)
    assert contaminated['maximum_observed_request_cohort_lower_bound']==0
    assert analysis.native_batch_lower_bound(rows,100,4,4,False)['available'] is False


def test_native_iteration_proof_rejects_multiple_engine_series():
    with pytest.raises(ValueError,match='One engine'):
        analysis.iteration_histogram('vllm:iteration_tokens_total_bucket{engine="0",le="16"} 1\n'
            'vllm:iteration_tokens_total_bucket{engine="1",le="16"} 1\n')


def test_active_phase_baseline_subtracts_upper_bound_on_old_tail():
    def sample(at,count,le16):
        return dict(at=at,raw_metrics=f'vllm:iteration_tokens_total_count {count}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="16.0"}} {le16}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {count}\n'
            'vllm:num_preemptions_total 0\n')
    rows=[sample(0,100,50),sample(1,120,60),sample(10,300,100),sample(11,320,110)]
    bound=analysis.native_batch_lower_bound(rows,10,4,4,False,separated_baseline=True)
    # New tail >= (300-110) - (120-50), even if the early
    # baseline's two metrics were collected at different update instants.
    assert bound['bounds'][0]['new_high_iteration_count_lower_bound']==120
    assert bound['maximum_observed_request_cohort_lower_bound']==4
    assert not analysis.native_batch_lower_bound(rows[:3],10,4,4,False,separated_baseline=True)['available']


@pytest.mark.parametrize('preemption_metric', ['', 'vllm:num_preemptions_total 1\n'])
def test_native_iteration_bound_rejects_missing_or_positive_preemptions(preemption_metric):
    def row(at, count, bucket):
        return dict(at=at, raw_metrics=f'vllm:iteration_tokens_total_count {count}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="16"}} {bucket}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {count}\n'
            + preemption_metric)
    result=analysis.native_batch_lower_bound(
        [row(0,2,1),row(10,10200,8990),row(11,10220,9000)],100,4,4,True)
    assert result['available'] is False
    assert 'preemptions' in result['reason']


def test_mtp_sequence_ratio_proves_cohort_without_prompt_token_contamination():
    def row(at,iterations,sequences):
        return dict(at=at,raw_metrics=f'vllm:iteration_tokens_total_count {iterations}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {iterations}\n'
            f'vllm:spec_decode_num_drafts_total{{engine="0"}} {sequences}\n'
            'vllm:num_preemptions_total 0\n')
    rows=[row(0,100,500),row(1,101,507),row(10,201,1263),row(11,202,1270)]
    result=analysis.native_verification_cohort_bound(rows,8)
    assert result['available'] and result['maximum_verification_request_cohort_lower_bound']==8
    assert result['verified_sequences_lower_bound']==756
    assert result['output_iterations_upper_bound']==104
    assert not analysis.native_verification_cohort_bound(rows,7)['available']
    assert not analysis.native_verification_cohort_bound(rows[:3],8)['available']


def test_mtp_cohort_rejects_multiple_counter_series_and_resets():
    def row(at,n,s):
        return dict(at=at,raw_metrics=f'vllm:iteration_tokens_total_count {n}\n'
            f'vllm:iteration_tokens_total_bucket{{engine="0",le="+Inf"}} {n}\n'
            f'vllm:spec_decode_num_drafts_total{{engine="0"}} {s}\n'
            'vllm:num_preemptions_total 0\n')
    rows=[row(0,10,10),row(1,11,18),row(2,12,26),row(3,13,34)]
    rows[-1]['raw_metrics']+='vllm:spec_decode_num_drafts_total{engine="1"} 2\n'
    assert not analysis.native_verification_cohort_bound(rows,8)['available']
    rows[-1]=row(3,5,34)
    with pytest.raises(ValueError,match='reset'):
        analysis.native_verification_cohort_bound(rows,8)
    rows[-1]=row(3,13,34)
    rows[-1]['raw_metrics']=rows[-1]['raw_metrics'].replace('num_preemptions_total 0','num_preemptions_total 1')
    assert not analysis.native_verification_cohort_bound(rows,8)['available']


def test_historical_view_requires_full_cpu_history_and_compact_active_view():
    profile={'max_model_len':262144,'kvmem':dict(select_budget=65536,gen_budget=32768,block_tokens=32)}
    geometry=dict(select_budget=65536,gen_budget=32768,historical_tokens=262144,
        cpu_offload_bytes=100_000_000_000,cpu_archive_bytes_per_token=34816,
        archive_codec='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1')
    archive=dict(archive_backend='cpu',repository_gc=False,disk_read_bytes=0,disk_write_bytes=0)
    view=dict(namespace=['run-task','run-task'],history=200000,window=98304)
    sample=dict(at=1,scheduler_resources=dict(session_views=[view],owned_blocks=76,free_blocks=800,total_blocks=914),
        archive=dict(archive,cpu_authority_bytes=200000*34816,pending_downloads=0,session_retiring_requests=0))
    proof=analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)
    assert proof['all_workers_long_history_compression_observed'] is True
    sample['archive']['pending_downloads']=2
    sample['archive']['session_retiring_requests']=1
    assert analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)[
        'all_workers_long_history_compression_observed'] is True
    sample['scheduler_resources']['session_views'].append(dict(view))
    sample['archive']['cpu_authority_bytes']=400000*34816
    assert analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],2)[
        'maximum_coherently_compressed_users']==1
    sample['scheduler_resources']['session_views'].pop()
    sample['archive']['cpu_authority_bytes']=98304*34816
    proof=analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)
    assert proof['maximum_coherently_compressed_users']==0
    sample['archive']['cpu_authority_bytes']=200000*34816
    view['window']=100000
    proof=analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)
    assert proof['budget_violations'] and not proof['all_workers_long_history_compression_observed']
    view['window']=98304;archive['disk_read_bytes']=1
    assert not analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)['contract_matches']


@pytest.mark.parametrize('limit',[131072,196608,262144])
def test_history_accounting_uses_declared_context_ceiling(limit):
    profile=dict(max_model_len=limit,kvmem=dict(select_budget=65536,gen_budget=32768,block_tokens=32))
    geometry=dict(select_budget=65536,gen_budget=32768,historical_tokens=limit,
        cpu_offload_bytes=100_000_000_000,cpu_archive_bytes_per_token=34816,
        archive_codec='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1')
    archive=dict(archive_backend='cpu',repository_gc=False,disk_read_bytes=0,disk_write_bytes=0)
    view=dict(namespace=['run-task','run-task'],history=limit,window=98304)
    sample=dict(at=1,scheduler_resources=dict(session_views=[view]),
        archive=dict(archive,cpu_authority_bytes=limit*34816))
    proof=analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)
    assert proof['all_workers_long_history_compression_observed'] and proof['historical_token_limit']==limit
    view['history']=limit+1
    proof=analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)
    assert proof['budget_violations'] and not proof['all_workers_long_history_compression_observed']
    geometry['historical_tokens']=limit-1
    assert not analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)['contract_matches']
    profile.pop('max_model_len')
    assert not analysis.historical_view_evidence([sample],profile,geometry,archive,'run',['task'],1)['contract_matches']
