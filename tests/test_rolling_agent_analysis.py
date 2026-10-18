import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('test_rolling_analysis',Path(__file__).parents[1]/'scripts/analyze_rolling_agents.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(request_id, begin, end, timeline, length, reason='stop'):
    return dict(request_id=request_id,response_id=request_id,kind='workload',begin=begin,end=end,
        input_tokens=100,status='completed',output_ids=[1]*length,timeline=timeline,
        generation_mode='natural',finish_reason=reason)


def test_deadline_counts_tokens_and_turns_separately_without_discarding_drain():
    rows = [row('a',10,15,[[12,5]],5),row('b',18,25,[[19,2],[22,8]],10,'length')]
    r = analysis.summarize(rows,[],[],[],[],dict(begin=10,end=20))
    assert r['output_tokens']==7 and r['completed_turns_in_window']==1 and r['drained_turns']==1
    assert r['completed_turns_per_second']==.1
    assert r['offered_cohort_final_output_lengths']['mean']==7.5
    assert r['offered_cohort_length_cap_fraction']==.5


def test_time_weighted_batch_clips_window_and_excludes_mixed_steps():
    steps = [dict(started_at=9,step_ms=2000,decode_requests=2,prefill_requests=0,prefill_tokens=0),
             dict(started_at=11,step_ms=3000,decode_requests=4,prefill_requests=0,prefill_tokens=0),
             dict(started_at=14,step_ms=2000,decode_requests=8,prefill_requests=1,prefill_tokens=100)]
    r = analysis.summarize([],[],steps,[],[],dict(begin=10,end=20))
    assert r['pure_decode_step_seconds']==4 and r['pure_decode_batch_time_weighted']==3.5
    assert r['pure_decode_batch_seconds']=={'2':1,'4':3}


def test_maintenance_owner_wait_is_not_gpu_idle_and_positions_match_own_request():
    rows = [row('a',10,19,[[18,10]],10)]
    refreshes = [dict(request_id='a-0-1234abcd',refresh_started_at=11,refresh_finished_at=14,
                      selection='pressure',history_begin=h,assembly_uploaded_bytes=20) for h in (99,120)]
    r = analysis.summarize(rows,[],[],refreshes,[],dict(begin=10,end=20))
    assert r['pressure_refresh_positions']==dict(post_prompt=1,final_prompt_block=1,earlier_prompt=0,unmatched=0)
    assert r['refresh_owner_wait_seconds']==6 and r['refresh_uploaded_bytes']==40
    assert r['refresh_count_normalized']['per_100_completed_turns']==200
    assert 'not GPU idle' in r['scope']


def test_rejects_fixed_quota_records():
    r = row('a',10,19,[[18,1]],1);r['generation_mode']='fixed'
    with pytest.raises(ValueError,match='natural workload'):
        analysis.summarize([r],[],[],[],[],dict(begin=10,end=20))


def test_missing_native_step_telemetry_is_explicitly_unavailable():
    r=analysis.summarize([],[],[],[],[],dict(begin=10,end=20))
    assert r['pure_decode_telemetry_available'] is False
    assert r['pure_decode_step_seconds'] is None and r['pure_decode_batch_seconds'] is None
