import importlib.util
import json
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('agent_analysis',Path(__file__).parents[1]/'scripts/analyze_agent_scheduling.py')
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def write_rows(path,values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values))


def restore(rid):
    return dict(request_id=rid,started=1.,finished=5.,uploaded_bytes=100,model_bytes=10,
        active_tokens=10,history_tokens=20,host_prepare_seconds=1.5,cancelled=False)


def test_restore_wait_and_service_partition_does_not_double_count_host_gather():
    result=analysis.restore_groups([restore('r')],[dict(request_id='r',stage='restore_submitted',at=2.)])
    row=result['requests'][0]
    assert row['pre_submit_seconds']==1. and row['service_seconds']==3.
    assert row['total_seconds']==row['pre_submit_seconds']+row['service_seconds']==4.
    assert row['host_prepare_seconds']==1.5  # Inside service, never added again.


@pytest.mark.parametrize('stages',[[],[dict(request_id='other',stage='restore_submitted',at=2.)],
    [dict(request_id='r',stage='restore_submitted',at=2.)]*2,
    [dict(request_id='r',stage='restore_submitted',at=6.)]])
def test_missing_ambiguous_or_misaligned_restore_evidence_fails(stages):
    with pytest.raises(ValueError):analysis.restore_groups([restore('r')],stages)


def test_prefill_rate_uses_sampled_pure_model_work_not_mixed_time_or_total_input():
    rows=[dict(decode_requests=0,prefill_requests=1,prefill_tokens=2048,model_ms=100.,step_ms=200.),
          dict(decode_requests=1,prefill_requests=1,prefill_tokens=2048,model_ms=1000.,step_ms=2000.),
          dict(decode_requests=0,prefill_requests=0,prefill_tokens=0,model_ms=1.,step_ms=2.)]
    result=analysis.step_groups(rows)
    assert result['groups']['prefill_only']['tokens_per_sampled_model_second']==20480.
    assert result['groups']['prefill_only']['tokens_per_sampled_step_second']==10240.
    assert result['groups']['mixed']['samples']==1
    assert result['groups']['maintenance_only']['samples']==1
    assert 'decode_only' not in result['groups']
    assert all(row['decode_context_bucket'] is None for row in result['matched_composition_bins'])


def test_same_size_query_and_ordinary_prefill_are_not_merged_in_cost_bins():
    base=dict(decode_requests=4,prefill_requests=2,prefill_tokens=4096,
              model_ms=100.,step_ms=110.,maintenance_host_ms=3.,
              decode_context_bucket=65536,prefill_context_bucket=32768,
              unknown_prefill_tokens=0)
    ordinary=dict(base,ordinary_prefill_tokens=4096,atomic_prefill_tokens=0)
    query=dict(base,ordinary_prefill_tokens=0,atomic_prefill_tokens=4096,model_ms=500.)
    report=analysis.step_groups([ordinary,query])
    bins={r['prefill_origin']:r for r in report['matched_composition_bins']}
    assert set(bins)=={'ordinary','atomic'}
    assert bins['ordinary']['model_ms']['p50']==100.
    assert bins['atomic']['model_ms']['p50']==500.
    groups={r['prefill_origin']:r for r in report['prefill_origin_groups']}
    assert groups['ordinary']['prefill_tokens']['maximum']==4096
    assert groups['ordinary']['sampled_model_seconds']==.1
    assert groups['ordinary']['maintenance_host_ms']['p50']==3.


def test_missing_or_partial_origin_evidence_stays_unknown_even_with_legacy_flag():
    base=dict(prefill_tokens=2048,ordinary_prefill_only=True)
    assert analysis.prefill_origin(base)=='unknown'
    assert analysis.prefill_origin(dict(base,ordinary_prefill_tokens=2048))=='unknown'
    assert analysis.prefill_origin(dict(base,ordinary_prefill_tokens=1024,
        atomic_prefill_tokens=0,unknown_prefill_tokens=1024))=='unknown'
    assert analysis.prefill_origin(dict(base,ordinary_prefill_tokens=1024,
        atomic_prefill_tokens=1024,unknown_prefill_tokens=0))=='ordinary_and_atomic'


@pytest.mark.parametrize('ordinary,atomic,unknown',[(2048,32,0),(-1,2049,0),(True,2047,0)])
def test_invalid_origin_partition_cannot_support_prefill_tuning(ordinary,atomic,unknown):
    with pytest.raises(ValueError,match='partition'):
        analysis.prefill_origin(dict(prefill_tokens=2048,ordinary_prefill_tokens=ordinary,
            atomic_prefill_tokens=atomic,unknown_prefill_tokens=unknown))


def test_case_mapping_excludes_warmup_even_inside_same_time_window(tmp_path):
    ids=['cmpl-test-0-1234abcd','cmpl-test-other-0-1234abcd','cmpl-warmup-0-1234abcd']
    write_rows(tmp_path/'async_restore.jsonl',[restore(rid) for rid in ids])
    write_rows(tmp_path/'async_refresh.jsonl',[dict(request_id=rid,stage='restore_submitted',at=2.) for rid in ids])
    trace=dict(response_id='cmpl-test',begin=.5,end=7.,timeline=[[6.,2],[6.5,1]],output_ids=[1,2,3])
    report=analysis.analyze(tmp_path,traces=[trace])
    assert [r['request_id'] for r in report['restores']['requests']]==ids[:1]
    assert report['clients']['accepted_output_tokens_per_second']==pytest.approx(3/6.5)
    assert report['clients']['output_event_gap_seconds']['p50']==.5
    assert report['policy_waits'] is None  # Absent telemetry is not zero wait.


def test_wait_reasons_are_overlapping_request_seconds_with_clipped_window():
    rows=[dict(at=0.,candidates=[dict(reason='cpu_history_budget')]*2),
          dict(at=5.,candidates=[dict(reason='completion_fence'),dict(reason=None)]),
          dict(at=9.,candidates=[])]
    result=analysis.policy_waits(rows,2.,7.)
    assert result['request_seconds']==dict(cpu_history_budget=6.,completion_fence=2.)


def test_restore_overlap_is_start_timestamp_correlation_with_context_left_unknown():
    steps=[dict(started_at=at,decode_requests=2,prefill_tokens=0,model_ms=ms)
           for at,ms in [(1.,20.),(2.,100.),(3.,200.),(5.,30.)]]
    result=analysis.restore_overlap_bins(steps,[restore('r')],
        [dict(request_id='r',stage='restore_submitted',at=2.)])
    groups={r['restore_service_pending_at_step_start']:r for r in result['bins']}
    assert groups[True]['model_ms']['p50']==150.
    assert groups[False]['model_ms']['p50']==25.
    assert all(r['decode_context_bucket'] is None for r in result['bins'])


def test_cost_coverage_excludes_decode_zero_cost_warmup_and_outside_window():
    def candidate(rid,phase,cost,reason=None):
        return dict(request_id=rid,phase=phase,estimated_ms=cost,reason=reason,waiting_seconds=.3)
    entries=[candidate('a','resident_admission',None),candidate('b','prefill',10.),
             candidate('d','decode',0.),candidate('w','cold_admission',20.),
             candidate('f','cpu_admission',None,'completion_fence')]
    rows=[dict(at=0.,candidates=entries),dict(at=3.,candidates=entries,
           maintenance_costs={'a':None,'b':10.,'w':5.})]
    result=analysis.policy_cost_coverage(rows,2.,4.,{'a','b','d','f'})
    assert result['ready_candidate_observations']==2
    assert result['ready_candidates_with_estimates']==1
    assert result['competing_ready_snapshots']==1
    assert result['competing_all_costs_known_snapshots']==0
    assert result['maintenance_cost_observations']==2 and result['maintenance_costs_known']==1
    assert set(result['by_phase'])=={'resident_admission','prefill'}
    assert analysis.policy_cost_coverage([],2.,4.) is None


def test_policy_case_identity_includes_cold_first_turn_and_excludes_similar_ids(tmp_path):
    def candidate(rid):
        return dict(request_id=rid,phase='cold_admission',estimated_ms=None,
                    reason='cpu_history_budget',waiting_seconds=2.)
    write_rows(tmp_path/'offline_decisions.jsonl',[dict(at=1.,candidates=[
        candidate('cmpl-test-0-1234abcd'),candidate('cmpl-test-other-0-1234abcd')])])
    trace=dict(response_id='cmpl-test',begin=1.,end=4.,timeline=[[2.,1],[3.,1]],output_ids=[1,2])
    report=analysis.analyze(tmp_path,traces=[trace])
    assert report['policy_waits']['request_seconds']=={'cpu_history_budget':3.}
    assert report['policy_cost_coverage']['ready_candidate_observations']==0
