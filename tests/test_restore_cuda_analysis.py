"""Synthetic interval/clock failures that would invalidate a CUDA gap diagnosis."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('restore_cuda_analysis',
    Path(__file__).resolve().parents[1]/'scripts/analyze_restore_cuda.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_nested_parallel_activity_uses_running_max_not_previous_end():
    # A LAG(end) algorithm incorrectly opens a gap between nested [2,3] and [4,5].
    busy = analysis.merge([(0,10),(2,3),(4,5),(9,12),(12,14),(18,22)],1,20)
    assert busy == [(1,14),(18,20)]
    assert analysis.complement(busy,1,20) == [(14,18)]
    assert analysis.duration(busy)+analysis.duration(analysis.complement(busy,1,20)) == 19


def test_overlap_is_clipped_unioned_and_does_not_add_nested_host_stages():
    out = analysis.describe_device([(0,2),(8,10)],[(1,7),(3,5)],
        {('prepare','resident','outer'):[(1,7)],('prepare','resident','inner'):[(3,5)]},0,10,20)
    assert out['gap_statistics']['total_ns'] == 6
    assert out['restore_host_union_ns'] == 6
    assert out['restore_direct_overlap_upper_bound_ns'] == 5
    assert sum(r['gap_overlap_ns'] for r in out['stage_overlap_totals']) == 7  # Deliberately non-additive.
    assert out['gaps_without_direct_restore_overlap_ns'] == 1


def test_capture_edges_are_not_presented_as_complete_interior_gaps():
    out = analysis.describe_device([(2,3),(5,8)],[(0,10)],{},0,10,20)
    assert out['gap_statistics']['total_ns'] == 6
    assert out['interior_gap_statistics']['total_ns'] == 2
    assert out['boundary_gap_ns'] == 4
    assert sum(r['boundary'] for r in out['longest_gaps']) == 2


def test_integer_epoch_mapping_preserves_nanoseconds_and_nonzero_report_start():
    origin = 1792158694000000000
    offset = 1787444242587402445
    anchor = dict(source='engine',mono_ns=4714451875500001,
                  wall_ns=4714451875500001+offset)
    clock = analysis.ClockMap(origin,[anchor])
    expected = 4714451875500001+offset-origin
    assert clock.report('4714451.875500001') == expected
    assert clock.mono(expected) == anchor['mono_ns']
    # A nonzero ANALYSIS_DETAILS.startTime bounds analysis; it is not another origin to subtract.
    assert clock.report('4714451.875500002')-clock.report('4714451.875500001') == 1
    assert clock.summary()['all_sources']['maximum_abs_residual_ns'] == 0
    second = dict(source='engine',mono_ns=anchor['mono_ns']+100,
                  wall_ns=anchor['wall_ns']+102)
    assert analysis.ClockMap(origin,[anchor,second]).offset_ns == offset+1


def test_raw_analysis_start_is_not_used_as_capture_relative_activity_origin():
    export = dict(EXPORT_PARAM_TIME_NORMALIZE='false',EXPORT_PARAM_TIME_SHIFT='0',
                  EXPORT_CONFIG_TIME_SHIFT='0',EXPORT_PARAM_TIMES_TO_EXPORT='')
    meta = {'META_DATA_EXPORT':dict(rows=[dict(name=k,value=v) for k,v in export.items()]),
            'ANALYSIS_DETAILS':dict(rows=[dict(startTime=470_000_000_000,stopTime=515_000_000_000,
                                               duration=45_000_000_000)])}
    assert analysis.capture_window(meta) == (0,45_000_000_000)
    meta['META_DATA_EXPORT']['rows'][0]['value'] = 'true'
    with pytest.raises(ValueError,match='unshifted'):
        analysis.capture_window(meta)


def test_clock_inconsistent_sources_remain_visible_instead_of_hidden_by_fit():
    rows = [dict(source='wrapper',mono_ns=0,wall_ns=100),
            dict(source='wrapper',mono_ns=1000,wall_ns=1100),
            dict(source='engine',mono_ns=1000,wall_ns=1300)]
    result = analysis.ClockMap(100,rows).summary()
    assert result['wall_minus_monotonic_ns'] == 100
    assert result['sources']['engine']['maximum_abs_residual_ns'] == 200
    with pytest.raises(ValueError):
        analysis.ns('NaN')


def test_offset_sensitivity_changes_host_gap_intersection():
    out = analysis.describe_device([(0,2_000_000),(4_000_000,6_000_000)],
                                   [(1_000_000,3_000_000)],{},0,6_000_000,20)
    shifts = {r['host_clock_shift_ns']:r['direct_overlap_ns'] for r in out['clock_sensitivity']}
    assert shifts[-1_000_000] == 0
    assert shifts[0] == 1_000_000
    assert shifts[1_000_000] == 2_000_000


def test_single_pass_snapshot_evidence_does_not_extend_ready_across_gap(tmp_path):
    path = tmp_path/'offline.jsonl'
    rows = [dict(at=t,step=i,candidates=[dict(request_id='r',phase='decode',reason=reason)],
                 scheduled={'r':1} if reason is None else {})
            for i,(t,reason) in enumerate([(0,None),(2,'blocked'),(4,None),(7,None),(9,'blocked')])]
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    clock = analysis.ClockMap(0,[dict(source='test',mono_ns=0,wall_ns=0)])
    gaps = [dict(start_ns=analysis.ns(3),end_ns=analysis.ns(6))]
    result = analysis.decision_evidence(path,gaps,clock)
    proof = gaps[0]['scheduler_snapshots']
    assert result['lines_scanned'] == 4
    assert proof['before']['at'] == 2
    assert proof['before']['ready_candidate_count'] == 0
    assert proof['inside_snapshot_count'] == 1
    assert proof['first_inside']['at'] == 4
    assert proof['after']['at'] == 7
    assert 'ready_duration' not in proof


def test_bad_nested_restore_interval_is_rejected(tmp_path):
    path = tmp_path/'restore.jsonl'
    stages = [dict(name='restore_main_thread_total',started_at=1,finished_at=2,thread_cpu_seconds=.5),
              dict(name='bad',started_at=.5,finished_at=1.5,thread_cpu_seconds=.2)]
    path.write_text(json.dumps(dict(request_id='r',phase='prepare',timing=dict(stages=stages)))+'\n')
    clock = analysis.ClockMap(0,[dict(source='test',mono_ns=0,wall_ns=0)])
    with pytest.raises(ValueError,match='outside'):
        analysis.restore_intervals(path,clock,0,analysis.ns(3))


def test_forward_bucket_keeps_postprocessing_and_leaves_unobserved_time_explicit():
    gap = dict(start_ns=analysis.ns(1),end_ns=analysis.ns(5),model_step_intersections=[
        dict(started_at=11,model_submitted_at=12,submitted_at=13),
        dict(started_at=14,model_submitted_at=15,submitted_at=16)])
    result = analysis.host_gap_classification(gap,analysis.ns(10))
    assert result['maintenance_host_ns'] == analysis.ns(2)
    assert result['forward_sampling_mtp_capture_host_ns'] == analysis.ns(1)
    assert result['outside_measured_step_ns'] == analysis.ns(1)
