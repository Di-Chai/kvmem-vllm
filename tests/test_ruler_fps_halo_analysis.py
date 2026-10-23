"""CPU-only final publication, atomic-group coverage and honest comparison tests."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

import analyze_ruler_fps_halo_answers as analysis


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


ROOT=Path(__file__).resolve().parents[1]
halo_tests=load(ROOT/'tests/test_ruler_fps_halo_answers.py','halo_analysis_fixture')
fps_tests=load(ROOT/'tests/test_ruler_fps_analysis.py','halo_old_answer_fixture')
save=analysis.campaign.save


def write_gate(root):
    required=('unchanged_frozen_runtime','original_inputs_seeds_sampling_verified','native_picker_replay_verified',
        'native_group_picker_replay_verified','independent_group_geometry_verified','official_scores_recomputed',
        'no_post_picker_neighbors','before_picker_expansion','pressure_cumulative_attention',
        'gpu_cleanup_verified','reference_answers_unchanged')
    for name in ('status.json',analysis.ARM+'/status.json'):save(root/name,dict(status='completed',cases=12))
    save(root/'execution-audit.json',dict(status='passed',cases=12,**{key:True for key in required},guided=False,
        graph_alpha=0.,propagation_applied=False,query_score_mode='key_fps_4',query_fps_halo_blocks=1))
    save(root/analysis.ARM/'cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))


@pytest.mark.parametrize('problem',['running','group_audit','budget_audit','cleanup'])
def test_report_requires_twelve_cases_atomic_group_audit_and_cleanup(tmp_path,problem):
    write_gate(tmp_path)
    if problem=='running':save(tmp_path/'status.json',dict(status='testing',completed=11))
    elif problem=='cleanup':save(tmp_path/analysis.ARM/'cleanup.json',dict(remaining_owned_groups={1:'live'},gpu_processes=[]))
    else:
        audit=analysis.read(tmp_path/'execution-audit.json')
        audit['native_group_picker_replay_verified' if problem=='group_audit' else 'no_post_picker_neighbors']=False
        save(tmp_path/'execution-audit.json',audit)
    with pytest.raises(ValueError):analysis.completed_gate(tmp_path)
    assert not (tmp_path/'analysis.json').exists()


def case_fixture(tmp_path):
    source=tmp_path/'old-fps';root=tmp_path/'halo'
    row,target,fixture,old_profile,snapshots,scorer=fps_tests.case_fixture(source,correct=False,partial_value=True)
    query,graph,inputs,profile=halo_tests.audit_fixture()
    query.update(selection='query',query_span=row['served_query'],guided_query_tokens=0,
        refresh_started_at=10.,refresh_finished_at=15.)
    query['query_key_fps_scoring'].update(cache_build_seconds=1.,scoring_seconds=.5,seconds=1.6,
        cache_bytes=4096,cache_capacity_bytes=8192,cache_hit_blocks=0,cache_rebuilt_blocks=6)
    graph['phase']='query'
    evidence=dict(bounded=[dict(selection='pressure',guided_query_tokens=0),query],
        graph=[dict(phase='pressure',seed_source='prefill',pressure_policy='attention',propagation_applied=False),graph],
        query_inputs=[inputs])
    response=dict(text='42',output_ids=[1,2],seconds=9.,finish_reason='stop')
    result=dict(official_case_score=100.,answer='42',thinking_complete=None,completion_tokens=2,seconds=9.,
        finish_reason='stop',value_covered=True,pressure_count=1,
        selection_audit=analysis.campaign.audit_selection(query,graph,inputs,profile))
    save(root/analysis.ARM/'profile.json',profile)
    request=dict(sampling=dict(fixture['sampling'],seed=row['seed']),query_span=row['served_query'],
        input_sha256=row['served_sha256'],profile_sha256=analysis.sha(root/analysis.ARM/'profile.json'))
    case=root/analysis.ARM/'cases'/row['id']
    for name,item in [('request.json',request),('response.json',response),('result.json',result),('evidence.json',evidence)]:save(case/name,item)
    (case/'query-inputs.jsonl').write_text(json.dumps(inputs)+'\n')
    save(case/'hashes.json',{name:analysis.sha(case/name) for name in (*analysis.campaign.CASE_FILES,'query-inputs.jsonl')})
    return root,row,target,fixture,profile,source,fixture,old_profile,snapshots,scorer


def test_group_restores_boundary_value_with_fixed_budget_and_reports_displaced_page(tmp_path):
    row=analysis.analyze_case(*case_fixture(tmp_path))
    assert row['correct'] and row['selection']['actual']['value']['state']=='full'
    assert row['controls']['key_fps_4']['correct'] is False
    assert row['controls']['key_fps_4']['value']['selected_tokens']==6
    assert row['selection']['same_seed_block_picker']['value']['selected_tokens']==6
    assert row['selection']['actual']['value']['selected_tokens']==7
    assert row['selection']['added_value_blocks']==[128]
    assert row['historical_fps_change']['added_starts']==[128]
    assert row['historical_fps_change']['displaced_starts']==[64]
    assert row['historical_fps_change']['seeds']['raw_seeds_exactly_equal']
    assert row['selection_audit']['charged_block_tokens']==96
    support=next(block for block in row['selection']['actual']['target_blocks'] if block['start']==128)
    assert [group['anchor'] for group in support['supporting_covered_groups']]==[128,160]
    summary=analysis.summarize([row])
    assert summary['same_seed_block_picker_full_value']==0 and summary['actual_group_full_value']==1
    assert summary['answer_changes']['key_fps_4']['improved']==['test-case']
    report=analysis.format_report(dict(cases=[row],summary=summary,native_full_context=None,query_fps_optimized=False))
    assert '不是 picker 显式选择该 anchor 的日志' in report
    assert '没有生成另一组答案' in report
    assert '原K预算未扩张' in report and '纯GPU内核' in report
    assert '6/7→7/7' in report


def test_answer_correctness_is_still_separate_from_group_value_coverage(tmp_path):
    row=analysis.analyze_case(*case_fixture(tmp_path))
    row['selection']['actual']['value']=dict(state='partial',selected_tokens=6,tokens=7,missing_positions=[128])
    summary=analysis.summarize([row])
    assert summary['arms'][analysis.ARM]['correct']==1
    assert summary['arms'][analysis.ARM]['full_value_coverage']==0
    assert summary['correct_without_full_value']==['test-case']
    report=analysis.format_report(dict(cases=[row],summary=summary,native_full_context=None,query_fps_optimized=False))
    assert '计入答对，不计入完整召回' in report


def test_historical_seed_drift_is_reported_not_hidden():
    values=dict(starts=[0,32],counts=[32,32],score_mask=[True,True],transport_mask=[True,True],
        mandatory_starts=[],budget=32,raw_scores=[.1,.9])
    other=deepcopy(values);other['raw_scores'][1]=.8
    result=analysis.seed_comparison(values,other)
    assert result['index_and_masks_equal'] and result['raw_seeds_exactly_equal'] is False
    assert result['max_abs_difference']==pytest.approx(.1)
    other['score_mask'][0]=False
    assert analysis.seed_comparison(values,other)['max_abs_difference'] is None


def test_inspection_is_readonly_and_final_artifacts_cannot_be_overwritten(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(analysis,'build_analysis',lambda *a,**k:dict(status='inspection',cases=[],summary=None))
    analysis.main(['--root',str(tmp_path),'--inspect-completed'])
    assert json.loads(capsys.readouterr().out)['status']=='inspection'
    assert not (tmp_path/'analysis.json').exists()
    (tmp_path/'report.md').write_text('immutable report')
    with pytest.raises(ValueError,match='do not overwrite'):analysis.main(['--root',str(tmp_path)])
    assert (tmp_path/'report.md').read_text()=='immutable report'
