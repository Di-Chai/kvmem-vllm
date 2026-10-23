"""CPU tests for actual-answer FPS reporting and immutable publication gates."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

import analyze_ruler_fps_answers as analysis

_spec = importlib.util.spec_from_file_location('fps_audit_test_fixture',
    Path(__file__).with_name('test_ruler_fps_answers.py'))
audit_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit_tests)


def save(path,value):
    analysis.campaign.save(path,value)


def write_gate(root):
    required = ('unchanged_frozen_runtime', 'original_inputs_seeds_sampling_verified',
        'native_picker_replay_verified', 'official_scores_recomputed', 'no_post_picker_neighbors',
        'pressure_cumulative_attention', 'gpu_cleanup_verified', 'reference_answers_unchanged',
        'overlap_reference_unchanged')
    save(root/'status.json',dict(status='completed',cases=12))
    save(root/analysis.ARM/'status.json',dict(status='completed',cases=12))
    save(root/'execution-audit.json',dict(status='passed',cases=12,**{key:True for key in required},
        guided=False,graph_alpha=0.,propagation_applied=False,query_score_mode=analysis.ARM))
    save(root/analysis.ARM/'cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))


@pytest.mark.parametrize('problem',['incomplete','audit','dirty','graph'])
def test_final_publication_requires_complete_audited_clean_campaign(tmp_path,problem):
    write_gate(tmp_path)
    if problem == 'incomplete': save(tmp_path/'status.json',dict(status='testing',completed=11))
    elif problem == 'dirty': save(tmp_path/analysis.ARM/'cleanup.json',dict(remaining_owned_groups={},gpu_processes=[42]))
    else:
        audit=analysis.read(tmp_path/'execution-audit.json')
        if problem == 'audit': audit['native_picker_replay_verified']=False
        else: audit['propagation_applied']=True
        save(tmp_path/'execution-audit.json',audit)
    with pytest.raises(ValueError): analysis.completed_gate(tmp_path)
    assert not (tmp_path/'analysis.json').exists()
    assert not (tmp_path/'report.md').exists()


def test_complete_gate_records_original_audit_and_cleanup_hashes(tmp_path):
    write_gate(tmp_path)
    proof=analysis.completed_gate(tmp_path)
    assert proof['execution_audit_sha256'] == analysis.sha(tmp_path/'execution-audit.json')
    assert proof['cleanup_sha256'] == analysis.sha(tmp_path/analysis.ARM/'cleanup.json')


def test_optional_ranks_exclude_mandatory_use_score_mask_and_keep_ties():
    inputs=dict(starts=[0,32,64,96],counts=[32]*4,raw_scores=[99.,2.,2.,1.],
        mandatory_starts=[0],score_mask=[False,True,True,True],transport_mask=[False,True,False,True])
    target=dict(key_token_span=[32,34],token_span=[63,66])
    result=analysis.describe_selection(inputs,[(0,32),(32,32)],target)
    assert result['key']['state']=='full' and result['value']['state']=='partial'
    assert result['value']['missing_positions']==[64,65]
    assert [b['optional_rank'] for b in result['target_blocks']]==[1,1]
    assert result['target_blocks'][1]['transport_allowed'] is False
    assert result['optional_cutoff']==2 and result['optional_cutoff_rank']==1
    assert result['optional_candidates']==3 and result['optional_selected_blocks']==1


def case_fixture(root,*,correct=False,partial_value=False):
    query,graph,inputs,profile=audit_tests.audit_fixture()
    row=dict(id='test-case',seed=42,served_sha256='original-input',served_query=[192,194],
        served_input_tokens=194,task='niah_multikey_2',outputs=['42'])
    fixture=dict(sampling=dict(max_tokens=8192),starts_in_thinking=False)
    query.update(selection='query',query_span=row['served_query'],guided_query_tokens=0,
        refresh_started_at=10.,refresh_finished_at=15.)
    query['query_key_fps_scoring'].update(cache_build_seconds=2.,scoring_seconds=1.,seconds=3.1,
        cache_bytes=4096,cache_capacity_bytes=8192,cache_hit_blocks=0,cache_rebuilt_blocks=6)
    graph['phase']='query'
    pressure=dict(selection='pressure',guided_query_tokens=0)
    pressure_graph=dict(phase='pressure',seed_source='prefill',pressure_policy='attention',propagation_applied=False)
    target_begin=query['selected_history_ranges'][0][0]
    target=dict(key_token_span=[target_begin+1,target_begin+3],token_span=[target_begin+4,target_begin+6])
    if partial_value:
        selected={start for start,_ in query['selected_history_ranges']}
        boundary=next(start+32 for start in selected if start+32 in inputs['starts'] and start+32 not in selected)
        target=dict(key_token_span=[boundary-10,boundary-7],token_span=[boundary-6,boundary+1])
    response=dict(text='42' if correct else 'wrong',output_ids=[1,2],seconds=10.,finish_reason='stop')
    result=dict(official_case_score=100. if correct else 0.,answer=response['text'],thinking_complete=None,
        completion_tokens=2,seconds=10.,finish_reason='stop',pressure_count=1,value_covered=not partial_value,
        selection_audit=analysis.campaign.audit_selection(query,graph,inputs,profile))
    save(root/analysis.ARM/'profile.json',profile)
    request=dict(input_sha256=row['served_sha256'],sampling=dict(fixture['sampling'],seed=row['seed']),
        query_span=row['served_query'],profile_sha256=analysis.sha(root/analysis.ARM/'profile.json'))
    evidence=dict(bounded=[pressure,query],graph=[pressure_graph,graph],query_inputs=[inputs])
    case=root/analysis.ARM/'cases'/row['id']
    for name,item in [('request.json',request),('response.json',response),('result.json',result),('evidence.json',evidence)]:
        save(case/name,item)
    (case/'query-inputs.jsonl').write_text(json.dumps(inputs)+'\n')
    save(case/'hashes.json',{name:analysis.sha(case/name) for name in (*analysis.campaign.CASE_FILES,'query-inputs.jsonl')})
    snapshots={}
    for arm in analysis.REFERENCES:
        old_case=root/'reference'/arm/row['id']
        old_query=deepcopy(query)
        old_query['query_scoring_method']='content_token_softmax_page_sum' if arm=='token_sum' else arm
        if arm=='overlap_16_8': old_query['query_window_scoring']=dict(seconds=4.)
        old_graph=dict(phase='query',alpha=0.)
        old_inputs=deepcopy(inputs);old_inputs['seed_source']=arm
        old_evidence=dict(bounded=[old_query],graph=[old_graph],query_inputs=[old_inputs])
        old_response=dict(response,text='42')
        old_result=dict(result,answer='42',official_case_score=100.)
        for name,item in [('request.json',request),('response.json',old_response),
                ('result.json',old_result),('evidence.json',old_evidence)]: save(old_case/name,item)
        snapshots[arm]={row['id']:dict(root=str(old_case),response=old_response,result=old_result,
            sha256={name:analysis.sha(old_case/name) for name in ('response.json','result.json')})}
    scorer=({'niah':{'metric_fn':lambda answers,gold:100. if answers[0] in gold[0] else 0.}},lambda text,cfg:text)
    return row,target,fixture,profile,snapshots,scorer


def test_actual_wrong_answer_is_not_inferred_correct_from_full_retrieval(tmp_path):
    args=case_fixture(tmp_path,correct=False)
    row=analysis.analyze_case(tmp_path,*args)
    assert row['selection']['value']['state']=='full' and row['selection']['key']['state']=='full'
    assert row['correct'] is False and row['answer']=='wrong'
    assert row['outcome']=='full_value_answer_incorrect'
    assert row['selection_audit']['propagation_applied'] is False
    summary=analysis.summarize([row])
    assert summary['full_value_but_wrong']==['test-case']
    assert summary['arms']['key_fps_4']['correct']==0
    assert summary['arms']['key_fps_4']['full_value_coverage']==1
    assert summary['cache_build_seconds']['median']==2
    assert summary['scoring_seconds']['median']==1
    assert summary['answer_changes']['overlap_16_8']['regressed']==['test-case']
    report=analysis.format_report(dict(cases=[row],summary=summary,native_full_context=None,fixed_capture=None))
    assert '尚未优化' in report and '默认关闭' in report and '并非纯 GPU 内核时间' in report
    assert 'wrong' in report


def test_correct_answer_with_six_of_seven_value_tokens_is_not_full_retrieval(tmp_path):
    args=case_fixture(tmp_path,correct=True,partial_value=True)
    row=analysis.analyze_case(tmp_path,*args)
    value=row['selection']['value']
    assert row['correct'] is True and row['official_case_score']==100
    assert value['state']=='partial' and value['selected_tokens']==6 and value['tokens']==7
    assert value['missing_positions']==[128]
    assert row['outcome']=='partial_value_answer_correct'
    summary=analysis.summarize([row])
    assert summary['arms']['key_fps_4']['correct']==1
    assert summary['arms']['key_fps_4']['full_value_coverage']==0
    assert summary['correct_without_full_value']==['test-case']
    assert not summary['full_value_but_wrong']
    report=analysis.format_report(dict(cases=[row],summary=summary,native_full_context=None,fixed_capture=None))
    assert '目标值未完整召回但答对：case' in report
    assert 'case 仅保留 6/7 个目标值 token' in report
    assert '计入答对题数，不计入完整召回题数' in report
    assert '| 每 block 4 个原始 Key（FPS） | 1/1 | 1/1 | 0/1 | 100.00 |' in report


def test_case_analysis_rejects_fps_method_relabeling_even_with_updated_file_hashes(tmp_path):
    args=case_fixture(tmp_path)
    case=tmp_path/analysis.ARM/'cases/test-case'
    evidence=analysis.read(case/'evidence.json')
    evidence['bounded'][1]['query_scoring_method']='overlap_16_8'
    save(case/'evidence.json',evidence)
    hashes=analysis.read(case/'hashes.json');hashes['evidence.json']=analysis.sha(case/'evidence.json')
    save(case/'hashes.json',hashes)
    with pytest.raises(ValueError,match='FPS public Query'):
        analysis.analyze_case(tmp_path,*args)


def test_saved_raw_response_change_is_detected_before_scoring(tmp_path):
    args=case_fixture(tmp_path)
    save(tmp_path/analysis.ARM/'cases/test-case/response.json',dict(text='42'))
    with pytest.raises(ValueError,match='Immutable evidence changed'):
        analysis.analyze_case(tmp_path,*args)


def test_native_reference_rechecks_raw_sampling_and_original_answer_hashes(tmp_path):
    fixture=dict(rows=[dict(id='a',task='niah_multikey_2',outputs=['42'],seed=7,
        served_sha256='input-digest',served_input_tokens=16)],sampling=dict(max_tokens=8192),starts_in_thinking=False)
    native=tmp_path/'native';overlap=tmp_path/'overlap';case=native/'run/answers/cases/a'
    request=dict(sampling=dict(max_tokens=8192,seed=7),input_tokens=16,input_sha256='input-digest')
    response=dict(text='42',output_ids=[1],seconds=3.,finish_reason='stop',usage=dict(prompt_tokens=16))
    result=dict(answer='42',official_case_score=100.,thinking_complete=None,completion_tokens=1,seconds=3.)
    for name,item in [('request.json',request),('response.json',response),('result.json',result)]: save(case/name,item)
    save(native/'status.json',dict(status='completed'))
    save(native/'run/cleanup.json',dict(remaining_owned_groups={},gpu_processes=[]))
    comparison=dict(same_inputs_seeds_sampling=True,cases=12,native_root=str(native),native_actual_concurrency=4,
        rows=[dict(id='a',native_score=100.,input_sha256='input-digest',seed=7,
            evidence_sha256={str(case/name):analysis.sha(case/name) for name in ('request.json','response.json','result.json')})])
    save(overlap/'native-context-comparison.json',comparison)
    scorer=({'niah':{'metric_fn':lambda answers,gold:100. if answers[0] in gold[0] else 0.}},lambda text,cfg:text)
    verified=analysis.native_reference(fixture,overlap,scorer)
    assert verified['correct']==1 and verified['actual_concurrency']==4
    assert verified['current_actual_concurrency']==1
    save(case/'response.json',dict(response,text='changed'))
    with pytest.raises(ValueError,match='source hash changed'):
        analysis.native_reference(fixture,overlap,scorer)


def test_inspection_is_read_only_and_final_artifacts_never_overwrite(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(analysis,'build_analysis',lambda *a,**k:dict(status='inspection',cases=[],summary=None))
    analysis.main(['--root',str(tmp_path),'--inspect-completed'])
    assert json.loads(capsys.readouterr().out)['status']=='inspection'
    assert not (tmp_path/'analysis.json').exists() and not (tmp_path/'report.md').exists()
    (tmp_path/'analysis.json').write_text('keep original')
    with pytest.raises(ValueError,match='do not overwrite'): analysis.main(['--root',str(tmp_path)])
    assert (tmp_path/'analysis.json').read_text()=='keep original'
