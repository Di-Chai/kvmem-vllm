"""The campaign scorer must not count an answer trapped in unfinished thought."""
import pytest

import recall_comparison_protocol as p
import ruler_protocol as ruler
import run_layerwise_query_diagnostic as driver


@pytest.mark.parametrize('text,expected,incomplete', [
    ('answer is still in unfinished thought', 0., 13),
    ('unrelated reasoning</think>answer', 100., 0),
])
def test_full_cohort_analysis_honors_thinking_and_long_caps(tmp_path, monkeypatch, text, expected, incomplete):
    monkeypatch.setattr(driver, 'verify', lambda root: None)
    metric = lambda predictions, refs: round(100*sum(any(v in prediction for v in values)
        for prediction, values in zip(predictions, refs))/len(predictions), 2)
    scorer = ({family:{'metric_fn':metric} for family in set(ruler.TASK_FAMILIES.values())},
              lambda text, config:text)
    monkeypatch.setattr(ruler, 'load_scorer', lambda root:scorer)
    rows = [dict(id=task+'-0000', task=task, index=0, seed=42, input='public question',
                 outputs=['answer'], served_query=[0,1])
            for task in ruler.TASK_FAMILIES]
    source = tmp_path/'source.json';p.save(source, {})
    fixture = dict(rows=rows, samples_per_task=1, starts_in_thinking=True,
        answer_prefix_policy='none', task_max_tokens=None,
        sampling=dict(max_tokens=8192, thinking_token_budget=4096),
        diagnostic_scope='test', diagnostic_limits=[], source_fixture=str(source),
        source_fixture_sha256=p.sha(source))
    for row in rows:
        path=tmp_path/'inputs'/(row['id']+'.i32');path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'test');row['served_sha256']=p.sha(path)
    p.save(tmp_path/'fixture.json', fixture)
    for row in rows:
        case=tmp_path/'formal/sequential/cases'/row['id']
        response=dict(text=text, response_id=row['id'])
        evidence=dict(layers=[dict(charged_tokens=36864,selected_tokens=36864) for _ in range(16)],
                      events=[dict(event='armed',actual_method='mean_k_sequential_layerwise')])
        p.save(case/'response.json', response);p.save(case/'evidence.json', evidence)
        p.save(case/'request.json', dict(sampling=p.case_sampling(fixture,row),
            input_sha256=row['served_sha256'],query_span=row['served_query'],
            fixture_sha256=p.sha(tmp_path/'fixture.json')))
        p.save(case/'result.json', dict(id=row['id'],task=row['task'],index=0,
            response_sha256=p.sha(case/'response.json'),evidence_sha256=p.sha(case/'evidence.json'),
            finish_reason='length',seconds=1.,completion_tokens=20,max_tokens=8192,
            **p.score_case(row,response,scorer,starts_in_thinking=True)))
    driver.analyze_balanced26(tmp_path)
    result=p.read(tmp_path/'analysis/comparison.json')
    assert result['macro_score']==expected
    assert result['thinking'] is True and result['answer_prefix']=='none'
    assert result['thinking_incomplete']==incomplete
    assert all(row['max_tokens']==8192 for row in result['per_task'].values())
