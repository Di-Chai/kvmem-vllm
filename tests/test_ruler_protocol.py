"""RULER cohort integrity, public queries and thinking-safe offline scoring."""
from copy import deepcopy

import pytest

from scripts import ruler_protocol as ruler


def cohort(count=1):
    return [dict(task=task, index=i, input='context\nQuestion: Public question?',
                 outputs=['alpha', 'beta'])
            for task in ruler.TASK_FAMILIES for i in range(count)]


def scoring_fixture():
    # A deliberately small scorer double exposes batch size and ordering.
    def coverage(preds, refs):
        return round(100 * sum(sum(r.lower() in p.lower() for r in rs) / len(rs)
                               for p, rs in zip(preds, refs)) / len(preds), 2)

    def alias(preds, refs):
        return round(100 * sum(any(r.lower() in p.lower() for r in rs)
                               for p, rs in zip(preds, refs)) / len(preds), 2)

    return ({family: {'metric_fn': alias if family == 'qa' else coverage}
             for family in ruler.TASK_FAMILIES.values()}, lambda text, _: text.strip())


def test_sample_count_is_per_task_not_total():
    assert ruler.validate_cohort(cohort(20), 20) == dict(
        tasks=13, samples_per_task=20, cases=260, official_default_sample_count=False)
    assert ruler.validate_cohort(cohort(500), 500)['cases'] == 6500


@pytest.mark.parametrize('damage', ['missing', 'extra', 'duplicate', 'no_gold', 'empty_gold'])
def test_reject_incomplete_or_invalid_cohort(damage):
    rows = cohort(2)
    if damage == 'missing':
        rows.pop()
    elif damage == 'extra':
        rows.append(dict(rows[-1], task='unknown'))
    elif damage == 'duplicate':
        rows[1] = deepcopy(rows[0])
    elif damage == 'no_gold':
        rows[0]['outputs'] = []
    else:
        rows[0]['outputs'] = ['']
    with pytest.raises(ValueError):
        ruler.validate_cohort(rows, 2)


@pytest.mark.parametrize('count', [0, -1, True, 2.5])
def test_sample_count_must_be_positive_integer(count):
    with pytest.raises(ValueError):
        ruler.validate_cohort([], count)


@pytest.mark.parametrize('task', ruler.TASK_FAMILIES)
def test_query_is_last_public_question_without_reference_answers(task):
    marker = ('What are all the special magic numbers for KEY mentioned in the provided text?'
              if task.startswith('niah') else 'Question: What is the public question?')
    prompt = 'History\n' + marker + '\nMore history\n' + marker
    result = ruler.public_query(task, prompt)
    assert result['text'] == marker
    assert prompt[slice(*result['char_span'])] == marker
    assert result['char_span'][0] > len(prompt) / 2


def test_missing_question_fails_instead_of_replaying_all_history():
    with pytest.raises(ValueError):
        ruler.public_query('qa_1', 'history without a question')


def test_upstream_singular_needle_question_is_supported():
    question = 'What is the special magic number for KEY mentioned in the provided text?'
    assert ruler.public_query('niah_single_1', 'history\n' + question)['text'] == question


def test_thinking_only_gold_is_not_an_answer():
    rows = cohort()
    predictions = {(r['task'], r['index']): 'alpha beta</think>wrong' for r in rows}
    result = ruler.score_suite(rows, predictions, scoring_fixture(),
                              samples_per_task=1, starts_in_thinking=True)
    assert result['macro_score'] == 0


def test_capped_thinking_scores_zero_and_is_reported():
    rows = cohort()
    predictions = {(r['task'], r['index']): '<think>alpha beta' for r in rows}
    result = ruler.score_suite(rows, predictions, scoring_fixture(),
                              samples_per_task=1, starts_in_thinking=False)
    assert result['macro_score'] == 0
    assert all(r['incomplete_thinking_cases'] == 1 for r in result['per_task'].values())


def test_qa_alternates_differ_from_multiple_required_references():
    rows = cohort()
    predictions = {(r['task'], r['index']): 'analysis</think>alpha' for r in rows}
    result = ruler.score_suite(rows, predictions, scoring_fixture(),
                              samples_per_task=1, starts_in_thinking=True)
    assert result['per_task']['qa_1']['score'] == 100
    assert result['per_task']['qa_1']['full_reference_recall_cases'] == 1
    assert result['per_task']['niah_multivalue']['score'] == 50
    assert result['per_task']['niah_multivalue']['full_reference_recall_cases'] == 0
    assert result['macro_score'] == pytest.approx((11 * 50 + 2 * 100) / 13)


def test_prediction_identity_is_checked_and_order_independent():
    rows = cohort(2)
    for row in rows:
        row['outputs'] = [str(row['index'])]
    predictions = {(r['task'], r['index']): str(r['index']) for r in reversed(rows)}
    settings = dict(samples_per_task=2, starts_in_thinking=False)
    assert ruler.score_suite(rows, predictions, scoring_fixture(), **settings)['macro_score'] == 100
    predictions.pop(next(iter(predictions)))
    with pytest.raises(ValueError):
        ruler.score_suite(rows, predictions, scoring_fixture(), **settings)


def test_modified_official_source_is_rejected(tmp_path):
    for relative in ruler.SOURCE_HASHES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('raise RuntimeError("must not execute modified source")')
    with pytest.raises(ValueError, match='Frozen RULER source changed'):
        ruler.load_scorer(tmp_path)
