"""Classic RULER v1 cohort and scoring checks, independent of GPU execution.

Sample count is explicit: covering all 13 tasks does not imply 500 examples
per task. Scores use the frozen upstream functions, with final-answer-only
grading for thinking models. Substring coverage is not exact-answer accuracy.
"""
import ast
from collections import Counter
import hashlib
from pathlib import Path
import re
import statistics


UPSTREAM_COMMIT = 'c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a'
TASK_FAMILIES = {
    'niah_single_1': 'niah',
    'niah_single_2': 'niah',
    'niah_single_3': 'niah',
    'niah_multikey_1': 'niah',
    'niah_multikey_2': 'niah',
    'niah_multikey_3': 'niah',
    'niah_multivalue': 'niah',
    'niah_multiquery': 'niah',
    'vt': 'variable_tracking',
    'cwe': 'common_words_extraction',
    'fwe': 'freq_words_extraction',
    'qa_1': 'qa',
    'qa_2': 'qa',
}
SOURCE_HASHES = {
    'scripts/eval/synthetic/constants.py':
        '6740467c17b8dc06b6b30f4f97e54ce8de81db0dd879f1538d0b6b5727f4bd5f',
    'scripts/eval/evaluate.py':
        'b9a5fcbded7209663d97670496c74dbcd6358ddbe3d9887a0c5f0c4f974bfe1c',
    'scripts/synthetic.yaml':
        '34bc71dcacdc41a829a170f04b528fbf48d62c616338005ab4991680fbf8cb0b',
}


def validate_cohort(rows, samples_per_task):
    """Require every task and unique cases, even before any GPU work starts."""
    if type(samples_per_task) is not int or samples_per_task <= 0:
        raise ValueError('samples_per_task must be an explicit positive integer')
    counts = Counter(row['task'] for row in rows)
    if counts != Counter({task: samples_per_task for task in TASK_FAMILIES}):
        raise ValueError(f'Incomplete or unbalanced 13-task RULER cohort: {dict(counts)}')
    ids = [(row['task'], row['index']) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate RULER case identity')
    for row in rows:
        references = row['outputs']
        if (not isinstance(references, list) or not references or
                any(not isinstance(value, str) or not value.strip() for value in references)):
            raise ValueError('Every RULER case needs nonempty reference answers')
        if not isinstance(row['input'], str) or not row['input'].strip():
            raise ValueError('Every RULER case needs the original public input')
    return dict(tasks=len(counts), samples_per_task=samples_per_task, cases=len(rows),
                official_default_sample_count=samples_per_task == 500)


def public_query(task, input_text):
    """Locate the trailing public task question without inspecting references."""
    family = TASK_FAMILIES[task]
    pattern = (r'\nWhat (?:are all|is) the special magic ' if family == 'niah'
               else r'\nQuestion: ')
    matches = list(re.finditer(pattern, input_text))
    if not matches:
        raise ValueError(f'Missing upstream public question for {task}')
    match = matches[-1]
    begin, end = match.start() + 1, len(input_text.rstrip())
    if end <= match.end():
        raise ValueError('Empty public question')
    return dict(text=input_text[begin:end], char_span=[begin, end])


def final_answer(text, *, starts_in_thinking):
    """A capped, unclosed thinking segment is not an answer to be graded."""
    thinking = starts_in_thinking or text.lstrip().startswith('<think>')
    if not thinking:
        return dict(text=text, thinking_complete=None)
    if '</think>' not in text:
        return dict(text='', thinking_complete=False)
    return dict(text=text.split('</think>', 1)[1], thinking_complete=True)


def load_scorer(upstream):
    """Load only the byte-verified official scorer and prediction normalizer."""
    upstream = Path(upstream)
    sources = {}
    for relative, expected in SOURCE_HASHES.items():
        data = (upstream / relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Frozen RULER source changed: {relative}')
        sources[relative] = data.decode('utf-8')
    scope = {'re': re}
    exec(compile(sources['scripts/eval/synthetic/constants.py'],
                 'frozen-ruler-metrics', 'exec'), scope)
    tree = ast.parse(sources['scripts/eval/evaluate.py'])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'postprocess_pred']
    if len(functions) != 1:
        raise ValueError('Missing unique upstream prediction normalizer')
    exec(compile(ast.Module(body=functions, type_ignores=[]),
                 'frozen-ruler-postprocess', 'exec'), scope)
    return scope['TASKS'], scope['postprocess_pred']


def score_suite(rows, predictions, scorer, *, samples_per_task, starts_in_thinking):
    """Score complete cohorts only; task scores use upstream batch rounding.

    Predictions are keyed by (task, index), so completion order cannot change
    reference association. Missing predictions must never silently disappear.
    """
    cohort = validate_cohort(rows, samples_per_task)
    expected = {(row['task'], row['index']) for row in rows}
    if set(predictions) != expected:
        raise ValueError('Missing or unexpected model predictions')
    metrics, postprocess = scorer
    tasks = {}
    for task, family in TASK_FAMILIES.items():
        selected = sorted((row for row in rows if row['task'] == task),
                          key=lambda row: row['index'])
        answers = [final_answer(predictions[(task, row['index'])],
                                starts_in_thinking=starts_in_thinking) for row in selected]
        config = metrics[family]
        cleaned = [postprocess(answer['text'], config) for answer in answers]
        references = [row['outputs'] for row in selected]
        fully_recalled = []
        for prediction, refs in zip(cleaned, references):
            matches = [ref.lower() in prediction.lower() for ref in refs]
            fully_recalled.append(any(matches) if family == 'qa' else all(matches))
        tasks[task] = dict(cases=len(selected),
            score=config['metric_fn'](cleaned, references),
            full_reference_recall_cases=sum(fully_recalled),
            incomplete_thinking_cases=sum(a['thinking_complete'] is False for a in answers))
    return dict(**cohort, upstream_commit=UPSTREAM_COMMIT,
        scoring_scope='Final answer only; frozen official substring metrics, not exact match.',
        macro_score=statistics.fmean(task['score'] for task in tasks.values()), per_task=tasks)
