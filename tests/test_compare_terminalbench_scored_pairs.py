import json
from pathlib import Path
import shutil
import sys

import pytest

from scripts import compare_terminalbench_scored_pairs as wrapper
from scripts.compare_terminalbench_policies import exact_mcnemar
from test_compare_terminalbench_policies import arm, edit, write


def outcome(reward=1, exception=None):
    return dict(trial='first', status='scored' if reward is not None else 'error',
                rewards=None if reward is None else {'reward': reward},
                passed=reward == 1, exception=exception)


def test_unavailable_partner_is_excluded_not_imputed_as_failure():
    baseline = dict(a=outcome(1), b=outcome(0), c=outcome(1))
    candidate = dict(a=outcome(None, 'RuntimeError'), b=outcome(1), c=outcome(1))
    quality, excluded = wrapper.quality_analysis(baseline, candidate, exact_mcnemar)
    assert quality['task_count'] == 2 and quality['coverage_fraction'] == 2/3
    assert quality['paired_counts'] == dict(both_pass=1, both_fail=0, candidate_only_pass=1, baseline_only_pass=0)
    assert quality['accuracy_delta_percentage_points'] == 50
    assert quality['exact_mcnemar']['discordant_pairs'] == 1
    assert excluded[0]['task'] == 'a' and excluded[0]['baseline']['eligible']
    assert excluded[0]['candidate']['reasons'] == ['unscored', 'exception_recorded']
    assert 'not an estimate of complete 89-task accuracy' in quality['selection_bias']


@pytest.mark.parametrize('row', [outcome(None, 'RuntimeError'), outcome(None, 'AgentTimeoutError'),
    outcome(1, 'AgentTimeoutError'), outcome(.5), outcome(True)])
def test_nonordinary_scores_cannot_create_quality_pairs(row):
    quality, excluded = wrapper.quality_analysis({'a': outcome()}, {'a': row}, exact_mcnemar)
    assert quality['task_count'] == 0 and quality['status'] == 'unavailable'
    assert quality['baseline_pass_rate'] is quality['candidate_pass_rate'] is None
    assert quality['accuracy_delta_percentage_points'] is quality['exact_mcnemar'] is None
    assert excluded and not excluded[0]['candidate']['eligible']


@pytest.fixture
def pair(tmp_path):
    baseline = arm(tmp_path/'baseline', 'lru', {0, 4})
    candidate = arm(tmp_path/'candidate', 'redundancy', {0, 1, 2, 3})
    shutil.copyfile(Path(wrapper.__file__).with_name('compare_terminalbench_policies.py'),
                    tmp_path/'compare_terminalbench_policies.py')
    for directory in (baseline, candidate):
        formal = wrapper.read(directory/'formal-events/summary.json')
        for row in formal['outcomes']:
            raw = dict(task_name=row['task'], trial_name=row['trial'],
                verifier_result=dict(rewards=row['rewards']) if row['rewards'] is not None else None,
                exception_info=dict(exception_type=row['exception'], exception_message='Original setup failure',
                                    exception_traceback='original traceback') if row['exception'] else None)
            write(directory/'jobs/formal'/row['trial']/'result.json', raw)
            write(directory/'formal-events'/(row['trial']+'.json'), raw)
    return baseline, candidate


def test_wrapper_validates_frozen_pair_and_preserves_all_original_artifacts(pair):
    before = {str(path): wrapper.sha(path) for root in pair for path in root.rglob('*') if path.is_file()}
    result = wrapper.compare(*pair)
    assert result['passed'] and result['attempts_complete'] and not result['scoring_complete']
    assert result['analysis_scope'] == wrapper.ANALYSIS_SCOPE
    assert result['quality']['task_count'] == 88
    assert result['quality']['paired_counts']['candidate_only_pass'] == 3
    assert result['quality']['paired_counts']['baseline_only_pass'] == 1
    assert result['quality']['accuracy_delta_percentage_points'] == 200/88
    assert result['quality']['exact_mcnemar']['p_value'] == .625
    assert result['operational']['candidate']['success_fraction_all_attempts'] == 4/89
    assert result['candidate']['scored'] == 88 and result['candidate']['unscored'] == 1
    assert result['operational']['candidate']['unscored_tasks'][0]['task'] == 'task-05'
    assert result['raw_outcome_evidence']['candidate']['task-05']['exception_info']['exception_traceback'] == 'original traceback'
    assert result['gc_triggered_subsets']['status'] == 'disabled'
    assert result['candidate']['gc_formal_delta']['retired'] == 100
    assert result['analysis_implementation']['comparator']['sha256'] == wrapper.sha(pair[1].parent/'compare_terminalbench_policies.py')
    assert before == {str(path): wrapper.sha(path) for root in pair for path in root.rglob('*') if path.is_file()}


def test_wrapper_cannot_relax_original_configuration_validation(pair):
    edit(pair[1], 'profile.json', lambda data: data['kvmem'].__setitem__('gen_budget', 1))
    with pytest.raises(ValueError):
        wrapper.compare(*pair)


@pytest.mark.parametrize('change', ['reward', 'exception_type', 'exception_details', 'missing_exception'])
def test_raw_evidence_changes_are_rejected(pair, change):
    root = pair[1]; trial = 'task-05-redundancy'
    path = root/'jobs/formal'/trial/'result.json'; raw = wrapper.read(path)
    if change == 'reward':
        raw['verifier_result'] = dict(rewards={'reward': 1})
    elif change == 'exception_type':
        raw['exception_info']['exception_type'] = 'OtherError'
    elif change == 'exception_details':
        raw['exception_info']['exception_traceback'] = 'changed traceback'
    else:
        raw['exception_info'] = None
    write(path, raw)
    with pytest.raises(ValueError):
        wrapper.compare(*pair)


def test_raw_pass_boolean_is_checked_even_with_permissive_original_loader(pair):
    root = pair[1]
    raw_run = dict(root=root, outcomes={'task-00': dict(task='task-00', trial='task-00-redundancy',
        status='scored', rewards={'reward': 1}, passed=False, exception=None)})
    with pytest.raises(ValueError, match='reward/pass mismatch'):
        wrapper.raw_outcome_evidence(raw_run)


def test_cli_never_replaces_an_existing_analysis(pair, tmp_path, monkeypatch):
    output = tmp_path/'existing.json'; output.write_text('original analysis')
    monkeypatch.setattr(sys, 'argv', ['wrapper', '--baseline', str(pair[0]), '--candidate', str(pair[1]), '--output', str(output)])
    with pytest.raises(ValueError, match='Never replace'):
        wrapper.main()
    assert output.read_text() == 'original analysis'
