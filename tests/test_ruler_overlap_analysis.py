"""Offline report separates implemented selection from inferred answer quality."""
from copy import deepcopy
import json

import pytest

import analyze_ruler_overlap_answers as analysis
from test_ruler_overlap_answers import audit_fixture


def test_native_graph_replay_identifies_displaced_value_without_predicting_answer():
    query, _, record, _ = audit_fixture()
    result = analysis.replay_selection(record, query['selected_history_ranges'], [98, 102], [92, 95])
    assert result['native_picker_replay_equal']
    assert result['pregraph_value']['state'] == 'full'
    assert result['actual_value']['state'] == 'none'
    assert result['graph_displaced_value_blocks'] == [96]
    target = next(block for block in result['target_blocks'] if block['value_tokens'])
    assert target['seed_rank'] == 3
    assert target['propagated_rank'] > 3
    assert 'correct' not in result


def test_replay_rejects_changed_actual_selection_even_when_total_budget_matches():
    query, _, record, _ = audit_fixture()
    selected = deepcopy(query['selected_history_ranges'])
    present = {start for start, _ in selected}
    selected[0][0] = next(start for start in record['starts'] if start not in present)
    with pytest.raises(ValueError, match='native SUM picker replay'):
        analysis.replay_selection(record, selected, [98, 102], [92, 95])


def test_disabled_graph_replay_preserves_raw_seed_ranks_and_selection():
    query, _, record, _ = audit_fixture(alpha=0)
    result = analysis.replay_selection(record, query['selected_history_ranges'], [98, 102], [92, 95])
    assert result['native_picker_replay_equal'] and result['graph_alpha'] == 0
    assert result['propagation']['identity']
    assert result['pregraph_value'] == result['actual_value']
    assert result['graph_displaced_value_blocks'] == result['graph_added_value_blocks'] == []
    assert all(block['seed_score'] == block['propagated_score'] and
        block['seed_rank'] == block['propagated_rank'] for block in result['target_blocks'])


def test_coverage_and_rank_report_partial_boundaries_and_transport_domain():
    assert analysis.span_coverage([[0, 32]], [29, 36]) == dict(tokens=7, selected_tokens=3,
        state='partial', missing_positions=[32, 33, 34, 35])
    assert analysis.span_coverage([[0, 32], [32, 32]], [29, 36])['state'] == 'full'
    assert analysis.ranks([100, 4, 4, 1], [False, True, True, True]) == [None, 1, 1, 3]


def test_completed_campaign_guard_creates_no_final_files(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(dict(status='testing')))
    with pytest.raises(ValueError, match='completed campaign'):
        analysis.main(['--root', str(tmp_path)])
    assert not (tmp_path/'analysis.json').exists() and not (tmp_path/'report.md').exists()


def test_immutable_case_hash_failure_is_not_treated_as_quality_failure(tmp_path):
    evidence = tmp_path/'evidence.json'
    evidence.write_text('{}')
    manifest = {'evidence.json':analysis.sha(evidence)}
    evidence.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='Immutable evidence changed'):
        analysis.verify_files(tmp_path, manifest)


def test_summary_quantiles_use_declared_linear_method():
    observed = analysis.distribution([0, 10, 20])
    assert observed['median'] == 10 and observed['p90'] == 18
