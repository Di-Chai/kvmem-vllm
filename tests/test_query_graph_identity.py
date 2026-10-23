"""Query diagnostics distinguish retained graphs from applied propagation."""
import json

import pytest
import torch

from vllm_kvmem.authority import _Ready, query_score_frame
from vllm_kvmem.query_window_scoring import score_authority_windows
from test_graph_lifecycle import selected_starts
from test_query_window_lifecycle import actual


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('trace', [False, True])
def test_alpha_zero_passes_exact_overlap_seeds_to_picker_and_reports_no_propagation(tmp_path, monkeypatch, native, trace):
    from vllm_kvmem import graph_scoring
    authority, archive, graph = actual(tmp_path)
    options = authority.graph_options
    options.graph_alpha = 0.
    options.graph_native_propagation = native
    options.graph_trace = trace
    authority.graph_report_path = tmp_path/'query_graph.jsonl'
    q = torch.tensor([[[[2., 1.], [1., -2.]], [[-1., 2.], [3., 1.]]]])
    with authority.index_snapshot() as (records, means):
        with query_score_frame(authority.memory, records) as frame:
            seeds, _ = score_authority_windows(authority, records, q, means, frame.score_mask)
        raw_by_slot = {record.handle.slot:value for record, value in zip(records, seeds.tolist())}
        slots = set(authority.memory.preview_selection(raw_by_slot, (), action=False, prefill_scores=None))
        expected_starts = sorted(record.history_begin for record in records if record.handle.slot in slots)
    calls = []
    original_score = graph_scoring.score_graph
    def capture_score(raw, *args, **kwargs):
        output = original_score(raw, *args, **kwargs)
        assert output.scores is raw  # No normalization/round-trip when alpha=0.
        assert output.diagnostics['identity'] is True
        calls.append(list(output.scores))
        return output
    monkeypatch.setattr(graph_scoring, 'score_graph', capture_score)
    original_select = authority.memory.select
    committed_inputs = []
    def capture_select(scores, *args, **kwargs):
        committed_inputs.append(dict(scores))
        return original_select(scores, *args, **kwargs)
    monkeypatch.setattr(authority.memory, 'select', capture_select)
    plan = authority.select_for_queries(q)
    assert calls == [seeds.tolist()]*(2 if trace else 1)
    assert committed_inputs == [raw_by_slot]
    assert selected_starts(authority, plan) == expected_starts
    report = authority.graph_selection_report
    assert report['identity'] is True and report['propagation_applied'] is False
    assert report['added_blocks'] == report['displaced_blocks'] == 0
    assert report['seed_source'] == 'overlap_16_8'
    saved = json.loads(authority.graph_report_path.read_text())
    assert saved['propagation_applied'] is False and saved['identity'] is True
    if trace:
        assert report['baseline_starts'] == report['selected_starts'] == expected_starts
    authority.memory.attach_completion(_Ready())
    assert authority.memory.commit_selection()
    del plan
    assert authority.close() and archive.close()


def test_nonidentity_query_reports_applied_propagation(tmp_path):
    authority, archive, _ = actual(tmp_path)
    plan = authority.select_for_queries(torch.tensor([[[[2., 1.], [1., -2.]]]]))
    report = authority.graph_selection_report
    assert report['alpha'] == .2 and report['identity'] is False
    assert report['propagation_applied'] is True
    authority.memory.attach_completion(_Ready())
    assert authority.memory.commit_selection()
    del plan
    assert authority.close() and archive.close()
