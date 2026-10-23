from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_group_recall import build_halo_groups, describe_halo_selection, halo_summary


def records(starts, counts=None):
    return [NS(history_begin=s, count=n, handle=NS(slot=100-i, generation=2+i))
        for i, (s, n) in enumerate(zip(starts, counts or [32]*len(starts)))]


def test_halo_uses_original_positions_not_reordered_slots_and_never_recurses():
    rows = records([0,32,64,96,128])
    spans, scores, meta = build_halo_groups(rows, [0,1,2,3,4])
    assert spans == [(0,64),(0,96),(32,128),(64,160),(96,160)]
    assert meta['group_members'][2] == [32,64,96]
    assert meta['anchor_starts'] == [0,32,64,96,128] and scores == [0,1,2,3,4]
    assert meta['atomic_groups'] and not meta['post_picker_expansion']


def test_holes_partial_tails_and_noncontiguous_physical_pages():
    rows = records([0,32,96,128,160], [32,7,32,32,5])
    spans, _, meta = build_halo_groups(rows, [1]*5)
    assert spans == [(0,39),(0,39),(96,160),(96,165),(128,165)]
    assert meta['group_members'][1] == [0,32]
    assert meta['group_members'][2] == [96,128]


def test_message_boundaries_and_unknown_or_straddling_blocks_do_not_bridge():
    rows = records([0,32,64,96,128,160])
    spans, _, _ = build_halo_groups(rows, [1]*6,
        message_spans=[(0,48,'system'),(48,128,'tool'),(160,192,'user')])
    assert spans == [(0,32),(32,64),(64,128),(64,128),(128,160),(160,192)]


@pytest.mark.parametrize('rows,scores,kwargs', [
    (records([0,32]), [1], {}), (records([32,0]), [1,1], {}),
    (records([1]), [1], {}), (records([0],[33]), [1], {}),
    (records([0]), [float('nan')], {}), (records([0]), [1], {'radius': True}),
    (records([0]), [1], {'radius': 2}),
])
def test_invalid_halo_inputs_fail(rows, scores, kwargs):
    with pytest.raises(ValueError): build_halo_groups(rows, scores, **kwargs)


def test_description_counts_displacement_and_deduplicated_physical_budget():
    rows = records([0,32,64,96,128])
    _, _, meta = build_halo_groups(rows, [0,.3,.2,.9,.1])
    slots = lambda starts: [r.handle.slot for r in rows if r.history_begin in starts]
    info = describe_halo_selection(meta, rows, slots([0,64,96,128]),
        slots([0,32,64,96]), slots([0]),128)
    assert info['selected_tokens'] == info['charged_tokens'] == 128
    assert info['added_starts'] == [128] and info['displaced_starts'] == [32]
    assert info['covered_anchor_starts'] == [96,128]
    assert 'groups' not in halo_summary(info)
    assert info['unused_block_slots'] == 0
    with pytest.raises(RuntimeError, match='orphaned'):
        describe_halo_selection(meta, rows, slots([0,64]), slots([0,64]), slots([0]),128)
    with pytest.raises(RuntimeError, match='budget'):
        describe_halo_selection(meta, rows, slots([0,32,64]), slots([0]), slots([0]),64)


def test_switches_are_independent_and_launch_roundtrip(tmp_path):
    from test_graph_options import model_profile, runtime, worker_payload
    from test_query_key_fps_integration import fps_settings
    from vllm_kvmem.serve import build_launch
    profile = model_profile(tmp_path)
    profile['kvmem'] = fps_settings(query_fps_halo_blocks=1, query_fps_optimized=True)
    original = deepcopy(profile)
    options = runtime(profile['kvmem'])
    assert options.query_fps_halo_blocks == 1 and options.query_fps_optimized
    launch = build_launch(profile, tmp_path, tmp_path/'run')
    worker = worker_payload(launch)
    assert profile == original
    assert worker['query_fps_halo_blocks'] == 1 and worker['query_fps_optimized'] is True
    assert worker_payload(build_launch(launch['profile'], tmp_path, tmp_path/'again')) == worker
    assert runtime(fps_settings(query_fps_optimized=True)).query_fps_halo_blocks == 0
    assert runtime(fps_settings(query_fps_halo_blocks=1)).query_fps_optimized is False


@pytest.mark.parametrize('update', [
    {'query_fps_halo_blocks':2}, {'query_fps_halo_blocks':True},
    {'query_fps_optimized':1}, {'query_fps_halo_blocks':-1},
    {'query_score_mode':'mean_k','query_fps_halo_blocks':1},
    {'query_score_mode':'overlap_16_8','query_fps_optimized':True},
])
def test_invalid_switches_are_rejected_before_serving(update):
    from test_graph_options import runtime
    from test_query_key_fps_integration import fps_settings
    cfg = fps_settings(); cfg.update(update)
    with pytest.raises(ValueError): runtime(cfg)


def test_actual_query_selects_atomic_neighbors_with_native_budget_and_exports(tmp_path, monkeypatch):
    from test_query_key_fps_integration import actual
    from test_graph_lifecycle import selected_starts
    from vllm_kvmem.authority import _Ready
    a, archive, graph = actual(tmp_path, select_budget=128)
    a.graph_options.query_fps_halo_blocks = 1
    a.graph_options.graph_trace = True
    a.graph_report_path = tmp_path/'block_graph.jsonl'
    values = torch.tensor([0., .5, .4, .9, .001, .002])
    monkeypatch.setattr('vllm_kvmem.query_key_fps.score_authority_fps', lambda *args: (values, {}))
    plan = a.select_for_queries(torch.ones(1,1,1,2), max_tokens=128)
    assert selected_starts(a, plan) == [0,64,96,128]
    assert plan.total_window_tokens == 128
    assert a.memory._store.runtime_select_budget == 0
    trace = json.loads((tmp_path/'block_graph_inputs.jsonl').read_text())
    grouped = trace['query_group_selection']
    assert grouped['group_scores'] == trace['raw_scores'] == values.tolist()
    assert grouped['selected_starts'] == [0,64,96,128]
    assert grouped['displaced_starts'] == [32] and grouped['added_starts'] == [128]
    assert grouped['covered_anchor_starts'] == [96]
    assert a.query_group_selection == halo_summary(grouped)
    assert a.graph_selection_report['identity'] and not a.graph_selection_report['propagation_applied']
    a.memory.attach_completion(_Ready()); assert a.memory.commit_selection()
    del plan
    assert a.close() and archive.close()


def test_group_trace_failure_precedes_assembly_and_keeps_budget(tmp_path, monkeypatch):
    from test_query_key_fps_integration import actual
    from vllm_kvmem.memory import Phase
    a, archive, graph = actual(tmp_path, select_budget=128)
    a.graph_options.query_fps_halo_blocks = 1
    def fail(*args, **kwargs): raise OSError('injected group trace error')
    monkeypatch.setattr('vllm_kvmem.graph_integration.emit_graph_report', fail)
    with pytest.raises(OSError, match='group trace'):
        a.select_for_queries(torch.ones(1,1,1,2), max_tokens=128)
    assert a.memory.phase is Phase.READY and a.memory._store.runtime_select_budget == 0
    assert a.close() and archive.close()
