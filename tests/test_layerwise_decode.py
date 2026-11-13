from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.layerwise_decode import DecodeHistory, sample_rows, validate_configuration
from vllm_kvmem.serving.action_attention import ActionHistory, ActionPacket
from vllm_kvmem.serving.block_graph import BlockGraph
from test_layerwise_pressure import controller, options, native_authority, finish
from test_graph_capture import action, options as action_options


def packet(c, *, thinking=True):
    o = action_options(action_interval=2, action_native_decode_commit=True,
                       pressure_policy='attention', guided_query_tokens=0, query_history=False)
    shared = ActionHistory(o, [10, 11, 12, 13, 14])
    shared.begin('r', [14, 12] if thinking else [14])
    shared.parser.prepare([20, 20, 20, 20])
    c.maps['r'] = (256, (((0, 1, 0, 32, 0), (1, 1, 64, 32, 32)),
                          ((0, 1, 0, 32, 0), (2, 1, 128, 32, 32))))
    c.rolling_decode = True
    p = ActionPacket(shared, [1, 3], [0, 999, 256], None, None,
                     source_positions=(257, 259), history_begin=256, candidate_count=4)
    masses = torch.tensor([[[.1, .8, .1], [0., 0., 1.]],
                           [[.2, .7, .1], [0., 0., 1.]]])
    p.masses = masses[0]
    c.attach_packet(p, 'r', None, [0, 32, 64], [32, 32, 4], masses)
    return p


def test_thinking_and_rejected_speculation_are_independent_of_legacy_scope():
    c = controller(); p = packet(c)
    assert p.state.parser.preview(3) == 0  # All accepted rows still thinking.
    assert p.history == [0, 64, 256]  # True layer zero, not global preselection.
    staged = c.prepare_accept(p, 3)
    assert all(h.samples == 0 for h in c.decode_histories['r'])
    c.commit_accept(staged)
    a, b = c.decode_histories['r']
    assert (a.samples, b.samples) == (1, 1)
    assert a.mass[64] == pytest.approx(.8) and b.mass[128] == pytest.approx(.7)
    assert a.mass[256] == pytest.approx(.1)  # Rejected row's unit mass excluded.
    c.commit_accept(c.prepare_accept(p, 3))
    assert c.decode_histories['r'][0].samples == 1


def test_invalid_second_layer_never_publishes_first_layer():
    c = controller(); p = packet(c)
    p.layerwise_decode[3][1, 0, 0] = float('nan')
    with pytest.raises((ValueError, RuntimeError)): c.prepare_accept(p, 3)
    assert all(h.samples == 0 and not h.mass for h in c.decode_histories['r'])


@pytest.mark.parametrize('abort', [False, True])
def test_decode_feedback_shares_graph_commit_and_abort(tmp_path, abort):
    c = controller(); p = packet(c)
    runtime = action(tmp_path); runtime.runner = NS(_kvmem_layerwise_pressure=c)
    graph = BlockGraph('rolling', 32, initial_position=256)
    before = p.state.parser.state
    tx = runtime.prepare_accept(NS(action_inputs={'r': p}), 'r', 3, graph=graph, history_begin=256)
    assert c.decode_histories['r'][0].samples == 0 and p.state.parser.state == before
    if abort:
        tx.abort()
        assert c.decode_histories['r'][0].samples == 0 and graph.accepted_through == 256
    else:
        tx.commit()
        assert c.decode_histories['r'][0].samples == 1 and graph.accepted_through == 259
        assert p.state.samples == 0  # Legacy non-thinking policy is unchanged.
    graph.close()


@pytest.mark.parametrize('abort', [False, True])
def test_native_segments_at_thinking_exit_keep_two_sampling_scopes(tmp_path, abort):
    import numpy as np
    c = controller(); p = packet(c)
    # Closing thinking makes the legacy parser sample at a different offset
    # from absolute-position rolling sampling. Native metadata uses int64.
    mask = p.state.parser.prepare([20, 13, 20, 20, 20, 20])
    p.rows = sorted(set(sample_rows(256, 6, 2)) |
                    {r for r in range(6) if mask >> r & 1})
    p.candidate_count = 6
    p.source_positions = tuple(256+r for r in p.rows)
    masses = torch.tensor([[[.1, .8, .1]]*len(p.rows),
                           [[.2, .7, .1]]*len(p.rows)])
    p.masses = masses[0]
    c.attach_packet(p, 'r', None, np.array([0, 32, 64]), np.array([32, 32, 6]), masses)
    runtime = action(tmp_path); runtime.runner = NS(_kvmem_layerwise_pressure=c)
    graph = BlockGraph('transition', 32, initial_position=256)
    tx = runtime.prepare_accept(NS(action_inputs={'r':p}), 'r', 4,
                                graph=graph, history_begin=256)
    if abort:
        tx.abort()
        assert p.state.samples == c.decode_histories['r'][0].samples == 0
        assert graph.accepted_through == 256
    else:
        tx.commit()
        assert p.state.samples == 1  # Only the accepted non-thinking sample.
        assert c.decode_histories['r'][0].samples == 2  # Includes thinking.
        assert graph.accepted_through == 260
    graph.close()


def test_decode_pressure_uses_only_each_layers_current_candidates():
    c = controller(); c.rolling_decode = True
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); hs = c._history('r', o)
    a.action_history = NS(prefill=hs[0])
    rows = list(a.records.values())
    def mapping(ids):
        return tuple((rows[i].handle.slot, rows[i].handle.generation,
                      rows[i].history_begin, 32, j*32) for j, i in enumerate(ids))
    # Native shared frame and real layer-zero content deliberately disagree.
    a.memory.select({}, pressure_scores=[(r.handle.slot, 1. if i in (1, 2) else 0.)
                                        for i, r in enumerate(rows)], max_tokens=96)
    finish(a.memory)
    c.maps['r'] = (256, (mapping((0, 3, 4)), mapping((0, 5, 6))))
    c.decode_histories['r'] = [DecodeHistory({0: .1, 96: .8, 128: .1, 32: 999}, 1, 256, 1),
                               DecodeHistory({0: .1, 160: .8, 192: .1, 32: 999}, 1, 256, 1)]
    job = NS(command=NS(request_id='r', pressure_stage='decode', mandatory_spans=(),
                       groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    try:
        c.prepare(NS(options=o), job)
        picked, evidence = job.layerwise_pressure_plan
        assert [{r.history_begin for r in layer} for layer in picked] == [{0, 96, 128}, {0, 160, 192}]
        assert all(e['score_scope'] == 'all_accepted_including_thinking' for e in evidence)
        slots = job.layerwise_decode_selection
        plan = a.memory.select({}, (rows[0].handle.slot,), action=False,
                               pressure_scores=[(s, 1.) for s in slots], max_tokens=96)
        assert {r.block_id for r in plan.remaps} == set(slots)
        finish(a.memory)
    finally:
        assert a.memory.close()


def test_sampling_uses_consumed_original_positions_not_dense_slots():
    assert sample_rows(100, 32, 16) == [11, 27]
    assert sample_rows(15, 1, 16) == [0]


def test_rolling_launch_requires_explicit_supported_contract():
    args = dict(query_mode='sequential', optimized=True, pressure=True,
                single_assembly=True, original_rope=False)
    profile = dict(draft_method='mtp', kvmem=dict(rolling_active_window=True,
        request_planning=True, async_refresh=True, action_attention=True,
        prefill_attention=True, pressure_policy='attention'))
    validate_configuration(profile, **args)
    with pytest.raises(ValueError): validate_configuration(profile, **dict(args, pressure=False))
    with pytest.raises(ValueError): validate_configuration(dict(profile, draft_method='dflash'), **args)


def test_rolling_native_producer_covers_each_target_decode_layer():
    from vllm_kvmem.serving.native_attention import statistic_layer_names
    names = tuple(f'language_model.model.layers.{n}.self_attn' for n in range(3, 64, 4))
    runner = NS(_kvmem_layers=names, _kvmem_layerwise_pressure=NS(names=names, rolling_decode=True))
    assert statistic_layer_names(runner, NS(action_layers='spread3')) == (names, names)


def test_finished_query_allows_only_explicit_rolling_decode():
    from vllm_kvmem.serving.layerwise_query import LayerwiseQuery
    query = LayerwiseQuery.__new__(LayerwiseQuery)
    query.finished = {'r'}
    visited = []
    query.pressure = NS(rolling_decode=True, prepare=lambda refresh, job: visited.append(job))
    job = NS(command=NS(request_id='r', selection='pressure', pressure_stage='decode'))
    query.prepare(None, job, None)
    assert visited == [job]
    # A cold prompt's tool history after a mid-prompt agent Query refreshes the
    # same per-layer maps through prefill pressure (dsh c8 crash, 2026-11-11).
    job.command.pressure_stage = 'prefill'
    query.prepare(None, job, None)
    assert visited == [job, job]
    query.pressure.rolling_decode = False
    with pytest.raises(RuntimeError, match='another refresh'): query.prepare(None, job, None)
    query.pressure.rolling_decode = True
    job.command.selection = 'query'
    with pytest.raises(RuntimeError, match='another refresh'): query.prepare(None, job, None)


def test_g4_keeps_16k_output_limit_and_schedules_decode_refresh_after_query():
    from vllm_kvmem.serving.request_plan import RequestPlan
    plan = RequestPlan((107324, 107359), block_tokens=32, gen_budget=4096,
                       output_tokens=16384, reserve_query_output=True, rolling_active_window=True)
    capacity, prompt = 16384 + 4096, 107366
    kept = plan.selection_budget(plan.replay_begin, prompt, capacity)
    assert kept == 16384 and plan.output_reserve == 0
    offset = plan.replay_begin - kept
    first = next(h for h in range(prompt, prompt+16384)
                 if plan.action(h, prompt, offset, capacity) == 'pressure')
    assert 0 < first-prompt < 4096 and plan.output_tokens == 16384


@pytest.mark.parametrize('tail', [1, 2, 7, 8, 15, 28, 31])
@pytest.mark.parametrize('count', [1, 2, 5, 32])
def test_native_merged_partial_tail_has_same_logical_identity(tail, count):
    from vllm_kvmem.serving.layerwise_decode import segment_identities
    from vllm_kvmem.serving.action_attention import window_segments
    from vllm_kvmem.serving.native_attention_planning import canonical_segments
    records = [(0, 1, 0, 32, 0), (1, 1, 128, tail, 32)]
    ls, lc, lh = window_segments(records, 128+tail, 32+tail, count, 32)
    canonical = canonical_segments(ls, lc, lh, 32+tail+count, 32)
    assert segment_identities(records, 128+tail, 32+tail, count,
                              canonical[:, 0], canonical[:, 1]) == canonical[:, 2].tolist()
    assert segment_identities(records, 128+tail, 32+tail, count, ls, lc) == lh


def test_native_merge_cannot_mix_two_layers_distinct_logical_blocks():
    from vllm_kvmem.serving.layerwise_decode import segment_identities
    with pytest.raises(RuntimeError, match='different logical blocks'):
        segment_identities([(0, 1, 0, 32, 0)], 128, 32, 5, [0], [37])
