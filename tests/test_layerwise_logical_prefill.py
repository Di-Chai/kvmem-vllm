"""Original 1K publication transactions retain every layer's score domain."""
from copy import copy
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_attention import ActionPacket, ActionRuntime
from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure
from vllm_kvmem.logical_action import LogicalScorePlan, project_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillEligibility, plan_logical_prefill
from vllm_kvmem.request_plan import RequestPlan
from test_logical_action import state, consumer, publish
from test_physical_prefill_options import payload, model_profile, build_launch, runtime, worker_payload


def fixture(chunks=8):
    group, reason = plan_logical_prefill(RequestPlan(None, block_tokens=32, gen_budget=16384),
        request_id='r', owner_epoch=(1, 0), scheduler_step=7, history_begin=8192,
        offset=8128, prompt_tokens=262144, capacity=53248, logical_chunk_tokens=1024,
        eligibility=LogicalPrefillEligibility(('r',), (1024,)*chunks, 8192))
    assert reason is None and group.count == chunks*1024
    plans = tuple(LogicalScorePlan(s, tuple(range(31, 1024, 32)),
        tuple(range(0, s.local_end, 32)), (32,)*(s.local_end//32),
        (0, 4096, *range(8192, s.history_end, 32))) for s in group.spans)
    rows = tuple(s.physical_offset+r for s, p in zip(group.spans, plans) for r in p.rows)
    # Each layer has different past blocks AND scores. No future scores enter
    # the old transactions, and every row is normalized in its causal domain.
    histories = [(0, 4096+32*l, *plans[-1].history[2:]) for l in range(16)]
    masses = torch.zeros((16, chunks*32, len(plans[-1].starts)))
    for l in range(16):
        for r in range(chunks*32):
            masses[l, r, 0] = .125
            masses[l, r, 1] = (l+1)/64
            masses[l, r, 2+r] = 1-.125-(l+1)/64
    return group, plans, rows, histories, masses


def controller():
    c = LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    c.names = tuple(f'l{i}' for i in range(16))
    c.histories = {'r': [state(True).prefill for _ in c.names]}
    c.install_acceptance()
    return c


@pytest.mark.parametrize('retained', [0, 1023, 1024, 1025, 4096, 8191, 8192])
def test_all_layer_histories_match_independent_original_transactions(tmp_path, retained):
    group, plans, rows, histories, masses = fixture()
    c = controller()
    expected = [copy(h) for h in c.histories['r']]
    parts = project_logical_scores(group, plans, masses[0], physical_rows=rows,
        layerwise_pressure=(c, 'r', histories, masses))
    action, graph = consumer(tmp_path)
    action.runner = NS(_kvmem_layerwise_pressure=c)
    current = state(True)
    try:
        for i, part in enumerate(parts):
            accepted = part.accepted_from_prefix(retained)
            p, s = part.plan, part.plan.span
            for l, h in enumerate(expected):
                h.accept(s.history_end, s.count, accepted, p.rows, p.starts,
                    histories[l][:len(p.starts)], s.local_begin,
                    masses[l, i*32:(i+1)*32, :len(p.starts)].clone())
            publish(action, graph, part.action_packet(current), accepted)
            assert [h.snapshot() for h in c.histories['r']] == [h.snapshot() for h in expected]
        assert graph.accepted_through == 8192+retained
        assert all(h.chunks == 4+retained//1024 for h in c.histories['r'])
    finally:
        graph.close()


def test_runtime_projection_keeps_all_layers_and_original_download_lease():
    group, plans, rows, histories, masses = fixture()
    c = controller()
    pending = NS(masses=masses[0], event=object(), layerwise_pressure=(c,'r',histories,masses))
    result = ActionRuntime._logical_packet_views({}, 'r',
        dict(logical_group=group, logical_score_plans=plans, rows=rows), pending)
    for i, part in enumerate(result.logical_slices['r']):
        packet = part.action_packet(state(True))
        owner, rid, ids, scores = packet.layerwise_pressure
        assert owner is c and rid == 'r' and packet.event is pending.event
        assert tuple(scores.shape) == (16, 32, 2+32*(i+1))
        assert scores.untyped_storage().data_ptr() == masses.untyped_storage().data_ptr()
        assert tuple(ids[15]) == tuple(histories[15][:2+32*(i+1)])
        assert all(max(h) < part.plan.span.history_end for h in ids)
        assert scores[0].data_ptr() == part.masses.data_ptr()


@pytest.mark.parametrize('bad', ['owner','layers','columns','history','separate_shared'])
def test_layerwise_projection_rejects_incompatible_owner_before_commit(bad):
    group, plans, rows, histories, masses = fixture()
    c = controller();rid='r';shared=masses[0]
    if bad == 'owner': rid='other'
    if bad == 'layers': masses=masses[:-1]
    if bad == 'columns': masses=masses[:, :, :-1]
    if bad == 'history': histories[0]=(32, *histories[0][1:])
    if bad == 'separate_shared': shared=shared.clone()
    with pytest.raises(ValueError):
        project_logical_scores(group, plans, shared, physical_rows=rows,
            layerwise_pressure=(c,rid,histories,masses))


def test_launch_1k_logical_8k_physical_does_not_increase_stats_or_kv(tmp_path):
    profile = model_profile(tmp_path)
    profile['prefill_chunk_tokens'] = 1024
    profile['kvmem'] = payload(prefill_physical_logical_tokens=1024,
        prefill_physical_max_chunks=8, action_native_max_rows=320,
        select_budget=36864, gen_budget=16384)
    launch = build_launch(profile,tmp_path,tmp_path/'launch')
    options = runtime(worker_payload(launch))
    assert launch['environment']['KVMEM_PREFILL_CHUNK_TOKENS'] == '1024'
    assert options.prefill_physical_logical_tokens == 1024
    assert options.action_native_max_rows == 320
    profile['prefill_chunk_tokens']=2048
    with pytest.raises(ValueError): build_launch(profile,tmp_path,tmp_path/'bad')
