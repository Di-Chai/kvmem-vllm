"""Real FlashInfer prefill + production Action staging, outside a model graph.

The queued tail simulates independent MTP GPU work only for event-order checks;
native MTP correctness is covered by wrapper tests and the full-model A/B run.
"""
import gc
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime, action_metadata
from vllm_kvmem.action_download_pool import ActionDownloadPool
from vllm_kvmem.prefill_attention import PrefillHistory
from test_native_attention_stats_gpu import geometry

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit native prefill CUDA qualification')


def fixture(tmp_path, *, mode='native'):
    bank, native_plan, execute, oracle, q, meta, output, (k, v), table = geometry(
        tmp_path, 'prefill', qlen=5, dim=256, heads=24, kvheads=4, page=16)
    options = bank.options
    for key, value in dict(action_scope='non-thinking', action_interval=1, action_decay=.5,
            prefill_attention=True, prefill_attention_interval=1, prefill_attention_stride=1,
            prefill_attention_half_life=4096, prefill_publication_pipeline=True,
            prefill_action_before_mtp=True, action_fused_scoring=False).items():
        setattr(options, key, value)
    state = ActionHistory(options, [10, 11, 12, 13, 14]); state.begin('r', [14])
    authority = NS(action_history=state, block_graph=None)
    name = 'model.layers.3.attn'
    trig = torch.cat((torch.ones(259, 32, device='cuda'), torch.zeros(259, 32, device='cuda')), -1)
    module = NS(num_kv_heads=4, head_dim=256, rotary_emb=NS(cos_sin_cache=trig),
        attn=NS(layer_name=name, kv_cache=torch.cat((k, v), -1).transpose(1, 2),
                impl=NS(scale=256**-.5), _k_scale=torch.tensor(.7, device='cuda'),
                _q_scale=torch.ones((), device='cuda')))
    positions = torch.arange(254, 259, device='cuda', dtype=torch.int64)
    batch = NS(num_reqs=1, num_tokens=5, req_ids=['r'], is_prefilling_np=[True],
        query_start_loc_np=np.array([0, 5], dtype=np.int32), num_computed_tokens_np=np.array([254]),
        input_ids=torch.tensor([20, 21, 22, 23, 24], device='cuda'), positions=positions)
    captured = [NS(name=name, query=q, positions=positions,
                   key=torch.zeros((5, 4, 256), device='cuda', dtype=torch.bfloat16))]
    plan = dict(native_plan, pipeline=True, history_begin=254, authority=authority, state=state,
                sources=(), history=native_plan['starts'])
    if mode == 'no_rows': plan['rows'] = []
    action = ActionRuntime.__new__(ActionRuntime)
    action.options, action.native, action.ids = options, bank, [10, 11, 12, 13, 14]
    action.runtime = NS(config=options.memory, output=tmp_path, _raise_publication_error=lambda: None)
    action.runner = NS(_kvmem_layers=[name], req_states=NS(req_id_to_index={'r': 0},
        prompt_len=NS(np=np.array([1024]))), block_tables=NS(block_tables=[NS(gpu=table)]))
    action.layers, action.groups = [(0, module)], {name: 0}
    action.q_dtypes = action.prefill_q_dtypes = {name: q.dtype}
    action.native_cache = {}
    action.capture_seconds = action.metadata_wait_seconds = 0.
    action.candidate_samples = action.packets = action.metadata_early_batches = 0
    action.download_pool = ActionDownloadPool(1024, 2)
    # Establish observed native backend independently of each frame's producer.
    bank.arm({'r': plan}); execute(); torch.cuda.current_stream().synchronize(); bank.disarm()

    def prepare(*, graph=None):
        action.metadata = action_metadata(batch, positions, True)
        action.prefill_plans = action.native_plans = (batch, {'r': plan})
        action.pipeline_requests = {'r'}
        bank.arm({'r': plan})
        if mode != 'missing':
            graph.replay() if graph is not None else execute()
        if mode == 'nan': bank.lses[name].fill_(float('nan'))
        if mode == 'reference': bank.reasons[name, 'prefill'] = 'forced_reference_qualification'
    return action, captured, batch, prepare, execute, oracle, q, plan


def snapshot(packets):
    result = {}
    for rid, packet in packets.items():
        packet.event.synchronize()
        result[rid] = dict(rows=tuple(packet.rows), history=tuple(packet.history),
            masses=packet.masses.clone(), prefill=packet.prefill,
            source_positions=packet.source_positions, history_begin=packet.history_begin,
            candidate_count=packet.candidate_count)
    return result


@pytest.mark.parametrize('mode', ['native', 'missing', 'nan', 'reference', 'no_rows'])
def test_actual_native_stage_matches_old_capture_and_fallback_with_bounded_downloads(tmp_path, mode):
    action, captured, batch, prepare, _, _, _, _ = fixture(tmp_path, mode=mode)
    prepare()
    baseline = action.capture(captured, batch)
    expected = snapshot(baseline)
    del baseline
    gc.collect()
    # A single score slot forces invalid-frame cancellation to drop all old
    # host/event aliases before the reference scorer may acquire its slot.
    action.download_pool = ActionDownloadPool(1024, 1)
    prepare()
    samples, packets = action.candidate_samples, action.packets
    assert action.stage_before_mtp(captured, batch)
    frame = action._mtp_capture_frame
    assert frame.ticket.dirty == (mode in ('native', 'missing', 'nan'))
    # Independent proposal kernels follow the complete-score event.
    torch.cuda._sleep(2_000_000)
    actual = snapshot(action.capture(captured, batch))
    assert actual.keys() == expected.keys()
    for rid in actual:
        for key in actual[rid]:
            if key == 'masses': assert torch.equal(actual[rid][key], expected[rid][key])
            else: assert actual[rid][key] == expected[rid][key]
    expected_count = 0 if mode == 'no_rows' else 5
    assert action.candidate_samples-samples == expected_count
    assert action.packets-packets == int(bool(expected_count))
    assert action.mtp_stage_stats.get('fallback_batches', 0) == int(mode in ('missing', 'nan'))
    assert action.mtp_stage_stats['staged_batches'] == action.mtp_stage_stats['resolved_batches'] == 1
    assert action._mtp_capture_frame is None and action.native._validation_ticket is None
    assert action.download_pool.snapshot()['event_count'] <= 1


@pytest.mark.parametrize('mode', ['native', 'reference', 'no_rows'])
def test_resolve_early_full_score_event_does_not_wait_for_mtp_tail(tmp_path, mode):
    action, captured, batch, prepare, _, _, _, _ = fixture(tmp_path, mode=mode)
    prepare()
    assert action.stage_before_mtp(captured, batch)
    early = action._mtp_capture_frame.ticket.event
    torch.cuda._sleep(500_000_000)
    tail = torch.cuda.Event(); tail.record()
    result = action.capture(captured, batch)
    assert early.query()
    assert not tail.query(), 'Early completion unexpectedly fenced the queued MTP tail'
    assert all(packet.event.query() for packet in result.values())
    tail.synchronize()


def test_consecutive_graph_replays_do_not_reuse_stale_flags_or_score_packets(tmp_path):
    action, captured, batch, prepare, execute, oracle, q, _ = fixture(tmp_path)
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): execute()
    previous = None
    for marker in (.5, 1.75):
        q.mul_(marker)
        action.native.lses['model.layers.3.attn'].fill_(float('nan'))
        prepare(graph=graph)
        assert action.stage_before_mtp(captured, batch)
        result = action.capture(captured, batch)['r']
        assert result.event.query()
        torch.testing.assert_close(result.masses, oracle().cpu(), rtol=.02, atol=2e-4)
        if previous is not None: assert not torch.equal(result.masses, previous)
        previous = result.masses.clone()
        del result
    assert action.mtp_stage_stats == {'staged_batches': 2, 'resolved_batches': 2}
    assert action.native.validation_failures == 0


@pytest.mark.parametrize('accepted', [0, 2, 5])
def test_same_prefill_acceptance_and_scores_after_early_submission(tmp_path, accepted):
    action, captured, batch, prepare, _, _, _, _ = fixture(tmp_path)
    outcomes = []
    for early in (False, True):
        prepare()
        if early: assert action.stage_before_mtp(captured, batch)
        packet = action.capture(captured, batch)['r']
        packet.event.synchronize()
        history = PrefillHistory(action.options)
        end, count, starts, local = packet.prefill
        changed = history.accept(end, count, accepted, packet.rows, starts, packet.history, local, packet.masses)
        outcomes.append((changed, history.snapshot(), history.scores()))
        del packet
    assert outcomes[0] == outcomes[1]
    assert outcomes[1][0] is (accepted == 5)


def test_cancel_completed_private_download_reuses_slot_after_real_dma(tmp_path):
    action, captured, batch, prepare, _, _, _, _ = fixture(tmp_path)
    action.download_pool = ActionDownloadPool(1024, 1)
    prepare()
    assert action.stage_before_mtp(captured, batch)
    slot = action.download_pool.entries[0]
    action.cancel_before_mtp()
    assert action.candidate_samples == action.packets == 0
    assert action._mtp_capture_frame is None and slot.cpu_done
    prepare()
    assert action.stage_before_mtp(captured, batch)
    result = action.capture(captured, batch)
    assert action.download_pool.entries[0] is slot and all(p.event.query() for p in result.values())
    assert action.mtp_stage_stats == {'staged_batches': 2, 'cancelled_batches': 1, 'resolved_batches': 1}


def test_injected_wait_failure_keeps_real_gpu_download_and_ticket_leased(tmp_path):
    action, captured, batch, prepare, _, _, _, _ = fixture(tmp_path)
    prepare()
    assert action.stage_before_mtp(captured, batch)
    frame = action._mtp_capture_frame
    original = frame.ticket.event
    calls = []
    def failed_wait():
        calls.append(True)
        raise RuntimeError('injected wait failure after real CUDA submission')
    frame.ticket.event = NS(synchronize=failed_wait)
    try:
        with pytest.raises(RuntimeError, match='injected wait'):
            action.capture(captured, batch)
        with pytest.raises(RuntimeError, match='quarantined'):
            action.cancel_before_mtp()
        assert len(calls) == 1
        assert action._mtp_capture_frame is frame and frame.packets
        assert action.download_pool.snapshot()['active_pinned_bytes'] > 0
        with pytest.raises(RuntimeError, match='unresolved'):
            action.prepare(batch, captured[0].positions)
    finally:
        # Fixture-only cleanup; production preserves the failed frame/leases.
        original.synchronize()
