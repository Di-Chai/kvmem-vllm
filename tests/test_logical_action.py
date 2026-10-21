"""CPU qualification of the opt-in logical projection and original consumers."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.action_attention import ActionPacket, ActionRuntime
from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.logical_action import LogicalScorePlan, plan_logical_scores, project_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.prefill_attention import PrefillHistory
from vllm_kvmem.request_plan import RequestPlan


def fixture(dtype=np.float32, *, chunks=2):
    spans = tuple(LogicalPrefillSpan(i, 8192+i*2048, 64+i*2048, 2048, i*2048)
                  for i in range(chunks))
    group = LogicalPrefillGroup(request_id='r', owner_epoch=(1, 0, 0), scheduler_step=7,
        offset=8128, prompt_tokens=max(16384, 8192+(chunks+1)*2048), capacity=49152, spans=spans,
        request_plan=RequestPlan(None, block_tokens=32, gen_budget=16384),
        coalescing_authorization_id=None)
    plans = tuple(LogicalScorePlan(span, tuple(range(31, 2048, 32)),
        tuple(range(0, span.local_end, 32)), (32,)*(span.local_end//32),
        (0, 4096, *range(8192, span.history_end, 32))) for span in spans)
    physical_rows = tuple(span.physical_offset+row for span, plan in zip(spans, plans) for row in plan.rows)
    masses = np.zeros((64*chunks, 2+64*chunks), dtype=dtype)
    for index in range(64*chunks):
        masses[index, 0] = .125 if index % 2 else .25
        masses[index, 1] = .25
        masses[index, 2+index] = 1-masses[index, 0]-masses[index, 1]
    return group, plans, physical_rows, masses


def state(native):
    options = NS(prefill_attention_half_life=8192, native_prefill_commit=native,
        prefill_attention_stride=32, prefill_attention_interval=2048,
        memory=NS(select_budget=32768, gen_budget=16384, block_tokens=32))
    history = PrefillHistory(options)
    history.values = {0: [.25, 1.], 4096: [.75, 1.]}
    history.end, history.samples, history.chunks = 8192, 64, 4
    return NS(prefill=history, prefill_replay=None)


def consumer(tmp_path):
    action = ActionRuntime.__new__(ActionRuntime)
    action.runtime = NS(output=tmp_path)
    graph = BlockGraph('owner', 32, topk=4)
    with graph.prepare(0, 8192) as update:
        update.commit()
    return action, graph


def publish(action, graph, pending, accepted):
    packet = NS(action_inputs={'r': pending})
    prepared = action.prepare_accept(packet, 'r', accepted, graph=graph,
                                    history_begin=pending.history_begin)
    prepared.commit()
    assert packet.action_inputs == {}
    return (pending.state.prefill.snapshot(), graph.rows, graph.partial,
            graph.accepted_through, graph.completed_through, graph.revision)


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('retained', [4096, 2048, 2049, 0])
def test_original_logical_acceptance_is_bitwise_identical_after_every_transaction(tmp_path, native, dtype, retained):
    group, plans, physical_rows, masses = fixture(dtype)
    projected = project_logical_scores(group, plans, masses, physical_rows=physical_rows)
    old_state, new_state = state(native), state(native)
    old_action, old_graph = consumer(tmp_path)
    new_action, new_graph = consumer(tmp_path)
    try:
        for i, (plan, selected) in enumerate(zip(plans, projected)):
            span = plan.span
            # Independent original 2K D2H payload, with the original column
            # domain. The new path is a view of one physical 4K payload.
            old_mass = masses[i*64:(i+1)*64, :len(plan.history)].copy()
            original = ActionPacket(old_state, list(plan.rows), list(plan.history), old_mass, None,
                (span.history_end, span.count, plan.starts, span.local_begin),
                tuple(span.history_begin+r for r in plan.rows), span.history_begin, span.count)
            accepted = min(2048, max(0, retained-i*2048))
            assert selected.accepted_from_prefix(retained) == accepted
            assert publish(old_action, old_graph, original, accepted) == publish(
                new_action, new_graph, selected.action_packet(new_state), accepted)
        assert new_state.prefill.chunks == 4 + retained//2048
        assert new_state.prefill.samples == 64 + 64*(retained//2048)
        assert new_graph.accepted_through == 8192+retained
    finally:
        old_graph.close()
        new_graph.close()


@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_prefix_projection_keeps_exact_domain_and_no_matrix_copy(backend):
    group, plans, rows, masses = fixture()
    if backend == 'torch':
        masses = torch.from_numpy(masses)
    event = object()  # It has no query/synchronize API; projection must not wait.
    parts = project_logical_scores(group, plans, masses, physical_rows=rows, event=event)
    assert [tuple(part.masses.shape) for part in parts] == [(64, 66), (64, 130)]
    assert 10240 not in parts[0].plan.history and 10240 in parts[1].plan.history
    for part in parts:
        assert part.event is event and part.source_owner is masses
        if backend == 'numpy':
            assert np.shares_memory(part.masses, masses)
        else:
            assert part.masses.untyped_storage().data_ptr() == masses.untyped_storage().data_ptr()
    masses[0, 0] = .375
    assert parts[0].masses[0, 0] == .375


def test_projection_preserves_noncontinuous_logical_history_and_duplicate_partial_segments():
    group, plans, rows, masses = fixture()
    # A reordered historical active view with two segments from the same
    # original page must retain both identities and their exact column order.
    adjusted = tuple(replace(plan, starts=(0, 16, *plan.starts[1:]),
        counts=(16, 16, *plan.counts[1:]), history=(4096, 4096, 0, *plan.history[2:])) for plan in plans)
    physical = np.column_stack((masses[:, 0]/2, masses[:, 0]/2, masses[:, 1:]))
    parts = project_logical_scores(group, adjusted, physical, physical_rows=rows)
    assert parts[0].plan.history[:3] == (4096, 4096, 0)
    assert parts[0].plan.counts[:3] == (16, 16, 32)
    assert parts[0].masses.shape == (64, 67)


@pytest.mark.parametrize('corruption', ['rows', 'columns', 'order', 'identity', 'block', 'policy', 'domain',
                                         'column_stride', 'unaligned_stride', 'row_overlap', 'dtype', 'list', 'grad'])
def test_all_projection_geometry_is_checked_before_original_state_can_change(tmp_path, corruption):
    group, plans, rows, masses = fixture()
    original = state(True)
    before = original.prefill.snapshot()
    if corruption == 'rows': masses = masses[:-1]
    elif corruption == 'columns': masses = masses[:, :-1]
    elif corruption == 'order': rows = (*rows[:-2], rows[-1], rows[-2])
    elif corruption == 'identity':
        plans = (plans[0], replace(plans[1], span=replace(plans[1].span,
            history_begin=plans[1].span.history_begin+32)))
    elif corruption == 'block':
        plans = (replace(plans[0], block_tokens=64,
            history=tuple(key-key%64 for key in plans[0].history)), plans[1])
    elif corruption == 'policy':
        group = replace(group, request_plan=replace(group.request_plan, block_tokens=64))
    elif corruption == 'domain': plans = (replace(plans[0], history=(4096, 0, *plans[0].history[2:])), plans[1])
    elif corruption == 'column_stride': masses = np.zeros((128, 260), dtype=np.float32)[:, ::2]
    elif corruption == 'unaligned_stride':
        masses = np.ndarray((128, 130), dtype=np.float32, buffer=bytearray(128*130*5),
                            strides=(130*5, 5))
    elif corruption == 'row_overlap': masses = np.broadcast_to(masses[:1], masses.shape)
    elif corruption == 'dtype': masses = masses.astype(np.int32)
    elif corruption == 'list': masses = masses.tolist()
    elif corruption == 'grad': masses = torch.from_numpy(masses).requires_grad_()
    with pytest.raises(ValueError):
        result = project_logical_scores(group, plans, masses, physical_rows=rows)
        # A later malformed span must never reach this caller's first mutation.
        first = result[0]
        original.prefill.accept(first.plan.span.history_end, 2048, 2048, first.plan.rows,
            first.plan.starts, first.plan.history, first.plan.span.local_begin, first.masses)
    assert original.prefill.snapshot() == before


def test_two_logical_packets_and_exported_views_retain_one_pool_lease_without_gc():
    from test_action_download_pool import Backend
    from vllm_kvmem.action_download_pool import ActionDownloadPool
    group, plans, rows, source = fixture()
    pool = ActionDownloadPool(131072, 1, backend=Backend())
    host, event = pool.copy(torch.from_numpy(source))
    event.synchronize()
    pointer = host.data_ptr()
    parts = project_logical_scores(group, plans, host, physical_rows=rows, event=event)
    first, second = (part.action_packet(state(True)) for part in parts)
    exported = second.masses.numpy()
    del parts, host, event
    with pytest.raises(BufferError): pool.copy(torch.from_numpy(source))
    del first, second
    with pytest.raises(BufferError): pool.copy(torch.from_numpy(source))
    del exported
    host, event = pool.copy(torch.from_numpy(source))
    assert host.data_ptr() == pointer and pool.snapshot()['reuses'] == 1
    assert pool.snapshot()['event_count'] == 1


@pytest.mark.parametrize('retained', [-1, 4097, True, 1.5])
def test_acceptance_rejects_a_prefix_outside_the_physical_packet(retained):
    group, plans, rows, masses = fixture()
    part = project_logical_scores(group, plans, masses, physical_rows=rows)[0]
    with pytest.raises(ValueError, match='physical group'):
        part.accepted_from_prefix(retained)


@pytest.mark.parametrize('interval', [2048, 4096, 8192])
def test_plan_uses_two_original_extend_calls_and_does_not_advance_state(interval):
    from vllm_kvmem.prefill_planning import freeze_prefill_frame
    group, expected, _, _ = fixture()
    current = state(True)
    current.prefill.options.prefill_attention_interval = interval
    current.options = current.prefill.options
    authority = object()
    frame = freeze_prefill_frame(authority, current, group.owner_epoch, 8192, 64,
                                [(0, 0, 0, 32, 0), (1, 0, 4096, 32, 32)])
    before = current.prefill.snapshot()
    plans, rows, successor = plan_logical_scores(group, frame)
    original = frame
    for plan, span in zip(plans, group.spans):
        old_rows, starts, counts, history, original = original.extend(
            span.history_begin, span.local_begin, 2048, 32)
        assert (plan.rows, plan.starts, plan.counts, plan.history) == tuple(
            tuple(value) for value in (old_rows, starts, counts, history))
    assert successor == original and successor.history_end == 12288
    assert frame.history_end == 8192 and frame.local_end == 64
    assert current.prefill.snapshot() == before
    assert rows == tuple(span.physical_offset+r for span, plan in zip(group.spans, plans) for r in plan.rows)
    if interval == 2048:
        assert plans == expected
    if interval == 4096:
        assert plans[0].rows == () and len(plans[1].rows) == 64
    if interval == 8192:
        assert rows == ()
    with pytest.raises(ValueError, match='stale prefix'):
        plan_logical_scores(group, successor)


@pytest.mark.parametrize('backends', [('fused', 'fused'), ('reference', 'fused'), ('fused', 'reference')])
def test_reference_projection_preserves_actual_scorer_backend_through_pad_and_cat(monkeypatch, backends):
    from vllm_kvmem.logical_action import reference_logical_scores
    group, plans, rows, _ = fixture()
    calls = []
    def score(query, positions, key, table, trig, key_scale, starts, counts, length, **kwargs):
        assert (query.shape[0], length) == (64, plans[len(calls)].span.local_end)
        values = torch.zeros((64, starts.numel()), dtype=torch.float32)
        values[:, 0] = 1
        values._action_scoring_backend = backends[len(calls)]
        calls.append(length)
        return values
    monkeypatch.setattr('vllm_kvmem.action_scoring.page_attention', score)
    result = reference_logical_scores(group, plans, torch.zeros((128, 1, 4)),
        torch.tensor([64+row for row in rows]), None, None, None, None,
        torch.tensor(plans[-1].starts), torch.tensor(plans[-1].counts),
        scale=.5, q_dtype=torch.bfloat16, q_scale=None, block_tokens=32, fused=True)
    assert result._action_scoring_backend == ('fused' if set(backends) == {'fused'} else 'reference')
    assert calls == [2112, 4160] and result.shape == (128, 130)
    assert not result[:64, 66:].any()
