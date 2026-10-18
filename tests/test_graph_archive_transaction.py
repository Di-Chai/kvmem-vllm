"""Graph admission precedes native raw-KV acceptance, across worker handoffs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.terminal import TerminalPrefix
from test_graph_capture import action, history, output_packet
from test_request_publication import fixture, packet, close


def install(runtime, owner, request_id, tmp_path, count=1):
    owner.block_graph = BlockGraph(request_id, 4, initial_position=4)
    runtime.action = action(tmp_path)
    state = owner.action_history = history()
    shared = packet(runtime, [(request_id, 4, count)])
    shared.action_inputs[request_id] = output_packet(state, [20] * count, 4,
        [0, 4], [[.5, .5]] * count).action_inputs['r']
    return shared, state


def discard(runtime):
    runtime.queue.clear()
    runtime.downloads.pending.clear()


def test_capacity_failure_is_before_raw_history_parser_legacy_and_graph(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    shared, state = install(runtime, a, 'a', tmp_path)
    graph = a.block_graph
    before = (a.history_end, state.parser.state, state.mass.copy(), state.samples,
              state.prefill.snapshot(), graph.binding, graph.bytes)
    graph.max_bytes_per_session = graph.bytes
    with pytest.raises(BufferError, match='capacity'):
        runtime._publish_claim(shared, [0], False)
    assert (a.history_end, state.parser.state, state.mass, state.samples,
            state.prefill.snapshot(), graph.binding, graph.bytes) == before
    assert not shared.graph_updates and graph._pending_update is None
    assert 'a' in shared.action_inputs and runtime.accepted['a'] == 4
    discard(runtime)
    close(runtime, (a, b), archive)


def test_later_owner_capacity_rejects_before_any_batch_raw_append(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    a.block_graph, b.block_graph = BlockGraph('a', 4, initial_position=4), BlockGraph('b', 4, initial_position=4)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    baseline = a.block_graph.bytes, b.block_graph.bytes
    b.block_graph.max_bytes_per_session = b.block_graph.bytes
    with pytest.raises(BufferError, match='capacity'):
        runtime._copy_source_owners(shared, [0, 1], False)
    assert a.history_end == b.history_end == 4
    assert (a.block_graph.bytes, b.block_graph.bytes) == baseline
    assert a.block_graph._pending_update is b.block_graph._pending_update is None
    assert not shared.graph_updates
    discard(runtime)
    close(runtime, (a, b), archive)


def test_archive_backpressure_aborts_unconsumed_reservation_and_releases_lock(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    a.block_graph = BlockGraph('a', 4, initial_position=4)
    shared = packet(runtime, [('a', 4, 1)])
    from vllm_kvmem import authority
    monkeypatch.setattr(authority, 'append_authority_batch', lambda entries: 0)
    assert runtime._copy_source_owners(shared, [0], False) == []
    assert not shared.graph_updates and a.block_graph._pending_update is None
    assert a.block_graph.bytes == a.block_graph.live_bytes
    with ThreadPoolExecutor(max_workers=1) as executor:
        saved = executor.submit(a.block_graph.snapshot).result(timeout=3)
        saved.close()
    assert a.history_end == a.block_graph.accepted_through == 4
    discard(runtime)
    close(runtime, (a, b), archive)


def test_append_exception_aborts_every_preflight_reservation(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    a.block_graph, b.block_graph = BlockGraph('a', 4, initial_position=4), BlockGraph('b', 4, initial_position=4)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    from vllm_kvmem import authority
    def failure(entries):
        raise RuntimeError('native append failed before committing')
    monkeypatch.setattr(authority, 'append_authority_batch', failure)
    with pytest.raises(RuntimeError, match='native append'):
        runtime._copy_source_owners(shared, [0, 1], False)
    assert not shared.graph_updates
    assert all(g._pending_update is None and g.bytes == g.live_bytes
               for g in (a.block_graph, b.block_graph))
    assert a.history_end == b.history_end == 4
    discard(runtime)
    close(runtime, (a, b), archive)


def test_early_copy_can_publish_on_other_thread_without_duplicate_parser_commit(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    runtime.options = replace(runtime.options, early_capture_slot_release=True)
    shared, state = install(runtime, a, 'a', tmp_path)
    baseline = state.parser.state, state.mass.copy(), state.samples
    assert runtime._copy_slot_owner_prefix('bulk')
    assert a.history_end == 5 and a.block_graph.accepted_through == 4
    assert shared.source_released and set(shared.graph_updates) == {0}
    assert (state.parser.state, state.mass, state.samples) == baseline
    reservation = shared.graph_updates[0].graph_update
    assert not reservation._locked
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(runtime.drain_requests, {'a'}, True).result(timeout=3)
    assert a.block_graph.accepted_through == 5 and a.block_graph.partial.samples == 1
    assert state.samples == 1 and not shared.graph_updates
    assert a.block_graph._pending_update is None
    close(runtime, (a, b), archive)


def test_partial_native_batch_keeps_only_consumed_graph_reservations(tmp_path, monkeypatch):
    runtime, a, b, archive = fixture(tmp_path)
    runtime.options = replace(runtime.options, early_capture_slot_release=True)
    a.block_graph, b.block_graph = BlockGraph('a', 4, initial_position=4), BlockGraph('b', 4, initial_position=4)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    from vllm_kvmem import authority
    original = authority.append_authority_batch
    monkeypatch.setattr(authority, 'append_authority_batch', lambda entries: original(entries[:1]))
    assert runtime._copy_slot_owner_prefix('bulk')
    assert set(shared.graph_updates) == set(shared.source_copied) == {0}
    assert a.history_end == 5 and b.history_end == 4
    assert a.block_graph._pending_update is not None and b.block_graph._pending_update is None
    assert b.block_graph.bytes == b.block_graph.live_bytes
    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(runtime.drain_requests, {'a'}, True).result(timeout=3)
    assert runtime.drain_requests({'b'}, wait=True)
    assert a.block_graph.accepted_through == b.block_graph.accepted_through == 5
    assert not shared.graph_updates
    close(runtime, (a, b), archive)


def test_epoch_change_rejects_detached_feedback_before_parser_or_query(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    runtime.options = replace(runtime.options, early_capture_slot_release=True)
    shared, state = install(runtime, a, 'a', tmp_path)
    assert runtime._copy_slot_owner_prefix('bulk')
    baseline = state.parser.state, state.mass.copy(), state.samples
    a.block_graph.invalidate()
    query_calls = []
    runtime._accept_query_input = lambda *args: query_calls.append(args)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match='Stale'):
            executor.submit(runtime._publish_claim, shared, [0], False).result(timeout=3)
    assert not query_calls and (state.parser.state, state.mass, state.samples) == baseline
    assert not shared.graph_updates and a.block_graph._pending_update is None
    assert a.block_graph.bytes == a.block_graph.live_bytes
    # A later branch invalidation does not roll back already copied raw bytes;
    # the stale semantic transaction is aborted and the failed owner retired.
    assert a.history_end == 5 and a.block_graph.accepted_through == 4
    discard(runtime)
    close(runtime, (a, b), archive)


def test_wholly_discarded_later_packet_does_not_advance_graph_watermark(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    a.block_graph = BlockGraph('a', 4, initial_position=4)
    runtime.action = action(tmp_path)
    state = a.action_history = history()
    shared = packet(runtime, [('a', 8, 4)])
    shared.action_inputs['a'] = output_packet(state, [20] * 4, 8, [0, 8],
        [[.5, .5]] * 4).action_inputs['r']
    runtime.terminal_prefixes['a'] = TerminalPrefix(4, 12, 4, 'FINISHED_ABORTED')
    assert runtime.drain_requests({'a'}, wait=True)
    assert a.history_end == a.block_graph.accepted_through == 4
    assert state.samples == 0 and runtime.terminal_discarded['a'] == 4
    assert not shared.graph_updates
    close(runtime, (a, b), archive)


def test_graph_close_aborts_detached_pending_charge_from_another_thread():
    graph = BlockGraph('a', 4)
    update = graph.prepare(0, 4)
    update.detach()
    assert graph.bytes > graph.live_bytes
    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(graph.close).result(timeout=3)
    assert update.done and graph.bytes == graph.budget.used == 0
    assert update.data is None


def test_explicit_release_marker_skips_new_graph_but_unowned_requests_still_reject(tmp_path):
    from types import SimpleNamespace
    runtime, a, b, archive = fixture(tmp_path)
    runtime.options = replace(runtime.options, graph_mode='collect',
        memory=replace(runtime.options.memory, repository_blocks=32),
        session_capacity=2, session_checkpoint_bytes=1 << 20, async_archive_retirement=True)
    runtime.config = runtime.options.memory
    runtime.runner._kvmem_sessions = SimpleNamespace(bindings={}, release_requests={'explicit-close'})
    runtime.runner._kvmem_layers = ('layer0', 'layer1')
    runtime.gc_config = runtime.gc_progress = None
    with pytest.raises(ValueError, match='owned session'):
        runtime.ensure_authority('ordinary-unowned')
    assert 'ordinary-unowned' not in runtime.authorities
    closing = runtime.ensure_authority('explicit-close')
    assert not hasattr(closing, 'block_graph')
    close(runtime, (a, b, closing), archive)


def test_graph_off_packet_does_not_allocate_graph_metadata(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('a', 4, 1)])
    assert shared.graph_updates is None
    assert runtime.drain_requests({'a'}, wait=True)
    assert shared.graph_updates is None
    close(runtime, (a, b), archive)
