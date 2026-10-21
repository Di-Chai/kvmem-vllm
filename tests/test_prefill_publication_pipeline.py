"""Real CPU ownership checks for bounded one-chunk-ahead prefill publication."""
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from vllm_kvmem.archive_admission import ArchiveAdmission
from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.terminal import TerminalPrefix
from test_request_publication import fixture, packet, close
from test_terminal_prefix import request


def enabled_fixture(tmp_path, *, threshold=4):
    runtime, a, b, archive = fixture(tmp_path, threshold=threshold)
    # These CPU fixtures intentionally do not construct CUDA ActionRuntime.
    # Exercise the runtime gates independently of serving option qualification.
    values = vars(runtime.options).copy()
    values['prefill_publication_pipeline'] = True
    runtime.options = SimpleNamespace(**values)
    return runtime, a, b, archive


def mark_pipeline(value, *request_ids):
    value.pipeline_requests = frozenset(request_ids or value.request_ids)
    return value


def test_pipeline_public_option_is_opt_in_and_rejects_unqualified_lifetimes():
    from test_options import config
    assert not RuntimeOptions().prefill_publication_pipeline
    payload = dict(prefill_publication_pipeline=True, async_cpu_scatter=True,
        split_capture_dma=True, action_attention=True, prefill_attention=True,
        session_capacity=1, session_checkpoint_bytes=4096, async_archive_retirement=True)
    assert RuntimeOptions.from_vllm(config({'kvmem': payload})).prefill_publication_pipeline
    for key, value in [('prefill_publication_pipeline', 1), ('async_cpu_scatter', False),
                       ('split_capture_dma', False), ('action_attention', False),
                       ('prefill_attention', False), ('early_capture_slot_release', True)]:
        with pytest.raises(ValueError, match='pipeline'):
            RuntimeOptions.from_vllm(config({'kvmem': {**payload, key: value}}))


def test_unconfirmed_eligible_packet_can_schedule_but_cannot_publish(tmp_path):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    pending = mark_pipeline(packet(runtime, [('a', 4, 4)], confirmed=False))
    try:
        # Feedback is consumed by the NEXT scheduler decision, which still
        # installs terminal clipping before it authorizes this old sample.
        assert runtime.publication_blocked_requests() == set()
        assert not runtime.prefill_pipeline_can_advance('a')
        assert not runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 4 and pending.download.owners
        runtime.begin_step()
        assert runtime.prefill_pipeline_can_advance('a')
        assert not runtime.publication_ready({'a'})
        assert runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 8
    finally:
        close(runtime, (a, b), archive)


@pytest.mark.parametrize('second_owner', ['a', 'b'])
def test_two_fixed_slots_bound_lookahead_even_across_owners(tmp_path, second_owner):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    first = mark_pipeline(packet(runtime, [('a', 4, 4)]))
    second = mark_pipeline(packet(runtime, [(second_owner, 8 if second_owner == 'a' else 4, 4)]))
    try:
        assert runtime.publication_blocked_requests() == {'a', second_owner}
        if second_owner == 'a':
            assert not runtime.prefill_pipeline_can_advance('a')
        assert len(runtime.downloads.pending) == len(runtime.downloads.buffers) == 2
        assert runtime.drain_requests({'a', 'b'}, wait=True)
        assert runtime.released == [first.download, second.download]
        assert runtime.publication_blocked_requests() == set()
    finally:
        close(runtime, (a, b), archive)


def test_nonpipeline_and_decode_owners_keep_original_fences(tmp_path):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    mixed = mark_pipeline(packet(runtime, [('a', 4, 4), ('b', 4, 4)]), 'a')
    try:
        assert runtime.publication_blocked_requests() == {'b'}
        assert runtime.prefill_pipeline_can_advance('a')
        assert not runtime.prefill_pipeline_can_advance('b')
        # Classification of CPU work must NOT use the relaxed scheduling gate:
        # otherwise a zero-forward poll would execute a large append in front.
        assert runtime._small_pending_requests() == set()
        assert not mixed.published
        assert runtime.drain_requests({'a', 'b'}, wait=True)
    finally:
        close(runtime, (a, b), archive)


def test_background_fifo_preserves_old_capacity_until_semantic_commit(tmp_path, monkeypatch):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    first = mark_pipeline(packet(runtime, [('a', 4, 4)]))
    entered, release = Event(), Event()
    original = a.memory.begin_append

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        if not entered.is_set():
            entered.set()
            assert release.wait(10)
        return result

    monkeypatch.setattr(a.memory, 'begin_append', delayed)
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        frozen = runtime.archive_capacity()['a']
        assert frozen.history_end == 4
        assert runtime.prefill_pipeline_can_advance('a')
        second = mark_pipeline(packet(runtime, [('a', 8, 4)], confirmed=False))
        assert not runtime.prefill_pipeline_can_advance('a')
        assert runtime.publication_blocked_requests() == {'a'}
        assert not runtime.drain_requests({'a'}, wait=False)
        assert not second.consuming and not second.published
        assert runtime.archive_capacity()['a'] == frozen
        # CPU archive credits charge BOTH suffix chunks beyond the published
        # watermark, including a partial block, rather than inventing capacity.
        gate = ArchiveAdmission(runtime.config.block_tokens, frozen.ready_records)
        gate.update(0, {'a': frozen}, {'a'})
        pending_blocks = (8 + runtime.config.block_tokens - 1) // runtime.config.block_tokens
        assert gate.allows('a', 8, 4) == (pending_blocks <= frozen.ready_records)
        release.set()
        runtime.join_cpu_scatter()
        assert a.history_end == 8 and runtime.released == [first.download]
        assert not second.published and second.download.owners
        runtime.begin_step()
        assert runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 12 and runtime.released == [first.download, second.download]
        assert runtime.last_steps['a'][-1]['history_begin'] == 8
        assert runtime.archive_capacity()['a'].history_end == 12
    finally:
        release.set()
        close(runtime, (a, b), archive)


def test_cancel_with_one_leading_chunk_clips_before_cpu_publication(tmp_path):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    first = mark_pipeline(packet(runtime, [('a', 4, 4)]))
    second = mark_pipeline(packet(runtime, [('a', 8, 4)], confirmed=False))
    try:
        prefix = TerminalPrefix.from_request(request(10, 0, 16, 0, 'FINISHED_ABORTED'))
        runtime.begin_step({'a'}, {'a': prefix})
        assert not runtime.prefill_pipeline_can_advance('a')
        runtime.finish({'a'}, {'a': prefix})
        report = runtime.completed[0]
        assert report['history_end'] == 10
        assert report['terminal_discarded_tokens'] == 2
        assert runtime.released == [first.download, second.download]
        assert runtime.last_steps.get('a') is None
    finally:
        close(runtime, (b,), archive)


def test_graph_budget_failure_does_not_publish_either_queued_chunk(tmp_path):
    runtime, a, b, archive = enabled_fixture(tmp_path)
    graph = a.block_graph = BlockGraph('pipeline-a', runtime.config.block_tokens, initial_position=4)
    first = mark_pipeline(packet(runtime, [('a', 4, 4)]))
    second = mark_pipeline(packet(runtime, [('a', 8, 4)]))
    before = graph.binding, graph.bytes, a.history_end, runtime.accepted['a']
    graph.max_bytes_per_session = graph.bytes
    try:
        with pytest.raises(BufferError, match='memory capacity'):
            runtime.drain_requests({'a'}, wait=True)
        assert (graph.binding, graph.bytes, a.history_end, runtime.accepted['a']) == before
        assert not first.published and not second.published and not second.consuming
        assert first.download.owners and second.download.owners and not runtime.released
        with pytest.raises(BufferError, match='memory capacity'):
            runtime.prefill_pipeline_can_advance('a')
    finally:
        # Failed publication quarantines both sources; test-owned fatal teardown.
        runtime.queue.clear()
        runtime.downloads.pending.clear()
        close(runtime, (a, b), archive)


def test_terminal_deferral_does_not_treat_pipeline_owner_as_tiny(tmp_path, monkeypatch):
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_kvmem.connector import ControlMetadata
    runtime, a, b, archive = enabled_fixture(tmp_path)
    mark_pipeline(packet(runtime, [('a', 4, 4)], confirmed=False))
    prefix = TerminalPrefix.from_request(request(8, 0, 16, 0, 'FINISHED_ABORTED'))
    output = replace(SchedulerOutput.make_empty(), finished_req_ids={'a'},
        kv_connector_metadata=ControlMetadata(terminal_prefixes={'a': prefix}))
    original = runtime.drain_requests
    seen = []

    def observe(request_ids, wait=False):
        seen.append(set(request_ids))
        return original(request_ids, wait)

    monkeypatch.setattr(runtime, 'drain_requests', observe)
    try:
        runtime.begin_step({'a'}, {'a': prefix})
        held = runtime.defer_terminals(output)
        assert seen == [set()]
        assert held.finished_req_ids == set()
        assert a.history_end == 4
        runtime.finish({'a'}, {'a': prefix})
    finally:
        close(runtime, (b,), archive)
