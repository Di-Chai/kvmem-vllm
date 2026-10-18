"""Two-phase DMA ownership with actual CPU authority copies, no GPU launch."""
from dataclasses import replace
import json
from threading import Event, current_thread
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.terminal import TerminalPrefix
from test_request_publication import fixture, packet, close
from test_authority import drain, setup


def enabled(tmp_path, *, budget=268435456):
    runtime, a, b, archive = fixture(tmp_path)
    runtime.options = replace(runtime.options, early_capture_slot_release=True,
                              early_capture_feedback_bytes=budget)
    runtime._early_owner_limit = 2
    runtime.downloads.buffers = [object()]  # Force source pressure in one shared slot.
    return runtime, a, b, archive


def cleanup_failed(runtime, owners, archive):
    executor = getattr(runtime, '_cpu_scatter_executor', None)
    if executor:
        executor.shutdown(wait=True)
    for owner in owners:
        drain(owner)
        assert owner.close()
    assert archive.close()


def test_reclaim_copies_all_owners_without_unrelated_score_or_early_ready(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    c, _ = setup(tmp_path/'c', 'c', archive)
    runtime.authorities['c'] = c
    runtime.accepted['c'] = runtime.rejected['c'] = 0
    runtime.last_steps['c'] = []
    shared = packet(runtime, [('b', 4, 4), ('a', 4, 1)])
    with runtime._publication_lock():
        runtime._early_source_submitted(shared)
    query = torch.arange(8, dtype=torch.float32)
    masses = torch.arange(4, dtype=torch.float32)
    shared.query_inputs['b'] = (object(), query)
    shared.action_inputs['b'] = NS(masses=masses)
    entered, proceed = Event(), Event()
    score_threads, observed = [], []

    def score(p, rid):
        value = p.query_inputs.pop(rid, None)
        if value:
            score_threads.append(current_thread().name)
            entered.set()
            assert proceed.wait(5)
            observed.append(value[1].clone())
            assert torch.equal(p.action_inputs[rid].masses, masses)
    runtime._accept_query_input = score
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert set(shared.source_copied) == {0} and not shared.source_released
        runtime.ensure_download_slot({'new-owner'})
        assert shared.source_released and set(shared.source_copied) == {0, 1}
        assert runtime.downloads.pending == {} and runtime.released == [shared.download]
        assert not runtime.publication_ready({'a', 'b'})
        assert runtime.publication_blocked_requests() == {'a', 'b'}
        assert runtime.archive_capacity()['b'].history_end == 4
        assert runtime.archive_capacity()['a'].history_end == 4
        # A different owner can use the reclaimed physical slot while b's
        # score is still blocked. Mutating old staging cannot affect feedback.
        shared.download.payload.fill_(255)
        other = packet(runtime, [('c', 0, 1)], confirmed=False)
        assert other.download.slot == shared.download.slot
        with runtime._publication_lock():
            runtime._early_source_submitted(other)
        events = [json.loads(line) for line in (tmp_path/'capture_feedback_events.jsonl').open()]
        new_submission = next(row for row in events if row['stage'] == 'source_submitted' and row['generation'] == 2)
        prior_release = next(row for row in events if row['stage'] == 'source_released' and row['generation'] == 1)
        score_copy = next(row for row in events if row['stage'] == 'source_copied' and 'b' in row['owner_storage_bytes'])
        assert score_copy['owner_storage_bytes']['b'] == {'query': 32, 'action': 16}
        assert new_submission['slot'] == prior_release['slot']
        assert new_submission['at'] >= prior_release['at']
        assert new_submission['pending_previous_generations'] == [1]
        assert not any(row['stage'] == 'semantic_complete' for row in events)
        metrics = runtime._publication_metrics_snapshot()
        assert metrics['main/bulk']['shared_source_reuse_before_semantic_packets'] == 1
        proceed.set()
        runtime.join_cpu_scatter()
        assert runtime.publication_ready({'b'})
        assert runtime.publication_ready({'a'})
        assert not runtime.publication_ready({'c'})  # unconfirmed different owner
        assert torch.equal(observed[0], query)
        assert runtime.accepted == {'a': 5, 'b': 8, 'c': 0}
        assert all(t.startswith('kvmem-scatter') for t in score_threads)
        assert len(runtime.released) == 1
        other.confirmed = True
        assert runtime.drain_requests({'c'}, wait=True)
        assert runtime.accepted['c'] == 1
        assert len(runtime.released) == 2 and not runtime._feedback_credits
    finally:
        proceed.set()
        close(runtime, (a, b, c), archive)


def test_partial_copy_retains_source_and_immutable_metadata(tmp_path, monkeypatch):
    runtime, a, b, archive = enabled(tmp_path)
    shared = packet(runtime, [('a', 4, 2), ('b', 4, 3)])
    from vllm_kvmem import authority
    append = authority.append_authority_batch
    monkeypatch.setattr(authority, 'append_authority_batch', lambda entries: append(entries[:1]))
    try:
        assert runtime._copy_slot_owner_prefix('bulk')
        assert set(shared.source_copied) == {0}
        assert not shared.source_released and not runtime.released
        assert not runtime.publication_ready({'a'})
        assert runtime._copy_slot_owner_prefix('bulk')
        assert shared.source_released and len(runtime.released) == 1
        shared.download.payload.fill_(255)
        assert runtime.drain_requests({'a', 'b'}, wait=True)
        assert runtime.accepted == {'a': 6, 'b': 7}
        assert runtime.last_steps['a'][-1]['history_begin'] == 4
        assert len(runtime.released) == 1
    finally:
        close(runtime, (a, b), archive)


@pytest.mark.parametrize('status', ['FINISHED_ABORTED', 'FINISHED_STOPPED'])
def test_terminal_clips_before_source_copy_and_never_publishes_discarded_rows(tmp_path, status):
    runtime, a, b, archive = enabled(tmp_path)
    shared = packet(runtime, [('a', 4, 4), ('b', 4, 1)], confirmed=False)
    try:
        assert not runtime._copy_slot_owner_prefix('bulk')
        runtime.begin_step({'a'}, {'a': TerminalPrefix(5, 8, 4, status)})
        assert runtime._copy_slot_owner_prefix('bulk')
        assert a.history_end == 5 and shared.source_released
        assert runtime.accepted['a'] == 4  # semantic fence still closed
        assert runtime.drain_requests({'a', 'b'}, wait=True)
        assert runtime.accepted['a'] == 5 and runtime.terminal_discarded['a'] == 3
        assert len(runtime.released) == 1
    finally:
        close(runtime, (a, b), archive)


def test_feedback_credit_bound_falls_back_to_owned_source_until_semantic_finish(tmp_path):
    runtime, a, b, archive = enabled(tmp_path, budget=8)
    shared = packet(runtime, [('a', 4, 1)])
    shared.query_inputs['a'] = (object(), torch.zeros(16))
    runtime._accept_query_input = lambda p, rid: p.query_inputs.pop(rid, None)
    try:
        assert runtime._copy_slot_owner_prefix('bulk')
        assert not shared.source_released and not runtime._feedback_credits
        assert not runtime.publication_ready({'a'})
        assert runtime.drain_requests({'a'}, wait=True)
        assert shared.source_released and len(runtime.released) == 1
        assert not runtime._feedback_credits
    finally:
        close(runtime, (a, b), archive)


def test_detached_packet_credit_is_reused_only_after_semantic_completion(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    first = packet(runtime, [('a', 4, 1)])
    try:
        assert runtime._copy_slot_owner_prefix('bulk') and first.source_released
        second = packet(runtime, [('b', 4, 1)])
        assert runtime._copy_slot_owner_prefix('bulk')
        assert not second.source_released
        snapshot = runtime._early_feedback_snapshot()
        assert snapshot['detached_packets'] == snapshot['packet_limit'] == 1
        assert runtime.drain_requests({'a'}, wait=True)
        assert runtime._copy_slot_owner_prefix('bulk') and second.source_released
        assert runtime._early_feedback_snapshot()['detached_packets'] == 1
        assert runtime.drain_requests({'b'}, wait=True)
        assert not runtime._feedback_credits and len(runtime.released) == 2
    finally:
        close(runtime, (a, b), archive)


def test_detached_owner_credit_counts_shared_packet_owners(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    runtime._early_owner_limit = 1
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    try:
        assert runtime._copy_slot_owner_prefix('bulk')
        assert not shared.source_released and not runtime._feedback_credits
        assert runtime.drain_requests({'a'}, wait=True)
        assert runtime._copy_slot_owner_prefix('bulk') and shared.source_released
        snapshot = runtime._early_feedback_snapshot()
        assert snapshot['detached_owners'] == snapshot['owner_limit'] == 1
        assert not runtime.publication_ready({'b'})
        assert runtime.drain_requests({'b'}, wait=True)
        assert not runtime._feedback_credits and len(runtime.released) == 1
    finally:
        close(runtime, (a, b), archive)


def test_failure_after_partial_copy_keeps_uncopied_shared_source_owned(tmp_path, monkeypatch):
    runtime, a, b, archive = enabled(tmp_path)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    from vllm_kvmem import authority
    append = authority.append_authority_batch
    monkeypatch.setattr(authority, 'append_authority_batch', lambda entries: append(entries[:1]))
    original = RuntimeError('second owner copy failed')
    def fail(*args, **kwargs):
        raise original
    try:
        assert runtime._copy_slot_owner_prefix('bulk')
        assert set(shared.source_copied) == {0}
        runtime._copy_source_owners = fail
        with pytest.raises(RuntimeError) as error:
            runtime._copy_slot_owner_prefix('bulk')
        assert error.value is original
        assert not shared.source_released and not runtime.released
        assert runtime.downloads.pending[shared.download.slot] is shared.download
        assert runtime.accepted == {'a': 4, 'b': 4}
        with pytest.raises(RuntimeError) as error:
            runtime.ensure_download_slot({'c'})
        assert error.value is original
    finally:
        cleanup_failed(runtime, (a, b), archive)


@pytest.mark.parametrize('stage', ['copy', 'score'])
def test_failure_preserves_exception_and_never_repeats_release_or_semantic_commit(tmp_path, stage):
    runtime, a, b, archive = enabled(tmp_path)
    shared = packet(runtime, [('a', 4, 1)])
    original = RuntimeError('owned '+stage+' failure')
    def fail(*args, **kwargs):
        raise original
    if stage == 'copy':
        runtime._copy_source_owners = fail
        operation = lambda: runtime._copy_slot_owner_prefix('bulk')
    else:
        runtime._accept_query_input = fail
        operation = lambda: runtime.drain_requests({'a'}, wait=True)
    try:
        with pytest.raises(RuntimeError) as error:
            operation()
        assert error.value is original
        for _ in range(2):
            with pytest.raises(RuntimeError) as error:
                runtime.drain_requests({'a'}, wait=True)
            assert error.value is original
        assert runtime.accepted['a'] == 4 and not shared.published
        assert len(runtime.released) == int(stage == 'score')
    finally:
        cleanup_failed(runtime, (a, b), archive)


def test_owner_fifo_prevents_copying_new_prefix_before_feedback_commit(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    first = packet(runtime, [('a', 4, 1)])
    try:
        assert runtime._copy_slot_owner_prefix('bulk') and first.source_released
        later = packet(runtime, [('a', 5, 1)])
        assert not runtime._copy_slot_owner_prefix('bulk')
        assert not later.source_copied and a.history_end == 5
        assert runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 6 and runtime.accepted['a'] == 6
        assert len(runtime.released) == 2
    finally:
        close(runtime, (a, b), archive)


def test_tiny_decode_keeps_foreground_publication_while_bulk_query_worker_is_blocked(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    bulk = packet(runtime, [('b', 4, 4)])
    bulk.query_inputs['b'] = (object(), torch.zeros(8))
    entered, proceed = Event(), Event()
    def score(p, rid):
        if p.query_inputs.pop(rid, None) is not None:
            entered.set()
            assert proceed.wait(5)
    runtime._accept_query_input = score
    try:
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert bulk.source_released and not runtime.publication_ready({'b'})
        tiny = packet(runtime, [('a', 4, 1)])
        tiny.lane = 'decode'
        del runtime.downloads.pending[tiny.download.slot]
        runtime.decode_downloads = NS(buffers=[object()], pending={tiny.download.slot: tiny.download})
        def release(download):
            assert runtime.decode_downloads.pending.pop(download.slot) is download
            runtime.released.append(download)
            return True
        runtime.decode_downloads.release = release
        assert runtime.publication_blocked_requests() == {'b'}
        assert runtime._small_pending_requests() == {'a'}
        # No-forward polling keeps the original foreground path available,
        # even though the sole background worker is scoring a bulk owner.
        runtime.advance_publication({})
        assert runtime.publication_ready({'a'}) and not runtime.publication_ready({'b'})
        assert runtime.accepted['a'] == 5 and not tiny.source_copied
        assert not runtime._cpu_scatter_future.done()
    finally:
        proceed.set()
        close(runtime, (a, b), archive)


def test_default_off_and_invalid_feedback_settings():
    assert not RuntimeOptions().early_capture_slot_release
    with pytest.raises(ValueError, match='requires async CPU scatter'):
        RuntimeOptions(early_capture_slot_release=True)
    with pytest.raises(ValueError, match='bytes must be positive'):
        RuntimeOptions(early_capture_feedback_bytes=0)
    with pytest.raises(ValueError, match='requires split capture DMA'):
        RuntimeOptions(early_capture_slot_release=True, async_cpu_scatter=True, split_capture_dma=False)


def test_cancellation_after_copy_still_waits_for_semantic_owner_before_terminal_replay(tmp_path):
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm_kvmem.connector import ControlMetadata
    runtime, a, b, archive = enabled(tmp_path)
    captured = packet(runtime, [('a', 4, 1)])
    try:
        assert runtime._copy_slot_owner_prefix('bulk') and captured.source_released
        prefix = TerminalPrefix(5, 5, 1, 'FINISHED_ABORTED')
        terminal = replace(SchedulerOutput.make_empty(), finished_req_ids={'a'},
                           kv_connector_metadata=ControlMetadata(terminal_prefixes={'a': prefix}))
        runtime.begin_step({'a'}, {'a': prefix})
        assert not runtime.defer_terminals(terminal).finished_req_ids
        assert not runtime.retirement_done('a') and not runtime.publication_ready({'a'})
        assert runtime.drain_requests({'a'}, wait=True)
        empty = replace(SchedulerOutput.make_empty(), kv_connector_metadata=ControlMetadata())
        assert runtime.defer_terminals(empty).finished_req_ids == {'a'}
        assert not runtime.defer_terminals(empty).finished_req_ids
        assert len(runtime.released) == 1
    finally:
        close(runtime, (a, b), archive)


def test_timestamped_counter_samples_and_stage_threads_are_independent_of_overwritten_progress(tmp_path):
    runtime, a, b, archive = enabled(tmp_path)
    packet(runtime, [('a', 4, 1)])
    try:
        assert runtime._copy_slot_owner_prefix('bulk')
        runtime._write_progress()
        before = json.loads((tmp_path/'progress.json').read_text())
        assert before['early_capture_feedback']['detached_packets'] == 1
        assert runtime.drain_requests({'a'}, wait=True)
        runtime._write_progress()
        samples = [json.loads(l) for l in (tmp_path/'publication_samples.jsonl').open()]
        assert len(samples) >= 2 and samples[-1]['at'] >= samples[0]['at']
        assert samples[-1]['feedback']['detached_packets'] == 0
        stages = [json.loads(l) for l in (tmp_path/'capture_feedback_events.jsonl').open()]
        assert {r['stage'] for r in stages} >= {
            'source_copied', 'source_released', 'semantic_owner_begin',
            'semantic_owner_committed', 'semantic_complete'}
        assert all(r['thread'] == current_thread().name for r in stages)
    finally:
        close(runtime, (a, b), archive)
