"""Actual host packet construction with private pre-validation DMA ownership."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.options import RuntimeOptions
from test_capture_dma_isolation import configure, make_ready
from test_options import config
from test_prefill_pipeline_options import profile
from test_request_publication import fixture, packet, close


def inputs(runtime, specs):
    starts = np.cumsum([0, *[row[2] for row in specs]])
    positions = torch.cat([torch.arange(begin, begin+count) for _, begin, count, _ in specs])
    captures = [NS(name=name, positions=positions,
        key=(torch.arange(positions.numel()*2).reshape(-1, 1, 2)+layer*13).bfloat16(),
        value=torch.ones(positions.numel(), 1, 2, dtype=torch.bfloat16))
        for layer, name in enumerate(runtime.names)]
    batch = NS(req_ids=tuple(row[0] for row in specs), num_reqs=len(specs), num_tokens=positions.numel(),
        query_start_loc=torch.tensor(starts), query_start_loc_np=starts, positions=positions,
        is_prefilling_np=np.array([row[3] for row in specs]), has_prefill=any(row[3] for row in specs),
        num_computed_tokens_np=np.array([row[1] for row in specs]),
        num_scheduled_tokens=np.array([row[2] for row in specs]))
    return captures, batch, torch.zeros(len(specs), dtype=torch.int32)


def prepare(runtime, batch, events, *, enabled=True, qualified=None, capture=None):
    values = dict(vars(runtime.options), prefill_publication_pipeline=True, prefill_capture_ahead=enabled)
    runtime.options = NS(**values)
    qualified = set(batch.req_ids) if qualified is None else set(qualified)
    results = {rid: object() for rid in batch.req_ids}
    original = runtime._capture_payload
    def payload(*args):
        events.append('pack')
        return original(*args)
    runtime._capture_payload = payload
    original_download = runtime.downloads.download
    def download(value):
        result = original_download(value)
        events.append('download')
        return result
    runtime.downloads.download = download
    def action(*_):
        events.append('action')
        if capture:
            capture()
        return results
    runtime.action = NS(capture=action, prefill_plans=(batch, {
        rid: {'pipeline': rid in qualified} for rid in batch.req_ids}), pipeline_requests=qualified,
        accept=lambda packet, rid, count: packet.action_inputs.pop(rid, None))
    return results


def test_capture_ahead_option_is_explicit_and_requires_pipeline_native_scores():
    assert not RuntimeOptions().prefill_capture_ahead
    value = profile()
    value['kvmem'].update(prefill_capture_ahead=True, action_native_scores='reuse')
    saved = deepcopy(value)
    assert RuntimeOptions.from_vllm(config(value)).prefill_capture_ahead
    assert value == saved
    for name, setting in [('prefill_capture_ahead', 1), ('prefill_publication_pipeline', False),
                          ('action_native_scores', 'off')]:
        with pytest.raises(ValueError, match='[Cc]apture.ahead'):
            RuntimeOptions.from_vllm(config({'kvmem': {**saved['kvmem'], name: setting}}))


@pytest.mark.parametrize('enabled', [False, True])
def test_private_payload_precedes_action_only_when_enabled_and_publishes_after_confirmation(tmp_path, enabled):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True), ('b', 4, 4, True)])
    events = []
    def verify_private():
        assert not runtime.queue and (a.history_end, b.history_end) == (4, 4)
        assert len(runtime.downloads.pending) == int(enabled)
    results = prepare(runtime, batch, events, enabled=enabled, capture=verify_private)
    try:
        runtime.submit(captures, batch, rejected)
        assert events == (['pack', 'download', 'action'] if enabled else ['action', 'pack', 'download'])
        current, = runtime.queue
        assert current.action_inputs == results and current.pipeline_requests == frozenset(('a', 'b'))
        assert not current.confirmed and not current.published
        assert not runtime.drain(wait=False) and (a.history_end, b.history_end) == (4, 4)
        make_ready(runtime)
        runtime.begin_step()
        assert runtime.drain(wait=True) and (a.history_end, b.history_end) == (8, 8)
        assert not runtime.downloads.pending
    finally:
        close(runtime, (a, b), archive)


@pytest.mark.parametrize('mixed', [False, True])
def test_replay_terminal_or_mixed_decode_keep_original_capture_order(tmp_path, mixed):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True), ('b', 4, 1, not mixed)])
    events = []
    prepare(runtime, batch, events, qualified={'a'})
    try:
        runtime.submit(captures, batch, rejected)
        assert events[0] == 'action'
        assert runtime.prefill_pipeline_stats.get('capture_ahead_batches', 0) == 0
        make_ready(runtime)
        runtime.begin_step()
        assert runtime.drain(wait=True)
    finally:
        close(runtime, (a, b), archive)


def test_native_invalid_fallback_result_is_the_only_published_action_packet(tmp_path):
    from vllm_kvmem.action_attention import ActionRuntime
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True)])
    events = []
    prepare(runtime, batch, events)
    prepared = runtime.action
    action = ActionRuntime.__new__(ActionRuntime)
    action.prefill_plans, action.pipeline_requests = prepared.prefill_plans, prepared.pipeline_requests
    action.accept = prepared.accept
    metadata = action.metadata = object()
    action.candidate_samples, action.packets, action.capture_seconds = 7, 3, 0.
    tentative, reference = object(), object()
    calls = []
    def score(*_):
        assert not runtime.queue and a.history_end == 4 and len(runtime.downloads.pending) == 1
        # The real capture wrapper restores metadata/counters before rerunning
        # an invalid native frame through its complete reference fallback.
        assert action.metadata is metadata and action.candidate_samples == 7 and action.packets == 3
        action.metadata = None
        action.candidate_samples += 2
        action.packets += 1
        calls.append('reference_complete' if calls else 'native_score')
        events.append(calls[-1])
        return {'a': reference if len(calls) == 2 else tentative}
    def invalid():
        events.append('native_invalid')
        return False
    action._capture = score
    action.native = NS(validate=invalid, disarm=lambda: None, report=lambda: None)
    runtime.action = action
    try:
        runtime.submit(captures, batch, rejected)
        assert events == ['pack', 'download', 'native_score', 'native_invalid', 'reference_complete']
        assert runtime.queue[0].action_inputs == {'a': reference} and not runtime.queue[0].confirmed
        assert action.candidate_samples == 9 and action.packets == 4 and action.prefill_plans is None
        make_ready(runtime)
        runtime.begin_step()
        assert runtime.drain(wait=True) and a.history_end == 8
    finally:
        close(runtime, (a, b), archive)


def test_action_failure_waits_only_own_private_dma_and_keeps_unrelated_owner(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    old = packet(runtime, [('b', 4, 4)], confirmed=False, ready=False)
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True)])
    events = []
    def fail():
        raise ValueError('injected Action validation failure')
    prepare(runtime, batch, events, capture=fail)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            task = executor.submit(runtime.submit, captures, batch, rejected)
            # The private entry is installed before Action is called.
            import time
            deadline = time.monotonic()+5
            while len(runtime.downloads.pending) != 2:
                assert time.monotonic() < deadline
                time.sleep(.001)
            staged = next(value for value in runtime.downloads.pending.values() if value is not old.download)
            assert staged.event.entered.wait(5)
            assert runtime.queue == [old] and (a.history_end, b.history_end) == (4, 4)
            assert staged.owners and old.download.owners and not runtime.downloads.released
            staged.event.ready.set()
            with pytest.raises(ValueError, match='injected Action'):
                task.result(5)
        assert runtime.queue == [old] and list(runtime.downloads.pending.values()) == [old.download]
        assert runtime.downloads.released == [staged]
        assert runtime.prefill_pipeline_stats['capture_ahead_discarded'] == 1
    finally:
        runtime.queue.clear()
        runtime.downloads.pending.clear()
        close(runtime, (a, b), archive)


def test_full_pool_reclaims_owner_prefix_before_packing_and_leaves_unrelated_dma_pending(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    old = packet(runtime, [('a', 4, 4)])
    old.pipeline_requests = frozenset(('a',))
    unrelated = packet(runtime, [('b', 4, 4)], confirmed=False, ready=False)
    captures, batch, rejected = inputs(runtime, [('a', 8, 4, True)])
    events = []
    def check():
        assert a.history_end == 8 and b.history_end == 4
        assert runtime.downloads.released == [old.download]
        assert runtime.queue == [unrelated]
    prepare(runtime, batch, events, capture=check)
    try:
        runtime.submit(captures, batch, rejected)
        assert events == ['pack', 'download', 'action']
        assert len(runtime.downloads.pending) == len(runtime.downloads.buffers) == 2
        current = runtime.queue[-1]
        current.download.event.ready.set()
        current.confirmed = True
        assert runtime.drain_requests({'a'}, wait=True) and a.history_end == 12
        assert runtime.queue == [unrelated] and unrelated.download.owners
    finally:
        runtime.queue.clear()
        runtime.downloads.pending.clear()
        close(runtime, (a, b), archive)


def test_full_unconfirmed_pool_does_not_pack_allocate_or_call_action(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    before = [packet(runtime, [('b', begin, 4)], confirmed=False, ready=False) for begin in (4, 8)]
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True)])
    events = []
    prepare(runtime, batch, events)
    try:
        with pytest.raises(RuntimeError, match='only unconfirmed'):
            runtime.submit(captures, batch, rejected)
        assert not events and runtime.queue == before and len(runtime.downloads.pending) == 2
        assert (a.history_end, b.history_end) == (4, 4)
    finally:
        runtime.queue.clear()
        runtime.downloads.pending.clear()
        close(runtime, (a, b), archive)


def test_failed_cleanup_quarantines_source_and_blocks_later_publication(tmp_path):
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    captures, batch, rejected = inputs(runtime, [('a', 4, 4, True)])
    events = []
    def fail():
        value, = runtime.downloads.pending.values()
        def broken():
            raise OSError('injected CUDA fence failure')
        value.event.synchronize = broken
        raise ValueError('Action failed')
    prepare(runtime, batch, events, capture=fail)
    try:
        with pytest.raises(OSError, match='CUDA fence'):
            runtime.submit(captures, batch, rejected)
        assert not runtime.queue and a.history_end == 4 and len(runtime.downloads.pending) == 1
        assert next(iter(runtime.downloads.pending.values())).owners and not runtime.downloads.released
        with pytest.raises(OSError, match='CUDA fence'):
            runtime.submit(captures, batch, rejected)
    finally:
        runtime.downloads.pending.clear()
        close(runtime, (a, b), archive)
