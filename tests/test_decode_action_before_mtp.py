"""KVMEM_DECODE_ACTION_BEFORE_MTP: decode scores are queued ahead of the drafts."""
from collections import Counter
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.serving.action_attention import ActionPacket, ActionRuntime, decode_action_before_mtp
from vllm_kvmem.serving.capture_runtime import CaptureRuntime
from vllm_kvmem.serving.native_attention import NativeAttentionStats
from test_prefill_action_before_mtp import install_wrapper


def fixture(monkeypatch, *, valid=True, owners=('a', 'b'), rows=True, eager=False):
    log = []
    class Fence:
        def record(self): log.append('complete recorded')
        def synchronize(self): log.append('complete waited')
    monkeypatch.setattr(torch.cuda, 'Event', Fence)
    class Flags:
        def copy_(self, *a, **kw): log.append('flags copied')
        def tolist(self): log.append('flags read'); return [int(valid)]
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.disabled, bank.mode, bank.rows = None, 'reuse', None
    bank.names, bank.counts, bank.host, bank.valid_flags = ['target'], Counter(), {'valid_flags': Flags()}, object()
    bank.validation_failures = 0
    bank.disarm()
    disarm = bank.disarm
    def finish(): log.append('disarm'); disarm()
    bank.disarm, bank.report = finish, lambda: log.append('report')
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.queue = []
    runtime.split_capture_dma = True
    def drain(request_ids, wait):
        assert wait
        log.append('drain ' + ','.join(request_ids))
        return True
    runtime.drain_requests = drain
    action = ActionRuntime.__new__(ActionRuntime)
    action.runtime, action.native = runtime, bank
    runtime.action = action
    action.options = NS(prefill_action_before_mtp=True, prefill_publication_pipeline=True)
    action.decode_before_mtp = True
    action.runner = NS(_kvmem_layers=['target'])
    action.candidate_samples = action.packets = 0
    action.capture_seconds = 0.
    batch = NS(num_reqs=len(owners), req_ids=list(owners), is_prefilling_np=[False]*len(owners))
    metadata = NS(batch=batch, indices=tuple(range(len(owners))))
    action.metadata = metadata
    action.prefill_plans = (batch, {})  # what prepare() leaves on a pure decode batch
    action.pipeline_requests = set()
    plans = {rid: dict(index=i) for i, rid in enumerate(owners)}
    action.native_plans = (batch, plans) if eager else None
    action.native_candidates = None if eager else (batch, plans)
    captured = [NS(name='target', key=torch.zeros((3, 1, 2), dtype=torch.bfloat16))]
    calls = []
    def capture(actual, actual_batch, trace):
        assert actual_batch is batch and action.metadata is metadata
        action.metadata = None
        action.native_plans, action.native_candidates = (batch, plans), None
        packets = {}
        for rid in owners:
            if rows and rid == owners[0]:
                log.extend(['score', 'score D2H'])
                bank.validation_dirty = not bank.frame_invalid
                packets[rid] = ActionPacket(NS(), [0], [0, 32], torch.tensor([[.25, .75]]), Fence(),
                    None, (), 4, 1)
                action.candidate_samples += 1
            else:
                packets[rid] = ActionPacket(NS(), [], [], None, None, source_positions=(),
                    history_begin=4, candidate_count=1)
        action.packets += len(packets)
        calls.append(bool(bank.frame_invalid))
        return packets
    action._capture = capture
    return action, captured, batch, log, calls


@pytest.mark.parametrize('eager', [False, True])
def test_decode_scores_and_fence_are_queued_before_the_drafts(monkeypatch, eager):
    action, captured, batch, log, calls = fixture(monkeypatch, eager=eager)
    assert action.stage_before_mtp(captured, batch)
    assert log == ['drain a,b', 'score', 'score D2H', 'flags copied', 'complete recorded']
    with pytest.raises(RuntimeError, match='unresolved'):
        action.prepare(batch, None)
    log.append('MTP queued')
    packets = action.capture(captured, batch)
    assert log[5:] == ['MTP queued', 'complete waited', 'flags read', 'disarm', 'report']
    assert set(packets) == {'a', 'b'} and packets['a'].rows == [0] and packets['b'].rows == []
    assert action.candidate_samples == 1 and action.packets == 2 and calls == [False]
    assert action.mtp_stage_stats == dict(decode_fenced_batches=1, decode_staged_batches=1, resolved_batches=1)
    assert action._mtp_capture_frame is None and action.native._validation_ticket is None
    assert action.native_plans is None and action.native_candidates is None and action.metadata is None


def test_a_step_without_native_rows_adds_no_fence(monkeypatch):
    action, captured, batch, log, calls = fixture(monkeypatch, rows=False)
    assert action.stage_before_mtp(captured, batch)
    assert log == ['drain a,b'] and action._mtp_capture_frame.ticket is None
    packets = action.capture(captured, batch)
    assert log == ['drain a,b', 'disarm', 'report']
    assert set(packets) == {'a', 'b'} and calls == [False]
    assert action.mtp_stage_stats == dict(decode_staged_batches=1, resolved_batches=1)


def test_invalid_native_mass_recomputes_through_the_reference_path(monkeypatch):
    action, captured, batch, log, calls = fixture(monkeypatch, valid=False)
    action.stage_before_mtp(captured, batch)
    log.append('MTP queued')
    packets = action.capture(captured, batch)
    assert calls == [False, True] and set(packets) == {'a', 'b'}
    assert log.count('flags copied') == log.count('flags read') == 1
    assert log.count('complete recorded') == log.count('complete waited') == 2
    assert action.candidate_samples == 1 and action.packets == 2
    assert action.mtp_stage_stats == dict(decode_fenced_batches=1, decode_staged_batches=1,
                                          fallback_batches=1, resolved_batches=1)


@pytest.mark.parametrize('change', ['off', 'verify', 'native_off', 'no_native', 'missing_metadata',
    'different_batch', 'partial_metadata', 'mixed', 'prefill_plans', 'unprepared', 'unsplit',
    'stream', 'original_rope', 'name', 'dtype'])
def test_only_prepared_pure_decode_batches_qualify(monkeypatch, change):
    action, captured, batch, log, _ = fixture(monkeypatch)
    if change == 'off': action.decode_before_mtp = False
    elif change == 'verify': action.native.mode = 'verify'
    elif change == 'native_off': action.native.disabled = 'unsupported'
    elif change == 'no_native': action.native = None
    elif change == 'missing_metadata': action.metadata = None
    elif change == 'different_batch': action.metadata.batch = object()
    elif change == 'partial_metadata': action.metadata.indices = (0,)
    elif change == 'mixed': batch.is_prefilling_np[1] = True
    elif change == 'prefill_plans': action.prefill_plans = (batch, {'a': {}})
    elif change == 'unprepared': action.native_candidates = (object(), {})
    elif change == 'unsplit': action.runtime.split_capture_dma = False
    elif change == 'stream': action.runtime._stream_active = object()
    elif change == 'original_rope': monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE', '1')
    elif change == 'name': captured[0].name = '__mtp__.wrong'
    elif change == 'dtype': captured[0].key = captured[0].key.float()
    assert not action.stage_before_mtp(captured, batch)
    assert not log and getattr(action, '_mtp_capture_frame', None) is None
    skips = [k for k in getattr(action, 'mtp_stage_stats', {}) if k.startswith('decode_skip:')]
    assert len(skips) == (change != 'off')


def test_cancel_fences_and_restores_tentative_counters(monkeypatch):
    for rows in (True, False):
        action, captured, batch, log, _ = fixture(monkeypatch, rows=rows)
        action.stage_before_mtp(captured, batch)
        action.cancel_before_mtp()
        assert log.index('complete waited') < log.index('disarm')
        assert action.candidate_samples == action.packets == 0
        assert action._mtp_capture_frame is None and action.native._validation_ticket is None
        assert action.mtp_stage_stats['cancelled_batches'] == 1


def test_wrapper_orders_decode_scores_before_the_proposal(monkeypatch):
    action, captured, batch, log, _ = fixture(monkeypatch)
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log)
    assert proposer.propose(batch) == 'drafts'
    assert log.index('complete recorded') < log.index('MTP queued') < log.index('complete waited')
    assert log.index('complete waited') < log.index('score consumed')
    assert len(action.runtime.queue) == 1
    assert runner._kvmem_pending_capture is None and runner._kvmem_mtp_rejected is None


@pytest.mark.parametrize('failure', ['mtp', 'before_queue', 'after_queue'])
def test_failures_poison_publication_and_leave_no_frame(monkeypatch, failure):
    action, captured, batch, log, _ = fixture(monkeypatch)
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log, failure=failure)
    with pytest.raises(OSError, match='failure'):
        proposer.propose(batch)
    with pytest.raises(OSError, match='failure'):
        action.runtime._raise_publication_error()
    assert action._mtp_capture_frame is None and runner._kvmem_pending_capture is None
    if failure == 'mtp': assert action.candidate_samples == action.packets == 0


def test_unconfirmed_predecessor_fails_before_any_score(monkeypatch):
    action, captured, batch, log, calls = fixture(monkeypatch)
    action.runtime.drain_requests = lambda request_ids, wait: False
    with pytest.raises(RuntimeError, match='scheduler confirmation'):
        action.stage_before_mtp(captured, batch)
    assert not calls and getattr(action, '_mtp_capture_frame', None) is None


def test_flag_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_DECODE_ACTION_BEFORE_MTP', raising=False)
    assert not decode_action_before_mtp()
    monkeypatch.setenv('KVMEM_DECODE_ACTION_BEFORE_MTP', '1')
    assert decode_action_before_mtp()
    monkeypatch.setenv('KVMEM_DECODE_ACTION_BEFORE_MTP', 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        decode_action_before_mtp()
