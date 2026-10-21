"""Same-forward Action staging preserves proposal/publication transactions."""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from vllm_kvmem.action_attention import ActionPacket, ActionRuntime
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.native_attention import NativeAttentionStats
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem import mtp_runtime
from test_options import config
from test_prefill_pipeline_options import profile


def fixture(monkeypatch, *, valid=True, owners=('a',)):
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
    action = ActionRuntime.__new__(ActionRuntime)
    action.runtime, action.native = runtime, bank
    runtime.action = action
    action.options = NS(prefill_action_before_mtp=True, prefill_publication_pipeline=True)
    action.runner = NS(_kvmem_layers=['target'], req_states=NS(
        req_id_to_index={rid: i for i, rid in enumerate(owners)}, prompt_len=NS(np=[100]*len(owners))))
    action.candidate_samples = action.packets = 0
    action.capture_seconds = 0.
    batch = NS(num_reqs=len(owners), req_ids=list(owners), is_prefilling_np=[True]*len(owners))
    metadata = NS(batch=batch, indices=tuple(range(len(owners))))
    action.metadata = metadata
    plans = {rid: dict(index=i, pipeline=True, count=3, history_begin=4,
                      state=NS(prefill_replay=None)) for i, rid in enumerate(owners)}
    action.prefill_plans = action.native_plans = (batch, plans)
    action.pipeline_requests = set(owners)
    captured = [NS(name='target', key=torch.zeros((3, 1, 2), dtype=torch.bfloat16))]
    weak_packets = []
    calls = []
    def capture(actual, actual_batch, trace):
        assert actual_batch is batch and action.metadata is metadata
        assert action.native_plans[0] is batch and action.prefill_plans[0] is batch
        if calls and bank.frame_invalid:
            assert all(ref() is None for ref in weak_packets), 'Tentative aliases retained into fallback'
        action.metadata = None
        log.extend(['score', 'mean', 'score D2H'])
        bank.validation_dirty = not bank.frame_invalid
        packet = ActionPacket(plans[owners[0]]['state'], [0, 2], [0, 32],
            torch.tensor([[.25, .75], [.5, .5]]), Fence(), (7, 3, [0, 32], 4), (4, 6), 4, 3)
        weak_packets.append(weakref.ref(packet))
        action.candidate_samples += 2
        action.packets += 1
        calls.append(bool(bank.frame_invalid))
        return {owners[0]: packet}
    action._capture = capture
    return action, captured, batch, log, calls


def test_stage_is_private_until_proposal_then_resolves_original_early_event_once(monkeypatch):
    action, captured, batch, log, calls = fixture(monkeypatch)
    assert action.stage_before_mtp(captured, batch)
    assert log == ['score', 'mean', 'score D2H', 'flags copied', 'complete recorded']
    assert not action.runtime.queue and action._mtp_capture_frame is not None
    with pytest.raises(RuntimeError, match='Only one'):
        action.stage_before_mtp(captured, batch)
    with pytest.raises(RuntimeError, match='unresolved'):
        action.prepare(batch, None)
    log.append('MTP queued')
    packets = action.capture(captured, batch)
    assert log == ['score', 'mean', 'score D2H', 'flags copied', 'complete recorded',
                   'MTP queued', 'complete waited', 'flags read', 'disarm', 'report']
    assert packets['a'].candidate_count == 3 and packets['a'].source_positions == (4, 6)
    assert action.candidate_samples == 2 and action.packets == 1 and calls == [False]
    assert action.mtp_stage_stats == {'staged_batches': 1, 'resolved_batches': 1}
    assert action._mtp_capture_frame is None and not action.runtime.queue


def test_invalid_native_releases_tentative_aliases_then_recomputes_and_fences_reference(monkeypatch):
    action, captured, batch, log, calls = fixture(monkeypatch, valid=False)
    action.stage_before_mtp(captured, batch)
    log.append('MTP queued')
    packets = action.capture(captured, batch)
    assert calls == [False, True]
    assert log.count('complete recorded') == log.count('complete waited') == 2
    assert log.count('flags copied') == log.count('flags read') == 1
    assert action.candidate_samples == 2 and action.packets == 1 and set(packets) == {'a'}
    assert action.mtp_stage_stats == dict(staged_batches=1, fallback_batches=1, resolved_batches=1)


@pytest.mark.parametrize('change', ['disabled', 'verify', 'native_off', 'missing_metadata',
    'different_batch', 'partial_metadata', 'mixed', 'terminal', 'replay', 'partial_plans', 'no_pipeline'])
def test_only_complete_ordinary_nonterminal_public_prefill_qualifies(monkeypatch, change):
    action, captured, batch, log, _ = fixture(monkeypatch, owners=('a', 'b'))
    if change == 'disabled': action.options.prefill_action_before_mtp = False
    elif change == 'verify': action.native.mode = 'verify'
    elif change == 'native_off': action.native.disabled = 'unsupported'
    elif change == 'missing_metadata': action.metadata = None
    elif change == 'different_batch': action.metadata.batch = object()
    elif change == 'partial_metadata': action.metadata.indices = (0,)
    elif change == 'mixed': batch.is_prefilling_np[1] = False
    elif change == 'terminal': action.runner.req_states.prompt_len.np[1] = 7
    elif change == 'replay': action.prefill_plans[1]['b']['state'].prefill_replay = (5, 9)
    elif change == 'partial_plans': action.prefill_plans[1].pop('b')
    elif change == 'no_pipeline': action.pipeline_requests.remove('b')
    assert not action.stage_before_mtp(captured, batch)
    assert not log and getattr(action, '_mtp_capture_frame', None) is None


@pytest.mark.parametrize('change', ['name', 'dtype'])
def test_target_capture_contract_is_checked_before_any_early_gpu_work(monkeypatch, change):
    action, captured, batch, log, _ = fixture(monkeypatch)
    if change == 'name': captured[0].name = '__mtp__.wrong'
    else: captured[0].key = captured[0].key.float()
    with pytest.raises(ValueError, match='ordered BF16'):
        action.stage_before_mtp(captured, batch)
    assert not log


def test_cancel_waits_only_staged_completion_and_restores_tentative_counters(monkeypatch):
    action, captured, batch, log, _ = fixture(monkeypatch)
    action.stage_before_mtp(captured, batch)
    action.cancel_before_mtp()
    assert log.index('complete waited') < log.index('disarm')
    assert action.candidate_samples == action.packets == 0
    assert action._mtp_capture_frame is None
    assert action.mtp_stage_stats == {'staged_batches': 1, 'cancelled_batches': 1}


@dataclass
class Batch:
    positions: object
    seq_lens: object
    seq_lens_cpu_upper_bound: object


def install_wrapper(monkeypatch, action, captured, batch, log, *, failure=None):
    monkeypatch.setattr('vllm_kvmem.draft_contract.verify_mtp_sources', lambda: None)
    module = NS()
    model = NS(named_modules=lambda: [('layer', module)])
    monkeypatch.setattr(mtp_runtime, 'install_qwen_capture', lambda _: ['layer'])
    monkeypatch.setattr(mtp_runtime, 'install_persistent_capture', lambda *_: [NS(name='layer')])
    class Proposer:
        input_buffers = NS(positions=None)
        def _prefill(self): pass
        def propose(self, actual):
            assert actual is batch
            log.append('MTP queued')
            if failure == 'mtp': raise OSError('MTP failure')
            return 'drafts'
    proposer = Proposer(); proposer.model = model
    runner = NS(speculative_config=NS(method='mtp'), speculator=proposer,
        _kvmem_layers=['target'], max_num_tokens=3, _kvmem_pending_capture=(captured, batch),
        _kvmem_mtp_rejected=2, _kvmem_capture_runtime=action.runtime)
    def submit(all_captured, actual, rejected):
        assert actual is batch and rejected == 2 and len(all_captured) == 2
        packets = action.capture(all_captured, batch)
        log.append('score consumed')
        if failure == 'before_queue': raise OSError('before queue failure')
        action.runtime.queue.append(NS(confirmed=False, action_inputs=packets))
        if failure == 'after_queue': raise OSError('after queue failure')
    action.runtime.submit = submit
    mtp_runtime.install_mtp_runtime(runner)
    return runner, proposer


def test_mtp_wrapper_submits_only_after_early_completion_and_retains_acceptance_filter(monkeypatch):
    action, captured, batch, log, _ = fixture(monkeypatch)
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log)
    assert proposer.propose(batch) == 'drafts'
    assert log.index('complete recorded') < log.index('MTP queued') < log.index('complete waited')
    assert log.index('complete waited') < log.index('score consumed')
    assert len(action.runtime.queue) == 1 and not action.runtime.queue[0].confirmed
    assert runner._kvmem_pending_capture is None and runner._kvmem_mtp_rejected is None


@pytest.mark.parametrize('failure', ['mtp', 'before_queue', 'after_queue'])
def test_proposal_and_post_consume_failures_poison_publication_without_committing(monkeypatch, failure):
    action, captured, batch, log, _ = fixture(monkeypatch)
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log, failure=failure)
    with pytest.raises(OSError, match='failure'):
        proposer.propose(batch)
    with pytest.raises(OSError, match='failure'):
        action.runtime._raise_publication_error()
    assert all(not packet.confirmed for packet in action.runtime.queue)
    assert len(action.runtime.queue) == int(failure == 'after_queue')
    assert action._mtp_capture_frame is None and runner._kvmem_pending_capture is None
    if failure == 'mtp': assert action.candidate_samples == action.packets == 0


def test_fence_failure_keeps_frame_ticket_and_packet_aliases_quarantined(monkeypatch):
    action, captured, batch, log, _ = fixture(monkeypatch)
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log)
    original_capture = action._capture
    class BrokenFence:
        attempts = 0
        def record(self): pass
        def synchronize(self):
            self.attempts += 1
            if self.attempts == 1: raise RuntimeError('injected CUDA completion failure')
    monkeypatch.setattr(torch.cuda, 'Event', BrokenFence)
    with pytest.raises(RuntimeError, match='quarantined'):
        proposer.propose(batch)
    frame = action._mtp_capture_frame
    assert frame is not None and frame.packets and frame.ticket.error
    assert frame.ticket.event.attempts == 1 and runner._kvmem_pending_capture is not None
    assert not action.runtime.queue and 'disarm' not in log
    assert action.mtp_stage_stats['quarantined_frames'] == 1


@pytest.mark.parametrize('failure', ['partial_stage', 'fallback'])
def test_partial_score_failure_completes_private_dma_before_cancel(monkeypatch, failure):
    action, captured, batch, log, _ = fixture(monkeypatch, valid=failure != 'fallback')
    original = action._capture
    def failed(*args):
        if failure == 'partial_stage' or action.native.frame_invalid:
            action.candidate_samples += 2
            log.append('partial score D2H')
            raise OSError('partial score failure')
        return original(*args)
    action._capture = failed
    runner, proposer = install_wrapper(monkeypatch, action, captured, batch, log)
    with pytest.raises(OSError, match='partial score'):
        proposer.propose(batch)
    assert log.index('partial score D2H') < len(log)-1
    assert log[-3:] == ['complete waited', 'disarm', 'report']
    assert action._mtp_capture_frame is None and not action.runtime.queue
    assert action.candidate_samples == action.packets == 0


def test_option_is_explicit_and_requires_same_forward_native_pipeline():
    assert not RuntimeOptions().prefill_action_before_mtp
    value = profile()
    value['kvmem'].update(prefill_action_before_mtp=True, action_native_scores='reuse')
    original = deepcopy(value)
    assert RuntimeOptions.from_vllm(config(value)).prefill_action_before_mtp
    assert value == original
    for key, bad in [('prefill_action_before_mtp', 1), ('action_native_scores', 'verify'),
                     ('action_native_scores', 'off'), ('prefill_publication_pipeline', False)]:
        altered = deepcopy(value)
        altered['kvmem'][key] = bad
        with pytest.raises(ValueError, match='boolean|requires the publication'):
            RuntimeOptions.from_vllm(config(altered))
