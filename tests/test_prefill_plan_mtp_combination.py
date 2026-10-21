"""CPU qualification of frozen-plan arming plus same-forward early scoring.

Only device kernels/events are modeled. Production prepare, coordinate checks,
capture/fallback, immutable planning, and delayed accepted-prefix commits run.
This proves ordering/semantics, not a performance gain from overlapping kernels.
"""
from collections import Counter
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

import vllm_kvmem.action_attention as module
import vllm_kvmem.action_scoring as scoring
from vllm_kvmem.memory import RequestMemory
from vllm_kvmem.native_attention import NativeAttentionStats
from test_core import append
from test_prefill_planning import runtime


def test_gpu_coordinate_fence_releases_event_lease_immediately():
    from test_prefill_plan_mtp_combination_gpu import CoordinateFence
    calls, waits, phase = [], [], ['early']
    class Event:
        def synchronize(self): calls.append('synchronized')
    leased = Event()
    reference = weakref.ref(leased)
    fence = CoordinateFence(leased, phase, waits, True)
    del leased
    assert reference() is not None
    fence.synchronize()
    assert waits == ['early'] and calls == ['synchronized']
    del fence
    assert reference() is None  # No gc.collect: no class cycle may retain a lease.


def setup(monkeypatch, tmp_path, *, plan_arm, early, retry=False, owners=('a',)):
    action, _, _, _ = runtime(owners, interval=7)
    action.options.prefill_native_plan_arm = plan_arm
    action.options.prefill_action_before_mtp = early
    action.runtime._raise_publication_error = lambda: None
    action.runtime.output = tmp_path
    action.runner._kvmem_layers = ['L0', 'L1', 'L2']
    action.runner.block_tables = NS(block_tables=[NS(gpu=torch.zeros((len(owners), 6), dtype=torch.int64))])
    memories = {rid: RequestMemory(rid, action.options.memory) for rid in owners}
    for rid, memory in memories.items():
        action.runtime.authorities[rid].memory = memory
    action.layers = [(i, NS(attn=NS(layer_name=f'L{i}', kv_cache=None, _k_scale=None,
        _q_scale=None, impl=NS(scale=.5)), num_kv_heads=1, head_dim=4,
        rotary_emb=NS(cos_sin_cache=None))) for i in range(3)]
    action.groups = {name: 0 for name in action.runner._kvmem_layers}
    action.q_dtypes = action.prefill_q_dtypes = {name: torch.bfloat16 for name in action.groups}
    action.metadata_early_batches = action.candidate_samples = action.packets = 0
    action.capture_seconds = 0.
    action.profile = action.metadata = None
    log, phase = [], ['prepare']
    original_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: original_empty(
        *a, **{k: v for k, v in kw.items() if k != 'pin_memory'}))
    class Fence:
        def record(self): log.append((phase[0], 'event record', id(self)))
        def synchronize(self): log.append((phase[0], 'event wait', id(self)))
    monkeypatch.setattr(torch.cuda, 'Event', Fence)
    # Source is CPU in this test; explicitly model metadata completion placement
    # to observe whether prepare fences it before the next target submission.
    original_metadata = module.action_metadata
    def metadata(*args, **kwargs):
        value = original_metadata(*args, **kwargs)
        value.event = NS(synchronize=lambda: log.append((phase[0], 'metadata wait')))
        log.append((phase[0], 'metadata enqueued'))
        return value
    monkeypatch.setattr(module, 'action_metadata', metadata)
    monkeypatch.setattr(module, 'native_vllm_fp8_views', lambda *_: (None, None))
    def score(q, positions, key, table, trig, ks, starts, counts, length, **kwargs):
        logits = torch.arange(length).float()[None, :] * (q.float().sum((1, 2))[:, None]/1000)
        logits.masked_fill_(torch.arange(length)[None, :] > positions[:, None], -torch.inf)
        probabilities = logits.softmax(-1)
        return torch.stack([probabilities[:, int(s):int(s+n)].sum(-1)
                            for s, n in zip(starts, counts)], 1)
    monkeypatch.setattr(scoring, 'page_attention', score)
    class Flags:
        def copy_(self, source, **kwargs): self.values = list(source)
        def tolist(self): return self.values
    class Native(NativeAttentionStats):
        def __init__(self):
            self.rows, self.disabled, self.mode = None, None, 'reuse'
            self.names, self.counts = action.runner._kvmem_layers, Counter()
            self.max_segments, self.context = 64, 24
            self.host, self.validation_failures = {'valid_flags': Flags()}, 0
            self.score_calls = 0
            self.disarm()
        def arm(self, plans):
            self.disarm()
            self.plans = plans
            self.valid_flags = [int(not retry)]*3
            log.append((phase[0], 'native arm'))
        def score(self, name, rid, prefill, *, defer_validation=False):
            self.score_calls += 1
            if self.frame_invalid: return None
            plan = self.plans[rid]
            rows = torch.tensor([plan['begin']+r for r in plan['rows']], dtype=torch.int64)
            q = self.captured[self.names.index(name)].query.index_select(0, rows)
            result = score(q, torch.tensor([plan['local_begin']+r for r in plan['rows']]),
                None, None, None, None, plan['starts'], plan['counts'], plan['local_begin']+plan['count'])
            self.validation_dirty = True
            self.counts['reuse:prefill'] += 1
            self.frame_success['reuse:prefill'] += 1
            return result
        def report(self): pass
    action.native = Native()
    return action, memories, log, phase


def batch_inputs(owners, begin, count, *, bad_coordinate=False):
    positions = torch.tensor(list(range(begin, begin+count))*len(owners), dtype=torch.int64)
    if bad_coordinate: positions[-1] += 9
    batch = NS(req_ids=list(owners), num_reqs=len(owners), is_prefilling_np=[True]*len(owners),
        query_start_loc_np=[i*count for i in range(len(owners)+1)],
        num_computed_tokens_np=[begin]*len(owners), positions=positions, input_ids=positions+20)
    captured = [NS(name=f'L{i}', query=(torch.arange(len(positions)*4).reshape(-1, 1, 4)+begin+i).bfloat16(),
                   positions=positions, key=torch.zeros((len(positions), 1, 4), dtype=torch.bfloat16))
                for i in range(3)]
    return batch, captured


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('owners', [('a',), ('a', 'b')])
def test_four_combinations_preserve_packets_and_delayed_accepted_scores(tmp_path, monkeypatch, retry, owners):
    outcomes = []
    for plan_arm, early in ((False, False), (False, True), (True, False), (True, True)):
        with monkeypatch.context() as patches:
            action, memories, log, phase = setup(patches, tmp_path, plan_arm=plan_arm,
                                                 early=early, retry=retry, owners=owners)
            pending, saved = [], []
            begin = 0
            def publish(item):
                origin, count, packets = item
                for rid in owners:
                    append(memories[rid], count, local=origin)
                    action.accept(NS(action_inputs=packets), rid, count)
            try:
                for count in (3, 2, 5, 1):
                    phase[0] = 'prepare'
                    log.clear()
                    batch, captured = batch_inputs(owners, begin, count)
                    action.native.captured = captured
                    action.prepare(batch, batch.positions)
                    assert (('prepare', 'metadata wait') not in log) is plan_arm
                    log.append(('target', 'submitted'))
                    phase[0] = 'early'
                    assert action.stage_before_mtp(captured, batch) is early
                    if early:
                        ticket = action._mtp_capture_frame.ticket
                        assert not any(entry[:2] == ('early', 'event wait') for entry in log)
                        assert ('early', 'metadata wait') in log
                        assert ('early', 'event record', id(ticket.event)) in log
                    log.append(('MTP', 'submitted'))
                    phase[0] = 'resolve'
                    packets = action.capture(captured, batch)
                    if early:
                        assert log.index(('MTP', 'submitted')) < log.index(('resolve', 'event wait', id(ticket.event)))
                    saved.append([(rid, tuple(p.rows), tuple(p.history), p.prefill, p.masses.tolist(),
                                   p.source_positions, p.history_begin, p.candidate_count)
                                  for rid, p in packets.items()])
                    assert action.prefill_plans is None and getattr(action, '_mtp_capture_frame', None) is None
                    # Keep one prior accepted chunk unpublished to exercise the
                    # same immutable planner across all four switch settings.
                    pending.append((begin, count, packets))
                    if len(pending) > 1: publish(pending.pop(0))
                    begin += count
                for item in pending: publish(item)
                outcomes.append((saved, [action.runtime.authorities[rid].action_history.prefill.snapshot()
                                         for rid in owners], action.candidate_samples, action.packets))
                assert action.native.counts['prepare:prefill_plan'] == (4 if plan_arm else 0)
                if early:
                    assert action.mtp_stage_stats['staged_batches'] == action.mtp_stage_stats['resolved_batches'] == 4
                    assert action.mtp_stage_stats.get('fallback_batches', 0) == (2 if retry else 0)
            finally:
                for memory in memories.values(): assert memory.close()
    assert all(outcome == outcomes[0] for outcome in outcomes[1:])


@pytest.mark.parametrize('early', [False, True])
def test_combined_plan_arm_still_checks_real_coordinates_before_any_packet(tmp_path, monkeypatch, early):
    action, memories, log, phase = setup(monkeypatch, tmp_path, plan_arm=True, early=early)
    batch, captured = batch_inputs(('a',), 0, 3, bad_coordinate=True)
    action.native.captured = captured
    try:
        action.prepare(batch, batch.positions)
        assert ('prepare', 'metadata wait') not in log
        phase[0] = 'early' if early else 'capture'
        with pytest.raises(RuntimeError, match='coordinates differ'):
            if early: action.stage_before_mtp(captured, batch)
            else: action.capture(captured, batch)
        assert action.native.score_calls == action.candidate_samples == action.packets == 0
        assert action.runtime.authorities['a'].action_history.prefill.end == 0
        if early:
            assert action._mtp_capture_frame is not None
            action.cancel_before_mtp()
            assert action._mtp_capture_frame is None
    finally:
        for memory in memories.values(): assert memory.close()
