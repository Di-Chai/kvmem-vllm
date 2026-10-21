"""Real prepare → frozen-plan native arm → FlashInfer → early Action scoring.

The short fixture supplies an already checked immutable planner result; the
CPU combination suite separately exercises its full multi-owner construction.
"""
import os

import pytest
import torch

import vllm_kvmem.action_attention as module
from test_prefill_action_before_mtp_gpu import fixture, snapshot

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit native prefill CUDA qualification')


class CoordinateFence:
    """Retain the leased event only as long as this wrapper is alive.

    A local class whose method closes over the event retains it in the class
    reference cycle until cyclic GC, unlike ordinary instance ownership.
    """
    def __init__(self, event, phase, waits, plan_arm):
        self.event, self.phase, self.waits, self.plan_arm = event, phase, waits, plan_arm

    def synchronize(self):
        self.waits.append(self.phase[0])
        if self.plan_arm and self.phase[0] == 'prepare':
            pytest.fail('Combined fast prepare waited on the metadata event')
        self.event.synchronize()


def through_prepare(action, batch, plan, monkeypatch, *, plan_arm, phase, waits):
    action.options.prefill_native_plan_arm = plan_arm
    def planned(actual):
        assert actual is batch
        action.prefill_plans = (batch, {'r': plan})
        action.pipeline_requests = {'r'}
    monkeypatch.setattr(action, '_prepare_prefill_pipeline', planned)
    original = module.action_metadata
    def metadata(*args, **kwargs):
        value = original(*args, **kwargs)
        value.event = CoordinateFence(value.event, phase, waits, plan_arm)
        return value
    monkeypatch.setattr(module, 'action_metadata', metadata)
    action.prepare(batch, batch.positions)


@pytest.mark.parametrize('missing', [False, True])
def test_real_prepare_plan_arm_and_early_submission_match_eager_arming(tmp_path, monkeypatch, missing):
    action, captured, batch, _, execute, _, _, plan = fixture(tmp_path)
    results = []
    for plan_arm in (False, True):
        phase, waits = ['prepare'], []
        with monkeypatch.context() as patches:
            through_prepare(action, batch, plan, patches, plan_arm=plan_arm, phase=phase, waits=waits)
            assert ('prepare' in waits) is (not plan_arm)
            phase[0] = 'target'
            if not missing: execute()
            phase[0] = 'early'
            assert action.stage_before_mtp(captured, batch)
            assert 'early' in waits  # Actual GPU coordinates remain checked before score packets escape.
            torch.cuda._sleep(2_000_000)
            phase[0] = 'resolve'
            packets = action.capture(captured, batch)
            results.append(snapshot(packets))
            del packets
            # Exactly two slots cover metadata and scores while the frame is
            # live; both are immediately reusable after consuming its packet.
            assert len(action.download_pool.entries) == 2
            assert all(entry.cpu_done for entry in action.download_pool.entries)
            assert action.download_pool.exhaustions == 0
    assert action.native.counts['prepare:eager'] == action.native.counts['prepare:prefill_plan'] == 1
    assert action.mtp_stage_stats['staged_batches'] == action.mtp_stage_stats['resolved_batches'] == 2
    assert action.mtp_stage_stats.get('fallback_batches', 0) == (2 if missing else 0)
    assert torch.equal(results[0]['r']['masses'], results[1]['r']['masses'])
    for key in results[0]['r']:
        if key != 'masses': assert results[0]['r'][key] == results[1]['r'][key]


def test_fast_prepare_keeps_gpu_coordinate_failure_before_scoring_or_publication(tmp_path, monkeypatch):
    action, captured, batch, _, execute, _, _, plan = fixture(tmp_path)
    batch.positions[-1].add_(7)
    phase, waits = ['prepare'], []
    through_prepare(action, batch, plan, monkeypatch, plan_arm=True, phase=phase, waits=waits)
    assert not waits and action.native.counts['prepare:prefill_plan'] == 1
    execute()
    phase[0] = 'early'
    try:
        with pytest.raises(RuntimeError, match='coordinates differ'):
            action.stage_before_mtp(captured, batch)
        assert waits == ['early']
        assert action.candidate_samples == action.packets == 0
        assert not action._mtp_capture_frame.packets
        assert not action.native.counts['reuse:prefill']
    finally:
        action.cancel_before_mtp()
    assert action._mtp_capture_frame is None
