"""Native FA2 scores detached across bank reuse and real archive publication."""
import gc
import os

import pytest
import torch

from vllm_kvmem.deferred_action_scoring import FrozenScoreLimits, FrozenScorePool
from test_prefill_action_before_mtp_gpu import snapshot
from test_prefill_early_ahead_combination_gpu import integrated, dispose, durable_snapshot
from test_prefill_plan_mtp_combination_gpu import through_prepare

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit detached prefill CUDA qualification')


def enable(action, captured):
    action.options.prefill_deferred_validation = True
    action._deferred_frames = []
    action._deferred_groups = (0,)
    bank = action.native
    action.deferred_score_pool = FrozenScorePool(FrozenScoreLimits(
        layers=1, groups=1, heads=24, dim=256, page_tokens=16, block_tokens=32,
        max_rows=bank.max_rows, max_owners=1, max_segments=bank.max_segments,
        max_length=bank.context), captured[0].query.device,
        fused=action.options.action_fused_scoring, max_bytes=64*1024**2,
        max_scratch_bytes=2*1024**3)


@pytest.mark.parametrize('mode', ['native', 'missing'])
@pytest.mark.parametrize('fused', [False, True])
def test_old_bank_can_be_rearmed_without_publishing_unvalidated_scores(tmp_path, monkeypatch, mode, fused):
    outcomes = []
    for detached in (False, True):
        runtime, action, captured, batch, rejected, execute, plan, authority, archive = integrated(
            tmp_path/str(detached), ahead=True, fused=True, mode=mode)
        action.options.action_fused_scoring = fused
        try:
            if detached:
                enable(action, captured)
            phase, waits = ['prepare'], []
            with monkeypatch.context() as patches:
                through_prepare(action, batch, plan, patches, plan_arm=True, phase=phase, waits=waits)
                phase[0] = 'target'
                if mode == 'native':
                    execute()
                phase[0] = 'stage'
                assert action.stage_before_mtp(captured, batch)
                private = action._mtp_capture_frame
                if detached:
                    assert private.deferred is not None and private.ticket is None
                    assert not private.deferred.ticket.resolved
                runtime.submit(captured, batch, rejected)
            packet, = runtime.queue
            if detached:
                frame = packet.action_validation
                assert frame is private.deferred and not frame.ticket.resolved
                # The same scratch bank and graph-stable public inputs are
                # allowed to change after snapshot enqueue on this stream.
                action.native.arm({'r': plan})
                captured[0].query.fill_(31)
                action.runner.block_tables.block_tables[0].gpu.fill_(0)
                module = action.layers[0][1]
                module.attn._k_scale.fill_(.125)
                module.attn._q_scale.fill_(2)
                action.native.valid_flags.fill_(1)
                assert not runtime.drain(wait=False)
                assert not frame.ticket.resolved and authority.history_end == 254
            assert not packet.confirmed and not runtime.drain(wait=True)
            runtime.begin_step()
            assert runtime.drain(wait=True)
            assert authority.history_end == 259
            action._collect_deferred_scores()
            authority.publish_sealed(lambda: archive.wait_for_capacity(1))
            outcomes.append((authority.action_history.prefill.snapshot(), durable_snapshot(authority)))
            if detached:
                assert frame.ready and frame.applied and frame.slot is None
                assert frame.outcome.valid is (mode == 'native')
                assert not action._deferred_frames
                assert action.mtp_stage_stats['deferred_resolved_batches'] == 1
                assert action.mtp_stage_stats.get('deferred_fallback_batches', 0) == int(mode == 'missing')
        finally:
            dispose(runtime, authority, archive)
    assert outcomes[0][0] == outcomes[1][0]
    for left, right in zip(outcomes[0][1], outcomes[1][1], strict=True):
        assert left[:3] == right[:3]
        assert torch.equal(left[3], right[3]) and torch.equal(left[4], right[4])


def test_two_detached_snapshots_can_resolve_after_new_graph_replay(tmp_path):
    from test_prefill_action_before_mtp_gpu import fixture
    action, captured, batch, prepare, execute, _, q, _ = fixture(tmp_path)
    enable(action, captured)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        execute()
    results, expected = [], []
    # A larger existing score download budget supports two pending frames;
    # the new frozen input pool itself remains exactly two slots.
    from vllm_kvmem.action_download_pool import ActionDownloadPool
    action.download_pool = ActionDownloadPool(4096, 4)
    for marker in (.5, 1.75):
        q.mul_(marker)
        prepare(graph=graph)
        # Capture the baseline from the same original Q/flags before changing
        # the live bank; independently freeze its resulting host scores.
        saved_pool = action.deferred_score_pool
        action.deferred_score_pool = None
        baseline = action.capture(captured, batch)
        expected.append(snapshot(baseline))
        del baseline
        gc.collect()
        action.deferred_score_pool = saved_pool
        prepare(graph=graph)
        assert action.stage_before_mtp(captured, batch)
        results.append(action.capture(captured, batch))
        assert not results[-1].validation.ticket.resolved
    q.fill_(0)
    for result, reference in zip(results, expected, strict=True):
        assert result.validation.resolve(wait=True)
        actual = snapshot(result)
        assert torch.equal(actual['r']['masses'], reference['r']['masses'])
    action._collect_deferred_scores()
    assert not action._deferred_frames
    assert not torch.equal(expected[0]['r']['masses'], expected[1]['r']['masses'])
