"""Real score/KV DMA ownership across delayed background-worker startup.

CPU tests exercise the runner's target boundary and planner admission. Here a
second real FA2 launch is queued before publishing the preceding packet; it is
an ordering/ownership qualification, not a full two-chunk model benchmark.
"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.connector import ControlMetadata
from vllm_kvmem.prefill_planning import PrefillPlanFrame
from test_prefill_early_ahead_combination_gpu import (
    integrated, dispose, durable_snapshot, through_prepare, snapshot,
)

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit native prefill CUDA qualification')


@pytest.mark.parametrize('mode', ['native', 'missing'])
def test_delayed_worker_preserves_real_scores_dma_and_archive(tmp_path, monkeypatch, mode):
    results = []
    for delayed in (False, True):
        args = integrated(tmp_path / str(delayed), ahead=True, fused=True, mode=mode)
        runtime, action, captured, batch, rejected, execute, plan, authority, archive = args
        runtime.options.async_cpu_scatter = True
        runtime.options.cpu_scatter_min_tokens = 1
        runtime.options.prefill_scatter_after_target = delayed
        runtime.runner._kvmem_sessions = NS()
        try:
            with monkeypatch.context() as patches:
                through_prepare(action, batch, plan, patches, plan_arm=True,
                                phase=['target'], waits=[])
                if mode == 'native':
                    execute()
                assert action.stage_before_mtp(captured, batch)
                runtime.submit(captured, batch, rejected)
            packet, = runtime.queue
            scores = snapshot(packet.action_inputs)
            assert not packet.confirmed and not runtime.drain(wait=False)
            runtime.begin_step()
            assert authority.history_end == 254

            # This immutable successor only describes the new public batch;
            # it grants no authority to the old packet beyond begin_step.
            successor = NS(req_ids=['r'], num_reqs=1, is_prefilling_np=np.array([True]),
                query_start_loc_np=np.array([0, 5]), num_computed_tokens_np=np.array([259]))
            successor_plan = dict(history_begin=259, local_begin=259, count=5,
                                  index=0, begin=0, pipeline=True)
            action.prefill_plans = action.native_plans = (successor, {'r': successor_plan})
            action.pipeline_requests = {'r'}
            action.prefill_plan_frames = {'r': PrefillPlanFrame(authority,
                authority.action_history, action._prefill_identity('r', authority.action_history),
                259, 259, 259, ())}
            runtime.runner.req_states = NS(req_id_to_index={'r': 0}, prompt_len=NS(np=np.array([1000])))
            runtime.runner._kvmem_window_adapter = NS(offsets={}, generations={})
            output = NS(num_scheduled_tokens={'r': 5}, total_num_scheduled_tokens=5,
                kv_connector_metadata=ControlMetadata(), finished_req_ids=set(),
                scheduled_new_reqs=[], scheduled_spec_decode_tokens={},
                scheduled_cached_reqs=NS(req_ids=['r'], resumed_req_ids=set(),
                    num_computed_tokens=[259], num_output_tokens=[0]))
            assert runtime.defer_prefill_scatter(output) is delayed
            runtime.advance_publication(output.num_scheduled_tokens, defer_scatter=delayed)
            if delayed:
                assert authority.history_end == 254 and len(runtime.downloads.pending) == 1
                assert packet.download.owners and not packet.published
                runtime.recheck_prefill_scatter(successor)
            # Queued CUDA work may change live native output buffers; preceding
            # CPU masses and packed KV remain owned by the preceding packet.
            execute()
            if delayed:
                assert runtime.resume_prefill_scatter(after_target=True, batch=successor)
            runtime.join_cpu_scatter()
            assert authority.history_end == 259 and not runtime.queue
            assert not runtime.downloads.pending
            authority.publish_sealed(lambda: archive.wait_for_capacity(1))
            results.append((scores, authority.action_history.prefill.snapshot(),
                            durable_snapshot(authority)))
            if delayed:
                assert runtime.prefill_scatter_after_target_stats == {
                    'deferred_batches': 1, 'after_target_started_batches': 1}
        finally:
            executor = getattr(runtime, '_cpu_scatter_executor', None)
            if executor is not None:
                executor.shutdown(wait=True)
            dispose(runtime, authority, archive)
    left, right = results
    assert left[1] == right[1]
    assert left[0]['r'].keys() == right[0]['r'].keys()
    for key, expected in left[0]['r'].items():
        actual = right[0]['r'][key]
        assert torch.equal(actual, expected) if isinstance(expected, torch.Tensor) else actual == expected
    for a, b in zip(left[2], right[2], strict=True):
        assert a[:3] == b[:3]
        assert torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])
