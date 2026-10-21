"""Early-score completion composes with real ahead pack/DMA and CPU sealing.

One actual FlashInfer sampled layer uses the deployed 24 Q / 4 KV heads and
256 head dimension. This qualifies the handoff, not full-model performance.
The immutable-plan construction and 17-layer pack geometry have separate tests.
"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.capture_pack import PackLimits
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.transfer import PackedDownloads
from test_prefill_action_before_mtp_gpu import fixture, snapshot
from test_prefill_plan_mtp_combination_gpu import through_prepare

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit native prefill CUDA qualification')


def integrated(tmp_path, *, ahead=True, fused=False, mode='native'):
    tmp_path.mkdir(parents=True, exist_ok=True)
    action, captured, batch, _, execute, _, _, plan = fixture(tmp_path, mode=mode)
    options = action.options
    options.memory = KVMemConfig(block_tokens=32, select_budget=512, gen_budget=128,
                                repository_blocks=32, sink_blocks=1, recent_blocks=0)
    for name, value in dict(prefill_capture_ahead=ahead, capture_fused_pack=fused,
            query_history=False, guided_query_tokens=0, batched_cpu_append=True,
            async_cpu_scatter=False, early_capture_slot_release=False).items():
        setattr(options, name, value)
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options, runtime.config = options, options.memory
    runtime.action, runtime.runner, runtime.output = action, action.runner, tmp_path
    action.runtime = runtime
    runtime.names = tuple(item.name for item in captured)
    runtime.layers, runtime.heads, runtime.dim = captured, 4, 256
    runtime.row_elements, runtime.resident_records = 1024, 16
    runtime.raw_dtype = torch.float8_e4m3fn
    device = captured[0].key.device
    runtime.runner.device = device
    runtime.runner.vllm_config = NS(max_concurrent_batches=1)
    runtime.scales = torch.tensor([.3], device=device)
    runtime.split_capture_dma, runtime.decode_downloads = False, None
    runtime._capture_pack_limits = PackLimits(1, 4, 256, 5, 1)
    runtime._capture_pack_cache = None
    runtime.downloads = PackedDownloads(runtime._capture_pack_limits.payload_bytes, 2, device)
    runtime.queue, runtime.terminal_prefixes, runtime.terminal_discarded = [], {}, {}
    runtime.accepted, runtime.rejected, runtime.last_steps = {'r': 254}, {'r': 0}, {'r': []}
    runtime.batches, runtime._progress_batch = 0, 0
    runtime._write_progress = lambda: None
    record_bytes = 32*4*256*2
    archive = runtime.archive = MemoryArchive(str(tmp_path/'archive'), record_bytes, 32,
        cpu_direct_record_seal=ahead, cpu_seal_hash_workers=8 if ahead else 1)
    archive.async_seal = True
    authority = RequestAuthority('r', options.memory, archive, 1, 4, 256,
                                 raw_dtype=runtime.raw_dtype, resident_records=16)
    prefix = torch.zeros((1, 254, 4, 256), dtype=torch.float32).to(runtime.raw_dtype)
    assert authority.append(prefix, prefix.view(torch.uint8), history_begin=0, local_rope_begin=0)
    authority.publish_sealed(lambda: archive.wait_for_capacity(1))
    authority.action_history = plan['state']
    runtime.authorities = {'r': authority}
    plan['authority'] = authority
    batch.query_start_loc = torch.tensor(batch.query_start_loc_np, device=device)
    batch.has_prefill, batch.num_scheduled_tokens = True, np.array([5])
    captured[0].key.copy_((torch.arange(5*4*256, device=device).reshape(5, 4, 256)%67-33).bfloat16())
    captured[0].value = (captured[0].key.float()/7).bfloat16()
    rejected = torch.zeros(1, device=device, dtype=torch.int32)
    return runtime, action, captured, batch, rejected, execute, plan, authority, archive


def dispose(runtime, authority, archive):
    # This is explicit fixture teardown after all assertions; normal execution
    # never releases an unconfirmed publication packet.
    for download in tuple(runtime.downloads.pending.values()):
        download.event.synchronize()
        assert runtime.downloads.release(download)
    runtime.queue.clear()
    assert runtime.downloads.close()
    authority.publish_sealed(lambda: archive.wait_for_capacity(1))
    assert authority.close() and archive.close()


def durable_snapshot(authority):
    result = []
    for record in sorted(authority.records.values(), key=lambda item: item.history_begin):
        raw, values = authority.read_record(record.handle)
        result.append((record.history_begin, record.count, record.digest,
                       raw.view(torch.uint8).clone(), values.clone()))
    return result


@pytest.mark.parametrize('mode', ['native', 'missing'])
def test_early_ticket_ahead_and_fused_pack_keep_scores_archive_bytes_and_confirmation(tmp_path, monkeypatch, mode):
    outcomes = []
    for ahead, fused in ((False, False), (True, False), (True, True)):
        args = integrated(tmp_path/f'{ahead}-{fused}', ahead=ahead, fused=fused, mode=mode)
        runtime, action, captured, batch, rejected, execute, plan, authority, archive = args
        try:
            seed_usage = archive.io_usage()
            # Compile the optional pack before the explicit event-order probe;
            # compilation time is unrelated to whether a recorded event moves.
            if fused:
                runtime._capture_payload(captured, batch, rejected, (0,))
                runtime._capture_pack_cache.event.synchronize()
                runtime.capture_pack_stats.clear()
            phase, waits = ['prepare'], []
            with monkeypatch.context() as patches:
                through_prepare(action, batch, plan, patches, plan_arm=True, phase=phase, waits=waits)
                assert not waits
                phase[0] = 'target'
                if mode == 'native': execute()
                phase[0] = 'early'
                assert action.stage_before_mtp(captured, batch)
                early = action._mtp_capture_frame.ticket
                # Only the native-success, ahead, non-JIT arm needs a delayed
                # tail. Invalid fallback correctly waits for its later work.
                delayed = mode == 'native' and ahead and not fused
                torch.cuda._sleep(500_000_000 if delayed else 2_000_000)
                tail = torch.cuda.Event(); tail.record()
                phase[0] = 'submit'
                runtime.submit(captured, batch, rejected)
            packet, = runtime.queue
            assert early.resolved and early.event.query()
            if delayed:
                assert not tail.query(), 'Ahead submission moved the early-score fence behind MTP'
                assert not packet.download.query()
            assert not packet.confirmed and not runtime.drain(wait=False)
            assert authority.history_end == 254 and not authority.action_history.prefill.chunks
            scores = snapshot(packet.action_inputs)
            packet.download.event.synchronize()
            payload = packet.download.payload.clone()
            assert not runtime.drain(wait=False) and authority.history_end == 254
            runtime.begin_step()
            assert runtime.drain(wait=True) and authority.history_end == 259
            assert not runtime.queue and not runtime.downloads.pending
            authority.publish_sealed(lambda: archive.wait_for_capacity(1))
            usage = archive.io_usage()
            # This five-row handoff completes exactly one 32-row record. It
            # exercises direct sealing, while the seven-record seed above
            # qualifies the configured parallel-hash path independently.
            assert usage['seal_direct_bytes']-seed_usage['seal_direct_bytes'] == (archive.record_bytes if ahead else 0)
            assert usage['cpu_seal_hash_workers'] == (8 if ahead else 1)
            assert (seed_usage['seal_hash_parallel_batches_completed'] > 0) is ahead
            assert usage['seal_hash_parallel_batches_completed'] == seed_usage['seal_hash_parallel_batches_completed']
            assert runtime.prefill_pipeline_stats.get('capture_ahead_batches', 0) == int(ahead)
            assert runtime.capture_pack_stats.get('fused_batches', 0) == int(fused)
            assert action.native.counts['prepare:prefill_plan'] == 1
            assert action.mtp_stage_stats == dict(staged_batches=1, resolved_batches=1,
                **({'fallback_batches': 1} if mode == 'missing' else {}))
            assert action.download_pool.exhaustions == 0
            assert all(entry.cpu_done for entry in action.download_pool.entries)
            outcomes.append((scores, payload, authority.action_history.prefill.snapshot(),
                             durable_snapshot(authority)))
        finally:
            dispose(runtime, authority, archive)
    expected = outcomes[0]
    for actual in outcomes[1:]:
        for key, value in actual[0]['r'].items():
            if key == 'masses': assert torch.equal(value, expected[0]['r'][key])
            else: assert value == expected[0]['r'][key]
        assert torch.equal(actual[1], expected[1]) and actual[2] == expected[2]
        assert len(actual[3]) == len(expected[3])
        for left, right in zip(actual[3], expected[3]):
            assert left[:3] == right[:3]
            assert torch.equal(left[3], right[3]) and torch.equal(left[4], right[4])


@pytest.mark.parametrize('dma_fails', [False, True])
def test_ahead_resolve_failure_retains_score_ticket_and_only_reclaims_completed_private_dma(tmp_path, monkeypatch, dma_fails):
    args = integrated(tmp_path)
    runtime, action, captured, batch, rejected, execute, plan, authority, archive = args
    score_event = None
    original_dma = []
    try:
        through_prepare(action, batch, plan, monkeypatch, plan_arm=True, phase=['early'], waits=[])
        execute()
        assert action.stage_before_mtp(captured, batch)
        frame = action._mtp_capture_frame
        score_event = frame.ticket.event
        waits = []
        def failed_score():
            waits.append('score')
            raise RuntimeError('injected early-score wait failure')
        frame.ticket.event = NS(synchronize=failed_score)
        if dma_fails:
            download = runtime.downloads.download
            def failed_dma(payload):
                result = download(payload)
                original_dma.append((result, result.event))
                def fail(): raise OSError('injected private-DMA fence failure')
                result.event = NS(query=result.event.query, synchronize=fail)
                return result
            monkeypatch.setattr(runtime.downloads, 'download', failed_dma)
        with pytest.raises((RuntimeError, OSError), match='failure') as error:
            runtime.submit(captured, batch, rejected)
        # Match the existing MTP wrapper's fatal-runtime treatment. Its actual
        # invocation and post-consume exceptions have separate host tests.
        runtime.quarantine_publication(error.value)
        with pytest.raises(RuntimeError, match='quarantined'):
            action.cancel_before_mtp()
        assert waits == ['score'] and frame.ticket.error is not None
        assert action._mtp_capture_frame is frame and frame.packets
        assert not runtime.queue and authority.history_end == 254
        assert len(runtime.downloads.pending) == int(dma_fails)
        assert all(item.owners for item in runtime.downloads.pending.values())
        assert any(not entry.cpu_done for entry in action.download_pool.entries)
        with pytest.raises((RuntimeError, OSError), match='failure'):
            runtime.begin_step()
    finally:
        # Fixture teardown waits the real events; it does not make a poisoned
        # production frame reusable or replace any frozen runtime state.
        if score_event is not None: score_event.synchronize()
        for item, event in original_dma: item.event = event
        dispose(runtime, authority, archive)
