"""Host handoff ordering for early scores and ahead capture publication.

Scores/events are host oracles here; real native scores, CUDA DMA and final
archive bytes are exercised in the accompanying opt-in GPU qualification.
"""
from dataclasses import replace

import pytest

from test_capture_dma_isolation import configure, make_ready
from test_prefill_action_before_mtp import fixture as staged_action
from test_prefill_capture_ahead import inputs
from test_request_publication import fixture as publication, close


@pytest.mark.parametrize('valid', [False, True])
def test_same_frame_early_score_then_ahead_dma_still_requires_scheduler_confirmation(tmp_path, monkeypatch, valid):
    action, captured, batch, log, calls = staged_action(monkeypatch, valid=valid)
    runtime, a, b, archive = publication(tmp_path)
    configure(runtime, split=True)
    runtime.options = replace(runtime.options, prefill_publication_pipeline=True, prefill_capture_ahead=True,
        async_cpu_scatter=True, split_capture_dma=True, prefill_attention=True,
        memory=replace(runtime.options.memory, action_attention=True, repository_blocks=16), action_native_scores='reuse',
        session_capacity=1, session_checkpoint_bytes=4096, async_archive_retirement=True)
    actual, actual_batch, rejected = inputs(runtime, [('a', 4, 4, True)])
    # Preserve the identities captured by the existing score oracle.
    vars(batch).update(vars(actual_batch))
    captured[:] = actual
    action.runner._kvmem_layers = list(runtime.names)
    action.runtime, runtime.action = runtime, action
    action.accept = lambda packet, rid, count: (
        packet.action_inputs.pop(rid), log.append(('accepted', rid, count)))
    original_pack, original_download = runtime._capture_payload, runtime.downloads.download
    def pack(*args):
        log.append('pack')
        return original_pack(*args)
    def download(payload):
        log.append('KV D2H')
        return original_download(payload)
    runtime._capture_payload, runtime.downloads.download = pack, download
    try:
        assert action.stage_before_mtp(captured, batch)
        log.append('MTP queued')
        runtime.submit(captured, batch, rejected)
        assert log.index('complete recorded') < log.index('MTP queued')
        assert log.index('MTP queued') < log.index('pack') < log.index('KV D2H') < log.index('complete waited')
        assert calls == ([False] if valid else [False, True])
        assert action.mtp_stage_stats.get('fallback_batches', 0) == int(not valid)
        packet, = runtime.queue
        assert not packet.confirmed and not packet.download.query()
        assert not runtime.drain(wait=False) and (a.history_end, b.history_end) == (4, 4)
        make_ready(runtime)
        assert not runtime.drain(wait=False) and a.history_end == 4
        runtime.begin_step()
        assert runtime.drain(wait=True) and (a.history_end, b.history_end) == (8, 4)
        assert log[-1] == ('accepted', 'a', 4)
        assert not runtime.downloads.pending and runtime.downloads.released == [packet.download]
        assert runtime.prefill_pipeline_stats['capture_ahead_batches'] == 1
    finally:
        close(runtime, (a, b), archive)
