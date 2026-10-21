"""Real CUDA pack/DMA lifecycle qualification; Action logic is a host oracle.

This does not qualify native attention numerics or model throughput. It uses
real captured CUDA tensors, producer ordering, and the fixed PackedDownloads
pool while retaining the scheduler-confirmation/CPU-authority fixture.
"""
import os
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                                reason='Explicit GPU qualification only')

from vllm_kvmem.terminal import TerminalPrefix
from vllm_kvmem.transfer import PackedDownloads
from test_capture_dma_isolation import configure
from test_prefill_capture_ahead import inputs, prepare as prepare_host
from test_request_publication import fixture, close


def gpu_runtime(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    runtime, a, b, archive = fixture(tmp_path)
    configure(runtime)
    device = torch.device('cuda', torch.cuda.current_device())
    runtime.runner.device = device
    runtime.scales = runtime.scales.to(device)
    runtime.downloads = PackedDownloads(4096, 2, device)
    runtime.decode_downloads = None  # Every qualified batch below is prefill.
    return runtime, a, b, archive


def prepare(runtime, *args, **kwargs):
    result = prepare_host(runtime, *args, **kwargs)
    runtime.action.invalidate_prefill_plans = lambda *_: None
    return result


def delayed_inputs(runtime, specs, marker):
    captures, batch, rejected = inputs(runtime, specs)
    device = runtime.runner.device
    batch.query_start_loc = batch.query_start_loc.to(device)
    batch.positions = batch.positions.to(device)
    for capture in captures:
        capture.positions = batch.positions
        capture.key = capture.key.to(device)
        capture.value = capture.value.to(device)
    rejected = rejected.to(device)
    # The pack must read these delayed producer writes, never old buffer bytes.
    torch.cuda._sleep(20_000_000)
    for capture in captures:
        capture.key.add_(marker)
        capture.value.add_(marker / 4)
    return captures, batch, rejected


def release_private(runtime):
    for item in tuple(runtime.queue):
        item.download.event.synchronize()
        assert runtime.downloads.release(item.download)
    runtime.queue.clear()


def shutdown(runtime, owners, archive):
    release_private(runtime)
    assert runtime.downloads.close()
    close(runtime, owners, archive)


@pytest.mark.parametrize('raw_dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_ahead_and_baseline_pack_identical_delayed_producer_bytes_without_publication(tmp_path, raw_dtype):
    snapshots = []
    for enabled in (False, True):
        runtime, a, b, archive = gpu_runtime(tmp_path/str(enabled))
        runtime.raw_dtype = raw_dtype
        producer = torch.cuda.Stream(device=runtime.runner.device)
        producer.wait_stream(torch.cuda.current_stream(runtime.runner.device))
        try:
            with torch.cuda.stream(producer):
                captures, batch, rejected = delayed_inputs(runtime, [('a', 4, 4, True)], 3)
                def action_fence():
                    # Stand-in for native validation's host fence; no score
                    # correctness claim is made by this pack/DMA component test.
                    done = torch.cuda.Event()
                    done.record(producer)
                    done.synchronize()
                    assert not runtime.queue and a.history_end == 4
                prepare(runtime, batch, [], enabled=enabled, capture=action_fence)
                runtime.submit(captures, batch, rejected)
            item, = runtime.queue
            assert not item.confirmed and not runtime.drain(wait=False)
            assert (a.history_end, b.history_end) == (4, 4)
            item.download.event.synchronize()
            snapshots.append(item.download.payload.clone())
            raw = torch.stack([(torch.arange(8).reshape(4, 1, 2)+layer*13+3).bfloat16()
                               for layer in range(2)]).float().clamp(-448, 448).to(raw_dtype)
            values = torch.full((2, 4, 1, 2), 1.75).to(torch.float8_e4m3fn)
            expected = torch.cat((torch.tensor([0, 4, 4, 4, 0], dtype=torch.int64).view(torch.uint8),
                                  raw.view(torch.uint8).flatten(), values.view(torch.uint8).flatten()))
            assert torch.equal(snapshots[-1], expected)
        finally:
            shutdown(runtime, (a, b), archive)
    assert torch.equal(*snapshots)


def test_cuda_dma_cancel_clips_prefix_then_reuses_same_fixed_slot(tmp_path):
    runtime, a, b, archive = gpu_runtime(tmp_path)
    pointers = tuple(buffer.data_ptr() for buffer in runtime.downloads.buffers)
    producer = torch.cuda.Stream(device=runtime.runner.device)
    producer.wait_stream(torch.cuda.current_stream(runtime.runner.device))
    try:
        with torch.cuda.stream(producer):
            captures, batch, rejected = delayed_inputs(runtime, [('a', 4, 4, True)], 2)
            prepare(runtime, batch, [])
            runtime.submit(captures, batch, rejected)
        first, = runtime.queue
        assert not runtime.drain(wait=False) and a.history_end == 4
        runtime.begin_step({'a'}, {'a': TerminalPrefix(6, 8, 0, 'FINISHED_ABORTED')})
        assert runtime.drain_requests({'a'}, wait=True)
        assert a.history_end == 6 and not runtime.downloads.pending
        assert runtime.last_steps['a'][-1]['accepted'] == 2
        with torch.cuda.stream(producer):
            captures, batch, rejected = delayed_inputs(runtime, [('b', 4, 4, True)], 9)
            prepare(runtime, batch, [])
            runtime.submit(captures, batch, rejected)
        second, = runtime.queue
        assert second.download.slot == first.download.slot
        assert tuple(buffer.data_ptr() for buffer in runtime.downloads.buffers) == pointers
        assert not second.confirmed and b.history_end == 4
        second.download.event.synchronize()
        raw_begin = 5*8
        raw = second.download.payload[raw_begin:raw_begin+32].view(torch.bfloat16).reshape(2, 4, 1, 2)
        expected = torch.stack([(torch.arange(8).reshape(4, 1, 2)+i*13+9).bfloat16() for i in range(2)])
        assert torch.equal(raw, expected)
        runtime.begin_step()
        assert runtime.drain_requests({'b'}, wait=True) and b.history_end == 8
    finally:
        shutdown(runtime, (a, b), archive)


def test_action_exception_waits_own_cuda_event_and_recycles_only_its_private_slot(tmp_path, monkeypatch):
    runtime, a, b, archive = gpu_runtime(tmp_path)
    pointers = tuple(buffer.data_ptr() for buffer in runtime.downloads.buffers)
    producer = torch.cuda.Stream(device=runtime.runner.device)
    producer.wait_stream(torch.cuda.current_stream(runtime.runner.device))
    original_event = None
    try:
        with torch.cuda.stream(producer):
            captures, batch, rejected = delayed_inputs(runtime, [('b', 4, 4, True)], 1)
            prepare(runtime, batch, [])
            runtime.submit(captures, batch, rejected)
        old, = runtime.queue
        original_event = old.download.event
        old.download.event = NS(query=original_event.query,
            synchronize=lambda: pytest.fail('Unrelated DMA was explicitly synchronized'))
        with monkeypatch.context() as guarded:
            guarded.setattr(torch.cuda, 'synchronize', lambda *a, **k: pytest.fail('Global GPU fence'))
            with torch.cuda.stream(producer):
                captures, batch, rejected = delayed_inputs(runtime, [('a', 4, 4, True)], 5)
                def fail_action():
                    assert runtime.queue == [old] and len(runtime.downloads.pending) == 2
                    raise ValueError('injected Action failure after private CUDA download')
                prepare(runtime, batch, [], capture=fail_action)
                with pytest.raises(ValueError, match='injected Action'):
                    runtime.submit(captures, batch, rejected)
        assert runtime.queue == [old] and len(runtime.downloads.pending) == 1
        assert runtime.downloads.pending[old.download.slot] is old.download
        assert old.download.owners and (a.history_end, b.history_end) == (4, 4)
        assert runtime.prefill_pipeline_stats['capture_ahead_discarded'] == 1
        with torch.cuda.stream(producer):
            captures, batch, rejected = delayed_inputs(runtime, [('a', 4, 4, True)], 11)
            prepare(runtime, batch, [])
            runtime.submit(captures, batch, rejected)
        replacement = runtime.queue[-1]
        replacement.download.event.synchronize()
        assert replacement.download.slot != old.download.slot
        assert tuple(buffer.data_ptr() for buffer in runtime.downloads.buffers) == pointers
        raw = replacement.download.payload[40:72].view(torch.bfloat16).reshape(2, 4, 1, 2)
        expected = torch.stack([(torch.arange(8).reshape(4, 1, 2)+i*13+11).bfloat16() for i in range(2)])
        assert torch.equal(raw, expected) and not replacement.confirmed
        old.download.event = original_event
    finally:
        if original_event is not None and runtime.queue and runtime.queue[0].request_ids == ('b',):
            runtime.queue[0].download.event = original_event
        shutdown(runtime, (a, b), archive)
