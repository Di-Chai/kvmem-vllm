import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1",
                                reason="Explicit GPU qualification only")

from vllm_kvmem.transfer import PackedDownloads


def test_completed_download_cannot_be_reused_until_cpu_consumer_releases():
    pool = PackedDownloads(4096, 1, torch.device("cuda"))
    source = torch.arange(4096, dtype=torch.int32, device="cuda").to(torch.uint8)
    download = pool.download(source)
    download.event.synchronize()
    assert pool.download(torch.zeros_like(source)) is None
    assert not pool.close()
    assert torch.equal(download.payload, source.cpu())
    saved = download.payload.clone()
    assert pool.release(download)
    second = pool.download(torch.zeros_like(source))
    with pytest.raises(ValueError, match="Stale"):
        pool.release(download)
    second.event.synchronize()
    assert not torch.any(second.payload).item()
    assert torch.equal(saved, source.cpu())
    assert pool.release(second)
    assert pool.close()


def test_decode_capture_dma_completes_before_delayed_bulk_stream(tmp_path):
    """The production split packet has its own event/slot, not a shared fence."""
    from types import SimpleNamespace as NS
    import numpy as np
    from vllm_kvmem.capture_runtime import CaptureRuntime
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.raw_dtype = torch.float8_e4m3fn
    runtime.names = ('target',)
    runtime.options = NS(query_history=False)
    runtime.queue = []; runtime.authorities = {}; runtime.batches = 0
    runtime.row_elements = 64
    runtime.scales = torch.ones(1, device='cuda')
    runtime.split_capture_dma = True; runtime.decode_capture_limit = 5
    runtime.downloads = PackedDownloads(65*64*2+72, 2, torch.device('cuda'))
    runtime.decode_downloads = PackedDownloads(5*64*2+72, 2, torch.device('cuda'))
    positions = torch.arange(65, device='cuda')
    captured = [NS(name='target', positions=positions,
        key=torch.arange(65*64, device='cuda').reshape(65,1,64).remainder(223).bfloat16(),
        value=torch.ones(65,1,64,device='cuda',dtype=torch.bfloat16))]
    batch = NS(num_reqs=2, num_tokens=65, req_ids=('bulk', 'decode'),
        query_start_loc=torch.tensor([0,64,65],device='cuda'),
        query_start_loc_np=np.array([0,64,65]), positions=positions,
        has_prefill=True, is_prefilling_np=np.array([True,False]))
    rejected = torch.zeros(2, dtype=torch.int64, device='cuda')
    # Warm the actual packing shapes before injecting a finite test-only delay.
    for _, indices in runtime._capture_groups(batch):
        runtime._capture_payload(captured, batch, rejected, indices)
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(runtime.downloads.stream):
            torch.cuda._sleep(2_000_000_000)
        runtime.submit(captured, batch, rejected)
        small, bulk = runtime.queue
        assert small.lane == 'decode' and bulk.lane == 'bulk'
        small.download.event.synchronize()
        assert not bulk.download.query(), 'Decode inherited the delayed bulk DMA fence'
        assert small.download.owners and bulk.download.owners
        assert not small.confirmed and not bulk.confirmed
        bulk.download.event.synchronize()
        for packet in runtime.queue:
            header = packet.download.payload[:40].view(torch.int64).tolist()
            assert header == ([0,1,64,64,0] if packet.lane == 'decode' else [0,64,0,0,0])
            expected = captured[0].key[64:] if packet.lane == 'decode' else captured[0].key[:64]
            raw = packet.download.payload[40:40+packet.tokens*64]
            assert torch.equal(raw[:expected.numel()], expected.float().to(torch.float8_e4m3fn).view(torch.uint8).flatten().cpu())
            assert runtime._download_pool(packet.lane).release(packet.download)
    finally:
        for pool in (runtime.downloads, runtime.decode_downloads):
            for download in list(pool.pending.values()):
                download.event.synchronize()
                pool.release(download)
            assert pool.close()
