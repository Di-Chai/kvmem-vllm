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
