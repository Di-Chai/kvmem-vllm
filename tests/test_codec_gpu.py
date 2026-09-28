import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


def test_fp8_reselection_uses_raw_source_without_drift():
    import torch
    from vllm_kvmem.codec import KVFormat, materialize_k, neox_rope
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(123)
    raw = torch.randn(128, 2, 128, dtype=torch.bfloat16, device=device, generator=generator)
    before = raw.clone()
    fmt = KVFormat(torch.float8_e4m3fn, 128, 2, 16)
    scale = torch.tensor(0.02, device=device)
    inv_freq = 1.0 / (10000 ** (torch.arange(32, device=device).float() / 32))
    def at(base):
        angles = (torch.arange(128, device=device) + base)[:, None] * inv_freq
        return materialize_k(raw, angles.cos(), angles.sin(), fmt, scale)
    original = at(4096)
    for base in (0, 8192, 16, 120000, 4096):
        result = at(base)
    assert torch.equal(result.view(torch.uint8), original.view(torch.uint8))
    assert torch.equal(raw, before)
    angles = torch.arange(128, device=device)[:, None] * inv_freq
    rotated = neox_rope(raw, angles.cos(), angles.sin())
    assert torch.equal(rotated[..., 64:], raw[..., 64:])
    expected = rotated.float()
    restored = at(0).float() * scale
    relative = ((restored - expected).square().mean() / expected.square().mean()).sqrt()
    assert relative.item() < 0.04


@pytest.mark.parametrize("layout", ["2PNHD", "P2NHD", "P2HND"])
def test_physical_views_alias_backend_pages(layout):
    import torch
    from vllm_kvmem.codec import KVFormat, paged_nhd_view
    shapes = {"2PNHD": (2, 3, 16, 2, 64), "P2NHD": (3, 2, 16, 2, 64), "P2HND": (3, 2, 2, 16, 64)}
    tensor = torch.zeros(shapes[layout], dtype=torch.float8_e4m3fn, device="cuda")
    k, v = paged_nhd_view(tensor, KVFormat(tensor.dtype, 64, 2, 16), layout)
    k[1].fill_(2)
    v[2].fill_(3)
    assert k[1].float().mean().item() == 2
    assert v[1].float().sum().item() == 0
    assert tensor.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()


def test_packed_upload_owns_payload_until_cuda_completion():
    import torch
    from vllm_kvmem.transfer import PackedTransfers
    device = torch.device("cuda:0")
    pool = PackedTransfers(65536, 2, device)
    source = torch.arange(65536, dtype=torch.int32).to(torch.uint8)
    destinations = [torch.empty_like(source, device=device) for _ in range(2)]
    txns = [pool.upload(source, dest, source_owners=(object(),)) for dest in destinations]
    for txn in txns:
        torch.cuda.current_stream().wait_event(txn.event)
    for dest in destinations:
        assert torch.equal(dest.cpu(), source)
    assert pool.close()
