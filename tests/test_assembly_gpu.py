import os
import pytest

pytestmark = pytest.mark.skipif(os.getenv("KVMEM_RUN_GPU_TESTS") != "1", reason="explicit GPU opt-in required")


@pytest.mark.parametrize("head_major", [False, True, "native"])
def test_assembly_matches_native_vllm_fp8_cache_and_retains_v(head_major):
    import torch
    from vllm import _custom_ops as ops
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.codec import neox_rope, native_vllm_fp8_views
    torch.manual_seed(418)
    raw = torch.randn(23, 4, 256, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(raw)
    angles = torch.randn(1024, 32, device="cuda")
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    positions = torch.arange(23, device="cuda", dtype=torch.int64) + 50
    slots = torch.tensor(list(range(32, 48)) + list(range(7)), device="cuda", dtype=torch.int64)
    scale = torch.tensor(0.02, device="cuda")
    def cache():
        shape = (4, 4, 16, 256) if head_major else (4, 16, 4, 256)
        x = torch.zeros(shape, dtype=torch.float8_e4m3fn, device="cuda")
        return x.transpose(1, 2) if head_major else x
    if head_major == "native":
        native = torch.zeros(4, 16, 4, 512, dtype=torch.uint8, device="cuda").transpose(1, 2)
        actual = torch.zeros_like(native)
        native_k, native_v = native_vllm_fp8_views(native, 4, 256)
        actual_k, actual_v = native_vllm_fp8_views(actual, 4, 256)
    else:
        native_k, native_v, actual_k, actual_v = [cache() for _ in range(4)]
    rotated = neox_rope(raw, trig[positions, :32], trig[positions, 32:])
    ops.reshape_and_cache_flash(rotated, value, native_k, native_v, slots, "fp8_e4m3", scale, scale)
    staged_v = (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    assemble_fp8(raw, staged_v, positions, slots, trig, actual_k, actual_v, scale)
    torch.cuda.synchronize()
    assert torch.equal(actual_k.view(torch.uint8), native_k.view(torch.uint8))
    assert torch.equal(actual_v.view(torch.uint8), native_v.view(torch.uint8))
    saved_v = actual_v.view(torch.uint8).clone()
    assemble_fp8(raw, None, positions + 17, slots, trig, actual_k, actual_v, scale)
    assert torch.equal(actual_v.view(torch.uint8), saved_v)
