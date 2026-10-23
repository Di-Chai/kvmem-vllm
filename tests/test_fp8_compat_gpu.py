"""Independent Torch byte oracles for software E4M3FN on pre-FP8 GPUs."""
import os

import pytest
import torch
import triton
import triton.language as tl

from vllm_kvmem.fp8_compat import fp8_storage, fp8_to_float, float_to_fp8

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')


@triton.jit
def _decode(Input, Output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Input+i, i < N, 0)
    tl.store(Output+i, fp8_to_float(value, True), i < N)


@triton.jit
def _encode(Input, Output, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(Input+i, i < N, 0.)
    tl.store(Output+i, float_to_fp8(value), i < N)


def test_all_fp8_encodings_decode_exactly_including_signed_zero_and_nans():
    source = torch.arange(256, dtype=torch.uint8)
    expected = source.view(torch.float8_e4m3fn).float()
    actual = torch.empty(256, dtype=torch.float32, device='cuda')
    _decode[(1,)](source.cuda(), actual, 256, 256)
    actual = actual.cpu()
    finite = ~expected.isnan()
    assert torch.equal(actual[finite].view(torch.int32), expected[finite].view(torch.int32))
    assert torch.equal(actual.isnan(), expected.isnan())
    assert torch.equal(actual.signbit(), expected.signbit())


def test_encoding_matches_cpu_torch_for_all_bf16_patterns_and_fp32_midpoints():
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16).float()
    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoints = (positive[:-1] + positive[1:]) * .5
    midpoints = torch.cat((midpoints, torch.tensor([464.])))
    boundary = torch.cat((torch.nextafter(midpoints, torch.full_like(midpoints, -torch.inf)),
                          midpoints, torch.nextafter(midpoints, torch.full_like(midpoints, torch.inf))))
    random = torch.randn(10000, generator=torch.Generator().manual_seed(7211)) * 200
    source = torch.cat((bf16, boundary, -boundary, random))
    expected = source.to(torch.float8_e4m3fn).view(torch.uint8)
    actual = torch.empty(source.numel(), dtype=torch.uint8, device='cuda')
    _encode[(triton.cdiv(source.numel(), 256),)](source.cuda(), actual, source.numel(), 256,
                                               enable_fp_fusion=False)
    assert torch.equal(actual.cpu(), expected)


def test_fp8_storage_keeps_address_and_strided_cache_geometry():
    original = torch.empty((5, 2, 16, 128), dtype=torch.float8_e4m3fn, device='cuda')
    key = original.transpose(1, 2)[..., :64]
    converted, software = fp8_storage(key)
    assert software == (torch.cuda.get_device_capability(key.device) < (8, 9))
    assert converted.data_ptr() == key.data_ptr()
    assert converted.shape == key.shape and converted.stride() == key.stride()
    assert converted.dtype == (torch.uint8 if software else key.dtype)
