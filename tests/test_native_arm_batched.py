"""KVMEM_NATIVE_ARM_BATCHED: one stacked LSE clear writes exactly what per-layer clears write."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='Explicit GPU qualification only')


@pytest.mark.parametrize('rows,active', [(7, 5), (64, 64), (130, 3)])
def test_stacked_clear_equals_per_layer_clears(rows, active):
    import torch
    import triton
    from vllm_kvmem.serving.native_attention import _clear_lse, _clear_lse_layers
    torch.manual_seed(rows)
    layers, heads, width = 17, 4, 600
    stack = torch.randn((layers, rows, heads, width), device='cuda')
    reference = stack.clone()
    meta = torch.zeros((rows, 3), dtype=torch.int32, device='cuda')
    meta[:, 1] = torch.randint(1, width + 1, (rows,), device='cuda', dtype=torch.int32)
    control = torch.tensor([active], dtype=torch.int32, device='cuda')
    tiles = triton.cdiv(int(meta[:active, 1].max()), 256)
    for layer in range(layers):
        _clear_lse[(active, heads, tiles)](reference[layer], meta, control, heads, width, 256)
    _clear_lse_layers[(active, heads, tiles * layers)](stack, meta, control, stack.stride(0), tiles, heads, width, 256)
    torch.cuda.synchronize()
    assert torch.equal(stack.isneginf(), reference.isneginf())
    assert torch.equal(torch.nan_to_num(stack, neginf=0.), torch.nan_to_num(reference, neginf=0.))
