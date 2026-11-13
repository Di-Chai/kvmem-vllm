"""CPU byte oracle at FP8 midpoints, including the Ada double-rounding case."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')


@pytest.mark.parametrize('implementation', ['direct', 'packed', 'split', 'fused'])
@pytest.mark.parametrize('scale_value', [1., .04052734375])
def test_assembly_rounds_fp32_to_fp8_once(implementation, scale_value):
    from vllm_kvmem.adapters.assembly import assemble_fp8
    from vllm_kvmem.serving.packed_assembly import assemble_packed_layer
    from vllm_kvmem.serving.split_assembly import (
        assemble_split_layer, assemble_split_layers, prepare_fused_layer_rows)
    from vllm_kvmem.serving.window_assembly import LayerWindow

    positive = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    midpoint = (positive[:-1] + positive[1:]) * .5
    edges = torch.cat((torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf)),
                       midpoint, torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf))))
    cosine = torch.cat((edges, -edges))
    count, page = cosine.numel(), 16
    # With raw K = [1, 0] and sin = 0, rotation yields [cosine, 0].
    # Synthetic trig deliberately places outputs next to every FP8 midpoint.
    raw_cpu = torch.tensor([1., 0.]).repeat(count, 1).reshape(count, 1, 2).to(torch.float8_e4m3fn)
    trig_cpu = torch.stack((cosine, torch.zeros_like(cosine)), dim=1)
    expected = (trig_cpu / scale_value).clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    raw = raw_cpu.cuda()
    values = torch.arange(count * 2, dtype=torch.int32).to(torch.uint8).reshape(count, 1, 2).cuda()
    capacity = (count + page - 1) // page
    key = torch.zeros((capacity, page, 1, 2), dtype=torch.float8_e4m3fn, device='cuda')
    value = torch.zeros_like(key)
    positions = torch.arange(count, dtype=torch.int64, device='cuda')
    table = torch.arange(capacity, dtype=torch.int32, device='cuda')
    layer = LayerWindow(key, value, trig_cpu.cuda(), torch.tensor(scale_value, device='cuda'), table)
    common = dict(layers=1, block_tokens=1, raw_dtype=torch.float8_e4m3fn)
    if implementation == 'direct':
        assemble_fp8(raw, values.view(torch.float8_e4m3fn), positions, positions,
                     layer.cos_sin, key, value, layer.key_scale)
    elif implementation == 'packed':
        payload = torch.cat((raw.view(torch.uint8).reshape(count, 2), values.reshape(count, 2)), dim=1)
        assemble_packed_layer(payload.flatten(), positions, layer, layer_index=0, **common)
    elif implementation == 'split':
        assemble_split_layer(raw.view(torch.uint8).flatten(), positions, values.flatten(),
                             positions, positions, layer, layer_index=0,
                             raw_record_bytes=2, value_record_bytes=2, **common)
    else:
        spec, reason = prepare_fused_layer_rows([layer])
        assert reason is None
        rows = torch.tensor(spec.rows, dtype=torch.int64, device='cuda')
        assemble_split_layers(raw.view(torch.uint8).flatten(), positions, values.flatten(),
                              positions, positions, rows, spec.geometry,
                              raw_record_bytes=2, value_record_bytes=2, **common)
    actual = key.view(torch.uint8).reshape(-1, 2).cpu()
    assert torch.equal(actual[:count], expected)
    assert not actual[count:].any()
    copied = value.view(torch.uint8).reshape(-1, 2).cpu()
    assert torch.equal(copied[:count], values.reshape(count, 2).cpu())
    assert not copied[count:].any()
