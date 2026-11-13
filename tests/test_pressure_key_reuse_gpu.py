"""KVMEM_LAYERWISE_PRESSURE_GPU_REUSE kernels against byte references (GPU)."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')

HEADS, DIM, HALF, PAGE, T = 4, 256, 32, 16, 32
ROW = T * HEADS * DIM


def trig_table(positions):
    inv = 1e7 ** (-torch.arange(0, 2 * HALF, 2, dtype=torch.float64) / (2 * HALF))
    angle = torch.arange(positions, dtype=torch.float64)[:, None] * inv[None, :]
    return torch.cat((angle.cos(), angle.sin()), 1).float().cuda()


def make_layer(blocks, scale):
    from vllm_kvmem.serving.window_assembly import LayerWindow
    pages = blocks * T // PAGE
    key = torch.zeros((pages, PAGE, HEADS, DIM), dtype=torch.float8_e4m3fn, device='cuda')
    table = torch.randperm(pages, generator=torch.Generator().manual_seed(3)).to(torch.int32).cuda()
    return LayerWindow(key, torch.zeros_like(key), trig_table(4096), torch.tensor(scale, device='cuda'), table)


def block_bytes(cache, table, base):
    positions = torch.arange(base, base + T, device='cuda')
    pages = table.long()[positions // PAGE]
    return cache.view(torch.uint8)[pages, positions % PAGE].clone()


def write_exact(layer, raw, values, bases):
    from vllm_kvmem.serving.split_assembly import assemble_split_layer
    n = bases.numel()
    index = torch.arange(n, device='cuda')
    assemble_split_layer(raw.flatten(), index, values.flatten(), index, bases, layer, layer_index=0,
                         layers=1, block_tokens=T, raw_record_bytes=ROW, value_record_bytes=ROW,
                         raw_dtype=torch.float8_e4m3fn)


def reuse_reference(old_key_bytes, delta, trig):
    x = old_key_bytes.view(torch.float8_e4m3fn).float()                 # (T, HEADS, DIM), scaled
    cos, sin = trig[delta, :HALF], trig[delta, HALF:2 * HALF]
    first, second = x[..., :HALF], x[..., HALF:2 * HALF]
    out = x.clone()
    out[..., :HALF] = first * cos + second * sin                          # rotate by -delta
    out[..., HALF:2 * HALF] = second * cos - first * sin
    return out.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)


def run_reuse(layer, raw, moves):
    """The assemble_reused GPU sequence without staging: whole snapshots, then reloads."""
    from vllm_kvmem.serving.pressure_key_reuse import REUSE, gather_rows, scatter_rows, write_keys
    upload = lambda xs: torch.tensor(xs, dtype=torch.int64, device='cuda')
    pages = layer.kernel_page_table
    reuse = [m for m in moves if m[5] == REUSE]
    reload = [m for m in moves if m[5] != REUSE]
    snapshot = torch.empty(len(moves) * ROW, dtype=torch.uint8, device='cuda')
    gather_rows(layer.value, upload([m[2] for m in moves]), pages, snapshot)
    scatter_rows(snapshot, upload([m[3] for m in moves]), pages, layer.value)
    if reuse:
        gather_rows(layer.key, upload([m[2] for m in reuse]), pages, snapshot[:len(reuse) * ROW])
        write_keys(snapshot, upload(list(range(len(reuse)))), upload([0] * len(reuse)),
                   upload([m[4] for m in reuse]), upload([m[3] for m in reuse]), layer, rows=ROW)
    if reload:
        payload = torch.cat([raw[m[1] // T].view(torch.uint8).flatten() for m in reload])
        n = len(reload)
        write_keys(payload, upload(list(range(n))), upload([1] * n), upload([0] * n),
                   upload([m[3] for m in reload]), layer, rows=ROW)


def test_in_place_reuse_matches_references():
    from vllm_kvmem.serving.pressure_key_reuse import REUSE, plan_layer
    torch.manual_seed(0)
    blocks, scale = 12, .0625
    layer = make_layer(blocks, scale)
    raw = (torch.randn(blocks, T, HEADS, DIM) * 2).to(torch.float8_e4m3fn).cuda()
    values = torch.randint(0, 255, (blocks, T, HEADS, DIM), dtype=torch.uint8).cuda()
    write_exact(layer, raw.view(torch.uint8), values, torch.arange(0, blocks * T, T, device='cuda'))
    kept = [b for b in range(blocks) if b not in (2, 5)]
    counts = {b * T: c for b, c in zip(kept, [0, 0, 0, 2, 1, 0, 2, 0, 1, 0])}
    moves, _, skipped = plan_layer([(b * T, T) for b in kept], [b * T for b in kept], counts, limit=2)
    assert skipped == 2 and {m[5] for m in moves} == {0, 1}
    old_key = {b: block_bytes(layer.key, layer.kernel_page_table, b * T) for b in range(blocks)}
    old_value = {b: block_bytes(layer.value, layer.kernel_page_table, b * T) for b in range(blocks)}
    reference = make_layer(blocks, scale)           # exact reload: the original kernel at the new position
    for m in moves:
        b = m[1] // T
        write_exact(reference, raw[b:b+1].view(torch.uint8), values[b:b+1], torch.tensor([m[3]], device='cuda'))
    run_reuse(layer, raw, moves)
    torch.cuda.synchronize()
    for b in kept[:skipped]:                        # unmoved prefix
        assert torch.equal(block_bytes(layer.key, layer.kernel_page_table, b * T), old_key[b])
        assert torch.equal(block_bytes(layer.value, layer.kernel_page_table, b * T), old_value[b])
    for m in moves:
        b, destination = m[1] // T, m[3]
        key = block_bytes(layer.key, layer.kernel_page_table, destination)
        if m[5] == REUSE:
            expected = reuse_reference(old_key[b], m[4], layer.cos_sin)
            assert torch.equal(key[..., 2 * HALF:], old_key[b][..., 2 * HALF:])   # position-free dims copied
        else:
            expected = block_bytes(reference.key, reference.kernel_page_table, destination)
        assert torch.equal(key, expected), (m, (key != expected).sum().item())
        assert torch.equal(block_bytes(layer.value, layer.kernel_page_table, destination), old_value[b])


def test_reused_keys_stay_close_to_a_fresh_rotation_after_many_moves():
    from vllm_kvmem.serving.pressure_key_reuse import REUSE
    torch.manual_seed(1)
    scale = .0625
    layer = make_layer(4, scale)
    raw = (torch.randn(1, T, HEADS, DIM) * 2).to(torch.float8_e4m3fn).cuda()
    values = torch.zeros((1, T, HEADS, DIM), dtype=torch.uint8).cuda()
    write_exact(layer, raw.view(torch.uint8), values, torch.tensor([96], device='cuda'))
    position = 96
    for step in range(32):                          # 96 -> 64 -> 32 -> 0, then wrap with an exact reload
        target = (position - T) % (4 * T)
        if target > position:
            write_exact(layer, raw.view(torch.uint8), values, torch.tensor([target], device='cuda'))
        else:
            run_reuse(layer, raw, [(0, 0, position, target, position - target, REUSE)])
        position = target
    fresh = make_layer(4, scale)
    write_exact(fresh, raw.view(torch.uint8), values, torch.tensor([position], device='cuda'))
    got = block_bytes(layer.key, layer.kernel_page_table, position).view(torch.float8_e4m3fn).float()
    want = block_bytes(fresh.key, fresh.kernel_page_table, position).view(torch.float8_e4m3fn).float()
    relative = ((got - want).norm() / want.norm()).item()
    assert relative < .08, relative
