import os
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
from vllm_kvmem.packed_assembly import read_packed_records
from test_authority import setup, payload, drain


@pytest.mark.parametrize('direct', [False, True])
def test_packed_records_preserve_order_bytes_and_owned_lifetime(tmp_path, direct):
    authority, archive = setup(tmp_path)
    assert authority.append(*payload(0, 9), history_begin=0, local_rope_begin=0)
    drain(authority)
    records = list(authority.records.values())
    handles = [records[1].handle, records[0].handle]
    destination = torch.empty(len(handles) * authority.record_bytes, dtype=torch.uint8) if direct else None
    packed, stats = read_packed_records(authority, handles, destination=destination)
    if direct:
        assert packed.data_ptr() == destination.data_ptr()
    expected = b''.join(authority.read_record(h)[0].view(torch.uint8).numpy().tobytes() +
                        authority.read_record(h)[1].numpy().tobytes() for h in handles)
    assert stats['bytes'] == len(expected)
    assert packed.numpy().tobytes() == expected
    with pytest.raises(ValueError, match='sealed'):
        read_packed_records(authority, [records[2].handle])
    assert authority.close() and archive.close()
    assert packed.numpy().tobytes() == expected  # Read payload owns bytes after retirement.


@pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1', reason='Explicit GPU qualification only')
@pytest.mark.parametrize('raw_dtype', [torch.bfloat16, torch.float16])
def test_packed_gpu_assembly_matches_reference_with_permuted_sources_and_pages(raw_dtype):
    from vllm_kvmem.assembly import assemble_fp8
    from vllm_kvmem.codec import native_vllm_fp8_views
    from vllm_kvmem.window_assembly import LayerWindow
    from vllm_kvmem.packed_assembly import assemble_packed_layer
    torch.manual_seed(129)
    layers, count, block, heads, dim = 2, 3, 16, 2, 64
    raw = torch.randn(count, layers, block, heads, dim, dtype=raw_dtype)
    values = torch.randn(count, layers, block, heads, dim).to(torch.float8_e4m3fn).view(torch.uint8)
    order = [2, 0, 1]
    encoded = b''.join(raw[i].view(torch.uint8).numpy().tobytes() + values[i].numpy().tobytes() for i in order)
    packed = torch.frombuffer(bytearray(encoded), dtype=torch.uint8).cuda()
    bases = torch.tensor([32, 0, 64], dtype=torch.int64, device='cuda')
    positions = torch.cat([torch.arange(int(base), int(base)+block, device='cuda') for base in bases])
    table = torch.tensor([7, 4, 9, 2, 6, 3, 10, 1, 8, 5], dtype=torch.int32, device='cuda')
    slots = table[positions // 8].long()*8 + positions % 8
    angles = torch.randn(128, 16, device='cuda')
    trig = torch.cat((angles.cos(), angles.sin()), dim=-1).to(raw_dtype)
    scale = torch.tensor(0.02, device='cuda')
    for i in range(layers):
        native = torch.zeros(12, heads, 8, dim * 2, dtype=torch.uint8, device='cuda')
        reference = native.clone()
        key, value = native_vllm_fp8_views(native, heads, dim)
        rk, rv = native_vllm_fp8_views(reference, heads, dim)
        layer = LayerWindow(key, value, trig, scale, table)
        assemble_packed_layer(packed, bases, layer, layer_index=i, layers=layers,
                              block_tokens=block, raw_dtype=raw_dtype)
        q = torch.cat([raw[j, i] for j in order]).cuda()
        v = torch.cat([values[j, i] for j in order]).view(torch.float8_e4m3fn).cuda()
        assemble_fp8(q, v, positions, slots, trig, rk, rv, scale)
        assert torch.equal(native, reference)
