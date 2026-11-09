"""KVMEM_LAYER0_FAST GPU assembly: byte-identical to the CPU path's packed chunks.

The fast path writes layer 0's window with one split-assembly launch over the
shadow rows; the CPU path uploads the same records in chunks and calls the same
kernel per chunk, skipping the plan's kept blocks. Run the fast path from a
separate thread on inference tensors, exactly as the refresh producer creates
them, and compare every K/V cache byte and the gathered next shadow.
"""
import threading

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='Requires CUDA')

BLOCK, HEADS, DIM, PAGE = 32, 2, 64, 16


def fp8_rows(n, generator):
    values = torch.randn((n, BLOCK * HEADS * DIM), generator=generator, device='cuda') * 4
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def layer_window(pages, generator):
    from vllm_kvmem.window_assembly import LayerWindow
    key = torch.zeros((pages, PAGE, HEADS, DIM), dtype=torch.uint8, device='cuda').view(torch.float8_e4m3fn)
    value = torch.zeros_like(key)
    cos_sin = torch.randn((4096, DIM), generator=generator, device='cuda')
    table = torch.randperm(pages, generator=generator, device='cuda').to(torch.int32)
    return LayerWindow(key, value, cos_sin, torch.tensor([0.75], device='cuda'), table)


def reference(raw, val, sources, skip, layer):
    """The CPU path: contiguous non-skipped runs, each uploaded and assembled as one chunk."""
    from vllm_kvmem.layerwise_pressure_efficiency import packed_layer_chunk
    from types import SimpleNamespace as NS
    row_bytes = raw.shape[1]
    total = len(sources) * BLOCK
    state = {'selected_tokens': total}
    run = []
    def flush():
        if not run:
            return
        cursor = run[0][0] * BLOCK
        payload = torch.cat([raw[[j for _, j in run]].reshape(-1), val[[j for _, j in run]].reshape(-1)])
        records = [NS(count=BLOCK) for _ in run]
        assert packed_layer_chunk(state, layer, payload, len(run) * row_bytes, cursor, records,
                                  row_bytes, 256)
        run.clear()
    for i, j in enumerate(sources):
        if i in skip:
            flush()
            continue
        run.append((i, j))
    flush()


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_fast_window_assembly_is_byte_identical_to_cpu_chunks(seed):
    from vllm_kvmem.layer0_fast import assemble_window
    generator = torch.Generator(device='cuda').manual_seed(seed)
    window, chunk = 24, 8
    pages = (window * BLOCK) // PAGE + 4
    with torch.inference_mode():                       # as the refresh producer allocates them
        raw = torch.cat([fp8_rows(window + chunk, generator)])
        val = fp8_rows(window + chunk, generator)
        out_raw, out_val = torch.zeros_like(raw), torch.zeros_like(val)
        fast_layer, cpu_layer = layer_window(pages, generator), None
        cpu_layer = type(fast_layer)(fast_layer.key.clone(), fast_layer.value.clone(), fast_layer.cos_sin,
                                     fast_layer.key_scale, fast_layer.kernel_page_table)
    picks = torch.randperm(window + chunk, generator=generator, device='cuda')[:window].sort().values.tolist()
    sources = picks                                     # rows in dense order
    skip = frozenset(i for i, j in enumerate(sources) if j < window and j == i)
    stream = torch.cuda.Stream()
    result = {}

    def fast():                                         # the fast-path thread
        with torch.inference_mode():
            result['event'] = assemble_window(raw, val, out_raw, out_val, sources, skip, fast_layer,
                                              raw.shape[1], stream)
    worker = threading.Thread(target=fast)
    worker.start(); worker.join()
    result['event'].synchronize()
    with torch.inference_mode():
        reference(raw, val, sources, skip, cpu_layer)
    torch.cuda.synchronize()
    assert torch.equal(fast_layer.key.view(torch.uint8), cpu_layer.key.view(torch.uint8))
    assert torch.equal(fast_layer.value.view(torch.uint8), cpu_layer.value.view(torch.uint8))
    assert torch.equal(out_raw[:window], raw[sources]) and torch.equal(out_val[:window], val[sources])
    assert len(skip) < window                           # the case really rebuilds blocks


def test_failed_shadow_rebuild_drops_the_shadow_and_never_raises(monkeypatch):
    """A rebuild runs after the producer's own upload of a layer; if the plan ended
    meanwhile (the reader refuses), the refresh must continue without a shadow."""
    from types import SimpleNamespace as NS
    from vllm_kvmem import layer0_fast
    monkeypatch.setenv('KVMEM_LAYER0_FAST', '1')
    monkeypatch.setenv('KVMEM_LAYER0_FAST_LAYERS', '2')
    fast = layer0_fast.Layer0Fast(NS(device=torch.device('cuda')))
    shadow = fast.shadow('r', 1, create=(torch.device('cuda'), 4, 2, 64))
    shadow.window, shadow.chunk = (0, 32, 64, 96), (128, 2, None)

    class Reader:
        native = True
        def contains(self, records): return True
        def read(self, *a, **k): raise ValueError('Prepared pressure reader belongs to a different plan, owner or prefix')

    authority = NS(raw_element_bytes=1, shape=(17, 32, 1, 2))
    records = [NS(count=32, archived=True, history_begin=32 * i) for i in range(4)]
    assert fast.rebuild_layer('r', 1, authority, records, Reader(), 4, 2) is False
    assert shadow.window is None and shadow.chunk is None
    assert any(k.startswith('rebuild_failed:ValueError') for k in fast.counts)
    assert fast.rebuild_layer('r', 2, authority, records, Reader(), 4, 2) is False   # beyond K: ignored
