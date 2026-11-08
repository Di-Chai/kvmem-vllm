"""KVMEM_STREAM_PUBLICATION: adopted pages equal the ordinary append, byte for byte."""
import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.streaming_publication import allocate_pages, verify_page

L, B, H, D = 5, 32, 2, 64


def owner(monkeypatch):
    monkeypatch.setenv('KVMEM_AUTHORITY_BATCHED_MEANS', '1')
    cfg = KVMemConfig(block_tokens=B, select_budget=B*64, gen_budget=B*16,
                      repository_blocks=512, sink_blocks=1, recent_blocks=0)
    archive = MemoryArchive('', L*B*H*D*2, 512, cpu_direct_record_seal=True)
    archive.async_seal = True
    return RequestAuthority('r', cfg, archive, L, H, D, raw_dtype=torch.float8_e4m3fn, scoring_layers=L-1,
                            mean_k_backend='qw3_cuda', resident_records=512, cpu_mean_index_simd=True), archive


def chunk(seed, tokens):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (L, tokens, H, D), dtype=torch.uint8, generator=g)
    raw[raw == 127] = 126; raw[raw == 255] = 254
    return raw.view(torch.float8_e4m3fn), torch.randint(0, 256, (L, tokens, H, D), dtype=torch.uint8, generator=g)


def stream_pages(a, raw, values, history_begin):
    """What the worker does: borrow pages, per-layer copy-only scatter, per-layer sums."""
    tokens = raw.shape[1]
    pages = allocate_pages(a, tokens // B, [])
    src_raw, src_val = raw.view(torch.uint8).numpy(), values.numpy()
    scratch = np.zeros((1, H, D), np.float32)
    for layer in range(L):
        one = slice(layer, layer+1)
        _native.scatter_history_batch([(src_raw[one], src_val[one], r.view(torch.uint8).numpy()[one],
            v.numpy()[one], None, s.numpy()[one] if layer < L-1 else scratch, p*B, 0, B, True)
            for p, (r, v, _, s, _) in enumerate(pages)])
        if layer < L-1:
            _native.finalize_mean_batch([(r.view(torch.uint8).numpy()[one], s.numpy()[one], B, True)
                                         for r, _, _, s, _ in pages], vectorized=True)
    a.register_prefilled({history_begin + p*B: page for p, page in enumerate(pages)})


def snapshot(a, archive):
    records = sorted(a.records.values(), key=lambda r: r.history_begin)
    a.publish_sealed(lambda: archive.wait_for_capacity(.05))
    payloads = a.read_records([r.handle for r in records])
    return [(r.history_begin, r.count, k[:, :r.count].contiguous().view(torch.uint8).numpy().tobytes(),
             v[:, :r.count].contiguous().numpy().tobytes(),
             None if r.mean_key is None else r.mean_key.contiguous().view(torch.int16).numpy().tobytes())
            for r, (k, v) in zip(records, payloads)]


@pytest.mark.parametrize('zero_copy', [False, True])
@pytest.mark.parametrize('verify', [False, True])
def test_streamed_pages_publish_identically(monkeypatch, verify, zero_copy):
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_VERIFY', '1' if verify else '0')
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_ZERO_COPY', '1' if zero_copy else '0')
    results = []
    for stream in (False, True):
        a, archive = owner(monkeypatch)
        history = 0
        for seed, tokens in ((1, 1024), (2, 512), (3, 1024)):
            raw, values = chunk(seed, tokens)
            if stream:
                stream_pages(a, raw, values, history)
            for begin in range(0, tokens, 512):     # Two logical spans per physical chunk.
                assert append_authority_batch([(a, raw[:, begin:begin+512], values[:, begin:begin+512],
                    dict(history_begin=history+begin, local_rope_begin=history+begin))]) == 1
            history += tokens
        if stream:
            assert a.streamed_pages == history // B and not a._prefilled
        results.append(snapshot(a, archive))
        if stream:
            adopted = getattr(archive, '_seal_zero_copy_records', 0)
            assert adopted == (history // B if zero_copy else 0)
            if zero_copy:
                assert not a._busy_record_buffers and not a._record_buffers   # no staging used
    assert results[0] == results[1]


def test_unadopted_pages_are_returned(monkeypatch):
    a, archive = owner(monkeypatch)
    raw, values = chunk(4, 256)
    stream_pages(a, raw, values, 0)
    busy = set(a._busy_record_buffers)
    assert len(busy) == 8
    # Only the first half is accepted; the rest is discarded at claim end.
    assert append_authority_batch([(a, raw[:, :128], values[:, :128],
        dict(history_begin=0, local_rope_begin=0))]) == 1
    assert a.discard_prefilled(range(0, 256, B)) == 4 and not a._prefilled


def test_verification_detects_a_wrong_page(monkeypatch):
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_VERIFY', '1')
    a, archive = owner(monkeypatch)
    raw, values = chunk(5, 64)
    stream_pages(a, raw, values, 0)
    a._prefilled[32][1].numpy()[0, 0, 0, 0] ^= 1
    with pytest.raises(RuntimeError, match='differs from the published packet'):
        append_authority_batch([(a, raw, values, dict(history_begin=0, local_rope_begin=0))])


def test_zero_copy_partial_pages_and_discard(monkeypatch):
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_ZERO_COPY', '1')
    a, archive = owner(monkeypatch)
    raw, values = chunk(6, 256)
    stream_pages(a, raw, values, 0)
    assert all(page[2] is None and type(page[4]) is bytes for page in a._prefilled.values())
    # Accept 4 pages plus a partial page: the seal mixes adopted and copied records.
    assert append_authority_batch([(a, raw[:, :144], values[:, :144],
        dict(history_begin=0, local_rope_begin=0))]) == 1
    assert a.discard_prefilled(range(0, 256, B)) == 4 and not a._prefilled
    a.publish_sealed(lambda: archive.wait_for_capacity(.05))
    assert archive._seal_zero_copy_records == 4


def test_private_record_views_keep_their_bytes_alive():
    import gc
    from vllm_kvmem.streaming_publication import private_record
    record, view = private_record(4096)
    tensor = torch.from_numpy(view[:2048])
    del record, view
    gc.collect()
    tensor.fill_(7)
    assert int(tensor.sum()) == 7 * 2048


def test_flag_validation(monkeypatch):
    from vllm_kvmem import streaming_publication as sp
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_ZERO_COPY', 'on')
    with pytest.raises(ValueError, match='ZERO_COPY'):
        sp.zero_copy_requested()
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_DELAY_MS', '900')
    with pytest.raises(ValueError, match='DELAY_MS'):
        sp.start_delay()


@pytest.mark.parametrize('simd', [False, True])
def test_native_stream_pages_match_the_python_worker(monkeypatch, simd):
    if getattr(_native, 'StreamPages', None) is None:
        pytest.skip('Requires the StreamPages native extension')
    a, _ = owner(monkeypatch)
    raw, values = chunk(7, 256)
    # Source as the worker sees it: a [:, :count] slice of a larger pinned-style buffer.
    big_raw = torch.zeros((L, 320, H, D), dtype=torch.uint8); big_raw[:, :256] = raw.view(torch.uint8)
    big_val = torch.zeros((L, 320, H, D), dtype=torch.uint8); big_val[:, :256] = values
    src_raw, src_val = big_raw[:, :256].numpy(), big_val[:, :256].numpy()
    results = []
    for native in (False, True):
        pages = allocate_pages(a, 8, [])
        pr = [p[0].view(torch.uint8).numpy() for p in pages]
        pv = [p[1].numpy() for p in pages]
        ps = [p[3].numpy() for p in pages]
        if native:
            stream = _native.StreamPages(src_raw, src_val, pr, pv, ps, B)
            assert (stream.layers, stream.scoring_layers) == (L, L-1)
            for layer in reversed(range(L)):                 # any layer order
                stream.layer(layer, simd)
        else:
            scratch = np.zeros((1, H, D), np.float32)
            for layer in range(L):
                one = slice(layer, layer+1)
                _native.scatter_history_batch([(src_raw[one], src_val[one], pr[p][one], pv[p][one], None,
                    ps[p][one] if layer < L-1 else scratch, p*B, 0, B, True) for p in range(8)])
                if layer < L-1:
                    _native.finalize_mean_batch([(pr[p][one], ps[p][one], B, True) for p in range(8)], vectorized=simd)
        results.append([(r.tobytes(), v.tobytes(), s.tobytes()) for r, v, s in zip(pr, pv, ps)])
        for page in pages:
            a._release_staging_slot(page[2])
    assert results[0] == results[1]


def test_native_stream_pages_reject_bad_geometry(monkeypatch):
    if getattr(_native, 'StreamPages', None) is None:
        pytest.skip('Requires the StreamPages native extension')
    a, _ = owner(monkeypatch)
    pages = allocate_pages(a, 2, [])
    pr = [p[0].view(torch.uint8).numpy() for p in pages]
    pv = [p[1].numpy() for p in pages]
    ps = [p[3].numpy() for p in pages]
    src = np.zeros((L, 3*B, H, D), np.uint8)                    # 3 pages of rows for 2 pages
    with pytest.raises(ValueError, match='source'):
        _native.StreamPages(src, src.copy(), pr, pv, ps, B)
    src = np.zeros((L, 2*B, H, D), np.uint8)
    with pytest.raises(ValueError, match='overlap'):
        _native.StreamPages(src, src.copy(), [pr[0], pr[0]], pv, ps, B)
    stream = _native.StreamPages(src, src.copy(), pr, pv, ps, B)
    with pytest.raises(IndexError):
        stream.layer(L, True)


@pytest.mark.parametrize('verify', [False, True])
def test_prescatter_job_publishes_identically(monkeypatch, verify):
    from types import SimpleNamespace as NS
    from vllm_kvmem.streaming_publication import PrescatterJob
    if getattr(_native, 'StreamPages', None) is None:
        pytest.skip('Requires the StreamPages native extension')
    monkeypatch.setenv('KVMEM_STREAM_PUBLICATION_VERIFY', '1' if verify else '0')
    results = []
    for prescatter in (False, True):
        a, archive = owner(monkeypatch)
        history = 0
        for seed, spans in ((1, (512, 512)), (2, (256, 128, 96))):
            tokens = sum(spans)
            raw, values = chunk(seed, tokens)
            job = None
            if prescatter:
                offsets, spec = 0, []
                for count in spans:
                    spec.append(NS(history_begin=history + offsets, count=count)); offsets += count
                job = PrescatterJob(a, NS(history_begin=history, spans=tuple(spec)), raw, values, True)
                a._prefill_wait = job.wait_range
            begin = 0
            for count in spans:
                assert append_authority_batch([(a, raw[:, begin:begin+count], values[:, begin:begin+count],
                    dict(history_begin=history+begin, local_rope_begin=history+begin))]) == 1
                begin += count
            if job is not None:
                job.thread.join(10)
                a._prefill_wait = None
                a.discard_prefilled(job.registered)
            history += tokens
        if prescatter:
            assert a.streamed_pages == history // B and not a._prefilled
        results.append(snapshot(a, archive))
    assert results[0] == results[1]
