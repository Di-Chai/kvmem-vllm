"""Explicit CUDA qualification, never part of the CPU-only candidate preparation.

These low-level checks do not substitute for real-model cross-round/H=M/cancel
qualification. Synchronization below is an oracle readback, not serving behavior.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_metadata_upload import PackedActionMetadata
from vllm_kvmem.action_scoring import page_attention
from vllm_kvmem.action_attention import ActionHistory


@pytest.fixture(autouse=True)
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip('Explicit Action metadata CUDA qualification')


def test_packed_metadata_exact_fields_big_int64_and_original_stream():
    p = PackedActionMetadata()
    stream = torch.cuda.Stream()
    starts, counts, rows = [0, 5, 16], [5, 11, 1], [0, 1, 7]
    with torch.cuda.stream(stream):
        with p.lease(starts, counts, rows, 2**31+17, 2**53+3, 'cuda') as lease:
            actual = [x.clone() for x in lease.views]
            event = torch.cuda.Event()
            event.record(stream)
            lease.complete(event)
    stream.synchronize()
    expected = (starts, counts, [2**31+17+r for r in rows], [2**53+3+r for r in rows])
    assert tuple(x.dtype for x in actual) == (torch.int32, torch.int32, torch.int64, torch.int64)
    assert tuple(x.cpu().tolist() for x in actual) == expected
    assert p.snapshot()['uploaded_bytes'] == 8*len(starts)+16*len(rows)
    p.poll()
    assert p.snapshot()['pending_leases'] == 0


@pytest.mark.parametrize('q_dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_three_layers_recursive_scoring_and_confirmed_prefix_equal_to_legacy(q_dtype, *, fused=False):
    p = PackedActionMetadata()
    g = torch.Generator(device='cuda').manual_seed(841)
    length, dim, heads, kvheads, page = 259, 128, 8, 2, 16
    physical = (length+page-1)//page+2
    starts = [0, 16, 21]+list(range(32, length, 16))
    counts = [b-a for a, b in zip(starts, starts[1:]+[length])]
    rows, begin, local_begin = list(range(73)), 4, 128  # >64 invokes every recursive tile.
    cache = [(torch.randn(physical, kvheads, page, dim*2, device='cuda', generator=g)*.5)
             .to(torch.float8_e4m3fn) for _ in range(3)]
    queries = [torch.randn(80, heads, dim, device='cuda', generator=g, dtype=torch.bfloat16) for _ in range(3)]
    table = torch.randperm(physical, device='cuda', generator=g, dtype=torch.int32)
    angle = torch.randn(length, dim//4, device='cuda', generator=g)
    trig = torch.cat((angle.cos(), angle.sin()), -1)
    ks, qs = torch.tensor(.8, device='cuda'), torch.tensor(.3, device='cuda')
    def score(metadata):
        s, c, ids, pos = metadata
        values = [page_attention(q.index_select(0, ids), pos, raw.transpose(1, 2)[..., :dim], table,
                  trig, ks, s, c, length, scale=dim**-.5, q_dtype=q_dtype, q_scale=qs,
                  block_tokens=16, fused=fused, complete_partition=True) for q, raw in zip(queries, cache)]
        return (values[0]+values[1]+values[2])/3
    legacy = (torch.tensor(starts, device='cuda', dtype=torch.int32),
              torch.tensor(counts, device='cuda', dtype=torch.int32),
              torch.tensor([begin+r for r in rows], device='cuda', dtype=torch.int64),
              torch.tensor([local_begin+r for r in rows], device='cuda', dtype=torch.int64))
    expected = score(legacy)
    other_rows = [0, 4, 9]
    other_legacy = (legacy[0], legacy[1], torch.tensor([2+r for r in other_rows], device='cuda'),
                    torch.tensor([64+r for r in other_rows], device='cuda'))
    expected_other = score(other_legacy)
    with p.lease(starts, counts, rows, begin, local_begin, 'cuda') as lease:
        actual = score(lease.views)
        host = torch.empty(actual.shape, dtype=actual.dtype, pin_memory=True)
        host.copy_(actual, non_blocking=True)
        # Keep the first owner's lease unsealed while the second owner's
        # complete three-layer scoring uses independently packed metadata.
        with p.lease(starts, counts, other_rows, 2, 64, 'cuda') as other:
            assert len(p.pending) == 2 and other.offset != lease.offset
            actual_other = score(other.views)
            host_other = torch.empty(actual_other.shape, dtype=actual_other.dtype, pin_memory=True)
            host_other.copy_(actual_other, non_blocking=True)
            other_event = torch.cuda.Event()
            other_event.record()
            other.complete(other_event)
        event = torch.cuda.Event()
        event.record()
        lease.complete(event)
    event.synchronize()
    assert torch.equal(actual, expected)
    assert torch.equal(host, expected.cpu())
    assert torch.equal(actual_other, expected_other)
    assert torch.equal(host_other, expected_other.cpu())
    # Production transactional parser only commits confirmed speculative rows;
    # exercise >1 accepted rows while rejecting the remaining sampled rows.
    summaries = []
    for masses in (host, expected.cpu()):
        opts = NS(action_scope='non-thinking', action_interval=1, action_decay=.5)
        state = ActionHistory(opts, [10, 11, 12, 13, 14])
        state.begin('request', [14, 13])
        state.parser.prepare([20]*5)
        state.accept(3, list(range(5)), starts, masses[:5].tolist())
        summaries.append((state.samples, state.mass))
    assert summaries[0] == summaries[1] and summaries[0][0] == 3
    p.poll()
    assert not p.pending


def test_real_busy_consumer_event_prevents_cancelled_lease_reuse():
    if not hasattr(torch.cuda, '_sleep'):
        pytest.skip('CUDA sleep required to prove busy consumer fence')
    p = PackedActionMetadata(max_bytes=48, max_leases=1)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with pytest.raises(RuntimeError, match='cancel before packet publication'):
            with p.lease([0], [1], [0], 0, 0, 'cuda') as lease:
                torch.cuda._sleep(100_000_000)
                observed = lease.views[3].clone()
                raise RuntimeError('cancel before packet publication')
        assert not lease.event.query(), 'No busy-event coverage; do not treat this run as qualified'
        with p.lease([0], [1], [0], 99, 99, 'cuda') as fallback:
            assert fallback is None
        assert p.pending[lease.offset] is lease
    stream.synchronize()
    assert observed.cpu().tolist() == [0]
    p.poll()
    assert not p.pending and p.snapshot()['fallbacks'] == {'lease_limit': 1}
