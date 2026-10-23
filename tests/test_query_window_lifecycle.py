from threading import RLock
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_window_scoring import WindowMeanCache, score_authority_windows


def synthetic(length=71, archive=None, value=1.):
    keys = torch.arange(length, dtype=torch.float32)[None, :, None, None]*value
    records = [NS(history_begin=p, count=min(32, length-p),
        handle=NS(slot=p//32, generation=1)) for p in range(0, length, 32)]
    reads = []
    def read(handles):
        reads.append(tuple(h.slot for h in handles))
        return [(keys[:, h.slot*32:h.slot*32+records[h.slot].count], None) for h in handles]
    a = NS(history_end=length, read_records=read, shape=(1,32,1,1), scoring_layers=1,
        mean_k_backend='torch_reference')
    archive = archive or NS(_lock=RLock())
    return a, archive, records, reads, keys


def test_cache_reuses_means_rechecks_masks_and_updates_partial_tail_and_generation():
    a, archive, records, reads, keys = synthetic()
    cache = a.query_window_cache = WindowMeanCache(archive,1,1,1,dtype=torch.float32,max_bytes=10000)
    means, windows, stats = cache.snapshot(a,records)
    assert stats['cache_rebuilt_blocks']==3
    assert means[0,3,0,0]==keys[:,24:40].mean()
    assert means[0,8,0,0]==keys[:,64:71].mean()
    _,_,stats=cache.snapshot(a,records)
    assert stats['cache_hit_blocks']==3 and len(reads)==1
    fallback=torch.tensor([15.5,47.5,67.]).reshape(1,3,1,1)
    q=torch.ones(1,1,1,1)
    first,stat=score_authority_windows(a,records,q,fallback,[True,True,True])
    second,stat=score_authority_windows(a,records,q,fallback,[True,False,True])
    assert len(reads)==1 and second[1]==0 and first[1]>0
    assert stat['valid_windows']==4  # first three, and the actual tail
    records[2].count=5;a.history_end=69
    _,_,stat=cache.snapshot(a,records)
    assert stat['cache_rebuilt_blocks']==2  # preceding cross-window and shortened tail
    records[2].handle.generation=2;keys[:,64:]=100
    new,_,stat=cache.snapshot(a,records)
    assert stat['cache_rebuilt_blocks']==2 and new[0,8,0,0]==100
    cache.clear()
    assert archive._query_window_cache_bytes==0


def test_global_capacity_evicts_inactive_owner_but_never_active_scoring():
    a, archive, records, _, _ = synthetic(64)
    b, _, other, _, _ = synthetic(64,archive,value=3)
    ca=WindowMeanCache(archive,1,1,1,dtype=torch.float32,max_bytes=4128)
    cb=WindowMeanCache(archive,1,1,1,dtype=torch.float32,max_bytes=4128)
    with ca.use():
        first,_,_=ca.snapshot(a,records)
        with cb.use(),pytest.raises(MemoryError,match='active scoring'):
            cb.snapshot(b,other)
        assert first[0,3,0,0]==31.5
        del first
    with cb.use():
        second,_,_=cb.snapshot(b,other)
        assert second[0,3,0,0]==94.5 and ca.bytes==0
        del second
    assert archive._query_window_cache_bytes==4128
    cb.clear();assert archive._query_window_cache_bytes==0


def actual(tmp_path, *, capacity=65536):
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from vllm_kvmem.block_graph import BlockGraph,GraphSample
    from test_authority import drain
    cfg=KVMemConfig(block_tokens=32,select_budget=64,gen_budget=128,
        repository_blocks=16,sink_blocks=1,recent_blocks=0,
        action_attention=True,action_fraction=0.)
    archive=AsyncArchive(str(tmp_path),1*32*1*2*3,16)
    a=RequestAuthority('window',cfg,archive,1,1,2,resident_records=16,
        query_score_mode='overlap_16_8',query_window_cache_bytes=capacity)
    torch.manual_seed(12)
    raw=torch.randn(1,192,1,2).bfloat16()
    assert a.append(raw,torch.zeros_like(raw,dtype=torch.uint8),history_begin=0,local_rope_begin=0)
    drain(a)
    graph=a.block_graph=BlockGraph(('tenant','window'),32)
    graph.prepare(0,192,[GraphSample(160,(32,64,96),(.3,.5,.2),'prefill')]).commit()
    a.graph_options=NS(graph_mode='multi',graph_scoring_version='v2',graph_aggregation='sum',
        graph_alpha=.2,graph_locality_weight=.1,graph_attention_weight=.2,
        graph_cross_message_weight=.25,graph_trace=False,graph_native_propagation=False)
    return a,archive,graph


def test_native_picker_exact_graph_scores_keep_budget_and_no_neighbor_expansion(tmp_path):
    from vllm_kvmem.authority import query_score_frame,_Ready
    from vllm_kvmem.graph_scoring import score_graph
    from test_graph_lifecycle import selected_starts
    a,archive,graph=actual(tmp_path)
    q=torch.tensor([[[[2.,1.],[1.,-2.]]]])
    with a.index_snapshot() as (records,means),graph.snapshot() as snapshot:
        with query_score_frame(a.memory,records) as frame:
            seeds,_=score_authority_windows(a,records,q,means,frame.score_mask)
            with score_graph(seeds.tolist(),[r.history_begin for r in records],frame.transport_mask,
                snapshot,mode='multi',scoring_version='v2',alpha=.2,aggregation='sum',block_tokens=32) as output:
                slots=set(a.memory.preview_selection({r.handle.slot:s for r,s in zip(records,output.scores)},
                    (),action=False,prefill_scores=None))
                expected=[r.history_begin for r in records if r.handle.slot in slots]
    plan=a.select_for_queries(q)
    assert selected_starts(a,plan)==expected and plan.total_window_tokens==64
    assert 0 in expected and len(expected)==2  # no appended neighbor after picking
    assert a.graph_selection_report['seed_source']=='overlap_16_8'
    assert a.graph_selection_report['aggregation']=='sum'
    assert a.query_window_stats['cache_hit_blocks']==6
    a.memory.attach_completion(_Ready());assert a.memory.commit_selection()
    del plan
    assert a.close() and archive._query_window_cache_bytes==0 and archive.close()


@pytest.mark.parametrize('fault',['read','capacity','stale_graph'])
def test_failures_precede_native_commit_and_leave_no_partial_update(tmp_path,monkeypatch,fault):
    from vllm_kvmem.authority import _Ready
    a,archive,graph=actual(tmp_path,capacity=1024 if fault=='capacity' else 65536)
    before=a.memory.phase,graph.binding,graph.budget.used
    if fault=='read':
        def fail(*args):raise OSError('injected archive read')
        monkeypatch.setattr(a,'read_records',fail)
    if fault=='stale_graph':
        plan=a.select_for_queries(torch.ones(1,1,1,2))
        graph.invalidate();a.memory.attach_completion(_Ready())
        with pytest.raises(RuntimeError,match='Stale'):a.memory.commit_selection()
        del plan
    else:
        with pytest.raises((OSError,MemoryError)):
            a.select_for_queries(torch.ones(1,1,1,2))
        assert (a.memory.phase,graph.binding,graph.budget.used)==before
    assert a.close() and archive._query_window_cache_bytes==0 and archive.close()


def test_archive_failure_after_first_cache_batch_preserves_retryable_verified_prefix(monkeypatch):
    a, archive, records, reads, keys = synthetic(40*32)
    cache = a.query_window_cache = WindowMeanCache(archive, 1, 1, 1,
        dtype=torch.float32, max_bytes=200000)
    original = a.read_records
    calls = 0
    def fail_second(handles):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('second cache batch read failed')
        return original(handles)
    monkeypatch.setattr(a, 'read_records', fail_second)
    with cache.use(), pytest.raises(OSError, match='second cache batch'):
        cache.snapshot(a, records)
    assert cache.active == 0
    assert all(signature is not None for signature in cache.signatures[:32])
    assert all(signature is None for signature in cache.signatures[32:])
    monkeypatch.setattr(a, 'read_records', original)
    with cache.use():
        means, _, stats = cache.snapshot(a, records)
        assert stats['cache_hit_blocks'] == 32 and stats['cache_rebuilt_blocks'] == 8
        assert means[0, 31*4+3, 0, 0] == keys[:, 31*32+24:32*32+8].mean()
        assert means[0, 39*4+3, 0, 0] == keys[:, 39*32+24:].mean()
        del means
    assert len(reads) == 2  # Successful prefix read and unfinished suffix retry.
    cache.clear()
    assert archive._query_window_cache_bytes == 0


def test_nonfinite_mean_never_publishes_half_filled_row_and_retry_rebuilds_it(monkeypatch):
    a, archive, records, _, keys = synthetic(128)
    cache = a.query_window_cache = WindowMeanCache(archive, 1, 1, 1,
        dtype=torch.float32, max_bytes=20000)
    saved = keys[:, 88].clone()
    keys[:, 88] = torch.nan
    with cache.use(), pytest.raises(ValueError, match='Nonfinite original Key'):
        cache.snapshot(a, records)
    # The first failing row has already received its earlier windows; it still
    # lacks a valid signature, so none of that partial row can become a hit.
    assert cache.signatures[0] is not None and cache.signatures[1] is not None
    assert cache.signatures[2] is None
    keys[:, 88] = saved
    with cache.use():
        means, _, stats = cache.snapshot(a, records)
        assert stats['cache_hit_blocks'] == 2 and stats['cache_rebuilt_blocks'] == 2
        assert bool(torch.isfinite(means).all())
        assert means[0, 8, 0, 0] == keys[:, 64:80].mean()
        del means
    cache.clear()


def test_concurrent_owner_cannot_evict_a_snapshot_between_read_and_cpu_consumption():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    a, archive, records, _, _ = synthetic(64)
    b, _, other, _, _ = synthetic(64, archive, value=-2.)
    ca = a.query_window_cache = WindowMeanCache(archive, 1, 1, 1,
        dtype=torch.float32, max_bytes=4128)
    cb = b.query_window_cache = WindowMeanCache(archive, 1, 1, 1,
        dtype=torch.float32, max_bytes=4128)
    acquired, release = Event(), Event()
    def hold_snapshot():
        with ca.use():
            means, _, _ = ca.snapshot(a, records)
            acquired.set()
            assert release.wait(5)
            assert ca.bytes == 4128 and means[0, 3, 0, 0] == 31.5
            del means
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(hold_snapshot)
        try:
            assert acquired.wait(5)
            with cb.use(), pytest.raises(MemoryError, match='active scoring'):
                cb.snapshot(b, other)
            assert archive._query_window_cache_bytes == 4128 and cb.active == 0
        finally:
            release.set()
        future.result(5)
    with cb.use():
        means, _, stats = cb.snapshot(b, other)
        assert ca.bytes == 0 and stats['cache_rebuilt_blocks'] == 2
        assert means[0, 3, 0, 0] == -63.
        del means
    cb.clear()
    assert archive._query_window_cache_bytes == 0


def test_real_native_rewind_clears_derived_means_and_rebuilds_changed_branch_generations(tmp_path):
    from vllm_kvmem.authority import RequestAuthority
    from vllm_kvmem.block_graph import BlockGraph
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.storage import AsyncArchive
    from test_authority import drain
    from test_authority_rewind import complete_rewind
    from test_core import Event
    cfg = KVMemConfig(block_tokens=32, select_budget=64, gen_budget=128,
        repository_blocks=16, sink_blocks=1, recent_blocks=0)
    archive = AsyncArchive(str(tmp_path), 32*2*3, 16)
    a = RequestAuthority('rewind-window', cfg, archive, 1, 1, 2, resident_records=8,
        query_score_mode='overlap_16_8', query_window_cache_bytes=65536)
    graph = a.block_graph = BlockGraph(('tenant', 'rewind-window'), 32)
    prefix = torch.ones(1, 64, 1, 2, dtype=torch.bfloat16)
    assert a.append(prefix, torch.zeros_like(prefix, dtype=torch.uint8), history_begin=0, local_rope_begin=0)
    drain(a)
    graph.prepare(0, 64, ()).commit()
    checkpoint, graph_checkpoint = a.memory.checkpoint(), graph.snapshot()
    seen_suffix_generations = set()
    try:
        for value in (10., -10., 20.):
            suffix = torch.full_like(prefix, value)
            assert a.append(suffix, torch.zeros_like(suffix, dtype=torch.uint8), history_begin=64, local_rope_begin=64)
            drain(a)
            graph.prepare(64, 128, ()).commit()
            suffix_handles = [r.handle for r in a.records.values() if r.history_begin >= 64]
            identities = {(handle.slot, handle.generation) for handle in suffix_handles}
            assert not (identities & seen_suffix_generations)
            seen_suffix_generations.update(identities)
            a.select_for_queries(torch.ones(1, 1, 1, 2), observe_only=True)
            assert a.query_window_stats['cache_rebuilt_blocks'] == 4
            torch.testing.assert_close(a.query_window_cache.tensor[0, 1, 3],
                torch.full((1, 2), (1+value)/2))
            old_graph_binding = graph.binding
            assert a.begin_rewind(checkpoint)
            assert a.query_window_cache.tensor is None and archive._query_window_cache_bytes == 0
            fence = Event()
            a.attach_rewind_completion(fence)
            assert not a.poll_rewind() and a.history_end == 128
            fence.done = True
            complete_rewind(a)
            assert a.history_end == 64 and len(a.records) == 2
            assert all(not a.memory.repository.pin(handle) for handle in suffix_handles)
            graph.restore(graph_checkpoint)
            assert graph.accepted_through == 64
            with pytest.raises(RuntimeError, match='Stale'):
                graph.validate_binding(old_graph_binding)
    finally:
        graph_checkpoint.close()
        del checkpoint
        assert a.close() and archive._query_window_cache_bytes == 0 and archive.close()
