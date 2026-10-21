"""CPU oracles for fixed opaque metadata leases, dispatch and failure fences."""
from types import SimpleNamespace as NS
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from vllm_kvmem import split_assembly as split
from vllm_kvmem import window_assembly as window
from test_assembly_read_prefetch import cuda, DeviceBytes, FakeEvent, make_split


def spec(cuda, rows=2, tag=1):
    tensors = tuple(DeviceBytes(torch.empty(16,dtype=torch.uint8),cuda.operations) for _ in range(rows*5))
    return NS(rows=tuple(tuple(tag*100+i*12+j for j in range(12)) for i in range(rows)),
              signature=(tag,),owners=tensors,geometry=NS(heads=1,dim=2,half=1,page=4))


def test_fixed_table_waits_for_full_consumer_without_blocking_or_reallocation(cuda):
    pool = window._AssemblyStaging(128,torch.device('cuda:0'))
    first,second = NS(_restore_lease=NS(generation=4)),NS(_restore_lease=NS(generation=5))
    a,b = NS(memory=NS(request_id='a')),NS(memory=NS(request_id='b'))
    p,q = object(),object()
    original,later = spec(cuda),spec(cuda,tag=2)
    pool.claim(first)
    table,reason = pool.prepare_fused_table(first,a,p,original)
    assert reason is None and pool.fused_active
    assert pool.fused_binding[:4] == (id(a),'a',id(p),4)
    assert a in pool.fused_owners and p in pool.fused_owners
    assert all(any(item is t for item in pool.fused_owners) for t in original.owners)
    pointer = pool.fused_host.data_ptr()
    old_bytes = pool.fused_host.clone()
    assert pool.fused_capacity() == (192,192)
    completion = FakeEvent(False)
    pool.finish_fused_table(first,completion)
    pool.release(first)
    assert not pool.close()
    pool.claim(second)
    assert pool.prepare_fused_table(second,b,q,later) == (None,'busy')
    assert torch.equal(pool.fused_host,old_bytes) and pool.fused_host.data_ptr() == pointer
    assert a in pool.fused_owners and not any(op[0].endswith('sync') for op in cuda.operations)
    pool.release(second)
    completion.complete = True
    pool.claim(second)
    updated,reason = pool.prepare_fused_table(second,b,q,later)
    assert reason is None and updated is table and pool.fused_host.data_ptr() == pointer
    assert pool.fused_binding[:4] == (id(b),'b',id(q),5)
    assert all(item is not a and item is not p for item in pool.fused_owners)
    assert torch.equal(pool.fused_host,torch.tensor(later.rows))
    pool.finish_fused_table(second,FakeEvent(True))
    pool.release(second)
    assert pool.close() and not pool.fused_owners and pool.fused_host is None


def test_table_capacity_change_declines_without_growing_staging(cuda):
    pool = window._AssemblyStaging(128,torch.device('cuda:0'))
    owner,authority,plan = NS(),NS(memory=NS(request_id='a')),object()
    pool.claim(owner)
    table,_ = pool.prepare_fused_table(owner,authority,plan,spec(cuda))
    pool.finish_fused_table(owner,FakeEvent(True))
    assert pool.prepare_fused_table(owner,authority,plan,spec(cuda,rows=3)) == (None,'capacity')
    assert pool.fused_device is table and pool.fused_capacity() == (192,192)
    pool.release(owner)
    assert pool.close()


@pytest.mark.parametrize('double_failure',[False,True])
def test_metadata_upload_failure_keeps_sources_until_proven_failure_fence(cuda,double_failure):
    pool = window._AssemblyStaging(128,torch.device('cuda:0'))
    owner,authority,plan = NS(),NS(memory=NS(request_id='a')),object()
    pool.claim(owner)
    # Preallocate the fixed table through the same successful lifecycle.
    table,_ = pool.prepare_fused_table(owner,authority,plan,spec(cuda))
    pool.finish_fused_table(owner,FakeEvent(True))
    table.fail_copy = True
    if double_failure:
        def fail_fence(): raise OSError('metadata fence failed')
        cuda.current.on_sync = fail_fence
    with pytest.raises(OSError,match='enqueue'):
        pool.prepare_fused_table(owner,authority,plan,spec(cuda,tag=2))
    assert pool.fused_active and authority in pool.fused_owners
    failed = FakeEvent(False)
    pool.release(owner,failed)
    assert not pool.close() and pool.fused_owners
    failed.complete = True
    if double_failure:
        assert pool.fused_poisoned and not pool.close()
        with pytest.raises(RuntimeError,match='unfenced'):
            pool.claim(NS())
    else:
        assert pool.close() and not pool.fused_owners


def test_missing_completion_never_recycles_opaque_pointers(cuda):
    pool = window._AssemblyStaging(128,torch.device('cuda:0'))
    owner,authority,plan = NS(),NS(memory=NS(request_id='a')),object()
    pool.claim(owner)
    pool.prepare_fused_table(owner,authority,plan,spec(cuda))
    pool.release(owner)
    assert pool.fused_poisoned and not pool.close()
    with pytest.raises(RuntimeError,match='unfenced'):
        pool.claim(NS())


@pytest.mark.parametrize('failure',[False,True])
def test_qualified_chunk_uses_one_launch_and_failure_never_falls_back(tmp_path,cuda,monkeypatch,failure):
    assembler,proxy,plan,authority,archive,_ = make_split(tmp_path,cuda,monkeypatch)
    assembler.fused_layers = True
    monkeypatch.setattr(split,'prepare_fused_layer_rows',lambda _: (spec(cuda),None))
    monkeypatch.setattr(split,'assemble_split_layer',lambda *a,**kw: pytest.fail('unexpected old layer launch'))
    calls = []
    def fused(*args,**kwargs):
        calls.append(kwargs)
        if failure: raise OSError('fused launch failed')
    monkeypatch.setattr(split,'assemble_split_layers',fused)
    if failure:
        with pytest.raises(OSError,match='fused launch failed'):
            assembler.assemble(proxy,plan)
        assert len(calls) == 1 and any(op[0]=='fail_transaction' for op in cuda.operations)
        assert assembler.read_prefetch_stats['assembly_prefetch_abandoned'] == 1
        assert not assembler.transfers.preparing and not assembler.staging.fused_active
    else:
        result = assembler.assemble(proxy,plan)
        assert len(calls) == result.read_stats['assembly_fused_chunks'] == 3
        assert result.read_stats['assembly_fused_layer_launches'] == 6
        assert result.read_stats['assembly_fused_fallback_chunks'] == 0
        assert result.read_stats['assembly_fused_metadata_bytes'] == 384
        assert result.device_temp_bytes == result.resident_v_reused_tokens*2*1*2+192
        assert assembler.close() and not assembler.staging.fused_owners
    assert authority.close() and archive.close()


@pytest.mark.parametrize('reason',['alias','geometry','strides'])
def test_whole_bundle_fallback_precedes_any_layer_writes(tmp_path,cuda,monkeypatch,reason):
    assembler,proxy,plan,authority,archive,_ = make_split(tmp_path,cuda,monkeypatch)
    assembler.fused_layers = True
    calls = []
    monkeypatch.setattr(split,'prepare_fused_layer_rows',lambda _: (None,reason))
    monkeypatch.setattr(split,'assemble_split_layer',lambda *a,**kw: calls.append(kw['layer_index']))
    monkeypatch.setattr(split,'assemble_split_layers',lambda *a,**kw: pytest.fail('ineligible bundle fused'))
    result = assembler.assemble(proxy,plan)
    assert calls == [0,1]*3
    assert result.read_stats['assembly_fused_chunks'] == 0
    assert result.read_stats['assembly_fused_fallback:'+reason] == 3
    assert result.read_stats['assembly_fused_metadata_bytes'] == 0
    assert assembler.staging.fused_host is None
    assert authority.close() and archive.close()


def test_partial_plan_declines_before_metadata_upload(tmp_path,cuda,monkeypatch):
    assembler,proxy,plan,authority,archive,_ = make_split(tmp_path,cuda,monkeypatch)
    assembler.fused_layers = True
    partial = [NS(n_tokens=4),NS(n_tokens=3)]
    monkeypatch.setattr(split,'prepare_fused_layer_rows',lambda _: pytest.fail('partial metadata table'))
    assert assembler._prepare_fused_layers(proxy,plan,partial) == (None,'partial')
    assert assembler.staging.fused_host is None
    assert authority.close() and archive.close()


def test_failed_record_cannot_release_an_unfenced_assembly(tmp_path,cuda,monkeypatch):
    assembler,proxy,plan,authority,archive,_ = make_split(tmp_path,cuda,monkeypatch)
    assembler.fused_layers = True
    monkeypatch.setattr(split,'prepare_fused_layer_rows',lambda _: (spec(cuda),None))
    def launch(*args,**kwargs):
        class BadEvent(FakeEvent):
            def record(self,stream):
                raise OSError('failure event record')
        monkeypatch.setattr(torch.cuda,'Event',lambda: BadEvent(True))
        raise ValueError('original kernel failure')
    monkeypatch.setattr(split,'assemble_split_layers',launch)
    with pytest.raises(ValueError,match='original kernel failure') as caught:
        assembler.assemble(proxy,plan)
    assert any('failure event record' in note for note in caught.value.__notes__)
    failed = next(op[1] for op in cuda.operations if op[0] == 'fail_transaction')
    assert not failed.query() and not assembler._completion.query()
    assert failed.owners == (assembler,plan,proxy)
    assert assembler.staging.fused_poisoned and assembler.staging.fused_owners
    assert not assembler.close() and not assembler.staging.close()
    with pytest.raises(RuntimeError,match='unfenced'):
        assembler.staging.claim(NS())
    # The proxy owns the failed transaction; the real CPU archive fixture did
    # not submit GPU work and can be closed independently.
    assert authority.close() and archive.close()


def test_poll_holds_lock_while_query_releases_python_execution(cuda):
    pool = window._AssemblyStaging(128,torch.device('cuda:0'))
    first,second,authority,plan = NS(),NS(),NS(memory=NS(request_id='a')),object()
    pool.claim(first)
    pool.prepare_fused_table(first,authority,plan,spec(cuda))
    entered,release = Event(),Event()
    class BlockingEvent(FakeEvent):
        def query(self):
            entered.set()
            assert release.wait(3)
            return True
    pool.finish_fused_table(first,BlockingEvent(True))
    pool.release(first)
    def next_owner():
        pool.claim(second)
        return pool.prepare_fused_table(second,authority,plan,spec(cuda,tag=2))
    with ThreadPoolExecutor(max_workers=2) as executor:
        polling = executor.submit(pool.poll_fused_table)
        assert entered.wait(2)
        preparing = executor.submit(next_owner)
        try:
            with pytest.raises(TimeoutError):
                preparing.result(.02)
        finally:
            release.set()
        assert polling.result(2)
        assert preparing.result(2)[1] is None
    assert pool.fused_active and pool.fused_binding[4] == (2,)
    pool.finish_fused_table(second,FakeEvent(True))
    pool.release(second)
    assert pool.close()
