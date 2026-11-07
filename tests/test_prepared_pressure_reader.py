"""Bounded per-plan source reuse without changing bytes, order or ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import record_key
from vllm_kvmem.memory import Phase
from vllm_kvmem.pressure_reader import prepare_pressure_reader
from test_paired_archive_reads import archive_fixture
from test_native_memory_storage import eventually


def authority_fixture(tmp_path, backend='cpu'):
    archive,repo,leases,values = archive_fixture(tmp_path, backend)
    records=[NS(handle=l.handle,history_begin=i*4,count=4,archived=True) for i,l in enumerate(leases)]
    plan=object();pins=[]
    def admitted(handle):
        pins.append(record_key(handle))
        return repo.pin(handle),None
    authority=NS(archive=archive,shape=(2,4,2,2),history_end=12,poll=lambda:None,
        records={record_key(r.handle):r for r in records},
        memory=NS(phase=Phase.ASSEMBLING,_plan=plan,request_id='owner',admitted_read=admitted))
    return authority,plan,records,pins,repo,leases,values


def test_each_complete_source_admitted_once_across_layers_and_copy_order_is_unchanged(tmp_path,monkeypatch):
    a,plan,records,pins,repo,leases,values=authority_fixture(tmp_path)
    layers=[[records[2],records[0]],[records[1],records[2]]]
    sources=[];original=a.archive._region_sources
    def observed(keys):sources.append(tuple(keys));return original(keys)
    monkeypatch.setattr(a.archive,'_region_sources',observed)
    with prepare_pressure_reader(a,plan,layers) as reader:
        assert len(pins)==reader.source_blocks==3 and len(sources)==1
        # Further lookups and lease cloning must not be needed by any layer.
        monkeypatch.setattr(a.archive,'_region_sources',lambda *a:pytest.fail('reborrowed sources'))
        a.memory.admitted_read=lambda *a:pytest.fail('repinned a prepared source')
        for group in (layers[0],layers[1],[records[2],records[2]]):
            out=torch.empty(len(group)*20,dtype=torch.uint8)
            data,stats=reader.read(group,((3,11),(47,9)),destination=out)
            order=[r.history_begin//4 for r in group]
            assert data.data_ptr()==out.data_ptr()
            assert data.numpy().tobytes()==b''.join(values[i][o:o+n] for o,n in ((3,11),(47,9)) for i in order)
            assert stats['prepared_cpu_region_read'] and stats['logical_bytes']==len(group)*20
            assert a.archive.io_usage()['archive_io_pending_bytes']==0
        assert reader.read_chunks==3 and not a.archive.close()
    assert a.archive._prepared_region_readers==0 and a.archive.close()


@pytest.mark.parametrize('change',['plan','namespace','prefix','phase','handle','record','position','partial'])
def test_stale_plan_owner_or_source_is_rejected_before_writing(tmp_path,change):
    a,plan,records,pins,repo,leases,values=authority_fixture(tmp_path)
    with prepare_pressure_reader(a,plan,[records]) as reader:
        group=[records[0]];out=torch.full((20,),73,dtype=torch.uint8)
        if change=='plan':a.memory._plan=object()
        if change=='namespace':a.memory.request_id='other'
        if change=='prefix':a.history_end+=4
        if change=='phase':a.memory.phase=Phase.READY
        if change=='handle':records[0].handle=records[1].handle
        if change=='record':group=[NS(**records[0].__dict__)]
        if change=='position':records[0].history_begin+=4
        if change=='partial':records[0].count=3
        with pytest.raises(ValueError):reader.read(group,((3,11),(47,9)),destination=out)
        assert torch.equal(out,torch.full_like(out,73))
    assert a.archive.close()


def test_partial_records_keep_the_original_path_and_native_payload_backend_is_not_broadened(tmp_path):
    a,plan,records,pins,repo,leases,values=authority_fixture(tmp_path)
    records[1].count=3
    records[2].archived=False
    with prepare_pressure_reader(a,plan,[records]) as reader:
        assert reader.source_blocks==len(pins)==1
    assert a.archive.close()
    native,*_=authority_fixture(tmp_path/'native','native')
    assert prepare_pressure_reader(native,native.memory._plan,[list(native.records.values())]) is None
    eventually(native.archive.close)


@pytest.mark.parametrize('failure',[False,True])
def test_source_retirement_copy_failure_and_close_preserve_original_admission(tmp_path,monkeypatch,failure):
    archive,repo,leases,values=archive_fixture(tmp_path,'cpu')
    reader=archive.prepare_region_reader('owner',leases)
    handle=leases[0].handle;entered,release=Event(),Event();gather=archive._gather_region
    def blocked(*args):
        if args[2]==32:
            entered.set();assert release.wait(5)
            if failure:raise OSError('gather failed')
        return gather(*args)
    monkeypatch.setattr(archive,'_gather_region',blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        out=bytearray(32)
        future=pool.submit(reader.read,[0],((0,16),(32,16)),destination=out)
        try:
            assert entered.wait(5)
            assert not reader.close() and not archive.close()
            assert archive.io_usage()['archive_io_pending_bytes']==32
            archive.max_pending=1
            assert reader.read([1],((0,16),(32,16)),destination=bytearray(32)) is None
            assert repo.retire(handle);leases[0].reset()
            assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r))==_native.ReclaimResult.READERS
        finally:release.set()
        if failure:
            with pytest.raises(OSError,match='gather failed'):future.result(5)
        else:assert bytes(future.result(5)[0])==values[0][:16]+values[0][32:48]
    assert archive.io_usage()['archive_io_pending_bytes']==0 and reader.close() and reader.close()
    assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r))==_native.ReclaimResult.RECLAIMED
    assert archive.close()


@pytest.mark.parametrize('fault',['index','region','destination','closed'])
def test_prepared_copy_validates_all_extents_before_writing(tmp_path,fault):
    archive,repo,leases,_=archive_fixture(tmp_path,'cpu')
    reader=archive.prepare_region_reader('owner',leases)
    out=bytearray(b'z'*32);indices=[0];regions=((0,16),(32,16))
    if fault=='index':indices=[3]
    if fault=='region':regions=((0,16),(60,16))
    if fault=='destination':out.append(122)
    if fault=='closed':assert reader.close()
    with pytest.raises((ValueError,RuntimeError)):reader.read(indices,regions,destination=out)
    assert out==b'z'*len(out) and archive.io_usage()['archive_io_pending_bytes']==0
    assert reader.close() and archive.close()


def test_foreign_unpublished_or_duplicate_generations_fail_without_leaking_archive_lifetime(tmp_path):
    a,plan,records,pins,repo,leases,_=authority_fixture(tmp_path)
    with pytest.raises(KeyError):a.archive.prepare_region_reader('foreign',leases)
    with pytest.raises(ValueError):a.archive.prepare_region_reader('owner',[leases[0],leases[0]])
    orphan=repo.reserve(12,4).commit()
    with pytest.raises(KeyError):a.archive.prepare_region_reader('owner',[orphan])
    records[0].handle=orphan.handle
    with pytest.raises(ValueError):prepare_pressure_reader(a,plan,[records])
    assert a.archive.close()


def test_failed_context_entry_releases_prepared_archive_lifetime(tmp_path):
    a,plan,records,pins,repo,leases,_=authority_fixture(tmp_path)
    reader=prepare_pressure_reader(a,plan,[records])
    a.memory._plan=object()
    with pytest.raises(ValueError):
        with reader:pytest.fail('Invalid context entered')
    assert reader.closed and a.archive._prepared_region_readers==0 and a.archive.close()


def test_later_publication_keeps_existing_read_path_instead_of_using_absent_prepared_sources(tmp_path):
    a,plan,records,pins,repo,leases,_=authority_fixture(tmp_path)
    records[1].archived=False
    with prepare_pressure_reader(a,plan,[records]) as reader:
        records[1].archived=True
        assert reader.contains([records[0]])
        assert not reader.contains([records[1]])
        assert reader.source_blocks==2
    assert a.archive.close()


def test_reader_preparation_failure_releases_already_cloned_sources(tmp_path):
    archive,repo,leases,_=archive_fixture(tmp_path,'cpu')
    cloned=[]
    class Borrowed:
        def __init__(self,lease,fail=False):self.lease,self.fail=lease,fail
        def __bool__(self):return bool(self.lease)
        @property
        def handle(self):return self.lease.handle
        def clone(self):
            if self.fail:raise MemoryError('clone allocation')
            lease=self.lease.clone();cloned.append(lease);return lease
    with pytest.raises(MemoryError,match='clone allocation'):
        archive.prepare_region_reader('owner',[Borrowed(leases[0]),Borrowed(leases[1],True)])
    assert len(cloned)==1 and not cloned[0]
    assert getattr(archive,'_prepared_region_readers',0)==0 and archive.close()


@pytest.mark.parametrize('backend',['legacy_gc','history_scope'])
def test_prepared_pressure_path_keeps_all_gc_admission_on_existing_backend(tmp_path,backend):
    a,plan,records,pins,repo,leases,_=authority_fixture(tmp_path)
    if backend=='legacy_gc':a.gc=object()
    else:a.memory._history_scope=object()
    assert prepare_pressure_reader(a,plan,[records]) is None
    assert not pins and a.archive.close()
