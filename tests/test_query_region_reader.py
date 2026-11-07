"""Lazy Query sources preserve bytes, admission and generation lifetimes."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import record_key
from vllm_kvmem.memory import Phase
from vllm_kvmem.query_region_reader import QueryRegionReader
from test_native_memory_storage import eventually
from test_paired_archive_reads import archive_fixture


def fixture(tmp_path, backend='cpu'):
    archive, repo, leases, values = archive_fixture(tmp_path, backend)
    records = [NS(handle=lease.handle, history_begin=i*4, count=4, archived=True)
               for i, lease in enumerate(leases)]
    pins, polls, generation = [], [], [7]
    def admitted(handle):
        pins.append(record_key(handle))
        return repo.pin(handle), None
    authority = NS(archive=archive, shape=(2,4,2,2), history_end=12,
        poll=lambda: polls.append(1), records={record_key(r.handle):r for r in records},
        memory=NS(phase=Phase.READY, request_id='owner', history_end=12, admitted_read=admitted))
    return authority, records, pins, polls, generation, repo, leases, values


def reader_for(authority, generation):
    return QueryRegionReader(authority, request_generation=generation[0],
                             generation_getter=lambda: generation[0])


def test_sources_are_lazy_unique_and_reused_for_independent_layer_regions(tmp_path, monkeypatch):
    a, records, pins, polls, generation, repo, leases, values = fixture(tmp_path)
    source_queries = []
    borrow = a.archive._region_sources
    def observed(keys):
        source_queries.append(tuple(keys))
        return borrow(keys)
    monkeypatch.setattr(a.archive, '_region_sources', observed)
    monkeypatch.setattr(a.archive.executor, 'submit', lambda *a, **k: pytest.fail('Unexpected I/O handoff'))
    reader = reader_for(a, generation)
    before = repo.stats['leases']
    assert not pins and not source_queries and reader.source_blocks == 0
    assert not a.archive.close()
    try:
        for indices, regions, new in [([2,0],((3,11),(47,9)),2),
                                      ([1,2],((0,16),(32,16)),1),
                                      ([2,2,0],((7,13),),0)]:
            target = torch.empty(len(indices)*sum(n for _,n in regions), dtype=torch.uint8)
            kwargs = {'region':regions[0]} if len(regions)==1 else {'regions':regions}
            data, stats = reader.read(a, [records[i].handle for i in indices],
                                      destination=target, **kwargs)
            assert data.data_ptr() == target.data_ptr()
            assert data.numpy().tobytes() == b''.join(values[i][o:o+n] for o,n in regions for i in indices)
            assert stats['query_lazy_region_read'] and stats['prepared_cpu_region_read']
            assert stats['inline_cpu_region_read'] and stats['cache_hit_bytes'] == target.numel()
            assert stats['logical_bytes'] == target.numel() and stats['bytes'] == 0
            assert stats['query_reader_new_blocks'] == new
            assert stats['query_reader_reused_blocks'] == len(indices)-new
            assert a.archive.io_usage()['archive_io_pending_bytes'] == 0
        assert len(pins) == reader.source_blocks == 3
        assert [len(keys) for keys in source_queries] == [2,1]
        assert polls == [1] and reader.read_chunks == 3 and reader.reused_blocks == 4
        assert repo.stats['leases'] == before+3
        # Equivalent handle wrappers must reuse the same generation too.
        reader.read(a, [leases[0].handle], destination=torch.empty(8,dtype=torch.uint8), region=(0,8))
        assert len(pins)==3 and repo.stats['leases']==before+3
    finally:
        assert reader.close() and reader.close()
        assert repo.stats['leases'] == before
        assert a.archive._prepared_region_readers == 0 and a.archive.close()


@pytest.mark.parametrize('mutation', ['generation','history','memory_history','phase','namespace',
    'memory_owner','archive_owner','record_table','record_identity','record_handle','record_begin',
    'record_count','unpublished','shape','foreign_call'])
def test_changed_query_scope_or_record_fails_closed_before_destination_write(tmp_path, mutation):
    a, records, pins, polls, generation, repo, leases, values = fixture(tmp_path)
    archive = a.archive
    reader = reader_for(a,generation)
    reader.read(a,[records[0].handle],destination=torch.empty(8,dtype=torch.uint8),region=(0,8))
    handle = records[0].handle
    caller = a
    if mutation=='generation': generation[0] += 1
    elif mutation=='history': a.history_end += 4
    elif mutation=='memory_history': a.memory.history_end += 4
    elif mutation=='phase': a.memory.phase = Phase.APPENDING
    elif mutation=='namespace': a.memory.request_id = 'recycled'
    elif mutation=='memory_owner': a.memory = NS(**vars(a.memory))
    elif mutation=='archive_owner': a.archive = object()
    elif mutation=='record_table': a.records = dict(a.records)
    elif mutation=='record_identity': a.records[record_key(handle)] = NS(**vars(records[0]))
    elif mutation=='record_handle': records[0].handle = leases[0].handle
    elif mutation=='record_begin': records[0].history_begin = 4
    elif mutation=='record_count': records[0].count = 3
    elif mutation=='unpublished': records[0].archived = False
    elif mutation=='shape': a.shape = (3,4,2,2)
    elif mutation=='foreign_call': caller = NS(**vars(a))
    target = torch.full((8,),122,dtype=torch.uint8)
    with pytest.raises(ValueError): reader.read(caller,[handle],destination=target,region=(0,8))
    assert target.tolist()==[122]*8 and reader.closed
    assert archive.io_usage()['archive_io_pending_bytes']==0
    assert archive._prepared_region_readers==0 and archive.close()


@pytest.mark.parametrize('backend',['disk','cache','native'])
def test_unsupported_archive_requires_explicit_failure(tmp_path, backend):
    a,records,pins,polls,generation,repo,leases,values = fixture(tmp_path,backend)
    try:
        with pytest.raises(ValueError,match='immutable CPU bytes'): reader_for(a,generation)
        assert not pins and getattr(a.archive,'_prepared_region_readers',0)==0
    finally:
        for lease in leases: lease.reset()
        eventually(a.archive.close)


@pytest.mark.parametrize('gc', ['repository','scope','worker'])
def test_independent_retirement_is_not_bypassed(tmp_path,gc):
    a,records,pins,polls,generation,repo,leases,values = fixture(tmp_path)
    if gc=='repository':a.gc=object()
    elif gc=='scope':a.memory._history_scope=object()
    else:a.archive._history_worker=object()
    try:
        with pytest.raises(ValueError,match='without GC'):reader_for(a,generation)
        assert not pins and getattr(a.archive,'_prepared_region_readers',0)==0
    finally:
        a.archive._history_worker=None
        assert a.archive.close()


@pytest.mark.parametrize('fault', ['partial','unpublished','unknown','bad_region','both_regions',
    'empty_regions','bad_count','destination','dtype','noncontiguous','paired_native'])
def test_invalid_extent_is_rejected_without_admitting_or_writing(tmp_path,fault):
    a,records,pins,polls,generation,repo,leases,values = fixture(tmp_path)
    reader=reader_for(a,generation)
    target=torch.full((16,),122,dtype=torch.uint8)
    destination=target
    handles=[records[0].handle]
    kwargs={'region':(0,16)}
    orphan=None
    if fault=='partial':records[0].count=3
    elif fault=='unpublished':records[0].archived=False
    elif fault=='unknown':orphan=repo.reserve(12,4).commit();handles=[orphan.handle]
    elif fault=='bad_region':kwargs={'region':(60,16)}
    elif fault=='both_regions':kwargs['regions']=((0,8),(32,8))
    elif fault=='empty_regions':kwargs={'regions':()}
    elif fault=='bad_count':kwargs={'regions':((0,16),)}
    elif fault=='destination':destination=target[:8]
    elif fault=='dtype':destination=target.to(torch.int16)
    elif fault=='noncontiguous':destination=target[::2]
    elif fault=='paired_native':kwargs['paired_native']=True
    with pytest.raises(ValueError):reader.read(a,handles,destination=destination,**kwargs)
    assert target.tolist()==[122]*16 and reader.closed and not pins
    assert a.archive.io_usage()['archive_io_pending_bytes']==0 and a.archive.close()


@pytest.mark.parametrize('failure',['raise','stale','payload','foreign','unpublished'])
def test_later_source_preparation_failure_releases_earlier_owned_leases(tmp_path,failure):
    a,records,pins,polls,generation,repo,leases,values = fixture(tmp_path)
    before=repo.stats['leases']
    admitted=a.memory.admitted_read
    calls=[]
    def broken(handle):
        calls.append(handle)
        if len(calls)==1:return admitted(handle)
        if failure=='raise':raise MemoryError('admission allocation')
        if failure=='stale':return None,None
        if failure=='payload':return repo.pin(handle),object()
        if failure=='foreign':return repo.pin(records[0].handle),None
        return admitted(handle)
    a.memory.admitted_read=broken
    if failure=='unpublished':
        a.archive._entries[('owner',*record_key(records[1].handle))].ready=False
    reader=reader_for(a,generation)
    target=torch.full((16,),122,dtype=torch.uint8)
    with pytest.raises((ValueError,KeyError,MemoryError)):
        reader.read(a,[r.handle for r in records[:2]],destination=target,region=(0,8))
    assert target.tolist()==[122]*16 and reader.closed and reader.source_blocks==0
    assert repo.stats['leases']==before
    assert a.archive._prepared_region_readers==0 and a.archive.close()


@pytest.mark.parametrize('failure',[False,True])
def test_active_producer_retains_sources_admission_and_releases_on_failure(tmp_path,monkeypatch,failure):
    a,records,pins,polls,generation,repo,leases,values = fixture(tmp_path)
    archive=a.archive
    reader=reader_for(a,generation)
    entered,release=Event(),Event()
    gather=archive._gather_region
    def blocked(*args):
        if args[2]==32:
            entered.set()
            assert release.wait(5)
            if failure:raise OSError('second field failed')
        return gather(*args)
    monkeypatch.setattr(archive,'_gather_region',blocked)
    target=torch.empty(32,dtype=torch.uint8)
    handle=records[0].handle
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(reader.read,a,[handle],destination=target,regions=((0,16),(32,16)))
        try:
            assert entered.wait(5)
            assert reader.active==1 and not reader.close() and not archive.close()
            assert archive.io_usage()['archive_io_pending_bytes']==32
            assert repo.retire(handle)
            leases[0].reset()
            assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r))==_native.ReclaimResult.READERS
            # Another owner's publication needs the same archive lock.
            extra=repo.reserve(12,4).commit()
            archive.write('other',[extra],b'x'*64).result(5)
        finally:
            release.set()
        if failure:
            with pytest.raises(OSError,match='second field failed'):future.result(5)
            assert reader.closed
        else:
            data,_=future.result(5)
            assert data.numpy().tobytes()==values[0][:16]+values[0][32:48]
            assert not reader.closed and reader.close()
    assert reader.active==0 and archive.io_usage()['archive_io_pending_bytes']==0
    assert repo.reclaim(handle,lambda r:archive.reclaim('owner',r))==_native.ReclaimResult.RECLAIMED
    extra.reset()
    assert archive.close()


def test_scope_retired_during_copy_does_not_return_a_stale_success(tmp_path,monkeypatch):
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    reader=reader_for(a,generation)
    gather=a.archive._gather_region
    def mutate(*args):
        result=gather(*args)
        generation[0]+=1
        return result
    monkeypatch.setattr(a.archive,'_gather_region',mutate)
    with pytest.raises(ValueError,match='generation'):
        reader.read(a,[records[0].handle],destination=torch.empty(8,dtype=torch.uint8),region=(0,8))
    assert reader.closed and a.archive.io_usage()['archive_io_pending_bytes']==0
    assert a.archive.close()


@pytest.mark.parametrize('timeout',[False,True])
def test_admission_retry_preserves_sources_and_timeout_releases_them(tmp_path,monkeypatch,timeout):
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    reader=reader_for(a,generation)
    admit=a.archive._admit
    attempts=[]
    def defer_once(size):
        attempts.append(size)
        return False if len(attempts)==1 else admit(size)
    waits=[]
    monkeypatch.setattr(a.archive,'_admit',defer_once)
    monkeypatch.setattr(a.archive,'wait_for_idle',lambda remaining: waits.append(remaining) or not timeout)
    target=torch.full((8,),122,dtype=torch.uint8)
    if timeout:
        with pytest.raises(TimeoutError):reader.read(a,[records[0].handle],destination=target,region=(0,8))
        assert target.tolist()==[122]*8 and reader.closed
    else:
        data,stats=reader.read(a,[records[0].handle],destination=target,region=(0,8))
        assert data.numpy().tobytes()==values[0][:8]
        assert stats['admission_wait_seconds']>=0 and len(attempts)==2
    assert len(waits)==1 and len(pins)==1
    assert a.archive.io_usage()['archive_io_pending_bytes']==0
    assert reader.close() and a.archive.close()


def test_poisoned_gather_pool_is_not_bypassed(tmp_path):
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    reader=reader_for(a,generation)
    a.archive._gather_unusable='earlier failure'
    target=torch.full((8,),122,dtype=torch.uint8)
    with pytest.raises(RuntimeError,match='pool unavailable'):
        reader.read(a,[records[0].handle],destination=target,region=(0,8))
    assert target.tolist()==[122]*8 and reader.closed
    assert a.archive.io_usage()['archive_io_pending_bytes']==0 and a.archive.close()


def test_failed_context_entry_releases_the_empty_archive_guard(tmp_path):
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    reader=reader_for(a,generation)
    generation[0]+=1
    with pytest.raises(ValueError):
        with reader:pytest.fail('stale context entered')
    assert reader.closed and a.archive._prepared_region_readers==0 and a.archive.close()


@pytest.mark.parametrize('mutation',['native_generation','archive_namespace','runtime_owner'])
def test_warm_import_uses_archive_namespace_and_native_request_generation_separately(tmp_path,mutation):
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    query=object.__new__(LayerwiseQuery)
    native_id='native-warm-prefix-request'
    query.runner=NS(_kvmem_request_generations={native_id:generation[0]},
        _kvmem_capture_runtime=NS(authorities={native_id:a}))
    state=dict(authority=a,request_id=native_id)
    reader=query._query_reader(state)
    assert reader.namespace=='owner' and reader.namespace!=native_id
    assert query._query_reader(state) is reader
    actual,_=reader.read(a,[records[0].handle],destination=torch.empty(8,dtype=torch.uint8),region=(0,8))
    assert actual.numpy().tobytes()==values[0][:8]
    if mutation=='native_generation':
        query.runner._kvmem_request_generations[native_id]+=1
    elif mutation=='archive_namespace':
        a.memory.request_id='different-archive-owner'
    else:
        query.runner._kvmem_capture_runtime.authorities[native_id]=NS(**vars(a))
    target=torch.full((8,),122,dtype=torch.uint8)
    with pytest.raises(ValueError):
        reader.read(a,[records[0].handle],destination=target,region=(0,8))
    assert target.tolist()==[122]*8 and reader.closed
    query._close_query_resources(state)
    assert 'query_region_reader' not in state
    assert a.archive._prepared_region_readers==0 and a.archive.close()


@pytest.mark.parametrize('missing',[False,True])
def test_controller_rejects_unbound_authority_before_creating_source_reader(tmp_path,missing):
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    a,records,pins,polls,generation,repo,leases,values=fixture(tmp_path)
    query=object.__new__(LayerwiseQuery)
    native_id='native-warm-prefix-request'
    query.runner=NS(_kvmem_request_generations={native_id:generation[0]},
        _kvmem_capture_runtime=NS(authorities={} if missing else {native_id:NS(**vars(a))}))
    state=dict(authority=a,request_id=native_id)
    with pytest.raises(ValueError,match='native authority binding'):
        query._query_reader(state)
    assert 'query_region_reader' not in state and not pins and not polls
    assert getattr(a.archive,'_prepared_region_readers',0)==0 and a.archive.close()
