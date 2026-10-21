import hashlib
import sys
from pathlib import Path
from threading import Event, Lock
import weakref

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import cpu_private_record_pack as candidate
from test_cpu_direct_record_seal import repository, reclaim_all
from vllm_kvmem import _native
from vllm_kvmem.memory_storage import MemoryArchive


def test_private_bytes_are_unobservable_until_all_ranges_finish():
    inputs = [bytearray(x) for x in (b'ab',b'cd',b'12',b'34',b'WX',b'YZ')]
    batch = _native.PrivateRecordPack(inputs, 4, 12)
    with pytest.raises(TypeError):
        memoryview(batch)
    with pytest.raises(AttributeError):
        batch.records
    with pytest.raises(BufferError):
        inputs[0].extend(b'!')
    batch.copy_range(2,3)
    with pytest.raises(RuntimeError, match='incomplete'):
        batch.finish()
    batch.copy_range(0,2)
    values = batch.finish()
    assert values == [b'abcd',b'1234',b'WXYZ']
    assert all(type(value) is bytes for value in values)
    inputs[0].extend(b'!')
    inputs[1][:] = b'??'
    assert values[0] == b'abcd'
    with pytest.raises(RuntimeError, match='closed'):
        batch.finish()
    assert batch.abort()


def test_range_claims_are_validated_before_any_new_claim():
    batch = _native.PrivateRecordPack([b'ab',b'cd']*3,4,12)
    batch.copy_range(0,2)
    with pytest.raises(RuntimeError, match='claimed'):
        batch.copy_range(1,3)
    batch.copy_range(2,3)
    assert batch.finish() == [b'abcd']*3


@pytest.mark.parametrize('sources,size,budget',[
    ([],4,4),([b'ab'],4,4),([b'ab',b'cd'],0,4),([b'ab',b'cd'],4,3),
    ([b'ab',b'c'],4,4),([b'abc',b'de'],4,4),
    ([np.zeros((2,2),dtype=np.uint8),b'ab'],4,4),
    ([np.arange(4,dtype=np.uint8)[::2],b'ab'],4,4),
])
def test_invalid_complete_input_fails_before_output_allocation(sources,size,budget):
    with pytest.raises((ValueError,TypeError,OverflowError)):
        _native.PrivateRecordPack(sources,size,budget)


def test_abort_releases_exported_sources_and_poisoned_outputs():
    raw = bytearray(b'ab')
    batch = _native.PrivateRecordPack([raw,b'cd',b'WX',b'YZ'],4,8)
    batch.copy_range(0,1)
    assert batch.abort() and batch.abort()
    raw.extend(b'!')
    with pytest.raises(ValueError, match='closed'):
        batch.copy_range(1,2)
    with pytest.raises(RuntimeError, match='closed'):
        batch.finish()


def archive(path, workers=8, capacity=17, **kwargs):
    return candidate.install(MemoryArchive(str(path),4,capacity,
        cpu_direct_record_seal=True,cpu_seal_pack_workers=workers,
        cpu_seal_hash_workers=8,**kwargs))


@pytest.mark.parametrize('workers,count',[(1,1),(2,17),(4,17),(8,17),(8,3)])
def test_exact_archive_bytes_digest_budget_and_retirement(tmp_path,workers,count):
    a=archive(tmp_path,workers,count)
    repo,leases,handles=repository(count)
    values=[bytes([i,255-i,i+1,254-i]) for i in range(count)]
    parts=[bytearray(x) for value in values for x in (value[:2],value[2:])]
    stats=a.write_from_views('owner',leases,parts).result(5)
    assert stats['record_digests'] == tuple(hashlib.sha256(v).hexdigest() for v in values)
    assert a.read('owner',leases[::-1]).result(5)[0] == b''.join(values[::-1])
    assert a.io_usage()['archive_io_peak_bytes'] == count*8
    assert a.io_usage()['seal_pack_parallel_shards_completed'] == (min(workers,count) if workers>1 and count>1 else 0)
    for part in parts:
        part.extend(b'!')
    reclaim_all(repo,leases,handles,a)
    assert a.close()


def test_worker_failure_drains_peers_before_rollback_and_releases_exports(tmp_path,monkeypatch):
    a=archive(tmp_path,2,2)
    repo,leases,handles=repository(2)
    entered,failed,release=Event(),Event(),Event()
    original=candidate._copy_private_range
    def work(owner,batch,begin,end):
        try:
            if begin==0:
                assert entered.wait(5)
                failed.set()
                raise OSError('private shard failed')
            entered.set()
            assert release.wait(5)
            return original(owner,batch,begin,end)
        finally:
            batch=None
    monkeypatch.setattr(candidate,'_copy_private_range',work)
    inputs=[np.array([i,i+1],dtype=np.uint8) for i in range(4)]
    refs=[weakref.ref(x) for x in inputs]
    job=a.write_from_views('owner',leases,inputs)
    inputs.clear()
    try:
        assert failed.wait(5)
        assert not job.done() and not a.close()
        assert a.io_usage()['archive_io_pending_bytes']==16
        assert not a.tier.records
        release.set()
        with pytest.raises(OSError,match='private shard'):
            job.result(5)
        assert all(ref() is None for ref in refs)
        assert not a._entries and a.tier.used_slots==0
        assert a.io_usage()['archive_io_pending_bytes']==0
        monkeypatch.setattr(candidate,'_copy_private_range',original)
        a.write_from_views('owner',leases,[b'ab',b'cd',b'WX',b'YZ']).result(5)
        reclaim_all(repo,leases,handles,a)
    finally:
        release.set()
        assert a.close()


def test_ambiguous_submit_drains_unknown_work_before_releasing_sources(tmp_path,monkeypatch):
    a=archive(tmp_path,2,2)
    repo,leases,handles=repository(2)
    entered,release=Event(),Event()
    submit=a._gather_executor.submit
    original=candidate._copy_private_range
    def work(owner,batch,begin,end):
        try:
            entered.set()
            assert release.wait(5)
            return original(owner,batch,begin,end)
        finally:
            batch=None
    def broken_submit(*args,**kwargs):
        submit(*args,**kwargs)
        assert entered.wait(5)
        raise OSError('submitted without future')
    monkeypatch.setattr(candidate,'_copy_private_range',work)
    monkeypatch.setattr(a._gather_executor,'submit',broken_submit)
    inputs=[np.array([i,i+1],dtype=np.uint8) for i in range(4)]
    refs=[weakref.ref(x) for x in inputs]
    job=a.write_from_views('owner',leases,inputs)
    inputs.clear()
    try:
        assert entered.wait(5) and not job.done()
        assert a.io_usage()['archive_io_pending_bytes']==16
        release.set()
        with pytest.raises(OSError,match='without future'):
            job.result(5)
        assert all(ref() is None for ref in refs)
        assert not a._entries and a.io_usage()['archive_io_pending_bytes']==0
        with pytest.raises(RuntimeError,match='unavailable'):
            a._pack_records([b'ab',b'cd',b'WX',b'YZ'])
        reclaim_all(repo,leases,handles,a)
    finally:
        release.set()
        assert a.close()


@pytest.mark.parametrize('failed',[False,True])
def test_real_option_dispatch_and_failed_future_source_release(tmp_path,monkeypatch,failed):
    a=MemoryArchive(str(tmp_path),4,2,cpu_direct_record_seal=True,
                    cpu_seal_serial_allocate=True,cpu_seal_hash_workers=8)
    repo,leases,handles=repository(2)
    inputs=[np.array([i,i+1],dtype=np.uint8) for i in range(4)]
    refs=[weakref.ref(x) for x in inputs]
    if failed:
        def error(owner,batch,begin,end):
            batch=None
            raise OSError('production private copy failed')
        monkeypatch.setattr(candidate,'_copy_private_range',error)
    job=a.write_from_views('owner',leases,inputs)
    inputs.clear()
    if failed:
        with pytest.raises(OSError,match='production private'):
            job.result(5)
        assert not a._entries and a.tier.used_slots==0
    else:
        job.result(5)
        assert a.read('owner',leases).result(5)[0] == bytes([0,1,1,2,2,3,3,4])
    assert all(ref() is None for ref in refs)
    usage=a.io_usage()
    assert usage['cpu_seal_serial_allocate']
    assert usage['seal_serial_allocate_batches_completed'] == (0 if failed else 1)
    assert usage['archive_io_pending_bytes']==0
    reclaim_all(repo,leases,handles,a)
    assert a.close()


def test_two_real_owners_share_pool_and_admission_without_partial_publication(tmp_path,monkeypatch):
    a=MemoryArchive(str(tmp_path),4,4,cpu_direct_record_seal=True,
        cpu_seal_serial_allocate=True,cpu_seal_hash_workers=8,gather_workers=2,
        max_pending=2,io_max_bytes=32)
    repo,leases,handles=repository(2)
    queued,release,lock=Event(),Event(),Lock()
    submit=a._gather_executor.submit
    original=candidate._copy_private_range
    jobs=[]
    def work(owner,batch,begin,end):
        try:
            assert release.wait(5)
            return original(owner,batch,begin,end)
        finally:
            batch=None
    def observed(*args,**kwargs):
        job=submit(*args,**kwargs)
        with lock:
            jobs.append(job)
            if len(jobs)==4:
                queued.set()
        return job
    monkeypatch.setattr(candidate,'_copy_private_range',work)
    monkeypatch.setattr(a._gather_executor,'submit',observed)
    first=a.write_from_views('owner',leases,[b'ab',b'cd',b'12',b'34'])
    second=a.write_from_views('other',leases,[b'WX',b'YZ',b'56',b'78'])
    try:
        assert queued.wait(5)
        assert len(a._gather_executor._threads)==2
        assert a.io_usage()['archive_io_pending_bytes']==32
        assert a.write_from_views('third',leases[:1],[b'!!',b'??']) is None
        assert not a.tier.records and not a.close()
        release.set()
        first.result(5);second.result(5)
        assert a.read('owner',leases).result(5)[0]==b'abcd1234'
        assert a.read('other',leases).result(5)[0]==b'WXYZ5678'
        assert a.io_usage()['seal_serial_allocate_batches_completed']==2
        for lease,handle in zip(leases,handles):
            repo.retire(handle);lease.reset()
            assert repo.reclaim(handle,lambda r:a.reclaim('owner',r) and a.reclaim('other',r)) == _native.ReclaimResult.RECLAIMED
        assert a.io_usage()['archive_io_pending_bytes']==0
    finally:
        release.set()
        assert a.close()
