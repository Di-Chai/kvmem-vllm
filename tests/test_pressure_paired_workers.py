"""Two existing gather workers preserve packed bytes and failure ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.pressure_reader import prepare_pressure_reader
from test_prepared_pressure_reader import authority_fixture
from test_query_paired_gather import reference_pair


def enable(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS','2')


@pytest.mark.parametrize('order',[(2,0,1),(1,1),(2,)])
def test_disjoint_native_ranges_preserve_order_and_separate_wall_from_work(tmp_path,monkeypatch,order):
    enable(monkeypatch);calls=[]
    def pair(*args):
        calls.append(args[-2:]);return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',pair,raising=False)
    a,p,records,_,repo,_,values=authority_fixture(tmp_path)
    before=repo.stats['leases']
    try:
        with prepare_pressure_reader(a,p,[records]) as reader:
            out=torch.empty(len(order)*20,dtype=torch.uint8)
            actual,stats=reader.read([records[i] for i in order],((3,11),(47,9)),destination=out)
            assert actual.numpy().tobytes()==b''.join(values[i][o:o+n] for o,n in ((3,11),(47,9)) for i in order)
            workers=min(2,len(order))
            expected=[(len(order)*i//workers,len(order)*(i+1)//workers) for i in range(workers)]
            assert sorted(calls)==expected
            assert stats['paired_native_worker_count']==workers
            assert stats['paired_native_copy_ns']==2*workers
            assert stats['cache_gather_tasks']==workers
            totals=reader.read_stats
            assert totals['pressure_paired_native_calls']==workers
            assert totals['pressure_paired_parallel_chunks']==int(workers==2)
            assert totals['pressure_paired_inline_chunks']==int(workers==1)
            assert totals['pressure_paired_copy_seconds']==2*workers/1e9
            assert totals['pressure_paired_gather_wall_seconds']>0
            assert totals['pressure_paired_native_timing_scope']=='sum_of_native_calls_not_elapsed_wall'
        assert repo.stats['leases']==before
        assert a.archive.io_usage()['archive_io_pending_bytes']==0
    finally:assert a.archive.close()


def test_failed_first_worker_does_not_release_second_worker_destination(tmp_path,monkeypatch):
    enable(monkeypatch)
    second,first_failed,release=Event(),Event(),Event()
    def pair(*args):
        if args[-2]==0:
            assert second.wait(5);first_failed.set();raise OSError('first shard failed')
        second.set();assert release.wait(5)
        return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',pair,raising=False)
    a,p,records,_,_,_,_=authority_fixture(tmp_path)
    reader=prepare_pressure_reader(a,p,[records]);out=torch.full((64,),73,dtype=torch.uint8)
    try:
        with ThreadPoolExecutor(max_workers=1) as outer:
            future=outer.submit(reader.read,records[:2],((0,16),(32,16)),destination=out)
            try:
                assert first_failed.wait(5) and not future.done()
                assert not reader.close() and not a.archive.close()
                assert a.archive.io_usage()['archive_io_pending_bytes']==64
            finally:release.set()
            with pytest.raises(OSError,match='first shard failed'):future.result(5)
        assert a.archive.io_usage()['archive_io_pending_bytes']==0
        assert not a.archive._gather_unusable  # Copy error does not poison a healthy pool.
        assert reader.close()
    finally:
        release.set();assert reader.close() and a.archive.close()


@pytest.mark.parametrize('raise_at',[1,2])
def test_unseen_submitted_worker_is_drained_before_failure_releases_sources(tmp_path,monkeypatch,raise_at):
    enable(monkeypatch)
    entered,submit_failed,release=[Event(),Event()],Event(),Event()
    def pair(*args):
        entered[args[-2]].set();assert release.wait(5)
        return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',pair,raising=False)
    a,p,records,_,_,_,_=authority_fixture(tmp_path)
    reader=prepare_pressure_reader(a,p,[records]);out=torch.full((64,),73,dtype=torch.uint8)
    original=a.archive._gather_executor.submit;calls=[]
    def submit(*args,**kwargs):
        future=original(*args,**kwargs);calls.append(1)
        if len(calls)==raise_at:
            assert entered[raise_at-1].wait(5)
            submit_failed.set();raise RuntimeError('submission returned no Future')
        return future
    monkeypatch.setattr(a.archive._gather_executor,'submit',submit)
    try:
        with ThreadPoolExecutor(max_workers=1) as outer:
            future=outer.submit(reader.read,records[:2],((0,16),(32,16)),destination=out)
            try:
                assert submit_failed.wait(5) and entered[raise_at-1].is_set()
                assert not future.done() and not reader.close()
                assert a.archive.io_usage()['archive_io_pending_bytes']==64
            finally:release.set()
            with pytest.raises(RuntimeError,match='submission returned no Future'):future.result(5)
        assert a.archive._gather_unusable.startswith('submission failed')
        assert a.archive._gather_shutdown_done.is_set()
        assert a.archive.io_usage()['archive_io_pending_bytes']==0
        out.fill_(73)
        with pytest.raises(RuntimeError,match='pool unavailable'):
            reader.read(records[:2],((0,16),(32,16)),destination=out)
        assert torch.all(out==73)
    finally:
        release.set();assert reader.close() and a.archive.close()


@pytest.mark.parametrize('case',['one_pool_worker','recursive','disabled','bad_count'])
def test_unsupported_parallel_dispatch_does_not_broaden_execution_policy(tmp_path,monkeypatch,case):
    enable(monkeypatch)
    monkeypatch.setattr(_native,'gather_record_regions',reference_pair,raising=False)
    a,p,records,_,_,_,_=authority_fixture(tmp_path)
    reader=None;out=torch.full((64,),73,dtype=torch.uint8)
    try:
        if case=='disabled':monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','0')
        if case=='bad_count':monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER_WORKERS','3')
        if case in ('disabled','bad_count'):
            with pytest.raises(ValueError,match='paired workers'):prepare_pressure_reader(a,p,[records])
        else:
            reader=prepare_pressure_reader(a,p,[records])
            if case=='one_pool_worker':a.archive.gather_workers=1
            if case=='recursive':a.archive._gather_worker.active=True
            with pytest.raises((ValueError,RuntimeError)):
                reader.read(records[:2],((0,16),(32,16)),destination=out)
        assert torch.all(out==73) and a.archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        a.archive._gather_worker.active=False
        if reader is not None:assert reader.close()
        assert a.archive.close()
