"""Prepared pressure paired gather keeps source, admission and fallback rules."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.serving.pressure_reader import prepare_pressure_reader
from test_prepared_pressure_reader import authority_fixture
from test_query_paired_gather import reference_pair


@pytest.mark.parametrize('enabled',[False,True])
def test_pressure_pair_reuses_union_leases_and_reports_transaction_counters(tmp_path,monkeypatch,enabled):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER',str(int(enabled)))
    calls=[]
    def pair(*args):calls.append(1);return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',pair,raising=False)
    a,plan,records,pins,repo,leases,values=authority_fixture(tmp_path)
    before=repo.stats['leases']
    groups=([records[2],records[0]],[records[1],records[2]])
    try:
        with prepare_pressure_reader(a,plan,groups) as reader:
            assert len(pins)==reader.source_blocks==3
            for group in groups:
                out=torch.empty(40,dtype=torch.uint8)
                actual,stats=reader.read(group,((3,11),(47,9)),destination=out)
                assert actual.data_ptr()==out.data_ptr()
                assert actual.numpy().tobytes()==b''.join(values[r.history_begin//4][o:o+n]
                    for o,n in ((3,11),(47,9)) for r in group)
                assert stats['prepared_cpu_region_read'] and stats['inline_cpu_region_read']
                assert stats.get('paired_native_gather',False)==enabled
            assert calls==([1,1] if enabled else [])
            if enabled:
                assert reader.read_stats['pressure_paired_gather_chunks']==2
                assert reader.read_stats['pressure_paired_fallback_chunks']==0
                assert reader.read_stats['pressure_paired_copied_bytes']==80
                assert reader.read_stats['pressure_paired_source_prepare_seconds']==2e-9
                assert reader.read_stats['pressure_paired_copy_seconds']==4e-9
                assert reader.read_stats['pressure_paired_gil_reacquire_seconds']==8e-9
            else:assert reader.read_stats=={}
        assert repo.stats['leases']==before and a.archive._prepared_region_readers==0
    finally:assert a.archive.close()


@pytest.mark.parametrize('reason',['threshold','whole_record'])
def test_pressure_pair_retains_existing_parallel_and_full_record_paths(tmp_path,monkeypatch,reason):
    from vllm_kvmem.core import memory_storage
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    monkeypatch.setattr(_native,'gather_record_regions',lambda *args:pytest.fail('Ineligible paired path'),raising=False)
    a,plan,records,_,_,_,values=authority_fixture(tmp_path)
    regions=((0,16),(32,16)) if reason=='threshold' else ((0,64),(0,64))
    if reason=='threshold':monkeypatch.setattr(memory_storage,'CPU_REGION_GATHER_PARALLEL_MIN_BYTES',48)
    try:
        with prepare_pressure_reader(a,plan,[records]) as reader:
            out=torch.empty(len(records)*sum(n for _,n in regions),dtype=torch.uint8)
            actual,stats=reader.read(records,regions,destination=out)
            assert actual.numpy().tobytes()==b''.join(v[o:o+n] for o,n in regions for v in values)
            assert stats['paired_native_gather'] is False
            assert stats['paired_native_fallback_reason']=='parallel_or_full_record_region'
            assert reader.read_stats['pressure_paired_gather_chunks']==0
            assert reader.read_stats['pressure_paired_fallback_chunks']==1
    finally:assert a.archive.close()


def test_candidate_environment_does_not_change_direct_prepared_query_reads(tmp_path,monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    monkeypatch.setattr(_native,'gather_record_regions',lambda *args:pytest.fail('Environment escaped pressure'),raising=False)
    a,_,_,_,_,leases,values=authority_fixture(tmp_path)
    reader=a.archive.prepare_region_reader('owner',leases)
    try:
        actual,stats=reader.read([2,0],((3,11),(47,9)),destination=bytearray(40))
        assert bytes(actual)==b''.join(values[i][o:o+n] for o,n in ((3,11),(47,9)) for i in (2,0))
        assert 'paired_native_gather' not in stats
    finally:
        assert reader.close() and a.archive.close()


@pytest.mark.parametrize('fail',[False,True])
def test_pressure_pair_retains_admission_and_sources_until_cpu_producer_drains(tmp_path,monkeypatch,fail):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    entered,release=Event(),Event()
    def blocked(*args):
        entered.set();assert release.wait(5)
        if fail:raise OSError('paired copy failed')
        return reference_pair(*args)
    monkeypatch.setattr(_native,'gather_record_regions',blocked,raising=False)
    a,p,records,_,repo,leases,values=authority_fixture(tmp_path)
    reader=prepare_pressure_reader(a,p,[records])
    out=torch.full((32,),73,dtype=torch.uint8)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future=executor.submit(reader.read,[records[0]],((0,16),(32,16)),destination=out)
            try:
                assert entered.wait(5)
                assert not reader.close() and not a.archive.close()
                assert a.archive.io_usage()['archive_io_pending_bytes']==32
                handle=leases[0].handle
                assert repo.retire(handle);leases[0].reset()
                assert repo.reclaim(handle,lambda r:a.archive.reclaim('owner',r))==_native.ReclaimResult.READERS
                a.archive.max_pending=1
                assert reader.reader.read([1],((0,16),(32,16)),destination=bytearray(32),paired_native=True) is None
            finally:release.set()
            if fail:
                with pytest.raises(OSError,match='paired copy failed'):future.result(5)
            else:
                actual,_=future.result(5)
                assert actual.numpy().tobytes()==values[0][:16]+values[0][32:48]
        assert a.archive.io_usage()['archive_io_pending_bytes']==0
        assert reader.close()
        assert repo.reclaim(handle,lambda r:a.archive.reclaim('owner',r))==_native.ReclaimResult.RECLAIMED
    finally:
        release.set();assert reader.close() and a.archive.close()


@pytest.mark.parametrize('failure',['extension','pool','source','owner','flag'])
def test_invalid_pair_or_stale_source_never_silently_writes(tmp_path,monkeypatch,failure):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    monkeypatch.setattr(_native,'gather_record_regions',reference_pair,raising=False)
    a,p,records,_,_,_,_=authority_fixture(tmp_path)
    out=torch.full((32,),73,dtype=torch.uint8)
    reader=None
    try:
        if failure=='flag':
            monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','invalid')
            with pytest.raises(ValueError,match='must be 0 or 1'):prepare_pressure_reader(a,p,[records])
        else:
            reader=prepare_pressure_reader(a,p,[records])
            if failure=='extension':monkeypatch.setattr(_native,'gather_record_regions',None)
            if failure=='pool':a.archive._gather_unusable='prior failure'
            if failure=='source':records[0].handle=records[1].handle
            if failure=='owner':a.memory.request_id='different-owner'
            with pytest.raises((RuntimeError,ValueError)):
                reader.read([records[0]],((0,16),(32,16)),destination=out)
        assert torch.all(out==73) and a.archive.io_usage()['archive_io_pending_bytes']==0
    finally:
        if reader is not None:assert reader.close()
        assert a.archive.close()


def test_actual_native_pair_on_prepared_pressure_sources(tmp_path,monkeypatch):
    if not callable(getattr(_native,'gather_record_regions',None)):
        pytest.skip('Updated native extension not built')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PAIRED_GATHER','1')
    a,p,records,_,_,_,values=authority_fixture(tmp_path)
    try:
        with prepare_pressure_reader(a,p,[records]) as reader:
            out=torch.empty(60,dtype=torch.uint8)
            actual,stats=reader.read(records,((3,11),(47,9)),destination=out)
            assert actual.numpy().tobytes()==b''.join(v[o:o+n] for o,n in ((3,11),(47,9)) for v in values)
            assert stats['paired_native_copied_bytes']==60
            counters=reader.read_stats
            assert counters['pressure_paired_gather_chunks']==1
            assert counters['pressure_paired_total_seconds']>=counters['pressure_paired_copy_seconds']
    finally:assert a.archive.close()
