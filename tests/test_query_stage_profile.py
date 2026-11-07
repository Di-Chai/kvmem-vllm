"""CPU oracles for nonblocking, bounded final-Query diagnostics."""
from contextlib import contextmanager
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.query_stage_profile import QueryStageProfile, no_stage
from test_assembly_read_prefetch import cuda, DeviceBytes, FakeEvent


class Events:
    def __init__(self):
        self.ready = False
        self.recorded = self.queried = 0

    def event(self):
        owner = self
        class Event:
            def record(self): owner.recorded += 1
            def query(self):
                owner.queried += 1
                return owner.ready
            def elapsed_time(self, end):
                assert owner.ready
                return .5
            def synchronize(self): pytest.fail('Probe added a CUDA fence')
        return Event()


def probe(tmp_path, **kwargs):
    events = Events()
    result = QueryStageProfile(tmp_path/'stages.jsonl', 'request-a',
        event_factory=events.event, capturing=lambda:False, **kwargs)
    return result, events


def test_inclusive_host_nesting_and_gpu_events_wait_for_existing_completion(tmp_path):
    p,e = probe(tmp_path)
    p.layer=3
    with p.stage('outer'):
        with p.stage('copy', gpu=True, transfer_bytes=64): pass
    assert e.recorded==2 and e.queried==0 and not p.output.exists()
    e.ready=True
    result=p.finish()
    outer,copy=result['stages']
    assert copy['parent']==outer['sequence']
    assert copy['layer']==3 and copy['transfer_bytes']==64 and copy['gpu_elapsed_ms']==.5
    assert outer['wall_seconds']>=copy['wall_seconds']>=0
    assert result['unresolved_gpu_events']==0
    assert json.loads(p.output.read_text())['request_id']=='request-a'


def test_capacity_reserves_nested_events_and_drops_telemetry_only(tmp_path):
    p,e=probe(tmp_path,max_events=1,max_rows=2)
    ran=[]
    with p.stage('outer',gpu=True):
        with p.stage('inner',gpu=True):
            with p.stage('dropped',gpu=True): ran.append(True)
    result=p.finish()
    assert ran==[True]
    assert result['dropped_gpu_events']==1 and result['dropped_rows']==1
    assert result['unresolved_gpu_events']==1 and e.recorded==2
    assert len(result['stages'])==2


def test_failure_is_preserved_and_graph_capture_cannot_allocate_events(tmp_path):
    p,e=probe(tmp_path)
    with pytest.raises(OSError,match='original'):
        with p.stage('read'):raise OSError('original')
    assert not p.stack and p.rows[0]['error_type']=='OSError'
    p.capturing=lambda:True
    with pytest.raises(RuntimeError,match='captured graph'):
        with p.stage('copy',gpu=True):pytest.fail('capture entered')
    assert e.recorded==0


def test_finish_never_queries_an_active_scope(tmp_path):
    p,e=probe(tmp_path)
    with p.stage('outer'):
        with pytest.raises(RuntimeError,match='active scope'):p.finish()
    assert not p.finished and e.queried==0
    p.finish()
    with pytest.raises(RuntimeError,match='already completed'):
        with p.stage('late'):pass


def test_packed_read_retains_begin_result_order_and_scalar_archive_evidence(tmp_path,monkeypatch):
    from vllm_kvmem import packed_assembly
    p,_=probe(tmp_path)
    payload=object();authority=object();handles=[object(),object()];calls=[]
    stats=dict(cache_copy_seconds=.004,read_queue_wait_seconds=.003,
               admission_wait_seconds=.002,cache_gather_tasks=4,owners=[object()])
    def begin(a,h,**kwargs):
        calls.append(('begin',a,h,kwargs))
        def result():
            calls.append(('result',))
            return payload,stats
        return NS(result=result)
    monkeypatch.setattr(packed_assembly,'begin_read_packed_records',begin)
    assert p.read_packed(authority,handles,regions=((0,4),(8,4)),destination=payload)==(payload,stats)
    assert calls==[('begin',authority,handles,dict(regions=((0,4),(8,4)),destination=payload)),('result',)]
    result=p.finish()
    assert [r['stage'] for r in result['stages']]==[
        'archive_source_prepare_and_admit','archive_result_wait_and_release','archive_report']
    assert result['stages'][-1]['stats']=={k:v for k,v in stats.items() if k!='owners'}


@pytest.mark.parametrize('max_rows',[1,8])
def test_prepared_read_reports_scalar_stats_without_replacing_its_callback(tmp_path,max_rows):
    p,e=probe(tmp_path,max_rows=max_rows)
    p.layer=5
    authority,handles,payload=object(),[object()],object()
    stats=dict(query_lazy_region_read=True,query_reader_new_blocks=1,
               query_reader_prepare_seconds=.002,cache_copy_seconds=.004,owners=[object()])
    calls=[]
    def reader(a,h,**kwargs):
        calls.append((a,h,kwargs))
        return payload,stats
    kwargs=dict(regions=((0,4),(8,4)),destination=payload)
    result,actual=p.read_prepared(reader,authority,handles,**kwargs)
    assert result is payload and actual is stats and calls==[(authority,handles,kwargs)]
    assert not e.recorded and not e.queried
    report=p.finish()
    assert report['stages'][0]['stage']=='archive_query_prepared_read'
    assert report['stages'][0]['layer']==5
    if max_rows==1:
        assert len(report['stages'])==1 and report['dropped_rows']==1
    else:
        assert report['stages'][1]['stage']=='archive_report'
        assert report['stages'][1]['stats']=={k:v for k,v in stats.items() if k!='owners'}


def test_prepared_read_failure_is_preserved_without_fabricating_success_stats(tmp_path):
    p,e=probe(tmp_path)
    failure=OSError('original prepared copy failure')
    calls=[]
    def reader(*args,**kwargs):
        calls.append(1)
        raise failure
    with pytest.raises(OSError) as caught:
        p.read_prepared(reader,object(),[object()],region=(0,4))
    assert caught.value is failure and calls==[1]
    report=p.finish()
    assert len(report['stages'])==1
    assert report['stages'][0]['stage']=='archive_query_prepared_read'
    assert report['stages'][0]['error_type']=='OSError'
    assert not e.recorded and not e.queried


def test_score_preserves_one_gpu_compute_download_and_materialization(tmp_path):
    p,e=probe(tmp_path)
    calls=[]
    def compute():
        calls.append('compute')
        def cpu():
            calls.append('cpu')
            return NS(tolist=lambda:calls.append('tolist') or [1.,2.])
        return NS(cpu=cpu)
    assert p.score(compute)==[1.,2.] and calls==['compute','cpu','tolist']
    e.ready=True
    assert [r['stage'] for r in p.finish()['stages']]==[
        'score_gpu_dispatch','score_download_wait','score_python_materialize']


def test_transfer_copy_probe_follows_ready_dependency_and_precedes_lifetime_fence(cuda):
    operations=cuda.operations
    class Profile:
        @contextmanager
        def stage(self,name,**kwargs):
            operations.append(('stage_begin',name,kwargs))
            yield
            operations.append(('stage_end',name))
    ready=FakeEvent(False)
    destination=DeviceBytes(torch.zeros(16,dtype=torch.uint8),operations)
    txn=cuda.pool.upload_from_reader(16,destination,lambda b:b.fill_(47),ready=ready,profile=Profile())
    assert torch.equal(destination.data,torch.full((16,),47,dtype=torch.uint8))
    names=[o[0] for o in operations]
    ready_index=next(i for i,o in enumerate(operations) if o[0]=='wait' and o[1] is ready)
    copy_begin=next(i for i,o in enumerate(operations) if o[:2]==('stage_begin','h2d_copy'))
    copy_end=next(i for i,o in enumerate(operations) if o[:2]==('stage_end','h2d_copy'))
    fence=next(i for i,o in enumerate(operations) if o[0]=='event' and o[1] is txn.event)
    assert ready_index<copy_begin<names.index('copy')<copy_end<fence
    assert 'event_sync' not in names and 'stream_sync' not in names
    assert cuda.pool.pending[txn.slot] is txn and not cuda.pool.preparing


def test_default_off_and_explicit_supported_query_profile(monkeypatch):
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    from test_layerwise_query import runner
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.delenv('KVMEM_QUERY_STAGE_PROFILE',raising=False)
    assert not LayerwiseQuery(runner()).stage_profile
    monkeypatch.setenv('KVMEM_QUERY_STAGE_PROFILE','1')
    with pytest.raises(ValueError,match='optimized sequential'):LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED','1')
    assert LayerwiseQuery(runner()).stage_profile
    monkeypatch.setenv('KVMEM_QUERY_STAGE_PROFILE','yes')
    with pytest.raises(ValueError,match='0 or 1'):LayerwiseQuery(runner())


def test_disabled_stage_never_reads_a_clock(monkeypatch):
    import vllm_kvmem.query_stage_profile as implementation
    monkeypatch.setattr(implementation.time,'perf_counter',lambda:pytest.fail('disabled timing'))
    with no_stage('unused',gpu=True): pass
