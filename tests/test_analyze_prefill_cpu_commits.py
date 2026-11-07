"""Synthetic timelines guard against treating overlapped CPU work as stalls."""
import json
import sqlite3

import pytest

from analyze_prefill_cpu_commits import analyze


def test_nested_intervals_gpu_overlap_and_confirmed_gil_are_distinct(tmp_path):
    db=sqlite3.connect(tmp_path/'trace.sqlite')
    db.executescript('''
      CREATE TABLE StringIds(id INTEGER,value TEXT);
      INSERT INTO StringIds VALUES (1,'cudaLaunchKernel'),(2,'model_gemm'),
        (3,'pthread_cond_timedwait'),(4,'take_gil');
      CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,correlationId INTEGER);
      INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (90000000,95000000,1,1,1),(490000000,495000000,1,1,2);
      CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,demangledName INTEGER,correlationId INTEGER);
      INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100000000,300000000,1,2,1),(500000000,900000000,1,2,2);
      CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
      INSERT INTO NVTX_EVENTS VALUES
        (280000000,460000000,2,'prefill_phase::layerwise_pressure.prepare_accept',NULL),
        (290000000,450000000,2,'prefill_phase::cpu_commit.buffered_accept',NULL),
        (301000000,400000000,2,'prefill_phase::cpu_commit.native_committed',NULL),
        (250000000,250000001,1,'prefill_pressure::suspended::p1',NULL),
        (260000000,260000001,1,'prefill_pressure::assembly_queued::p1',NULL),
        (280000000,280000001,1,'prefill_pressure::assembly_started::p1',NULL),
        (261000000,601000000,3,'prefill_phase::refresh._assemble',NULL),
        (262000000,278000000,3,'prefill_phase::layerwise_pressure.prepare',NULL),
        (600000000,600000001,1,'prefill_pressure::ready::p1',NULL);
      CREATE TABLE OSRT_CALLCHAINS(id INTEGER,symbol INTEGER);
      INSERT INTO OSRT_CALLCHAINS VALUES(1,4);
      CREATE TABLE OSRT_API(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,callchainId INTEGER);
      INSERT INTO OSRT_API VALUES(320000000,370000000,1,3,1),
        (600000000,700000000,1,3,NULL);
    ''');db.commit();db.close()
    (tmp_path/'probe').mkdir()
    (tmp_path/'probe/cpu-commits-1.json').write_text(json.dumps({'rows':[
        dict(tid=2,label='cpu_commit.native_committed',start_ns=301000000,end_ns=400000000,cpu_ns=50000000)]}))
    x=analyze(tmp_path)
    assert x['total']==dict(wall_ms=800,gpu_busy_ms=600,gpu_idle_ms=200,
        main_confirmed_gil_ms=50,main_confirmed_gil_gpu_idle_ms=50)
    assert x['pressure']['wall_ms']==350 and x['pressure']['gpu_busy_ms']==150
    native=next(r for r in x['scopes'] if r['label']=='cpu_commit.native_committed')
    outer=next(r for r in x['scopes'] if r['label']=='layerwise_pressure.prepare_accept')
    assert native['inclusive_ms']==99 and outer['exclusive_ms']==20
    assert x['confirmed_main_gil_overlaps_publisher_native_ms']==50
    assert x['confirmed_main_gil_overlaps_publisher_python_ms']==0
    assert x['cpu_commit_timing'][0]['thread_cpu_ms']==50
    assert x['preparation_before_started_marker']==[dict(command='p1',
        queued_to_started_ms=20,queued_to_worker_entry_ms=1,
        layerwise_preparation_ms=16,main_confirmed_gil_during_preparation_ms=0)]
    # Another owner's preparation between queued and our worker entry must
    # not be attributed to this command, nor counted as its queue end.
    db=sqlite3.connect(tmp_path/'trace.sqlite')
    db.executescript('''
      INSERT INTO NVTX_EVENTS VALUES
        (260100000,260900000,3,'prefill_phase::refresh._assemble',NULL),
        (260200000,260800000,3,'prefill_phase::layerwise_pressure.prepare',NULL),
        (261000000,261000001,3,'prefill_pressure::assembly_worker_started::p1',NULL);
    ''');db.commit();db.close()
    assert analyze(tmp_path)['preparation_before_started_marker']==x['preparation_before_started_marker']
    # GC is clipped to the measured GPU window and remains nonadditive with
    # the containing CPU scope. A low GPU sample is not itself GC attribution.
    db=sqlite3.connect(tmp_path/'trace.sqlite')
    db.executescript('''
      INSERT INTO NVTX_EVENTS VALUES
        (330000000,350000000,2,'prefill_phase::python.gc.generation2',NULL),
        (90000000,105000000,1,'prefill_phase::python.gc.generation0',NULL);
    ''');db.commit();db.close()
    y=analyze(tmp_path)
    assert y['python_gc']['wall_ms']==25
    assert y['python_gc']['gpu_idle_ms']==20
    assert y['python_gc']['main_confirmed_gil_gpu_idle_ms']==20
    assert y['python_gc']['generations']['0']['max_ms']==5
    assert y['python_gc']['generations']['1']['count']==0
    assert y['python_gc']['generations']['2']['count']==1
    assert y['gpu_gaps_at_least_ms']['100']==dict(count=1,total_ms=200)


def test_no_activity_is_not_reported_as_zero_gpu_work(tmp_path):
    db=sqlite3.connect(tmp_path/'trace.sqlite')
    db.execute('CREATE TABLE StringIds(id INTEGER,value TEXT)');db.close()
    with pytest.raises(AssertionError,match='No GPU trace'):analyze(tmp_path)
