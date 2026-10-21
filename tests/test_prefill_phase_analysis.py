"""Synthetic time-domain checks; no GPU or model execution."""
import json
from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_prefill_phases as phases


def test_exclusive_scopes_partition_nested_same_start_and_reject_crossing():
    result = phases.exclusive([dict(start=2,end=4,label='inner'),
                               dict(start=2,end=6,label='outer')])
    assert result == [(2,4,'inner'),(4,6,'outer')]
    with pytest.raises(ValueError, match='cross'):
        phases.exclusive([dict(start=2,end=5,label='one'),dict(start=3,end=6,label='two')])


def fixture(tmp_path):
    pid = 41*(1<<24)
    main, background = pid+7, pid+8
    path = tmp_path/'trace.sqlite'
    connection = sqlite3.connect(path)
    connection.executescript('''
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,
            globalPid INTEGER,correlationId INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,
            correlationId INTEGER,nameId INTEGER);
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,textId INTEGER,text TEXT,globalTid INTEGER);
        CREATE TABLE OSRT_API(start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER,callchainId INTEGER);
        CREATE TABLE OSRT_CALLCHAINS(id INTEGER,symbol INTEGER);
    ''')
    connection.executemany('INSERT INTO StringIds VALUES (?,?)', [(1,'some_model_kernel'),
        (2,'cudaLaunchKernel'),(3,'take_gil'),(4,'pthread_cond_timedwait')])
    connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)', [
        (100,200,0,pid,1,1),(300,400,0,pid,2,1)])
    connection.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)', [
        (50,60,main,1,2),(280,310,main,2,2)])
    connection.executemany('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)', [
        (180,320,0,'prefill_phase::outer',main),
        (220,260,0,'prefill_phase::inner',main),
        (190,290,0,'prefill_phase::archive',background),
        (390,None,0,'prefill_phase::unfinished',background)])
    connection.execute('INSERT INTO OSRT_API VALUES (?,?,?,?,?)',(235,245,main,4,1))
    connection.execute('INSERT INTO OSRT_CALLCHAINS VALUES (?,?)',(1,3))
    connection.commit();connection.close()
    (tmp_path/'timeline-summary.json').write_text(json.dumps(dict(actual_geometry_verified=True,
        profiled_worker_shapes=[dict(start_ns=40,end_ns=350)],devices={'0':{}})))
    return main,background


def test_correlated_gpu_tail_and_deepest_main_gap_remain_separate_from_background(tmp_path):
    main,background=fixture(tmp_path)
    result=phases.analyze(tmp_path)
    by_name={row['label']:row for row in result['phase_rows']}
    assert result['main_launch_global_tid']==main
    assert result['gpu_gaps_ms']==100/1e6
    assert result['unsubmitted_gaps_ms']==80/1e6
    assert result['unsubmitted_gaps_outside_main_named_phases_ms']==0
    assert by_name['outer']['host_exclusive_in_gpu_window_ms']==100/1e6
    assert by_name['inner']['host_exclusive_in_gpu_window_ms']==40/1e6
    assert by_name['outer']['coincident_unsubmitted_gaps_ms']==40/1e6
    assert by_name['inner']['coincident_unsubmitted_gaps_ms']==40/1e6
    assert by_name['inner']['coincident_main_confirmed_gil_wait_ms']==10/1e6
    assert by_name['archive']['coincident_main_confirmed_gil_wait_ms']==10/1e6
    assert not by_name['archive']['main_launch_thread']
    # Entire kernel 300..400 belongs to API's outer scope even though CPU
    # returns at 320. Background overlap is concurrent, not extra idle.
    assert by_name['outer']['linked_gpu_activity_union_ms']==100/1e6
    assert by_name['archive']['coincident_gpu_gaps_ms']==90/1e6
    assert result['incomplete_phase_ranges']==[dict(label='unfinished',global_tid=background)]


def test_absent_phase_ranges_are_not_a_zero_cost_result(tmp_path):
    fixture(tmp_path)
    connection=sqlite3.connect(tmp_path/'trace.sqlite')
    connection.execute('DELETE FROM NVTX_EVENTS')
    connection.commit();connection.close()
    with pytest.raises(ValueError,match='No completed'):
        phases.analyze(tmp_path)


def test_unverified_actual_geometry_is_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path/'timeline-summary.json').write_text(json.dumps(dict(actual_geometry_verified=False)))
    with pytest.raises(ValueError,match='active shapes'):
        phases.analyze(tmp_path)


def test_prepare_fence_separates_prior_gpu_tail_from_remaining_host_work(tmp_path):
    main,_=fixture(tmp_path)
    connection=sqlite3.connect(tmp_path/'trace.sqlite')
    connection.execute("UPDATE NVTX_EVENTS SET text='prefill_phase::action.prepare' WHERE text='prefill_phase::outer'")
    connection.execute("UPDATE NVTX_EVENTS SET text='prefill_phase::native_stats.arm' WHERE text='prefill_phase::inner'")
    connection.execute('INSERT INTO StringIds VALUES (?,?)',(5,'cudaEventSynchronize'))
    connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?,?)',(201,215,main,3,5))
    connection.commit();connection.close()
    (tmp_path/'timeline-summary.json').write_text(json.dumps(dict(actual_geometry_verified=True,
        profiled_worker_shapes=[dict(start_ns=40,end_ns=170),dict(start_ns=180,end_ns=350)],devices={'0':{}})))
    row=phases.analyze(tmp_path)['action_prepare_fences'][0]
    assert row['step']==1 and row['host_total_ms']==140/1e6
    assert row['host_residual_excluding_cuda_sync_ms']==126/1e6
    assert row['previous_gpu_end_after_execute_ms']==20/1e6
    assert row['previous_gpu_busy_inside_prepare_ms']==20/1e6
    assert row['native_arm_host_ms']==40/1e6
