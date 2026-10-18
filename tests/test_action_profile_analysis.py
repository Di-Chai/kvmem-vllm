import importlib.util
from pathlib import Path

import pytest


path=Path(__file__).resolve().parents[1]/'scripts/analyze_action_capture.py'
spec=importlib.util.spec_from_file_location('analyze_action_capture',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
analyze=module.analyze


def fixture():
    def scope(batch,identity,kind,stage,at,wall,**extra):
        return dict(batch_id=batch,scope_id=identity,kind=kind,layer='L3',stage=stage,
                    started_at=at,host_wall_seconds=wall,host_thread_cpu_seconds=wall/2,**extra)
    outer=scope(1,1,'output','score.layer_total',110,2.)
    inner=scope(1,2,'output','score.softmax',111,1.)
    prefill=scope(2,3,'prefill','metadata.starts_h2d',210,3.,
                  gpu_event_dropped=True,transfer_bytes=64)
    return [dict(event='batch',batch_id=1,started_at=100,stages=[outer,inner]),
            dict(event='batch',batch_id=2,started_at=200,stages=[prefill]),
            dict(event='gpu_stage',emitted_at=900,gpu_elapsed_ms=4.,**outer),
            dict(event='gpu_stage',emitted_at=901,gpu_elapsed_ms=2.,**inner),
            dict(event='summary',counters={'gpu_event_drops':1,'graph_capture_skips':2},
                 pending_gpu_event_pairs=3,max_gpu_event_pairs=256,sample_interval=8)]


def test_late_cuda_rows_use_original_scope_start_and_do_not_double_count_host():
    result=analyze(fixture(),start=105,end=120)
    stages={row['stage']:row for row in result['stages']}
    assert set(stages)=={'score.layer_total','score.softmax'}
    assert stages['score.layer_total']['host_wall_seconds']==2.
    assert stages['score.layer_total']['host_calls']==stages['score.layer_total']['gpu_calls']==1
    assert stages['score.softmax']['host_thread_cpu_seconds']==.5
    assert stages['score.softmax']['gpu_elapsed_ms']==2.
    assert result['selected_host_gpu_event_drops']==0
    assert result['lifetime']['counters']['gpu_event_drops']==1
    assert result['lifetime']['pending_gpu_event_pairs']==3
    assert 'idle' not in result and 'total_runtime' not in result
    assert 'Nested scopes overlap' in result['interpretation']


def test_batch_basis_and_phase_transfer_aggregation():
    result=analyze(fixture(),start=95,end=105,time_basis='batch')
    assert {row['stage'] for row in result['stages']}=={'score.layer_total','score.softmax'}
    assert all(row['gpu_calls']==1 for row in result['stages'])
    result=analyze(fixture(),start=200,end=211)
    assert len(result['stages'])==1
    stage=result['stages'][0]
    assert stage['kind']=='prefill' and stage['layer']=='L3'
    assert stage['transfer_calls']==1 and stage['transfer_bytes']==64
    assert stage['gpu_calls']==0 and result['selected_host_gpu_event_drops']==1


def test_boundaries_unknown_summary_and_orphan_gpu_rows_are_explicit():
    rows=fixture()
    assert not analyze(rows,start=120,end=200)['stages']
    result=analyze(rows[2:4],start=100,end=120,time_basis='batch')
    assert not result['stages'] and result['scopes_without_requested_timestamp']==2
    assert result['completed_gpu_scopes_without_host_record']==2
    assert result['lifetime']['counters'] is None
    assert result['lifetime']['pending_gpu_event_pairs'] is None
    assert len(analyze(rows[2:4],start=100,end=120)['stages'])==2


@pytest.mark.parametrize('kwargs',[{'start':float('nan')},{'end':float('inf')},
                                 {'start':2,'end':1},{'time_basis':'write_time'}])
def test_invalid_windows_are_rejected(kwargs):
    with pytest.raises(ValueError):analyze(fixture(),**kwargs)


def test_duplicate_completed_intervals_are_not_silently_counted_twice():
    rows=fixture();rows.append(rows[2])
    with pytest.raises(ValueError,match='Duplicate completed'):analyze(rows)
