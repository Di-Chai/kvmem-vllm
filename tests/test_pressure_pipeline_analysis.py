"""Early dispatch and final materialization are distinct timing boundaries."""
import json
import pytest

from analyze_mab_prefill_concurrency import measure


def fixture(root, pipeline):
    def write(name, value, rows=False):
        p=root/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(''.join(json.dumps(v)+'\n' for v in value) if rows else json.dumps(value))
    capture='run/server/capture/'
    write(capture+'prefill_phase_gate.jsonl', [
        dict(event='history_prefill_arrived',request_id='r',at=9.),
        dict(event='history_prefill_released',expected=1,owners=['r'],at=10.,history_tokens=100)],True)
    write(capture+'public_work.json',dict(created_wall_time=100.,created_at=0.))
    write('run/answers/cases/case/attempt.json',dict(at=101.))
    stages=[('suspended',3.),('assembly_queued',3.1),('assembly_started',4.),
            ('assembly_submitted',5.)]
    stages += [('pipeline_dispatch',4.2),('pipeline_completed',5.1)] if pipeline else [('ready',5.1)]
    write(capture+'async_refresh.jsonl',[
        dict(stage=s,at=t,request_id='r',command_id=1) for s,t in stages],True)
    write(capture+'bounded_refresh.jsonl',[dict(selection='query',request_id='r',command_id=2)],True)
    write(capture+'model_step_latency.jsonl',[dict(started_at=2.,decode_requests=0,
        prefill_requests=1,prefill_tokens=100,model_ms=5.,step_ms=6.,
        batch_geometry=[dict(request_id='r',prefill=True,query_tokens=100,active_seq_len=96)])],True)
    write('run/gpu.jsonl',[dict(at=103.,devices=[dict(gpu_utilization_percent=75)])],True)


def test_pipeline_preserves_prefill_numerator_and_interval_without_negative_completion_wait(tmp_path):
    a,b=tmp_path/'baseline',tmp_path/'pipeline';fixture(a,False);fixture(b,True)
    old,new=measure(a),measure(b)
    assert old['tokens_per_second']==new['tokens_per_second']==100/9
    assert old['seconds']==new['seconds']==9.
    assert new['pressure_timings']['suspended_to_resumable']['mean_seconds']==pytest.approx(1.2)
    assert old['pressure_timings']['suspended_to_resumable']['mean_seconds']==pytest.approx(2.1)
    assert 'assembly_submitted_to_ready' not in new['pressure_timings']
    assert new['pressure_timings']['dispatch_to_materialization_reaped']['mean_seconds']==pytest.approx(.9)


def test_missing_producer_completion_cannot_pass_pipeline_audit(tmp_path):
    fixture(tmp_path,True)
    path=tmp_path/'run/server/capture/async_refresh.jsonl'
    path.write_text(''.join(s+'\n' for s in path.read_text().splitlines() if 'pipeline_completed' not in s))
    with pytest.raises(KeyError): measure(tmp_path)
