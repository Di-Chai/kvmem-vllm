import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

spec=importlib.util.spec_from_file_location('offline_monitor',Path(__file__).parents[1]/'scripts/monitor_offline_inference.py')
monitor=importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_rates_use_elapsed_wall_time_and_publish_real_warmup_span():
    counter=monitor.CounterWindow()
    counter.observe(0,100)
    assert counter.rate(60)==(None,0)
    counter.observe(2,120)
    assert counter.rate(60)==(10,2)
    counter.observe(12,140)
    assert counter.rate(10)==(2,10)


def test_counter_reset_or_missing_source_never_becomes_negative_or_fake_zero():
    counter=monitor.CounterWindow()
    counter.observe(0,100)
    counter.observe(1,None)
    assert counter.rate(1)==(None,0)
    counter.observe(2,110)
    assert counter.rate(10)==(5,2)
    counter.observe(3,2)
    assert counter.rate(60)==(None,0) and counter.resets==1
    counter.observe(4,2)
    assert counter.rate(1)==(0,1)


def test_metrics_use_accepted_outputs_and_filter_other_models_and_draft_counts():
    text='''# HELP ignored
vllm:generation_tokens_total{engine="0",model_name="ours"} 8.0
vllm:generation_tokens_total{engine="1",model_name="ours"} 2.0
vllm:generation_tokens_total{engine="0",model_name="other"} 1000.0
vllm:prompt_tokens_total{engine="0",model_name="ours"} 262144.0
vllm:spec_decode_num_draft_tokens_total{model_name="ours"} 500
vllm:num_requests_running{model_name="ours"} 4
'''
    assert monitor.parse_metrics(text,'ours')==dict(accepted_output_total=10.,native_running=4.)


def test_jsonl_cursor_retries_incomplete_tail_but_rejects_invalid_full_record(tmp_path):
    source=tmp_path/'source.jsonl'
    source.write_bytes(b'{"n":1}\n{"n":')
    cursor=monitor.JsonlCursor(source)
    assert cursor.read()==[{'n':1}]
    assert cursor.read()==[]
    with source.open('ab') as output:output.write(b'2}\n')
    assert cursor.read()==[{'n':2}]
    with source.open('ab') as output:output.write(b'invalid\n')
    with pytest.raises(json.JSONDecodeError):cursor.read()


@pytest.mark.parametrize('source_age',[0,10])
def test_collector_records_sources_and_queue_without_inventing_first_sample_rates(tmp_path,monkeypatch,source_age):
    campaign=tmp_path/'campaign'
    arm=campaign/'multi'
    capture=arm/'server/capture'
    capture.mkdir(parents=True)
    qualification=arm/'qualification'
    qualification.mkdir()
    now=time.monotonic()-source_age
    def write(path,value):path.write_text(json.dumps(value))
    write(campaign/'manifest.json',dict(arms=['multi'],source_git_commit='test'))
    write(campaign/'status.json',dict(status='running'))
    write(arm/'status.json',dict(status='testing'))
    write(arm/'profile.json',dict(port=1,served_model_name='ours'))
    write(capture/'public_work.json',dict(at=now,counter_id='epoch1',completed=dict(public_prefill_tokens=12345)))
    write(qualification/'queue.json',dict(counts=dict(running=8,waiting=52)))
    write(qualification/'progress.json',dict(completed=2))
    (campaign/'multi-resources.jsonl').write_text(json.dumps(dict(at=now,gpu_utilization=50,gpu_free_mib=10000,gpu_power_w=300))+'\n')
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'vllm:generation_tokens_total{model_name="ours"} 6\n'
    monkeypatch.setattr(monitor.urllib.request,'build_opener',lambda *args:SimpleNamespace(open=lambda *args,**kw:Response()))
    output=tmp_path/'telemetry'
    monitor.run(SimpleNamespace(campaign=campaign,output=output,interval=.05,duration=.01))
    row=json.loads((output/'latest.json').read_text())
    assert row['gpu_utilization']==(50 if source_age==0 else None)
    assert row['prefill_completed_total']==(12345 if source_age==0 else None)
    assert row['accepted_output_total']==6 and row['outer_running']==8 and row['outer_waiting']==52
    assert row['prefill_tps_1s'] is None and row['decode_tps_60s'] is None
    assert json.loads((output/'status.json').read_text())['reason']=='duration_limit'
