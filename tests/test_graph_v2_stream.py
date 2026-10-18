"""CPU tests for real admission/release fences and honest fixed-window accounting."""
from copy import deepcopy
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    client = importlib.import_module('qualify_graph_v2_stream')
    control = importlib.import_module('benchmark_graph_v2_stream')
finally:
    sys.path.pop(0)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def rows(count=128):
    return [dict(id=str(i),original_case=str(i%8)) for i in range(count)]


def test_controller_bootstraps_without_importing_an_installed_serving_package(tmp_path):
    # Frozen controllers start before the child service's PYTHONPATH is set.
    # Importing the qualifier for four constants accidentally loaded the old
    # installed package instead of the frozen child's runtime.
    package = tmp_path/'vllm_kvmem'
    package.mkdir()
    (package/'__init__.py').write_text("raise RuntimeError('serving package imported by controller')\n")
    result = subprocess.run([sys.executable, '-B', str(SCRIPTS/'benchmark_graph_v2_stream.py'), '--help'],
        env=dict(os.environ, PYTHONPATH=str(tmp_path), CUDA_VISIBLE_DEVICES='',
                 PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'prepare' in result.stdout and 'arm-run' in result.stdout


def test_continuous_refill_waits_for_own_release_without_waiting_for_other_owner(tmp_path):
    admitted_third = threading.Event()
    queue = client.WindowQueue(rows(),{str(i):10 for i in range(128)},20,
        slots=2,warmup=2,seconds=.06)
    recorder = client.Recorder(tmp_path)

    def work(row):
        if row['id'] == '2':
            admitted_third.set()
        time.sleep(.005)
        return row

    def release(row,result):
        if row['id'] == '0':
            assert admitted_third.wait(1), 'Another owner must refill while this release waits'
            assert queue.states['0']['state'] == 'retiring'
            assert queue.inflight == 2 and queue.reserved == 20
        return result

    results = queue.run(work,release,recorder)
    events = client.q.records(tmp_path/'events.jsonl')
    third = next(r for r in events if r['kind']=='admitted' and r['source_request']=='2')
    first_release = next(r for r in events if r['kind']=='released' and r['source_request']=='0')
    assert third['at'] < first_release['at']
    assert queue.peak == 2 and queue.peak_bytes == 20
    assert queue.inflight == queue.reserved == 0
    assert queue.begin is not None and queue.end-queue.begin == pytest.approx(.06)
    assert all(r['admitted_at'] < queue.end for r in events if r['kind']=='admitted')
    assert len(results) < 128 and any(r['state']=='not_admitted_deadline' for r in queue.states.values())


def test_deadline_stops_new_admission_while_existing_request_drains(tmp_path):
    queue = client.WindowQueue(rows(),{str(i):1 for i in range(128)},2,slots=2,warmup=1,seconds=.02)
    recorder = client.Recorder(tmp_path)
    def work(row):
        time.sleep(.08 if row['id']=='1' else .005)
        return row
    queue.run(work,lambda row,result:result,recorder)
    events = client.q.records(tmp_path/'events.jsonl')
    closed = next(r for r in events if r['kind']=='admission_closed')
    drained = next(r for r in events if r['kind']=='drain_finished')
    assert closed['at'] >= queue.end
    assert drained['at'] > queue.end
    assert any(r['kind']=='released' and r['at'] > queue.end for r in events)
    assert not any(r['kind']=='admitted' and r['admitted_at'] >= queue.end for r in events)


def test_failure_publishes_before_cleanup_and_never_refunds_unreleased_history(tmp_path):
    queue = client.WindowQueue(rows(8),{str(i):1 for i in range(8)},1,slots=1,warmup=1,seconds=.01)
    called = []
    def fail(row):
        called.append(row['id'])
        raise RuntimeError('real stream failed')
    with pytest.raises(RuntimeError,match='real stream failed'):
        queue.run(fail,lambda *args:pytest.fail('failed request cannot release'),client.Recorder(tmp_path))
    assert called == ['0'] and queue.reserved == queue.inflight == 1
    assert (tmp_path/'client-failure.json').exists()
    with pytest.raises(ValueError,match='No queue reuse'):
        queue.run(fail,lambda *args:None,client.Recorder(tmp_path))


def publication(sequence,at,submitted,completed):
    return dict(schema_version=1,counter_id='worker',sequence=sequence,at=at,
        submitted=dict(public_prefill_tokens=submitted),completed=dict(public_prefill_tokens=completed),
        completed_fence=dict(completion_lower_bound=at-.2,completion_upper_bound=at))


def test_completed_work_uses_original_publications_and_exposes_boundary_uncertainty():
    source = [publication(1,9,100,80),publication(2,11,130,110),
              publication(3,309,600,580),publication(4,311,650,620)]
    report = client.public_prefill_evidence(source,10,310)
    assert report['completed_observed_delta'] == 500
    assert report['completed_observed_tokens_per_second'] == 500/300
    assert report['completed_within_window_lower_bound'] == 450
    assert report['completed_within_window_upper_bound'] == 570
    assert report['boundary_brackets_complete']
    assert report['boundary_snapshots']['before_end']['completed_fence']['completion_upper_bound'] == 309
    missing = client.public_prefill_evidence(source[1:-1],10,310)
    assert missing['completed_observed_delta'] is None
    assert missing['completed_within_window_upper_bound'] is None
    assert not missing['boundary_brackets_complete']


@pytest.mark.parametrize('damage',['counter_reset','watermark','duplicate'])
def test_public_work_identity_and_publication_monotonicity_are_required(damage):
    source = [publication(1,9,100,80),publication(2,11,130,110)]
    if damage=='counter_reset': source[1]['counter_id']='another worker'
    elif damage=='watermark': source[1]['completed']['public_prefill_tokens']=20
    else: source[1]['sequence']=1
    with pytest.raises(ValueError): client.public_prefill_evidence(source,10,310)


def test_unique_frontier_lower_bound_caps_inputs_and_excludes_carry_in_owners():
    cases = dict(old=dict(response_id='cmpl-old',input_tokens=100),new=dict(response_id='cmpl-new',input_tokens=100))
    states = dict(old=dict(admitted_at=1),new=dict(admitted_at=11))
    samples = [dict(read_started_at=12,at=12.1,progress=dict(accepted_history={'cmpl-old-0':60,'cmpl-new-0':20})),
               dict(read_started_at=309,at=309.1,progress=dict(accepted_history={'cmpl-old-0':130,'cmpl-new-0':120}))]
    report = client.frontier_evidence(samples,cases,states,10,310)
    assert report['unique_input_tokens'] is None
    assert report['unique_input_lower_bound'] == 100
    assert report['owners']['old']['observed_frontier_delta'] == 40
    assert report['owners']['old']['proved_unique_lower_bound'] == 0
    samples[-1]['progress']['accepted_history']['cmpl-new-0']=10
    assert client.frontier_evidence(samples,cases,states,10,310)['unique_input_lower_bound'] is None


def test_missing_native_to_source_mapping_does_not_invent_unique_input():
    sample = dict(read_started_at=12,at=12.1,progress=dict(accepted_history={'opaque-native-owner':100}))
    report = client.frontier_evidence([sample],{'a':dict(response_id='cmpl-a',input_tokens=100)},
        {'a':dict(admitted_at=11)},10,310)
    assert report['unique_input_tokens'] is report['unique_input_lower_bound'] is None


def test_stale_progress_carries_source_publication_time_not_read_time(tmp_path,monkeypatch):
    path = tmp_path/'progress.json'
    save(path,dict(accepted_history={'cmpl-owner-0':100}))
    os.utime(path,(100.,100.))
    monkeypatch.setattr(client.time,'time',lambda:250.)
    value,source = client.read_observation(path)
    assert source['source_published_wall_time']==100.
    assert source['age_seconds']==150. and source['stale']
    assert source['source_monotonic_at'] is None
    sample = dict(read_started_at=12,at=12.1,progress=value,source_observations=dict(progress=source))
    report = client.frontier_evidence([sample],{'owner':dict(response_id='cmpl-owner',input_tokens=100)},
        {'owner':dict(admitted_at=1)},10,310)
    owner = report['owners']['owner']
    assert owner['stale_observations']==1 and owner['source_time_unavailable']==0
    assert owner['observed_frontiers'][0]['source']['source_published_wall_time']==100.
    assert owner['proved_unique_lower_bound']==0


@pytest.mark.parametrize('usage_count',[2,3])
def test_sse_counts_actual_accepted_ids_and_rejects_native_usage_disagreement(tmp_path,monkeypatch,usage_count):
    events = [dict(id='cmpl-session',choices=[dict(index=0,token_ids=[11,22],text='answer',finish_reason='stop')]),
              dict(id='cmpl-session',choices=[],usage=dict(prompt_tokens=2,completion_tokens=usage_count))]
    body = b''.join(b'data: '+json.dumps(e).encode()+b'\n' for e in events)+b'data: [DONE]\n'
    monkeypatch.setattr(client.urllib.request,'build_opener',lambda *args:SimpleNamespace(open=lambda *args,**kwargs:io.BytesIO(body)))
    call = lambda:client.stream(dict(served_model_name='model',port=1),'source','session',[1,2],[0,1],
        dict(max_tokens=8192,seed=1000),1,client.Recorder(tmp_path))
    if usage_count != 2:
        with pytest.raises(ValueError,match='native usage'):call()
    else:
        result = call()
        journal = client.q.records(tmp_path/'events.jsonl')
        assert result['output_ids']==[11,22]
        assert [r['token_ids'] for r in journal if r['kind']=='accepted_tokens']==[[11,22]]


@pytest.mark.parametrize('token_times,expected', [((9,311),0),((10,310),1)])
def test_no_output_in_window_is_valid_and_boundaries_are_half_open(tmp_path,token_times,expected):
    capture,output = tmp_path/'capture',tmp_path/'output'
    events = [dict(kind='accepted_tokens',at=token_times[0],source_request='one',token_ids=[1]),
              dict(kind='accepted_tokens',at=token_times[1],source_request='one',token_ids=[2]),
              dict(kind='released',at=312,source_request='one')]
    lines(output/'events.jsonl',events)
    observations = [dict(read_started_at=t,at=t+.1,progress=dict(accepted_history={'cmpl-one-0':n}),
        public_work=publication(i,t,n,n),archive=dict(seal_direct_bytes=0),nvml=dict(gpu_used_mib=10,
            gpu_free_mib=20,gpu_utilization_percent=90,gpu_power_w=500)) for i,(t,n) in enumerate(((11,10),(309,100)),1)]
    lines(output/'observations.jsonl',observations)
    lines(capture/'public_work.jsonl',[publication(1,9,100,100),publication(2,11,200,180),
        publication(3,309,900,880),publication(4,311,950,920)])
    lines(capture/'model_step_latency.jsonl',[dict(started_at=12,prefill_tokens=100,decode_requests=0,
        prefill_requests=8,model_ms=20),dict(started_at=312,prefill_tokens=0,decode_requests=8,model_ms=1)])
    cases = {'one':dict(response_id='cmpl-one',input_tokens=100,output_tokens=2,output_ids_sha256=client.digest([1,2]))}
    report = client.measurements(capture,output,cases,SimpleNamespace(begin=10,end=310,states={'one':dict(admitted_at=1)}))
    assert report['accepted_output_tokens']==expected
    assert report['accepted_output_tokens_per_second']==expected/300
    assert report['completed_requests']==0 and report['native']['pure_decode_steps']==0
    assert report['native']['pure_prefill_gpu_tokens_per_second']==5000
    assert report['native']['actual_batch_peak']==8
    assert report['zero_output_is_valid']


def test_repeated_input_aliases_are_original_bytes_with_distinct_source_identities(tmp_path):
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    original = []
    for i in range(8):
        path = inputs/f'{i}.i32'
        path.write_bytes(bytes((i,0,0,0)))
        original.append(dict(id=str(i),served_sha256=control.sha(path),seed=1000,served_query=[0,1]))
    fixture = dict(rows=original,input_directory=str(inputs),sampling=dict(temperature=1.,max_tokens=8192))
    frozen = control.repeat_inputs(fixture,tmp_path,32)
    assert len({r['source_request'] for r in frozen['rows']})==32
    assert frozen['sampling']==fixture['sampling']
    for row in frozen['rows']:
        path = Path(frozen['input_directory'])/(row['id']+'.i32')
        assert path.read_bytes()==(inputs/(row['original_case']+'.i32')).read_bytes()
        assert row['seed']==1000 and row['served_query']==[0,1]
    assert control.job_order(3)==[dict(id=f'p{i}-{role}',pair=i,role=role) for i in (1,2,3)
        for role in (('candidate','baseline') if i==2 else ('baseline','candidate'))]


def test_zero_baseline_rate_remains_null():
    assert control.ratio(0,0) is None
    assert control.ratio(5,0) is None
    assert control.ratio(None,1) is None
    assert control.ratio(0,1)==0


@pytest.mark.parametrize('candidate',list(control.pairs.CANDIDATES))
def test_pair_comparison_preserves_zero_output_and_requires_observed_candidate(tmp_path,candidate):
    manifest = dict(pairs=1,candidate=candidate)
    for role in ('baseline','candidate'):
        directory = tmp_path/f'p1-{role}'/'qualification'
        metrics = dict(schema=client.SCHEMA,raw_cache_exercised=role=='candidate',
            direct_seal_bytes_observed_delta=10 if role=='candidate' else 0,
            accepted_output_tokens_per_second=0,native=dict(actual_batch_peak=8),
            public_prefill=dict(boundary_brackets_complete=True,
                completed_observed_tokens_per_second=100 if role=='baseline' else 200))
        save(directory/'measurements.json',metrics)
        save(directory/'results.json',dict(status='passed',schema=client.SCHEMA,
            measurements_sha256=control.sha(directory/'measurements.json'),cases={
                'same':dict(input_sha256='same input',seed=1000,original_case='original',input_tokens=262101,
                    output_ids_sha256='same output',selection_sha256='same selection')}))
    report = control.comparison(tmp_path,manifest)
    pair = report['pairs'][0]
    assert pair['accepted_output_rate_ratio'] is None
    assert pair['completed_public_prefill_observed_rate_ratio']==2
    assert pair['verdict']==('inconclusive' if candidate=='async-action-metadata' else 'descriptive_fixed_window')
    assert not report['performance_gain_established']
    path = tmp_path/'p1-candidate/qualification/results.json'
    changed = control.read(path)
    changed['cases']['same']['selection_sha256']='drift'
    save(path,changed)
    assert control.comparison(tmp_path,manifest)['pairs'][0]['verdict']=='inconclusive'


@pytest.mark.parametrize('status',['running','failed','completed_pending_review'])
def test_started_attempt_is_not_retried_or_allowed_to_start_gpu(tmp_path,monkeypatch,status):
    save(tmp_path/'status.json',dict(status=status))
    monkeypatch.setattr(control,'validate',lambda root:dict(candidate='raw-k-cache'))
    monkeypatch.setattr(control,'__file__',str(tmp_path/'frozen/benchmark_graph_v2_stream.py'))
    monkeypatch.setattr(control.subprocess,'Popen',lambda *args,**kwargs:pytest.fail('GPU must not start'))
    with pytest.raises(ValueError,match='No retry'):
        control.run(SimpleNamespace(output=tmp_path,candidate=None))
