import json
from io import BytesIO

import pytest

from scripts.terminalbench_dashboard import CounterWindow, Dashboard, GaugeWindow, generation_counter, gpu_view, make_handler, parse_metrics


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


def campaign(tmp_path):
    write(tmp_path/'experiment-plan.json', dict(concurrency=8,budgets=dict(K=36864,G=16384)))
    write(tmp_path/'pair-status.json', dict(status='running',arm='thinking',started_at=80))
    for arm in ('thinking','lru'):
        write(tmp_path/arm/'manifest.json', dict(launched_task_names=['a','b']))
        write(tmp_path/arm/'status.json', dict(status='formal' if arm=='thinking' else 'prepared'))
    root=tmp_path/'thinking'
    write(root/'formal-boundary.json', dict(at=100,capture=dict(archive=dict(historical_gc=dict(
        retired=100,redundancy_retired=30,lru_fallback_retired=70)))))
    write(root/'model/server/capture/progress.json', dict(historical_gc=dict(
        retired=107,redundancy_retired=32,lru_fallback_retired=75)))
    write(root/'model/server/capture/public_work.json',dict(counter_id='first',wall_time=110,
        completed=dict(public_prefill_tokens=1000,public_non_prefill_tokens=999999)))
    lines(root/'formal-events/events.jsonl',[
        dict(at=101,event='start',task='terminal-bench/a',trial='a__trial'),
        dict(at=102,event='agent-start',task='terminal-bench/a',trial='a__trial'),
        dict(at=99,event='agent-start',task='terminal-bench/b',trial='preflight')])
    lines(root/'bridge/requests.jsonl',[
        dict(at=90,event='session_opened',owner='smoke',session_id='smoke__agent'),
        dict(at=95,owner='smoke',request_id='smoke-request',output_tokens=5000),
        dict(at=102,event='session_opened',owner='owner-a',session_id='a__trial__agent'),
        dict(at=108,owner='owner-a',request_id='request-a',output_tokens=123,content='PRIVATE CHAT',control_token='SECRET'),
        dict(at=108,owner='owner-a',request_id='request-a',output_tokens=123)])
    return root


def test_generation_uses_only_single_accepted_output_counter():
    raw='''# HELP ignored
vllm:prompt_tokens_total{engine="0"} 9999999
vllm:generation_tokens_total{engine="0"} 1234
vllm:spec_decode_num_draft_tokens_total 99999
'''
    assert generation_counter(raw)==1234
    assert generation_counter(raw+'vllm:generation_tokens_total{engine="1"} 300\n') is None
    assert generation_counter('vllm:generation_tokens_total NaN') is None
    assert generation_counter('vllm:prompt_tokens_total 200') is None


def test_native_metrics_keep_model_requests_separate_from_generation_and_trials():
    assert parse_metrics('''vllm:generation_tokens_total{engine="0"} 1200
vllm:num_requests_running{engine="0"} 4
vllm:num_requests_waiting{engine="0"} 0
vllm:prompt_tokens_total 99999''')==dict(generation=1200,running=4,waiting=0)
    assert parse_metrics('vllm:generation_tokens_total 100')['running'] is None


def test_task_twelve_and_model_eight_are_distinct_configuration_fields(tmp_path):
    campaign(tmp_path)
    write(tmp_path/'experiment-plan.json',dict(concurrency=12,model_concurrency=8))
    view=Dashboard(tmp_path,metrics_fetcher=lambda:0,clock=lambda:110).sample()
    config=view['campaign']['config']
    assert config['task_concurrency']==config['max_concurrency']==12
    assert config['model_concurrency']==8


def test_model_concurrency_falls_back_to_actual_profile_for_legacy_plan(tmp_path):
    campaign(tmp_path)
    for arm in ('thinking','lru'):
        write(tmp_path/arm/'profile.json',dict(max_num_seqs=8))
    config=Dashboard(tmp_path,metrics_fetcher=lambda:0,clock=lambda:110).sample()['campaign']['config']
    assert config['task_concurrency']==8 and config['model_concurrency']==8


def test_remote_worker_staleness_is_independent_of_local_observer(tmp_path):
    campaign(tmp_path)
    now=[110]
    write(tmp_path/'worker-status.json',dict(host='dell@10.1.139.69',last_sync_at=105,
          preserved=72,restarted=6,remote_total=17,container_admission_bytes=48*1024**3,
          control_token='PRIVATE'))
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:now[0])
    dashboard.sample()
    assert not dashboard.snapshot()['stale']
    assert dashboard.snapshot()['campaign']['remote_worker']['container_admission_bytes']==48*1024**3
    assert 'PRIVATE' not in json.dumps(dashboard.snapshot())
    now[0]=181
    dashboard.sample()  # Local observer is fresh, but its remote source is stale.
    assert dashboard.snapshot()['stale']
    assert dashboard.snapshot()['arms'][0]['engine']['running'] is None
    write(tmp_path/'pair-status.json',dict(status='completed',arm='lru'))
    dashboard.sample()
    assert not dashboard.snapshot()['stale']  # Final results no longer need remote polling.


def test_gauge_window_is_time_weighted_clipped_and_reset_for_other_engine():
    window=GaugeWindow()
    window.add(100,2,0,'thinking')
    assert window.view(100,True)['running']==2
    assert window.view(100,True)['running_mean_60s'] is None
    window.add(102,6,1,'thinking')
    window.add(108,4,0,'thinking')
    assert window.view(108,True)['running_mean_60s']==5
    assert window.view(108,True)['waiting_mean_60s']==.75
    assert window.view(108,True)['sample_window_seconds']==8
    assert window.view(122,True)['running'] is None
    window.add(110,8,0,'lru')
    assert window.view(110,True)['running_mean_60s'] is None
    assert window.view(110,False)['running'] is None
    for at in range(112,172,2):window.add(at,4,0,'lru')
    assert window.view(172,True)['running_mean_60s']==4


def test_dashboard_exposes_real_model_concurrency_and_marks_it_stale(tmp_path):
    campaign(tmp_path)
    now=[110]
    metrics=[dict(generation=20000,running=4,waiting=0)]
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:metrics[0],clock=lambda:now[0])
    view=dashboard.sample()
    assert view['arms'][0]['engine']['running']==4
    assert view['arms'][0]['counts']['agents']==1
    assert view['arms'][1]['engine']['running'] is None
    now[0]=112;metrics[0]=dict(generation=20200,running=2,waiting=1)
    view=dashboard.sample()
    assert view['arms'][0]['engine']['running']==2
    assert view['arms'][0]['engine']['waiting']==1
    assert view['arms'][0]['engine']['running_mean_60s']==4
    assert view['arms'][0]['throughput']['generation_tps_60s']==100
    now[0]=140
    assert dashboard.snapshot()['arms'][0]['engine']['running'] is None


def test_memory_wait_event_without_task_name_is_not_environment_setup(tmp_path):
    root=campaign(tmp_path)
    with (root/'formal-events/events.jsonl').open('a') as stream:
        for row in [dict(at=103,event='start',task='terminal-bench/b',trial='b__trial'),
                    dict(at=104,event='waiting_for_memory',trial='b__trial',
                         requested=8*1024**3,reserved=18*1024**3,available=49*1024**3,
                         control_token='PRIVATE')]:
            stream.write(json.dumps(row)+'\n')
    view=Dashboard(tmp_path,metrics_fetcher=lambda:0,clock=lambda:110).sample()
    arm=view['arms'][0]; task=next(t for t in arm['tasks'] if t['name']=='b')
    assert task['phase']=='waiting_memory' and task['memory_wait']['reserved']==18*1024**3
    assert arm['counts']['waiting_memory']==1 and arm['counts']['setup']==0
    assert arm['counts']['executing']==1 and arm['counts']['active']==2
    assert 'PRIVATE' not in json.dumps(view)


def test_memory_wait_transitions_on_later_incremental_events(tmp_path):
    root=campaign(tmp_path)
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:0,clock=lambda:110)
    dashboard.sample()
    with (root/'formal-events/events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(at=105,event='waiting_for_memory',trial='a__trial',
                                    requested=1024,reserved=2048,available=4096))+'\n')
    assert dashboard.sample()['arms'][0]['counts']['waiting_memory']==1
    with (root/'formal-events/events.jsonl').open('a') as stream:
        for row in [dict(at=106,event='waiting_for_memory',trial='unknown',requested=999),
                    dict(at=107,event='environment-start',trial='a__trial')]:
            stream.write(json.dumps(row)+'\n')
    arm=dashboard.sample()['arms'][0]
    assert arm['counts']['waiting_memory']==0 and arm['counts']['setup']==1
    assert arm['tasks'][0]['phase']=='setup' and 'memory_wait' not in arm['tasks'][0]


def test_stopped_memory_wait_never_appears_as_live_work(tmp_path):
    root=campaign(tmp_path)
    with (root/'formal-events/events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(at=105,event='waiting_for_memory',trial='a__trial'))+'\n')
    write(tmp_path/'user-reconfiguration.json',dict(at=112,reason='user_requested_efficiency_optimization'))
    arm=Dashboard(tmp_path,metrics_fetcher=lambda:0,clock=lambda:120).sample()['arms'][0]
    assert arm['counts']['active']==arm['counts']['waiting_memory']==arm['counts']['executing']==0
    assert arm['tasks'][0]['phase']=='interrupted'


def test_user_efficiency_stop_preserves_completed_results_but_excludes_cancellation(tmp_path):
    root=campaign(tmp_path)
    with (root/'formal-events/events.jsonl').open('a') as stream:
        for row in [dict(at=103,event='start',task='terminal-bench/b',trial='b__trial'),
                    dict(at=104,event='agent-start',task='terminal-bench/b',trial='b__trial'),
                    dict(at=110,event='end',task='terminal-bench/b',trial='b__trial'),
                    dict(at=116,event='end',task='terminal-bench/a',trial='a__trial')]:
            stream.write(json.dumps(row)+'\n')
    write(root/'formal-events/status.json',dict(outcomes=[
        dict(task='terminal-bench/b',passed=True,rewards=dict(reward=1),status='scored',exception=None),
        dict(task='terminal-bench/a',passed=False,rewards=None,status='error',exception='CancelledError')]))
    write(tmp_path/'user-reconfiguration.json',dict(at=112,reason='user_requested_efficiency_optimization',
        status='stopped',control_token='PRIVATE'))
    calls=[]
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:calls.append(1),clock=lambda:120)
    value=dashboard.sample()
    assert calls==[]
    assert value['campaign']['status']=='efficiency_optimization'
    assert value['campaign']['active_arm'] is None
    assert value['campaign']['completed']==value['campaign']['passed']==1
    arm=value['arms'][0]
    assert arm['phase']=='stopped'
    assert arm['counts']['errors']==0 and arm['counts']['interrupted']==1
    assert arm['counts']['active']==0
    assert arm['tasks'][0]['phase']=='interrupted'
    assert arm['tasks'][0]['passed'] is None and arm['tasks'][0]['exception'] is None
    assert arm['tasks'][0]['elapsed_seconds']==11
    assert value['arms'][1]['phase']=='not_started'
    assert value['arms'][1]['counts']['queued']==2
    assert value['arms'][1]['tasks'][0]['phase']=='not_started'
    assert 'PRIVATE' not in json.dumps(value)


def test_user_stop_does_not_hide_real_infrastructure_error_before_stop(tmp_path):
    root=campaign(tmp_path)
    with (root/'formal-events/events.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(at=109,event='end',task='terminal-bench/a',trial='a__trial'))+'\n')
    write(root/'formal-events/status.json',dict(outcomes=[dict(task='terminal-bench/a',passed=False,
        rewards=None,status='error',exception='EnvironmentBuildError')]))
    write(tmp_path/'user-reconfiguration.json',dict(at=112,reason='user_requested_efficiency_optimization'))
    value=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:120).sample()
    assert value['arms'][0]['counts']['errors']==1
    assert value['campaign']['completed']==1


def test_cancellation_without_user_stop_is_not_silently_removed(tmp_path):
    root=campaign(tmp_path)
    write(root/'formal-events/status.json',dict(outcomes=[dict(task='terminal-bench/a',passed=False,
        rewards=None,status='error',exception='CancelledError')]))
    write(tmp_path/'user-reconfiguration.json',dict(at=112,reason='unverified_reason'))
    value=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:120).sample()
    assert value['campaign']['status']=='running'
    assert value['arms'][0]['counts']['errors']==1


def test_authorized_switch_shows_cancellation_and_keeps_full_attempt_denominator(tmp_path):
    root=campaign(tmp_path)
    write(root/'status.json',dict(status='completed'))
    write(root/'user-interruption.json',dict(reason='user_requested_switch_to_lru',
        cancelled_trials=['a__trial'],control_token='PRIVATE'))
    write(root/'formal-events/status.json',dict(outcomes=[
        dict(task='terminal-bench/a',trial='a__trial',passed=False,rewards=None,status='error',exception='CancelledError'),
        dict(task='terminal-bench/b',trial='b__trial',passed=True,rewards=dict(reward=1),status='scored',exception=None)]))
    value=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:120).sample()
    arm=value['arms'][0]
    assert arm['counts']['completed']==2 and arm['counts']['passed']==1
    assert arm['counts']['errors']==0 and arm['counts']['interrupted']==1
    task=next(t for t in arm['tasks'] if t['name']=='a')
    assert task['phase']=='interrupted' and task['exception']=='CancelledError'
    assert task['interruption_reason']=='user_requested_switch_to_lru'
    assert 'PRIVATE' not in json.dumps(value)


def test_switch_marker_does_not_hide_an_unrelated_cancelled_trial(tmp_path):
    root=campaign(tmp_path)
    write(root/'user-interruption.json',dict(reason='user_requested_switch_to_lru',cancelled_trials=['b__trial']))
    write(root/'formal-events/status.json',dict(outcomes=[dict(task='terminal-bench/a',trial='a__trial',
        passed=False,rewards=None,status='error',exception='CancelledError')]))
    value=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:120).sample()
    assert value['arms'][0]['counts']['errors']==1 and value['arms'][0]['counts']['interrupted']==0


def test_counter_window_bootstrap_reset_stale_and_formal_only():
    window=CounterWindow()
    window.add(100,100000,'thinking')
    assert window.view(100,True)[0] is None
    window.add(102,100200,'thinking')
    assert window.view(102,True)[0]==100
    assert window.view(120,True)[0] is None
    assert window.view(103,False)[0] is None
    window.add(104,100201,'lru')
    assert window.view(104,True)[0] is None
    window.add(106,5,'lru')
    assert window.view(106,True)[0] is None
    window.add(130,1005,'lru')
    assert window.view(130,True)[0] is None


def test_counter_window_uses_recent_elapsed_window_not_sample_count():
    window=CounterWindow()
    for at in range(0,101,2):
        window.add(at,at*10,'engine')
    assert window.view(100,True)==(10,100,60,False)


def test_dashboard_excludes_preflight_deduplicates_ledger_and_preserves_unknown(tmp_path):
    root=campaign(tmp_path)
    now=[110]
    counter=[20000]
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:counter[0],clock=lambda:now[0])
    value=dashboard.sample()
    arm=value['arms'][0]
    assert arm['counts']['agents']==1
    assert arm['counts']['queued']==1
    assert arm['throughput']['generation_tps_60s'] is None
    assert arm['throughput']['output_tokens']==123
    assert arm['gc']['retired']==7
    assert arm['gc']['redundancy_retired']==2
    assert arm['tasks'][0]['gc_retired'] is None
    assert value['arms'][1]['throughput']['output_tokens'] is None
    assert 'SECRET' not in json.dumps(value) and 'PRIVATE CHAT' not in json.dumps(value)
    now[0]=112;counter[0]=20200
    write(root/'model/server/capture/public_work.json',dict(counter_id='first',wall_time=112,
        completed=dict(public_prefill_tokens=1100,public_non_prefill_tokens=99999999)))
    value=dashboard.sample()
    assert value['arms'][0]['throughput']['generation_tps_60s']==100
    assert value['arms'][0]['throughput']['prefill_tps_60s']==50
    assert value['arms'][0]['throughput']['output_tokens']==123
    assert value['arms'][1]['throughput']['generation_tps_60s'] is None
    now[0]=140
    assert dashboard.snapshot()['stale']
    assert dashboard.snapshot()['arms'][0]['throughput']['generation_tps_60s'] is None


def test_formal_outcomes_and_phase_switch_do_not_poll_previous_model(tmp_path):
    root=campaign(tmp_path)
    calls=[]
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:calls.append(1) or 100,clock=lambda:110)
    dashboard.sample()
    write(root/'formal-events/status.json',dict(outcomes=[dict(task='terminal-bench/a',
        passed=False,rewards=None,status='agent_timeout',exception='AgentTimeoutError')]))
    write(tmp_path/'pair-status.json',dict(status='running',arm='lru'))
    value=dashboard.sample()
    assert len(calls)==1
    assert value['arms'][0]['counts']['timeouts']==1
    assert value['arms'][0]['tasks'][0]['phase']=='timeout'
    assert value['arms'][0]['throughput']['generation_tps_60s'] is None


def test_gpu_window_uses_formal_samples_and_caps_unobserved_gaps():
    rows=[dict(at=90,devices=[dict(gpu_utilization_percent=100,memory_used_mib=2)]),
          dict(at=100,devices=[dict(gpu_utilization_percent=20,memory_used_mib=3)]),
          dict(at=110,devices=[dict(gpu_utilization_percent=80,memory_used_mib=4)])]
    view=gpu_view(rows,112,100,True)
    assert view['utilization_percent_60s']==pytest.approx((20*2.5+80*2)/4.5)
    assert view['memory_used_mib']==4
    assert gpu_view(rows,130,100,True)['stale']
    assert gpu_view(rows,112,None,True)['utilization_percent_60s'] is None


@pytest.mark.parametrize('path,expected', [('/',200),('/api/status',200),('/../pair-status.json',404)])
def test_http_routes_are_read_only_and_allowlisted(tmp_path,path,expected):
    campaign(tmp_path)
    page=tmp_path/'index.html'
    page.write_text('<!doctype html><title>Terminal Bench</title>')
    dashboard=Dashboard(tmp_path,metrics_fetcher=lambda:100,clock=lambda:110)
    dashboard.sample()
    handler=make_handler(dashboard,page).__new__(make_handler(dashboard,page))
    handler.path=path
    handler.wfile=BytesIO()
    responses=[]
    handler.send_response=lambda code:responses.append(code)
    handler.send_error=lambda code:responses.append(code)
    handler.send_header=lambda *_args:None
    handler.end_headers=lambda:None
    handler.do_GET()
    assert responses==[expected]
    if path=='/api/status':
        assert json.loads(handler.wfile.getvalue())['campaign']['active_arm']=='thinking'
    elif path=='/':
        assert handler.wfile.getvalue()==page.read_bytes()
