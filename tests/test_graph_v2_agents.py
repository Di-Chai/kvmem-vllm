"""CPU-only checks for fixed8 natural V2 performance campaigns."""
import asyncio
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    control = importlib.import_module('benchmark_graph_v2_agents')
    checks = importlib.import_module('graph_v2_agent_checks')
finally:
    sys.path.pop(0)


def base():
    return dict(model_format='safetensors',model='model',tokenizer='tokenizer',hf_config_path='model',
        execution_mode='compiled_graph',draft_method='mtp',draft_tokens=4,
        kvmem=dict(mean_k_backend='qw3_cuda',archive_backend='cpu',request_planning=True,
            async_archive_retirement=True,async_refresh=True,async_restore=True,cpu_history_coordination=True,
            async_seal=True,direct_pinned_reads=True,native_action_metadata=False))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))


@pytest.mark.parametrize('candidate',list(checks.pairs.CANDIDATES))
def test_agent_profiles_preserve_one_variable_full_history_and_default_false(candidate):
    configs = checks.profiles(base(),1234,candidate)
    checks.validate_profiles(configs,candidate)
    for profile in configs.values():
        assert profile['max_model_len']==262144
        assert profile['kvmem']['session_checkpoint_bytes']==4*1024**3
        assert profile['kvmem']['action_profiling'] is False
        assert not profile['kvmem']['rolling_active_window']
        assert profile['kvmem']['select_budget']==36864
    changed = next(reversed(configs.values()))
    changed['kvmem']['native_action_metadata']=True
    with pytest.raises(ValueError,match='more than one'): checks.validate_profiles(configs,candidate)


@pytest.mark.parametrize('field,value',[
    ('session_checkpoint_bytes',1),('session_capture_terminal',False),('action_profiling',True),
    ('async_restore',False),('active_resident_pressure_only',True),('rolling_active_window',True)])
def test_agent_profile_rejects_history_loss_or_independent_policy_changes(field,value):
    configs=checks.profiles(base(),1234,'raw-k-cache')
    for profile in configs.values(): profile['kvmem'][field]=value
    with pytest.raises(ValueError): checks.validate_profiles(configs,'raw-k-cache')


def test_pair_order_and_conservative_full_eight_history_budget():
    assert [j['role'] for j in checks.job_order(3)]==['baseline','candidate','candidate','baseline','baseline','candidate']
    assert len(checks.job_order(1))==2
    for invalid in (0,2,4,True,1.):
        with pytest.raises(ValueError): checks.job_order(invalid)
    budget=checks.host_budget()
    assert budget['history_limits']==[262144]*8
    assert budget['archive_bytes']==8*(262144+64)*34816
    assert budget['required_available_bytes']==budget['archive_bytes']+8*805306368+32*1024**3


def test_guarded_trace_limit_extension_rejects_any_other_graph_semantics():
    old = ('def record_query_inputs(options, records):\n'
           '    if not options.graph_trace or len(records) > 2048: return\n'
           '    return 262144\n'
           'def record_pressure_inputs(options, records):\n'
           '    if not options.graph_trace or len(records) > 2048: return\n'
           '    return 262144\n')
    new = ('_MAX_REPLAY_NODES = 8192\n_MAX_REPLAY_RECORD_BYTES = 2 * 1024 * 1024\n' +
           old.replace('2048', '_MAX_REPLAY_NODES').replace('262144', '_MAX_REPLAY_RECORD_BYTES'))
    assert checks.trace_export_limit_extension(old,new)['non_trace_ast_identical']
    for changed in (new.replace('8192','16384'),new+'x = _MAX_REPLAY_NODES\n',
                    new.replace('not options.graph_trace','options.graph_trace'),new+'def score(): return 1\n'):
        with pytest.raises(ValueError):checks.trace_export_limit_extension(old,changed)


def qualification(tmp_path):
    profile=next(iter(checks.profiles(base(),1234,'raw-k-cache').values()))
    old=deepcopy(profile)
    old.update(max_model_len=524288,verify_session_restore=True)
    old['kvmem'].update(session_checkpoint_bytes=8*1024**3,action_profiling=True)
    old['kvmem'].pop('cpu_direct_record_seal')
    old['kvmem'].pop('async_action_metadata')
    source='src/vllm_kvmem/options.py'
    (tmp_path/'runtime'/source).parent.mkdir(parents=True)
    (tmp_path/'runtime'/source).write_text('old default false baseline')
    save(tmp_path/'profile.json',old)
    save(tmp_path/'fixture.json',{'fixture':'original qualification'})
    save(tmp_path/'status.json',{'status':'passed'})
    save(tmp_path/'cleanup.json',dict(remaining_owned_process_groups={},remaining_gpu_pids=[],server_returncode=0))
    graph=dict(scoring_version='v2',alpha=.2,pressure_policy='graph')
    evidence=[dict(agent=a,turn=t,query=[graph],pressure=[graph] if not t else [],
        **(dict(target_restore=[dict(byte_equal=True)],state_restore=[dict(recurrent_byte_equal=True)]) if t else {}))
        for a in range(2) for t in range(3)]
    save(tmp_path/'qualification/results.json',dict(status='passed',owners=2,turns=6,errors=[],
        all_owned_sessions_released=True,evidence=evidence))
    save(tmp_path/'manifest.json',dict(synchronous=False,qualification_script='qualify_graph_v2_sessions.py',
        profile_sha256=checks.sha(tmp_path/'profile.json'),fixture_sha256=checks.sha(tmp_path/'fixture.json'),
        runtime_sha256={source:checks.sha(tmp_path/'runtime'/source)}))
    return profile,{source:'new default false extension'}


def test_prior_qualification_normalizes_missing_false_without_claiming_candidate_passed(tmp_path):
    profile,sources=qualification(tmp_path)
    report=checks.qualification_proof(tmp_path,profile,sources,False)
    assert report['passed_prior_lifecycle']
    assert report['candidate_prequalified'] is False and report['current_profile_prequalified'] is False
    assert report['exact_source_match'] is False
    assert 'kvmem.session_checkpoint_bytes' in report['profile_deltas']


@pytest.mark.parametrize('mutation',['byte','release','profile','critical_source','source_tamper'])
def test_qualification_fails_closed_on_missing_proof_or_unqualified_core(tmp_path,mutation):
    profile,sources=qualification(tmp_path)
    if mutation in ('byte','release'):
        path=tmp_path/'qualification/results.json'; value=checks.read(path)
        if mutation=='byte': value['evidence'][1]['state_restore'][0]['recurrent_byte_equal']=False
        else: value['all_owned_sessions_released']=False
        save(path,value)
    elif mutation=='profile': profile['kvmem']['native_action_metadata']=True
    elif mutation=='critical_source': sources['src/vllm_kvmem/authority.py']='unqualified'
    else: (tmp_path/'runtime/src/vllm_kvmem/options.py').write_text('changed after qualification')
    with pytest.raises(ValueError): checks.qualification_proof(tmp_path,profile,sources,False)


def test_fixture_rejects_changed_original_bytes_before_parsing(tmp_path):
    path=tmp_path/'fixture.json'; save(path,{'agent_plans':[]})
    with pytest.raises(ValueError,match='unchanged original'): checks.fixture_proof(path)


def test_actual_worker_payload_is_checked_not_only_requested_configuration(tmp_path):
    profile=checks.profiles(base(),1234,'cpu-direct-record-seal')['seal-copy']
    expected=dict(profile=profile,vllm_argv=['vllm','serve','model'])
    save(tmp_path/'server/launch.json',expected)
    path=tmp_path/'server/capture/cache-geometry.json'
    geometry=dict(select_budget=36864,gen_budget=16384,historical_tokens=524288,
        cpu_archive_bytes_per_token=34816,cpu_offload_bytes=100000000000)
    save(path,geometry)
    proof=checks.actual_profile(tmp_path,expected)
    assert proof['geometry']['historical_tokens']==524288
    assert proof['logical_max_model_len']==262144
    save(path,dict(geometry,historical_tokens=262144))
    with pytest.raises(ValueError,match='geometry'):checks.actual_profile(tmp_path,expected)
    save(path,geometry)
    changed=deepcopy(expected); changed['profile']['kvmem']['cpu_direct_record_seal']=True
    save(tmp_path/'server/launch.json',changed)
    with pytest.raises(ValueError,match='Actual launched'): checks.actual_profile(tmp_path,expected)


def test_missing_counter_is_unobserved_not_measured_zero():
    assert checks.measured_total([{}],'uploaded_bytes')['value'] is None
    assert checks.measured_total([dict(uploaded_bytes=0)],'uploaded_bytes') == dict(
        value=0,observed_rows=1,total_rows=1,observed=True)


def graph_fixture(tmp_path):
    query=[19007,20241]
    fixture=dict(agent_plans=[dict(query_plan=dict(kvmem_query_span=query))])
    rows=[dict(kind='warmup',agent=0,epoch=0,turn=i,request_id=str(i),response_id='cmpl-'+str(i),
        query_plan=dict(kvmem_query_span=query),input_sha256='input'+str(i),output_sha256='output'+str(i)) for i in range(2)]
    graph=dict(mode='multi',scoring_version='v2',score_domain_version='qw3-native-v2',alpha=.2,
        pressure_policy='graph',phase='query',seed_source='mean_k',graph_global_bytes=1000,graph_bytes=500)
    lines(tmp_path/'block_graph.jsonl',[graph])
    lines(tmp_path/'bounded_refresh.jsonl',[dict(request_id='cmpl-0-x',selection='query',query_span=query,
        selected_history_ranges=[[0,32]])])
    lines(tmp_path/'async_restore.jsonl',[dict(request_id='cmpl-1-x',history_tokens=220000,cancelled=False)])
    return fixture,rows


def test_query_stays_at_original_early_position_and_continuation_restores_it(tmp_path):
    fixture,rows=graph_fixture(tmp_path)
    result=checks.graph_evidence(tmp_path,fixture,rows)
    assert result['original_query_positions_preserved']
    assert result['requests'][1]['query_executions']==0
    assert result['requests'][0]['selection_sha256']
    rows[1]['query_plan']['kvmem_query_span']=[219872,219999]
    with pytest.raises(ValueError,match='positions changed'): checks.graph_evidence(tmp_path,fixture,rows)


def test_restored_history_must_contain_original_query(tmp_path):
    fixture,rows=graph_fixture(tmp_path)
    lines(tmp_path/'async_restore.jsonl',[dict(request_id='cmpl-1-x',history_tokens=1024)])
    with pytest.raises(ValueError,match='restore'): checks.graph_evidence(tmp_path,fixture,rows)


def measurement_fixture(tmp_path):
    capture=tmp_path/'server/capture'
    save(tmp_path/'tools/window.json',dict(begin=10.,end=310.,agents=8,warmup_turns=2))
    lines(capture/'model_step_latency.jsonl',[
        dict(started_at=11.,model_ms=1.,decode_requests=7,prefill_requests=0,prefill_tokens=0),
        dict(started_at=12.,model_ms=1.,decode_requests=2,prefill_requests=6,prefill_tokens=6144)])
    lines(capture/'bounded_refresh.jsonl',[dict(refresh_started_at=11.,refresh_finished_at=13.,
        assembly_uploaded_bytes=1024,assembly_host_prepare_seconds=.01,raw_cache_hit_tokens=0)])
    lines(tmp_path/'agent_counters.jsonl',[dict(at=11.,nvml=dict(gpu_free_mib=5000,gpu_used_mib=60000,
        gpu_utilization_percent=95,gpu_power_w=150))])
    save(tmp_path/'tools/traces.json',[dict(kind='workload',agent=i,begin=11.,end=312.) for i in range(8)])
    profile=next(iter(checks.profiles(base(),1234,'raw-k-cache').values()))
    result=dict(output_tokens=3000,output_tokens_per_second=10.,offered_new_input_tokens=300,archive_final={},
        ttft_observed_p50_seconds=1.,ttft_observed_p95_seconds=2.,output_event_gap_p50_seconds=.01,
        output_event_gap_p95_seconds=.02,ttft_right_censored=[],output_gap_right_censored=[])
    return profile,result


def test_measured_accepted_tps_and_actual_pure_decode_are_not_speculative_proposals(tmp_path):
    profile,result=measurement_fixture(tmp_path)
    report=checks.measurements(tmp_path,profile,result)
    assert report['output_tokens_per_second']==10.
    assert report['native']['actual_c8_compute_observed'] is True
    assert report['native']['pure_decode_steps']==1
    assert report['native']['pure_decode_batch_histogram']['7']==1
    assert report['native']['pure_decode_batch_histogram']['8']==0
    assert report['seal']['observed'] is False
    assert report['seal']['seconds']['seal_pack_seconds'] is None
    assert report['cpu_gather']['observed'] is False
    assert report['gpu_idle_seconds'] is None
    assert report['independent_owners']['http_inflight_peak']==8
    assert report['latency']['measurement_complete']
    assert report['latency']['request_p95_seconds']==301.


def test_peak_below_eight_is_measured_not_a_protocol_failure(tmp_path):
    profile,result=measurement_fixture(tmp_path)
    lines(tmp_path/'server/capture/model_step_latency.jsonl',[
        dict(started_at=11.,model_ms=1.,decode_requests=7,prefill_requests=0,prefill_tokens=0)])
    assert checks.measurements(tmp_path,profile,result)['native']['actual_c8_compute_observed'] is False


def test_comparison_never_promotes_unexercised_or_drifting_candidate(tmp_path):
    manifest=dict(pairs=1,jobs=checks.job_order(1),candidate='raw-k-cache')
    for role in ('baseline','candidate'):
        save(tmp_path/f'p1-{role}'/'v2-evidence.json',dict(measured=dict(raw_cache_exercised=False,
            output_tokens=3000,output_tokens_per_second=10.,native=dict(actual_c8_compute_observed=True),
            latency=dict(measurement_complete=True,request_p95_seconds=2.,ttft_p95_seconds=1.,intertoken_event_gap_p95_seconds=.1)),
            graph=dict(requests=[dict(agent=0,epoch=0,turn=0,selection_sha256='same')])))
        save(tmp_path/f'p1-{role}'/'tools/traces.json',[dict(kind='workload',agent=0,epoch=0,turn=0,
            input_sha256='same',output_sha256='same',query_plan={})])
    result=control.comparison(tmp_path,manifest)
    assert result['verdict']=='inconclusive' and result['geometric_mean_usable_ratio'] is None
    path=tmp_path/'p1-candidate/v2-evidence.json'; value=checks.read(path)
    value['measured']['raw_cache_exercised']=True; save(path,value)
    assert control.comparison(tmp_path,manifest)['verdict']=='screening_observation'
    value['graph']['requests'][0]['selection_sha256']='drift'; save(path,value)
    assert control.comparison(tmp_path,manifest)['verdict']=='inconclusive'


@pytest.mark.parametrize('status',['failed','started','loading','running','completed_pending_review'])
def test_started_campaign_cannot_be_reentered(tmp_path,monkeypatch,status):
    save(tmp_path/'status.json',dict(status=status))
    monkeypatch.setattr(control,'ROOT',tmp_path/'frozen')
    monkeypatch.setattr(control,'validate',lambda root:dict(jobs=checks.job_order(1)))
    with pytest.raises(ValueError,match='No retry'):
        asyncio.run(control.run(SimpleNamespace(output=tmp_path,candidate=None)))


def test_cancellation_waits_for_owned_cleanup_and_post_stop(tmp_path,monkeypatch):
    events=[]
    async def launch(*args,**kwargs):
        try: await asyncio.Event().wait()
        finally:
            await asyncio.sleep(.01)
            events.append('owned-stopped')
    def stopped(*args): events.append('post-stop')
    monkeypatch.setattr(control.ready,'run_arm',launch)
    monkeypatch.setattr(control.chain,'post_stop_proof',stopped)
    async def scenario():
        task=asyncio.create_task(control.guarded_arm(tmp_path,'job',{}))
        await asyncio.sleep(.001); task.cancel()
        await asyncio.sleep(.002); task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
    asyncio.run(scenario())
    assert events==['owned-stopped','post-stop']
