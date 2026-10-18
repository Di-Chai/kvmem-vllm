"""CPU-only collection protocol tests; no service or GPU operations."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS=Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    control=importlib.import_module('benchmark_graph_collection')
    client=importlib.import_module('qualify_graph_collection')
finally:
    sys.path.pop(0)


def base():
    return dict(model_format='safetensors',model='model',tokenizer='tokenizer',hf_config_path='model',
        execution_mode='compiled_graph',draft_method='mtp',draft_tokens=4,
        kvmem=dict(mean_k_backend='qw3_cuda',archive_backend='cpu',request_planning=True,
            async_refresh=True,async_restore=True,async_seal=True,direct_pinned_reads=True,
            async_archive_retirement=True))


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_only_graph_mode_changes_and_attention_pressure_remains_enabled():
    configs=control.profiles(base(),1234)
    control.validate_profiles(configs)
    for variant,profile in configs.items():
        kv=profile['kvmem']
        assert kv['graph_mode']==variant
        assert kv['pressure_policy']=='attention'
        assert kv['graph_scoring_version']=='v2' and kv['graph_alpha']==.2
        assert kv['action_attention'] and kv['prefill_attention']
        assert kv['action_fraction']==0 and kv['query_history'] is False
        assert kv['gpu_raw_k_cache_bytes']==0
        assert all(kv[k] is False for k in ('cpu_direct_record_seal','async_action_metadata','native_action_metadata','action_profiling'))
        assert 'action_pressure' not in kv


def test_launch_audit_does_not_require_lazy_action_config_before_first_request(tmp_path,monkeypatch):
    profile=control.profiles(base(),1234)['off']
    root=tmp_path/'off';capture=root/'server/capture'
    identity={'version':'frozen-test'}
    launch=dict(profile=profile,vllm_argv=[],
        environment={'KVMEM_REQUEST_PLAN_CONFIG':json.dumps({'pressure_policy':'attention'})})
    save(root/'expected-launch.json',launch)
    save(root/'server/launch.json',launch)
    save(root/'manifest.json',dict(schema=control.SCHEMA,arm='off',native_identity=identity,
        expected_launch_sha256=control.sha(root/'expected-launch.json')))
    save(capture/'cache-geometry.json',dict(select_budget=36864,gen_budget=16384,
        historical_tokens=524288,cpu_archive_bytes_per_token=34816,cpu_offload_bytes=100000000000))
    monkeypatch.setattr(control,'native_identity',lambda:identity)
    monkeypatch.setattr(client.old,'audit_launch',lambda *args:{'status':'passed'})
    assert not (capture/'action_config.json').exists()
    proof=client.audit_launch(profile,capture,'off')
    assert 'Deferred' in proof['attention_runtime_audit']
    with pytest.raises(FileNotFoundError): client.audit_attention_runtime(capture)
    save(capture/'action_config.json',dict(prefill_attention=True,action_pressure=True,fraction=0,
        scope='non-thinking',async_action_metadata=False,native_action_metadata=False))
    assert client.audit_attention_runtime(capture)['status']=='passed'
    save(capture/'action_config.json',dict(prefill_attention=False))
    with pytest.raises(ValueError,match='attention collection'):
        client.audit_attention_runtime(capture)


@pytest.mark.parametrize('field,value',[('graph_max_total_bytes',2*268435456),('cpu_gather_workers',16)])
def test_pair_cannot_hide_other_graph_or_runtime_changes(field,value):
    configs=control.profiles(base(),1234)
    configs['collect']['kvmem'][field]=value
    with pytest.raises(ValueError,match='Only graph_mode'): control.validate_profiles(configs)


@pytest.mark.parametrize('field,value',[('pressure_policy','graph'),('action_attention',False),
    ('prefill_attention',False),('cpu_direct_record_seal',True),('gpu_raw_k_cache_bytes',4*1024**3),
    ('async_action_metadata',True),('native_action_metadata',True),('action_profiling',True),('async_restore',False)])
def test_collection_does_not_turn_into_a_second_performance_candidate(field,value):
    configs=control.profiles(base(),1234)
    for profile in configs.values(): profile['kvmem'][field]=value
    with pytest.raises(ValueError): control.validate_profiles(configs)


def events(variant='off'):
    profile=control.profiles(base(),1234)[variant]
    query=[300,310]
    common=dict(history_begin=320,selected_tokens=32,query_tokens=32,selected_history_ranges=[[0,32]],
        mandatory_fit=dict(hard_blocks=[0]),raw_cache_hit_tokens=0,attention_source='prefill')
    selected=dict(common,selection='query',query_span=query,replay_span=[288,320],mean_k_backend='qw3_cuda',
        query_history_enabled=False,guided_query_tokens=0,action_attention_enabled=True,
        action_history_pages=0,action_fraction=0,prefill_attention_samples=10)
    pressure=dict(common,selection='pressure',query_span=None)
    action=dict(samples=0,prefill_samples=10)
    graph=[]
    if variant=='collect':
        version=dict(scoring_version='v2',score_domain_version='qw3-native-v2',alpha=.2,pressure_policy='attention')
        selected.update(version,phase='query',seed_source='mean_k')
        pressure.update(version,phase='pressure',seed_source='prefill',score_mask_count=10,transport_mask_count=0)
        graph=[dict(version,phase='query',seed_source='mean_k',mode='collect',selection_unchanged=True,
            graph_live_bytes=100,graph_global_bytes=200,snapshot_bytes=100,accepted_through=320,observed_sources=10,edges=20),
            dict(version,phase='pressure',seed_source='prefill',mode='collect',propagation_applied=False,
                graph_live_bytes=100,candidate_count=10,score_mask_count=10,transport_mask_count=0,selected_tokens=32)]
        action['block_graph']=dict(prefill_samples=10,output_samples=0,observed_sources=10,total_bytes=200)
    return profile,[selected,pressure],graph,action,query


def test_off_audit_accepts_absent_graph_fields_and_does_not_invent_a_seed():
    profile,refresh,graph,action,query=events()
    assert not {'phase','scoring_version','seed_source','alpha'} & refresh[0].keys()
    proof=client.audit_events(refresh,graph,action,profile,'off',query)
    assert proof['graph']==[] and proof['graph_evidence']==dict(expected=False,observed=False)
    assert proof['aligned_domain_event_observed'] is False
    assert proof['pressure_seed_evidence']['exact_runtime_seed_observed'] is False
    assert proof['pressure_seed_evidence']['observed_attention_source_fields']==['prefill']
    assert 'seed_source' not in proof['pressure'][0]


def test_collect_requires_actual_graph_collection_and_unchanged_selection():
    profile,refresh,graph,action,query=events('collect')
    proof=client.audit_events(refresh,graph,action,profile,'collect',query)
    assert proof['pressure_seed_evidence']['sources']==['prefill']
    graph[0]['selection_unchanged']=False
    with pytest.raises(ValueError,match='preserve'): client.audit_events(refresh,graph,action,profile,'collect',query)


@pytest.mark.parametrize('failure',['event','state','query','inherited','overbudget','raw_cache','mandatory','no_pressure'])
def test_off_still_enforces_real_attention_selection_budget_and_no_graph(failure):
    profile,refresh,graph,action,query=events()
    if failure=='event': graph.append(dict(phase='query'))
    elif failure=='state': action['block_graph']={}
    elif failure=='query': refresh[0]['query_span']=[1,2]
    elif failure=='inherited': refresh[0]['action_history_pages']=1
    elif failure=='overbudget': refresh[1]['selected_tokens']=36896
    elif failure=='raw_cache': refresh[1]['raw_cache_hit_tokens']=32
    elif failure=='mandatory': refresh[1]['mandatory_fit']={'hard_blocks':[1]}
    else: refresh.pop()
    with pytest.raises(ValueError): client.audit_events(refresh,graph,action,profile,'off',query)


def test_off_has_no_global_graph_budget_and_both_variants_release_graph_bytes():
    configs=control.profiles(base(),1234)
    archive=dict(archive_backend='cpu',repository_gc=False,disk_read_bytes=0,disk_write_bytes=0,
        nvme_capacity_bytes=0,block_graph_bytes=0,block_graph_limit_bytes=0)
    client.archive_contract(archive,'off',configs['off'])
    archive['block_graph_limit_bytes']=268435456
    client.archive_contract(archive,'collect',configs['collect'])
    with pytest.raises(ValueError): client.archive_contract(archive,'off',configs['off'])
    archive['block_graph_bytes']=1
    with pytest.raises(ValueError): client.archive_contract(archive,'collect',configs['collect'])


def comparison_fixture(root):
    cases={str(i):dict(input_sha256=str(i),input_tokens=262144,seed=1000,query_span=[262000,262100],
        output_ids_sha256='same-output',selection_sha256='same-selection',attention_samples={'samples':2,'prefill_samples':10}) for i in range(8)}
    for variant,wall in (('off',100.),('collect',110.)):
        save(root/variant/'qualification/collection.json',dict(schema=control.SCHEMA,variant=variant,
            cases=cases,wall_seconds=wall,collection_exercised=variant=='collect',off_has_no_graph_state=variant=='off',
            actual_concurrency=dict(executed_request_peak=8)))
        save(root/variant/'nvml-summary.json',dict(samples=100))
        save(root/variant/'qualification/results.json',dict(status='passed',all_owned_sessions_released=True,
            fixture_sha256='same-fixture',collection_sha256=control.sha(root/variant/'qualification/collection.json')))


def test_pair_reports_only_descriptive_cold_collection_cost(tmp_path):
    comparison_fixture(tmp_path)
    result=control.comparison(tmp_path)
    assert result['verdict']=='comparable_cold_collection_smoke'
    assert result['observed_collection_extra_wall_seconds']==10.
    assert result['raw_wall_ratio_collect_over_off']==1.1
    assert result['performance_gain_established'] is False


@pytest.mark.parametrize('failure',['output','selection','attention','unobserved','batch7'])
def test_drift_or_missing_measurement_is_inconclusive_without_retry(tmp_path,failure):
    comparison_fixture(tmp_path)
    path=tmp_path/'collect/qualification/collection.json'; data=json.loads(path.read_text())
    if failure=='output': data['cases']['0']['output_ids_sha256']='different'
    elif failure=='selection': data['cases']['0']['selection_sha256']='different'
    elif failure=='attention': data['cases']['0']['attention_samples']['samples']=3
    elif failure=='unobserved': data['collection_exercised']=False
    else: data['actual_concurrency']['executed_request_peak']=7
    save(path,data)
    results=tmp_path/'collect/qualification/results.json'
    binding=json.loads(results.read_text()); binding['collection_sha256']=control.sha(path); save(results,binding)
    report=control.comparison(tmp_path)
    assert report['verdict']=='inconclusive' and report['inconclusive_reasons']
    assert report['observed_collection_extra_wall_seconds'] is None


@pytest.mark.parametrize('status',['running','failed','completed'])
def test_started_collection_campaign_cannot_be_reentered(tmp_path,monkeypatch,status):
    save(tmp_path/'status.json',dict(status=status))
    for name in control.VARIANTS: save(tmp_path/name/'status.json',dict(status='prepared'))
    monkeypatch.setattr(control,'validate',lambda root:{})
    monkeypatch.setattr(control,'__file__',str(tmp_path/'frozen/benchmark_graph_collection.py'))
    with pytest.raises(ValueError,match='cannot be retried'): control.run(SimpleNamespace(output=tmp_path))
