from copy import deepcopy
from pathlib import Path

import pytest

from run_concurrent_layerwise_ruler import audit_shared_events, query_mode, server_command
from run_layerwise_matched_performance import MODES, assert_matched, controls, paired_backend_audit


def base():
    return (dict(rows=[dict(id='case',served_sha256='input',seed=17)],
                 sampling=dict(temperature=1.,top_p=.95,max_tokens=8192),
                 layerwise_workers=1,layerwise_prefill_pressure=True,
                 layerwise_pressure_single_assembly=True,layerwise_pressure_fast_metadata=True),
            dict(model='model',execution_mode='compiled_graph',max_num_seqs=1,
                 kvmem=dict(select_budget=36864,gen_budget=16384,block_tokens=32,sink_blocks=41,
                    recent_blocks=0,pressure_policy='attention',prefill_attention=True,
                    query_score_mode='mean_k',prefill_attention_stride=32,graph_alpha=0.)))


def shared_evidence():
    _, profile = base()
    request_id = 'cmpl-owned-0-request'
    span = [262080,262110]
    common = dict(request_id=request_id,history_begin=262080,selected_tokens=36864,
        selected_history_ranges=[[i,32] for i in range(0,36864,32)],
        guided_fallback=None,cancelled=False,private_forward_steps=0)
    bounds = [dict(common,selection='pressure',attention_source='prefill',prefill_attention_samples=100),
              dict(common,selection='query',query_span=span,query_scoring_method='mean_k')]
    return [],bounds,dict(response_id='cmpl-owned'),dict(served_query=span),profile


def test_control_disables_hooks_instead_of_global_diagnostic_work():
    fixture, profile = base()
    variants = [controls(fixture,profile,mode,i) for i,mode in enumerate(MODES)]
    assert_matched(variants)
    commands = [server_command(Path('/root'),Path('/attempt'),f) for f,_ in variants]
    assert not any('layerwise' in arg for arg in commands[0])
    assert '--layerwise-query-diagnostic' in commands[1]
    assert '--layerwise-prefill-pressure' not in commands[1]
    assert '--layerwise-pressure-single-assembly' in commands[2]
    assert '--layerwise-prefill-pressure' in commands[2]
    assert fixture['layerwise_workers']==1 and profile['max_num_seqs']==1


def test_paired_archive_reads_are_enabled_for_both_layerwise_query_controls():
    fixture,profile=base()
    fixture.update(layerwise_paired_reads=True,layerwise_batched_reads=True,
                   layerwise_packed_assembly=True,layerwise_prepared_index=True)
    variants=[controls(fixture,profile,mode,i) for i,mode in enumerate(MODES)]
    assert_matched(variants)
    commands=[server_command(Path('/root'),Path('/attempt'),f) for f,_ in variants]
    assert '--layerwise-paired-reads' not in commands[0]
    assert all('--layerwise-paired-reads' in command for command in commands[1:])
    assert fixture['layerwise_paired_reads']


def test_buffered_history_execution_is_limited_to_independent_pressure():
    fixture,profile=base();fixture['layerwise_buffered_history']=True
    variants=[controls(fixture,profile,mode,i) for i,mode in enumerate(MODES)]
    assert_matched(variants)
    commands=[server_command(Path('/root'),Path('/attempt'),f) for f,_ in variants]
    assert all('--layerwise-buffered-history' not in command for command in commands[:2])
    assert '--layerwise-buffered-history' in commands[2]


def test_native_history_measurement_requires_actual_updates_in_every_layer():
    from run_layerwise_matched_performance import buffered_backend_audit
    fixture=dict(layerwise_prefill_pressure=True,layerwise_buffered_history=True)
    event=dict(event='pressure_selected',layers=[dict(layer=i,attention_state_backend='native_buffered',
        buffered_commits=104) for i in range(16)])
    assert buffered_backend_audit(fixture,dict(events=[event]))['passed']
    event['layers'][-1]['buffered_commits']=0
    with pytest.raises(ValueError):buffered_backend_audit(fixture,dict(events=[event]))
    event['layers'][-1].update(buffered_commits=104,attention_state_backend='native_dictionary')
    with pytest.raises(ValueError):buffered_backend_audit(fixture,dict(events=[event]))
    with pytest.raises(ValueError):buffered_backend_audit(fixture,dict(events=[]))


@pytest.mark.parametrize('independent',[False,True])
def test_timed_paired_backend_requires_actual_query_and_pressure_reads(independent):
    fixture=dict(layerwise_paired_reads=True,layerwise_prefill_pressure=independent)
    evidence=dict(layers=[dict(layer=i,assembly_backend={'paired_read_chunks':5}) for i in range(16)],
        refresh=[dict(selection='pressure',assembly_read_stats={'layerwise_paired_read_chunks':85 if independent else 0})])
    assert paired_backend_audit(fixture,evidence)['passed']
    bad=deepcopy(evidence);bad['layers'][-1]['assembly_backend']['paired_read_chunks']=0
    with pytest.raises(ValueError,match='Query'):paired_backend_audit(fixture,bad)
    if independent:
        bad=deepcopy(evidence);bad['refresh'][0]['assembly_read_stats']={}
        with pytest.raises(ValueError,match='pressure'):paired_backend_audit(fixture,bad)
        with pytest.raises(ValueError,match='pressure'):
            paired_backend_audit(fixture,dict(evidence,refresh=[]))


@pytest.mark.parametrize('drift', ['budget','sampling','input','precision','chunks','workers','scoring'])
def test_hidden_control_drift_is_rejected(drift):
    fixture, profile=base()
    arms=[controls(fixture,profile,mode,i) for i,mode in enumerate(MODES)]
    f,p=arms[-1]
    if drift=='budget':p['kvmem']['gen_budget']=8192
    elif drift=='sampling':f['sampling']['temperature']=0
    elif drift=='input':f['rows'][0]['served_sha256']='other'
    elif drift=='precision':p['kv_cache_dtype']='bf16'
    elif drift=='chunks':p['kvmem']['prefill_chunk_tokens']=2048
    elif drift=='workers':p['max_num_seqs']=4
    else:p['kvmem']['prefill_attention_stride']=16
    with pytest.raises(ValueError):assert_matched(arms)


def test_plain_query_cannot_leak_into_formal_or_layerwise_pressure():
    fixture,profile=base()
    f,_=controls(fixture,profile,'shared_global',0)
    for key,value in [('qualification_only',False),('performance_comparison',False),
                       ('layerwise_prefill_pressure',True),('layerwise_pressure_single_assembly',True)]:
        bad=deepcopy(f);bad[key]=value
        with pytest.raises(ValueError):query_mode(bad)


def test_plain_shared_policy_is_verified_from_native_refresh_evidence():
    assert audit_shared_events(*shared_evidence()) == []


@pytest.mark.parametrize('bad', ['hook','foreign','method','pressure_fallback','missing_pressure',
    'missing_sink','duplicate','future','over_budget','wrong_extent','guided','query','no_samples','private'])
def test_shared_control_rejects_wrong_actual_method_or_scope(bad):
    events,bounds,response,row,profile=shared_evidence()
    if bad=='hook':events.append(dict(event='layer_selected'))
    elif bad=='foreign':bounds[0]['request_id']='cmpl-other-0-request'
    elif bad=='method':bounds[1]['query_scoring_method']='key_fps_4'
    elif bad=='pressure_fallback':bounds[0]['attention_source']='none'
    elif bad=='missing_pressure':bounds=bounds[1:]
    elif bad=='missing_sink':bounds[0]['selected_history_ranges'][0][0]=40000
    elif bad=='duplicate':bounds[0]['selected_history_ranges'][0][0]=32
    elif bad=='future':bounds[0]['selected_history_ranges'][0][0]=262080
    elif bad=='over_budget':bounds[0]['selected_history_ranges'].append([40000,32])
    elif bad=='wrong_extent':bounds[0]['selected_tokens']=36832
    elif bad=='guided':bounds[1]['guided_fallback']='capacity'
    elif bad=='query':bounds[1]['query_span']=[1,2]
    elif bad=='no_samples':bounds[0]['prefill_attention_samples']=0
    else:bounds[0]['private_forward_steps']=1
    with pytest.raises(ValueError):audit_shared_events(events,bounds,response,row,profile)
