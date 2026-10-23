from copy import deepcopy
from pathlib import Path

import pytest

from ruler_guided_pressure_protocol import PROFILE, audit_case, expected_profile, validate_execution
from test_ruler_pooling_protocol import contract


def fixture():
    _,f=contract('max')
    f.update(execution_profile=PROFILE)
    return expected_profile(f['reference_profile']),f


def evidence():
    public=dict(request_id='req',command_id=1,selection='query',query_span=[100,102],
        guided_query_tokens=0,query_score_tokens=2,mean_k_backend='qw3_cuda',
        query_history_enabled=False,seed_source='mean_k')
    pressure=dict(request_id='req',command_id=2,selection='pressure',history_begin=64,
        guided_scope='prefill-pressure',guided_context_end=64,public_input_replayed_for_guidance=0,
        guided_published_tokens=2,guided_query_tokens=2,seed_source='guided_query',
        pressure_policy='guided_query',query_score_tokens=2,guided_natural_termination=True,
        guided_fallback=None,selected_tokens=32,score_domain_version='qw3-native-v2')
    graph=[dict(request_id='req',phase='query',aggregation='max',alpha=.2,seed_source='mean_k'),
        dict(request_id='req',phase='pressure',seed_source='guided_query',
             propagation_applied=False,transport_mask_count=0)]
    return [public,pressure],graph


def test_corrected_fixture_and_fixed_diagnostic_prefix_are_valid():
    p,f=fixture();validate_execution(p,f);validate_execution(p,f,diagnostic_cases=8)
    with pytest.raises(ValueError):validate_execution(p,f,diagnostic_gen_budget=65536)


@pytest.mark.parametrize('field,value',[('guided_reselect','boundary'),('guided_query_tokens',0),
    ('graph_aggregation','sum'),('pressure_policy','graph'),('gen_budget',65536)])
def test_configuration_cannot_silently_return_to_old_wrong_test(field,value):
    p,f=fixture();p['kvmem'][field]=value
    with pytest.raises(ValueError):validate_execution(p,f)


def audit(p,refresh,graphs):
    return audit_case(Path('/unused'),'req',p,[100,102],require_pressure=True,
        event_reader=lambda p:refresh if p.name=='bounded_refresh.jsonl' else graphs)


def test_real_generated_pressure_and_original_final_query_are_required():
    p,_=fixture();r,g=evidence()
    assert audit(p,r,g)['seed_sources']==['guided_query','mean_k']


@pytest.mark.parametrize('field,value',[('guided_context_end',66),
    ('public_input_replayed_for_guidance',2),('seed_source','prefill'),
    ('query_score_tokens',0),('guided_natural_termination',False)])
def test_future_input_contamination_or_fake_guidance_is_rejected(field,value):
    p,_=fixture();r,g=evidence();r[1][field]=value
    with pytest.raises(ValueError):audit(p,r,g)


def test_guidance_at_final_query_is_rejected():
    p,_=fixture();r,g=evidence();r[0]['guided_query_tokens']=1
    with pytest.raises(ValueError):audit(p,r,g)


def test_unfinished_query_falls_back_explicitly_and_never_scores_partial_rows():
    p,_=fixture();r,g=evidence()
    r[1].update(guided_published_tokens=0,guided_fallback='unterminated-output',
                seed_source='prefill',query_score_tokens=0)
    g[1]['seed_source']='prefill'
    audit(p,r,g)
    r[1]['query_score_tokens']=1
    with pytest.raises(ValueError):audit(p,r,g)
