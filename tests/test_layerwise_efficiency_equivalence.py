from copy import deepcopy

import pytest

from run_layerwise_pressure_efficiency import full_prefix_candidate_proof, signature


def evidence():
    armed=dict(event='armed',layers=16,budget=64,gen_budget=32,selected_tokens=64,query_span=[128,130],
        fixed_blocks=1,candidate_blocks=4,actual_method='mean_k_sequential_layerwise')
    entry=dict(selection='query',history_begin=128,selected_tokens=64,query_tokens=2,query_span=[128,130],
        query_score_tokens=2,guided_query_tokens=0,guided_fallback=None,query_history_enabled=False,
        mean_k_backend='qw3_cuda',replay_span=[128,130],mandatory_fit=dict(hard_blocks=[0]))
    layers=[dict(layer=i,selected_starts=[0,64],charged_tokens=64,query_source='same_forward_current_layer',
                 predecessor_layers_completed=i) for i in range(16)]
    return dict(events=[armed],refresh=[entry],layers=layers)


def test_final_query_scope_is_part_of_execution_equivalence():
    original=evidence();baseline=signature(original)
    assert baseline==signature(deepcopy(original))
    for section,key,new in [('events','actual_method','mean_k_global'),
                            ('events','candidate_blocks',3),('refresh','history_begin',160),
                            ('refresh','query_score_tokens',1),('refresh','guided_fallback','legacy')]:
        changed=deepcopy(original);changed[section][0][key]=new
        assert signature(changed)!=baseline


def test_full_prefix_count_and_native_unique_archive_geometry_identify_candidates():
    profile=dict(kvmem=dict(select_budget=64,gen_budget=32,block_tokens=32,repository_gc=False))
    proof=full_prefix_candidate_proof(evidence(),profile)
    assert proof['verified'] and proof['candidate_count']==4


@pytest.mark.parametrize('fault',['gc','missing_candidate','cold','missing_sink','duplicate','budget'])
def test_candidate_proof_rejects_unsupported_or_invalid_domains(fault):
    data=evidence();profile=dict(kvmem=dict(select_budget=64,gen_budget=32,block_tokens=32))
    if fault=='gc':profile['kvmem']['repository_gc']=True
    elif fault=='missing_candidate':data['events'][0]['candidate_blocks']=3
    elif fault=='cold':data['layers'][-1]['selected_starts']=[0,128]
    elif fault=='missing_sink':data['layers'][-1]['selected_starts']=[32,64]
    elif fault=='duplicate':data['layers'][-1]['selected_starts']=[0,0]
    elif fault=='budget':data['events'][0]['budget']=96
    with pytest.raises(ValueError):full_prefix_candidate_proof(data,profile)
