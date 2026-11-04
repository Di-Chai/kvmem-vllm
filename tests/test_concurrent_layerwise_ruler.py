from copy import deepcopy

import pytest

from run_concurrent_layerwise_ruler import audit_events


def evidence():
    rid='cmpl-owned-0-request'
    layers=[dict(event='layer_selected',request_id=rid,layer=i,predecessor_layers_completed=i,
        query_source='same_forward_current_layer',charged_tokens=36864,selected_tokens=36864) for i in range(16)]
    chosen=list(range(0,36864,32));candidate=chosen+[40000]
    pressure=dict(event='pressure_selected',request_id=rid,actual_method='accumulated_attention_per_layer',
        event_type='prefill_pressure',accepted_history=53184,selected_tokens=36864,fallback_reason=None,
        layers=[dict(layer=i,charged_tokens=36864,selected_tokens=36864,accepted_score_end=53184,
            selected_starts=chosen,candidate_starts=candidate) for i in range(16)])
    bounds=[dict(request_id=rid,selection='pressure'),dict(request_id=rid,selection='query',query_span=[262080,262100])]
    return layers+[pressure],bounds,dict(response_id='cmpl-owned'),dict(served_query=[262080,262100])


def test_concurrent_audit_accepts_only_complete_owner_scoped_layer_evidence():
    args=evidence()
    assert len(audit_events(*args,True))==16


@pytest.mark.parametrize('violation',['foreign_owner','layer_order','future_score','cold_candidate','budget','missing_pressure'])
def test_concurrent_audit_rejects_cross_owner_or_invalid_layer_state(violation):
    events,bounds,response,row=deepcopy(evidence())
    if violation=='foreign_owner':events[0]['request_id']='cmpl-other-0-request'
    elif violation=='layer_order':events[1]['predecessor_layers_completed']=0
    elif violation=='future_score':events[-1]['layers'][3]['accepted_score_end']=54000
    elif violation=='cold_candidate':events[-1]['layers'][4]['candidate_starts']=[]
    elif violation=='budget':events[-1]['layers'][5]['charged_tokens']=36896
    else:events.pop()
    with pytest.raises(ValueError):audit_events(events,bounds,response,row,True)
