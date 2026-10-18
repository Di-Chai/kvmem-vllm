"""Fairness and request/evidence gates for the long-input scoring comparison."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
try:
    import benchmark_graph_niah as controller
    import qualify_graph_niah as client
finally:
    sys.path.pop(0)


def base():
    return dict(model='model',tokenizer='tokenizer',execution_mode='compiled_graph',
        draft_method='mtp',draft_tokens=4,kvmem=dict(mean_k_backend='qw3_cuda',
        archive_backend='cpu',request_planning=True))


def test_profiles_differ_only_in_two_declared_scoring_fields():
    configs = controller.profiles(base(),1234)
    controller.validate_profiles(configs)
    for arm,p in configs.items():
        assert p['kvmem']['prefill_attention'] and p['kvmem']['action_attention']
        assert p['kvmem']['select_budget'] == 36864 and p['kvmem']['gen_budget'] == 16384
        assert not p['kvmem']['action_pressure'] and not p['kvmem']['query_history']
        assert p['kvmem']['guided_query_tokens'] == 0


@pytest.mark.parametrize('key,value',[('gpu_memory_utilization',.8),('draft_tokens',3),('prefill_chunk_tokens',2048)])
def test_non_scoring_profile_changes_are_rejected(key,value):
    configs = controller.profiles(base(),1234)
    configs['legacy'][key] = value
    with pytest.raises(ValueError):
        controller.validate_profiles(configs)


@pytest.mark.parametrize('key,value',[('query_history',True),('action_pressure',True),
    ('guided_query_tokens',512),('graph_mode','off'),('action_fraction',.5)])
def test_graph_does_not_silently_run_with_incompatible_settings(key,value):
    cfg = controller.profiles(base(),1234)['multi']
    cfg['kvmem'][key] = value
    with pytest.raises(ValueError):
        client.validate_profile(cfg,'multi')


def test_stream_sends_only_original_input_query_and_frozen_sampling(monkeypatch):
    captured = {}
    events = [dict(id='cmpl-owner',choices=[dict(index=0,text='answer',token_ids=[9,10],finish_reason='stop')],
                   usage=dict(prompt_tokens=3,completion_tokens=2))]
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def __iter__(self):
            yield ('data: '+json.dumps(events[0])+'\n').encode()
            yield b'data: [DONE]\n'
    def opened(request,timeout):
        captured.update(json.loads(request.data))
        return Response()
    monkeypatch.setattr(client.urllib.request,'build_opener',lambda *a:SimpleNamespace(open=opened))
    cfg = dict(port=1234,served_model_name='test')
    result = client.stream(cfg,'owner',[1,2,3],[1,2],dict(max_tokens=8192,seed=1000,
        temperature=1.,top_p=.95,top_k=20,thinking_token_budget=4096),5)
    assert captured['prompt'] == [1,2,3] and captured['kvmem_query_span'] == [1,2]
    assert captured['seed'] == 1000 and captured['thinking_token_budget'] == 4096
    assert not set(captured)&{'expected','needles','gold','answer'}
    assert result['output_ids'] == [9,10]


def test_inference_stage_validates_original_ids_and_query_before_sending(tmp_path,monkeypatch):
    import numpy as np
    inputs=tmp_path/'inputs'
    inputs.mkdir()
    source=inputs/'case.i32'
    np.array([1,2,3],dtype='<i4').tofile(source)
    row=dict(id='case',task='niah_single_2',seed=1000,position_percent=10,
             served_sha256=client.sha(source),served_input_tokens=3,
             served_query=[1,2],question='query')
    fixture=dict(input_directory=str(inputs),sampling=dict(max_tokens=8192),request_timeout_seconds=10)
    profile=dict(kvmem=dict(gen_budget=16384))
    args=SimpleNamespace(output=tmp_path/'output',arm='multi')
    sent={}
    def stream(profile,session,tokens,query,sampling,timeout):
        sent.update(tokens=tokens,query=query,sampling=sampling)
        return dict(response_id='cmpl-test',output_ids=[9])
    monkeypatch.setattr(client,'stream',stream)
    tokenizer=SimpleNamespace(decode=lambda ids,skip_special_tokens:'query' if ids==[2] else 'wrong')
    result,response,tokens=client.infer_case(args,fixture,profile,row,tokenizer)
    assert tokens==sent['tokens']==[1,2,3] and sent['query']==[1,2]
    assert sent['sampling']==dict(seed=1000,max_tokens=8192)
    assert response['response_id']=='cmpl-test' and result['input_sha256']==row['served_sha256']
    assert (args.output/'cases/case/response.json').is_file()
    assert not (args.output/'cases/case/result.json').exists()  # Retirement not yet acknowledged.


def selection_events():
    owner = 'cmpl-owner-0-suffix'
    return dict(bounded_refresh=[dict(request_id=owner,selection='query',query_span=[96,99],
        replay_span=[96,99],
        mean_k_backend='qw3_cuda',query_history_enabled=False,guided_query_tokens=0,
        selected_tokens=36864,action_attention_enabled=True,action_history_pages=0,
        prefill_attention_samples=100,attention_source='prefill',selected_history_ranges=[[0,128]])],
        action_attention=[dict(request_id=owner,samples=2,prefill_samples=100)],
        block_graph=[dict(request_id=owner,mode='multi',observed_sources=100,edges=400,
                          accepted_through=96,added_blocks=1)])


def audited(monkeypatch,events):
    monkeypatch.setattr(client.q,'records',lambda path:events[path.stem])
    monkeypatch.setattr(client.q,'expected_output_samples',lambda *a:2)
    return client.audit_selection(Path('capture'),dict(response_id='cmpl-owner',output_ids=[1,2]),
        dict(served_query=[96,99],needles=[dict(start=10,end=15)]),
        controller.profiles(base(),1234)['multi'],'multi',[1,2,3],SimpleNamespace(token_to_id=lambda x:1))


def test_real_graph_evidence_is_required_and_other_users_are_excluded(monkeypatch):
    events = selection_events()
    events['block_graph'].append({**events['block_graph'][0],'request_id':'cmpl-other-0-suffix'})
    result = audited(monkeypatch,events)
    assert len(result['graph']) == 1 and result['all_needles_selected']


def test_unaligned_query_requires_only_the_aligned_replay_prefix(monkeypatch):
    events = selection_events()
    events['bounded_refresh'][0].update(query_span=[123,127],replay_span=[96,127])
    monkeypatch.setattr(client.q,'records',lambda path:events[path.stem])
    monkeypatch.setattr(client.q,'expected_output_samples',lambda *a:2)
    result = client.audit_selection(Path('capture'),dict(response_id='cmpl-owner',output_ids=[1,2]),
        dict(served_query=[123,127],needles=[dict(start=10,end=15)]),
        controller.profiles(base(),1234)['multi'],'multi',[1]*128,SimpleNamespace(token_to_id=lambda x:1))
    assert result['graph'][0]['accepted_through'] == 96


def released_state():
    return dict(at=11, requests=7, session_owners=[dict(namespace=['other','other'])],
                reservations=['cmpl-other-0-native'],
                cpu_history_admission=dict(lease_views=[dict(key=['session','other','other'],
                                                            request_id='cmpl-other-0-native')]))


def owner_released(state):
    return client.owner_released(state,'owner',dict(response_id='cmpl-owner'),
                                 dict(retirement_ack_ready_at=10))


def test_per_owner_release_does_not_wait_for_unrelated_live_requests():
    assert owner_released(released_state())


@pytest.mark.parametrize('mutation',['old_snapshot','missing_owner_view','missing_history_view',
    'active_owner','retiring_owner','idle_owner','uncached_owner','history_credit','gpu_reservation'])
def test_refill_never_precedes_this_owners_release_fences(mutation):
    state = released_state()
    if mutation == 'old_snapshot':state['at'] = 9
    elif mutation == 'missing_owner_view':state.pop('session_owners')
    elif mutation == 'missing_history_view':state['cpu_history_admission'].pop('lease_views')
    elif mutation.endswith('_owner'):
        state['session_owners'].append(dict(namespace=['owner','owner'],
                                           phase=mutation.removesuffix('_owner')))
    elif mutation == 'history_credit':
        state['cpu_history_admission']['lease_views'].append(dict(
            key=['session','owner','owner'],request_id=None,phase='idle'))
    else:state['reservations'].append('cmpl-owner-0-native')
    assert not owner_released(state)


@pytest.mark.parametrize('mutation',[ 'missing_graph','stale_graph','wrong_query','foreign_feedback','replay_samples'])
def test_invalid_or_ineffective_campaign_is_rejected(monkeypatch,mutation):
    events = selection_events()
    if mutation == 'missing_graph':events['block_graph'] = []
    elif mutation == 'stale_graph':events['block_graph'][0]['accepted_through'] = 95
    elif mutation == 'wrong_query':events['bounded_refresh'][0]['query_span'] = [1,2]
    elif mutation == 'foreign_feedback':events['bounded_refresh'][0]['action_history_pages'] = 1
    else:events['action_attention'][0]['prefill_samples'] = 101
    with pytest.raises(ValueError):
        audited(monkeypatch,events)
