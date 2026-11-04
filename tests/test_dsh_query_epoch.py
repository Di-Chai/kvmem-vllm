"""Actual user identity stays independent from user-role plugin notices."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.dsh_query_epoch import apply_query_provenance, prepare_with_query_provenance


def fixture():
    body=dict(prompt_cache_key='s',messages=[dict(role='system',content='system'),
        dict(role='user',content='real task'),dict(role='assistant',content='read'),
        dict(role='tool',content='result'),dict(role='user',content='plugin notice')])
    hashes=[hashlib.sha256(s.encode()).hexdigest() for s in ('real task','plugin notice')]
    body['kvmem_query_provenance']=dict(schema_version=1,session_id='s',
        user_messages_sha256=hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest(),
        query_user_ordinal=0,query_message_id='manual-message')
    plan={'kvmem_query_span':[90,100],'kvmem_reselect':'auto'}
    prepared=([1]*120,plan,dict(messages=[{}]*4,spans=[dict(index=0,role='user',begin=10,
        content_begin=20,content_end=30,end=40)]),None,None,True,{(0,10):5,(0,20):9})
    return body,prepared


def test_plugin_notice_keeps_original_query_coordinates_and_token_ids():
    body,p=fixture();tokens=p[0].copy();diag=apply_query_provenance(body,p,required=True)
    assert p[1]['kvmem_query_span']==[5,9] and p[0]==tokens
    assert diag['message_id']=='manual-message'


@pytest.mark.parametrize('damage',['digest','session','ordinal','identity','span'])
def test_invalid_source_proof_fails_before_changing_plan(damage):
    body,p=fixture();before=copy.deepcopy(p[1]);proof=body['kvmem_query_provenance']
    if damage=='digest':body['messages'][1]['content']='changed'
    if damage=='session':proof['session_id']='other'
    if damage=='ordinal':proof['query_user_ordinal']=True
    if damage=='identity':proof['query_message_id']=''
    if damage=='span':p[2]['spans']=[]
    with pytest.raises(ValueError):apply_query_provenance(body,p,required=True)
    assert p[1]==before


def test_auxiliary_call_without_a_real_user_has_no_query_and_legacy_is_explicit():
    body,p=fixture();body['kvmem_query_provenance'].update(query_user_ordinal=None,query_message_id=None)
    apply_query_provenance(body,p,required=True)
    assert 'kvmem_query_span' not in p[1] and p[1]['kvmem_reselect']=='off'
    body.pop('kvmem_query_provenance')
    assert apply_query_provenance(body,p)['status']=='legacy-role-query'
    with pytest.raises(ValueError,match='required'):apply_query_provenance(body,p,required=True)


class CharacterTokenizer:
    def convert_tokens_to_ids(self, token):
        return 999

    def encode(self, text, **kwargs):
        return list(text.encode())

    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])


def planner_with_feedback(*, feedback_size=256):
    from benchmarks.deepswe.claude_bridge import Planner, Trajectory
    from scripts.terminalbench_query import TaskQuery
    from test_deepswe_prefix_reuse import renderer

    def protocol(*, request):
        result = renderer(request=request)
        result['query_index'] = max(i for i, m in enumerate(request['messages']) if m['role'] == 'user')
        return result

    planner = Planner(protocol, CharacterTokenizer(), gen_budget=128)
    state = Trajectory()
    source = TaskQuery('s', 'task', 'solve', 'solve')
    body = dict(prompt_cache_key='s', messages=[dict(role='user', content='solve')])
    body['kvmem_query_provenance'] = source.proof(body['messages'])
    first, _ = prepare_with_query_provenance(planner, body, state, body, required=True)
    planner.commit(state, first, body, dict(content='answer'), [17, 18, planner.end])
    following = dict(prompt_cache_key='s', messages=body['messages'] + [
        dict(role='assistant', content='answer'), dict(role='user', content='X'*feedback_size)])
    following['kvmem_query_provenance'] = source.proof(following['messages'])
    return planner, state, source, first, following


def test_verified_task_query_precedes_legacy_feedback_capacity_check_and_keeps_exact_ids():
    planner, state, _, first, body = planner_with_feedback()
    before = copy.deepcopy(body)
    with pytest.raises(ValueError, match='Query exceeds G replay capacity'):
        planner.prepare(body, state)
    prepared, proof = prepare_with_query_provenance(planner, body, state, body, required=True)
    assert prepared[5] and prepared[0][:len(state.tokens)+len(state.output)] == state.tokens+state.output
    assert prepared[1]['kvmem_query_span'] == first[1]['kvmem_query_span']
    assert prepared[1]['kvmem_query_id'] == first[1]['kvmem_query_id']
    assert proof['status'] == 'verified-user-source' and proof['user_ordinal'] == 0
    assert body == before and state.turns == 1


@pytest.mark.parametrize('damage', ['missing', 'session', 'digest', 'ordinal', 'identity'])
def test_long_feedback_cannot_bypass_validation_with_missing_or_bad_provenance(damage):
    planner, state, _, _, body = planner_with_feedback()
    if damage == 'missing':
        body.pop('kvmem_query_provenance')
        body['defer_validation'] = True  # An arbitrary raw field grants nothing.
    else:
        key, value = {'session': ('session_id', 'other'), 'digest': ('user_messages_sha256', 'bad'),
                      'ordinal': ('query_user_ordinal', True), 'identity': ('query_message_id', '')}[damage]
        body['kvmem_query_provenance'][key] = value
    before = copy.deepcopy(state)
    with pytest.raises(ValueError):
        prepare_with_query_provenance(planner, body, state, body)
    assert state == before


def test_resolved_real_question_still_obeys_full_query_capacity_and_headroom():
    planner, state, _, _, body = planner_with_feedback()
    body['kvmem_query_provenance'].update(query_user_ordinal=1, query_message_id='new-human-source')
    with pytest.raises(ValueError, match='Query exceeds G replay capacity'):
        prepare_with_query_provenance(planner, body, state, body, required=True)
    # This fits G itself, but still lacks the ordinary one-block replay reserve.
    planner, state, _, _, body = planner_with_feedback(feedback_size=100)
    body['kvmem_query_provenance'].update(query_user_ordinal=1, query_message_id='new-human-source')
    with pytest.raises(ValueError, match='pressure headroom'):
        prepare_with_query_provenance(planner, body, state, body, required=True)


def test_off_and_missing_required_sources_remain_distinct():
    planner, state, _, _, body = planner_with_feedback()
    body['kvmem_query_provenance'].update(query_user_ordinal=None, query_message_id=None)
    prepared, source = prepare_with_query_provenance(planner, body, state, body, required=True)
    assert prepared[1]['kvmem_reselect'] == 'off' and 'kvmem_query_span' not in prepared[1]
    assert source['user_ordinal'] is None
    body.pop('kvmem_query_provenance')
    with pytest.raises(ValueError, match='source metadata is required'):
        prepare_with_query_provenance(planner, body, state, body, required=True)
