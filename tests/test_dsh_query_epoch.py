"""Actual user identity stays independent from user-role plugin notices."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from scripts.dsh_query_epoch import apply_query_provenance


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
