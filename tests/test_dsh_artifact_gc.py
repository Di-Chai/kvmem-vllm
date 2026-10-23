import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from diagnostics.dsh_artifact_gc import file_body_tokens, token_digest, verified_candidate


def fixture(tmp_path):
    root = tmp_path/'workspace'
    root.mkdir()
    path = root/'app.js'
    content = ('const message="测试";\n'*20)
    path.write_text(content)
    raw = '<tool_call>\n<function=write>\n<parameter=content>\n'+content+'\n</parameter>\n</function>\n</tool_call>'
    ids = list(raw.encode())
    prefix = [1]*96
    result = f'<path>{path}</path>\n<type>file</type>\n<content>\nCreated file\n</content>'
    response = dict(choices=[dict(message=dict(tool_calls=[dict(id='real-write-id',function=dict(name='write',arguments=json.dumps(dict(file_path=str(path),content=content))))]))])
    out = tmp_path/'bridge'
    directory = out/'owner'/'00001'
    directory.mkdir(parents=True)
    (directory/'response.json').write_text(json.dumps(response))
    (directory/'native.json').write_text(json.dumps(dict(token_ids=ids,text=raw)))
    state = SimpleNamespace(turns=1,tokens=prefix,output=ids)
    rb = len(prefix)+len(ids)+1
    tokens = prefix+ids+[2]*192
    prepared = (tokens,{},dict(spans=[dict(role='tool',index=2,begin=0,end=160,content_begin=0,content_end=160)],
        messages=[{}, {}, dict(tool_call_id='real-write-id')]),None,None,True,{(2,0):rb,(2,160):rb+160})
    body = dict(messages=[dict(role='tool',tool_call_id='real-write-id',content=result)])
    return root,out,state,prepared,body,path


def test_actual_file_equality_has_one_observation_and_no_invented_cold_score(tmp_path):
    root,out,state,prepared,body,path = fixture(tmp_path)
    value = verified_candidate(root,out,'owner',state,prepared,body,lambda i:bytes([i]))
    assert value['importance'] == 1 and value['cold'] is False and value['observations'] == 1
    assert value['prefix_sha256'] == token_digest(prepared[0])
    assert value['relative_path'] == 'app.js'
    assert value['source_begin'] < value['source_end'] <= value['result_begin']
    assert (value['source_end']//32-(value['source_begin']+31)//32) > 0


@pytest.mark.parametrize('case', ['changed_file','failed_result','wrong_id','non_reused','foreign_file','symlink','prefix_changed'])
def test_candidate_fails_closed(tmp_path,case):
    root,out,state,prepared,body,path = fixture(tmp_path)
    if case == 'changed_file':path.write_text('changed')
    if case == 'failed_result':body['messages'][0]['content'] = 'Error writing file'
    if case == 'wrong_id':body['messages'][0]['tool_call_id'] = 'other'
    if case == 'non_reused':prepared = (*prepared[:5],False,prepared[6])
    if case == 'foreign_file':root = root/'subdir';root.mkdir()
    if case == 'symlink':
        target = root/'target.js';path.rename(target);path.symlink_to(target)
    if case == 'prefix_changed':
        prepared[0][0] = 3
        with pytest.raises(ValueError,match='exact accepted prefix'):
            verified_candidate(root,out,'owner',state,prepared,body,lambda i:bytes([i]))
        return
    assert verified_candidate(root,out,'owner',state,prepared,body,lambda i:bytes([i])) is None


def test_literal_body_mapping_keeps_partial_boundary_tokens():
    chunks=[b'<parameter=content>\na',b'bc',b'de',b'f\n</parameter>']
    raw=b''.join(chunks).decode()
    assert file_body_tokens(list(range(4)),raw,'abcdef',lambda i:chunks[i]) == (1,3)
    with pytest.raises(ValueError,match='accepted token bytes'):
        file_body_tokens([0],raw,'abcdef',lambda i:b'wrong')


def test_experiment_profile_is_explicit_and_defaults_remain_conservative():
    from vllm_kvmem.historical_gc import HistoricalGcSettings
    default = HistoricalGcSettings(policy='statistical')
    assert default.policy_params['min_cold_observations'] == 3
    profile = HistoricalGcSettings(policy='statistical',policy_params=dict(min_observations=1,min_cold_observations=0,max_importance=1.0))
    assert profile.policy_params['min_cold_observations'] == 0
    for params in [dict(min_observations=0),dict(min_cold_observations=-1),dict(min_cold_observations=False)]:
        with pytest.raises(ValueError):HistoricalGcSettings(policy='statistical',policy_params=params)
