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
    (directory/'native.json').write_text(json.dumps(dict(token_ids=ids,text=raw,usage=dict(prompt_tokens=len(prefix)))))
    (directory/'accepted.json').write_text(json.dumps(dict(token_ids=ids)))
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


def test_committed_multi_call_and_updated_artifact_are_supported(tmp_path):
    from diagnostics.dsh_artifact_gc import verified_candidates
    root,out,state,prepared,body,path=fixture(tmp_path)
    p=out/'owner/00001/response.json'
    response=json.loads(p.read_text())
    response['choices'][0]['message']['tool_calls'].append(dict(id='other',function=dict(name='bash',arguments='{}')))
    p.write_text(json.dumps(response))
    body['messages'][0]['content']=body['messages'][0]['content'].replace('Created file','Updated file')
    values=verified_candidates(root,out,'owner',state,prepared,body,lambda i:bytes([i]),set())
    assert len(values)==1
    assert verified_candidates(root,out,'owner',state,prepared,body,lambda i:bytes([i]),{values[0]['span_id']})==[]


def test_restored_authority_matches_current_binding_instead_of_archive_id():
    from diagnostics.dsh_artifact_gc import Controller
    class Object:pass
    runner=Object();authority=Object()
    authority.memory=SimpleNamespace(request_id='original-archive-id')
    authority.history_end=0
    tokens=(1,)*160
    runner._kvmem_capture_runtime=SimpleNamespace(authorities={'current-request':authority})
    runner._kvmem_sessions=SimpleNamespace(bindings={'current-request':SimpleNamespace(request_id='current-request',namespace=('owner','owner'),generation=4,tokens=tokens)})
    plan=dict(span_id='real-file',namespace=['owner','owner'],source_begin=32,source_end=64,result_begin=96,result_end=128,prompt_tokens=160,
        prefix_sha256=token_digest(tokens),source='verified_dsh_write_artifact',importance=1.0,cold=False,observations=1)
    controller=Controller();controller.arm(runner,plan);controller.poll(authority)
    assert controller.states[0]['error'] is None
    assert controller.states[0]['checked_request']=='current-request'
    assert controller.states[0]['generation']==4


@pytest.mark.parametrize('count,protect_source',[(256,False),(233,False),(256,True)])
def test_handoff_uses_native_picker_and_preserves_every_other_record(count,protect_source):
    from diagnostics.dsh_artifact_gc import Controller
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory import RequestMemory
    from test_core import append,Event
    memory=RequestMemory('handoff',KVMemConfig(block_tokens=32,select_budget=256,gen_budget=256,repository_blocks=16,sink_blocks=1,recent_blocks=1))
    spans=append(memory,count)
    records={(s.handle.slot,s.handle.generation):SimpleNamespace(handle=s.handle,history_begin=i*32,count=min(32,count-i*32)) for i,s in enumerate(spans)}
    handles=[s.handle for s in spans]
    candidates={(h.slot,h.generation) for h in handles[2:4]}
    authority=SimpleNamespace(memory=memory,records=records,history_end=count,_verified_artifact_namespace=('owner','owner'))
    controller=Controller();controller.states=[dict(flushed=True,error=None,plan=dict(namespace=['owner','owner']),candidate_handles=[list(k) for k in candidates],events=[])]
    mandatory=[handles[3]] if protect_source else []
    expected=set(records)-candidates
    if protect_source:expected.add((handles[3].slot,handles[3].generation))
    try:
        plan=controller.select_window(authority,mandatory,256)
        memory.attach_completion(Event(True));assert memory.commit_selection()
        assert {(r[0],r[1]) for r in memory._store.working_records()}==expected
        assert memory.config.select_budget==256 and memory._store.runtime_select_budget==0
        assert controller.states[0]['events'][0]['excluded_blocks']==(1 if protect_source else 2)
    finally:
        assert memory.close()


def test_force_query_is_scheduled_below_k_but_auto_and_other_boundaries_are_unchanged():
    from dataclasses import replace
    from vllm_kvmem.request_plan import RequestPlan
    plan=RequestPlan(query=(64,80),block_tokens=32,gen_budget=16384,reselect='auto')
    assert not plan.refresh_required(64,96,0,53248)
    forced=replace(plan,reselect='force')
    assert forced.refresh_required(64,96,0,53248)
    assert not forced.refresh_required(32,96,0,53248)
    assert not forced.refresh_required(96,96,0,53248)


def test_pinned_bridge_accepts_force_only_for_explicit_artifact_experiment():
    from scripts.dsh_bridge import validate_system_pin_budget
    protocol=SimpleNamespace(pin_system=True,system_pin_policy=dict(block_tokens=32,gen_budget=16384,query_replay_budget=128,
        select_budget=36864,sink_blocks=1,recent_blocks=1,rounding_reserve_blocks=2))
    data=dict(prompt=[1]*256,max_tokens=32,kvmem_reselect='force',kvmem_query_mode='agent',kvmem_query_id='real-query',
        kvmem_query_span=[224,240],kvmem_mandatory_parts='none',kvmem_mandatory_spans=[dict(begin=0,end=64,kind='system')])
    with pytest.raises(ValueError,match='verified agent Query'):
        validate_system_pin_budget(data,protocol)
    protocol.artifact_gc_force_allowed=True
    assert validate_system_pin_budget(data,protocol)['declared_select_budget']==36864
