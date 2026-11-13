from pathlib import Path

from run_layerwise_memory import normalize_cuda_alias
import pytest


def test_restore_diagnostic_survives_serving_env_reset_without_changing_model_controls(tmp_path,monkeypatch):
    from copy import deepcopy
    from run_layerwise_memory import restore_verification_profile,server_command
    from vllm_kvmem.serving.serve import build_launch
    from test_serving_launcher import profile
    original=profile(tmp_path);before=deepcopy(original)
    monkeypatch.setenv('KVMEM_VERIFY_SESSION_RESTORE_DIGEST','1')
    # The ambient variable was deliberately ignored: this reproduces the failure.
    normal=build_launch(original,tmp_path,tmp_path/'normal')
    assert 'KVMEM_VERIFY_SESSION_RESTORE_DIGEST' not in normal['environment']
    diagnostic=restore_verification_profile(original)
    verified=build_launch(diagnostic,tmp_path,tmp_path/'verified')
    assert verified['environment']['KVMEM_VERIFY_SESSION_RESTORE_DIGEST']=='1'
    assert verified['environment']['KVMEM_VERIFY_SESSION_RESTORE']=='1'
    assert verified['vllm_argv']==normal['vllm_argv']
    assert original==before
    assert diagnostic==dict(before,verify_session_restore=True,verify_session_restore_digest=True)
    cmd=server_command(tmp_path,dict(restore_digest_qualification=True))
    assert cmd[cmd.index('--config')+1]==str(tmp_path/'verification-profile.json')
    cmd=server_command(tmp_path,{})
    assert cmd[cmd.index('--config')+1]==str(tmp_path/'profile.json')


def test_supplement_uses_new_pair_and_preserves_even_unaudited_saved_response(tmp_path,monkeypatch):
    import run_layerwise_memory as driver
    import recall_comparison_protocol as p
    rows=[dict(id=f'long_q{i:04}',input_tokens=750000+i,token_sha256=str(i)) for i in range(4)]
    largest=dict(id='largest_q0000',input_tokens=757259,token_sha256='largest');rows.append(largest)
    fixture=dict(suite='memoryagentbench',qualification=True,execution_flags={})
    monkeypatch.setattr(driver,'verify',lambda root:(fixture,{}))
    monkeypatch.setattr(driver,'memory_pressure_audit',lambda *args:None)
    for name in ('profile.json','fixture.json','runtime-proof.json','prepared-proof.json','run/server/launch.json'):
        p.save(tmp_path/name,{})
    p.save(tmp_path/'status.json',dict(status='failed'))
    p.save(tmp_path/'run/cleanup.json',dict(gpu_processes=[],remaining_owned_groups=[]))
    for row in rows[:2]+[largest]:
        p.save(tmp_path/'run/answers/cases'/row['id']/'response.json',dict(first_answer=row['id']))
    directory=tmp_path/'run/answers/cases'/largest['id']
    p.save(directory/'result.json',dict(status='completed',token_sha256='largest',original_input_intact=True,
        prompt_tokens=757259,layerwise=dict(events=[],layers=[dict(layer=i) for i in range(16)])))
    p.save(directory/'events.json',[])
    pending,proof=driver.supplemental_qualification(tmp_path,rows)
    assert pending==rows[2:4]
    assert proof['preserved_response_ids']==sorted(r['id'] for r in rows[:2]+[largest])
    assert proof['resampled_previous_answers']==0
    # q1 had a saved answer but no result.json; it must never be generated again.
    assert str(tmp_path/'run/answers/cases/long_q0001/response.json') in proof['files']
    p.save(tmp_path/'run/cleanup.json',dict(gpu_processes=[123],remaining_owned_groups=[]))
    with pytest.raises(ValueError,match='still owns'):driver.supplemental_qualification(tmp_path,rows)


@pytest.mark.parametrize('paired',[False,True])
def test_memory_preserves_qualified_paired_read_backend_through_launch(paired):
    from run_layerwise_memory import inherited_execution,server_command
    origin=dict(layerwise_prefill_pressure=True,layerwise_pressure_single_assembly=True,
        layerwise_pressure_fast_metadata=True,layerwise_packed_assembly=True,
        layerwise_prepared_index=True,layerwise_batched_reads=True,layerwise_paired_reads=paired,
        layerwise_workers=1,rows=[{'id':'ruler-only'}],sampling={'max_tokens':8192})
    execution=inherited_execution(origin)
    command=server_command(Path('/memory'),dict(execution_flags=execution))
    assert ('--layerwise-paired-reads' in command)==paired
    assert '--layerwise-prefill-pressure' in command
    assert '--layerwise-query-optimized' in command
    assert '--layerwise-query-native-batches' in command
    assert not {'rows','sampling','layerwise_workers'} & execution.keys()
    assert '--config' in command and '/memory/profile.json' in command


def test_historical_memory_runtime_does_not_enable_new_backend_implicitly():
    from run_layerwise_memory import inherited_execution,server_command
    execution=inherited_execution({'layerwise_prefill_pressure':True})
    assert execution['layerwise_paired_reads'] is False
    assert '--layerwise-paired-reads' not in server_command(Path('/memory'),dict(execution_flags=execution))


@pytest.mark.parametrize('marker',['process.json','run/server.log'])
def test_memory_run_refuses_to_overwrite_a_previous_attempt(tmp_path,monkeypatch,marker):
    import sys
    import run_layerwise_memory as driver
    monkeypatch.setattr(sys,'path',list(sys.path))
    monkeypatch.setattr(driver,'verify',lambda root:({},{}))
    path=tmp_path/marker;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original attempt')
    with pytest.raises(ValueError,match='first Memory attempt'):driver.run(tmp_path)
    assert path.read_text()=='original attempt'
    assert not (tmp_path/'toolchain-alias-audit.json').exists()


def test_memory_inherits_explicit_buffered_pressure_and_rejects_shared_policy():
    from run_layerwise_memory import inherited_execution,server_command
    flags=inherited_execution(dict(layerwise_prefill_pressure=True,layerwise_buffered_history=True))
    assert '--layerwise-buffered-history' in server_command(Path('/memory'),dict(execution_flags=flags))
    flags['layerwise_prefill_pressure']=False
    with pytest.raises(ValueError):server_command(Path('/memory'),dict(execution_flags=flags))


def test_reuses_same_toolkit_alias_without_changing_runtime_controls(tmp_path):
    installation=tmp_path/'cuda-13.0';installation.mkdir()
    preferred=tmp_path/'cuda';preferred.symlink_to(installation)
    alternate=tmp_path/'cuda-13';alternate.symlink_to(installation)
    environment=dict(CUDA_HOME=str(alternate),PATH='/original/path',MAX_JOBS='4',
                     FLASHINFER_NVCC_THREADS='1',MODEL_CONTROL='unchanged')
    actual=normalize_cuda_alias(environment,preferred)
    assert actual['CUDA_HOME']==str(preferred)
    assert actual['PATH']==str(preferred/'bin')+':/original/path'
    assert {k:v for k,v in actual.items() if k not in ('CUDA_HOME','PATH')}=={
        k:v for k,v in environment.items() if k not in ('CUDA_HOME','PATH')}
    assert environment['CUDA_HOME']==str(alternate)


def test_never_replaces_a_different_toolkit(tmp_path):
    original=tmp_path/'cuda-12';original.mkdir()
    preferred=tmp_path/'cuda-13';preferred.mkdir()
    environment=dict(CUDA_HOME=str(original),PATH='/original/path')
    assert normalize_cuda_alias(environment,preferred)==environment


def test_absent_toolkit_is_not_invented(tmp_path):
    environment=dict(PATH='/original/path')
    assert normalize_cuda_alias(environment,tmp_path/'missing')==environment


def test_recovery_preserves_completed_answers_without_sending_them_again(monkeypatch,tmp_path):
    import json
    import run_layerwise_memory as driver
    import recall_comparison_protocol as p
    root=tmp_path/'campaign';saved=root/'recovered-answers/cases/old';saved.mkdir(parents=True)
    record=dict(id='old',status='completed',usage={'completion_tokens':1},seconds=1.,
                finish_reason='stop',hit_output_cap=False,prefix_reused=False,
                answer='Preserve this answer even when it is incorrect.')
    p.save(saved/'result.json',record)
    p.save(saved/'response.json',{'original_first_response':True})
    profile={'kvmem':{'cpu_read_cache_bytes':32*1024**3}}
    p.save(root/'profile.json',profile)
    fixture=dict(suite='longmemeval_s',rows=[{'id':'old'}],bindings=[],reference=str(tmp_path),
                 source_cohort_sha256='original',case_ids=['old'],qualification=False)
    p.save(root/'recovery-proof.json',dict(recovered_files={
        str(f.relative_to(root/'recovered-answers')):p.sha(f) for f in saved.iterdir()}))
    monkeypatch.setattr(driver,'verify',lambda root:(fixture,profile))
    def must_not_infer(row):
        raise AssertionError('A completed answer was sent for generation again')
    monkeypatch.setattr(driver,'source',must_not_infer)
    driver.infer(root,tmp_path/'unused-capture')
    actual=root/'run/answers/cases/old'
    assert (actual/'response.json').read_bytes()==(saved/'response.json').read_bytes()
    assert json.loads((actual/'result.json').read_text())==record
    audit=p.read(root/'execution-audit.json')
    assert audit['inherited_answers']==1 and audit['regenerated_completed_answers']==0


def prefix_event(kind='pressure_prefix_captured'):
    return dict(event=kind,accepted_history=256000,layer_names=[f'layer{i}' for i in range(16)],
        layer_tokens=[36864]*16,map_sha256='frozen-maps',attention_sha256='frozen-scores',
        actual_method='accumulated_attention_per_layer',fallback_reason=None)


def test_mab_prefix_requires_identical_per_layer_maps_and_scores_across_questions():
    from run_layerwise_memory import memory_prefix_audit
    saved=memory_prefix_audit([prefix_event()],256000,None)
    assert memory_prefix_audit([prefix_event('pressure_prefix_restored'),prefix_event()],256000,saved)==saved
    for field in ['map_sha256','attention_sha256']:
        restored=prefix_event('pressure_prefix_restored');restored[field]='changed'
        with pytest.raises(ValueError,match='changed'):
            memory_prefix_audit([restored,prefix_event()],256000,saved)
    with pytest.raises(ValueError,match='Missing'):
        memory_prefix_audit([prefix_event()],256000,saved)


def test_memory_pressure_audit_accepts_short_context_without_pressure():
    from run_layerwise_memory import memory_pressure_audit
    assert memory_pressure_audit([],[],True)==[]


def test_buffered_memory_pressure_can_use_a_verified_native_prefix_restore():
    from run_layerwise_memory import memory_pressure_audit
    selected=list(range(0,1312,32))
    layers=[dict(layer=i,charged_tokens=1312,selected_tokens=1312,accepted_score_end=50000,
        candidate_starts=selected,selected_starts=selected,attention_state_backend='native_buffered',
        buffered_commits=0,buffered_restores=1) for i in range(16)]
    event=dict(event='pressure_selected',event_type='prefill_pressure',fallback_reason=None,
        actual_method='accumulated_attention_per_layer',accepted_history=50000,selected_tokens=1312,layers=layers)
    bounds=[dict(selection='pressure')]
    assert memory_pressure_audit([event],bounds,True,True)==bounds
    layers[-1]['buffered_restores']=0
    with pytest.raises(ValueError,match='buffered history'):memory_pressure_audit([event],bounds,True,True)


def test_memory_pressure_audit_rejects_shared_fallback_missing_layer_and_future_scores():
    from run_layerwise_memory import memory_pressure_audit
    from copy import deepcopy
    selected=list(range(0,1312,32))
    event=dict(event='pressure_selected',event_type='prefill_pressure',fallback_reason=None,
        actual_method='accumulated_attention_per_layer',accepted_history=50000,selected_tokens=1312,
        layers=[dict(layer=i,charged_tokens=1312,selected_tokens=1312,accepted_score_end=50000,
                     candidate_starts=selected,selected_starts=selected) for i in range(16)])
    bounds=[dict(selection='pressure')]
    assert memory_pressure_audit([event],bounds,True)==bounds
    for fault in ['shared','missing','future','budget']:
        broken=deepcopy(event)
        if fault=='shared':broken['actual_method']='accumulated_attention'
        elif fault=='missing':broken['layers'].pop()
        elif fault=='future':broken['layers'][-1]['accepted_score_end']=60000
        elif fault=='budget':broken['layers'][-1]['charged_tokens']=40000
        with pytest.raises(ValueError):memory_pressure_audit([broken],bounds,True)


def restore_digest():
    return dict(request_id='cmpl-second-0-worker',history_tokens=256000,working_tokens=47104,
        sha256_equal=True,recurrent_byte_equal=True,working_attention_layers=17,mtp_attention_layers=1,
        recurrent_tensors=96,draft_layers=0,target_kv_bytes=47104*17*4*256*2,sha256=['a'*64]*34,
        verification='sha256_all_target_bytes_and_exact_model_tensors',oracle_kind='fp8_immutable_rebuild')


def test_memory_restore_gate_binds_the_actual_warm_request_and_entire_prefix_frame():
    from run_layerwise_memory import memory_restore_digest_audit
    row=restore_digest()
    assert memory_restore_digest_audit([row],'cmpl-second',256000,47104)==row


@pytest.mark.parametrize('fault',['missing','foreign','duplicate','position','extent','sha','recurrent','draft','bytes','layers','oracle'])
def test_memory_restore_gate_rejects_missing_partial_or_other_request_proof(fault):
    from run_layerwise_memory import memory_restore_digest_audit
    row=restore_digest();rows=[row]
    if fault=='missing':rows=[]
    elif fault=='foreign':row['request_id']='cmpl-first-0-worker'
    elif fault=='duplicate':rows.append(dict(row))
    elif fault=='position':row['history_tokens']+=1024
    elif fault=='extent':row['working_tokens']-=32
    elif fault=='sha':row['sha256_equal']=False
    elif fault=='recurrent':row['recurrent_byte_equal']=False
    elif fault=='draft':row['draft_layers']=1
    elif fault=='bytes':row['target_kv_bytes']-=1
    elif fault=='layers':row['sha256'].pop()
    elif fault=='oracle':row['oracle_kind']='metadata_only'
    with pytest.raises(ValueError):memory_restore_digest_audit(rows,'cmpl-second',256000,47104)


@pytest.mark.parametrize('fault',[None,'unfinished','gpu_owner','no_warm_restore','input_changed','missing_digest'])
def test_formal_memory_requires_completed_matching_model_qualification(monkeypatch,tmp_path,fault):
    import run_layerwise_memory as driver
    import recall_comparison_protocol as p
    root=tmp_path/'qualification';case=root/'run/answers/cases/second';case.mkdir(parents=True)
    fixture=dict(qualification=True,suite='memoryagentbench',cases=1,
                 rows=[dict(id='second',token_sha256='original-input')])
    monkeypatch.setattr(driver,'verify',lambda root:(fixture,dict(kvmem={})))
    for name in ('prepared-proof.json','runtime-proof.json','fixture.json','profile.json'):
        p.save(root/name,{})
    p.save(root/'status.json',dict(status='running' if fault=='unfinished' else 'completed'))
    p.save(root/'execution-audit.json',dict(status='passed',cases=1))
    p.save(root/'run/cleanup.json',dict(gpu_processes=[42] if fault=='gpu_owner' else [],remaining_owned_groups=[]))
    result=dict(status='completed',token_sha256='changed' if fault=='input_changed' else 'original-input',
                prefix_reused=fault!='no_warm_restore',prefix_tokens=256000,
                prefix_restore_digest=restore_digest(),layerwise_prefix=dict(layer_tokens=[47104]*16))
    if fault=='missing_digest':result.pop('prefix_restore_digest')
    p.save(case/'result.json',result);p.save(case/'response.json',dict(id='cmpl-second'))
    for name in ('request.json','terminal.json','events.json'):p.save(case/name,{})
    if fault:
        with pytest.raises((ValueError,KeyError)):driver.completed_memory_qualification(root,'memoryagentbench')
    else:
        _,_,proof=driver.completed_memory_qualification(root,'memoryagentbench')
        assert proof['verified_warm_restores']==1
        assert proof['files'][str(case/'response.json')]==p.sha(case/'response.json')


@pytest.mark.parametrize('active_budget',[36864,24576,16384])
def test_full_lme_output_cap_and_query_tail_fit_without_post_query_pressure(active_budget):
    from run_layerwise_memory import memory_generation_capacity_audit
    kv=dict(select_budget=active_budget,gen_budget=16384,block_tokens=32)
    query=dict(history_begin=200000)
    result=memory_generation_capacity_audit(query,[dict(charged_tokens=active_budget-160)]*16,200113,16384,kv)
    assert result['speculation_headroom_tokens']==47
    assert result['max_output_tokens']==16384 and result['no_post_query_pressure_needed']
    # The shorter-output RULER path's full history allowance is not enough to
    # reserve the original LME cap plus its complete replay suffix.
    with pytest.raises(ValueError):
        memory_generation_capacity_audit(query,[dict(charged_tokens=active_budget)]*16,200113,16384,kv)


@pytest.mark.parametrize('budget',[24576,16384])
def test_lme_budget_sweep_preserves_every_other_model_control(budget):
    from copy import deepcopy
    from run_layerwise_memory import budget_variant_profile
    original=dict(model='frozen-nvfp4',max_num_seqs=8,query_replay_budget=16384,
        kvmem=dict(select_budget=36864,gen_budget=16384,sink_blocks=41,recent_blocks=0))
    before=deepcopy(original)
    result=budget_variant_profile(original,budget)
    assert result['kvmem']['select_budget']==budget
    result['kvmem']['select_budget']=36864
    assert result==before and original==before
    original['kvmem']['gen_budget']=8192
    with pytest.raises(ValueError):budget_variant_profile(original,budget)


@pytest.mark.parametrize('budget',[24576,16384])
def test_pressure_audit_enforces_the_smaller_active_budget(budget):
    from run_layerwise_memory import memory_pressure_audit
    starts=list(range(0,1312,32))
    layer=dict(charged_tokens=budget,selected_tokens=budget,accepted_score_end=100000,
        candidate_starts=starts,selected_starts=starts)
    event=dict(event='pressure_selected',event_type='prefill_pressure',fallback_reason=None,
        actual_method='accumulated_attention_per_layer',selected_tokens=budget,accepted_history=100000,
        layers=[dict(layer,layer=i) for i in range(16)])
    refresh=[dict(selection='pressure')]
    assert memory_pressure_audit([event],refresh,True,select_budget=budget)==refresh
    # An overflow that the old 36K-only upper bound would miss must fail.
    event['layers'][5]['charged_tokens']+=32
    with pytest.raises(ValueError):memory_pressure_audit([event],refresh,True,select_budget=budget)
