import json

import pytest

from scripts.compare_terminalbench_policies import compare,exact_mcnemar


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def arm(root,policy,passed):
    root.mkdir()
    names=[f'task-{index:02}' for index in range(89)]
    outcomes=[dict(task='terminal-bench/'+name,trial=f'{name}-{policy}',
        rewards={'reward':int(index in passed)},exception=None,status='scored',passed=index in passed)
        for index,name in enumerate(names)]
    # Preserve an infrastructure failure as its own category, while keeping
    # all 89 attempts in the primary pass@1 denominator.
    outcomes[5].update(rewards=None,exception='EnvironmentBuildError',status='error')
    dataset={f'tasks/{name}/{file}':name+'/'+file for name in names
             for file in ('instruction.md','task.toml','tests/test.sh')}
    manifest=dict(tasks=89,launched_task_names=names,dataset='Terminal-Bench 2.1',
        dataset_commit='dataset-commit',harbor_commit='harbor-commit',harbor_version='one',
        attempts=1,max_concurrency=8,container_memory_admission_bytes=24*1024**3,
        output_cap=131072,agent_time_limit='official',official_timing_comparable=True,
        thinking_budget=4096,seed=1000,query_mode='task',
        frozen=dict(dataset=dataset,runtime={'source.py':'same-source-hash'}))
    timing=dict(agent_time_limit='official',official_timing_comparable=True,
        scope='Agent execution only; verifier/setup/tool timeouts retained',
        tasks={name:dict(original_sha256=dataset[f'tasks/{name}/task.toml'],
        effective_sha256=dataset[f'tasks/{name}/task.toml'],verifier_unchanged=True,environment_unchanged=True,
        original_agent_timeout_sec=900,effective_agent_timeout_sec=900) for name in names})
    job=dict(n_attempts=1,n_concurrent_trials=8,retry=dict(max_retries=0),
        jobs_dir=str(root/'jobs'),job_name='formal',timeout_multiplier=1,
        tasks=[dict(path=str(root/'dataset/tasks'/name)) for name in names],
        environment=dict(type='docker',extra_docker_compose=[str(root/'download-cache-compose.json')]),
        agents=[dict(name='terminus-2',kwargs=dict(temperature=.6,llm_call_kwargs=dict(seed=1000,top_p=.95)))])
    profile=dict(max_num_seqs=8,max_model_len=10000000,kvmem=dict(select_budget=36864,gen_budget=16384,
        repository_blocks=8192,block_tokens=32,cpu_read_cache_bytes=96*1024**3,session_capacity=8,
        historical_gc=dict(mode='delete',policy=policy,ready_low=512,ready_high=1024,
            policy_params=dict(keep_mandatory=True,**(dict(thinking_only=True,lru_fallback=True,
                feature_cache_bytes=256*1024**2) if policy=='redundancy' else {})))))
    gc=dict(retired=100,redundancy_retired=0 if policy=='lru' else 30,lru_fallback_retired=0 if policy=='lru' else 70,
        cpu_released=200,slots_released=200,policy_failures=0)
    audit=dict(passed=True,tasks=89,attempts=89,retries=0,actual_seed=1000,output_cap=131072,thinking_budget=4096,
        agent_time_limit='official',official_timing_comparable=True,
        cleanup=dict(remaining_processes=[],remaining_owned_gpu_pids=[]),
        formal_window_seconds=100,agent_window_seconds=90,
        accepted_completed_response_tokens=900,accepted_completed_response_tokens_per_wall_second=10,
        incomplete_native_requests=[],gc_formal_delta=gc,
        efficiency=dict(gpu=dict(agent=dict(status='unavailable',reason='test'))))
    counts=dict(passed=len(passed),scored=88,task_timeouts=0,errors=1,retries=0)
    formal=dict(expected=89,outcomes=outcomes,**counts)
    result=dict(**formal,tasks=[dict(**row,agent_seconds=20) for row in outcomes],measurement=audit)
    for name,value in [('manifest.json',manifest),('profile.json',profile),('job.json',job),
                       ('agent-time-policy.json',timing),('execution-audit.json',audit),
                       ('formal-events/summary.json',formal),('results.json',result),
                       ('status.json',dict(status='completed'))]:
        write(root/name,value)
    (root/'bridge').mkdir()
    (root/'bridge/requests.jsonl').write_text(json.dumps(dict(event='session_opened',owner='native-owner',
        session_id=outcomes[0]['trial']+'__agent'))+'\n')
    return root


@pytest.fixture
def pair(tmp_path):
    return arm(tmp_path/'baseline','lru',{0,4}),arm(tmp_path/'candidate','redundancy',{0,1,2,3})


def edit(root,name,change):
    path=root/name
    value=json.loads(path.read_text())
    change(value)
    write(path,value)


def configure_concurrency(root,tasks=12,sessions=12,bridge=16):
    edit(root,'manifest.json',lambda d:d.update(max_concurrency=tasks,task_concurrency=tasks,
        model_concurrency=8,session_capacity=sessions,bridge_max_sessions=bridge))
    edit(root,'job.json',lambda d:d.__setitem__('n_concurrent_trials',tasks))
    edit(root,'profile.json',lambda d:d['kvmem'].__setitem__('session_capacity',sessions))
    job=json.loads((root/'job.json').read_text())
    profile=json.loads((root/'profile.json').read_text())
    write(root/'jobs/formal/config.json',job)
    write(root/'bridge-process.json',dict(argv=['python','--max-sessions',str(bridge)]))
    connector=dict(kv_connector_extra_config=dict(kvmem=dict(session_capacity=sessions)))
    write(root/'model/server/launch.json',dict(profile=profile,
        vllm_argv=['vllm','--max-num-seqs','8','--kv-transfer-config',json.dumps(connector)]))


@pytest.mark.parametrize('tasks,sessions',[(8,8),(12,12),(12,16),(16,16)])
def test_explicit_task_and_model_limits_reconcile_with_actual_launches(pair,tasks,sessions):
    for root in pair:
        configure_concurrency(root,tasks,sessions)
    result=compare(*pair)
    assert result['validation']['concurrency']==dict(task_concurrency=tasks,model_concurrency=8,
        session_capacity=sessions,bridge_max_sessions=16)
    assert result['candidate']['attempts']==89


def test_legacy_manifest_is_only_the_previous_eight_trial_configuration(pair):
    assert compare(*pair)['validation']['concurrency']==dict(task_concurrency=8,model_concurrency=8,
        session_capacity=8,bridge_max_sessions=16)
    for root in pair:
        edit(root,'manifest.json',lambda d:d.__setitem__('max_concurrency',12))
        edit(root,'job.json',lambda d:d.__setitem__('n_concurrent_trials',12))
    with pytest.raises(ValueError,match='legacy manifests require eight'):
        compare(*pair)


@pytest.mark.parametrize('filename,change,message',[
    ('manifest.json',lambda d:d.pop('model_concurrency'),'Incomplete explicit'),
    ('manifest.json',lambda d:d.__setitem__('task_concurrency',True),'positive integers'),
    ('manifest.json',lambda d:d.__setitem__('max_concurrency',8),'max_concurrency differs'),
    ('manifest.json',lambda d:d.__setitem__('model_concurrency',12),'eight native model'),
    ('manifest.json',lambda d:d.__setitem__('session_capacity',8),'exceeds persistent session'),
    ('manifest.json',lambda d:d.__setitem__('bridge_max_sessions',8),'exceeds persistent session'),
    ('job.json',lambda d:d.__setitem__('n_concurrent_trials',8),'Job task concurrency'),
    ('job.json',lambda d:d['tasks'].reverse(),'Job task order'),
    ('profile.json',lambda d:d.__setitem__('max_num_seqs',12),'Profile model concurrency'),
    ('profile.json',lambda d:d['kvmem'].__setitem__('session_capacity',16),'Profile session capacity'),
    ('jobs/formal/config.json',lambda d:d.__setitem__('n_concurrent_trials',8),'Actual Harbor task concurrency'),
    ('jobs/formal/config.json',lambda d:d['tasks'].reverse(),'Actual Harbor task queue'),
    ('bridge-process.json',lambda d:d.__setitem__('argv',['python','--max-sessions','8']),'Actual bridge session capacity'),
    ('model/server/launch.json',lambda d:d['vllm_argv'].__setitem__(2,'12'),'Actual native model concurrency'),
    ('model/server/launch.json',lambda d:d['profile']['kvmem'].__setitem__('session_capacity',8),'Actual launch profile'),
    ('model/server/launch.json',lambda d:d['vllm_argv'].__setitem__(4,json.dumps(dict(
        kv_connector_extra_config=dict(kvmem=dict(session_capacity=8))))),'Actual native connector session capacity'),
])
def test_matching_but_inconsistent_concurrency_is_rejected(pair,filename,change,message):
    # Identical bad settings in both arms cannot be excused by pair equality.
    for root in pair:
        configure_concurrency(root)
        edit(root,filename,change)
    with pytest.raises(ValueError,match=message):
        compare(*pair)


@pytest.mark.parametrize('relative',['jobs/formal/config.json','bridge-process.json','model/server/launch.json'])
def test_new_concurrency_manifest_requires_actual_launch_evidence(pair,relative):
    for root in pair:
        configure_concurrency(root)
    (pair[1]/relative).unlink()
    with pytest.raises(ValueError,match='Missing actual concurrency evidence'):
        compare(*pair)


def test_legacy_actual_evidence_is_checked_when_available(pair):
    write(pair[1]/'jobs/formal/config.json',dict(n_concurrent_trials=12,tasks=[]))
    with pytest.raises(ValueError,match='Actual Harbor task concurrency'):
        compare(*pair)


def test_different_but_internally_consistent_session_capacity_cannot_be_paired(pair):
    configure_concurrency(pair[0],sessions=12)
    configure_concurrency(pair[1],sessions=16)
    with pytest.raises(ValueError,match='Concurrency configuration differs'):
        compare(*pair)


def test_different_but_internally_consistent_task_order_cannot_be_paired(pair):
    for root in pair:
        configure_concurrency(root)
    edit(pair[1],'manifest.json',lambda d:d['launched_task_names'].reverse())
    for name in ('job.json','jobs/formal/config.json'):
        edit(pair[1],name,lambda d:d['tasks'].reverse())
    with pytest.raises(ValueError,match='task queue order differs'):
        compare(*pair)


def test_pairs_by_task_identity_and_preserves_errors_gc_and_denominator(pair):
    baseline,candidate=pair
    # Different completion order is normal and must not pair unrelated tasks.
    edit(candidate,'formal-events/summary.json',lambda d:d['outcomes'].reverse())
    result=compare(baseline,candidate)
    assert result['baseline']['pass_at_1']==pytest.approx(2/89)
    assert result['candidate']['pass_at_1']==pytest.approx(4/89)
    assert result['accuracy_delta_percentage_points']==pytest.approx(200/89)
    assert result['paired_counts']==dict(both_pass=1,both_fail=84,candidate_only_pass=3,baseline_only_pass=1)
    assert result['paired_task_lists']['candidate_only_pass']==['task-01','task-02','task-03']
    assert result['baseline']['errors']==result['candidate']['errors']==1
    assert result['candidate']['unscored_tasks']==['task-05']
    assert result['exact_mcnemar']['p_value']==.625
    assert result['candidate']['gc_formal_delta']['retired']==100
    assert result['candidate']['gc_formal_delta']['cpu_released']==200
    assert result['candidate']['per_task_gc']['task-00']['bridge_owners']==['native-owner']
    assert result['candidate']['per_task_gc']['task-00']['retired'] is None
    assert result['candidate']['per_task_gc']['task-00']['status']=='unavailable'
    assert result['validation']['shared_cpu_cache_cap_bytes']==96*1024**3


@pytest.mark.parametrize('left,right,expected',[(0,0,1),(1,0,1),(5,0,.0625),(0,5,.0625),(3,1,.625),(4,4,1)])
def test_exact_two_sided_mcnemar_known_binomial_cases(left,right,expected):
    result=exact_mcnemar(left,right)
    assert result['p_value']==expected
    assert result['discordant_pairs']==left+right


@pytest.mark.parametrize('left,right',[(-1,2),(1.0,2),(True,0)])
def test_exact_mcnemar_rejects_invalid_counts(left,right):
    with pytest.raises(ValueError,match='nonnegative integers'):
        exact_mcnemar(left,right)


def test_duplicate_task_is_rejected_even_with_89_rows(pair):
    baseline,candidate=pair
    edit(candidate,'formal-events/summary.json',lambda d:d['outcomes'][-1].update(task=d['outcomes'][0]['task']))
    with pytest.raises(ValueError,match='duplicate task'):
        compare(baseline,candidate)


def test_task_set_mismatch_is_rejected(pair):
    baseline,candidate=pair
    edit(candidate,'manifest.json',lambda d:d['launched_task_names'].__setitem__(-1,'different-task'))
    for filename in ('formal-events/summary.json','results.json'):
        def change(d):
            for row in d['outcomes']:
                if row['task']=='terminal-bench/task-88':row['task']='terminal-bench/different-task'
            for row in d.get('tasks',[]):
                if row['task']=='terminal-bench/task-88':row['task']='terminal-bench/different-task'
        edit(candidate,filename,change)
    with pytest.raises(ValueError,match='Paired task sets differ'):
        compare(baseline,candidate)


@pytest.mark.parametrize('filename,change,message',[
    ('manifest.json',lambda d:d['frozen']['dataset'].__setitem__('tasks/task-00/tests/test.sh','changed'),'Frozen dataset'),
    ('manifest.json',lambda d:d['frozen']['runtime'].__setitem__('source.py','changed'),'Frozen runtime'),
    ('job.json',lambda d:d['agents'][0]['kwargs']['llm_call_kwargs'].__setitem__('seed',1001),'Jobs differ'),
    ('job.json',lambda d:d.__setitem__('timeout_multiplier',2),'timeout multiplier changed'),
    ('profile.json',lambda d:d['kvmem'].__setitem__('cpu_read_cache_bytes',97*1024**3),'Model profiles differ'),
    ('profile.json',lambda d:d['kvmem']['historical_gc'].__setitem__('ready_low',513),'Model profiles differ'),
    ('agent-time-policy.json',lambda d:d['tasks']['task-00'].__setitem__('original_sha256','changed'),'deadlines or resource audit differs'),
])
def test_non_policy_differences_are_rejected(pair,filename,change,message):
    baseline,candidate=pair
    edit(candidate,filename,change)
    with pytest.raises(ValueError,match=message):
        compare(baseline,candidate)


def test_execution_audit_cannot_be_replaced_with_unverified_result_fields(pair):
    baseline,candidate=pair
    edit(candidate,'results.json',lambda d:d['measurement'].__setitem__('accepted_completed_response_tokens',1))
    with pytest.raises(ValueError,match='results and execution audit differ'):
        compare(baseline,candidate)


def test_reward_and_reported_pass_must_agree(pair):
    baseline,candidate=pair
    edit(candidate,'formal-events/summary.json',lambda d:d['outcomes'][0].__setitem__('passed',False))
    with pytest.raises(ValueError,match='reward/pass mismatch'):
        compare(baseline,candidate)


def test_same_docker_tag_with_different_image_identity_is_rejected(pair):
    baseline,candidate=pair
    for root,identity in [(baseline,'sha256:old'),(candidate,'sha256:new')]:
        write(root/'images/task-00.json',dict(task='task-00',image='same:tag',id=identity))
    with pytest.raises(ValueError,match='image identities differ'):
        compare(baseline,candidate)


def test_scored_pass_at_agent_deadline_remains_a_pass(pair):
    baseline,candidate=pair
    for filename in ('formal-events/summary.json','results.json'):
        def change(d):
            for row in d['outcomes']+d.get('tasks',[]):
                if row['task']=='terminal-bench/task-01':
                    row['exception']='AgentTimeoutError'
        edit(candidate,filename,change)
    result=compare(baseline,candidate)
    assert result['candidate']['passed']==4
    assert result['candidate']['agent_timeout_exceptions']==1
    assert result['candidate']['task_timeouts']==0  # This counter is unscored timeouts only.
    assert 'task-01' in result['paired_task_lists']['candidate_only_pass']


def configure_unlimited(root, mode='none'):
    from scripts.terminalbench_policy import configure_task_deadlines
    names=json.loads((root/'manifest.json').read_text())['launched_task_names']
    for name in names:
        path=root/'dataset/tasks'/name/'task.toml'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 30\n[environment]\nmemory_mb = 1024\n')
    timing=configure_task_deadlines(root/'dataset/tasks',mode)
    write(root/'agent-time-policy.json',timing)
    def manifest(d):
        d.update(agent_time_limit=mode,official_timing_comparable=False)
        for name,row in timing['tasks'].items():
            d['frozen']['dataset'][f'tasks/{name}/task.toml']=row['effective_sha256']
    edit(root,'manifest.json',manifest)
    edit(root,'execution-audit.json',lambda d:d.update(agent_time_limit=mode,official_timing_comparable=False))
    edit(root,'results.json',lambda d:d['measurement'].update(agent_time_limit=mode,official_timing_comparable=False))
    if mode=='relaxed':
        write(root/'formal-deadline-hook.json',dict(agent_time_limit='relaxed',original_phase_sha256='test'))


def test_paired_relaxed_deadlines_are_custom_and_must_match(pair):
    for root in pair:
        configure_unlimited(root, 'relaxed')
    result=compare(*pair)
    assert result['agent_time_limit']=='relaxed'
    assert result['official_timing_comparable'] is False
    assert '2x official' in result['timing_scope']
    configure_unlimited(pair[1], 'none')
    with pytest.raises(ValueError, match='Frozen dataset|deadline|timing'):
        compare(*pair)


def test_paired_unlimited_agent_time_is_explicitly_custom_local(pair):
    for root in pair:
        configure_unlimited(root)
    result=compare(*pair)
    assert result['agent_time_limit']=='none'
    assert result['official_timing_comparable'] is False
    assert result['validation']['official_timing_comparable'] is False
    assert 'custom local evaluation' in result['timing_scope']
    assert 'not directly comparable with official timed scores' in result['timing_scope']
    assert result['baseline']['attempts']==result['candidate']['attempts']==89
    assert any('verifier/setup/tool timeouts retained' in row for row in result['caveats'])


def test_official_timed_pair_stays_official(pair):
    result=compare(*pair)
    assert result['agent_time_limit']=='official'
    assert result['official_timing_comparable'] is True


def test_unlimited_and_official_cannot_be_paired(pair):
    configure_unlimited(pair[1])
    with pytest.raises(ValueError,match='Frozen dataset|deadline|timing'):
        compare(*pair)


@pytest.mark.parametrize('filename,change,message',[
    ('agent-time-policy.json',lambda d:d.__setitem__('scope','all deadlines removed'),'only Agent timing'),
    ('agent-time-policy.json',lambda d:d.__setitem__('official_timing_comparable',True),'comparability'),
    ('agent-time-policy.json',lambda d:d['tasks']['task-00'].__setitem__('effective_agent_timeout_sec',900),'still enabled'),
    ('agent-time-policy.json',lambda d:d['tasks']['task-00'].__setitem__('verifier_unchanged',False),'proof is incomplete'),
    ('job.json',lambda d:d['agents'][0].__setitem__('override_timeout_sec',1000),'timing override'),
    ('job.json',lambda d:d.__setitem__('timeout_multiplier',2),'timeout multiplier'),
])
def test_unlimited_requires_matching_agent_only_audit_and_no_hidden_override(pair,filename,change,message):
    for root in pair:
        configure_unlimited(root)
        edit(root,filename,change)
    with pytest.raises(ValueError,match=message):
        compare(*pair)


def test_unlimited_effective_task_file_must_match_frozen_audit(pair):
    for root in pair:
        configure_unlimited(root)
    (pair[1]/'dataset/tasks/task-00/task.toml').write_text('[agent]\ntimeout_sec = 900\n')
    with pytest.raises(ValueError,match='does not match its frozen audit'):
        compare(*pair)


def update_audit(root,change):
    audit=json.loads((root/'execution-audit.json').read_text())
    change(audit)
    write(root/'execution-audit.json',audit)
    edit(root,'results.json',lambda d:d.__setitem__('measurement',audit))


def scope_journal(root,policy,task_counts):
    bridge=[dict(event='session_opened',owner='smoke-owner',session_id='smoke-trial__agent')]
    records=[dict(event='history_scope_closed',at=50,namespace='preflight',mode='delete',policy=policy,
                  stats=dict(retired=900)),
             dict(event='history_scope_closed',at=80,namespace='smoke',mode='delete',policy=policy,
                  session_namespace=['smoke-owner','smoke-owner'],session_generation=1,stats=dict(retired=5))]
    totals=dict(retired=0,redundancy_retired=0,lru_fallback_retired=0,lru_representative_retired=0,
                cpu_released=0,slots_released=0,policy_failures=0)
    for name,(retired,redundancy) in task_counts.items():
        owner='owner-'+name
        bridge.extend([dict(event='session_opened',owner=owner,session_id=f'{name}-{policy}__agent'),
                       dict(owner=owner,request_id='cmpl-'+name,output_tokens=10)])
        stats=dict(retired=retired,redundancy_retired=redundancy,
            lru_fallback_retired=retired-redundancy if policy=='redundancy' else 0,
            lru_representative_retired=0,cpu_released=retired+20,slots_released=retired+20,policy_failures=0)
        for key,value in stats.items():totals[key]+=value
        records.append(dict(event='history_scope_closed',at=110,namespace='cmpl-'+name+'-0',
            mode='delete',policy=policy,session_namespace=['unrelated-cache',owner],session_generation=1,stats=stats))
    (root/'bridge/requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in bridge))
    path=root/'model/server/capture/historical_gc_scopes.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    write(root/'formal-boundary.json',dict(at=100,capture=dict(archive=dict(historical_gc=dict(active_scopes=0)))))
    def update(d):
        d['gc_formal_delta']=totals
        d['gc']=dict(active_scopes=0,scope_metrics=dict(path=str(path),records=len(records),errors=0,last_error=None))
    update_audit(root,update)
    return path


@pytest.fixture
def scoped_pair(pair):
    baseline,candidate=pair
    scope_journal(baseline,'lru',{'task-00':(2,0),'task-04':(3,0)})
    scope_journal(candidate,'redundancy',{'task-00':(1,1),'task-01':(4,4),'task-02':(2,0)})
    return pair


def test_scope_gc_excludes_smoke_and_preflight_and_reports_exposed_task_pairs(scoped_pair):
    result=compare(*scoped_pair)
    baseline,candidate=result['baseline'],result['candidate']
    assert baseline['gc_scope_attribution']['status']==candidate['gc_scope_attribution']['status']=='measured'
    assert candidate['gc_scope_attribution']['excluded_before_formal_scope_count']==2
    assert baseline['per_task_gc']['task-04']['retired']==3
    assert baseline['per_task_gc']['task-04']['counters']['lru_fallback_retired']==0
    assert baseline['per_task_gc']['task-04']['lru_retired']==3
    assert candidate['per_task_gc']['task-02']['lru_retired']==2
    assert candidate['per_task_gc']['task-01']['retired']==4
    assert candidate['per_task_gc']['task-01']['counters']['cpu_released']==24
    assert candidate['per_task_gc']['task-88']['retired']==0
    exposed=result['gc_triggered_subsets']
    assert exposed['status']=='measured'
    assert exposed['triggered_tasks']['candidate']==['task-00','task-01','task-02']
    assert exposed['union_triggered']['tasks']==['task-00','task-01','task-02','task-04']
    assert exposed['union_triggered']['paired_counts']==dict(both_pass=1,both_fail=0,candidate_only_pass=2,baseline_only_pass=1)
    assert exposed['neither_triggered']['task_count']==85
    assert exposed['candidate_redundancy_triggered']['tasks']==['task-00','task-01']
    assert exposed['per_arm_used_vs_unused']['candidate']['not_triggered']['passed']==1


def rewrite_scopes(root,change):
    path=root/'model/server/capture/historical_gc_scopes.jsonl'
    records=[json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    path.write_text(''.join(json.dumps(row)+'\n' for row in records))
    update_audit(root,lambda d:d['gc']['scope_metrics'].__setitem__('records',len(records)))


def test_scope_duplicate_with_same_namespace_generation_is_not_double_counted(scoped_pair):
    baseline,candidate=scoped_pair
    rewrite_scopes(candidate,lambda rows:rows.append(dict(rows[-1])))
    result=compare(baseline,candidate)
    audit=result['candidate']['gc_scope_attribution']
    assert audit['status']=='incomplete' and len(audit['duplicate_scopes'])==1
    assert audit['observed_formal_counter_sums']['retired']==7
    assert result['candidate']['per_task_gc']['task-02']['retired'] is None
    assert result['gc_triggered_subsets']['status']=='unavailable'
    assert result['candidate']['passed']==4  # Journal defects do not rewrite actual task rewards.


def test_unmapped_scope_after_formal_boundary_is_explicit_not_assigned_by_time(scoped_pair):
    baseline,candidate=scoped_pair
    rewrite_scopes(candidate,lambda rows:rows.append(dict(event='history_scope_closed',at=120,
        namespace='unknown-native',mode='delete',policy='redundancy',stats=dict(retired=0))))
    result=compare(baseline,candidate)
    audit=result['candidate']['gc_scope_attribution']
    assert audit['status']=='incomplete'
    assert audit['unmatched_formal_scopes'][0]['namespace']=='unknown-native'
    assert result['gc_triggered_subsets']['status']=='unavailable'


@pytest.mark.parametrize('change,reason',[
    (lambda d:d['gc_formal_delta'].__setitem__('retired',8),'scope sum differs'),
    (lambda d:d['gc']['scope_metrics'].__setitem__('errors',1),'writer reported errors'),
    (lambda d:d['gc']['scope_metrics'].__setitem__('records',999),'count differs'),
    (lambda d:d['gc'].__setitem__('active_scopes',1),'zero active scopes'),
])
def test_scope_accounting_or_writer_failures_withhold_exposure_analysis(scoped_pair,change,reason):
    baseline,candidate=scoped_pair
    update_audit(candidate,change)
    result=compare(baseline,candidate)
    audit=result['candidate']['gc_scope_attribution']
    assert audit['status']=='incomplete'
    assert any(reason in message for message in audit['issues'])
    assert result['gc_triggered_subsets']['status']=='unavailable'


def test_legacy_scope_without_binding_uses_exact_native_response_identity(scoped_pair):
    baseline,candidate=scoped_pair
    rewrite_scopes(candidate,lambda rows:rows[-1].pop('session_namespace'))
    result=compare(baseline,candidate)
    assert result['candidate']['gc_scope_attribution']['status']=='measured'
    assert result['candidate']['per_task_gc']['task-02']['retired']==2


def test_present_but_conflicting_binding_does_not_fall_back_to_request_guess(scoped_pair):
    baseline,candidate=scoped_pair
    rewrite_scopes(candidate,lambda rows:rows[-1].__setitem__('session_namespace',['owner-task-01','owner-task-02']))
    result=compare(baseline,candidate)
    assert result['candidate']['gc_scope_attribution']['status']=='incomplete'
    assert result['candidate']['gc_scope_attribution']['unmatched_formal_scopes']


@pytest.mark.parametrize('retired',[0,1])
def test_release_control_scope_requires_explicit_cleanup_identity_and_zero_retirements(scoped_pair,retired):
    baseline,candidate=scoped_pair
    journal=candidate/'bridge/requests.jsonl'
    with journal.open('a') as stream:
        stream.write(json.dumps(dict(event='history_released',owner='owner-task-01',
            session_id='task-01-redundancy__agent',cleanup_request_id='cmpl-cleanup-task-01'))+'\n')
    rewrite_scopes(candidate,lambda rows:rows.append(dict(event='history_scope_closed',at=130,
        namespace='cmpl-cleanup-task-01-0',mode='delete',policy='redundancy',
        stats=dict(retired=retired,cpu_released=1,slots_released=1))))
    result=compare(baseline,candidate)
    task=result['candidate']['per_task_gc']['task-01']
    audit=result['candidate']['gc_scope_attribution']
    assert task['release_control_scopes']==1 and task['closed_history_scopes']==1
    assert audit['mapped_formal_scopes'][-1]['kind']=='release_control'
    assert not audit['unmatched_formal_scopes']
    if retired:
        assert audit['status']=='incomplete'
        assert any('Release control scope unexpectedly' in reason for reason in audit['issues'])
        assert result['gc_triggered_subsets']['status']=='unavailable'
    else:
        assert audit['status']=='measured'
        assert task['retired']==4
        assert result['gc_triggered_subsets']['status']=='measured'


def test_control_scope_cannot_substitute_for_missing_history_scope(scoped_pair):
    baseline,candidate=scoped_pair
    with (candidate/'bridge/requests.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event='history_released',owner='owner-task-02',
            session_id='task-02-redundancy__agent',cleanup_request_id='cmpl-cleanup-task-02'))+'\n')
    def change(rows):
        rows[-1]=dict(event='history_scope_closed',at=130,namespace='cmpl-cleanup-task-02-0',
                      mode='delete',policy='redundancy',stats=dict(retired=0))
    rewrite_scopes(candidate,change)
    # Even if aggregate retirement happens to agree, the missing history
    # owner must prevent a false "this task never used GC" conclusion.
    def totals(d):
        d['gc_formal_delta']['retired']-=2
        d['gc_formal_delta']['lru_fallback_retired']-=2
    update_audit(candidate,totals)
    result=compare(baseline,candidate)
    assert result['candidate']['gc_scope_attribution']['status']=='incomplete'
    assert result['candidate']['gc_scope_attribution']['formal_owners_without_closed_scope']==['owner-task-02']
    assert result['gc_triggered_subsets']['status']=='unavailable'
