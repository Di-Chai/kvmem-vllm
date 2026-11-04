"""Audit two complete Terminal-Bench policy runs and report paired pass@1.

One attempted trajectory per task is retained, including unscored failures.
This does not regenerate tasks, rescore answers, or infer per-task GC from
global counters shared by concurrent tasks.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from math import comb,isfinite
from pathlib import Path
import time
import tomllib


TASK_COUNT = 89
CONCURRENCY_FIELDS = ('task_concurrency','model_concurrency','session_capacity','bridge_max_sessions')


def read(path):
    return json.loads(path.read_text())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(candidate_only,baseline_only):
    """Two-sided exact conditional binomial test of discordant task pairs."""
    require(type(candidate_only) is int and type(baseline_only) is int and
            candidate_only>=0 and baseline_only>=0,'Discordant counts must be nonnegative integers')
    discordant=candidate_only+baseline_only
    p_value=(min(1.,2*sum(comb(discordant,k) for k in range(min(candidate_only,baseline_only)+1))/2**discordant)
             if discordant else 1.)
    return dict(method='exact two-sided McNemar (conditional Binomial(n, 0.5))',
        discordant_pairs=discordant,candidate_only=candidate_only,baseline_only=baseline_only,
        p_value=p_value)


def normalized_paths(value,root):
    """Ignore only the output root, keeping sampling, resources and task order."""
    if isinstance(value,dict):
        return {key:normalized_paths(child,root) for key,child in value.items()}
    if isinstance(value,list):
        return [normalized_paths(child,root) for child in value]
    if isinstance(value,str):
        path=str(root)
        if value==path or value.startswith(path+'/'):
            return '$RUN'+value[len(path):]
    return value


def task_name(value):
    return value.removeprefix('terminal-bench/')


def outcome_map(rows,label):
    require(len(rows)==TASK_COUNT,f'{label}: expected exactly 89 outcomes')
    output={}
    trials=set()
    for row in rows:
        name=task_name(row['task'])
        require(name not in output,f'{label}: duplicate task {name}')
        require(row['trial'] not in trials,f'{label}: duplicate trial {row["trial"]}')
        rewards=row.get('rewards')
        passed=bool(rewards and rewards.get('reward')==1)
        require(type(row.get('passed')) is bool and row['passed']==passed,f'{label}: reward/pass mismatch for {name}')
        status=('scored' if rewards is not None else
                'agent_timeout' if row.get('exception')=='AgentTimeoutError' else 'error')
        require(row['status']==status,f'{label}: reward/status mismatch for {name}')
        output[name]=row
        trials.add(row['trial'])
    return output


def outcome_identity(row):
    return {key:row.get(key) for key in ('trial','passed','status','rewards','exception')}


def concurrency_config(run):
    """Reconcile declared limits with the saved job and actual launch evidence."""
    manifest,profile,job=run['manifest'],run['profile'],run['job']
    present=[key in manifest for key in CONCURRENCY_FIELDS]
    explicit=any(present)
    require(not explicit or all(present),'Incomplete explicit concurrency manifest')
    limits=({key:manifest[key] for key in CONCURRENCY_FIELDS} if explicit else
            dict(task_concurrency=8,model_concurrency=8,session_capacity=8,bridge_max_sessions=16))
    require(all(type(value) is int and value>0 for value in limits.values()),
            'Concurrency limits must be positive integers')
    tasks,model,sessions,bridge=(limits[key] for key in CONCURRENCY_FIELDS)
    require(type(manifest.get('max_concurrency')) is int and manifest['max_concurrency']==tasks,
            'Manifest max_concurrency differs from task concurrency; legacy manifests require eight trials')
    require(model==8,'Expected eight native model sequences')
    require(tasks<=min(sessions,bridge),'Task concurrency exceeds persistent session or bridge capacity')
    require(type(job.get('n_concurrent_trials')) is int and job['n_concurrent_trials']==tasks,
            'Job task concurrency differs from manifest')
    require(type(profile.get('max_num_seqs')) is int and profile['max_num_seqs']==model,
            'Profile model concurrency differs from manifest')
    require(type(profile['kvmem'].get('session_capacity')) is int and
            profile['kvmem']['session_capacity']==sessions,
            'Profile session capacity differs from manifest')
    require([Path(row['path']).name for row in job['tasks']]==manifest['launched_task_names'],
            'Job task order differs from manifest')

    def evidence(relative):
        path=run['root']/relative
        require(path.is_file() or not explicit,'Missing actual concurrency evidence: '+relative)
        return read(path) if path.is_file() else None

    def argument(argv,flag):
        require(isinstance(argv,list) and argv.count(flag)==1 and argv.index(flag)+1<len(argv),
                'Missing or ambiguous actual launch argument: '+flag)
        return argv[argv.index(flag)+1]

    actual_job=evidence('jobs/formal/config.json')
    if actual_job is not None:
        require(type(actual_job.get('n_concurrent_trials')) is int and actual_job['n_concurrent_trials']==tasks,
                'Actual Harbor task concurrency differs from manifest')
        require(actual_job.get('tasks')==job['tasks'],'Actual Harbor task queue differs from frozen job')
    actual_bridge=evidence('bridge-process.json')
    if actual_bridge is not None:
        require(argument(actual_bridge.get('argv'),'--max-sessions')==str(bridge),
                'Actual bridge session capacity differs from manifest')
    launch=evidence('model/server/launch.json')
    if launch is not None:
        argv=launch.get('vllm_argv')
        require(argument(argv,'--max-num-seqs')==str(model),'Actual native model concurrency differs from manifest')
        actual_profile=launch.get('profile',{})
        require(type(actual_profile.get('max_num_seqs')) is int and actual_profile['max_num_seqs']==model and
                type(actual_profile.get('kvmem',{}).get('session_capacity')) is int and
                actual_profile['kvmem']['session_capacity']==sessions,
                'Actual launch profile concurrency differs from manifest')
        connector=json.loads(argument(argv,'--kv-transfer-config'))
        actual_sessions=connector.get('kv_connector_extra_config',{}).get('kvmem',{}).get('session_capacity')
        require(type(actual_sessions) is int and actual_sessions==sessions,
                'Actual native connector session capacity differs from manifest')
    return limits


GC_COUNTERS=('retired','redundancy_retired','lru_fallback_retired','lru_representative_retired',
             'lossy_retired','cpu_released','slots_released','policy_failures')


def gc_attribution(root,outcomes,audit,policy):
    """Attribute closed native scopes through the bridge's stable owner identity."""
    sessions={row['trial']+'__agent':name for name,row in outcomes.items()}
    owner_tasks={};native_links={};started_owners=set();issues=[]
    journal=root/'bridge/requests.jsonl'
    bridge_rows=[json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    for row in bridge_rows:
        if row.get('event')=='session_opened':
            owner,task=row['owner'],sessions.get(row.get('session_id'))
            if owner in owner_tasks and owner_tasks[owner]!=task:
                issues.append('Bridge owner maps to multiple sessions: '+owner)
            owner_tasks[owner]=task
    for row in bridge_rows:
        control=row.get('event')=='history_released'
        request_id=row.get('cleanup_request_id') if control else row.get('request_id')
        if request_id and row.get('owner') in owner_tasks:
            owner=row['owner'];link=dict(owner=owner,kind='release_control' if control else 'history')
            if request_id in native_links and native_links[request_id]!=link:
                issues.append('Native request maps to multiple owners: '+request_id)
            native_links[request_id]=link
            if owner_tasks[owner] is not None and not control:
                started_owners.add(owner)
    owners={name:sorted(owner for owner,task in owner_tasks.items() if task==name) for name in outcomes}
    per_task={name:dict(status='unavailable',retired=None,bridge_owners=owners[name],closed_scopes=0,
                       release_control_scopes=0)
              for name in sorted(outcomes)}
    path=root/'model/server/capture/historical_gc_scopes.jsonl'
    if not path.exists():
        return per_task,dict(status='unavailable',reason='No per-scope GC journal; global concurrent counters cannot be apportioned')
    try:
        records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError,ValueError) as error:
        return per_task,dict(status='incomplete',issues=['Cannot read GC scope journal: '+repr(error)])
    before=read(root/'formal-boundary.json') if (root/'formal-boundary.json').exists() else {}
    start=before.get('at')
    if type(start) not in (int,float) or not isfinite(start):
        return per_task,dict(status='incomplete',issues=['Missing formal GC boundary timestamp'])
    before_gc=before.get('capture',{}).get('archive',{}).get('historical_gc',{})
    final_gc=audit.get('gc') or {}
    if before_gc.get('active_scopes')!=0 or final_gc.get('active_scopes')!=0:
        issues.append('Formal GC boundaries do not prove zero active scopes')
    writer=final_gc.get('scope_metrics')
    if not writer:
        issues.append('Missing GC scope journal writer audit')
    else:
        if writer.get('errors')!=0 or writer.get('last_error'):
            issues.append('GC scope journal writer reported errors')
        if writer.get('records')!=len(records):
            issues.append('GC scope journal count differs from writer audit')
    counters={name:Counter() for name in outcomes}
    seen=set();mapped_owners=set();excluded=[];unmatched=[];duplicates=[];scopes=Counter()
    controls=Counter();mapped_scopes=[]
    for row in records:
        identity=(row.get('namespace'),row.get('session_generation'))
        if (row.get('event')!='history_scope_closed' or not isinstance(identity[0],str) or
                (identity[1] is not None and type(identity[1]) is not int)):
            issues.append('Malformed GC scope identity/event')
            continue
        if identity in seen:
            duplicates.append(dict(namespace=identity[0],session_generation=identity[1]))
            continue
        seen.add(identity)
        at=row.get('at')
        if type(at) not in (int,float) or not isfinite(at):
            issues.append('Invalid GC scope close timestamp: '+identity[0])
            continue
        if at<start:
            excluded.append(identity[0])
            continue
        stats=row.get('stats') or {}
        if not isinstance(stats,dict) or 'retired' not in stats or any(type(stats.get(key,0)) is not int or stats.get(key,0)<0 for key in GC_COUNTERS):
            issues.append('Invalid GC scope counters: '+identity[0])
            continue
        context=row.get('session_namespace')
        matches=set()
        linked=[link for request,link in native_links.items()
                if identity[0]==request or identity[0].startswith(request+'-')]
        kind='release_control' if any(link['kind']=='release_control' for link in linked) else 'history'
        if context is not None:
            # SessionBinding.namespace is (cache_salt, session_id). The bridge
            # uses the same stable owner for both; never guess a task by timing.
            if isinstance(context,list) and len(context)==2 and all(isinstance(value,str) for value in context):
                matches={value for value in context if value in owner_tasks}
        else:
            matches={link['owner'] for link in linked}
        formal_matches={owner for owner in matches if owner_tasks[owner] is not None}
        tasks={owner_tasks[owner] for owner in formal_matches}
        if len(tasks)!=1 or len(matches)!=len(formal_matches):
            unmatched.append({key:row.get(key) for key in ('namespace','session_namespace','session_generation','at','policy')}
                             |dict(retired=stats['retired']))
            continue
        task=next(iter(tasks))
        if linked and not {link['owner'] for link in linked}.issubset(formal_matches):
            issues.append('GC scope binding conflicts with native response owner: '+identity[0])
        if row.get('policy')!=policy or row.get('mode')!='delete':
            issues.append('GC scope policy differs from frozen profile: '+identity[0])
        if kind=='release_control' and stats['retired']!=0:
            issues.append('Release control scope unexpectedly retired policy blocks: '+identity[0])
        counters[task].update({key:stats.get(key,0) for key in GC_COUNTERS})
        scopes[task]+=1
        if kind=='release_control':
            controls[task]+=1
        else:
            mapped_owners.update(formal_matches)
        mapped_scopes.append(dict(namespace=identity[0],session_generation=identity[1],task=task,
                                  bridge_owners=sorted(formal_matches),kind=kind,retired=stats['retired']))
    if duplicates:
        issues.append('Duplicate closed GC scopes')
    if unmatched:
        issues.append('Unmatched scopes at or after the formal GC boundary')
    missing_owners=sorted(started_owners-mapped_owners)
    if missing_owners:
        issues.append('Completed native requests lack an attributed closed scope')
    totals={key:sum(value[key] for value in counters.values()) for key in GC_COUNTERS}
    expected=audit.get('gc_formal_delta')
    if not isinstance(expected,dict) or 'retired' not in expected:
        issues.append('Missing formal GC counter delta')
    else:
        # CPU/slot releases remain visible but include normal task-end cleanup.
        for key in ('retired','redundancy_retired','lru_fallback_retired','lru_representative_retired','policy_failures'):
            if key in expected and totals[key]!=expected[key]:
                issues.append('Attributed scope sum differs from formal GC delta: '+key)
    complete=not issues
    for name in per_task:
        per_task[name].update(status='measured' if complete else 'incomplete',closed_scopes=scopes[name],
            closed_history_scopes=scopes[name]-controls[name],release_control_scopes=controls[name],
            retired=counters[name]['retired'] if complete else None,
            counters=dict(counters[name]) if complete else None,
            observed_counters=dict(counters[name]),
            lru_retired=(counters[name]['retired'] if policy=='lru' else counters[name]['lru_fallback_retired']) if complete else None)
    return per_task,dict(status='measured' if complete else 'incomplete',issues=issues,
        journal=str(path),journal_sha256=sha(path),writer_audit=writer,
        total_scope_records=len(records),excluded_before_formal_scope_count=len(excluded),
        excluded_before_formal_namespaces=excluded,unmatched_formal_scopes=unmatched,
        mapped_formal_scopes=mapped_scopes,
        duplicate_scopes=duplicates,formal_owners_without_closed_scope=missing_owners,
        observed_formal_counter_sums=totals,expected_formal_counters=expected,
        scope='Only closed scopes at or after the zero-active-scope formal boundary are attributed. '
              'Pure LRU deletion uses retired; lru_fallback_retired applies to the redundancy policy.')


def load_run(root):
    manifest=read(root/'manifest.json')
    result=read(root/'results.json')
    audit=read(root/'execution-audit.json')
    formal=read(root/'formal-events/summary.json')
    require(read(root/'status.json')['status']=='completed',f'{root}: run is not completed')
    names=manifest['launched_task_names']
    require(len(names)==len(set(names))==TASK_COUNT and manifest['tasks']==TASK_COUNT,
            f'{root}: expected the full unique 89-task set')
    require(audit['passed'] is True and audit['tasks']==audit['attempts']==TASK_COUNT and audit['retries']==0,
            f'{root}: completed execution audit does not cover one attempt per task')
    require(result['measurement']==audit,f'{root}: results and execution audit differ')
    require(not audit['cleanup']['remaining_processes'] and not audit['cleanup']['remaining_owned_gpu_pids'],
            f'{root}: owned process cleanup is incomplete')
    outcomes=outcome_map(formal['outcomes'],str(root))
    require(set(outcomes)==set(names),f'{root}: launched and scored task sets differ')
    counts=dict(passed=sum(row['passed'] for row in outcomes.values()),
        scored=sum(row['status']=='scored' for row in outcomes.values()),
        task_timeouts=sum(row['status']=='agent_timeout' for row in outcomes.values()),
        errors=sum(row['status']=='error' for row in outcomes.values()))
    for key,value in counts.items():
        require(formal[key]==result[key]==value,f'{root}: inconsistent {key} count')
    require(formal['expected']==TASK_COUNT and formal['retries']==result['retries']==0,
            f'{root}: formal task count or retry count differs')
    for key in ('outcomes','tasks'):
        actual=outcome_map(result[key],f'{root}/results.{key}')
        require({name:outcome_identity(row) for name,row in actual.items()}==
                {name:outcome_identity(row) for name,row in outcomes.items()},f'{root}: result task details differ')
    durations={task_name(row['task']):row.get('agent_seconds') for row in result['tasks']}
    profile=read(root/'profile.json')
    per_task_gc,scope_audit=gc_attribution(root,outcomes,audit,profile['kvmem']['historical_gc']['policy'])
    metrics=dict(**counts,attempts=TASK_COUNT,pass_at_1=counts['passed']/TASK_COUNT,
        agent_timeout_exceptions=sum(row.get('exception')=='AgentTimeoutError' for row in outcomes.values()),
        unscored_tasks=[name for name,row in outcomes.items() if row['status']!='scored'],
        formal_window_seconds=audit.get('formal_window_seconds'),
        agent_window_seconds=audit.get('agent_window_seconds'),
        summed_agent_seconds=(sum(durations.values()) if all(value is not None for value in durations.values()) else None),
        accepted_completed_response_tokens=audit.get('accepted_completed_response_tokens'),
        completed_response_tokens_per_agent_window_second=audit.get('accepted_completed_response_tokens_per_wall_second'),
        incomplete_native_requests=audit.get('incomplete_native_requests'),
        gc_formal_delta=audit.get('gc_formal_delta'),
        gc_release_scope=audit.get('gc_release_scope','CPU/slot releases include normal task-end cleanup'),
        efficiency=audit.get('efficiency',dict(status='unavailable',reason='No efficiency metrics in execution audit')),
        per_task_gc=per_task_gc,gc_scope_attribution=scope_audit)
    images={}
    for path in (root/'images').glob('*.json'):
        proof=read(path)
        if proof.get('task') in outcomes and proof.get('id'):
            images[proof['task']]=proof['id']
    return dict(root=root,manifest=manifest,profile=profile,job=read(root/'job.json'),
        timing=read(root/'agent-time-policy.json'),audit=audit,outcomes=outcomes,metrics=metrics,images=images)


def verify_pair(baseline,candidate):
    a,b=baseline,candidate
    require(set(a['outcomes'])==set(b['outcomes']),'Paired task sets differ')
    for label in ('dataset','runtime'):
        require(a['manifest']['frozen'][label]==b['manifest']['frozen'][label],f'Frozen {label} differs between policies')
    dataset=a['manifest']['frozen']['dataset']
    for name in a['outcomes']:
        require(f'tasks/{name}/instruction.md' in dataset,f'Missing frozen original instruction: {name}')
        require(any(path.startswith(f'tasks/{name}/tests/') for path in dataset),f'Missing frozen verifier: {name}')
        require(f'tasks/{name}/task.toml' in dataset,f'Missing frozen task resource configuration: {name}')
    require(a['timing']==b['timing'],'Original/effective task deadlines or resource audit differs')
    require(set(a['timing']['tasks'])==set(a['outcomes']),'Deadline audit does not cover all paired tasks')
    for name,row in a['timing']['tasks'].items():
        require(row.get('original_sha256') and row.get('effective_sha256')==dataset[f'tasks/{name}/task.toml'] and
                row.get('verifier_unchanged') is True and row.get('environment_unchanged') is True,
                'Original task resource/deadline proof is incomplete: '+name)
    timing_mode=a['manifest'].get('agent_time_limit','official')
    require(timing_mode in ('official','none','relaxed'),'Unknown Agent timing policy')
    for run in (a,b):
        require(run['manifest'].get('agent_time_limit','official')==timing_mode and
                run['timing'].get('agent_time_limit')==timing_mode,
                'Agent timing mode differs between manifest and policy audit')
        comparable=timing_mode=='official'
        require(run['manifest'].get('official_timing_comparable') is comparable and
                run['timing'].get('official_timing_comparable') is comparable and
                run['audit'].get('agent_time_limit')==timing_mode and
                run['audit'].get('official_timing_comparable') is comparable,
                'Agent timing comparability or execution audit is inconsistent')
        require(run['timing'].get('scope')=='Agent execution only; verifier/setup/tool timeouts retained',
                'Missing audit that only Agent timing changes; other timeout protections must be retained')
        require(run['job'].get('timeout_multiplier',1)==1,'Non-Agent timeout multiplier changed')
        require(run['job'].get('agent_timeout_multiplier') in (None,1),'Agent multiplier outside audited task policy')
        if timing_mode=='relaxed':
            require(run['timing'].get('agent_multiplier')==2.0,'Invalid relaxed Agent multiplier')
            hook=read(run['root']/'formal-deadline-hook.json')
            require(hook.get('agent_time_limit')=='relaxed' and hook.get('original_phase_sha256'),
                    'Missing pause-aware Agent deadline hook')
        for agent in run['job']['agents']:
            require(agent.get('override_timeout_sec') is None and agent.get('max_timeout_sec') is None,
                    'Agent timing override enabled outside audited task policy')
        for name,row in run['timing']['tasks'].items():
            original=row.get('original_agent_timeout_sec')
            require(original is None or (type(original) in (int,float) and isfinite(original) and original>0),
                    'Invalid original Agent deadline: '+name)
            require('effective_agent_timeout_sec' in row and 'original_agent_timeout_sec' in row,
                    'Missing audited Agent deadline values: '+name)
            if timing_mode=='official':
                require(row['original_sha256']==row['effective_sha256'] and
                        row['effective_agent_timeout_sec']==original,
                        'Official Agent deadlines were modified: '+name)
            elif timing_mode=='none':
                require(row['effective_agent_timeout_sec'] is None,
                        'Agent deadline is still enabled in custom timing audit: '+name)
                require((row['original_sha256']==row['effective_sha256'])==(original is None),
                        'Custom timing audit does not show the expected Agent deadline removal: '+name)
                task_path=run['root']/'dataset/tasks'/name/'task.toml'
                require(task_path.is_file() and sha(task_path)==row['effective_sha256'],
                        'Custom timing task does not match its frozen audit: '+name)
                config=tomllib.loads(task_path.read_text())
                require('timeout_sec' not in config.get('agent',{}) and not config.get('steps'),
                        'Custom timing task still has an Agent deadline or unsupported steps: '+name)
            else:
                require(original is not None and isfinite(original*2) and
                        row['effective_agent_timeout_sec']==original*2,
                        'Relaxed Agent deadline must be exactly 2x official: '+name)
                task_path=run['root']/'dataset/tasks'/name/'task.toml'
                require(task_path.is_file() and sha(task_path)==row['effective_sha256'],
                        'Relaxed timing task does not match its frozen audit: '+name)
                config=tomllib.loads(task_path.read_text())
                require(config.get('agent',{}).get('timeout_sec')==original*2 and not config.get('steps'),
                        'Relaxed Agent deadline mismatch or unsupported steps: '+name)
    for key in ('dataset','dataset_commit','harbor_commit','harbor_version','attempts','max_concurrency',
                'container_memory_admission_bytes','output_cap','agent_time_limit','official_timing_comparable',
                'thinking_budget','seed','query_mode'):
        require(a['manifest'].get(key)==b['manifest'].get(key),f'Run manifest setting differs: {key}')
    concurrency=[concurrency_config(run) for run in (a,b)]
    require(concurrency[0]==concurrency[1],'Concurrency configuration differs between policies')
    require(a['manifest']['launched_task_names']==b['manifest']['launched_task_names'],
            'Manifest task queue order differs between policies')
    require(a['manifest']['attempts']==1,'Manifest must declare one attempt per task')
    for key,manifest_key in [('actual_seed','seed'),('output_cap','output_cap'),('thinking_budget','thinking_budget')]:
        require(a['audit'].get(key)==b['audit'].get(key)==a['manifest'][manifest_key],
                'Actual audited sampling differs: '+key)
    require(normalized_paths(a['job'],a['root'])==normalized_paths(b['job'],b['root']),
            'Jobs differ in sampling, task order, timeouts, resources or concurrency')
    require(a['job']['n_attempts']==1 and a['job']['retry']['max_retries']==0,
            'Expected one attempt and no task retry')
    profiles=[deepcopy(run['profile']) for run in (a,b)]
    policies=[]
    for profile in profiles:
        kv=profile['kvmem']
        require(kv['select_budget']==36864 and kv['gen_budget']==16384 and
                kv['repository_blocks']*kv['block_tokens']==262144 and profile['max_num_seqs']==8,
                'Requested K36K/G16K/history256K/eight-sequence geometry differs')
        gc=kv['historical_gc']
        policies.append(dict(policy=gc.pop('policy'),policy_params=gc.pop('policy_params',{})))
    require(profiles[0]==profiles[1],'Model profiles differ beyond historical policy and its parameters')
    require(policies[0]['policy']=='lru' and policies[1]['policy']=='redundancy','Expected baseline LRU and candidate redundancy')
    require(policies[1]['policy_params'].get('thinking_only') is True and
            policies[1]['policy_params'].get('lru_fallback') is True,'Candidate must use thinking redundancy with LRU fallback')
    require(a['images']==b['images'],'Verified task image identities differ')
    return dict(passed=True,full_task_set_equal=True,frozen_dataset_equal=True,frozen_runtime_equal=True,
        concurrency=concurrency[0],
        agent_time_limit=timing_mode,official_timing_comparable=timing_mode=='official',
        timing_scope={
            'none': 'custom local evaluation; Agent execution deadline disabled; verifier/setup/tool timeouts retained; not directly comparable with official timed scores',
            'relaxed': 'custom local evaluation; Agent deadlines 2x official, recorded pauses excluded; verifier/setup/tool timeouts retained; not directly comparable with official timed scores',
            'official': 'official per-task Agent execution deadlines retained'}[timing_mode],
        original_resource_deadline_proof_equal=True,jobs_equal_after_run_path_normalization=True,
        image_identity=dict(status='verified' if len(a['images'])==TASK_COUNT else
                                   'partial' if a['images'] else 'unavailable',
                            verified_tasks=len(a['images']),same_verified_ids=True),
        shared_cpu_cache_cap_bytes=profiles[0]['kvmem']['cpu_read_cache_bytes'],
        common_profile=profiles[0],policies=dict(baseline=policies[0],candidate=policies[1]),
        feature_budget_scope='Policy parameters may reserve different redundancy features within the identical shared CPU cap; '
                             'policy overhead is part of this comparison.')


def paired_subset(names,baseline,candidate):
    paired=dict(both_pass=[],both_fail=[],candidate_only_pass=[],baseline_only_pass=[])
    for name in sorted(names):
        left,right=baseline['outcomes'][name]['passed'],candidate['outcomes'][name]['passed']
        key=('both_pass' if left and right else 'candidate_only_pass' if right else
             'baseline_only_pass' if left else 'both_fail')
        paired[key].append(name)
    n=len(names)
    left=len(paired['both_pass'])+len(paired['baseline_only_pass'])
    right=len(paired['both_pass'])+len(paired['candidate_only_pass'])
    return dict(tasks=sorted(names),task_count=n,baseline_passed=left,candidate_passed=right,
        baseline_pass_rate=left/n if n else None,candidate_pass_rate=right/n if n else None,
        paired_task_lists=paired,paired_counts={key:len(value) for key,value in paired.items()},
        exact_mcnemar=exact_mcnemar(len(paired['candidate_only_pass']),len(paired['baseline_only_pass'])))


def gc_triggered_subsets(baseline,candidate):
    runs=dict(baseline=baseline,candidate=candidate)
    statuses={key:run['metrics']['gc_scope_attribution']['status'] for key,run in runs.items()}
    if any(value!='measured' for value in statuses.values()):
        return dict(status='unavailable',attribution_statuses=statuses,
            reason='Both arms require complete, reconciled per-scope GC attribution before selecting GC-triggered subsets')
    all_names=set(baseline['outcomes'])
    triggered={key:{name for name,row in run['metrics']['per_task_gc'].items() if row['retired']>0}
               for key,run in runs.items()}
    union=triggered['baseline']|triggered['candidate']
    redundancy={name for name,row in candidate['metrics']['per_task_gc'].items()
                if row['counters'].get('redundancy_retired',0)>0}
    usage={}
    for key,run in runs.items():
        usage[key]={}
        for label,names in [('triggered',triggered[key]),('not_triggered',all_names-triggered[key])]:
            passed=sum(run['outcomes'][name]['passed'] for name in names)
            usage[key][label]=dict(tasks=sorted(names),task_count=len(names),passed=passed,
                                  pass_rate=passed/len(names) if names else None)
    return dict(status='measured',triggered_tasks={key:sorted(value) for key,value in triggered.items()},
        per_arm_used_vs_unused=usage,
        union_triggered=paired_subset(union,baseline,candidate),
        neither_triggered=paired_subset(all_names-union,baseline,candidate),
        candidate_redundancy_triggered=paired_subset(redundancy,baseline,candidate),
        scope='These subsets are selected after observing policy behavior and generated trajectories. '
              'They are descriptive exposure analyses, not randomized estimates of causal quality effects.')


def compare(baseline_root,candidate_root):
    baseline,candidate=[load_run(path.resolve()) for path in (baseline_root,candidate_root)]
    validation=verify_pair(baseline,candidate)
    paired=dict(both_pass=[],both_fail=[],candidate_only_pass=[],baseline_only_pass=[])
    task_rows=[]
    for name in sorted(baseline['outcomes']):
        a,b=baseline['outcomes'][name],candidate['outcomes'][name]
        key=('both_pass' if a['passed'] and b['passed'] else 'candidate_only_pass' if b['passed'] else
             'baseline_only_pass' if a['passed'] else 'both_fail')
        paired[key].append(name)
        task_rows.append(dict(task=name,pair_class=key,baseline=outcome_identity(a),candidate=outcome_identity(b)))
    statistic=exact_mcnemar(len(paired['candidate_only_pass']),len(paired['baseline_only_pass']))
    return dict(created_at=time.time(),passed=True,scope='Full 89-task paired pass@1; one fresh attempt per task per policy',
        agent_time_limit=validation['agent_time_limit'],
        official_timing_comparable=validation['official_timing_comparable'],
        timing_scope=validation['timing_scope'],
        baseline=dict(root=str(baseline['root']),policy='lru',**baseline['metrics']),
        candidate=dict(root=str(candidate['root']),policy='thinking redundancy with LRU fallback',**candidate['metrics']),
        accuracy_delta_percentage_points=100*(candidate['metrics']['pass_at_1']-baseline['metrics']['pass_at_1']),
        paired_task_lists=paired,paired_counts={key:len(value) for key,value in paired.items()},
        exact_mcnemar=statistic,tasks=task_rows,validation=validation,
        gc_triggered_subsets=gc_triggered_subsets(baseline,candidate),
        evidence_sha256={label:{name:sha(run['root']/name) for name in
            ('manifest.json','profile.json','job.json','agent-time-policy.json','results.json','execution-audit.json','formal-events/summary.json')}
            for label,run in [('baseline',baseline),('candidate',candidate)]},
        caveats=[validation['timing_scope'],
                 'Each task has n=1 attempted trajectory per policy. Stochastic generation and dynamic batching can change trajectories; '
                 'this paired result does not establish repeatable quality superiority.',
                 'Overall pass@1 uses all 89 attempted tasks as denominator. Infrastructure/unscored errors and Agent timeouts are also reported separately.',
                 'The exact McNemar test conditions on discordant task pairs; its p-value does not replace repeated-run validation.',
                 'Natural-generation token rates include tool waiting and task tails, and exclude tokens from incomplete responses.',
                 'Policy retirements exclude warmup/preflight/smoke; CPU/slot releases also include ordinary task-end cleanup.',
                 'Per-task GC is available only when closed native scopes reconcile with formal aggregate retirement counters and the journal writer reports no errors.',
                 'GC-triggered subsets are selected after observing policy behavior; differences there are descriptive and do not establish causality.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=compare(args.baseline,args.candidate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    temporary.replace(args.output)
    print(json.dumps(dict(passed=True,output=str(args.output),
        baseline_passed=result['baseline']['passed'],candidate_passed=result['candidate']['passed'],
        paired_counts=result['paired_counts'],exact_mcnemar=result['exact_mcnemar']),ensure_ascii=False))


if __name__=='__main__':
    main()
