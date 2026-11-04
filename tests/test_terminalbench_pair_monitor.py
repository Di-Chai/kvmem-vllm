import asyncio
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from scripts import monitor_terminalbench_pair as monitor
from scripts import compare_terminalbench_scored_pairs as wrapper
from scripts.monitor_deepswe_delivery import deliver


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    write(root / 'pair-status.json', dict(status='running', arm='thinking', at=1))
    write(root / 'pair-process.json', dict(pid=1, created=2))
    write(root / 'thinking/status.json', dict(status='formal', at=1))
    write(root / 'lru/status.json', dict(status='prepared'))
    for name in ('queue-process.json', 'formal-process.json', 'model/server-process.json',
                 'bridge-process.json', 'gpu-monitor-process.json'):
        write(root / 'thinking' / name, dict(pid=1, created=2))
    write(root / 'thinking/formal-events/status.json', dict(outcomes=[], active=['working']))


def inspect(root, **kwargs):
    return monitor.inspect(root, probe=lambda _: dict(state='alive'),
                           resource_probe=lambda _: {}, **kwargs)


def test_first_baseline_normal_progress_and_long_tools_remain_quiet(tmp_path):
    fixture(tmp_path)
    baseline = inspect(tmp_path, now=10)
    assert monitor.findings(baseline, {}) == []
    later = inspect(tmp_path, now=100000)
    assert not later['issues'] and not monitor.findings(later, baseline)
    write(tmp_path / 'thinking/formal-events/status.json', dict(outcomes=[
        dict(task='incorrect-answer', trial='trial', status='scored', passed=False,
             rewards={'reward': 0}, exception=None)]))
    assert not monitor.findings(inspect(tmp_path, now=100001), later)


def test_new_failure_alerts_once_without_recovery_noise(tmp_path):
    fixture(tmp_path); previous = inspect(tmp_path)
    write(tmp_path / 'pair-status.json', dict(status='failed', error='engine exited'))
    failed = inspect(tmp_path)
    assert monitor.findings(failed, previous)[0]['code'] == 'pair_failed'
    assert not monitor.findings(failed, failed)
    fixture(tmp_path)
    assert not monitor.findings(inspect(tmp_path), failed)


def test_new_unscored_and_cleanup_failure_are_actionable(tmp_path):
    fixture(tmp_path); previous = inspect(tmp_path)
    write(tmp_path / 'thinking/formal-events/status.json', dict(outcomes=[
        dict(task='task', trial='one', status='error', exception='VerifierTimeoutError')],
        release_errors=[dict(trial='one', error='release failed')]))
    codes = {x['code'] for x in monitor.findings(inspect(tmp_path), previous)}
    assert codes == {'thinking:release_errors', 'thinking:task:one'}


@pytest.mark.parametrize('damage',[None,'raw','authorization','audit'])
def test_only_audited_user_cancellation_is_quiet(tmp_path,damage):
    fixture(tmp_path);arm=tmp_path/'thinking';transition=tmp_path/'user-stop'
    raw_path=arm/'jobs/formal/one/result.json'
    write(raw_path,dict(trial_name='one',verifier_result=None,exception_info=dict(exception_type='CancelledError')))
    write(transition/'authorization.json',dict(human_authorization='explicit stop'))
    write(transition/'audit.json',dict(passed=True,original_completed_evidence_preserved=True,user_cancelled=[
        dict(trial='one',raw_path='jobs/formal/one/result.json',raw_sha256=monitor.sha(raw_path))]))
    write(arm/'user-interruption.json',dict(reason='user_requested_switch_to_lru',cancelled_trials=['one'],
        audit=str(transition/'audit.json'),audit_sha256=monitor.sha(transition/'audit.json'),
        authorization_sha256=monitor.sha(transition/'authorization.json')))
    write(arm/'formal-events/status.json',dict(outcomes=[dict(task='task',trial='one',status='error',
                                                            exception='CancelledError',rewards=None)]))
    if damage=='raw':raw_path.write_text(raw_path.read_text()+' ')
    elif damage=='authorization':write(transition/'authorization.json',dict(human_authorization='different'))
    elif damage=='audit':write(transition/'audit.json',dict(passed=False))
    current=inspect(tmp_path)
    assert bool(current['issues']) is (damage is not None)
    assert current['arms']['thinking']['user_cancelled']==(1 if damage is None else 0)


def test_owned_process_disappearance_and_final_review_shutdown(tmp_path):
    fixture(tmp_path)
    def probe(record):
        return dict(state='identity_changed' if record.get('pid') == 3 else 'alive')
    write(tmp_path / 'thinking/formal-process.json', dict(pid=3, created=4))
    current = monitor.inspect(tmp_path, probe=probe, resource_probe=lambda _: {})
    assert any(x['code'] == 'process:thinking/formal' for x in current['issues'])
    write(tmp_path / 'thinking/status.json', dict(status='needs_final_review', at=1))
    current = monitor.inspect(tmp_path, probe=probe, resource_probe=lambda _: {})
    assert not current['issues']


def test_pid_only_pair_child_is_never_an_owned_identity(tmp_path):
    fixture(tmp_path); (tmp_path / 'pair-process.json').unlink()
    write(tmp_path / 'pair-child.json', dict(pid=999, arm='thinking'))
    calls = []
    value = monitor.inspect(tmp_path, probe=lambda r: (calls.append(r), dict(state='alive'))[1],
                            resource_probe=lambda _: {})
    assert not value['issues'] and not any(r['pid'] == 999 for r in calls)


@pytest.mark.parametrize('change,expected', [
    ('created', 'identity_changed'), ('argv', 'identity_changed'), ('cwd', 'identity_changed'),
    ('zombie', 'exited'), ('denied', 'visibility_denied'), ('gone', 'missing_or_exited'),
    ('valid', 'alive'),
])
def test_identity_checks_creation_command_and_cwd(monkeypatch, tmp_path, change, expected):
    record = dict(pid=123, created=10, argv=['python', 'owned.py'], cwd=str(tmp_path))
    class Process:
        pid = 123
        def create_time(self):
            if change == 'denied':
                raise monitor.psutil.AccessDenied(123)
            if change == 'gone':
                raise monitor.psutil.NoSuchProcess(123)
            return 11 if change == 'created' else 10
        def status(self):
            return monitor.psutil.STATUS_ZOMBIE if change == 'zombie' else 'running'
        def cmdline(self):
            return ['unrelated'] if change == 'argv' else record['argv']
        def cwd(self):
            return '/' if change == 'cwd' else str(tmp_path)
    monkeypatch.setattr(monitor.psutil, 'Process', lambda _: Process())
    assert monitor.owned_process(record)['state'] == expected
    assert monitor.owned_process({'pid': 123})['state'] == 'unverifiable_identity'


def complete_fixture(root):
    fixture(root); comparison = dict(passed=True, validation=dict(passed=True), evidence_sha256={})
    names = [f'task-{index:02d}' for index in range(89)]
    for arm, label in (('thinking', 'candidate'), ('lru', 'baseline')):
        directory = root / arm; cleanup = dict(remaining_processes=[], remaining_owned_gpu_pids=[])
        outcomes = [dict(task='terminal-bench/' + name, trial=name + '__first', status='scored',
                         rewards={'reward': int(i % 2 == 0)}, passed=i % 2 == 0, exception=None)
                    for i, name in enumerate(names)]
        audit = dict(passed=True, tasks=89, attempts=89, retries=0, cleanup=cleanup)
        formal = dict(expected=89, scored=89, passed=45, errors=0, retries=0, outcomes=outcomes)
        write(directory / 'status.json', dict(status='completed'))
        write(directory / 'manifest.json', dict(tasks=89, launched_task_names=names))
        write(directory / 'profile.json', {})
        write(directory / 'job.json', dict(n_attempts=1, retry=dict(max_retries=0)))
        write(directory / 'agent-time-policy.json', dict(agent_time_limit='none'))
        write(directory / 'execution-audit.json', audit)
        write(directory / 'formal-events/summary.json', formal)
        write(directory / 'formal-events/status.json', formal)
        write(directory / 'cleanup.json', cleanup)
        write(directory / 'results.json', dict(**formal, measurement=audit, tasks=outcomes))
        evidence = {}
        for row in outcomes:
            rel = 'jobs/formal/' + row['trial'] + '/result.json'
            write(directory / rel, dict(task_name=row['task'], trial_name=row['trial'],
                                       verifier_result=dict(rewards=row['rewards']), exception_info=None))
            write(directory / 'jobs/formal' / row['trial'] / 'config.json', {})
            evidence[rel] = monitor.sha(directory / rel)
        write(directory / 'evidence-hashes.json', evidence)
        (directory / 'report.txt').write_text('All first-attempt scores verified.')
        write(directory / 'final-review.json', dict(status='completed', expected=89, scored=89,
            report=str(directory / 'report.txt'), report_sha256=monitor.sha(directory / 'report.txt'),
            evidence_sha256=monitor.sha(directory / 'evidence-hashes.json')))
        comparison[label] = dict(root=str(directory), scored=89, passed=45, errors=0)
        comparison['evidence_sha256'][label] = {name: monitor.sha(directory / name) for name in monitor.EVIDENCE_FILES}
    write(root / 'accuracy-comparison.json', comparison)
    write(root / 'pair-status.json', dict(status='completed', result=str(root / 'accuracy-comparison.json')))


def test_full_completion_uses_bound_controller_artifacts(tmp_path):
    fixture(tmp_path); previous = inspect(tmp_path)
    complete_fixture(tmp_path)
    current = inspect(tmp_path)
    assert current['complete'] and not current['issues']
    assert current['completion']['cohorts']['thinking'] == dict(scored=89, unscored=0, passed=45,
        attempts=89, errors=0, task_timeouts=0)
    assert monitor.findings(current, previous)[0]['code'] == 'complete'
    assert not monitor.findings(current, current)


def wrapped_complete_fixture(root):
    complete_fixture(root)
    comparison = monitor.read(root/'accuracy-comparison.json')
    comparison.update(schema=wrapper.SCHEMA, attempts_complete=True, scoring_complete=False,
        analysis_scope=wrapper.ANALYSIS_SCOPE, raw_outcome_evidence={}, operational={},
        gc_triggered_subsets=dict(status='disabled'), analysis_implementation={})
    for label, name in (('wrapper', 'compare_terminalbench_scored_pairs.py'),
                        ('comparator', 'compare_terminalbench_policies.py')):
        source = Path(wrapper.__file__).with_name(name)
        shutil.copyfile(source, root/name)
        comparison['analysis_implementation'][label] = dict(path=str(root/name), sha256=monitor.sha(root/name))
    write(root/'prelaunch-audit.json', dict(comparator_sha256=monitor.sha(root/'compare_terminalbench_policies.py')))
    cohorts = {}
    for arm, label in (('thinking', 'candidate'), ('lru', 'baseline')):
        directory = root/arm; formal = monitor.read(directory/'formal-events/summary.json')
        if arm == 'thinking':
            formal['outcomes'][0].update(status='error', rewards=None, passed=False, exception='RuntimeError')
            formal.update(scored=88, passed=44, errors=1)
        formal['task_timeouts'] = 0
        for path in ('formal-events/summary.json', 'formal-events/status.json'):
            write(directory/path, formal)
        write(directory/'results.json', dict(**formal, tasks=formal['outcomes'],
                                             measurement=monitor.read(directory/'execution-audit.json')))
        evidence = {}; raw_proofs = {}; outcomes = monitor.outcome_map(formal['outcomes'], arm)
        cohorts[label] = outcomes
        for name, row in outcomes.items():
            relative = 'jobs/formal/'+row['trial']+'/result.json'
            event = 'formal-events/'+row['trial']+'.json'
            exception = dict(exception_type='RuntimeError', exception_message='original setup failure',
                             exception_traceback='original traceback') if row['exception'] else None
            raw = dict(task_name=row['task'], trial_name=row['trial'],
                       verifier_result=dict(rewards=row['rewards']), exception_info=exception)
            for path in (relative, event):
                write(directory/path, raw); evidence[path] = monitor.sha(directory/path)
            raw_proofs[name] = dict(trial=row['trial'], path=relative, sha256=evidence[relative],
                event_path=event, event_sha256=evidence[event], exception_info=exception)
        comparison['raw_outcome_evidence'][label] = raw_proofs
        write(directory/'evidence-hashes.json', evidence)
        review = monitor.read(directory/'final-review.json')
        review.update(scored=formal['scored'], task_timeouts=0,
            infrastructure_or_unscored_errors=formal['errors'], evidence_sha256=monitor.sha(directory/'evidence-hashes.json'))
        write(directory/'final-review.json', review)
        unscored = sorted(name for name, row in outcomes.items() if row['status'] != 'scored')
        counts = dict(attempts=89, scored=formal['scored'], passed=formal['passed'],
                      unscored=len(unscored), errors=formal['errors'], task_timeouts=0)
        comparison[label] = dict(root=str(directory), unscored_tasks=unscored, **counts)
        comparison['operational'][label] = dict(**counts, success_fraction_all_attempts=counts['passed']/89,
            unscored_tasks=[dict(task=name, outcome=monitor.outcome_identity(outcomes[name]),
                                exception_info=raw_proofs[name]['exception_info']) for name in unscored])
        comparison['evidence_sha256'][label] = {name: monitor.sha(directory/name) for name in monitor.EVIDENCE_FILES}
    comparison['quality'], comparison['excluded_tasks'] = wrapper.quality_analysis(
        cohorts['baseline'], cohorts['candidate'], monitor.exact_mcnemar)
    write(root/'accuracy-comparison.json', comparison)
    return comparison


def test_all_first_attempts_with_explicit_unscored_complete_without_inventing_a_loss(tmp_path):
    fixture(tmp_path); previous = inspect(tmp_path)
    wrapped_complete_fixture(tmp_path)
    current = inspect(tmp_path)
    assert current['complete'], current['issues']
    assert current['completion']['attempts_complete'] and not current['completion']['scoring_complete']
    assert current['completion']['jointly_scored_tasks'] == 88
    assert current['completion']['cohorts']['thinking']['unscored'] == 1
    message = next(item['message'] for item in monitor.findings(current, previous) if item['code'] == 'complete')
    assert '89 original first attempts' in message and '"thinking": 1' in message
    assert 'jointly scored quality pairs: 88' in message
    assert monitor.findings(current, current) == []


@pytest.mark.parametrize('change', ['legacy', 'quality', 'coverage', 'exclusions', 'scoring_complete',
    'operational', 'wrapper_hash', 'comparator_hash', 'prelaunch_hash', 'exception_details',
    'raw_cohort', 'unscored_list', 'unscored_details', 'gc_quality', 'final_review'])
def test_wrapper_completion_rejects_misleading_or_unbound_analysis(tmp_path, change):
    comparison = wrapped_complete_fixture(tmp_path)
    if change == 'legacy':
        comparison.pop('schema')
    elif change == 'quality':
        comparison['quality']['paired_counts']['baseline_only_pass'] += 1
    elif change == 'coverage':
        comparison['quality']['coverage_fraction'] = 1
    elif change == 'exclusions':
        comparison['excluded_tasks'] = []
    elif change == 'scoring_complete':
        comparison['scoring_complete'] = True
    elif change == 'operational':
        comparison['operational']['candidate']['success_fraction_all_attempts'] = 44/88
    elif change in ('wrapper_hash', 'comparator_hash'):
        comparison['analysis_implementation'][change.removesuffix('_hash')]['sha256'] = 'changed'
    elif change == 'prelaunch_hash':
        write(tmp_path/'prelaunch-audit.json', dict(comparator_sha256='changed'))
    elif change == 'exception_details':
        comparison['raw_outcome_evidence']['candidate']['task-00']['exception_info']['exception_traceback'] = 'changed'
    elif change == 'raw_cohort':
        comparison['raw_outcome_evidence']['candidate']['extra-task'] = {}
    elif change == 'unscored_list':
        comparison['candidate']['unscored_tasks'] = []
    elif change == 'unscored_details':
        comparison['operational']['candidate']['unscored_tasks'] = []
    elif change == 'gc_quality':
        comparison['gc_triggered_subsets']['status'] = 'measured'
    elif change == 'final_review':
        review = monitor.read(tmp_path/'thinking/final-review.json')
        review['infrastructure_or_unscored_errors'] = 0
        write(tmp_path/'thinking/final-review.json', review)
    write(tmp_path/'accuracy-comparison.json', comparison)
    current = inspect(tmp_path)
    assert not current['complete'] and any(item['code'] == 'completion_invalid' for item in current['issues'])


@pytest.mark.parametrize('change', ['missing_audit', 'raw_reward', 'extra_attempt', 'unfinished_attempt', 'report',
                                    'comparison_path', 'comparison_failed', 'missing_score',
                                    'audit_retry', 'inherited', 'evidence', 'wrong_arm'])
def test_incomplete_or_changed_completion_never_announced(tmp_path, change):
    complete_fixture(tmp_path); arm = tmp_path / 'thinking'
    if change == 'missing_audit':
        (arm / 'execution-audit.json').unlink()
    elif change == 'raw_reward':
        path = next((arm / 'jobs/formal').glob('*/result.json'))
        write(path, dict(trial_name=path.parent.name, verifier_result={'rewards': {'reward': 0}}))
    elif change == 'extra_attempt':
        write(arm / 'jobs/formal/task-00__second/result.json', {})
    elif change == 'unfinished_attempt':
        write(arm / 'jobs/formal/task-00__second/config.json', {})
    elif change == 'report':
        (arm / 'report.txt').write_text('changed')
    elif change == 'comparison_path':
        write(tmp_path / 'pair-status.json', dict(status='completed', result='/tmp/foreign-comparison.json'))
    elif change in ('comparison_failed', 'wrong_arm'):
        value = monitor.read(tmp_path / 'accuracy-comparison.json')
        if change == 'wrong_arm':
            value['candidate']['root'] = str(tmp_path / 'lru')
        else:
            value['passed'] = False
        write(tmp_path / 'accuracy-comparison.json', value)
    elif change == 'missing_score':
        value = monitor.read(arm / 'formal-events/summary.json'); value['outcomes'].pop()
        write(arm / 'formal-events/summary.json', value)
    elif change == 'audit_retry':
        value = monitor.read(arm / 'execution-audit.json'); value['retries'] = 1
        write(arm / 'execution-audit.json', value)
    elif change == 'inherited':
        write(arm / 'preserved-outcomes.json', [dict(task='old-run')])
    else:
        write(arm / 'evidence-hashes.json', {})
    current = inspect(tmp_path)
    assert not current['complete'] and any(x['code'] == 'completion_invalid' for x in current['issues'])


def test_resource_pressure_is_a_threshold_event_not_progress_timer(tmp_path):
    fixture(tmp_path)
    value = monitor.inspect(tmp_path, probe=lambda _: dict(state='alive'),
        resource_probe=lambda _: dict(memory_available_bytes=1024, disk_free_bytes=2**40))
    assert {x['code'] for x in value['issues']} == {'resource:memory_available_bytes'}


def inspect_lru(root, **kwargs):
    return monitor.inspect(root,selected_arms=('lru',),probe=lambda _:dict(state='alive'),
                           resource_probe=lambda _: {},**kwargs)


def test_lru_scope_excludes_old_thinking_failures(tmp_path):
    fixture(tmp_path)
    write(tmp_path/'pair-status.json',dict(status='running',arm='lru'))
    write(tmp_path/'thinking/status.json',dict(status='failed',error='old failure'))
    write(tmp_path/'thinking/formal-events/status.json',dict(outcomes=[dict(trial='old',task='old',
        status='error',exception='CancelledError')],network_errors=['old network failure']))
    value=inspect_lru(tmp_path)
    assert value['scope']=='lru' and set(value['arms'])=={'lru'} and not value['issues']
    message=monitor.event_message(value,[dict(code='lru:new',message='new LRU event')])
    assert message.startswith('【Terminal Bench LRU 巡检】')
    assert 'old failure' not in message and 'old network failure' not in message


def test_lru_scope_keeps_new_task_failure_alerts(tmp_path):
    fixture(tmp_path);previous=inspect_lru(tmp_path)
    write(tmp_path/'lru/formal-events/status.json',dict(outcomes=[dict(trial='lru-first',task='new-task',
        status='error',exception='RuntimeError')]))
    current=inspect_lru(tmp_path)
    assert [e['code'] for e in monitor.findings(current,previous)]==['lru:task:lru-first']
    assert not monitor.findings(current,current)


def test_lru_completion_notice_precedes_pending_comparison_and_is_deduplicated(tmp_path):
    complete_fixture(tmp_path)
    write(tmp_path/'pair-status.json',dict(status='running',arm='lru'))
    value=inspect_lru(tmp_path)
    assert value['selected_arm_complete'] is True and value['complete'] is False
    assert value['selected_arm_completion']['scored']==89
    previous=dict(issues=[],complete=False,selected_arm_complete=False)
    events=monitor.findings(value,previous)
    assert len(events)==1 and events[0]['code']=='selected_arm_complete'
    assert 'passed 45' in events[0]['message'] and 'comparison is pending' in events[0]['message']
    assert not monitor.findings(value,value)


def test_lru_final_comparison_notice_does_not_repeat_arm_completion(tmp_path):
    complete_fixture(tmp_path);value=inspect_lru(tmp_path)
    events=monitor.findings(value,dict(issues=[],complete=False,selected_arm_complete=False))
    assert len(events)==1 and events[0]['code']=='complete'
    assert events[0]['message'].startswith('LRU test and paired comparison are complete')


@pytest.mark.parametrize('damage',['raw_result','extra_attempt','report','cleanup'])
def test_lru_completion_requires_original_attempts_and_final_audit(tmp_path,damage):
    complete_fixture(tmp_path);write(tmp_path/'pair-status.json',dict(status='running',arm='lru'))
    root=tmp_path/'lru'
    if damage=='raw_result':
        p=next((root/'jobs/formal').glob('*/result.json'));p.write_text(p.read_text()+' ')
    elif damage=='extra_attempt':write(root/'jobs/formal/extra/result.json',{})
    elif damage=='report':(root/'report.txt').write_text('changed report')
    else:write(root/'cleanup.json',dict(remaining_processes=[123],remaining_owned_gpu_pids=[]))
    value=inspect_lru(tmp_path)
    assert value['selected_arm_complete'] is False
    assert any(e['code']=='lru:completion_invalid' for e in value['issues'])


def test_lru_scope_is_bound_in_durable_monitor_state(tmp_path):
    a=args(tmp_path/'run');a.arm='lru'
    state=monitor.step(a,{},now=10,inspector=lambda *_,**__:snapshot())
    assert state['binding']['arm']=='lru'
    a.arm='thinking'
    with pytest.raises(ValueError,match='another root/thread'):
        monitor.step(a,state,now=11,inspector=lambda *_,**__:snapshot())


def args(root):
    state = root.parent / 'monitor-state'; state.mkdir(exist_ok=True)
    return SimpleNamespace(root=root, state_dir=state, thread_id='authorized-thread', socket=Path('/fake'), expires_after=604800)


def snapshot(issues=None, complete=False):
    return dict(at=10, issues=issues or [], complete=complete, arms={}, pair_status='running')


def test_outbox_is_durable_and_uncertain_delivery_reuses_client_id(tmp_path):
    a = args(tmp_path / 'run'); calls = []; current = snapshot()
    async def notifier(thread, socket, message, key, **kwargs):
        persisted = monitor.read(a.state_dir / 'state.json')
        assert persisted['outbox'][0]['key'] == key
        calls.append(key)
        if len(calls) == 1:
            raise TimeoutError('reply lost after queue/add')
        return dict(delivery='persisted_user_message')
    inspector = lambda *_, **__: current
    state = monitor.step(a, {}, now=10, inspector=inspector, notifier=notifier)
    assert not calls
    current = snapshot([dict(code='failed', message='failed')])
    state = monitor.step(a, state, now=20, inspector=inspector, notifier=notifier)
    assert state['outbox'] and state['delivery_error']
    state = monitor.step(a, monitor.read(a.state_dir / 'state.json'), now=30, inspector=inspector, notifier=notifier)
    assert len(calls) == 2 and calls[0] == calls[1] and not state['outbox']
    monitor.step(a, state, now=40, inspector=inspector, notifier=notifier)
    assert len(calls) == 2


def test_new_event_while_delivery_pending_is_not_lost(tmp_path):
    a = args(tmp_path / 'run'); current = snapshot(); calls = []
    async def fail(*values, **kwargs):
        calls.append(values[3]); raise ConnectionError('offline')
    inspector = lambda *_, **__: current
    state = monitor.step(a, {}, now=1, inspector=inspector, notifier=fail)
    current = snapshot([dict(code='one', message='one')])
    state = monitor.step(a, state, now=2, inspector=inspector, notifier=fail)
    current = snapshot([dict(code='one', message='one'), dict(code='two', message='two')])
    state = monitor.step(a, state, now=3, inspector=inspector, notifier=fail)
    assert len(state['outbox']) == 2 and calls[0] == calls[1]


def test_expiry_is_fixed_across_restart_and_does_not_touch_experiment(tmp_path):
    a = args(tmp_path / 'run'); a.expires_after = 10; calls = []
    inspector = lambda *_, **__: (calls.append(True), snapshot())[1]
    state = monitor.step(a, {}, now=100, inspector=inspector)
    a.expires_after = 999999
    state = monitor.step(a, state, now=110, inspector=inspector)
    assert state['status'] == 'expired' and state['expires_at'] == 110 and len(calls) == 1
    assert not a.root.exists()


def test_monitor_state_cannot_be_retargeted_silently(tmp_path):
    a = args(tmp_path / 'run'); state = monitor.step(a, {}, now=1, inspector=lambda *_, **__: snapshot())
    a.thread_id = 'different'
    with pytest.raises(ValueError, match='another root/thread'):
        monitor.step(a, state, now=2)


def test_active_chat_notice_is_queued_without_interrupt_or_turn_start():
    calls = []
    async def rpc(method, params):
        calls.append((method, params))
        if method in ('thread/turns/list', 'thread/queue/list'):
            return dict(data=[])
        if method == 'thread/queue/add':
            return dict(queuedSubmission=dict(id='queued'))
        if method == 'thread/read':
            return dict(thread=dict(status=dict(type='active')))
        raise AssertionError('Unexpected interrupt/start RPC: ' + method)
    result = asyncio.run(deliver(rpc, 'authorized-thread', monitor.PREFIX + '\n快照：\n{"at":1}', 'stable-id', monitor.PREFIX))
    assert result['started'] is False
    assert [name for name, _ in calls] == ['thread/turns/list', 'thread/queue/list', 'thread/queue/add', 'thread/read']
