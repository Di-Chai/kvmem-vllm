"""Runtime failures retain their full operational outcome and missing reward."""
from copy import deepcopy
import json

import pytest

from scripts import recover_terminalbench_query_audit as recovery
from scripts import recover_terminalbench_recorder_audit as recorder
from scripts import terminalbench_query_incident as incident
from test_terminalbench_network_audit_recovery import campaign as network_campaign
from test_terminalbench_network_audit_recovery import read, sha, write
from test_terminalbench_setup_audit_recovery import campaign as setup_campaign
from test_terminalbench_recorder_audit_recovery import campaign as recorder_campaign


def jsonlines(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row)+'\n' for row in values))


@pytest.fixture
def campaign(recorder_campaign, monkeypatch):
    c = recorder_campaign; arm = c['arm']; index = 4
    monkeypatch.setattr(recovery, 'INCIDENTS', recorder.INCIDENTS)
    monkeypatch.setattr(recovery, 'ADAPTER_SHA', recorder.ADAPTER_SHA)
    monkeypatch.setattr(recovery, 'ANALYSIS_SHA', recorder.ANALYSIS_SHA)
    item = deepcopy(incident.INCIDENT); trial, owner = item['trial'], item['owner']
    old = c['trials'][index]; name = trial.split('__')[0]
    (arm/'jobs/formal'/old).rename(arm/'jobs/formal'/trial)
    (arm/'formal-events'/(old+'.json')).unlink()
    raw = deepcopy(c['results'][index])
    raw.update(trial_name=trial, task_name=item['task'], verifier=None, verifier_result=None,
        agent_result=dict(metadata=dict(n_episodes=635)),
        exception_info=dict(exception_type='ValueError', exception_message=incident.ERROR,
                            exception_traceback='terminalbench_query.py: source.proof(messages)'))
    raw['config']['trial_name'] = trial; raw['config']['task']['path'] = str(arm/'dataset/tasks'/name)
    for path in (arm/'jobs/formal'/trial/'result.json', arm/'formal-events'/(trial+'.json')):
        write(path, raw)
    write(arm/'jobs/formal'/trial/'config.json', raw['config'])
    trajectory = arm/'jobs/formal'/trial/'agent/trajectory.json'
    write(trajectory, dict(steps=[{}]*635))
    item.update(raw_sha256=sha(arm/'jobs/formal'/trial/'result.json'),
                event_sha256=sha(arm/'formal-events'/(trial+'.json')), trajectory_sha256=sha(trajectory))
    monkeypatch.setattr(incident, 'INCIDENT', item)
    c['names'][index] = name; c['trials'][index] = trial; c['results'][index] = raw
    c['outcomes'][index] = dict(task=item['task'], trial=trial, rewards=None, exception='ValueError', status='error', passed=False)
    own = []
    for row in c['events']:
        row = deepcopy(row)
        if row.get('trial') == old:
            if row['event'] == 'verification-start': continue
            row.update(trial=trial, task=item['task'])
            if row['event'] == 'kv_released': row['owners'] = [owner]
        own.append(row)
    own.extend([dict(event='environment-start', at=10.5, trial=trial),
                dict(event='empty_networks_reclaimed', at=21, trial=trial, count=1)])
    c['events'] = own; jsonlines(arm/'formal-events/events.jsonl', own)
    status = read(arm/'formal-events/status.json'); status.update(outcomes=c['outcomes'], passed=42)
    write(arm/'formal-events/status.json', status)
    harbor = read(arm/'formal-events/harbor-result.json'); harbor['trial_results'] = c['results']
    harbor['stats']['n_errored_trials'] = 4
    write(arm/'formal-events/harbor-result.json', harbor)
    write(arm/'jobs/formal/result.json', {k:v for k,v in harbor.items() if k!='trial_results'})
    bridge = [dict(owner=owner, turn=n) for n in range(1,636)]
    bridge += [dict(owner=owner, error=incident.STREAM_ERROR)]*9
    bridge += [dict(owner=owner, event=e, session_id=trial+'__agent') for e in ('session_opened','history_released')]
    jsonlines(arm/'bridge/requests.jsonl', bridge)
    sources = [dict(event='task_query_bound', session_id=trial+'__agent')]
    sources += [dict(event='query_source', session_id=trial+'__agent', query_user_ordinal=0,
                     query_message_id='task-query', user_messages_sha256='wire-hash')]*638
    jsonlines(arm/'formal-query-sources.jsonl', sources)
    prefix = arm/'bridge'/owner
    request = dict(messages=[dict(role='user', content='instruction')])
    write(prefix/'00635/request.json', request)
    write(prefix/'00635/native.json', dict(finish_reason='length', usage=dict(completion_tokens=131072), token_ids=[1]*131072))
    write(prefix/'00635/accepted.json', dict(token_ids=[1]*131072))
    followup = dict(messages=request['messages']+[dict(role='assistant',content='truncated'),
        dict(role='user',content='ERROR!! NONE of the actions exceeded output')],
        kvmem_query_provenance=dict(user_messages_sha256='wire-hash'))
    for n in range(9):
        directory = prefix/('00636' if n == 0 else '00636-attempt-'+str(n))
        write(directory/'request.json', followup); write(directory/'plan.json', {})
    adapter = arm/'runtime/scripts/terminalbench_query.py'; adapter.write_text('# frozen query adapter\n')
    for folder in (arm,c['lru']):
        manifest = read(folder/'manifest.json'); manifest['launched_task_names'] = c['names']
        if folder == arm: manifest['frozen']['runtime']['scripts/terminalbench_query.py'] = sha(adapter)
        write(folder/'manifest.json', manifest)
    proof = dict(at=16, status='checked', project=trial.lower()+'__env',
        network=dict(Id=item['network'],Containers={}), retained_containers={},
        environment_dir=str(arm/'dataset/tasks'/name/'environment'))
    directory = arm/'formal-events/network-cleanup'/trial
    write(directory/(item['network']+'.json'), proof)
    write(directory/(item['network']+'.removed.json'), dict(proof,status='removed',finished_at=20,
                                                           retained_filesystem_diffs_unchanged=True))
    c.update(query=item, query_prefix=prefix, bridge_rows=bridge)
    return c


def test_four_original_errors_preserved_in_complete_first_attempt_cohort(campaign):
    c = campaign
    result = recovery.validate_results(c['arm'])
    assert result['summary']['scored'] == 85 and result['summary']['errors'] == 4
    assert len(result['summary']['outcomes']) == 89
    proofs = result['unscored_error_preservation']
    assert len(proofs) == 4 and all(p['reward'] is None for p in proofs)
    assert proofs[-1]['classification'] == 'agent_runtime_error'
    assert proofs[-1]['model_execution_occurred'] is True


@pytest.mark.parametrize('damage', ['raw_bytes','trajectory','verification','owner_release','bridge_error',
    'query_source','successful_recovery','different_followup','length_result','network','frozen_query'])
def test_runtime_error_requires_specific_original_evidence(campaign, damage):
    c = campaign; arm = c['arm']; item = c['query']; trial = item['trial']
    if damage == 'raw_bytes':
        p=arm/'jobs/formal'/trial/'result.json'; p.write_text(p.read_text()+' ')
    elif damage == 'trajectory': write(arm/'jobs/formal'/trial/'agent/trajectory.json',dict(steps=[]))
    elif damage == 'verification': c['events'].append(dict(event='verification-start',trial=trial))
    elif damage == 'owner_release':
        next(r for r in c['events'] if r.get('trial')==trial and r['event']=='kv_released')['owners']=[]
    elif damage == 'bridge_error':
        c['bridge_rows'][-4]['error']='different error'; jsonlines(arm/'bridge/requests.jsonl',c['bridge_rows'])
    elif damage == 'query_source': jsonlines(arm/'formal-query-sources.jsonl',[])
    elif damage == 'successful_recovery': write(c['query_prefix']/'00636/accepted.json',dict(token_ids=[1]))
    elif damage == 'different_followup': write(c['query_prefix']/'00636/request.json',dict(messages=[]))
    elif damage == 'length_result': write(c['query_prefix']/'00635/native.json',dict(finish_reason='stop'))
    elif damage == 'network':
        write(arm/'formal-events/network-cleanup'/trial/(item['network']+'.removed.json'),dict(status='removed'))
    elif damage == 'frozen_query': (arm/'runtime/scripts/terminalbench_query.py').write_text('# changed\n')
    with pytest.raises((ValueError, KeyError, OSError)):
        incident.validate(arm,c['events'])


@pytest.mark.parametrize('damage', ['fifth_error','missing_agent','extra_verifier','harbor_errors'])
def test_runtime_exception_does_not_relax_remaining_tasks(campaign, damage):
    c=campaign;arm=c['arm']
    if damage=='fifth_error':
        status=read(arm/'formal-events/status.json');status['outcomes'][5].update(rewards=None,status='error',exception='ValueError',passed=False)
        write(arm/'formal-events/status.json',status)
    elif damage in ('missing_agent','extra_verifier'):
        rows=c['events'];t=c['query']['trial']
        if damage=='missing_agent': rows=[r for r in rows if not(r.get('trial')==t and r['event']=='agent-start')]
        else: rows=rows+[dict(event='verification-start',trial=t)]
        jsonlines(arm/'formal-events/events.jsonl',rows)
    else:
        harbor=read(arm/'formal-events/harbor-result.json');harbor['stats']['n_errored_trials']=3
        write(arm/'formal-events/harbor-result.json',harbor)
    with pytest.raises((ValueError, KeyError)):
        recovery.validate_results(arm)


def test_recovery_keeps_runtime_error_and_never_relaunches_completed_task(campaign):
    c=campaign;pair=c['pair'];state=c['recovery']
    failed=read(pair/'pair-status.json');write(pair/'pair-status.json',dict(status='running',arm='thinking',started_at=1))
    plan=recovery.prepare(pair,state,c['analysis']);write(pair/'pair-status.json',failed)
    audited=recovery.audit(pair,state,plan,live_probe=lambda _:False)
    hashes=audited['source_hashes']

    def finalize(arm,_):
        summary=read(arm/'formal-events/summary.json')
        assert summary['errors']==4 and summary['scored']==85
        assert summary['derived']['no_reward_imputed']
        write(arm/'status.json',dict(status='completed'))
        write(arm/'results.json',dict(scored=85,errors=4))

    recovery.recover(pair,state,plan,finalizer=finalize,live_probe=lambda _:False)
    assert all(sha(c['arm']/name)==digest for name,digest in hashes.items())
    assert not (c['lru']/'started').exists()
    assert read(state/'original/pair-status.json')['status']=='failed'
