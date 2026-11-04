"""Only explicit human interruption may change the candidate stopping rule."""
from copy import deepcopy
import json

import pytest

from scripts import run_terminalbench_user_switch as switch
from test_terminalbench_network_audit_recovery import campaign as network_campaign
from test_terminalbench_network_audit_recovery import read, sha, write
from test_terminalbench_setup_audit_recovery import campaign as setup_campaign
from test_terminalbench_recorder_audit_recovery import campaign as recorder_campaign
from test_terminalbench_query_audit_recovery import campaign as query_campaign, jsonlines


@pytest.fixture
def campaign(query_campaign):
    c=query_campaign;arm=c['arm'];transition=c['pair']/'user-stop';transition.mkdir()
    for index,trial in zip((5,6,7),sorted(switch.AUTHORIZED)):
        old=c['trials'][index];raw=deepcopy(c['results'][index]);name=trial.split('__')[0]
        (arm/'jobs/formal'/old).rename(arm/'jobs/formal'/trial)
        (arm/'formal-events'/(old+'.json')).unlink()
        raw.update(trial_name=trial,task_name='terminal-bench/'+name,verifier=None,verifier_result=None,
                   exception_info=dict(exception_type='CancelledError',exception_message='',exception_traceback='cancelled'))
        raw['config']['trial_name']=trial;raw['config']['task']['path']=str(arm/'dataset/tasks'/name)
        c['trials'][index]=trial;c['names'][index]=name;c['results'][index]=raw
        c['outcomes'][index]=dict(task=raw['task_name'],trial=trial,rewards=None,exception='CancelledError',status='error',passed=False)
        events=[]
        for row in c['events']:
            row=deepcopy(row)
            if row.get('trial')==old:
                if row['event']=='verification-start':continue
                row.update(trial=trial,task=raw['task_name'])
            events.append(row)
        events.extend([dict(event='cancel',trial=trial,at=12),dict(event='empty_networks_reclaimed',trial=trial,at=21,count=1)])
        c['events']=events
        write(arm/'formal-events/network-cleanup'/trial/'network.removed.json',
              dict(status='removed',retained_filesystem_diffs_unchanged=True))
    # Match the production cutoff's 53 successes without changing missing rewards.
    needed=53-sum(row['passed'] for row in c['outcomes'])
    for index in range(8,89):
        if needed and not c['outcomes'][index]['passed']:
            c['outcomes'][index].update(passed=True,rewards=dict(reward=1))
            c['results'][index]['verifier_result']=dict(rewards=dict(reward=1));needed-=1
    assert needed==0
    for raw in c['results']:
        trial=raw['trial_name']
        write(arm/'jobs/formal'/trial/'result.json',raw)
        write(arm/'jobs/formal'/trial/'config.json',raw['config'])
        write(arm/'formal-events'/(trial+'.json'),raw)
    jsonlines(arm/'formal-events/events.jsonl',c['events'])
    status=read(arm/'formal-events/status.json');status.update(outcomes=c['outcomes'],passed=53)
    write(arm/'formal-events/status.json',status)
    aggregate=read(arm/'jobs/formal/result.json');aggregate['finished_at']=None
    aggregate['stats'].update(n_errored_trials=7,n_cancelled_trials=3)
    write(arm/'jobs/formal/result.json',aggregate)
    (arm/'formal-events/harbor-result.json').unlink()
    for folder in (arm,c['lru']):
        manifest=read(folder/'manifest.json');manifest['launched_task_names']=c['names'];write(folder/'manifest.json',manifest)
    write(arm/'status.json',dict(status='failed',error="RuntimeError('formal exited -2; preserve logs, no retry')"))
    originals={}
    for trial in set(c['trials'])-switch.AUTHORIZED:
        for relative in (f'thinking/jobs/formal/{trial}/result.json',f'thinking/formal-events/{trial}.json',f'thinking/jobs/formal/{trial}/config.json'):
            originals[relative]=sha(c['pair']/relative)
    auth=dict(human_authorization=switch.AUTHORIZATION,active_at_stop=sorted(switch.AUTHORIZED),
        authorized_trials=sorted(switch.AUTHORIZED),task_retries_allowed=False,completed_at_stop=86,
        scored_at_stop=82,passed_at_stop=53,completed_evidence_sha256=originals,at=0,
        formal_identity=read(arm/'formal-process.json'),original_pair=read(c['pair']/'pair-process.json'),
        original_suite=read(arm/'queue-process.json'))
    write(transition/'authorization.json',auth)
    write(transition/'stop-signal.json',dict(signal='SIGINT',identity=auth['formal_identity'],at=1,only_remaining_trials=sorted(switch.AUTHORIZED)))
    c['transition']=transition
    return c


def test_explicit_stop_preserves_all_scores_and_raw_cancellations(campaign):
    c=campaign;r=switch.audit(c['pair'],c['transition'],live_probe=lambda _:False)
    assert r['summary']['expected']==89 and r['summary']['scored']==82 and r['summary']['passed']==53
    assert r['summary']['errors']==7 and r['summary']['cancelled_by_user']==3
    assert len(r['user_cancelled'])==3 and all(x['reward'] is None for x in r['user_cancelled'])
    assert r['missing_rewards_imputed'] is False


@pytest.mark.parametrize('damage',['authorization','changed_score','live_process','missing_release',
    'lru_attempted','imputed_cancellation','wrong_aggregate','existing_summary'])
def test_switch_refuses_unapproved_or_incomplete_transition(campaign,damage):
    c=campaign;arm=c['arm'];transition=c['transition']
    if damage=='authorization':
        a=read(transition/'authorization.json');a['human_authorization']='keep waiting';write(transition/'authorization.json',a)
    elif damage=='changed_score':
        p=next((arm/'jobs/formal'/t/'result.json') for t in c['trials'] if t not in switch.AUTHORIZED)
        p.write_text(p.read_text()+' ')
    elif damage=='missing_release':
        rows=[r for r in c['events'] if not(r.get('trial') in switch.AUTHORIZED and r['event']=='kv_released')]
        jsonlines(arm/'formal-events/events.jsonl',rows)
    elif damage=='lru_attempted':(c['lru']/'started').write_text('started')
    elif damage=='imputed_cancellation':
        trial=next(iter(switch.AUTHORIZED));p=arm/'jobs/formal'/trial/'result.json'
        raw=read(p);raw['verifier_result']=dict(rewards=dict(reward=0));write(p,raw);write(arm/'formal-events'/(trial+'.json'),raw)
    elif damage=='wrong_aggregate':
        p=arm/'jobs/formal/result.json';raw=read(p);raw['stats']['n_cancelled_trials']=0;write(p,raw)
    elif damage=='existing_summary':write(arm/'formal-events/summary.json',dict(expected=89))
    with pytest.raises((ValueError,KeyError,OSError)):
        switch.audit(c['pair'],transition,live_probe=lambda _:damage=='live_process')
