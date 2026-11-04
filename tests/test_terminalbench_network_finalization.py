from copy import deepcopy
import json
import shutil

import pytest

from scripts import finalize_terminalbench_network_reconciliation as recovery
from test_terminalbench_network_audit_recovery import campaign as base_campaign, write, read, sha


@pytest.fixture
def campaign(base_campaign):
    c=base_campaign;arm=c['arm']
    for index,raw in enumerate(c['results']):
        raw['verifier']=dict(started_at='start',finished_at='finish')
        if index in (1,2):
            raw['verifier_result']=None
            kind='VerifierTimeoutError' if index==1 else 'BadGatewayError'
            raw['exception_info']=dict(exception_type=kind,exception_message='original failure',exception_traceback='original trace')
            c['outcomes'][index]=recovery.original_outcome(raw)
            if index==2:raw['verifier']=None
        trial=raw['trial_name'];write(arm/'jobs/formal'/trial/'result.json',raw);write(arm/'formal-events'/(trial+'.json'),raw)
    status=read(arm/'formal-events/status.json');status.update(outcomes=c['outcomes'],passed=sum(r['passed'] for r in c['outcomes']))
    write(arm/'formal-events/status.json',status)
    harbor=read(arm/'formal-events/harbor-result.json');harbor['trial_results']=c['results'];harbor['stats']['n_errored_trials']=2
    write(arm/'formal-events/harbor-result.json',harbor)
    write(arm/'jobs/formal/result.json',{k:v for k,v in harbor.items() if k!='trial_results'})
    c['events']=[r for r in c['events'] if not(r.get('trial')==c['trials'][2] and r['event']=='verification-start')]
    (arm/'formal-events/events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in c['events']))
    (arm/'bridge').mkdir();(arm/'bridge/requests.jsonl').write_text('')
    c['plan']=dict(network=dict(error=c['error']))
    return c


def test_integrity_audit_retains_original_runtime_and_verifier_failures(campaign):
    c=campaign;result=recovery.audit_results(c['arm'],c['plan'])
    assert result['summary']['expected']==89 and result['summary']['scored']==87 and result['summary']['errors']==2
    failures=result['original_exceptions_preserved']
    assert {x['exception_info']['exception_type'] for x in failures}=={'VerifierTimeoutError','BadGatewayError'}
    assert all(x['reward'] is None and 'not presumed policy-independent' in x['causal_classification'] for x in failures)


@pytest.mark.parametrize('damage',['missing_result','extra_attempt','changed_copy','imputed_reward','retries',
    'unfinished','missing_kv','extra_cleanup_error','cancelled','missing_exception'])
def test_integrity_audit_cannot_hide_or_replace_original_outcomes(campaign,damage):
    c=campaign;arm=c['arm'];trial=c['trials'][1]
    if damage=='missing_result':(arm/'jobs/formal'/trial/'result.json').unlink()
    elif damage=='extra_attempt':write(arm/'jobs/formal/second/result.json',{})
    elif damage=='changed_copy':write(arm/'formal-events'/(trial+'.json'),{})
    elif damage in ('imputed_reward','cancelled','missing_exception'):
        raw=read(arm/'jobs/formal'/trial/'result.json')
        if damage=='imputed_reward':raw['verifier_result']=dict(rewards=dict(reward=0))
        elif damage=='cancelled':raw['exception_info']['exception_type']='CancelledError'
        else:raw['exception_info']=None
        write(arm/'jobs/formal'/trial/'result.json',raw)
        write(arm/'formal-events'/(trial+'.json'),raw)
        harbor=read(arm/'formal-events/harbor-result.json');harbor['trial_results'][1]=raw
        write(arm/'formal-events/harbor-result.json',harbor)
        write(arm/'jobs/formal/result.json',{k:v for k,v in harbor.items() if k!='trial_results'})
    elif damage=='retries':write(arm/'job.json',dict(n_attempts=1,retry=dict(max_retries=1)))
    elif damage=='unfinished':
        s=read(arm/'formal-events/status.json');s['completed']=88;write(arm/'formal-events/status.json',s)
    elif damage=='missing_kv':
        (arm/'formal-events/events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in c['events'] if r['event']!='kv_released'))
    else:
        s=read(arm/'formal-events/status.json');s['network_errors'].append(dict(error='other'));write(arm/'formal-events/status.json',s)
    with pytest.raises((ValueError,KeyError,OSError)):
        recovery.audit_results(arm,c['plan'])


@pytest.fixture
def receipt(base_campaign):
    c=base_campaign;arm=c['arm'];trial=c['trials'][0];cid=recovery.base.CID;nid=recovery.base.NETWORK
    directory=arm/'network-audit-recovery'/trial/'independent-cleanup';directory.mkdir(parents=True)
    checked=read(c['checked']);removed=dict(checked,status='removed',retained_filesystem_diffs_unchanged=True,finished_at=100)
    write(directory/(nid+'.json'),checked);write(directory/(nid+'.removed.json'),removed)
    source=arm/'runtime/scripts/terminalbench_network.py';source.write_text('# frozen network implementation\n')
    manifest=read(arm/'manifest.json');manifest['frozen']['runtime']['scripts/terminalbench_network.py']=sha(source);write(arm/'manifest.json',manifest)
    p=directory.parent/'result.json';write(p,dict(passed=True,kind='independent-pre-removal-network-audit-reconciliation',
        original_network_error_retained=True,no_task_or_verifier_execution=True,trial=trial,container_id=cid,network_id=nid,
        source_sha256=sha(source),original_result_sha256=sha(arm/'jobs/formal'/trial/'result.json'),reward_preserved=0,records=[removed]))
    return c,p


@pytest.mark.parametrize('damage',[None,'score','source','receipt','filesystem'])
def test_network_reconciliation_is_specific_and_retains_original_score(receipt,damage):
    c,p=receipt;arm=c['arm']
    if damage=='score':
        raw=arm/'jobs/formal'/c['trials'][0]/'result.json';raw.write_text(raw.read_text()+' ')
    elif damage=='source':(arm/'runtime/scripts/terminalbench_network.py').write_text('# changed\n')
    elif damage=='receipt':write(p,dict(passed=False))
    elif damage=='filesystem':
        target=p.parent/'independent-cleanup'/(recovery.base.NETWORK+'.removed.json')
        data=read(target);data['retained_filesystem_diffs_unchanged']=False;write(target,data)
    if damage:
        with pytest.raises((ValueError,KeyError)):recovery.network_proof(arm,p)
    else:
        result=recovery.network_proof(arm,p)
        assert result['error']==c['error'] and result['source_hashes']


def test_prepare_pins_existing_scores_and_keeps_monitor_expiration(receipt,monkeypatch):
    c,p=receipt;pair=c['pair'];lru=pair/'lru'
    shutil.copytree(c['arm'],lru,dirs_exist_ok=True)
    # Network validation is independently covered above. Here the plan's
    # score snapshot, source binding and inherited lifetime are under test.
    monkeypatch.setattr(recovery,'network_proof',lambda *_:dict(error=c['error'],source_hashes={}))
    for name in ['results.json','execution-audit.json','formal-events/summary.json','user-interruption.json']:
        write(pair/'thinking'/name,{})
    analyzer=pair/'setup-audit-continuation-code/compare_terminalbench_scored_pairs.py'
    analyzer.parent.mkdir();analyzer.write_text('# frozen analyzer\n')
    write(pair/'pair-status.json',dict(status='running',arm='lru'))
    write(pair/'monitor-current.json',dict(state=str(pair/'monitor-state.json')))
    expiry=recovery.time.time()+1200;write(pair/'monitor-state.json',dict(expires_at=expiry))
    plan=recovery.prepare(pair,c['recovery'],p)
    assert plan['expires_at']==expiry and plan['task_retries_allowed'] is False
    assert len([name for name in plan['immutable'] if name.startswith('lru/jobs/formal/') and name.endswith('result.json')])==89
