import asyncio
import json

import pytest

from scripts.terminalbench_queue_probe import Probe,admission_summary,gauge_summary


class Response:
    def __init__(self,value):self.value=value
    def raise_for_status(self):pass
    def json(self):return self.value


class Client:
    def __init__(self):self.calls=[];self.prior={};self.released=[]
    async def post(self,path,json):
        assert path=='/v1/completions'
        self.calls.append(json)
        session=json['session_id']
        if json.get('kv_transfer_params',{}).get('kvmem_release_session'):
            self.released.append(session)
            return Response(dict(id='release-'+session,usage=dict(completion_tokens=1)))
        assert 'allowed_token_ids' not in json and 'logit_bias' not in json
        assert json['ignore_eos'] is True
        if session in self.prior:
            previous=self.prior[session]
            assert json['prompt'][:len(previous)]==previous
        tokens=list(range(json['max_tokens']))
        self.prior[session]=list(json['prompt'])+tokens
        return Response(dict(id='request-'+str(len(self.calls)),usage=dict(completion_tokens=len(tokens)),
            choices=[dict(token_ids=tokens,finish_reason='length')]))


def config():
    return dict(items=24,turns=3,output_tokens=2,tool_wait_seconds=0,eos_token=99,
        prompt=[10,11],query_span=[0,1],additions=[[20],[30]],temperature=0,seed=1000,
        qualification_owners=12,qualification_output_tokens=2)


def test_natural_queue_runs_each_item_once_with_exact_own_prefix_and_release(tmp_path):
    client=Client();probe=Probe(client,'test',config(),tmp_path,tmp_path)
    asyncio.run(probe.queue(12))
    assert len(probe.records)==72
    assert len(client.released)==len(set(client.released))==24
    assert not probe.sessions
    assert len(client.calls)==96
    for item in range(24):
        rows=[row for row in probe.records if row['item']==item]
        assert [row['turn'] for row in rows]==[1,2,3]
        files=[json.loads((tmp_path/'requests'/f'{item}-{turn}.json').read_text()) for turn in (1,2,3)]
        assert files[1]['input_ids']==files[0]['input_ids']+files[0]['accepted_ids']+[20]
        assert files[2]['input_ids']==files[1]['input_ids']+files[1]['accepted_ids']+[30]


def test_twelve_owner_qualification_keeps_all_sessions_between_rounds(tmp_path):
    client=Client();probe=Probe(client,'test',config(),tmp_path,tmp_path)
    snapshots=[]
    async def idle(sessions,turn):
        assert len(sessions)==12
        assert not client.released
        snapshots.append((turn,set(sessions.values())))
    probe.wait_idle=idle
    asyncio.run(probe.qualify())
    assert len(probe.records)==36
    assert snapshots[0][1]==snapshots[1][1]==snapshots[2][1]
    assert len(client.released)==12 and not probe.sessions


def test_failed_queue_does_not_retry_generation_and_releases_owners(tmp_path):
    class FailedClient(Client):
        async def post(self,path,json):
            if not json.get('kv_transfer_params'):
                self.calls.append(json)
                raise ValueError('injected generation failure')
            return await super().post(path,json)
    client=FailedClient();probe=Probe(client,'test',config(),tmp_path,tmp_path)
    with pytest.raises(ValueError,match='injected'):
        asyncio.run(probe.queue(12))
    generation=[row for row in client.calls if not row.get('kv_transfer_params')]
    assert len({row['session_id'] for row in generation})==len(generation)
    assert not probe.sessions
    assert set(client.released)=={row['session_id'] for row in generation}


def test_observed_concurrency_is_weighted_by_time_and_gaps_are_not_extended():
    samples=[dict(at=0,metrics=dict(running=2)),dict(at=1,metrics=dict(running=6)),
             dict(at=3,metrics=dict(running=4))]
    result=gauge_summary(samples,0,4,'running')
    assert result['mean']==4.5 and result['peak']==6
    assert gauge_summary(samples,0,30,'running')['observed_seconds']==5.5


def test_shared_cpu_admission_checks_capacity_and_uncached_owner():
    sample=dict(scheduler_resources=dict(session_owners=[dict(phase='idle')]*12,
        cpu_history_admission=dict(capacity_records=100,reserved_records=50,
            capacity_bytes=1000,reserved_bytes=500,idle_protected_records=24)))
    result=admission_summary([sample])
    assert result['peak_session_owners']==12
    assert result['capacity_violations']==0
    assert result['peak_idle_protected_records']==24
    sample['scheduler_resources']['cpu_history_admission']['reserved_records']=101
    sample['scheduler_resources']['session_owners'][0]['phase']='uncached'
    result=admission_summary([sample])
    assert result['capacity_violations']==1 and result['uncached_owner_observed']


def test_expected_uncached_release_control_is_not_a_lost_history_owner():
    state=dict(session_owners=[dict(namespace=['owner','owner'],phase='uncached')],
        cpu_history_admission=dict(capacity_records=100,reserved_records=0,capacity_bytes=1000,reserved_bytes=0))
    assert admission_summary([dict(snapshot=dict(scheduler_resources=state))])['uncached_owner_observed']
    assert not admission_summary([dict(snapshot=dict(scheduler_resources=state),
        release_control_sessions=['owner'])])['uncached_owner_observed']
