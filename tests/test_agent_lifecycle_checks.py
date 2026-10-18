import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('lifecycle_checks', Path(__file__).parents[1]/'scripts/agent_lifecycle_checks.py')
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


def quiescent_state():
    return dict(requests=0, refresh_pending=0, connector_pending=0, owned_blocks=0, reservations=[],
        total_blocks=100, free_blocks=84, active_residency=dict(idle_blocks=15, max_blocks=20, expired=0),
        cpu_history_admission=dict(active=0, retiring=0, waiting=0, reserved_bytes=40,
            progress_reserve_bytes=10, available_bytes=50, capacity_bytes=100, last_observed_archive_bytes=30))


def write_rows(path, values):
    path.write_text(''.join(json.dumps(value)+'\n' for value in values))


@pytest.fixture
def capture(tmp_path):
    (tmp_path/'cache-geometry.json').write_text(json.dumps(dict(cpu_archive_bytes_per_token=12, groups=[
        dict(recurrent=True,layers=['model.linear_attn']),
        dict(recurrent=False,layers=['model.self_attn','mtp.layers.0.self_attn'])])))
    rid='cmpl-known-0-1234abcd'
    write_rows(tmp_path/'async_restore.jsonl',[dict(request_id=rid, history_tokens=100,active_tokens=100,
        resident_active_tokens=0,cancelled=True,uploaded_bytes=800,model_bytes=8,started=1.,finished=2.)])
    write_rows(tmp_path/'verified_state_restores.jsonl',[dict(request_id=rid, history_tokens=100,
        working_tokens=100,checkpoint_kind='M',pressure_restore=False,recurrent_byte_equal=True,recurrent_tensors=2)])
    write_rows(tmp_path/'verified_restores.jsonl',[dict(request_id=rid,history_tokens=100,working_tokens=100,
        byte_equal=True,recurrent_tensors=2,working_attention_layers=2,mtp_attention_layers=1,
        target_kv_bytes=800,draft_layers=0)])
    return tmp_path


def test_cancelled_cold_restore_requires_complete_target_mtp_and_recurrent_bytes(capture):
    proof=lifecycle.exact_restore(capture,'cmpl-known',100,resident=False,cancelled=True)
    assert proof['uploaded_bytes']==800


@pytest.mark.parametrize('filename,change',[
    ('async_restore.jsonl',dict(cancelled=False)),
    ('async_restore.jsonl',dict(resident_active_tokens=100)),
    ('async_restore.jsonl',dict(uploaded_bytes=0)),
    ('async_restore.jsonl',dict(history_tokens=99)),
    ('verified_state_restores.jsonl',dict(checkpoint_kind='P')),
    ('verified_state_restores.jsonl',dict(recurrent_tensors=1)),
    ('verified_restores.jsonl',dict(mtp_attention_layers=0)),
    ('verified_restores.jsonl',dict(target_kv_bytes=799)),
    ('verified_restores.jsonl',dict(byte_equal=False)),
])
def test_successful_http_cannot_replace_cancellation_byte_oracles(capture,filename,change):
    values=lifecycle.rows(capture/filename);values[0].update(change)
    write_rows(capture/filename,values)
    with pytest.raises(AssertionError):
        lifecycle.exact_restore(capture,'cmpl-known',100,resident=False,cancelled=True)


def test_quiescent_page_conservation_allows_only_bounded_idle_views():
    lifecycle.validate_quiescent(quiescent_state())


@pytest.mark.parametrize('change',[
    dict(free_blocks=83), dict(free_blocks=85), dict(reservations=['aborted']), dict(owned_blocks=1),
    dict(connector_pending=1), dict(active_residency=dict(idle_blocks=15,max_blocks=14)),
    dict(cpu_history_admission=dict(active=0,retiring=0,waiting=0,reserved_bytes=40,
        progress_reserve_bytes=10,available_bytes=50,capacity_bytes=100,last_observed_archive_bytes=41)),
])
def test_leaked_pages_premature_credits_and_pending_fences_fail_quiescence(change):
    state=quiescent_state();state.update(change)
    with pytest.raises(AssertionError):lifecycle.validate_quiescent(state)


def test_only_unfinished_json_tail_may_be_ignored(tmp_path):
    path=tmp_path/'log.jsonl'
    path.write_text('{"ok": 1}\n{"partial"')
    assert lifecycle.rows(path)==[dict(ok=1)]
    path.write_text('{"partial"\n{"ok": 1}\n')
    with pytest.raises(json.JSONDecodeError):lifecycle.rows(path)


@pytest.mark.parametrize('pressure_only', [False, True])
def test_cancellation_followup_cannot_use_newer_or_recomputed_history(capture, pressure_only):
    # Full report validation must consult worker evidence, including the exact
    # old checkpoint after abort, including its retained full byte oracle.
    base=lifecycle.rows(capture/'async_restore.jsonl')[0]
    state=lifecycle.rows(capture/'verified_state_restores.jsonl')[0]
    full=lifecycle.rows(capture/'verified_restores.jsonl')[0]
    checks={};restores=[];states=[];fulls=[];terminals=[];stages=[]
    for mode in ('ttl','cold','resident'):
        response='cmpl-'+mode;rid=response+'-0-1234abcd'
        proof={**base,'request_id':rid,'cancelled':mode=='cold',
            'resident_active_tokens':100 if mode=='resident' else 0,'uploaded_bytes':0 if mode=='resident' else 800}
        restores.append(proof);states.append({**state,'request_id':rid});fulls.append({**full,'request_id':rid})
        item=dict(first=dict(input_tokens=37,output_ids=[1]*64),restore=proof)
        if mode=='ttl':
            before=quiescent_state();after=copy.deepcopy(before);after['active_residency']['expired']=1
            item.update(before=before,after=after,second=dict(response_id=response,output_ids=[1]*32))
            checks['ttl']=item
        else:
            terminal=dict(request_id=rid,status='FINISHED_ABORTED',lifetime_reservation_retained=True)
            terminals.append(terminal)
            stage=dict(request_id=rid,stage='restore_submitted',at=1.1)
            if mode=='cold':stages.append(stage)
            follow_id='cmpl-follow-'+mode
            follow={**base,'request_id':follow_id+'-0-1234abcd','cancelled':False}
            restores.append(follow)
            states.append({**state,'request_id':follow['request_id']})
            fulls.append({**full,'request_id':follow['request_id']})
            item.update(response_id=response,cancelled_at=1.5,stage=stage if mode=='cold' else None,
                terminal=terminal,resources=quiescent_state(),followup=dict(response_id=follow_id,output_ids=[1]*32),
                followup_restore=follow)
            checks[mode+'_cancel']=item
    checks['cleanup']=quiescent_state()
    for filename,values in [('async_restore.jsonl',restores),('verified_state_restores.jsonl',states),
            ('verified_restores.jsonl',fulls),('scheduler_finished.jsonl',terminals),('async_refresh.jsonl',stages)]:
        write_rows(capture/filename,values)
    if pressure_only:
        checks = {key:checks[key] for key in ('resident_cancel','cleanup')}
        checks['cleanup']['active_residency']['retention_policy']='pressure'
    report=dict(passed=True,checks=checks,pressure_only=pressure_only)
    assert lifecycle.validate_report(capture,report)
    wrong = copy.deepcopy(report);wrong['pressure_only'] = not pressure_only
    with pytest.raises(AssertionError,match='Incomplete lifecycle suite'):
        lifecycle.validate_report(capture,wrong)
    if pressure_only:
        wrong=copy.deepcopy(report);wrong['checks']['cleanup']['active_residency']['retention_policy']='ttl'
        with pytest.raises(AssertionError,match='TTL cache'):lifecycle.validate_report(capture,wrong)
    follow['history_tokens']=101
    write_rows(capture/'async_restore.jsonl',restores)
    with pytest.raises(AssertionError,match='complete CPU checkpoint'):lifecycle.validate_report(capture,report)
