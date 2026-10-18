import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('restore_gate', Path(__file__).parents[1] / 'scripts/check_coordination_restore.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


@pytest.fixture
def run(tmp_path):
    write(tmp_path / 'manifest.json', dict(arms=['B3'], cases=['agents', 'burst'], pairs=1,
          workload=dict(turns=3, agent_tokens=2)))
    write(tmp_path / 'status.json', dict(status='completed'))
    write(tmp_path / 'B3.json', dict(verify_session_restore=True, draft_method='mtp', draft_tokens=4,
          kvmem=dict(async_restore=True, select_budget=64, gen_budget=32)))
    server = tmp_path / 'pair-00-B3/server/capture'
    write(server / 'cache-geometry.json', dict(cpu_archive_bytes_per_token=12, groups=[
        dict(recurrent=True, layers=['model.linear_attn']),
        dict(recurrent=False, layers=['model.self_attn', 'mtp.layers.0.self_attn'])]))
    restores, states, fulls = [], [], []
    for case in ('agents', 'burst'):
        traces = []
        for user in range(8):
            for turn in range(3):
                response = f'cmpl-{case}-{user}-{turn}'
                request = response + '-0-1234abcd'
                tokens = 32 + turn * 10
                traces.append(dict(response_id=response, session=f'session-{user}', turn=turn,
                    input_tokens=tokens, output_ids=[1, 2], usage=dict(completion_tokens=2), timeline=[[1., 2]]))
                if turn:
                    history = tokens - 9
                    restores.append(dict(request_id=request, history_tokens=history, active_tokens=history,
                        cancelled=False, uploaded_bytes=100, model_bytes=8, started=1., finished=2.))
                    states.append(dict(request_id=request, history_tokens=history, working_tokens=history,
                        checkpoint_kind='M', pressure_restore=False, recurrent_byte_equal=True, recurrent_tensors=2))
                    fulls.append(dict(request_id=request, history_tokens=history, working_tokens=history,
                        target_kv_bytes=history * 8, working_attention_layers=2, mtp_attention_layers=1,
                        recurrent_tensors=2, draft_layers=0, byte_equal=True))
        write(tmp_path / f'pair-00-B3/{case}/traces.json', traces)
        write(tmp_path / f'pair-00-B3/{case}/results.json', dict(requests=24, output_tokens=48))
    write_rows(server / 'async_restore.jsonl', restores)
    write_rows(server / 'verified_state_restores.jsonl', states)
    write_rows(server / 'verified_restores.jsonl', fulls)
    return tmp_path


def mutate_rows(root, filename, change):
    path = root / 'pair-00-B3/server/capture' / filename
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    write_rows(path, rows)


def test_exact_complete_cross_round_restore_proof(run):
    result = gate.check(run)
    assert result['passed']
    assert result['proved_restores'] == result['expected_restores'] == 32


@pytest.fixture
def agent_run(run):
    manifest = json.loads((run/'manifest.json').read_text())
    manifest.update(kind='agent_scheduling', arms=['A1'], cases=['agents'],
                    workload=dict(turns=3, output_tokens=2, concurrency=4))
    write(run/'manifest.json', manifest)
    (run/'B3.json').rename(run/'A1.json')
    (run/'pair-00-B3').rename(run/'pair-00-A1')
    root = run/'pair-00-A1'
    traces = json.loads((root/'agents/traces.json').read_text())
    traces = [r for r in traces if int(r['session'].split('-')[-1]) < 4]
    write(root/'agents/traces.json',traces)
    write(root/'agents/results.json',dict(requests=12,output_tokens=24))
    path = root/'server/capture/async_restore.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for row in rows:
        row.update(resident_active_tokens=row['active_tokens'], uploaded_bytes=0)
    write_rows(path,rows)
    return run


def test_four_agent_hot_restore_requires_full_target_mtp_and_state_oracles(agent_run):
    report = gate.check(agent_run)
    assert report['passed'] and report['expected_restores'] == report['proved_restores'] == 8


@pytest.mark.parametrize('change', [dict(resident_active_tokens=0), dict(uploaded_bytes=1)])
def test_hot_restore_cannot_pass_without_actual_zero_upload_residency(agent_run, change):
    path = agent_run/'pair-00-A1/server/capture/async_restore.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for row in rows: row.update(change)
    write_rows(path,rows)
    with pytest.raises(gate.EvidenceError): gate.check(agent_run)


@pytest.mark.parametrize('filename', ['async_restore.jsonl', 'verified_state_restores.jsonl', 'verified_restores.jsonl'])
def test_http_success_does_not_replace_missing_restore_oracle(run, filename):
    mutate_rows(run, filename, lambda rows: rows.pop(0))
    with pytest.raises(gate.EvidenceError):
        gate.check(run)


@pytest.mark.parametrize('filename', ['async_restore.jsonl', 'verified_state_restores.jsonl', 'verified_restores.jsonl'])
def test_duplicate_worker_proof_is_rejected(run, filename):
    mutate_rows(run, filename, lambda rows: rows.append(copy.deepcopy(rows[0])))
    with pytest.raises(gate.EvidenceError):
        gate.check(run)


@pytest.mark.parametrize('filename,change', [
    ('async_restore.jsonl', dict(cancelled=True)),
    ('async_restore.jsonl', dict(history_tokens=1)),
    ('async_restore.jsonl', dict(request_id='cmpl-agents-0-1-0-1234abcd-extra')),
    ('async_restore.jsonl', dict(request_id='cmpl-agents-0-1')),
    ('verified_state_restores.jsonl', dict(recurrent_byte_equal=False)),
    ('verified_state_restores.jsonl', dict(checkpoint_kind='P')),
    ('verified_restores.jsonl', dict(byte_equal=False)),
    ('verified_restores.jsonl', dict(mtp_attention_layers=0)),
    ('verified_restores.jsonl', dict(working_attention_layers=1)),
    ('verified_restores.jsonl', dict(target_kv_bytes=8)),
    ('verified_restores.jsonl', dict(recurrent_tensors=1)),
])
def test_incorrect_or_fuzzy_evidence_cannot_pass(run, filename, change):
    mutate_rows(run, filename, lambda rows: rows[0].update(change))
    with pytest.raises(gate.EvidenceError):
        gate.check(run)


def test_missing_turn_cannot_lower_required_proof_count(run):
    path = run / 'pair-00-B3/agents/traces.json'
    traces = json.loads(path.read_text())
    traces.pop()
    write(path, traces)
    with pytest.raises(gate.EvidenceError, match='incomplete 8-user'):
        gate.check(run)


def test_incomplete_stream_is_rejected(run):
    path = run / 'pair-00-B3/agents/traces.json'
    traces = json.loads(path.read_text())
    traces[0]['timeline'] = [[1., 1]]
    write(path, traces)
    with pytest.raises(gate.EvidenceError, match='incomplete output stream'):
        gate.check(run)


def test_mismatch_report_is_fatal_even_with_positive_proofs(run):
    write(run / 'pair-00-B3/server/capture/restore_mismatch.json', [])
    with pytest.raises(gate.EvidenceError, match='mismatch report'):
        gate.check(run)


def test_verification_must_be_enabled(run):
    profile = json.loads((run / 'B3.json').read_text())
    profile['verify_session_restore'] = False
    write(run / 'B3.json', profile)
    with pytest.raises(gate.EvidenceError, match='oracles'):
        gate.check(run)


def test_unfinished_run_cannot_unlock_performance(run):
    write(run / 'status.json', dict(status='running'))
    with pytest.raises(gate.EvidenceError, match='not completed'):
        gate.check(run)


@pytest.fixture
def pressure_run(agent_run):
    manifest=json.loads((agent_run/'manifest.json').read_text())
    manifest['arms']=['Ppressure'];manifest['workload']['retirement_barrier']=True
    write(agent_run/'manifest.json',manifest)
    profile=json.loads((agent_run/'A1.json').read_text());profile['kvmem']['offline_agent_policy']=True
    write(agent_run/'Ppressure.json',profile)
    (agent_run/'pair-00-A1').rename(agent_run/'pair-00-Ppressure')
    root=agent_run/'pair-00-Ppressure'
    restores=[json.loads(s) for s in (root/'server/capture/async_restore.jsonl').read_text().splitlines()]
    restores[0].update(resident_active_tokens=0,uploaded_bytes=100)
    write_rows(root/'server/capture/async_restore.jsonl',restores)
    write_rows(root/'server/capture/offline_decisions.jsonl',[dict(scheduled=[r['request_id'] for r in restores])])
    state=dict(requests=0,refresh_pending=0,connector_pending=0,owned_blocks=0,reservations=[],
        free_blocks=89,total_blocks=100,active_residency=dict(idle_blocks=10,max_blocks=20,pressure_evictions=1),
        cpu_history_admission=dict(active=0,retiring=0,waiting=0,capacity_bytes=100,reserved_bytes=40,
            progress_reserve_bytes=10,available_bytes=50,last_observed_archive_bytes=30))
    write(root/'agents/resources-cleanup.json',state)
    write(root/'agents/retirement-barrier.json',state)
    state['active_residency']['pressure_evictions']=0
    write(root/'agents/resources-before.json',state)
    return agent_run


def test_forced_pressure_requires_hot_and_cold_exact_restores_and_page_conservation(pressure_run):
    assert gate.check(pressure_run)['passed']


@pytest.mark.parametrize('failure',['no_pressure','leak','no_cold_restore','no_policy','missing_lifecycle'])
def test_pressure_oracle_cannot_pass_without_observed_lifecycle(pressure_run,failure):
    root=pressure_run/'pair-00-Ppressure'
    if failure in ('no_pressure','leak'):
        path=root/'agents/retirement-barrier.json';state=json.loads(path.read_text())
        if failure=='no_pressure':state['active_residency']['pressure_evictions']=0
        else:state['free_blocks']-=1
        write(path,state)
    elif failure=='no_cold_restore':
        path=root/'server/capture/async_restore.jsonl';values=[json.loads(s) for s in path.read_text().splitlines()]
        values[0].update(resident_active_tokens=values[0]['active_tokens'],uploaded_bytes=0)
        write_rows(path,values)
    elif failure=='no_policy':write_rows(root/'server/capture/offline_decisions.jsonl',[dict(scheduled=[])])
    else:
        path=pressure_run/'manifest.json';manifest=json.loads(path.read_text())
        manifest['workload']['retirement_barrier']=False;write(path,manifest)
    with pytest.raises(gate.EvidenceError):gate.check(pressure_run)


def test_distinct_tenant_gate_rejects_identical_histories_despite_passing_byte_oracles(pressure_run):
    import hashlib
    path=pressure_run/'manifest.json';manifest=json.loads(path.read_text())
    manifest['workload']['distinct_users']=True;write(path,manifest)
    path=pressure_run/'pair-00-Ppressure/agents/traces.json'
    traces=json.loads(path.read_text())
    for row in traces:row['input_sha256']=hashlib.sha256(row['session'].encode()).hexdigest()
    write(path,traces)
    assert gate.check(pressure_run)['passed']
    for row in traces:row['input_sha256']='0'*64
    write(path,traces)
    with pytest.raises(gate.EvidenceError,match='distinct complete input histories'):gate.check(pressure_run)


@pytest.fixture
def pressure_isolation(pressure_run):
    import hashlib
    root=pressure_run
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['arms']=['P1']
    manifest['workload'].update(retirement_barrier=False,distinct_users=True)
    write(root/'manifest.json',manifest)
    profile=json.loads((root/'Ppressure.json').read_text())
    profile['kvmem']['active_resident_pressure_only']=True
    write(root/'P1.json',profile)
    (root/'pair-00-Ppressure').rename(root/'pair-00-P1')
    case=root/'pair-00-P1/agents'
    traces=json.loads((case/'traces.json').read_text())
    for row in traces:
        row.update(begin=row['turn']*10.,end=row['turn']*10.+4.,
                   input_sha256=hashlib.sha256(row['session'].encode()).hexdigest())
    write(case/'traces.json',traces)
    path=root/'pair-00-P1/server/capture/async_restore.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    for row in rows:row.update(resident_active_tokens=row['active_tokens'],uploaded_bytes=0)
    write_rows(path,rows)
    before=json.loads((case/'resources-before.json').read_text())
    before['active_residency'].update(retention_policy='pressure',expired=0)
    write(case/'resources-before.json',before)
    write(case/'resources-after.json',before)
    return root


def test_pressure_only_requires_hot_return_beyond_former_ttl_with_complete_byte_oracles(pressure_isolation):
    assert gate.check(pressure_isolation)['proved_restores']==8


@pytest.mark.parametrize('failure',['ttl_expiry','ttl_mode','short_wait','cold_return','lifecycle_mode'])
def test_pressure_only_evidence_cannot_silently_fall_back_to_ttl(pressure_isolation,failure):
    root=pressure_isolation;case=root/'pair-00-P1/agents'
    if failure in ('ttl_expiry','ttl_mode'):
        path=case/'resources-after.json';state=json.loads(path.read_text())
        state['active_residency'].update({'expired':1} if failure=='ttl_expiry' else {'retention_policy':'ttl'})
        write(path,state)
    elif failure=='short_wait':
        path=case/'traces.json';rows=json.loads(path.read_text())
        for row in rows:row['end']=row['begin']+5.
        write(path,rows)
    elif failure=='cold_return':
        path=root/'pair-00-P1/server/capture/async_restore.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        rows[0].update(resident_active_tokens=0,uploaded_bytes=100)
        write_rows(path,rows)
    else:
        manifest=json.loads((root/'manifest.json').read_text());manifest['workload']['lifecycle_checks']=True
        write(root/'manifest.json',manifest)
        write(root/'pair-00-P1/lifecycle/results.json',dict(passed=True,pressure_only=False))
    with pytest.raises(gate.EvidenceError):gate.check(root)
