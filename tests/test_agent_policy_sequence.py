import importlib.util
import json
from pathlib import Path

import pytest


spec=importlib.util.spec_from_file_location('policy_sequence',Path(__file__).parents[1]/'scripts/run_agent_policy_sequence.py')
sequence=importlib.util.module_from_spec(spec);spec.loader.exec_module(sequence)


def save(path,value):path.write_text(json.dumps(value))


@pytest.fixture
def formal_pair(tmp_path):
    for arm in ('A0','P1'):
        path=tmp_path/('deepswe-'+arm);path.mkdir()
        manifest=dict(reference_sha256='reference',tasks=[dict(task_id=str(i)) for i in range(16)],
            claude_sha256='claude',workers=4,sampling='fixed',seed=1000,attempts_per_task=1,
            setup_sha256={'setup':'sha'},protocol_provenance={'protocol':'sha'},cpu_admission={'capacity':100},
            termination_policies={'guard':32},carried_results=[],
            profile=dict(max_num_seqs=4,kvmem=dict(select_budget=131072,gen_budget=131072,
                offline_agent_policy=arm=='P1',active_resident_ttl_ms=0,
                active_resident_pressure_only=arm=='P1')))
        save(path/'manifest.json',manifest)
        save(path/'formal-evidence.json',dict(source_sha256={'bridge':'same-source'},protocol_sha256={'protocol':'sha'}))
        summary=dict(run_status='completed',completed_tasks=16,infrastructure_errors=0,carried_completed_tasks=0,
            disk_read_bytes=0,disk_write_bytes=0,mixed_end_to_end_output_tokens_per_second=120 if arm=='P1' else 100,
            elapsed_seconds=80 if arm=='P1' else 100,passed_tasks=9 if arm=='P1' else 8,
            tasks=[dict(task=str(i),state='completed',reward=int(i<8)) for i in range(16)])
        save(path/'live-summary.json',summary)
    return tmp_path


def test_formal_comparison_uses_fresh_same_bridge_baseline_and_all_task_identities(formal_pair):
    result=sequence.formal_comparison(formal_pair)
    assert result['baseline_complete'] and result['identical_bridge_and_protocol']
    assert result['output_throughput_ratio']==1.2 and result['makespan_ratio']==.8
    assert result['passed_task_delta']==1 and len(result['tasks'])==16


@pytest.mark.parametrize('filename,change',[
    ('formal-evidence.json',dict(source_sha256={'bridge':'old-quadratic-bridge'})),
    ('manifest.json',dict(workers=8)),
    ('manifest.json',dict(carried_results=['previous-attempt'])),
    ('manifest.json',dict(attempts_per_task=2)),
    ('manifest.json',dict(seed=1234)),
    ('live-summary.json',dict(run_status='running')),
    ('live-summary.json',dict(completed_tasks=15)),
    ('live-summary.json',dict(carried_completed_tasks=1)),
    ('live-summary.json',dict(disk_write_bytes=1)),
    ('live-summary.json',dict(infrastructure_errors=1)),
])
def test_unmatched_or_incomplete_formal_runs_cannot_produce_speedup(formal_pair,filename,change):
    path=formal_pair/'deepswe-P1'/filename
    data=json.loads(path.read_text());data.update(change);save(path,data)
    with pytest.raises(ValueError):sequence.formal_comparison(formal_pair)


def test_native_model_setting_changes_and_missing_task_fail_formal_pair(formal_pair):
    path=formal_pair/'deepswe-P1/manifest.json';original=json.loads(path.read_text())
    changed=json.loads(path.read_text());changed['profile']['kvmem']['gen_budget']=8192;save(path,changed)
    with pytest.raises(ValueError,match='model settings'):sequence.formal_comparison(formal_pair)
    save(path,original)
    path=formal_pair/'deepswe-P1/live-summary.json';summary=json.loads(path.read_text())
    summary['tasks'][0]['task']='different-task';save(path,summary)
    with pytest.raises(ValueError,match='identities'):sequence.formal_comparison(formal_pair)


def test_sequence_separates_finite_ttl_regression_from_pressure_only_candidate(tmp_path):
    commands = sequence.commands(tmp_path)
    assert commands['correctness'][commands['correctness'].index('--arms')+1:][:2] == ['A0','Pttl']
    assert '--lifecycle-checks' in commands['correctness']
    assert '--distinct-users' in commands['isolation'] and '--lifecycle-checks' in commands['isolation']
    assert 'P1' in commands['isolation'] and 'Ppressure' in commands['pressure']
    assert not {'Pttl', 'Ppressure'} & set(commands['performance'])
    assert commands['performance'][commands['performance'].index('--pairs')+1] == '3'
