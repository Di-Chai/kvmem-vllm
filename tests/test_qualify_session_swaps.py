"""Host validation of the session qualification evidence gates."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import qualify_session_swaps as qualify
from test_stress_request_swaps import snapshot


def test_repeat_workload_preserves_stress_prompt_and_stable_explicit_query_origin():
    prompt, begin = qualify.prompt_layout('repeat', 13, 4, [1, 2, 3], [])
    assert prompt == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1]
    assert begin == 8 and prompt[begin:]
    assert qualify.REPEAT_SENTENCE == ' A quiet river flows past the village.'


def test_code_workload_preserves_question_suffix_and_original_prompt_extent():
    prompt, begin = qualify.prompt_layout('code', 13, 4, [1, 2, 3], [8, 9])
    assert prompt == [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 8, 9]
    assert begin == 8
    with pytest.raises(ValueError): qualify.prompt_layout('code', 2, 4, [1], [8, 9])
    with pytest.raises(ValueError): qualify.prompt_layout('repeat', 13, 4, [], [])


def retained_idle():
    value = snapshot()
    value['session_owners'] = [dict(namespace=['s', 's'], phase='idle')]
    value['cpu_history_admission'].update(leases=1, reserved_bytes=4096)
    return value


def test_idle_pm_history_is_retained_but_active_or_retiring_work_is_not_settled():
    state = retained_idle()
    assert not qualify.active_resources(state)
    state['session_owners'][0]['phase'] = 'active'
    assert qualify.active_resources(state)
    state['session_owners'][0]['phase'] = 'idle'
    state['cpu_history_admission']['retiring'] = 1
    assert qualify.active_resources(state)


def test_restore_proof_requires_exact_request_namespace_kind_and_positive_history(tmp_path):
    path = tmp_path / 'sync_restore_timing.jsonl'
    row = dict(request_id='cmpl-a-0-aabbccdd', checkpoint_kind='M', history_tokens=128,
               namespace=['s', 's'], generation=2)
    path.write_text(json.dumps(row) + '\n')
    assert qualify.restore_proof(tmp_path, 'a', 's', 'M') == row
    for base, namespace, kind in [('aa', 's', 'M'), ('a', 'other', 'M'), ('a', 's', 'P')]:
        with pytest.raises(ValueError): qualify.restore_proof(tmp_path, base, namespace, kind)
    path.write_text(json.dumps({**row, 'history_tokens': 0}) + '\n')
    with pytest.raises(ValueError): qualify.restore_proof(tmp_path, 'a', 's', 'M')


def test_same_session_serialization_requires_retirement_ack_before_successor_restore():
    qualify.serialization_proof({'retirement_ack_ready_at': 10}, {'started_at': 10.1})
    with pytest.raises(ValueError):
        qualify.serialization_proof({'retirement_ack_ready_at': 10}, {'started_at': 9.9})
    with pytest.raises(ValueError): qualify.serialization_proof({}, {'started_at': 11})


@pytest.mark.parametrize('actions', [['capture', 'drop'], ['capture', 'drop', 'restore'],
                                    ['restore', 'drop'], ['capture', 'restore'], []])
def test_canceled_session_requires_capture_restore_drop_in_order(actions):
    rows = [dict(request_id='a', history_end=42, action=action) for action in actions]
    with pytest.raises(ValueError): qualify.cancel_restore_proof(rows, 'a', 42)


def test_cancel_restore_evidence_keeps_exact_owner_and_history():
    rows = [dict(request_id=owner, history_end=history, action=action)
            for owner, history in [('a', 42), ('a', 43), ('peer', 42)]
            for action in ['capture', 'restore', 'drop']]
    assert qualify.cancel_restore_proof(rows, 'a', 42) == rows[:3]


def test_partial_jsonl_write_is_not_mistaken_for_committed_receipt(tmp_path):
    path = tmp_path / 'receipt.jsonl'
    path.write_text('{"request_id":"a"}\n{"request_')
    assert qualify.records(path) == [{'request_id': 'a'}]
    path.write_text('{"request_id":"a"}\n{bad}\n')
    with pytest.raises(json.JSONDecodeError): qualify.records(path)


def test_terminal_proof_rejects_plain_retirement_without_pm_capture():
    with pytest.raises(ValueError):
        qualify.terminal_proof({'completed': [dict(request_id='cmpl-a-0-aabbccdd', history_end=32)]}, 'a')


def worker_report():
    return dict(at=1, request_checkpoints=dict(requests={},
        resources=dict(active=0, resources=dict(checkpoint_bytes=dict(live=0, promised=0))),
        storage=dict(entries=0, cpu_resident_bytes=0, cpu_charged_bytes=0, cpu_promised_bytes=0,
            logical_bytes=0, ssd_used_bytes=0, ssd_pending_write_bytes=0, io_pending=0,
            disk_write_bytes=100, disk_read_bytes=100, spills=1, loads=1, io_failures=0)),
        archive_io_pending_bytes=0, cpu_authority_bytes=0, logical_allocated_bytes=0,
        ssd_used_bytes=0, ssd_reserved_bytes=0, ssd_retired_pending_bytes=0, ssd_file_bytes=0)


@pytest.mark.parametrize('key', ['entries', 'cpu_resident_bytes', 'cpu_charged_bytes', 'cpu_promised_bytes',
    'logical_bytes', 'ssd_used_bytes', 'ssd_pending_write_bytes', 'io_pending'])
def test_storage_release_gate_checks_every_physical_and_promised_owner(key):
    before, after = worker_report(), worker_report()
    after['request_checkpoints']['storage'][key] = 1
    assert qualify.worker_checkpoint_outstanding(after)
    with pytest.raises(ValueError): qualify.checkpoint_storage_proof(before, after)


@pytest.mark.parametrize('key', ['live', 'promised'])
def test_worker_ledger_must_release_even_when_storage_and_scheduler_look_empty(key):
    report = worker_report()
    report['request_checkpoints']['resources']['resources']['checkpoint_bytes'][key] = 1
    assert qualify.worker_checkpoint_outstanding(report)


def test_ssd_qualification_requires_observed_read_and_write_deltas_in_this_flow():
    before, after = worker_report(), worker_report()
    with pytest.raises(ValueError, match='actual spill'): qualify.checkpoint_storage_proof(before, after)
    storage = after['request_checkpoints']['storage']
    storage.update(disk_write_bytes=200, disk_read_bytes=180, spills=2, loads=2)
    proof = qualify.checkpoint_storage_proof(before, after)
    assert proof['deltas'] == dict(disk_write_bytes=100, disk_read_bytes=80, spills=1, loads=1, io_failures=0)
    storage['io_failures'] = 1
    with pytest.raises(ValueError, match='I/O failed'): qualify.checkpoint_storage_proof(before, after)


def test_archive_ssd_payload_is_not_confused_with_request_checkpoint_storage():
    report = worker_report()
    report['ssd_used_bytes'] = 4096
    assert qualify.worker_checkpoint_outstanding(report) == {'history_ssd_used_bytes': 4096}
