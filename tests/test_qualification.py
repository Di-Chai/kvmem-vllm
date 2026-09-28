import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("qualification_script",
    Path(__file__).parents[1] / "scripts/qualify_engine.py")
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def fixture():
    report = {"dflash": True, "single": {"result": {
        "id": "r", "usage": {"prompt_tokens": 14, "completion_tokens": 128}}}, "concurrent": []}
    archived = {"active_requests": 0, "archive_used_slots": 0, "pending_downloads": 0,
        "max_concurrent_batches": 2, "decode_query_len": 8,
        "completed": [{"request_id": "r-0-worker", "sealed_records_verified": 1,
            "history_end": 153, "accepted_tokens": 153, "rejected_tokens": 10,
            "last_steps": [{"history_begin": 149, "accepted": 4, "rejected": 4, "scheduled": 8}]}]}
    return report, archived


def test_terminal_extent_accounts_for_actual_native_inflight_batches():
    report, archived = fixture()
    qualification.validate_archive(report, archived)
    serial = copy.deepcopy(archived)
    serial["max_concurrent_batches"] = 1
    with pytest.raises(RuntimeError, match="extent"):
        qualification.validate_archive(report, serial)


def test_terminal_extent_checks_each_request_and_native_packet_not_just_total():
    report, archived = fixture()
    archived["completed"][0]["last_steps"][0]["history_begin"] = 148
    with pytest.raises(RuntimeError, match="packet"):
        qualification.validate_archive(report, archived)
    report, archived = fixture()
    archived["completed"][0]["request_id"] = "another-request-0-worker"
    with pytest.raises(RuntimeError, match="archive"):
        qualification.validate_archive(report, archived)


def test_short_natural_eos_can_leave_only_a_partial_tail_but_no_sealed_record_may_be_omitted():
    report, archived = fixture()
    report["concurrent"].append({"result": {"id": "short", "usage": {"prompt_tokens": 14, "completion_tokens": 1}}})
    archived["completed"].append({"request_id": "short-0-worker", "sealed_records_verified": 0,
        "history_end": 14, "accepted_tokens": 14, "rejected_tokens": 0,
        "last_steps": [{"history_begin": 0, "accepted": 14, "rejected": 0, "scheduled": 14}]})
    qualification.validate_archive(report, archived)
    archived["completed"][0]["sealed_records_verified"] = 0
    with pytest.raises(RuntimeError, match="Incomplete archive"):
        qualification.validate_archive(report, archived)


def test_frozen_workload_preserves_exact_ids_and_validates_hash_and_query(tmp_path):
    import hashlib
    import json
    path = tmp_path / 'request.json'
    payload = {"id": "real-history", "input_ids": [11, 0, 248319], "query_begin": 1, "query_end": 3}
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result, provenance = qualification.read_frozen(path, digest, 128)
    assert result == payload and provenance['tokens'] == 3 and not provenance['tool_execution']
    with pytest.raises(ValueError, match='SHA256'):
        qualification.read_frozen(path, '0' * 64, 128)
    payload['query_end'] = 4
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='boundaries'):
        qualification.read_frozen(path, hashlib.sha256(path.read_bytes()).hexdigest(), 128)


def test_concurrent_only_archive_validation_does_not_invent_a_single_request():
    report, archived = fixture()
    report['concurrent'] = [report.pop('single')]
    qualification.validate_archive(report, archived)
    archived['completed'].append(copy.deepcopy(archived['completed'][0]))
    with pytest.raises(RuntimeError, match='request sets'):
        qualification.validate_archive(report, archived)


def bounded_fixture():
    report = {'concurrent': [{'result': {'id': f'r{i}', 'usage': {'prompt_tokens': 134000}}}
                             for i in range(4)]}
    ids = [f'r{i}-0-worker' for i in range(4)]
    workers = [{'request_id': request_id, 'command_id': i, 'history_begin': 133120,
                'selected_tokens': 131072, 'query_tokens': 256, 'batch_requests': ids}
               for i, request_id in enumerate(ids)]
    schedulers = [{'request_id': request_id, 'command_id': i, 'selected_tokens': 131072,
                   'replay_tokens': 256, 'groups': [{'offset': 2048, 'blocks': 1026, 'block_tokens': 128}]}
                  for i, request_id in enumerate(ids)]
    return report, workers, schedulers


def test_long_c4_requires_exact_transactions_and_real_native_co_batch():
    report, workers, schedulers = bounded_fixture()
    result = qualification.validate_bounded(report, workers, schedulers, require_concurrent=True)
    assert result['compressed_requests'] == result['max_refresh_batch_requests'] == 4
    with pytest.raises(RuntimeError, match='Duplicate'):
        qualification.validate_bounded(report, workers + [workers[0]], schedulers)
    wrong = copy.deepcopy(schedulers)
    wrong[0]['command_id'] = 999
    with pytest.raises(RuntimeError, match='transaction sets'):
        qualification.validate_bounded(report, workers, wrong)
    serial = copy.deepcopy(workers)
    for worker in serial:
        worker['batch_requests'] = [worker['request_id']]
    with pytest.raises(RuntimeError, match='four-request'):
        qualification.validate_bounded(report, serial, schedulers, require_concurrent=True)


def test_long_c4_requires_compression_for_each_history_and_bounded_pages():
    report, workers, schedulers = bounded_fixture()
    with pytest.raises(RuntimeError, match='lacks an actual'):
        qualification.validate_bounded(report, workers[:-1], schedulers[:-1])
    schedulers[0]['groups'][0]['blocks'] += 1
    with pytest.raises(RuntimeError, match='allocation'):
        qualification.validate_bounded(report, workers, schedulers)


def test_bounded_dflash_reselection_cannot_score_unverified_candidates():
    report, workers, schedulers = bounded_fixture()
    report['dflash'] = True
    for item, worker, scheduler in zip(report['concurrent'], workers, schedulers):
        item['result']['usage']['prompt_tokens'] = 132000
        worker.update(query_tokens=8, query_score_tokens=1, draft_groups=[4])
        scheduler.update(replay_tokens=8)
        scheduler['groups'][0]['blocks'] = 1025
    qualification.validate_bounded(report, workers, schedulers, require_concurrent=True)
    workers[0]['query_score_tokens'] = 8
    with pytest.raises(RuntimeError, match='unverified draft'):
        qualification.validate_bounded(report, workers, schedulers)


def test_publication_only_measurements_cannot_claim_checksum_verification_or_skip_publication():
    report, archived = fixture()
    archived['archive_verification'] = 'publication_only'
    record = archived['completed'][0]
    record.update(sealed_records_archived=1, sealed_records_verified=0)
    qualification.validate_archive(report, archived)
    record['sealed_records_verified'] = 1
    with pytest.raises(RuntimeError, match='Incomplete archive'):
        qualification.validate_archive(report, archived)
    record.update(sealed_records_archived=0, sealed_records_verified=0)
    with pytest.raises(RuntimeError, match='Incomplete archive'):
        qualification.validate_archive(report, archived)


def test_cancellation_requires_actual_refresh_overlap_and_native_abort_status():
    cancelled = [{'request_id': 'cmpl-victim', 'cancelled_at': 20,
                  'trigger': {'request_id': 'cmpl-victim-0-worker', 'command_id': 7}}]
    workers = [{'request_id': 'cmpl-victim-0-worker', 'command_id': 7,
                'refresh_started_at': 10, 'refresh_finished_at': 30}]
    finished = [{'request_id': 'cmpl-victim-0-worker', 'at': 31, 'status': 'FINISHED_ABORTED'}]
    qualification.validate_cancellation(cancelled, workers, finished)
    finished[0]['status'] = 'FINISHED_LENGTH_CAPPED'
    with pytest.raises(RuntimeError, match='acknowledge'):
        qualification.validate_cancellation(cancelled, workers, finished)
    finished[0]['status'] = 'FINISHED_ABORTED'
    cancelled[0]['cancelled_at'] = 35
    with pytest.raises(RuntimeError, match='during'):
        qualification.validate_cancellation(cancelled, workers, finished)


def test_cancelled_prefill_archive_is_checked_without_fabricating_http_completion_tokens():
    report, archived = fixture()
    report['dflash'] = False
    report['cancelled'] = [{'request_id': 'r', 'prompt_tokens': 180,
                            'trigger': {'history_begin': 128}}]
    del report['single']
    qualification.validate_archive(report, archived)
    report['cancelled'][0]['prompt_tokens'] = 153
    with pytest.raises(RuntimeError, match='stop with'):
        qualification.validate_archive(report, archived)


def test_final_scheduler_gate_requires_physical_gpu_pages_not_only_request_cleanup():
    snapshot = dict(requests=0, refresh_pending=0, connector_pending=0,
                    reservations=[], owned_blocks=0, free_blocks=0, total_blocks=17)
    with pytest.raises(RuntimeError, match='GPU pages'):
        qualification.validate_released_scheduler(snapshot)
    snapshot['free_blocks'] = 16
    qualification.validate_released_scheduler(snapshot)
