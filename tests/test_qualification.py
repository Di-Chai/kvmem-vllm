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
