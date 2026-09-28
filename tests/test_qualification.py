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
