import json
import asyncio
import sys
from types import SimpleNamespace

import pytest

from monitor_ruler_pooling_queue import ARMS, findings, inspect, wake


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def setup_run(root):
    write(root / "status.json", dict(status="running", arm="max", controller_pid=20))
    write(root / "queue-process.json", dict(pid=10))
    write(root / "max/status.json", dict(status="testing", server_pid=30, client_pid=40, at=500))
    write(root / "max/run/answers/progress.json", dict(completed=11, expected=260, at=900))


def check(root, dead=(), now=1000):
    return inspect(root, now=now, is_alive=lambda pid: pid is not None and pid not in dead,
                   disk_usage=lambda path: SimpleNamespace(free=100 * 1024**3))


def test_healthy_run_is_quiet(tmp_path):
    setup_run(tmp_path)
    snap = check(tmp_path)
    assert not snap["issues"]
    assert snap["arms"]["max"]["completed"] == 11
    assert findings(snap, {}) == []


def test_failed_and_dead_processes_are_actionable(tmp_path):
    setup_run(tmp_path)
    write(tmp_path / "max/run/answers/failure.json", dict(error="protocol failure"))
    snap = check(tmp_path, dead=(10, 20, 30, 40))
    assert len(snap["issues"]) == 5
    assert any("client failure.json" in issue for issue in findings(snap, {}))


def test_no_progress_wakes_agent_but_does_not_restart_anything(tmp_path):
    setup_run(tmp_path)
    snap = check(tmp_path, now=2101)
    assert snap["issues"] == ["max: no case completed for over 20 minutes; investigate before restarting"]
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "running"


def test_final_audit_does_not_require_completed_client_to_be_alive(tmp_path):
    setup_run(tmp_path)
    write(tmp_path / "max/run/answers/progress.json", dict(completed=260, expected=260, at=900))
    assert check(tmp_path, dead=(40,), now=4000)["issues"] == []


def test_completion_notifies_once_and_only_after_all_arms(tmp_path):
    setup_run(tmp_path)
    write(tmp_path / "status.json", dict(status="completed"))
    assert not check(tmp_path)["complete"]
    for arm in ARMS:
        write(tmp_path / arm / "status.json", dict(status="completed"))
    snap = check(tmp_path)
    assert snap["complete"]
    assert len(findings(snap, {})) == 4
    assert findings(snap, snap) == []


def test_low_disk_notifies_agent(tmp_path):
    setup_run(tmp_path)
    snap = inspect(tmp_path, now=1000, is_alive=lambda _: True,
                   disk_usage=lambda _: SimpleNamespace(free=1024**3))
    assert snap["issues"] == ["Experiment disk has less than 10 GiB free"]


def test_live_process_with_fatal_log_wakes_agent(tmp_path):
    setup_run(tmp_path)
    path = tmp_path / "max/run/server.log"
    path.write_text("normal\n" * 20000 + "Traceback (most recent call last):\n")
    assert len(check(tmp_path)["issues"]) == 1


@pytest.mark.parametrize("active,existing", [(True, False), (False, False), (False, True)])
def test_wake_uses_existing_chat_and_preserves_queue_order(monkeypatch, active, existing):
    class Socket:
        def __init__(self):
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def send(self, text):
            self.sent.append(json.loads(text))

        async def recv(self):
            last = self.sent[-1]
            method = last["method"]
            result = {}
            if method == "thread/queue/list":
                result = dict(data=[dict(id="old", input=[dict(text="【KVMem RULER 定时巡检】existing")])]
                              if existing else [])
            elif method == "thread/queue/add":
                result = dict(queuedSubmission=dict(id="new"))
            elif method == "thread/read":
                result = dict(thread=dict(status=dict(type="active" if active else "idle")))
            return json.dumps(dict(id=last["id"], result=result))

    socket = Socket()
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(unix_connect=lambda *a, **kw: socket))
    result = asyncio.run(wake("same-thread", "/unused.sock", "message", "stable-key"))
    assert result["started"] is (not active)
    methods = [r["method"] for r in socket.sent]
    assert ("thread/queue/add" in methods) is (not existing)
    if not active:
        assert socket.sent[-1]["params"] == dict(threadId="same-thread")
    for request in socket.sent:
        if request["method"].startswith("thread/"):
            assert request["params"]["threadId"] == "same-thread"
