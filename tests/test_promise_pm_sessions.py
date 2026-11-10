"""KVMEM_PROMISE_PM_SESSIONS: promised refreshes with P/M checkpoints, settled before every capture."""
import threading
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import pressure_promise as pp


def sessions(freeze=False, captures=(), terminals=(), checkpoint_at=None):
    entries = {'ns': NS(position=checkpoint_at)} if checkpoint_at is not None else {}
    return NS(options=NS(session_freeze_prompt=freeze), captures={r: 1 for r in captures},
              terminal_captures={r: 1 for r in terminals}, bindings={'r': NS(namespace='ns')},
              pool=NS(entries=entries))


def test_reasons(monkeypatch):
    monkeypatch.delenv('KVMEM_PROMISE_PM_SESSIONS', raising=False)
    assert pp.session_reason(None, 'r', 4096) is None
    assert pp.session_reason(sessions(freeze=True), 'r', 4096) is None
    assert pp.session_reason(sessions(), 'r', 4096) == 'session_checkpoint'          # original rule
    assert pp.session_reason(sessions(freeze=True, captures=['r']), 'r', 4096) == 'session_checkpoint'
    monkeypatch.setenv('KVMEM_PROMISE_PM_SESSIONS', '1')
    assert pp.session_reason(sessions(), 'r', 4096) is None
    assert pp.session_reason(sessions(captures=['r']), 'r', 4096) == 'session_checkpoint'
    assert pp.session_reason(sessions(terminals=['r']), 'r', 4096) == 'session_checkpoint'
    assert pp.session_reason(sessions(checkpoint_at=4096), 'r', 4096) == 'session_checkpoint_position'
    assert pp.session_reason(sessions(checkpoint_at=2048), 'r', 4096) is None
    monkeypatch.setenv('KVMEM_PROMISE_PM_SESSIONS', 'yes')
    with pytest.raises(ValueError):
        pp.session_reason(sessions(), 'r', 4096)


class Event:
    def __init__(self):
        self.synced = False

    def synchronize(self):
        self.synced = True


def lease(layers=3):
    return NS(promised=True, condition=threading.Condition(), events=[None] * layers,
              committed=threading.Event(), error=None)


def runner_with(active):
    return NS(_kvmem_layerwise_pressure=NS(pipeline=NS(active=active)))


def test_settle_waits_for_commit_and_every_layer():
    item = lease()
    runner = runner_with({'r': item})
    done = []
    worker = threading.Thread(target=lambda: done.append(pp.settle_before_checkpoint(runner, 'r')))
    worker.start()
    events = [Event() for _ in item.events]
    for index, event in enumerate(events):
        assert not done
        with item.condition:
            item.events[index] = event
            item.condition.notify_all()
    worker.join(.05)
    assert not done                       # every layer published, selection not committed yet
    item.committed.set()
    worker.join(2)
    assert done == [True] and all(e.synced for e in events)


def test_settle_is_a_noop_without_a_promised_lease_and_raises_on_failure():
    assert pp.settle_before_checkpoint(runner_with({}), 'r') is False
    assert pp.settle_before_checkpoint(NS(), 'r') is False
    plain = lease(); plain.promised = False
    assert pp.settle_before_checkpoint(runner_with({'r': plain}), 'r') is False
    failed = lease(); failed.error = ValueError('boom')
    with pytest.raises(RuntimeError, match='producer failed'):
        pp.settle_before_checkpoint(runner_with({'r': failed}), 'r')
