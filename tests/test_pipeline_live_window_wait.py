"""An early pipeline ACK must not refresh a live P/M checkpoint before every layer
is materialized.

dsh c8 crash (2026-11-10 15:20): PressurePipeline.dispatch -> SessionWorker.
selection_committed -> PromptCheckpointPool.update_live_window -> memory.
checkpoint_window raised 'Whole-window access preceded layer materialization
completion' because only captures waited for the whole lease.
"""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.pressure_pipeline import PressurePipeline, updates_live_window


class Committed(Exception):
    """Raised once dispatch reaches commit_layerwise_selection (no wait)."""


def make(monkeypatch, *, position, history=100, ready=False, freeze=False, captures=(), binding=True):
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1')
    lease = NS(dispatchable=lambda: True, query=lambda: ready)

    def commit(_lease):
        raise Committed

    job = NS(layerwise_pipeline=lease, cancelled=False, pressure_oracle=None, command=NS(request_id='r'),
             authority=NS(memory=NS(commit_layerwise_selection=commit)), history=history)
    sessions = NS(options=NS(session_freeze_prompt=freeze), captures=set(captures), terminal_captures=set(),
                  bindings={'r': NS(namespace='ns')} if binding else {},
                  pool=NS(entries={'ns': NS(position=position, request_id='r')}))
    pipeline = PressurePipeline.__new__(PressurePipeline)
    pipeline.active = {}
    pipeline.runner = NS(_kvmem_sessions=sessions)
    events = []
    refresh = NS(emit=lambda _job, name, **_kw: events.append(name))
    return pipeline, refresh, job, sessions, events


def test_live_checkpoint_at_position_waits_for_whole_lease(monkeypatch):
    pipeline, refresh, job, sessions, events = make(monkeypatch, position=100)
    assert updates_live_window(sessions, 'r', 100)
    assert pipeline.dispatch(refresh, job) is False
    assert pipeline.dispatch(refresh, job) is False
    assert events == ['pipeline_checkpoint_boundary_wait']   # reported once
    assert pipeline.active == {}


def test_live_checkpoint_dispatches_once_materialized(monkeypatch):
    pipeline, refresh, job, _sessions, _events = make(monkeypatch, position=100, ready=True)
    with pytest.raises(Committed):
        pipeline.dispatch(refresh, job)


@pytest.mark.parametrize('kwargs', [dict(position=96), dict(position=100, freeze=True),
                                    dict(position=100, binding=False)])
def test_other_refreshes_keep_the_early_ack(monkeypatch, kwargs):
    pipeline, refresh, job, sessions, events = make(monkeypatch, **kwargs)
    assert not updates_live_window(sessions, 'r', 100)
    with pytest.raises(Committed):
        pipeline.dispatch(refresh, job)
    assert events == []


def test_capture_still_waits(monkeypatch):
    pipeline, refresh, job, _sessions, events = make(monkeypatch, position=96, captures=('r',))
    assert pipeline.dispatch(refresh, job) is False
    assert events == ['pipeline_checkpoint_boundary_wait']
