"""KVMEM_GIL_GATE_MS: background checkpoints park only while the engine runs host work."""
import threading
import time

import pytest

from vllm_kvmem.serving import gil_gate


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(gil_gate, '_gate', None)
    yield
    monkeypatch.setattr(gil_gate, '_gate', None)


def run_in_thread(fn):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault('v', fn()))
    t.start(); t.join(5)
    return out.get('v')


def test_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_GIL_GATE_MS', raising=False)
    assert gil_gate.gate() is None
    gil_gate.checkpoint()                       # no-op
    with gil_gate.engine_waiting():
        pass
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '0')    # =0 overrides the default-on table: off
    assert gil_gate.configured() is None and gil_gate.gate() is None
    for bad in ('1001', 'x', '-1', '0,5'):
        monkeypatch.setenv('KVMEM_GIL_GATE_MS', bad)
        with pytest.raises(ValueError):
            gil_gate.configured()


def test_closed_gate_parks_background_until_opened_and_never_the_engine(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    g = gil_gate.gate()
    g.close()                                   # this (engine) thread closes
    gil_gate.checkpoint()                       # the engine itself is never parked
    done = []
    t = threading.Thread(target=lambda: (gil_gate.checkpoint(), done.append(time.perf_counter())))
    started = time.perf_counter()
    t.start()
    time.sleep(.2)
    assert not done                             # still parked
    g.open()                                    # the engine blocks on the GPU
    t.join(5)
    assert .15 < done[0] - started < .9 and g.parked == 1 and g.timeouts == 0


def test_a_missed_open_only_delays_by_the_bound(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '50')
    g = gil_gate.gate()
    g.close()
    started = time.perf_counter()
    run_in_thread(gil_gate.checkpoint)
    assert .04 < time.perf_counter() - started < .5 and g.timeouts == 1


def test_engine_waits_on_background_work_open_the_gate(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    g = gil_gate.gate()
    g.close()
    with gil_gate.engine_waiting():
        started = time.perf_counter()
        run_in_thread(gil_gate.checkpoint)      # open while the engine waits
        assert time.perf_counter() - started < .1
        g.close()                               # nested close is ignored while waiting
        assert g.event.is_set()
    g.close()
    assert not g.event.is_set()


def test_only_the_engine_thread_opens_or_closes(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    g = gil_gate.gate()
    g.close()
    run_in_thread(g.open)                       # a background thread cannot open it
    assert not g.event.is_set()
    g.open()
    run_in_thread(g.close)                      # nor close it
    assert g.event.is_set()


def test_schedule_hold_reopens_an_idle_engine_and_step_requests_are_never_parked(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000,30')
    assert gil_gate.configured() == (1.0, .03)
    g = gil_gate.gate()
    g.end_step()                                # results in; schedule follows
    started = time.perf_counter()
    run_in_thread(gil_gate.checkpoint)          # idle engine: reopens after the hold
    assert time.perf_counter() - started < .3 and g.timeouts == 0
    g.begin_step({'a': 1, 'b': 2})              # launching a step for a and b
    started = time.perf_counter()
    run_in_thread(lambda: gil_gate.checkpoint('a'))   # a's own layers: never parked
    assert time.perf_counter() - started < .1
    with gil_gate.engine_waiting():             # a guard wait opens the gate
        assert g.event.is_set()
    assert not g.event.is_set()                 # and the launch closes it again


def test_a_single_live_request_keeps_the_gate_open(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    g = gil_gate.gate()
    g.begin_step({'a': 1}, active=1)            # N=1: everything is a's critical path
    assert g.event.is_set()
    g.end_step()
    assert g.event.is_set()
    g.begin_step({'a': 1}, active=2)            # another live request: gate as usual
    assert not g.event.is_set()


def test_chunk_checkpoint_is_opt_in(monkeypatch):
    from vllm_kvmem.serving import gil_gate
    calls = []
    monkeypatch.setattr(gil_gate, 'checkpoint', lambda request_id=None: calls.append(request_id))
    monkeypatch.delenv('KVMEM_GIL_GATE_CHUNKS', raising=False)
    gil_gate.chunk_checkpoint('r')
    assert calls == []
    monkeypatch.setenv('KVMEM_GIL_GATE_CHUNKS', '1')
    gil_gate.chunk_checkpoint('r')
    assert calls == ['r']
    monkeypatch.setenv('KVMEM_GIL_GATE_CHUNKS', 'on')
    import pytest
    with pytest.raises(ValueError):
        gil_gate.chunk_checkpoint('r')


def test_drain_open_is_opt_in_and_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '50')
    monkeypatch.delenv('KVMEM_GIL_GATE_DRAIN_OPEN', raising=False)
    g = gil_gate.gate()
    g.close()
    with gil_gate.engine_blocked():
        assert not g.event.is_set()                 # off: the gate stays closed
    g.begin_step(['a', 'b'])
    started = time.perf_counter()
    run_in_thread(lambda: gil_gate.checkpoint_owners(['a']))
    assert time.perf_counter() - started > .04      # off: plain checkpoint, parked
    monkeypatch.setenv('KVMEM_GIL_GATE_DRAIN_OPEN', 'x')
    with pytest.raises(ValueError):
        gil_gate.drain_open_enabled()


def test_engine_blocked_opens_then_restores_and_ignores_other_threads(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    monkeypatch.setenv('KVMEM_GIL_GATE_DRAIN_OPEN', '1')
    g = gil_gate.gate()
    g.close()
    with gil_gate.engine_blocked():
        assert g.event.is_set()
        started = time.perf_counter()
        run_in_thread(gil_gate.checkpoint)          # background runs while the engine waits
        assert time.perf_counter() - started < .1
    assert not g.event.is_set() and g.blocked_opens == 1
    g.open()                                        # an open gate is left open
    with gil_gate.engine_blocked():
        pass
    assert g.event.is_set() and g.blocked_opens == 1
    g.close()

    def other():
        with gil_gate.engine_blocked():
            return g.event.is_set()
    assert run_in_thread(other) is False            # only the engine thread opens


def test_launching_step_owners_are_never_parked(monkeypatch):
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '200')
    monkeypatch.setenv('KVMEM_GIL_GATE_DRAIN_OPEN', '1')
    g = gil_gate.gate()
    g.begin_step(['a', 'b'])
    assert not g.event.is_set()
    started = time.perf_counter()
    run_in_thread(lambda: gil_gate.checkpoint_owners(['z', 'b']))
    assert time.perf_counter() - started < .1 and g.owner_passes == 1
    started = time.perf_counter()
    run_in_thread(lambda: gil_gate.checkpoint_owners(['z']))
    assert time.perf_counter() - started > .15 and g.parked == 1


def test_engine_drain_wait_opens_the_gate(monkeypatch):
    """CaptureRuntime.drain_requests(wait=True) on the engine thread: a parked
    publisher proceeds, and the gate is closed again afterwards."""
    from vllm_kvmem.serving.capture_runtime import CaptureRuntime
    monkeypatch.setenv('KVMEM_GIL_GATE_MS', '1000')
    monkeypatch.setenv('KVMEM_GIL_GATE_DRAIN_OPEN', '1')
    g = gil_gate.gate()
    g.begin_step(['x'])                             # launching another request's step
    condition = threading.Condition()
    state = {'published': False}

    class Fake:
        def _publication_lock(self):
            return condition

        def _raise_publication_error(self):
            pass

        def _claim_publications(self, wanted, wait):
            return None, not state['published'], False

    def publisher():
        gil_gate.checkpoint_owners(['a'])           # parked until the engine waits
        with condition:
            state['published'] = True
            condition.notify_all()
    t = threading.Thread(target=publisher)
    started = time.perf_counter()
    t.start()
    assert CaptureRuntime.drain_requests(Fake(), ['a'], wait=True) is True
    t.join(5)
    assert time.perf_counter() - started < .5 and g.blocked_opens >= 1
    assert not g.event.is_set()                     # the launch continues with the gate closed
