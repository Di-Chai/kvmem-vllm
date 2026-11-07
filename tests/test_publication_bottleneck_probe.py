from copy import copy
from types import SimpleNamespace as NS

import pytest

from prefill_phase_probe import PhaseScopes
from publication_bottleneck_probe import PublicationTimings


def test_timing_keeps_staged_owner_and_nested_return(monkeypatch):
    worker = NS(profiler=NS(is_running=True))
    events = []
    timing = PublicationTimings(PhaseScopes(worker, push=events.append, pop=lambda: events.append('pop')))
    class Owner:
        def __init__(self): self.value = 0
        def inner(self, amount): self.value += amount; return self.value
        def outer(self, amount): return self.inner(amount)
    for name in ('inner', 'outer'):
        monkeypatch.setattr(Owner, name, getattr(Owner, name))
        timing.wrap(Owner, name, name)
        wrapped = getattr(Owner, name)
        timing.wrap(Owner, name, name)
        assert getattr(Owner, name) is wrapped
    live = Owner(); staged = copy(live)
    assert staged.outer(3) == 3 and live.value == 0
    inner, outer = timing.rows
    assert outer['start_ns'] <= inner['start_ns'] <= inner['end_ns'] <= outer['end_ns']
    assert outer['cpu_ns'] >= inner['cpu_ns'] >= 0
    assert events == ['prefill_phase::outer', 'prefill_phase::inner', 'pop', 'pop']
    worker.profiler.is_running = False
    assert staged.outer(4) == 7 and len(timing.rows) == 2


def test_exception_and_context_do_not_change_operation():
    worker = NS(profiler=NS(is_running=True)); events=[]
    timing = PublicationTimings(PhaseScopes(worker, push=events.append, pop=lambda: events.append('pop')))
    error = ValueError('original')
    class Owner:
        def call(self, value): raise error
    timing.wrap(Owner, 'call', 'failure', context=lambda self, value: dict(input=value))
    with pytest.raises(ValueError) as caught: Owner().call(12)
    assert caught.value is error and timing.rows[0]['failed']
    assert timing.rows[0]['input'] == 12 and events[-1] == 'pop'
