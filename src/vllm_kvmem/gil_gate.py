"""Opt-in KVMEM_GIL_GATE_MS: keep kvmem background Python off the GIL while the
engine thread prepares and launches a step.

The engine thread opens the gate while it blocks on the GPU (`sample_tokens`)
or on kvmem work (pipeline layer guards, the pressure handoff wait), and
closes it again when it resumes host work. Background threads (refresh,
publication) call `checkpoint()` before each unit of work. A closed gate parks
them on an Event, without the GIL, for at most KVMEM_GIL_GATE_MS milliseconds.
The bound means a missed open can only delay work, never deadlock.

This is scheduling only: the same work runs in the same order on each thread,
so selections, bytes and outputs are unchanged.
"""
import os
import threading
import time

from .tensor_parallel import active as _tp_active
from . import settings as _settings

_gate = None


class Gate:
    def __init__(self, max_wait_seconds, schedule_hold_seconds=.02):
        self.max_wait = max_wait_seconds
        self.schedule_hold = schedule_hold_seconds
        self.event = threading.Event()
        self.event.set()
        self.engine = None              # ident of the engine thread (first to close)
        self.depth = 0                  # nested engine waits that opened the gate
        self.deadline = None            # a closed gate reopens by itself at this time
        self.batch = frozenset()        # requests of the step being launched
        self.parked = 0
        self.parked_seconds = 0.0
        self.timeouts = 0

    def close(self, hold=None):
        if self.engine is None:
            self.engine = threading.get_ident()
        elif threading.get_ident() != self.engine:
            return
        if self.depth == 0:
            self.deadline = None if hold is None else time.perf_counter() + hold
            self.event.clear()

    def begin_step(self, requests, active=2):
        """Engine: host work for a step's launch starts. Production that this
        step's forward will wait for (its requests' layers) is never parked.
        With fewer than two live kvmem requests every background unit is on
        that request's critical path, so the gate stays open."""
        if self.engine is not None and threading.get_ident() != self.engine:
            return
        self.active = active
        if active < 2:
            self.batch = frozenset()
            self.open()
            return
        self.batch = frozenset(requests)
        self.close()

    def end_step(self):
        """Engine: the step's results arrived; the update and the next schedule
        follow. If no launch follows (idle engine), the gate reopens itself."""
        self.batch = frozenset()
        if getattr(self, 'active', 2) < 2:
            self.open()
            return
        self.close(self.schedule_hold)

    def open(self):
        if self.engine is not None and threading.get_ident() != self.engine:
            return
        self.event.set()

    def checkpoint(self, request_id=None):
        if self.event.is_set() or threading.get_ident() == self.engine:
            return
        if request_id is not None and request_id in self.batch:
            return                      # the step being launched waits for this work
        started = time.perf_counter()
        timeout = self.max_wait
        bounded = True                  # False when the gate's own deadline comes first
        deadline = self.deadline
        if deadline is not None:
            if deadline - started <= 0:
                return
            if deadline - started < timeout:
                timeout, bounded = deadline - started, False
        if not self.event.wait(timeout) and bounded:
            self.timeouts += 1          # a missed open, cut short by the bound
        self.parked += 1
        self.parked_seconds += time.perf_counter() - started

    def checkpoint_owners(self, request_ids):
        """checkpoint() for work that publishes these owners: never parked when
        one of them is in the step being launched (its forward waits for it)."""
        batch = self.batch
        if batch and not self.event.is_set():
            for request_id in request_ids:
                if request_id in batch:
                    self.owner_passes += 1
                    return
        self.checkpoint()

    owner_passes = 0
    blocked_opens = 0

    class _Blocked:
        """Engine blocked on a background condition: open the gate meanwhile and
        restore the closed state afterwards (unlike _Waiting, which re-closes
        only for a launching step and so suits waits inside the launch)."""
        __slots__ = ('gate', 'opened')

        def __init__(self, gate):
            self.gate, self.opened = gate, False

        def __enter__(self):
            gate = self.gate
            if gate.engine is not None and threading.get_ident() == gate.engine and not gate.event.is_set():
                self.opened = True
                gate.blocked_opens += 1
                gate.event.set()

        def __exit__(self, *exc):
            gate = self.gate
            if self.opened and gate.depth == 0:
                gate.event.clear()
            return False

    class _Waiting:
        def __init__(self, gate):
            self.gate = gate

        def __enter__(self):
            gate = self.gate
            if gate.engine is not None and threading.get_ident() == gate.engine:
                gate.depth += 1
                gate.event.set()

        def __exit__(self, *exc):
            gate = self.gate
            if gate.engine is not None and threading.get_ident() == gate.engine:
                gate.depth -= 1
                if gate.depth == 0 and gate.batch:
                    gate.deadline = None    # back to launching the step: close again
                    gate.event.clear()

    def engine_waiting(self):
        """Context for an engine-thread wait on background work: open meanwhile."""
        return Gate._Waiting(self)


def configured():
    """KVMEM_GIL_GATE_MS='max_wait[,schedule_hold]' in ms (hold default 20).
    '' or '0' turns the gate off (the default-on table is overridden with =0)."""
    value = _settings.get('KVMEM_GIL_GATE_MS', '')
    if value in ('', '0'):
        return None
    parts = value.split(',')
    if (len(parts) > 2 or not all(p.isdigit() for p in parts) or not 1 <= int(parts[0]) <= 1000 or
            (len(parts) == 2 and not 0 <= int(parts[1]) <= 1000)):
        raise ValueError('KVMEM_GIL_GATE_MS must be "max_wait[,schedule_hold]" in ms, max_wait 1..1000')
    return int(parts[0]) / 1e3, (int(parts[1]) if len(parts) == 2 else 20) / 1e3


def gate():
    """The process gate, created on first use when configured; None when off."""
    global _gate
    if _gate is None:
        config = configured()
        if config is None:
            return None
        _gate = Gate(*config)
    return _gate


def checkpoint(request_id=None):
    g = _gate if _gate is not None else gate()
    if g is not None:
        g.checkpoint(request_id)


class _Null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NULL = _Null()


def engine_waiting():
    g = _gate if _gate is not None else gate()
    return g.engine_waiting() if g is not None else _NULL


def drain_open_enabled():
    """Opt-in KVMEM_GIL_GATE_DRAIN_OPEN=1 (with the gate on): an engine-thread
    drain_requests(wait=True) opens the gate while it waits, and publication of
    an owner in the step being launched is never parked. Without it the engine
    can wait on a publication that is itself parked on the closed gate (up to
    KVMEM_GIL_GATE_MS per logical span). Scheduling only: same work, same order."""
    value = _settings.get('KVMEM_GIL_GATE_DRAIN_OPEN', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_GIL_GATE_DRAIN_OPEN must be 0 or 1')
    # A local owner becoming runnable cannot release the publication gate
    # on behalf of its TP peers. Keep the ordinary gate in that group.
    return value == '1' and not _tp_active()


def engine_blocked():
    """Context for an engine-thread wait on a background condition (opt-in
    KVMEM_GIL_GATE_DRAIN_OPEN): opens a closed gate meanwhile, then restores it."""
    g = _gate if _gate is not None else gate()
    if g is None or not drain_open_enabled():
        return _NULL
    return Gate._Blocked(g)


def checkpoint_owners(request_ids):
    """checkpoint() for the publication of these owners (opt-in
    KVMEM_GIL_GATE_DRAIN_OPEN: owners of the launching step pass)."""
    g = _gate if _gate is not None else gate()
    if g is None:
        return
    if drain_open_enabled():
        g.checkpoint_owners(request_ids)
    else:
        g.checkpoint()


def chunk_checkpoint(request_id=None):
    """Opt-in KVMEM_GIL_GATE_CHUNKS=1 (with the gate on): also park between the
    staged chunks of one layer's assembly, so a background layer that started
    before the engine closed the gate stops within one chunk, not one layer."""
    value = _settings.get('KVMEM_GIL_GATE_CHUNKS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_GIL_GATE_CHUNKS must be 0 or 1')
    if value == '1':
        checkpoint(request_id)
