"""Opt-in cpu_session_ttl_ms: idle sessions move to SSD and come back before their next turn."""
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.session_tiering import SessionTiering


class Inline:
    """Executor running jobs when the test says so."""
    def __init__(self):
        self.pending = []

    def submit(self, fn, *args):
        f = Future()
        self.pending.append((f, fn, args))
        return f

    def run(self):
        while self.pending:
            f, fn, args = self.pending.pop(0)
            if f.cancelled():
                continue
            if not f.running():
                f.set_running_or_notify_cancel()
            f.set_result(fn(*args))

    def start(self):
        for f, _, _ in self.pending:
            if not f.running() and not f.cancelled():
                f.set_running_or_notify_cancel()

    def shutdown(self, wait=True):
        pass


class Tier:
    def __init__(self):
        self.ram, self.ssd = set(), set()

    def spill(self, allocations):
        moved = [a for a in allocations if a in self.ram]
        self.ram.difference_update(moved); self.ssd.update(moved)
        return len(moved)

    def promote(self, allocations):
        moved = [a for a in allocations if a in self.ssd]
        self.ssd.difference_update(moved); self.ram.update(moved)
        return len(moved)


def session(name, slots, *, partial=()):
    records = {}
    for slot in slots:
        records[slot] = NS(archived=True, raw=None, handle=NS(slot=slot, generation=1))
    for slot in partial:
        records[slot] = NS(archived=False, raw=b'x', handle=NS(slot=slot, generation=1))
    return NS(memory=NS(request_id=name), records=records)


def setup(ttl=10.0):
    now = [0.0]
    tier, ex = Tier(), Inline()
    entries = {}
    archive = NS(tier=tier, _entries=entries)
    t = SessionTiering(archive, ttl, clock=lambda: now[0], executor=ex)
    return t, tier, ex, entries, now


def register(entries, tier, authority):
    for r in authority.records.values():
        if r.archived:
            alloc = ('alloc', authority.memory.request_id, r.handle.slot)
            entries[(authority.memory.request_id, r.handle.slot, 1)] = NS(allocation=alloc, ready=True)
            tier.ram.add(alloc)


def test_idle_session_spills_after_ttl_and_promotes_before_its_next_turn():
    t, tier, ex, entries, now = setup()
    auth = session('s1', [1, 2, 3], partial=[4])
    register(entries, tier, auth)
    t.idle('ns1', auth)
    now[0] = 5.0; t.tick(); ex.run()
    assert not tier.ssd                                   # before the TTL nothing moves
    now[0] = 11.0; t.tick(); ex.run(); t.tick()
    assert len(tier.ssd) == 3 and t.snapshot()['spilled_now'] == 1   # partial record never spilled
    assert not t.ready('ns1', auth)                       # promotion starts; the turn waits
    ex.run()
    assert t.ready('ns1', auth) and not tier.ssd and len(tier.ram) == 3
    s = t.snapshot()
    assert s['spilled_sessions'] == 1 and s['promoted_sessions'] == 1 and s['promoted_records'] == 3


def test_active_session_never_spills_and_spill_in_flight_is_awaited():
    t, tier, ex, entries, now = setup()
    auth = session('s2', [7])
    register(entries, tier, auth)
    t.idle('ns2', auth)
    now[0] = 1.0
    assert t.ready('ns2', auth)                           # returned before the TTL
    now[0] = 100.0; t.tick(); ex.run()
    assert not tier.ssd
    t.idle('ns2', auth); now[0] = 200.0; t.tick()         # spill submitted ...
    ex.start()                                            # ... and already running
    assert not t.ready('ns2', auth)                       # waits for the spill, then promotes
    ex.run(); assert not t.ready('ns2', auth)
    ex.run(); assert t.ready('ns2', auth) and not tier.ssd


def test_ensure_blocks_until_in_ram_and_validation():
    t, tier, ex, entries, now = setup()
    auth = session('s3', [9])
    register(entries, tier, auth)
    t.idle('ns3', auth); now[0] = 50.0; t.tick(); ex.run(); t.tick()
    assert tier.ssd

    class Eager(Inline):
        def submit(self, fn, *args):
            f = Future(); f.set_result(fn(*args)); return f
    t.executor = t.promoter = Eager()
    t.ensure('ns3', auth)
    assert not tier.ssd
    with pytest.raises(ValueError):
        SessionTiering(NS(tier=None), 1.0)
    with pytest.raises(ValueError):
        SessionTiering(NS(tier=tier), 0)


def test_returning_session_cancels_its_pending_spill_and_promotions_do_not_queue_behind_spills():
    t, tier, ex, entries, now = setup()
    a, b = session('a', [1, 2]), session('b', [3])
    register(entries, tier, a); register(entries, tier, b)
    t.idle('na', a); t.idle('nb', b)
    now[0] = 50.0; t.tick()                      # one spill at a time: a's queued, not started
    assert t.snapshot()['jobs'] == 1
    assert t.ready('na', a)                      # cancelled before it ran: still in RAM
    assert t.snapshot()['spills_cancelled'] == 1 and tier.ram >= {('alloc', 'a', 1), ('alloc', 'a', 2)}
    t.tick(); ex.run(); t.tick()                 # then b's spill runs
    assert ('alloc', 'b', 3) in tier.ssd


def test_no_spill_starts_while_a_promotion_runs():
    t, tier, ex, entries, now = setup()
    a, b = session('a', [1]), session('b', [2])
    register(entries, tier, a); register(entries, tier, b)
    t.idle('na', a); now[0] = 50.0; t.tick(); ex.run(); t.tick()
    assert ('alloc', 'a', 1) in tier.ssd
    assert not t.ready('na', a)                  # promotion submitted, not finished
    t.idle('nb', b); now[0] = 100.0; t.tick()
    assert ('alloc', 'b', 2) in tier.ram and t.snapshot()['jobs'] == 1   # b waits for the device
    ex.run(); assert t.ready('na', a)
    t.tick(); ex.run(); t.tick()
    assert ('alloc', 'b', 2) in tier.ssd


def test_prefetch_promotes_a_returning_session_before_its_restore_asks():
    t, tier, ex, entries, now = setup()
    auth = session('p', [1, 2])
    register(entries, tier, auth)
    t.idle('np', auth); now[0] = 50.0; t.tick(); ex.run(); t.tick()
    assert len(tier.ssd) == 2
    t.prefetch('np')                              # the turn arrived at the scheduler
    assert t.snapshot()['jobs'] == 1 and t.snapshot()['prefetched'] == 1
    ex.run()                                      # promoted while the request queued
    assert t.ready('np', auth) and not tier.ssd   # the restore finds it in RAM: no wait
    assert t.snapshot()['promote_waits'] == 0
    t.prefetch('np')                              # nothing spilled: no-op
    assert t.snapshot()['jobs'] == 0


def test_prefetch_cancels_a_queued_spill_and_follows_a_running_one():
    t, tier, ex, entries, now = setup()
    a, b = session('a', [1]), session('b', [2])
    register(entries, tier, a); register(entries, tier, b)
    t.idle('na', a); now[0] = 50.0; t.tick()      # a's spill queued, not started
    t.prefetch('na')
    assert t.snapshot()['spills_cancelled'] == 1 and ('alloc', 'a', 1) in tier.ram
    t.idle('nb', b); now[0] = 100.0; t.tick(); ex.start()   # b's spill is running
    t.prefetch('nb')
    assert t.wanted == {'nb'}
    ex.run(); t.tick()                            # reaped, then promoted right away
    assert t.snapshot()['prefetched'] == 1 and not t.wanted
    ex.run(); t.tick()
    assert t.ready('nb', b) and ('alloc', 'b', 2) in tier.ram
    t.prefetch('gone'); t.forget('nb')            # unknown and evicted sessions are ignored
    assert not t.wanted and t.snapshot()['jobs'] == 0


def test_tiering_prefetch_flag(monkeypatch):
    from vllm_kvmem.session_tiering import tiering_prefetch
    monkeypatch.delenv('KVMEM_TIERING_PREFETCH', raising=False)
    assert tiering_prefetch()
    monkeypatch.setenv('KVMEM_TIERING_PREFETCH', '0')
    assert not tiering_prefetch()
    monkeypatch.setenv('KVMEM_TIERING_PREFETCH', 'on')
    with pytest.raises(ValueError):
        tiering_prefetch()
