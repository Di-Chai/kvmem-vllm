"""Opt-in concurrent prefill-pressure lanes: admission and credit rules."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.async_refresh import AsyncRefresh
from vllm_kvmem.serving.maintenance import MaintenancePolicy


def policy():
    options = NS(maintenance_max_steps=8, maintenance_busy_ms=2., maintenance_idle_ms=8.,
                 maintenance_aging_ms=250., maintenance_inflight_bytes=0,
                 memory=NS(block_tokens=32, select_budget=4096, gen_budget=1024))
    return MaintenancePolicy(options, bytes_per_token=4)


def job(rid, history=2048):
    return NS(command=NS(request_id=rid, groups=[(0, NS(window_tokens=4096))]), history=history)


def test_single_lane_credit_is_unchanged():
    p = policy()
    a, b = job('a'), job('b')
    assert p.acquire(a) and not p.acquire(b)
    assert p.owner == 'a' and p.inflight_bytes == p.cost(a)
    with pytest.raises(ValueError, match='Foreign'):
        p.release(b)
    p.release(a)
    assert p.owner is None and p.inflight_bytes == 0 and p.acquire(b)


def test_two_lanes_admit_two_pressure_owners_and_never_mix_with_ordinary_work():
    p = policy()
    a, b, c, ordinary = job('a'), job('b'), job('c'), job('q')
    assert p.acquire(a, lanes=2) and p.acquire(b, lanes=2)
    assert not p.acquire(c, lanes=2)              # both lanes busy
    assert not p.acquire(ordinary)                # ordinary job needs an idle lane set
    assert p.inflight_bytes == p.cost(a) + p.cost(b)
    p.release(a)
    assert p.owner == 'b' and p.acquire(c, lanes=2)
    p.release(b); p.release(c)
    assert p.acquire(ordinary) and not p.acquire(a, lanes=2)   # nor pressure behind ordinary
    with pytest.raises(ValueError, match='positive'):
        p.acquire(a, lanes=0)


def manager(lanes, jobs=(), active=()):
    m = AsyncRefresh.__new__(AsyncRefresh)
    pipeline = NS(active={rid: NS(job=j) for rid, j in active})
    m.runner = NS(_kvmem_layerwise_pressure=NS(lanes=lanes, pipeline=pipeline))
    m.jobs = {j.command.request_id: j for j in jobs}
    return m


def pjob(rid, *, lane=0, future=None, done=False, selection='pressure', stage='prefill'):
    return NS(command=NS(request_id=rid, selection=selection, pressure_stage=stage),
              guided_pressure=False, future=future, done=done, lane=lane)


def test_lane_claims_follow_running_and_dispatched_producers():
    first = pjob('a', future=object())
    assert manager(2, [first])._claim_lane(pjob('b')) == 1
    # A dispatched pipeline job is done but still owns lane 0 scratch.
    dispatched = pjob('a', future=object(), done=True)
    assert manager(2, [], [('a', dispatched)])._claim_lane(pjob('b')) == 1
    both = manager(2, [pjob('a', future=object()), pjob('b', lane=1, future=object())])
    assert both._claim_lane(pjob('c')) is None
    # A non-pressure producer keeps exclusive use of the refresh machinery.
    query = pjob('q', future=object(), selection='query')
    assert manager(2, [query])._claim_lane(pjob('b')) is None
    assert not manager(1)._multi_lane(pjob('b'))
    assert manager(2)._multi_lane(pjob('b')) and not manager(2)._multi_lane(pjob('b', stage='decode'))


def test_stream_is_per_lane_thread_local():
    import threading
    m = AsyncRefresh.__new__(AsyncRefresh)
    m.stream = 'lane0'
    seen = []
    def worker():
        m.__dict__.setdefault('_lane_local', threading.local()).index = 1
        seen.append(m.stream); m.stream = 'lane1'; seen.append(m.stream)
    t = threading.Thread(target=worker); t.start(); t.join()
    assert seen == [None, 'lane1'] and m.stream == 'lane0'


def test_lane_staging_is_only_defined_for_refresh_assembly():
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    with pytest.raises(ValueError, match='only defined for refresh'):
        NativeWindowAssembler.from_runner(NS(), 'r', (), arena='pressure', staging_lane=1)


def test_lanes_flag_requires_the_guarded_pipeline(monkeypatch):
    from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure
    query = NS(runner=NS(model_config=NS(enforce_eager=True)), names=('l0',), optimized=True,
               native_batches=True, paired_reads=True)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LANES', '3')
    with pytest.raises(ValueError, match='1 or 2'):
        LayerwisePrefillPressure(query)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LANES', '2')
    with pytest.raises(ValueError, match='guarded pipeline handoff'):
        LayerwisePrefillPressure(query)
