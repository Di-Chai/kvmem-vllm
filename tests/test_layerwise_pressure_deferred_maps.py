"""KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS: same maps/evidence, built after the plan/ACK."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.layerwise_pressure import DeferredPreparedCommit, PendingMaps


def run(deferred, build_before_commit):
    from test_layerwise_pressure import controller, options, native_authority, finish
    c = controller(); c.prepared_commit = True; c.deferred_maps = deferred
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); hs = c._history('r', o)
    rows = list(a.records.values())
    for i, history in enumerate(hs):
        history.values = {r.history_begin: [1. if j in ((2, 3) if i == 0 else (5, 6)) else .01, 1.]
                          for j, r in enumerate(rows)}
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    emitted = []; c.query.emit = lambda event, **fields: emitted.append((event, fields))
    try:
        c.prepare(NS(options=o), job)
        assert isinstance(job.layerwise_prepared_commit, DeferredPreparedCommit) == deferred
        chosen = job.layerwise_pressure_plan[0]
        slots = {r.handle.slot for r in chosen[0]}
        a.memory.select({}, pressure_scores=[(r.handle.slot, float(r.handle.slot in slots)) for r in rows],
                        max_tokens=96)
        finish(a.memory)
        job.selected_tokens = 96
        job.layerwise_single_assembly_done = True
        if deferred and build_before_commit:
            job.layerwise_prepared_commit.value()
        c.committed(job)
        entry = c.maps['r']
        if deferred and not build_before_commit:
            assert isinstance(entry, PendingMaps)
        resolved = c.resolved_maps('r')
        return resolved, emitted[-1]
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('build_before_commit', [False, True])
def test_deferred_maps_resolve_to_the_eager_maps_and_evidence(build_before_commit):
    eager_maps, eager_event = run(False, False)
    maps, event = run(True, build_before_commit)
    assert maps == eager_maps and event == eager_event


def test_deferred_builder_runs_once_and_reraises_failures():
    calls = []
    def ok():
        calls.append(1); return 'maps'
    d = DeferredPreparedCommit(ok, frozenset())
    assert not d.ready and d.value() == 'maps' and d.value() == 'maps' and d.ready and calls == [1]
    def bad():
        raise RuntimeError('geometry')
    d = DeferredPreparedCommit(bad, frozenset())
    for _ in range(2):
        with pytest.raises(RuntimeError, match='geometry'):
            d.value()


def test_deferred_maps_wait_for_every_lazily_selected_layer():
    """KVMEM_LAYER0_FAST lets the next capture ask for the maps while the producer
    is still selecting later layers; the build must see all of them."""
    import threading, time
    chosen, selected = [['layer0']], threading.Event()
    d = DeferredPreparedCommit(lambda: tuple(map(tuple, chosen)), frozenset(), selected)
    result = {}
    consumer = threading.Thread(target=lambda: result.setdefault('maps', d.value()))
    consumer.start()
    time.sleep(0.05)
    assert consumer.is_alive() and not d.ready           # waits, holding no lock
    for layer in range(1, 16):                            # the producer keeps selecting
        chosen.append([f'layer{layer}'])
    selected.set()
    consumer.join(5)
    assert len(result['maps']) == 16 and d.ready


def test_deferred_maps_release_fails_waiting_consumers_and_times_out():
    import threading
    selected = threading.Event()
    d = DeferredPreparedCommit(lambda: 'maps', frozenset(), selected)
    errors = []
    def consume():
        try:
            d.value()
        except RuntimeError as error:
            errors.append(str(error))
    consumer = threading.Thread(target=consume)
    consumer.start()
    d.release(RuntimeError('producer failed'))
    consumer.join(5)
    assert errors == ['producer failed']
    with pytest.raises(RuntimeError, match='producer failed'):
        d.value()
    stuck = DeferredPreparedCommit(lambda: 'maps', frozenset(), threading.Event())
    stuck.SELECTION_TIMEOUT = 0.01
    with pytest.raises(RuntimeError, match='did not finish'):
        stuck.value()
    built = DeferredPreparedCommit(lambda: 'maps', frozenset(), threading.Event())
    built._selected.set()
    assert built.value() == 'maps'
    built.release(RuntimeError('late'))                   # after a build: no effect
    assert built.value() == 'maps'


def test_discarded_deferred_maps_drop_their_builder_and_refuse_late_use():
    from vllm_kvmem.layerwise_pressure import DeferredPreparedCommit
    import pytest as _pytest
    built = []
    deferred = DeferredPreparedCommit(lambda: built.append(1) or 'maps', frozenset())
    deferred.discard()
    assert deferred._build is None and not built
    with _pytest.raises(RuntimeError, match='discarded with their request'):
        deferred.value()
    ready = DeferredPreparedCommit(lambda: 'maps', frozenset())
    assert ready.value() == 'maps'
    ready.discard()
    assert ready.value() == 'maps'                    # a built value stays valid
