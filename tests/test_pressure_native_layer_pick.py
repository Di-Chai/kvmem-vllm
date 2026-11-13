"""KVMEM_PRESSURE_NATIVE_LAYER_PICK: the fused native layer pick equals the Python path.

Same chosen records per layer, evidence, resolved maps and store calls, and the
same exceptions in the same order for invalid candidates or scores.
"""
import random
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import _native
from vllm_kvmem import _prefill_state_native as prefill_native
from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory

pytestmark = pytest.mark.skipif(
    not hasattr(_native.Store, 'pressure_layer_pick') or not hasattr(prefill_native.State, 'scores_for'),
    reason='Requires the rebuilt native extensions')

COUNT, BUDGET = 512, 256          # 16 blocks, 8 selected (fits K + G of native_authority)


class CountingStore:
    """Forwards every call; counts the fused native pick."""
    def __init__(self, store):
        self.__dict__['_store'], self.__dict__['fused'] = store, 0

    def pressure_layer_pick(self, *args, **kwargs):
        result = self._store.pressure_layer_pick(*args, **kwargs)
        self.__dict__['fused'] += result is not None
        return result

    def __getattr__(self, name):
        return getattr(self._store, name)


def scores(kind, layer, rows):
    rnd = random.Random(7 + layer)
    values = {}
    for j, r in enumerate(rows):
        mass = rnd.choice([0., .25, .5, 1., rnd.random()])        # many ties and zeros
        if kind == 'nan' and layer == 1 and j == 5: mass = float('nan')
        if kind == 'negative' and layer == 0 and j == 3: mass = -1.
        if kind == 'zero': mass = 0.
        values[r.history_begin] = [mass, 1.]
    return values


def run(monkeypatch, *, fused, lazy, kind='ok', missing=False):
    from test_layerwise_pressure import controller, options, native_authority, finish
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '1')
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LAYER_PICK', '1' if fused else '0')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST', '1' if lazy else '0')
    c = controller(); c.prepared_commit = True; c.deferred_maps = True; c.fast_prepare = True
    a = native_authority(count=COUNT, budget=BUDGET, sink=1)
    a.publish_sealed = lambda wait: None
    store = a.memory._store = CountingStore(a.memory._store)
    o = options(a.memory.config); o.native_prefill_commit = True
    c._history('r', o)
    rows = list(a.records.values())
    hs = []
    for layer in range(len(c.names)):
        h = BufferedPrefillHistory(o)
        h.values = scores(kind, layer, rows)
        h.end, h.samples, h.chunks = COUNT, 4, 1
        hs.append(h)
    if missing:
        rows[4].count = 31        # the candidate no longer matches its accepted record
    c.histories['r'] = hs
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=BUDGET)),)), history=COUNT, authority=a)
    if lazy:
        job.promised_lease = object()
    emitted = []; c.query.emit = lambda event, **fields: emitted.append((event, fields))
    try:
        try:
            c.prepare(NS(options=o), job)
            chosen, evidence = job.layerwise_pressure_plan
            slots = {r.handle.slot for r in chosen[0]}
            a.memory.select({}, pressure_scores=[(r.handle.slot, float(r.handle.slot in slots)) for r in rows],
                            max_tokens=BUDGET)
            finish(a.memory)
            job.selected_tokens = BUDGET
            job.layerwise_single_assembly_done = True
            c.committed(job)
            if lazy:
                job.layerwise_select_layer(1)
            job.layerwise_prepared_commit.value()
        except (RuntimeError, ValueError) as error:
            return ('error', type(error), str(error)), store.fused
        return ([[r.history_begin for r in layer] for layer in chosen], list(job.layerwise_zero_slots),
                emitted[-1], c.resolved_maps('r')), store.fused
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('lazy', [False, True])
def test_fused_layer_pick_matches_python_selection_maps_and_evidence(monkeypatch, lazy):
    python, calls = run(monkeypatch, fused=False, lazy=lazy)
    native, fused_calls = run(monkeypatch, fused=True, lazy=lazy)
    assert calls == 0 and fused_calls == 2            # both layers took the native path
    assert native == python
    assert python[0][0] != python[0][1]                # the layers really differ


@pytest.mark.parametrize('kind', ['nan', 'negative', 'zero'])
@pytest.mark.parametrize('lazy', [False, True])
def test_invalid_scores_raise_the_same_error(monkeypatch, kind, lazy):
    python, _ = run(monkeypatch, fused=False, lazy=lazy, kind=kind)
    native, _ = run(monkeypatch, fused=True, lazy=lazy, kind=kind)
    assert python[0] == 'error' and native == python


@pytest.mark.parametrize('lazy', [False, True])
def test_changed_candidate_raises_the_same_error(monkeypatch, lazy):
    python, _ = run(monkeypatch, fused=False, lazy=lazy, missing=True)
    native, _ = run(monkeypatch, fused=True, lazy=lazy, missing=True)
    assert python[0] == 'error' and native == python


def test_unexpected_shapes_fall_back_before_any_side_effect():
    from test_recall_experiment import native_authority
    a = native_authority(count=COUNT, budget=BUDGET, sink=1)
    store = a.memory._store
    called = []
    try:
        mapping = [(0, 0, 0, 32, 0)]
        assert store.pressure_layer_pick(mapping, {}, [], [], BUDGET, {}, [], lambda: called.append(1)) is None
        assert store.pressure_layer_pick([[0, 0, 0, 32, 0]], {}, [1.], [], BUDGET, {}, [],
                                         lambda: called.append(1)) is None
        assert store.pressure_layer_pick(mapping, {}, [1], [], BUDGET, {}, [], lambda: called.append(1)) is None
        assert store.pressure_layer_pick(mapping, {0: [None, 0]}, [1.], [], BUDGET, {}, [],
                                         lambda: called.append(1)) is None
        assert not called
    finally:
        assert a.memory.close()


def test_flag_validation(monkeypatch):
    from vllm_kvmem.serving.layerwise_pressure import native_layer_pick
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LAYER_PICK', 'on')
    with pytest.raises(ValueError, match='NATIVE_LAYER_PICK'):
        native_layer_pick()
