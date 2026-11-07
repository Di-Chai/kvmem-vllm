"""Batched pressure selection is the original picker over one stable snapshot."""
import random
from types import SimpleNamespace as NS

import pytest

from test_layerwise_pressure import controller, native_authority, options, finish
from vllm_kvmem.memory import Phase


def require_batch(memory):
    if not hasattr(memory._store, 'pick_pressure_batch'):
        pytest.skip('The isolated batched pressure native extension is required')


def pick_batch(store, mandatory, layers, budget):
    return store.pick_pressure_batch(mandatory,
        [[slot for slot, _ in layer] for layer in layers],
        [[score for _, score in layer] for layer in layers], budget)


@pytest.mark.parametrize('budget', [96, 256])
@pytest.mark.parametrize('count', [233, 256])
@pytest.mark.parametrize('seed', [0, 17, 832])
def test_native_batch_reuses_original_order_ties_mandatory_and_cold_mask(budget, count, seed):
    authority = native_authority(count=count, budget=budget, sink=1)
    memory = authority.memory
    try:
        require_batch(memory)
        rng = random.Random(seed)
        slots = [r.handle.slot for r in authority.records.values()]
        mandatory = [slots[-1]]
        layers = []
        for _ in range(16):
            candidates = rng.sample(slots, 5)
            pairs = [(slot, rng.choice([0., -0., 1e-300, .25, .25, 1.])) for slot in candidates]
            pairs.append(pairs[0])  # Preserve native duplicate-caller semantics too.
            layers.append(pairs)
        expected = [memory._store.pick_pressure(mandatory, scores, budget) for scores in layers]
        before = tuple(memory._store.working_records())
        runtime_budget = memory._store.runtime_select_budget
        actual = pick_batch(memory._store, mandatory, layers, budget)
        assert actual == expected
        assert tuple(memory._store.working_records()) == before
        assert memory._store.runtime_select_budget == runtime_budget
        assert memory.phase is Phase.READY
        # A failed later layer must not install an earlier selection.
        with pytest.raises((ValueError, RuntimeError)):
            pick_batch(memory._store, mandatory, layers + [[(slots[0], float('nan'))]], budget)
        assert tuple(memory._store.working_records()) == before
        with pytest.raises(ValueError, match='at least one layer'):
            pick_batch(memory._store, mandatory, [], budget)
        for layer_slots, layer_scores in (([[slots[0]]], []), ([[slots[0]]], [[]])):
            with pytest.raises(ValueError, match='counts differ'):
                memory._store.pick_pressure_batch(mandatory, layer_slots, layer_scores, budget)
    finally:
        assert memory.close()


@pytest.mark.parametrize('buffered', [False, True])
@pytest.mark.parametrize('independent_maps', [False, True])
def test_controller_batch_preserves_each_selected_record_and_original_evidence(buffered, independent_maps):
    c = controller()
    c.names = tuple(f'layer{i}' for i in range(16))
    c.buffered_history = buffered
    authority = native_authority(count=256, budget=96, sink=1)
    authority.publish_sealed = lambda wait: None
    memory = authority.memory
    try:
        require_batch(memory)
        opts = options(memory.config)
        histories = c._history('r', opts)
        records = list(authority.records.values())
        if independent_maps:
            memory.select({}, pressure_scores=[(r.handle.slot, float(i in (1, 2)))
                                               for i, r in enumerate(records)], max_tokens=96)
            finish(memory)
            def mapping(indices):
                return tuple((records[i].handle.slot, records[i].handle.generation,
                              records[i].history_begin, 32, j*32) for j, i in enumerate(indices))
            c.maps['r'] = (256, tuple(mapping((0, 1, 2) if layer % 2 == 0 else (0, 5, 6))
                                     for layer in range(16)))
        rng = random.Random(4407)
        for history in histories:
            history.values = {r.history_begin: [rng.choice([.01, .25, .25, 1.]), 1.]
                              for r in records}
            history.end, history.samples, history.chunks = 256, 8, 1
            if buffered:
                history.successful_buffered_commits = 1
        authority.action_history = NS(prefill=histories[0])
        job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                           groups=((0, NS(window_tokens=96)),)), history=256, authority=authority)
        c.prepare(NS(options=opts), job)
        original, original_evidence = job.layerwise_pressure_plan
        c.batch_selection = True
        original_store = memory._store
        calls = []
        class StoreProxy:
            def __getattr__(self, name):
                return getattr(original_store, name)

            def pick_pressure_batch(self, *args):
                calls.append(args)
                return original_store.pick_pressure_batch(*args)

            def pick_pressure(self, *args):
                pytest.fail('The batch must not repeat a Python/native call per layer')
        memory._store = StoreProxy()
        c.prepare(NS(options=opts), job)
        assert len(calls) == 1
        candidate, evidence = job.layerwise_pressure_plan
        assert candidate == original
        assert all(a is b for left, right in zip(candidate, original) for a, b in zip(left, right))
        assert all(row.pop('pressure_selection_backend') == 'native_batch_snapshot' for row in evidence)
        assert evidence == original_evidence
        assert memory.phase is Phase.READY
        memory._store = original_store
    finally:
        assert memory.close()


@pytest.mark.parametrize('fault', ['phase', 'gc', 'scope', 'missing_picker', 'decode', 'layers'])
def test_batch_scope_fails_explicitly_before_native_ranking(fault):
    from vllm_kvmem.pressure_selection_batch import prepare_pressure_batch
    c = controller()
    calls = []
    store = NS(pick_pressure_batch=lambda *args: calls.append(args))
    memory = NS(phase=Phase.READY, _store=store)
    authority = NS(memory=memory)
    job = NS(command=NS(request_id='r', pressure_stage='prefill'), authority=authority)
    histories = [object(), object()]
    if fault == 'phase': memory.phase = Phase.APPENDING
    elif fault == 'gc': authority.gc = object()
    elif fault == 'scope': memory._history_scope = object()
    elif fault == 'missing_picker': del store.pick_pressure_batch
    elif fault == 'decode': job.command.pressure_stage = 'decode'
    else: histories.pop()
    with pytest.raises(ValueError):
        prepare_pressure_batch(c, job, histories, 0, {}, (), 96)
    assert not calls


def test_candidate_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION', raising=False)
    assert not controller().batch_selection
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION', '1')
    assert controller().batch_selection
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION', 'invalid')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        controller()
