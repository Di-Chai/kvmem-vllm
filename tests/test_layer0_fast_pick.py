"""KVMEM_LAYER0_FAST: the position-space pressure pick equals the store's pick.

The layer-0 fast path picks before the chunk's records have slots. Its pick
must equal KvMemStore::pick_prefill_pressure_blocks on the real store for every
input the pressure selection can supply: sink prefix, mandatory blocks, ties,
zero and positive scores, and candidates that are a subset of the history.
"""
import random

import pytest

from vllm_kvmem import _native

pytestmark = pytest.mark.skipif(not hasattr(_native, 'pressure_pick_positions'),
                                reason='Requires the rebuilt native extension')

BLOCK = 32


def store_fixture(blocks, sink):
    from vllm_kvmem.core.config import KVMemConfig
    from vllm_kvmem.core.memory import RequestMemory
    from test_core import append
    count = blocks * BLOCK
    memory = RequestMemory('pick', KVMemConfig(block_tokens=BLOCK, select_budget=count, gen_budget=256,
        repository_blocks=blocks + 8, sink_blocks=sink, recent_blocks=0, action_fraction=0))
    memory._test_spans = append(memory, count)
    return memory


def store_pick(memory, records, mandatory, candidates, scores, budget):
    slot_of = {r[2]: r[0] for r in records}
    begin_of = {r[0]: r[2] for r in records}
    picked = memory._store.pick_pressure([slot_of[b] for b in mandatory],
                                         [(slot_of[b], s) for b, s in zip(candidates, scores)], budget)
    return sorted(begin_of[slot] for slot in picked)


def position_pick(memory, mandatory, candidates, scores, budget):
    config = memory.config
    return _native.pressure_pick_positions(memory.history_end, BLOCK, config.sink_blocks,
        config.select_budget // BLOCK, list(mandatory), list(candidates), list(scores), budget)


@pytest.mark.parametrize('seed', range(12))
def test_position_pick_equals_store_pick(seed):
    rnd = random.Random(seed)
    blocks = rnd.choice([40, 96, 160])
    sink = rnd.choice([0, 1, 3, 8])
    memory = store_fixture(blocks, sink)
    try:
        records = list(memory._store.working_records())
        begins = [r[2] for r in records]
        assert begins == [i * BLOCK for i in range(blocks)]
        for _ in range(20):
            budget = BLOCK * rnd.randint(max(sink, 1) + 2, blocks - 1)
            candidates = sorted(rnd.sample(begins, rnd.randint(1, blocks)))
            if rnd.random() < .5:
                candidates = list(reversed(candidates))
            if rnd.random() < .3:
                rnd.shuffle(candidates)
            palette = [0., .25, .5, 1., 2.] if rnd.random() < .5 else None
            scores = [rnd.choice(palette) if palette else rnd.random() for _ in candidates]
            mandatory = sorted(rnd.sample(begins, rnd.randint(0, 3)))
            expected = store_pick(memory, records, mandatory, candidates, scores, budget)
            assert position_pick(memory, mandatory, candidates, scores, budget) == expected
    finally:
        memory.close()


def test_whole_history_fits_and_invalid_inputs():
    memory = store_fixture(40, 2)
    try:
        records = list(memory._store.working_records())
        begins = [r[2] for r in records]
        candidates, scores = begins[5:20], [1.] * 15
        budget = 40 * BLOCK
        assert (position_pick(memory, [], candidates, scores, budget) ==
                store_pick(memory, records, [], candidates, scores, budget))
        with pytest.raises(ValueError, match='invalid pressure action score'):
            position_pick(memory, [], [0], [float('nan')], 10 * BLOCK)
        with pytest.raises(ValueError, match='budget'):
            position_pick(memory, [], [0], [1.], 10 * BLOCK + 1)
        with pytest.raises(RuntimeError, match='outside the pressure snapshot'):
            position_pick(memory, [41 * BLOCK], [0], [1.], 10 * BLOCK)
    finally:
        memory.close()


def test_duplicate_candidates_keep_the_store_order():
    memory = store_fixture(40, 1)
    try:
        records = list(memory._store.working_records())
        candidates = [64, 96, 64, 128, 96]
        scores = [1., .5, 1., 0., .5]
        budget = 3 * BLOCK
        assert (position_pick(memory, [], candidates, scores, budget) ==
                store_pick(memory, records, [], candidates, scores, budget))
    finally:
        memory.close()


@pytest.mark.parametrize('kind', ['random', 'ties', 'front'])
def test_plan_layer0_matches_cpu_selection_and_store_plan_skips(monkeypatch, kind):
    """Same window as the lazy CPU selection; same skipped blocks as the native plan."""
    from types import SimpleNamespace as NS
    from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
    from vllm_kvmem.serving.layerwise_pressure import native_lookup_scores
    from test_layerwise_pressure import controller, options
    from test_recall_experiment import native_authority, finish
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '1')
    monkeypatch.setenv('KVMEM_PRESSURE_RECORD_INDEX_CACHE', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST', '1')
    count, budget = 512, 256
    c = controller(); c.prepared_commit = True; c.deferred_maps = True; c.fast_prepare = True
    a = native_authority(count=count, budget=budget, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); o.native_prefill_commit = True
    c._history('r', o)
    rows = list(a.records.values())
    rnd = random.Random(kind)
    hs = []
    for layer in range(len(c.names)):
        h = BufferedPrefillHistory(o)
        if kind == 'random':
            values = {r.history_begin: [rnd.random(), 1.] for r in rows}
        elif kind == 'ties':
            values = {r.history_begin: [rnd.choice([0., .5, 1.]), 1.] for r in rows}
        else:   # high scores at the front keep many blocks at their dense position
            values = {r.history_begin: [1. / (1 + j), 1.] for j, r in enumerate(rows)}
        h.values = values
        h.end, h.samples, h.chunks = count, 4, 1
        hs.append(h)
    c.histories['r'] = hs
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=budget)),)), history=count, authority=a,
             promised_lease=object())
    c.query.emit = lambda *x, **k: None
    try:
        working = list(a.memory._store.working_records())
        assert [r[4] for r in working] == [BLOCK * i for i in range(len(working))]
        window = [r[2] for r in working]
        scores = native_lookup_scores(hs[0], working)
        c.prepare(NS(options=o), job)
        chosen, _ = job.layerwise_pressure_plan
        # This first frame holds the whole history (16 blocks for an 8-block
        # budget), so apply plan_layer0's position pick and skip rule directly;
        # the steady-state shape (window == budget) is covered below.
        picked = _native.pressure_pick_positions(count, BLOCK, a.memory.config.sink_blocks,
            a.memory.config.select_budget // BLOCK, [], window, list(scores), budget)
        position = {b: i for i, b in enumerate(window)}
        skip = frozenset(i for i, b in enumerate(picked) if position[b] == i)
        assert picked == [r.history_begin for r in chosen[0]]
        slots = {r.handle.slot for r in chosen[0]}
        plan = a.memory.select({}, pressure_scores=[(r.handle.slot, float(r.handle.slot in slots)) for r in rows],
                               max_tokens=budget)
        by_slot = {r.handle.slot: r.history_begin for r in rows}
        assert {picked[i] for i in skip} == {by_slot[r.block_id] for r in plan.remaps if r.skip}
        assert [by_slot[r.block_id] for r in plan.remaps] == picked
        finish(a.memory)
    finally:
        assert a.memory.close()


def test_plan_layer0_sources_and_skips_with_a_staged_chunk():
    """Window rows keep their row when their dense position is unchanged; chunk rows follow."""
    from vllm_kvmem.serving.layer0_fast import plan_layer0, _Decline
    window = [0, 32, 64, 128, 192, 256]                    # previous window, 6 blocks
    chunk_begin, chunk_blocks = 320, 4                      # staged chunk: 320..447
    mapping = [(0, 0, b, 32, 32 * i) for i, b in enumerate(window)] + \
              [(-1, -1, chunk_begin + 32 * c, 32, 32 * (6 + c)) for c in range(chunk_blocks)]
    scores = [9., 1., 0., 5., 0., 3., 8., 0., 7., 0.]
    picked, sources, skip = plan_layer0(448, mapping, scores, window, chunk_begin, chunk_blocks,
                                        [], 1, 14, 6 * BLOCK)
    expected = _native.pressure_pick_positions(448, BLOCK, 1, 14, [], [r[2] for r in mapping], scores, 6 * BLOCK)
    assert picked == expected
    position = {b: i for i, b in enumerate(window)}
    for i, (b, j) in enumerate(zip(picked, sources)):
        assert j == (position[b] if b in position else len(window) + (b - chunk_begin) // BLOCK)
        assert (i in skip) == (b in position and position[b] == i)
    with pytest.raises(_Decline):
        plan_layer0(448, mapping, scores, window, chunk_begin, chunk_blocks, [], 1, 14, 5 * BLOCK)
