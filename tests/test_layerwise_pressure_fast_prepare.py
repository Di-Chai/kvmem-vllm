"""KVMEM_LAYERWISE_PRESSURE_FAST_PREPARE: identical native-picker inputs and plans."""
import random
from types import SimpleNamespace as NS

import pytest

from test_layerwise_pressure import controller, native_authority, options


def plan(fast, seed, layers=16, extra=None):
    c = controller(); c.names = tuple(f'layer{i}' for i in range(layers)); c.fast_prepare = fast
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); hs = c._history('r', o)
    rows = list(a.records.values()); rng = random.Random(seed)
    for history in hs:
        # Exact ties, zero exposure (excluded score -> 0.0) and missing keys.
        history.values = {r.history_begin: [rng.choice([.01, .2, .2, 1.]), rng.choice([1., 1., 0.])]
                          for r in rows if rng.random() > .1}
        history.values[rows[0].history_begin] = [1., 1.]
    if extra:
        extra(hs, rows)
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    try:
        c.prepare(NS(options=o), job)
        chosen, evidence = job.layerwise_pressure_plan
        return [[r.history_begin for r in group] for group in chosen], evidence
    finally:
        assert a.memory.close()


@pytest.mark.parametrize('seed', [0, 1, 7, 20260722])
def test_fast_prepare_matches_original_selection_order_and_evidence(seed):
    assert plan(True, seed) == plan(False, seed)


@pytest.mark.parametrize('bad', ['nan', 'negative', 'all_zero'])
def test_fast_prepare_keeps_the_original_failures(bad):
    def corrupt(hs, rows):
        if bad == 'nan':
            hs[3].values[rows[5].history_begin] = [float('nan'), 1.]
        elif bad == 'negative':
            hs[3].values[rows[5].history_begin] = [-1., 1.]
        else:
            hs[3].values = {r.history_begin: [0., 1.] for r in rows}
    match = 'no accepted attention' if bad == 'all_zero' else 'Invalid layer cumulative attention'
    for fast in (False, True):
        with pytest.raises(RuntimeError, match=match):
            plan(fast, 3, extra=corrupt)
