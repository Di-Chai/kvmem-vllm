"""KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST: same per-layer selections and evidence."""
from types import SimpleNamespace as NS

import pytest


def run(monkeypatch, lazy):
    from test_layerwise_pressure import controller, options, native_authority, finish
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST', '1' if lazy else '0')
    c = controller(); c.prepared_commit = True; c.deferred_maps = True
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
    if lazy:
        job.promised_lease = object()
    emitted = []; c.query.emit = lambda event, **fields: emitted.append((event, fields))
    try:
        c.prepare(NS(options=o), job)
        chosen, evidence = job.layerwise_pressure_plan
        assert len(chosen) == (1 if lazy else 2)
        slots = {r.handle.slot for r in chosen[0]}
        # Layer-0 plan and host-map commit, as the producer does before uploads.
        a.memory.select({}, pressure_scores=[(r.handle.slot, float(r.handle.slot in slots)) for r in rows],
                        max_tokens=96)
        finish(a.memory)
        job.selected_tokens = 96
        job.layerwise_single_assembly_done = True
        c.committed(job)
        assert bool(emitted) == (not lazy)
        if lazy:
            with pytest.raises(RuntimeError, match='layer order'):
                job.layerwise_select_layer(2)
            job.layerwise_select_layer(1)       # After the committed layer-0 host map.
            assert len(emitted) == 1
            with pytest.raises(RuntimeError, match='layer order'):
                job.layerwise_select_layer(1)
        job.layerwise_prepared_commit.value()
        return ([[r.history_begin for r in layer] for layer in chosen], emitted[-1],
                c.resolved_maps('r'))
    finally:
        assert a.memory.close()


def test_layer_zero_first_matches_eager_selection_maps_and_evidence(monkeypatch):
    eager = run(monkeypatch, False)
    lazy = run(monkeypatch, True)
    assert lazy[0] == eager[0] and lazy[1] == eager[1] and lazy[2] == eager[2]
    assert eager[0][0] != eager[0][1]          # The two layers really differ.


def test_layer_zero_first_requires_a_promised_job(monkeypatch):
    from test_layerwise_pressure import controller, options, native_authority
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST', '1')
    c = controller(); c.prepared_commit = True; c.deferred_maps = True
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    o = options(a.memory.config); hs = c._history('r', o)
    for history in hs:
        history.values = {r.history_begin: [1., 1.] for r in a.records.values()}
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    c.query.emit = lambda *a, **k: None
    try:
        c.prepare(NS(options=o), job)
        assert len(job.layerwise_pressure_plan[0]) == 2 and not hasattr(job, 'layerwise_select_layer')
    finally:
        assert a.memory.close()
