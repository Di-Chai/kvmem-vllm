"""Opt-in KVMEM_LAYERWISE_SESSIONS: per-layer state of multi-turn session checkpoints."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import layerwise_checkpoints as lc
from vllm_kvmem.layerwise_checkpoints import (
    capture, capture_for_pool, finish_async_restore, install_restored, prepare_restore,
    refresh_committed_snapshot, require_supported_restore, restore_pressure_sync,
    restored_decode, sessions_enabled,
)
from vllm_kvmem.layerwise_decode import DecodeHistory
from test_layerwise_checkpoints import prefix
from test_layerwise_pressure import options


@pytest.fixture(autouse=True)
def _no_gc(monkeypatch):
    # GC holds stay inert unless a test opts in explicitly.
    monkeypatch.delenv('KVMEM_LAYERWISE_HISTORICAL_GC', raising=False)
    monkeypatch.delenv('KVMEM_LAYERWISE_SESSIONS', raising=False)


def sessions(monkeypatch, on=True):
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1' if on else '0')


def with_decode(c, rid='old', end=280):
    c.decode_histories[rid] = [DecodeHistory({64: 0.5, 0: 0.25}, 4, end, 2),
                               DecodeHistory({128: 0.75}, 3, end, 1)]
    return c


def emitted(c):
    events = []
    c.query.emit = lambda event, **fields: events.append((event, fields))
    return events


# 1. flag
def test_sessions_flag_defaults_off_and_is_validated(monkeypatch):
    assert sessions_enabled() is False
    sessions(monkeypatch)
    assert sessions_enabled() is True
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', 'yes')
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_SESSIONS'):
        sessions_enabled()


# 2. capture
def test_terminal_capture_requires_the_flag(monkeypatch):
    c, a = prefix()
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_SESSIONS'):
        capture(c, 'old', a, 288, terminal=True)
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_SESSIONS'):
        capture(c, 'old', a, 288, committed=True)


def test_terminal_capture_after_the_query_snapshots_sorted_decode_histories(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    c.query.finished.add('old')                      # after the final Query: allowed for M
    snapshot = capture(c, 'old', a, 288, terminal=True)
    assert snapshot.terminal is True
    assert snapshot.decode == ((((0, 0.25), (64, 0.5)), 4, 280, 2), (((128, 0.75),), 3, 280, 1))
    # Snapshot rows are independent of later mutation of the live history.
    c.decode_histories['old'][0].mass[0] = 99.
    assert snapshot.decode[0][0][0] == (0, 0.25)
    assert snapshot.bytes > lc.LayerwisePrefix(288, snapshot.names, snapshot.maps, snapshot.histories).bytes


def test_p_capture_still_refuses_after_the_query_but_committed_skips_it(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); c.query.finished.add('old')
    with pytest.raises(ValueError, match='precede the final Query'):
        capture(c, 'old', a, 288)
    snapshot = capture(c, 'old', a, 288, committed=True)
    assert snapshot.terminal is False and snapshot.position == 288


@pytest.mark.parametrize('fault', ['end', 'key'])
def test_future_decode_evidence_is_refused(monkeypatch, fault):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c, end=320 if fault == 'end' else 280)
    if fault == 'key':
        c.decode_histories['old'][1].mass[288] = 0.1
    with pytest.raises(ValueError, match='future decode attention'):
        capture(c, 'old', a, 288, terminal=True)


def test_frozen_p_capture_cannot_carry_decode_attention_without_the_flag():
    c, a = prefix(); with_decode(c)
    with pytest.raises(ValueError, match='cannot carry decode attention'):
        capture(c, 'old', a, 288)


def test_decode_layer_count_must_match(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c); c.decode_histories['old'].pop()
    with pytest.raises(ValueError, match='rolling decode histories'):
        capture(c, 'old', a, 288, terminal=True)


# 3. capture_for_pool
def adapter(c, frozen):
    return NS(runner=NS(_kvmem_layerwise_pressure=c,
                        _kvmem_sessions=NS(options=NS(session_freeze_prompt=frozen))))


@pytest.mark.parametrize('terminal,frozen', [(True, True), (False, False), (True, False)])
def test_pool_capture_without_the_flag_keeps_frozen_p_only(terminal, frozen):
    c, a = prefix()
    with pytest.raises(ValueError, match='frozen pre-Query P'):
        capture_for_pool(adapter(c, frozen), 'old', a, 288, terminal=terminal)
    assert capture_for_pool(adapter(c, True), 'old', a, 288).terminal is False


def test_pool_capture_with_the_flag_passes_terminal_through(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    snapshot = capture_for_pool(adapter(c, False), 'old', a, 288, terminal=True)
    assert snapshot.terminal is True and snapshot.decode is not None
    assert capture_for_pool(NS(runner=NS()), 'old', a, 288, terminal=True) is None


# 4. restored_decode / install_restored
def test_restored_decode_rebuilds_independent_decode_histories(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    decode = restored_decode(snapshot)
    assert [(d.mass, d.samples, d.end, d.chunks) for d in decode] == [
        ({0: 0.25, 64: 0.5}, 4, 280, 2), ({128: 0.75}, 3, 280, 1)]
    decode[0].mass[0] = 7.
    assert restored_decode(snapshot)[0].mass[0] == 0.25
    assert restored_decode(lc.LayerwisePrefix(288, ('a', 'b'), (), ())) is None
    with pytest.raises(ValueError, match='decode history count'):
        restored_decode(lc.LayerwisePrefix(288, ('a', 'b'), (), (), decode=snapshot.decode[:1]))


def test_install_restored_publishes_maps_histories_and_decode(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    install_restored(c, 'fresh', a, frame, histories, snapshot)
    assert c.maps['fresh'] == frame and c.histories['fresh'] is histories
    assert [d.mass for d in c.decode_histories['fresh']] == [{0: 0.25, 64: 0.5}, {128: 0.75}]
    # Without KVMEM_LAYERWISE_HISTORICAL_GC no hold is taken.
    from vllm_kvmem.layerwise_gc_holds import holds_of
    assert holds_of(c).holds == {}


def test_install_restored_without_decode_leaves_decode_histories_alone(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix()
    snapshot = capture(c, 'old', a, 288, terminal=True)
    assert snapshot.decode is None
    c.maps.pop('old'); c.histories.pop('old')
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    install_restored(c, 'fresh', a, frame, histories, snapshot)
    assert 'fresh' not in c.decode_histories


# 5. restore_pressure_sync
def pressure_setup(monkeypatch, *, flag=True):
    sessions(monkeypatch, flag)
    c, a = prefix(); with_decode(c)
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1')
    snapshot = capture(c, 'old', a, 288, terminal=True)
    sessions(monkeypatch, flag)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c), options=options(),
                bindings={'fresh': NS(pressure_restore=True)})
    checkpoint = NS(kind='M', authority=a, layerwise_pressure=snapshot, position=288)
    return c, a, snapshot, worker, checkpoint


def test_compacting_restore_installs_histories_and_decode_without_per_layer_maps(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    events = emitted(c)
    restore_pressure_sync(worker, 'fresh', checkpoint)
    assert 'fresh' not in c.maps                      # every layer resumes from the shared frame
    assert c.histories['fresh'][1].values == {0: [0.1, 1.], 128: [0.9, 1.]}
    assert [d.mass for d in c.decode_histories['fresh']] == [{0: 0.25, 64: 0.5}, {128: 0.75}]
    (event, fields), = events
    assert event == 'pressure_prefix_restored'
    assert fields['actual_method'] == 'shared_compacted_frame_per_layer_histories'
    assert fields['accepted_history'] == 288 and fields['selected_tokens'] == 96


def test_compacting_restore_requires_the_flag(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch, flag=False)
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_SESSIONS'):
        restore_pressure_sync(worker, 'fresh', checkpoint)
    assert 'fresh' not in c.histories


@pytest.mark.parametrize('owner', ['maps', 'histories', 'finished'])
def test_compacting_restore_refuses_a_non_fresh_owner(monkeypatch, owner):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    if owner == 'maps': c.maps['fresh'] = (288, ())
    elif owner == 'histories': c.histories['fresh'] = []
    else: c.query.finished.add('fresh')
    with pytest.raises(ValueError, match='fresh native request owner'):
        restore_pressure_sync(worker, 'fresh', checkpoint)
    assert 'fresh' not in c.decode_histories


def test_compacting_restore_refuses_an_unrewound_authority(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    a.history_end = 320
    with pytest.raises(ValueError, match='rewind'):
        restore_pressure_sync(worker, 'fresh', checkpoint)


# 6. finish_async_restore
def test_async_finish_routes_compacting_imports_to_the_shared_frame_path(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    calls = []
    monkeypatch.setattr(lc, 'restore_pressure_sync', lambda *args: calls.append(('pressure', args[1])))
    monkeypatch.setattr(lc, 'restore_sync', lambda *args: calls.append(('copy', args[1])))
    finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=True)
    assert calls == [('pressure', 'fresh')]


def test_async_finish_resident_import_installs_cpu_state_only(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    worker.bindings['fresh'].pressure_restore = False
    def assemble(*args, **kwargs):
        raise AssertionError('a resident import must not rewrite any layer page')
    c.query.assemble_layer_pinned = assemble
    events = emitted(c)
    finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=False)
    assert c.maps['fresh'] == (288, snapshot.maps)
    assert c.histories['fresh'][1].values == {0: [0.1, 1.], 128: [0.9, 1.]}
    assert [d.mass for d in c.decode_histories['fresh']] == [{0: 0.25, 64: 0.5}, {128: 0.75}]
    (event, fields), = events
    assert fields['actual_method'] == 'resident_terminal_pages_per_layer'
    assert fields['map_sha256'] == snapshot.evidence()['map_sha256']


def test_async_finish_resident_import_requires_the_flag(monkeypatch):
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch, flag=False)
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_SESSIONS'):
        finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=False)


def test_async_finish_copy_import_rebuilds_target_layers_like_restore_sync(monkeypatch):
    import torch
    c, a, snapshot, worker, checkpoint = pressure_setup(monkeypatch)
    worker.bindings['fresh'].pressure_restore = False
    worker.runner.device = 'cpu'
    worker.options.session_freeze_prompt = False      # a non-frozen session
    copies, events = [], emitted(c)
    c.query.assemble_layer_pinned = lambda state, layer, records: copies.append(
        (layer, [r.history_begin for r in records]))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    finish_async_restore(worker, 'fresh', checkpoint, resident=False, pressure=False)
    assert copies == [(1, [0, 128, 256])]              # layers 1..N only
    assert c.maps['fresh'] == (288, snapshot.maps)
    assert [d.mass for d in c.decode_histories['fresh']] == [{0: 0.25, 64: 0.5}, {128: 0.75}]
    assert events[-1][1]['actual_method'] == 'accumulated_attention_per_layer'


def test_async_finish_without_layerwise_state_is_a_no_op():
    worker = NS(runner=NS())
    assert finish_async_restore(worker, 'fresh', NS(), resident=False, pressure=True) is None


# 7. require_supported_restore
def restore_worker(c, *, frozen=False, pressure=False):
    return NS(runner=NS(_kvmem_layerwise_pressure=c), options=NS(session_freeze_prompt=frozen),
              bindings={'fresh': NS(pressure_restore=pressure)})


@pytest.mark.parametrize('kind', ['P', 'M'])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_session_restores_of_p_and_m_are_supported_with_the_flag(monkeypatch, kind, asynchronous):
    sessions(monkeypatch)
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    checkpoint = NS(kind=kind, layerwise_pressure=snapshot)
    assert require_supported_restore(restore_worker(c, pressure=True), 'fresh', checkpoint,
                                     asynchronous=asynchronous) is None


def test_unknown_checkpoint_kind_is_refused_with_the_flag(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    with pytest.raises(ValueError, match='Unknown layerwise checkpoint kind'):
        require_supported_restore(restore_worker(c), 'fresh', NS(kind='X', layerwise_pressure=snapshot))


@pytest.mark.parametrize('fault', ['async', 'terminal', 'mutable', 'pressure'])
def test_without_the_flag_only_synchronous_frozen_p_restores(fault):
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    worker = restore_worker(c, frozen=fault != 'mutable', pressure=fault == 'pressure')
    checkpoint = NS(kind='M' if fault == 'terminal' else 'P', layerwise_pressure=snapshot)
    with pytest.raises(ValueError, match='synchronous frozen P'):
        require_supported_restore(worker, 'fresh', checkpoint, asynchronous=fault == 'async')
    assert require_supported_restore(restore_worker(c, frozen=True), 'fresh',
                                     NS(kind='P', layerwise_pressure=snapshot)) is None


# 8. refresh_committed_snapshot
def test_committed_selection_retakes_the_live_p_snapshot(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix()
    before = capture(c, 'old', a, 288)
    c.query.finished.add('old')                       # the commit happened after the Query
    # Layer 0 stays the native frame; layer 1 adopts a different committed window.
    c.maps['old'] = (256, (((0, 1, 0, 32, 0), (1, 1, 64, 32, 32)),
                           ((0, 1, 0, 32, 0), (1, 1, 64, 32, 32))))
    checkpoint = NS(layerwise_pressure=before, authority=a, position=288)
    refresh_committed_snapshot(NS(runner=NS(_kvmem_layerwise_pressure=c)), 'old', checkpoint)
    after = checkpoint.layerwise_pressure
    assert after is not before and after.position == 288 and after.terminal is False
    assert [row[2] for row in after.maps[1]] == [0, 64, 256]
    assert [row[2] for row in before.maps[1]] == [0, 128, 256]


def test_committed_snapshot_refresh_requires_the_flag():
    c, a = prefix()
    checkpoint = NS(layerwise_pressure=capture(c, 'old', a, 288), authority=a, position=288)
    with pytest.raises(ValueError, match='cannot adopt a later shared window'):
        refresh_committed_snapshot(NS(runner=NS(_kvmem_layerwise_pressure=c)), 'old', checkpoint)


def test_committed_snapshot_refresh_ignores_checkpoints_without_layer_state():
    c, a = prefix()
    checkpoint = NS(layerwise_pressure=None, authority=a, position=288)
    assert refresh_committed_snapshot(NS(runner=NS(_kvmem_layerwise_pressure=c)), 'old', checkpoint) is None
    assert refresh_committed_snapshot(NS(runner=NS()), 'old', NS(layerwise_pressure=object())) is None


# 9. finished request -> stash -> terminal capture
def test_finished_state_is_stashed_for_the_terminal_capture_then_consumed(monkeypatch):
    from vllm_kvmem.layerwise_gc_holds import holds_of
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    holds = holds_of(c)
    holds.holds['old'] = object()                       # the live window hold
    lc.retire_state(c, 'old')
    assert 'old' not in c.histories and 'old' not in c.decode_histories and 'old' in holds.retired
    assert 'old' in holds.holds                         # kept until the capture
    snapshot = capture_for_pool(adapter(c, False), 'old', a, 288, terminal=True)
    assert snapshot.terminal and snapshot.decode is not None
    assert 'old' not in holds.retired and 'old' not in holds.holds
    assert 'old' not in c.histories and 'old' not in c.maps   # live tables stay clean


def test_stash_is_bounded_and_releases_the_oldest_hold(monkeypatch):
    from vllm_kvmem.layerwise_gc_holds import holds_of
    sessions(monkeypatch)
    monkeypatch.setattr(lc, 'MAX_RETIRED', 2)
    c, a = prefix()
    holds = holds_of(c)
    for rid in ('a', 'b', 'c'):
        c.histories[rid] = list(c.histories['old'])
        holds.holds[rid] = object()
        lc.retire_state(c, rid)
    assert list(holds.retired) == ['b', 'c'] and 'a' not in holds.holds


def test_finish_drops_the_record_index_cached_under_the_session_namespace(monkeypatch):
    sessions(monkeypatch)
    c, a = prefix()
    cache = c.__dict__.setdefault('_record_index_cache', {})
    cache['first-turn'] = ('records of the whole session',)
    authority = NS(memory=NS(request_id='first-turn'))
    lc.retire_state(c, 'old', authority)              # a later turn of the same session finishes
    assert 'first-turn' not in cache


def test_retired_blocks_are_pruned_from_every_layer_like_the_shared_history(monkeypatch):
    from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure
    from vllm_kvmem.layerwise_decode import DecodeHistory
    sessions(monkeypatch)
    c, a = prefix()
    pressure = LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    pressure.histories = {'old': c.histories['old']}
    pressure.decode_histories = {'old': [DecodeHistory({0: .5, 64: .25}, 2, 280, 1) for _ in c.histories['old']]}
    authority = NS(history_end=288)
    pressure.bind_history_owner('old', authority)
    assert authority.history_pruners == [pressure]
    before = [dict(h.values) for h in pressure.histories['old']]
    live = {k for k in before[0] if k != 0}                  # block 0 retired
    pressure.prune_layer_histories(authority, live)
    for old, history in zip(before, pressure.histories['old']):
        assert history.values == {k: v for k, v in old.items() if k in live or k >= 288}
    assert all(0 not in h.mass and 64 in h.mass for h in pressure.decode_histories['old'])
    pressure.prune_layer_histories(NS(history_end=288), set())   # another authority: untouched
    assert pressure.histories['old'][0].values == {k: v for k, v in before[0].items() if k in live}


# Probe-mode final Query: layer zero selects with its own Query, the native
# frame keeps the global selection until the next pressure realigns them.
def diverged_prefix():
    c, a = prefix()
    origin, layers = c.maps['old']
    c.maps['old'] = (origin, (((0, 1, 0, 32, 0), (2, 1, 128, 32, 32)), layers[1]))
    return c, a


def test_layer_zero_divergence_requires_equal_length_and_partial_geometry():
    shared = ((0, 1, 0, 32, 0), (1, 1, 64, 32, 32), (3, 1, 256, 16, 64))
    assert lc.layer0_diverged(shared, shared) is False
    assert lc.layer0_diverged(shared, ((0, 1, 0, 32, 0), (2, 1, 128, 32, 32), (3, 1, 256, 16, 64))) is True
    with pytest.raises(ValueError, match='native layer-zero frame'):
        lc.layer0_diverged(shared, ((0, 1, 0, 32, 0), (3, 1, 256, 16, 32)))
    with pytest.raises(ValueError, match='native layer-zero frame'):
        lc.layer0_diverged(shared, ((0, 1, 0, 32, 0), (1, 1, 64, 16, 32), (2, 1, 128, 32, 48)))


def test_diverged_layer_zero_is_captured_only_with_the_flag(monkeypatch):
    c, a = diverged_prefix()
    with pytest.raises(ValueError, match='native layer-zero frame'):
        capture(c, 'old', a, 288)
    sessions(monkeypatch)
    c, a = diverged_prefix(); with_decode(c); c.query.finished.add('old')
    snapshot = capture(c, 'old', a, 288, terminal=True)
    assert [row[2] for row in snapshot.maps[0]] == [0, 128, 256]


def test_copy_restore_rebuilds_a_diverged_layer_zero_too(monkeypatch):
    import torch
    sessions(monkeypatch)
    c, a = diverged_prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c, device='cpu'), options=options(),
                bindings={'fresh': NS(pressure_restore=False)})
    worker.options.session_freeze_prompt = False
    checkpoint = NS(kind='M', authority=a, layerwise_pressure=snapshot, position=288)
    copies = []
    c.query.assemble_layer_pinned = lambda state, layer, records: copies.append(
        (layer, [r.history_begin for r in records]))
    c.query.emit = lambda *args, **kwargs: None
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    finish_async_restore(worker, 'fresh', checkpoint, resident=False, pressure=False)
    # The native import wrote the shared frame (0, 64, 256); layer zero is (0, 128, 256).
    assert copies == [(0, [0, 128, 256]), (1, [0, 128, 256])]
    assert c.maps['fresh'] == (288, snapshot.maps)


@pytest.mark.parametrize('flag,single,layer0,expected', [
    (True, True, (0, 3, 4), True),      # diverged session state: full layer-zero rebuild
    (True, True, (0, 1, 2), False),     # aligned: native skip hints still apply
    (False, False, (0, 3, 4), False),   # flag off: original behavior, no check
    (True, False, (0, 3, 4), 'error'),  # needs the single per-layer assembly
])
def test_prefill_pressure_marks_a_diverged_layer_zero(monkeypatch, flag, single, layer0, expected):
    from test_layerwise_pressure import controller, native_authority, finish
    sessions(monkeypatch, flag)
    c = controller(); c.single_assembly = single
    a = native_authority(count=256, budget=96, sink=1)
    a.publish_sealed = lambda wait: None
    rows = list(a.records.values()); o = options(a.memory.config)
    hs = c._history('r', o)
    a.memory.select({}, pressure_scores=[(r.handle.slot, 1. if i in (1, 2) else 0.)
                                         for i, r in enumerate(rows)], max_tokens=96)
    finish(a.memory)
    mapping = lambda indices: tuple((rows[i].handle.slot, rows[i].handle.generation,
                                     rows[i].history_begin, 32, j * 32) for j, i in enumerate(indices))
    c.maps['r'] = (256, (mapping(layer0), mapping((0, 5, 6))))
    hs[0].values = {r.history_begin: [.5, 1.] for r in rows}
    hs[1].values = {0: [.1, 1.], 160: [.5, 1.], 192: [.4, 1.]}
    a.action_history = NS(prefill=hs[0])
    job = NS(command=NS(request_id='r', pressure_stage='prefill', mandatory_spans=(),
                        groups=((0, NS(window_tokens=96)),)), history=256, authority=a)
    try:
        if expected == 'error':
            with pytest.raises(ValueError, match='single pressure assembly'):
                c.prepare(NS(options=o), job)
        else:
            c.prepare(NS(options=o), job)
            assert job.layerwise_layer0_diverged is expected
    finally:
        assert a.memory.close()


def test_restore_oracle_accepts_a_diverged_layer_zero_and_the_draft_follows_it(monkeypatch):
    sessions(monkeypatch)
    c, a = diverged_prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps['fresh'] = (288, snapshot.maps)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c), runtime=NS(names=('a', 'b', 'draft')))
    # A diverged restore rebuilds the draft from layer zero, so all three agree.
    assert lc.oracle_record_groups(worker, 'fresh', a) == ((snapshot.maps[0], (0, 1, 2)),)


def draft_controller(c, calls):
    c.runner = NS(_kvmem_archive_layers=tuple(c.names) + ('mtp',), device='cpu')
    c.query.assemble_layer_pinned = lambda state, layer, records, **kw: calls.append(
        (layer, [r.history_begin for r in records], kw.get('layer_name')))
    c.query.emit = lambda *args, **kwargs: None
    return c


def test_diverged_copy_restore_rebuilds_layer_zero_and_the_draft(monkeypatch):
    import torch
    sessions(monkeypatch)
    c, a = diverged_prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    calls = []; draft_controller(c, calls)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c, device='cpu'), options=options(),
                bindings={'fresh': NS(pressure_restore=False)})
    worker.options.session_freeze_prompt = False
    checkpoint = NS(kind='M', authority=a, layerwise_pressure=snapshot, position=288)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    finish_async_restore(worker, 'fresh', checkpoint, resident=False, pressure=False)
    assert calls == [(0, [0, 128, 256], None), (1, [0, 128, 256], None), (2, [0, 128, 256], 'mtp')]


@pytest.mark.parametrize('diverged', [True, False])
def test_resident_restore_rebuilds_only_a_diverged_draft(monkeypatch, diverged):
    import torch
    sessions(monkeypatch)
    c, a = diverged_prefix() if diverged else prefix(); with_decode(c)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    calls = []; draft_controller(c, calls)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c, device='cpu'), options=options(),
                bindings={'fresh': NS(pressure_restore=False)})
    checkpoint = NS(kind='M', authority=a, layerwise_pressure=snapshot, position=288)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=lambda: None))
    finish_async_restore(worker, 'fresh', checkpoint, resident=True, pressure=False)
    assert calls == ([(2, [0, 128, 256], 'mtp')] if diverged else [])
    assert c.maps['fresh'] == (288, snapshot.maps)


@pytest.mark.parametrize('case', ['diverged', 'aligned', 'flag_off', 'pressure'])
def test_realign_commits_layer_zero_as_the_native_frame_after_publication(monkeypatch, case):
    from vllm_kvmem import layerwise_no_probe
    sessions(monkeypatch, case != 'flag_off')
    c, a = diverged_prefix() if case != 'aligned' else prefix()
    commits, committed = [], []
    monkeypatch.setattr(layerwise_no_probe, 'commit_layer_zero', lambda controller, state: commits.append(
        ([r.history_begin for r in state['layer_zero_records']], state['selected_tokens'])))
    c.query.emit = lambda *args, **kwargs: None
    origin, layers = c.maps['old']
    from vllm_kvmem.layerwise_pressure import extend_records
    c.maps['fresh'] = (288, tuple(tuple(extend_records(rows, origin, 288)) for rows in layers))
    # Resolve the extension rows to real records as a restore installs them.
    c.maps['fresh'] = (288, tuple(tuple((a.records[next(k for k, r in a.records.items() if r.history_begin == row[2])].handle.slot, 1,
                                          row[2], row[3], row[4]) for row in rows) for rows in c.maps['fresh'][1]))
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c),
                bindings={'fresh': NS(pressure_restore=case == 'pressure')},
                selection_committed=lambda rid, history: committed.append((rid, history)))
    checkpoint = NS(authority=a, layerwise_pressure=object(), position=288)
    assert lc.realign_after_restore(worker, 'fresh', checkpoint) is (case == 'diverged')
    assert commits == ([([0, 128, 256], 96)] if case == 'diverged' else [])
    assert committed == ([('fresh', 288)] if case == 'diverged' else [])


@pytest.mark.parametrize('flag', [True, False])
def test_query_commit_drops_stale_layer_maps_only_with_the_flag(monkeypatch, flag):
    from vllm_kvmem.layerwise_pressure import PendingMaps
    sessions(monkeypatch, flag)
    c, a = prefix()
    discarded = []
    c.maps['p'] = PendingMaps(NS(discard=lambda: discarded.append(True)), NS(history=256))
    runner = NS(_kvmem_layerwise_pressure=c)
    assert lc.query_frame_committed(runner, 'old') is flag
    assert lc.query_frame_committed(runner, 'p') is flag
    assert ('old' in c.maps, 'p' in c.maps, discarded) == ((False, False, [True]) if flag else (True, True, []))
    assert lc.query_frame_committed(runner, 'absent') is False
    assert lc.query_frame_committed(NS(), 'old') is False


# Historical GC retires a block while a session restore is in flight. The
# restoring request is bound as a history owner only at install time, so the
# authority pruned the shared prefill history alone.
def retiring_prefix(monkeypatch):
    import types
    from vllm_kvmem.authority import RequestAuthority
    sessions(monkeypatch)
    c, a = prefix(); with_decode(c)
    a.records[(4, 1)] = NS(history_begin=192, count=32, handle=NS(slot=4, generation=1))
    for history in c.histories['old']:
        history.values[192] = [0.3, 1.]
    for history in c.decode_histories['old']:
        history.mass[192] = 0.125
    a.history_live = types.MethodType(RequestAuthority.history_live, a)
    snapshot = capture(c, 'old', a, 288, terminal=True)
    c.maps.pop('old'); c.histories.pop('old'); c.decode_histories.pop('old')
    return c, a, snapshot


def retire_192(a):
    """What RequestAuthority._drain_gc does for a request without bound owners."""
    del a.records[(4, 1)]
    a.action_history.prefill.values = {k: v for k, v in a.action_history.prefill.values.items()
                                       if k in a.history_live() or k >= a.history_end}
    assert 192 not in a.action_history.prefill.values


def test_restore_applies_a_retirement_prune_missed_before_prepare(monkeypatch):
    c, a, snapshot = retiring_prefix(monkeypatch)
    retire_192(a)                                    # between the claim and prepare_restore
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    assert all(192 not in h.values for h in histories)
    assert histories[0].snapshot() == a.action_history.prefill.snapshot()
    assert set(histories[1].values) == {0, 128}       # live keys of other layers stay
    install_restored(c, 'fresh', a, frame, histories, snapshot)
    # The decode histories were unaffected by the comparison: untouched here.
    assert capture(c, 'fresh', a, 288, committed=True).histories[0] == snapshot.histories[0][:0] + (
        tuple(row for row in snapshot.histories[0][0] if row[0] != 192),) + snapshot.histories[0][1:]


def test_restore_applies_a_retirement_prune_missed_between_prepare_and_install(monkeypatch):
    c, a, snapshot = retiring_prefix(monkeypatch)
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    assert all(192 in h.values for h in histories)
    retire_192(a)                                    # authority.poll() inside the draft rebuild
    install_restored(c, 'fresh', a, frame, histories, snapshot)
    assert all(192 not in h.values for h in c.histories['fresh'])
    assert all(192 not in h.mass for h in c.decode_histories['fresh'])
    assert [set(h.mass) for h in c.decode_histories['fresh']] == [{0, 64}, {128}]
    assert a.history_pruners == [c]
    # The realign capture (the check that failed) now passes.
    realigned = capture(c, 'fresh', a, 288, committed=True)
    assert all(192 not in {row[0] for row in rows} for rows, *_ in realigned.histories)


def test_restore_without_a_missed_prune_keeps_stale_keys_of_both_histories(monkeypatch):
    c, a, snapshot = retiring_prefix(monkeypatch)
    del a.records[(4, 1)]                            # retired, but nothing pruned yet
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    install_restored(c, 'fresh', a, frame, histories, snapshot)
    assert all(192 in h.values for h in c.histories['fresh'])
    assert 192 in a.action_history.prefill.values


def test_restore_still_refuses_scores_that_differ_beyond_a_retirement(monkeypatch):
    c, a, snapshot = retiring_prefix(monkeypatch)
    a.action_history.prefill.values[64] = [0.5, 1.]  # a live block's score changed
    with pytest.raises(ValueError, match='differ from the shared checkpoint'):
        prepare_restore(c, 'fresh', a, snapshot, options())
    c, a, snapshot = retiring_prefix(monkeypatch)
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    a.action_history.prefill.values.pop(192)         # pruned although the block is live
    with pytest.raises(ValueError, match=r'layer-only keys 1, shared-only keys 0'):
        install_restored(c, 'fresh', a, frame, histories, snapshot)
