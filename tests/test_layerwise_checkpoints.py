from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving.layerwise_checkpoints import (
    capture, prepare_restore, restore_sync, require_supported_restore, oracle_record_groups,
)
from test_layerwise_pressure import controller, options


def prefix():
    c = controller()
    c.query.finished = set()
    c.maps['old'] = (256, (((0, 1, 0, 32, 0), (1, 1, 64, 32, 32)),
                           ((0, 1, 0, 32, 0), (2, 1, 128, 32, 32))))
    records = [NS(history_begin=begin, count=32, handle=NS(slot=i, generation=1))
               for i, begin in enumerate((0, 64, 128, 256))]
    shared = ((0, 1, 0, 32, 0), (1, 1, 64, 32, 32), (3, 1, 256, 32, 64))
    a = NS(history_end=288, records={(r.handle.slot, 1): r for r in records},
           memory=NS(config=NS(block_tokens=32), _store=NS(working_records=lambda: shared)))
    histories = c._history('old', options())
    for i, history in enumerate(histories):
        history.values = {0: [0.1, 1.], (64 if i == 0 else 128): [0.9, 1.]}
        history.end, history.samples, history.chunks = 256, 8, 1
    a.action_history = NS(prefill=histories[0])
    return c, a


def test_prefix_survives_finish_and_query_mutations_without_changing_next_owner():
    c, a = prefix()
    snapshot = capture(c, 'old', a, 288)
    assert snapshot.maps[0][-1] == (3, 1, 256, 32, 64)
    assert [row[2] for row in snapshot.maps[1]] == [0, 128, 256]
    assert snapshot.bytes == 8 + 2 * 3 * 40 + 2 * (24 + 2 * 24)
    original = a.action_history.prefill.snapshot()
    c.histories['old'][1].values[128][0] = 999.
    c.maps.pop('old'); c.histories.pop('old'); c.query.finished.add('old')
    chosen, frame, histories = prepare_restore(c, 'fresh', a, snapshot, options())
    assert [r.history_begin for r in chosen[1]] == [0, 128, 256]
    assert histories[0].snapshot() == original
    assert histories[1].values[128] == [0.9, 1.]
    assert frame == (288, snapshot.maps)
    assert 'fresh' not in c.maps and 'fresh' not in c.histories
    histories[1].values[128][0] = -1.
    assert snapshot.histories[1][0][1] == (128, 0.9, 1.)


def test_uncompressed_prefix_keeps_identical_maps_and_independent_scores():
    c, a = prefix(); c.maps.clear()
    snapshot = capture(c, 'old', a, 288)
    assert snapshot.maps[0] == snapshot.maps[1]
    assert snapshot.histories[0] != snapshot.histories[1]


@pytest.mark.parametrize('fault', ['missing_history', 'future_score', 'wrong_boundary', 'after_query', 'shared_score'])
def test_capture_rejects_unqualified_state(fault):
    c, a = prefix(); boundary = 288
    if fault == 'missing_history': c.histories.pop('old')
    elif fault == 'future_score': c.histories['old'][1].end = 320
    elif fault == 'wrong_boundary': boundary = 256
    elif fault == 'after_query': c.query.finished.add('old')
    elif fault == 'shared_score':
        a.action_history.prefill = NS(snapshot=lambda: ({}, 0, 0, 0))
    with pytest.raises(ValueError): capture(c, 'old', a, boundary)


@pytest.mark.parametrize('fault', ['missing', 'layer_order', 'layer_count', 'missing_archive', 'rewind',
                                 'existing_owner', 'shared_frame', 'future_score', 'duplicate_block'])
def test_restore_fails_atomically_instead_of_using_a_shared_map(fault):
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    before = dict(c.maps), dict(c.histories)
    if fault == 'missing': snapshot = None
    elif fault == 'layer_order': snapshot = replace(snapshot, names=('b', 'a'))
    elif fault == 'layer_count': snapshot = replace(snapshot, histories=snapshot.histories[:1])
    elif fault == 'missing_archive': a.records.pop((2, 1))
    elif fault == 'rewind': a.history_end = 320
    elif fault == 'existing_owner': c.query.finished.add('fresh')
    elif fault == 'shared_frame': a.memory._store.working_records = lambda: snapshot.maps[1]
    elif fault == 'future_score':
        rows, _, samples, chunks = snapshot.histories[1]
        snapshot = replace(snapshot, histories=(snapshot.histories[0], (rows, 320, samples, chunks)))
    elif fault == 'duplicate_block':
        first, _, last = snapshot.maps[1]
        snapshot = replace(snapshot, maps=(snapshot.maps[0], (first, (*first[:4], 32), last)))
    with pytest.raises(ValueError): prepare_restore(c, 'fresh', a, snapshot, options())
    assert (c.maps, c.histories) == before


def test_restore_resolves_archive_handles_without_changing_logical_selection():
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    # Archive handles are validated against the authoritative retained records,
    # while text coordinates and dense RoPE positions remain frozen.
    a.records[(2, 1)].handle = NS(slot=17, generation=4)
    chosen, frame, _ = prepare_restore(c, 'fresh', a, snapshot, options())
    assert frame[1][1][1] == (17, 4, 128, 32, 32)
    assert chosen[1][1].handle.slot == 17


@pytest.mark.parametrize('fail', [False, True])
def test_sync_restore_publishes_per_layer_state_only_after_completed_fence(monkeypatch, fail):
    import torch
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c, device='cuda'),
                options=options(), bindings={'fresh': NS(pressure_restore=False)})
    worker.options.session_freeze_prompt = True
    checkpoint = NS(kind='P', authority=a, layerwise_pressure=snapshot)
    events = []
    def assemble(state, layer, records):
        assert 'fresh' not in c.maps and 'fresh' not in c.histories
        assert state['selected_tokens'] == 96 and layer == 1
        assert [r.history_begin for r in records] == [0, 128, 256]
        if fail: raise RuntimeError('copy failed')
        events.append('copy')
    def synchronize():
        assert 'fresh' not in c.maps and 'fresh' not in c.histories
        events.append('fence')
    def emit(event, **fields):
        assert 'fresh' in c.maps and 'fresh' in c.histories
        assert fields['map_sha256'] == snapshot.evidence()['map_sha256']
        assert fields['attention_sha256'] == snapshot.evidence()['attention_sha256']
        events.append('publish')
    c.query.assemble_layer_pinned = assemble; c.query.emit = emit
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda _: NS(synchronize=synchronize))
    require_supported_restore(worker, 'fresh', checkpoint)
    if fail:
        with pytest.raises(RuntimeError, match='copy failed'): restore_sync(worker, 'fresh', checkpoint)
        assert 'fresh' not in c.maps and 'fresh' not in c.histories
    else:
        restore_sync(worker, 'fresh', checkpoint)
        assert events == ['copy', 'fence', 'publish']
        assert c.maps['fresh'] == (288, snapshot.maps)


@pytest.mark.parametrize('fault', ['async', 'pressure', 'mutable', 'terminal', 'no_snapshot', 'no_controller'])
def test_unsupported_restore_is_rejected_before_assembly(fault):
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c), options=NS(session_freeze_prompt=True),
                bindings={'fresh': NS(pressure_restore=fault == 'pressure')})
    checkpoint = NS(kind='M' if fault == 'terminal' else 'P', layerwise_pressure=snapshot)
    if fault == 'mutable': worker.options.session_freeze_prompt = False
    elif fault == 'no_snapshot': checkpoint.layerwise_pressure = None
    elif fault == 'no_controller': del worker.runner._kvmem_layerwise_pressure
    with pytest.raises(ValueError):
        require_supported_restore(worker, 'fresh', checkpoint, asynchronous=fault == 'async')


@pytest.mark.parametrize('foreground_busy',[False,True])
def test_sync_prefix_restore_does_not_claim_a_background_pressure_producers_scratch(monkeypatch,foreground_busy):
    import torch
    from vllm_kvmem.serving.window_assembly import _AssemblyStaging
    c,a=prefix();snapshot=capture(c,'old',a,288)
    worker=NS(runner=NS(_kvmem_layerwise_pressure=c,device='cpu'),
        options=options(),bindings={'fresh':NS(pressure_restore=False)})
    worker.options.session_freeze_prompt=True
    checkpoint=NS(kind='P',authority=a,layerwise_pressure=snapshot)
    pools={name:_AssemblyStaging(256,'cpu') for name in ('pressure','layerwise')}
    pressure_owner,foreground_owner=object(),object()
    pools['pressure'].claim(pressure_owner)
    if foreground_busy:pools['layerwise'].claim(foreground_owner)
    copies=[]
    def assemble(state,layer,records):
        staging=pools[state['assembly_arena']];owner=object()
        staging.claim(owner)
        try:copies.append((layer,[r.history_begin for r in records]))
        finally:staging.release(owner)
    c.query.assemble_layer_pinned=assemble
    c.query.emit=lambda *args,**kwargs:None
    monkeypatch.setattr(torch.cuda,'current_stream',lambda _:NS(synchronize=lambda:None))
    try:
        if foreground_busy:
            with pytest.raises(RuntimeError,match='already has a CPU producer'):
                restore_sync(worker,'fresh',checkpoint)
            assert 'fresh' not in c.maps and 'fresh' not in c.histories
        else:
            restore_sync(worker,'fresh',checkpoint)
            assert copies==[(1,[0,128,256])]
            assert c.maps['fresh']==(288,snapshot.maps)
            assert c.histories['fresh'][1].snapshot()==c.histories['old'][1].snapshot()
        assert pools['pressure'].owner is pressure_owner
    finally:
        pools['pressure'].release(pressure_owner)
        if foreground_busy:pools['layerwise'].release(foreground_owner)


def test_restore_oracle_uses_own_target_frames_and_shared_draft_frame():
    c, a = prefix(); snapshot = capture(c, 'old', a, 288)
    c.maps['fresh'] = (288, snapshot.maps)
    worker = NS(runner=NS(_kvmem_layerwise_pressure=c), runtime=NS(names=('a', 'b', 'draft')))
    groups = oracle_record_groups(worker, 'fresh', a)
    assert groups == ((snapshot.maps[0], (0, 2)), (snapshot.maps[1], (1,)))
    del worker.runner._kvmem_layerwise_pressure
    assert oracle_record_groups(worker, 'fresh', a) == ((snapshot.maps[0], (0, 1, 2)),)


def pool_prefix(tmp_path):
    from test_prompt_checkpoints import saved
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    c = controller(); c.query.finished = set()
    hs = c._history('a', options())
    for history in hs:
        history.values = {0: [0.7, 1.], 4: [0.3, 1.]}
        history.end, history.samples, history.chunks = 8, 2, 1
    authority.action_history = NS(prefill=hs[0], snapshot=lambda pos: ({}, 0, hs[0].snapshot()))
    shared = pool.capture(old.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    adapter.runner._kvmem_layerwise_pressure = c
    adapter.runner._kvmem_sessions = NS(options=NS(session_freeze_prompt=True))
    return pool, shared, authority, archive, adapter


def test_pool_accounts_for_per_layer_state_and_releases_it(tmp_path):
    pool, shared, authority, archive, adapter = pool_prefix(tmp_path)
    previous_bytes = pool.payload_bytes
    checkpoint = pool.capture(shared.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert checkpoint.layerwise_pressure is not None
    assert pool.payload_bytes == previous_bytes + checkpoint.layerwise_pressure.bytes
    assert checkpoint.layerwise_pressure.maps[0] == checkpoint.layerwise_pressure.maps[1]
    with pytest.raises(ValueError, match='Frozen layerwise'):
        pool.update_live_window(checkpoint)
    pool.drop_live(checkpoint)
    assert checkpoint.layerwise_pressure is None and pool.payload_bytes == 0
    assert authority.close() and archive.close()


def test_pool_extra_metadata_admission_failure_preserves_original_checkpoint(tmp_path):
    pool, shared, authority, archive, adapter = pool_prefix(tmp_path)
    previous_bytes = pool.payload_bytes
    pool.max_payload_bytes = previous_bytes
    with pytest.raises(BufferError):
        pool.capture(shared.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert pool.entries[shared.namespace] is shared and pool.payload_bytes == previous_bytes
    assert shared.layerwise_pressure is None
    pool.drop_live(shared)
    assert authority.close() and archive.close()


def test_tiny_lme_checkpoint_quota_declines_before_unsupported_terminal_metadata(tmp_path):
    pool, shared, authority, archive, adapter = pool_prefix(tmp_path)
    pool.max_payload_bytes = 1
    adapter.runner._kvmem_sessions.options.session_freeze_prompt = False
    with pytest.raises(BufferError):
        pool.capture(shared.namespace, 'a', authority, list(range(11)), adapter,
                     replace=True, terminal=True)
    assert pool.entries[shared.namespace] is shared
    pool.drop_live(shared)
    assert authority.close() and archive.close()
