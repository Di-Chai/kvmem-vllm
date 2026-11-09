"""KVMEM_LAYERWISE_PRESSURE_GPU_REUSE: block classification and reuse counters (CPU)."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import pressure_key_reuse as reuse


def test_flags_default_off_validate_and_yield_to_byte_verification(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', raising=False)
    monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_REUSE_LIMIT', raising=False)
    assert reuse.enabled() is False and reuse.reuse_limit() == 32
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', '1')
    assert reuse.enabled() is True
    monkeypatch.setenv('KVMEM_VERIFY_SESSION_RESTORE', '1')
    assert reuse.enabled() is False
    for bad in ('2', 'yes'):
        monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', bad)
        with pytest.raises(ValueError):
            reuse.enabled()
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_REUSE_LIMIT', '-1')
    with pytest.raises(ValueError):
        reuse.reuse_limit()


def test_unmoved_prefix_is_skipped_moved_blocks_reuse_until_the_limit_then_reload():
    # Old window: blocks at 0, 32, 64, ..., 320; blocks 64 and 160 are evicted.
    chosen = [(b, 32) for b in (0, 32, 96, 128, 192, 224)]
    sources = [0, 32, 96, 128, 192, 224]
    counts = {0: 5, 96: 1, 128: 2, 192: 0, 224: 1}
    moves, new, skipped = reuse.plan_layer(chosen, sources, counts, limit=2)
    assert skipped == 2 and new[0] == 5 and new[32] == 0            # untouched prefix keeps counts
    assert [(m[1], m[2], m[3], m[4], m[5]) for m in moves] == [
        (96, 96, 64, 32, reuse.REUSE), (128, 128, 96, 32, reuse.RELOAD),
        (192, 192, 128, 64, reuse.REUSE), (224, 224, 160, 64, reuse.REUSE)]
    assert (new[96], new[128], new[192], new[224]) == (2, 0, 1, 2)   # reuse +1, reload resets


def test_untrusted_pages_reload_every_moving_block_and_limit_zero_disables_reuse():
    chosen, sources = [(0, 32), (64, 32)], [0, 64]
    moves, new, _ = reuse.plan_layer(chosen, sources, {64: 0}, limit=32, untrusted=True)
    assert [m[5] for m in moves] == [reuse.RELOAD] and new[64] == 0
    moves, _, _ = reuse.plan_layer(chosen, sources, {}, limit=0)
    assert [m[5] for m in moves] == [reuse.RELOAD]


@pytest.mark.parametrize('sources', [[32, 0], [0, 96, 64 + 32]])
def test_compaction_must_shift_left_non_decreasingly(sources):
    chosen = [(i * 32, 32) for i in range(len(sources))]
    with pytest.raises(ValueError):
        reuse.plan_layer(chosen, sources, {}, limit=32)


def test_counter_lifecycle_untrusted_until_each_layer_refreshes(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', '1')
    controller = NS()
    counts, untrusted = reuse.counters(controller, 'r', 0)
    assert counts == {} and untrusted is False
    reuse.commit_counts(controller, 'r', 0, {64: 3})
    reuse.restored(controller, 'r', resident=True)                  # resident restore: unknown history
    assert reuse.counters(controller, 'r', 0) == ({}, True)
    reuse.commit_counts(controller, 'r', 0, {64: 0})
    assert reuse.counters(controller, 'r', 0) == ({64: 0}, False)
    assert reuse.counters(controller, 'r', 1)[1] is True            # other layers still untrusted
    reuse.restored(controller, 'r', resident=False)                 # copy restore: exact pages
    assert reuse.counters(controller, 'r', 1) == ({}, False)
    reuse.cancelled(NS(_kvmem_layerwise_pressure=controller), NS(command=NS(selection='pressure', request_id='r')))
    assert reuse.counters(controller, 'r', 2)[1] is True
    reuse.forget(controller, 'r')
    assert 'r' not in controller.reuse_counts and 'r' not in controller.reuse_untrusted


@pytest.mark.parametrize('reuse_flag', ['0', '1'])
def test_resident_sources_keep_layer_zero_first_only_with_gpu_reuse(monkeypatch, reuse_flag):
    """Lazy layer-0-first selection stays on with resident sources (GPU reuse);
    the frozen old windows equal what eager selection captures per layer."""
    from test_layerwise_pressure import controller, options, native_authority
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_LAYER0_FIRST', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', reuse_flag)
    def prepare(promised):
        c = controller(); c.prepared_commit = True; c.deferred_maps = True; c.resident_values = True
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
        if promised:
            job.promised_lease = object()
        c.query.emit = lambda *args, **kwargs: None
        try:
            c.prepare(NS(options=o), job)
            return len(job.layerwise_pressure_plan[0]), job.layerwise_pressure_value_sources
        finally:
            assert a.memory.close()
    eager_layers, eager_sources = prepare(False)
    lazy_layers, lazy_sources = prepare(True)
    assert eager_layers == 2
    assert lazy_layers == (1 if reuse_flag == '1' else 2)      # layer-0-first only with GPU reuse
    assert lazy_sources == eager_sources and len(lazy_sources[3]) == 2


def _native_or_skip(name):
    try:
        from vllm_kvmem import _native
    except ImportError:
        pytest.skip('native extension not built')
    fn = getattr(_native, name, None)
    if fn is None:
        pytest.skip(f'native extension lacks {name}')
    return fn


def _window(seed, blocks=40, evict=0.3):
    """Old dense window of whole blocks plus a compacting selection of it."""
    import random
    rng = random.Random(seed)
    begins = sorted(rng.sample(range(0, 4096 * 32, 32), blocks))
    mapping = tuple((0, i, b, 32, 32 * i) for i, b in enumerate(begins))
    kept = [i for i in range(blocks) if rng.random() > evict]
    chosen = [NS(history_begin=begins[i], count=32, archived=True) for i in kept]
    return mapping, chosen, [32 * i for i in kept], 32 * blocks, rng


@pytest.mark.parametrize('seed', range(20))
def test_native_plan_matches_the_python_reference(seed):
    native_bases = _native_or_skip('reuse_resident_bases')
    _native_or_skip('reuse_layer_plan')
    from vllm_kvmem.pressure_resident_values import resident_bases as python_bases
    mapping, chosen, sources, local, rng = _window(seed)
    assert native_bases(mapping, chosen, local, 32) == python_bases(chosen, mapping, local) == sources
    counts = {r.history_begin: rng.randrange(0, 4) for r in chosen if rng.random() < .8}
    for untrusted in (False, True):
        for limit in (0, 2, 32):
            got = reuse.layer_plan(chosen, sources, counts, limit, untrusted)
            native = reuse._native
            try:
                reuse._native = lambda name: None              # force the Python reference
                want = reuse.layer_plan(chosen, sources, counts, limit, untrusted)
            finally:
                reuse._native = native
            assert got[:5] == want[:5] and got[5].dtype == want[5].dtype
            assert got[5].tolist() == want[5].tolist()


@pytest.mark.parametrize('case', ['partial', 'unarchived', 'absent', 'duplicate', 'gap', 'short'])
def test_native_bases_fallbacks_and_errors_match_python(case):
    native_bases = _native_or_skip('reuse_resident_bases')
    from vllm_kvmem.pressure_resident_values import resident_bases as python_bases
    mapping, chosen, _, local, _ = _window(7, blocks=12, evict=0)
    if case == 'partial':
        chosen[3] = NS(history_begin=chosen[3].history_begin, count=16, archived=True)
    elif case == 'unarchived':
        chosen[5] = NS(history_begin=chosen[5].history_begin, count=32, archived=False)
    elif case == 'absent':
        chosen[2] = NS(history_begin=10 ** 9, count=32, archived=True)
    elif case == 'duplicate':
        chosen[4] = chosen[3]
    elif case == 'gap':
        mapping = mapping[:3] + tuple((a, b, c, d, e + 32) for a, b, c, d, e in mapping[3:])
    elif case == 'short':
        local += 32
    try:
        want = python_bases(chosen, mapping, local)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error)):
            native_bases(mapping, chosen, local, 32)
    else:
        assert native_bases(mapping, chosen, local, 32) == want


def test_native_plan_rejects_a_right_shift_like_python():
    _native_or_skip('reuse_layer_plan')
    chosen = [NS(history_begin=b, count=32) for b in (0, 32)]
    with pytest.raises(ValueError, match='shift kept blocks left'):
        reuse.layer_plan(chosen, [32, 0], {}, 32, False)
