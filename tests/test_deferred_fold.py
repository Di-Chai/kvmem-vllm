"""KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD: layer 0 folds at publication, others on first use."""
from copy import copy
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import _prefill_state_native as native
from vllm_kvmem import deferred_fold
from vllm_kvmem.buffered_prefill_attention import BufferedPrefillHistory, accept_buffered_batch
from vllm_kvmem.deferred_fold import DeferredFoldHistory, accept_deferred, deferrable, force_all
from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure, _LayerwiseAcceptance

L = 16


@pytest.fixture(autouse=True)
def batch_native():
    if not hasattr(native, 'committed_batch'):
        pytest.skip('Requires the native batch acceptance extension')


def options(layer=0):
    return NS(native_prefill_commit=True, prefill_attention_half_life=8192.+layer,
              pressure_policy='attention', guided_query_tokens=0, query_history=False, prefill_attention=True)


def bits(values):
    return [(key, struct.pack('<dd', *pair)) for key, pair in values.items()]


def state(history):
    return (bits(history.values), history.end, history.samples, history.chunks,
            [(key, struct.pack('<d', score)) for key, score in history.scores().items()],
            history.successful_buffered_commits, history.successful_buffered_batch_commits,
            history.successful_buffered_restores, getattr(history, 'last_commit_backend', None))


def spans(steps, seed=5):
    rng = np.random.default_rng(seed)
    for step in range(1, steps+1):
        segments = 40 + 7*step
        logical = tuple(tuple(rng.integers(0, 120, size=segments).tolist()) for _ in range(L))
        starts = list(range(0, segments*32, 32))
        backing = torch.from_numpy(rng.random((L, 9, segments+3)))
        masses = backing[:, 1:7, :segments]                     # strided view, like a logical slice
        masses = masses / masses.sum(dim=2, keepdim=True)
        rows = list(range(31, 192, 32))
        yield step*1024, 1024, rows, starts, logical, 4096+step, masses


def run(kind, steps=6, observe_after=None):
    cls = DeferredFoldHistory if kind == 'deferred' else BufferedPrefillHistory
    committed = [cls(options(layer)) for layer in range(L)]
    observed = {}
    for index, (end, count, rows, starts, logical, local, masses) in enumerate(spans(steps)):
        staged = [copy(h) for h in committed]
        if kind == 'deferred':
            assert deferrable(staged, end, count, count, rows, logical, masses)
            flags = accept_deferred(staged, end, count, count, rows, starts, logical, local, masses)
        else:
            flags = accept_buffered_batch(staged, end, count, count, rows, starts, logical, local, masses)
        assert flags == [True]*L
        committed = staged
        if observe_after and index in observe_after:
            observed[index] = [state(committed[layer]) for layer in observe_after[index]]
    return committed, observed


@pytest.mark.parametrize('order', ['reverse', 'force_all', 'interleaved'])
def test_deferred_histories_equal_eager_every_double(order):
    peek = {2: [5, 9], 4: [1, 15, 9]} if order == 'interleaved' else None
    eager, eager_seen = run('eager', observe_after=peek)
    lazy, lazy_seen = run('deferred', observe_after=peek)
    assert lazy_seen == eager_seen
    assert lazy[0].pending_count() == 0
    if order == 'reverse':
        assert all(h.pending_count() == 6 for h in lazy[1:])
        actual = [state(h) for h in reversed(lazy)][::-1]
    else:
        if order == 'force_all':
            force_all(lazy)
            assert all(h.pending_count() == 0 for h in lazy)
        actual = [state(h) for h in lazy]
    assert actual == [state(h) for h in eager]


def test_force_all_uses_one_batched_call_per_span(monkeypatch):
    lazy, _ = run('deferred', steps=3)
    calls = []
    original = native.committed_batch
    def observed(states, *args):
        calls.append(len(states))
        return original(states, *args)
    monkeypatch.setattr(native, 'committed_batch', observed)
    force_all(lazy)
    assert calls == [L-1]*3


def test_staged_copy_never_changes_committed_state():
    committed, _ = run('deferred', steps=2)
    before_pending = [h.pending_count() for h in committed]
    end, count, rows, starts, logical, local, masses = list(spans(3))[-1]
    staged = [copy(h) for h in committed]
    accept_deferred(staged, end, count, count, rows, starts, logical, local, masses)
    force_all(staged)
    assert [h.pending_count() for h in committed] == before_pending
    eager, _ = run('eager', steps=2)
    assert [state(h) for h in committed] == [state(h) for h in eager]


def test_writes_apply_pending_first():
    lazy, _ = run('deferred', steps=3)
    eager, _ = run('eager', steps=3)
    # A write (restore/values) applies queued folds before replacing state.
    lazy[3].values = {1: [.5, 1.]}
    assert lazy[3].pending_count() == 0 and bits(lazy[3].values) == bits({1: [.5, 1.]})
    assert lazy[3].end == eager[3].end


def test_pending_queue_is_bounded(monkeypatch):
    monkeypatch.setattr(deferred_fold, 'MAX_PENDING', 2)
    lazy, _ = run('deferred', steps=2)
    end, count, rows, starts, logical, local, masses = list(spans(3))[-1]
    assert not deferrable(lazy, end, count, count, rows, logical, masses)


def test_ineligible_spans_keep_the_eager_path():
    lazy = [DeferredFoldHistory(options(layer)) for layer in range(L)]
    end, count, rows, starts, logical, local, masses = next(spans(1))
    assert not deferrable(lazy, end, count, count-1, rows, logical, masses)        # partial accept
    assert not deferrable(lazy, end, count, count, [], logical, masses)            # no sampled rows
    assert not deferrable(lazy, 0, count, count, rows, logical, masses)            # stale end
    assert not deferrable([BufferedPrefillHistory(options())]*L, end, count, count, rows, logical, masses)
    # The eager batch still accepts deferred histories (forcing them first).
    assert accept_buffered_batch(lazy, end, count, count, rows, starts, logical, local, masses) == [True]*L


def controller(monkeypatch, deferred=True):
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD', '1' if deferred else '0')
    query = NS(native_batches=True, optimized=True, names=tuple(range(L)),
               runner=NS(model_config=NS(enforce_eager=True)))
    return LayerwisePrefillPressure(query)


def test_controller_folds_only_layer_zero_at_publication(monkeypatch):
    c = controller(monkeypatch)
    histories = c._history('owner', options())
    assert all(type(h) is DeferredFoldHistory for h in histories)
    backing = torch.full((L, 3, 8), .25, dtype=torch.float32)
    masses = backing[:, 1:2, ::2]
    pending = NS(prefill=(512, 512, [0, 32, 64, 96], 0), rows=[511],
                 layerwise_pressure=(c, 'owner', [[layer*1024+i*32 for i in range(4)] for layer in range(L)],
                                     masses))
    calls = []
    original = native.committed_batch
    monkeypatch.setattr(native, 'committed_batch', lambda states, *a: (calls.append(len(states)),
                                                                       original(states, *a))[1])
    staged = c.prepare_accept(pending, 512)
    assert calls == [1]
    transaction = NS(commit=lambda: None, abort=lambda: None)
    _LayerwiseAcceptance(transaction, c, staged).commit()
    committed = c.histories['owner']
    assert [h.pending_count() for h in committed] == [0] + [1]*(L-1)
    assert committed[4].chunks == 1 and calls == [1, 1]          # first read folds layer 4 only
    force_all(committed)
    assert calls == [1, 1, 3, 11]                                 # runs 1-3 and 5-15
    assert all(h.chunks == 1 for h in committed)


def test_flag_validation(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD', 'yes')
    with pytest.raises(ValueError, match='DEFERRED_FOLD'):
        deferred_fold.enabled()
    monkeypatch.setenv('KVMEM_LAYERWISE_BUFFERED_HISTORY', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT', '0')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_FOLD', '1')
    query = NS(native_batches=True, optimized=True, names=(0, 1), runner=NS(model_config=NS(enforce_eager=True)))
    with pytest.raises(ValueError, match='batched layer acceptance'):
        LayerwisePrefillPressure(query)


@pytest.mark.parametrize('fault', ['nan_layer1', 'negative_layer5', 'sum_layer15', 'nan_layer0', 'shape_layer3'])
def test_invalid_spans_fail_at_publication_like_the_eager_batch(fault):
    end, count, rows, starts, logical, local, masses = next(spans(1))
    masses = masses.clone()
    logical = list(logical)
    if fault == 'nan_layer1': masses[1, 0, 0] = float('nan')
    elif fault == 'negative_layer5': masses[5, 1, 0] = -.25
    elif fault == 'sum_layer15': masses[15, 2, 0] += .5
    elif fault == 'nan_layer0': masses[0, 0, 1] = float('nan'); masses[3, 0, 0] = float('nan')
    elif fault == 'shape_layer3': logical[3] = logical[3][:-1]
    outcomes = []
    for kind in ('eager', 'deferred'):
        cls = DeferredFoldHistory if kind == 'deferred' else BufferedPrefillHistory
        staged = [cls(options(layer)) for layer in range(L)]
        before = [h.chunks for h in staged]
        try:
            if kind == 'deferred':
                assert deferrable(staged, end, count, count, rows, logical, masses)
                accept_deferred(staged, end, count, count, rows, starts, logical, local, masses)
            else:
                accept_buffered_batch(staged, end, count, count, rows, starts, logical, local, masses)
            outcomes.append('accepted')
        except (RuntimeError, ValueError) as error:
            outcomes.append((type(error), str(error)))
    assert outcomes[0] == outcomes[1] and outcomes[0] != 'accepted'


def test_deferred_spans_release_the_download_pool_lease():
    """Queued folds must not pin action-download slots (N>=4 exhausted the pool)."""
    import gc
    from vllm_kvmem.action_download_pool import ActionDownloadPool

    class Event:
        def query(self): return True
        def synchronize(self): pass

    class Backend:
        def allocate(self, size): return torch.empty(size, dtype=torch.uint8)
        def event(self): return Event()
        def record(self, event, source): pass

    items = list(spans(3))
    size = max(m.numel() * m.element_size() for *_, m in items)
    pool = ActionDownloadPool(1 << (size - 1).bit_length(), 1, backend=Backend())
    lazy = [DeferredFoldHistory(options(layer)) for layer in range(L)]
    for end, count, rows, starts, logical, local, masses in items:
        host, event = pool.copy(masses.contiguous())        # one slot: must be free again
        event.synchronize()
        staged = [copy(h) for h in lazy]
        assert deferrable(staged, end, count, count, rows, logical, host)
        accept_deferred(staged, end, count, count, rows, starts, logical, local, host)
        lazy = staged
        del host, event
        gc.collect()
    assert all(h.pending_count() == 3 for h in lazy[1:])
    host, event = pool.copy(items[0][-1].contiguous())     # still free with every fold pending
    del host, event
    eager, _ = run('eager', steps=3)
    force_all(lazy)
    assert [state(h) for h in lazy] == [state(h) for h in eager]


def test_deferred_span_copy_is_private_and_exact():
    end, count, rows, starts, logical, local, masses = next(spans(1))
    span = deferred_fold._Span(end, tuple(rows), tuple(starts), local, tuple(logical), masses)
    expected = masses.numpy()[1:]
    assert span.layers(1, L - 1).flags['C_CONTIGUOUS']
    assert span.layers(1, L - 1).dtype == expected.dtype
    assert span.layers(1, L - 1).tobytes() == np.ascontiguousarray(expected).tobytes()
    assert not np.shares_memory(span.buffer, masses.numpy())
    assert span.layers(7, 3).tobytes() == np.ascontiguousarray(expected[6:9]).tobytes()
    with pytest.raises(RuntimeError, match='retain'):
        span.layers(0, 1)
