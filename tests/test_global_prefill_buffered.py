"""KVMEM_GLOBAL_PREFILL_BUFFERED: the global prefill history on the native state is identical."""
import struct
from copy import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import _prefill_state_native as native
from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
from vllm_kvmem.serving.prefill_attention import PrefillHistory

pytestmark = pytest.mark.skipif(not hasattr(native.State, 'same_state'), reason='Requires rebuilt extension')


def options():
    return NS(native_prefill_commit=True, prefill_attention_half_life=8192., memory=None)


def bits(snapshot):
    values, end, samples, chunks = snapshot
    return sorted((k, struct.pack('<dd', *v)) for k, v in values.items()), end, samples, chunks


@pytest.mark.parametrize('export', ['0', '1'])
def test_buffered_global_history_matches_dict_backend(monkeypatch, export):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', export)
    rng = np.random.default_rng(11)
    plain, buffered = PrefillHistory(options()), BufferedPrefillHistory(options())
    for step in range(1, 9):
        segments = 30 + 9 * step
        keys = rng.integers(0, 200, size=segments).tolist()
        starts = list(range(0, segments * 32, 32))
        masses = torch.from_numpy(rng.random((6, segments + 2)))[:, :segments]
        masses = masses / masses.sum(dim=1, keepdim=True)
        rows = list(range(31, 192, 32))
        staged = [copy(plain), copy(buffered)]
        results = [h.accept(step * 1024, 1024, 1024, rows, starts, keys, 512 * step, masses) for h in staged]
        assert results == [True, True]
        plain, buffered = staged
        assert bits(plain.snapshot()) == bits(buffered.snapshot())
        sp, sb = plain.scores(), buffered.scores()
        assert sorted(sp) == sorted(sb) and all(struct.pack('<d', sp[k]) == struct.pack('<d', sb[k]) for k in sp)
        assert plain.page_count() == buffered.page_count()


def test_same_state_has_dict_equality_semantics():
    a = native.State.from_values({1: [0., 1.], 2: [.5, 2.]})
    b = native.State.from_values({2: [.5, 2.], 1: [-0., 1.]})
    c = native.State.from_values({1: [0., 1.], 2: [.5, 2.000001]})
    d = native.State.from_values({1: [0., 1.]})
    nan = native.State.from_values({1: [float('nan'), 1.]})
    assert a.same_state(b) and b.same_state(a)
    assert not a.same_state(c) and not a.same_state(d) and not nan.same_state(nan)
    assert a.size() == 2 and d.size() == 1


def test_same_snapshot_both_native(monkeypatch):
    from vllm_kvmem.serving.layerwise_pressure import same_snapshot
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    g, h = BufferedPrefillHistory(options()), BufferedPrefillHistory(options())
    g.values = h.values = {4: [1., 2.], 8: [0., 1.]}
    g.end = h.end = 64; g.samples = h.samples = 3; g.chunks = h.chunks = 1
    for flag in ('0', '1'):
        monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', flag)
        assert same_snapshot(g, h) == (g.snapshot() == h.snapshot()) == True
        h.chunks = 2
        assert same_snapshot(g, h) == (g.snapshot() == h.snapshot()) == False
        h.chunks = 1
