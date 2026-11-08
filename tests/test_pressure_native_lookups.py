"""KVMEM_PRESSURE_NATIVE_LOOKUPS: scores, snapshot check and seed equal the dict paths."""
import math
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import _prefill_state_native as native
from vllm_kvmem.buffered_prefill_attention import BufferedPrefillHistory
from vllm_kvmem.layerwise_pressure import native_lookup_scores, same_snapshot
from vllm_kvmem.prefill_attention import PrefillHistory

pytestmark = pytest.mark.skipif(not hasattr(native.State, 'scores_for'),
                                reason='Requires the rebuilt prefill state extension')


def options():
    return NS(native_prefill_commit=True, prefill_attention_half_life=8192., memory=None)


VALUES = {0: [1.5, 3.], 32: [0., 2.], 64: [2., 0.], 96: [-0., 4.], 128: [1e-300, 3.], 160: [7., 7.]}


def buffered(monkeypatch, values=VALUES):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT', '1')
    h = BufferedPrefillHistory(options())
    h.values = {k: list(v) for k, v in values.items()}
    h.end, h.samples, h.chunks = 4096, 12, 3
    return h


def test_scores_for_matches_dict_get_bitwise(monkeypatch):
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '1')
    h = buffered(monkeypatch)
    begins = [160, 0, 32, 64, 96, 128, 999, 0]
    mapping = [(0, 0, b, 32, 0) for b in begins]
    expected = [h.scores().get(b, 0.) for b in begins]
    actual = native_lookup_scores(h, mapping)
    assert [math.copysign(1, x) for x in actual] == [math.copysign(1, x) for x in expected]
    assert actual == expected
    monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', '0')
    assert native_lookup_scores(h, mapping) is None


@pytest.mark.parametrize('change', ['none', 'value', 'missing', 'extra', 'end', 'samples', 'negzero'])
def test_same_snapshot_matches_python_comparison(monkeypatch, change):
    h = buffered(monkeypatch)
    g = PrefillHistory(options())
    g.values = {k: list(v) for k, v in VALUES.items()}
    g.end, g.samples, g.chunks = 4096, 12, 3
    if change == 'value': g.values[32][0] = 1e-12
    elif change == 'missing': del g.values[64]
    elif change == 'extra': g.values[192] = [0., 0.]
    elif change == 'end': g.end += 1
    elif change == 'samples': g.samples += 1
    elif change == 'negzero': g.values[32][0] = -0.
    expected = g.snapshot() == h.snapshot()
    for flag in ('0', '1'):
        monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', flag)
        assert same_snapshot(g, h) == expected


@pytest.mark.parametrize('prefill', [{4: (2., 4.), 8: (1., 2.), 12: (5., 0.)}, {}])
def test_pressure_seed_lookup_matches_dict(tmp_path, monkeypatch, prefill):
    from test_graph_lifecycle import fixture
    from test_graph_pressure_v2 import configure
    from vllm_kvmem.prefill_attention import pressure_seed
    a, archive, graph = fixture(tmp_path, sink=1)
    state = configure(a, policy='attention')
    state.prefill.values = {k: list(v) for k, v in prefill.items()}
    candidates = tuple(a.memory._store.working_records())
    results = []
    for flag in ('0', '1'):
        monkeypatch.setenv('KVMEM_PRESSURE_NATIVE_LOOKUPS', flag)
        results.append(pressure_seed(a, candidates))
    assert results[0] == results[1]
    assert a.close() and archive.close()
