"""Native score commit compared bit-for-bit with retained Python policy."""
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.prefill_attention import PrefillHistory


def histories(monkeypatch):
    options = NS(prefill_attention_half_life=8192)
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_COMMIT', '0')
    python = PrefillHistory(options)
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_COMMIT', '1')
    native = PrefillHistory(options)
    return python, native


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('layout', ['contiguous', 'reversed', 'strided'])
def test_double_updates_match_reference_across_decay_duplicate_pages_and_causality(monkeypatch, dtype, layout):
    python, native = histories(monkeypatch)
    rng = np.random.default_rng(812)
    for turn, count in enumerate((64, 129, 7, 1)):
        # Duplicate page identities model a partial selected page and suffix.
        keys = [int(v)*32 for v in rng.integers(0, 97, size=137)]
        starts = list(range(0, 137*32, 32))
        rows = list(range(7, count*32, 32))
        masses = rng.random((count, 274)).astype(dtype)
        if layout == 'strided': masses = masses[:, ::2]
        else: masses = masses[:, :137].copy()
        if layout == 'reversed': masses = masses[::-1, ::-1]
        masses /= masses.sum(axis=1, keepdims=True)
        # ndarray input preserves arbitrary buffer strides in native mode;
        # reference receives the exact widened Python float values.
        args = (8192*(turn+1), count*32, count*32, rows, starts, keys, 320)
        assert python.accept(*args, masses.tolist())
        assert native.accept(*args, masses)
        assert native.snapshot() == python.snapshot()
        assert list(native.values) == list(python.values)
        assert native.scores() == python.scores()


def test_real_prefill_shape_tensor_matches_reference(monkeypatch):
    python, native = histories(monkeypatch)
    generator = torch.Generator().manual_seed(5)
    matrix = torch.rand((64, 5375), generator=generator)
    matrix /= matrix.sum(dim=1, keepdim=True)
    history = list(range(0, 5375*32, 32))
    for state in (python, native):
        state.end = 170000
        state.values = {key: [.123456789, .875] for key in history}
    args = (172048, 2048, 2048, list(range(31, 2048, 32)), history, history, 170000)
    assert python.accept(*args, matrix)
    assert native.accept(*args, matrix)
    assert native.snapshot() == python.snapshot()


@pytest.mark.parametrize('invalid', ['nan', 'infinity', 'negative', 'sum', 'columns', 'missing', 'extra'])
def test_failed_commit_never_publishes_partial_decay_or_rows(monkeypatch, invalid):
    python, native = histories(monkeypatch)
    masses = np.full((4, 8), .125)
    if invalid == 'nan': masses[-1, -1] = math.nan
    if invalid == 'infinity': masses[-1, -1] = math.inf
    if invalid == 'negative': masses[-1, -1] = -.125
    if invalid == 'sum': masses[-1, -1] = .5
    if invalid == 'columns': masses = masses[:, :-1]
    if invalid == 'missing': masses = masses[:-1]
    if invalid == 'extra': masses = np.vstack([masses, masses[:1]])
    for state in (python, native):
        state.values = {0: [.7, .9], 5000: [.1, .2]}
        snapshot = state.snapshot()
        with pytest.raises(RuntimeError):
            state.accept(1024, 32, 32, [7, 15, 23, 31], list(range(8)),
                         list(range(8)), 0, masses.tolist() if state is python else masses)
        assert state.snapshot() == snapshot


@pytest.mark.parametrize('mass', [math.nextafter(.995, -math.inf), .995,
                                  math.nextafter(.995, math.inf),
                                  math.nextafter(1.005, -math.inf), 1.005,
                                  math.nextafter(1.005, math.inf)])
def test_validation_threshold_has_same_double_semantics(monkeypatch, mass):
    python, native = histories(monkeypatch)
    outcomes = []
    for state in (python, native):
        try:
            state.accept(4, 4, 4, [3], [0], [0], 0, [[mass]])
        except RuntimeError:
            outcomes.append('error')
        else:
            outcomes.append(state.snapshot())
    assert outcomes[0] == outcomes[1]


def test_rejected_chunk_and_old_end_never_convert_or_read_mass(monkeypatch):
    _, native = histories(monkeypatch)
    class NotReady:
        def numpy(self): raise AssertionError('Rejected transaction read GPU/CPU feedback')
    native.end = 4
    assert not native.accept(8, 4, 3, [3], [0], [0], 0, NotReady())
    assert not native.accept(4, 4, 4, [3], [0], [0], 0, NotReady())
    assert native.snapshot() == ({}, 4, 0, 0)


def test_native_buffer_validation_and_owned_input_state(monkeypatch):
    _, native = histories(monkeypatch)
    previous = {0: [.25, .5]}
    with pytest.raises(ValueError, match='float32/float64'):
        _native.commit_prefill_attention(previous, .5, [3], [0], [0], 0,
                                         np.array([[1]], dtype=np.int32))
    assert previous == {0: [.25, .5]}
    snapshot = native.snapshot()
    native.accept(4, 4, 4, [3], [0], [0], 0, [[1.]])
    assert snapshot == ({}, 0, 0, 0)
    saved = native.snapshot()
    native.restore(saved, 4)
    native.values[0][0] = .3
    assert saved[0][0] == (1., 1.)


def test_direct_options_default_and_environment_override(monkeypatch):
    monkeypatch.delenv('KVMEM_NATIVE_PREFILL_COMMIT', raising=False)
    assert not PrefillHistory(NS(native_prefill_commit=False)).native_commit
    assert PrefillHistory(NS(native_prefill_commit=True)).native_commit
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_COMMIT', '1')
    assert PrefillHistory(NS(native_prefill_commit=False)).native_commit
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_COMMIT', 'invalid')
    with pytest.raises(ValueError, match='0 or 1'): PrefillHistory(NS())


def test_action_runtime_passes_tensor_without_tolist_and_reports_commit_cost(tmp_path, monkeypatch):
    from vllm_kvmem.action_attention import ActionRuntime, ActionPacket
    _, history = histories(monkeypatch)
    mass = torch.ones((1, 1))
    monkeypatch.setattr(torch.Tensor, 'tolist', lambda _: (_ for _ in ()).throw(AssertionError('scalar materialization')))
    runtime = ActionRuntime.__new__(ActionRuntime)
    runtime.runtime = NS(output=tmp_path)
    state = NS(prefill=history)
    packet = NS(action_inputs={'r': ActionPacket(state, [3], [0], mass, None, (4, 4, [0], 0))})
    runtime.accept(packet, 'r', 4)
    assert history.values == {0: [1., 1.]}
    assert runtime.prefill_commit_native_calls == runtime.prefill_commit_calls == 1
    assert runtime.prefill_commit_host_seconds >= 0
