"""Dense host feedback must preserve the scalar graph transaction exactly."""
import math

import numpy as np
import pytest

from vllm_kvmem import _native
from vllm_kvmem.block_graph import BlockGraph, GraphSample


def commit(graph, begin, end, positions, targets, masses, kind='prefill', dense=True):
    if dense:
        update = graph.prepare_accept(begin, end, positions, targets, masses, kind)
    else:
        update = graph.prepare(begin, end, tuple(GraphSample(p, tuple(targets), v, kind)
                               for p, v in zip(positions, masses)))
    with update:
        update.commit()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('strided', [False, True])
def test_dense_matches_reference_through_partial_replay_restore_and_duplicate_targets(dtype, strided):
    fast, slow = BlockGraph('fast', 32, topk=4), BlockGraph('slow', 32, topk=4)
    rng = np.random.default_rng(915)
    checkpoints = []
    try:
        for begin, end, positions in [(0, 51, [47, 32]), (51, 97, [95, 51, 96]),
                                      (97, 131, [100, 130]), (32, 131, [32, 96]),
                                      (131, 160, [159, 140]), (160, 161, [])]:
            targets = (0, 0, 32, 64, 96, 128, 160)
            values = rng.random((len(positions), len(targets))).astype(dtype)
            for p, row in zip(positions, values):
                row[np.asarray(targets) > p] = 0
                row /= math.fsum(row)
            if strided:
                backing = np.zeros((len(positions), len(targets)*2), dtype=dtype)
                backing[:, ::2] = values
                values = backing[:, ::2]
            for g, dense in [(fast, True), (slow, False)]:
                commit(g, begin, end, positions, targets, values, dense=dense)
            assert fast.rows == slow.rows
            assert fast.partial == slow.partial
            assert fast.accepted_through == slow.accepted_through
            assert fast.live_bytes == slow.live_bytes
            if end == 97:
                checkpoints = [fast.snapshot(), slow.snapshot()]
        for g, snap in zip((fast, slow), checkpoints):
            g.restore(snap)
        values = np.array([[.2, .3, .5]], dtype=dtype)
        for g, dense in [(fast, True), (slow, False)]:
            commit(g, 97, 128, [99], [0, 32, 96], values, 'output', dense)
        assert fast.rows == slow.rows and fast.partial == slow.partial
    finally:
        for snap in checkpoints:
            snap.close()
        fast.close(); slow.close()
    assert fast.budget.used == slow.budget.used == 0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dense_complete_source_topk_after_all_samples_not_before(dtype):
    targets = tuple(range(0, 40, 4))
    values = np.array([[.1, .175, .175, .175, .175, 0, 0, 0, 0, .2],
                       [.1, 0, 0, 0, 0, .175, .175, .175, .175, .2]], dtype=dtype)
    fast, slow = BlockGraph('a', 4), BlockGraph('b', 4)
    try:
        for g, dense in [(fast, True), (slow, False)]:
            commit(g, 0, 44, [40, 41], targets, values, dense=dense)
        assert fast.rows == slow.rows
        assert fast.rows[0].targets == (36, 0, 4, 8)
    finally:
        fast.close(); slow.close()


@pytest.mark.parametrize('values,positions,targets', [
    ([[math.nan, 0]], [8], [0, 4]), ([[math.inf, 0]], [8], [0, 4]),
    ([[-.1, 1.1]], [8], [0, 4]), ([[.2, .2]], [8], [0, 4]),
    ([[.5, .5]], [8], [0, 12]), ([[.5, .5]], [8], [0, 5]),
    ([[.5, .5], [.5, .5]], [8, 8], [0, 4]),
    ([[.5, .5]], [12], [0, 4]), ([[.5, .5]], [8], [0]),
])
def test_dense_invalid_input_is_atomic(values, positions, targets):
    graph = BlockGraph('s', 4)
    try:
        before = graph.binding, graph.bytes
        with pytest.raises(ValueError):
            graph.prepare_accept(0, 12, positions, targets, np.array(values, dtype='float32'))
        assert (graph.binding, graph.bytes) == before
        assert graph._pending_update is None and not graph.rows
    finally:
        graph.close()


def test_dense_legacy_replay_still_validates_mass_but_does_not_add_rows():
    graph = BlockGraph('s', 4)
    try:
        commit(graph, 0, 12, [8], [0, 4], np.array([[.5, .5]]))
        before = graph.binding, graph.rows, graph.bytes
        commit(graph, 0, 12, [8], [0, 4], np.array([[.5, .5]]))
        assert (graph.binding, graph.rows, graph.bytes) == before
        with pytest.raises(ValueError):
            graph.prepare_accept(0, 12, [8], [0, 4], np.array([[.5, math.nan]]))
        assert (graph.binding, graph.rows, graph.bytes) == before
    finally:
        graph.close()


def test_dense_capacity_is_reserved_before_native_work_and_abort_returns_every_byte(monkeypatch):
    graph = BlockGraph('s', 32, initial_position=4096)
    values = np.full((4, 128), 1/128, dtype='float32')
    positions, targets = [4096, 4100, 4128, 4130], list(range(0, 4096, 32))
    original = _native.prepare_block_graph
    calls = []
    def observed(*args):
        calls.append(True)
        return original(*args)
    monkeypatch.setattr(_native, 'prepare_block_graph', observed)
    try:
        before = graph.bytes, graph.binding
        limit = graph.max_bytes_per_session
        graph.max_bytes_per_session = graph.bytes
        with pytest.raises(BufferError):
            graph.prepare_accept(4096, 4140, positions, targets, values)
        assert not calls and (graph.bytes, graph.binding) == before
        graph.max_bytes_per_session = limit
        update = graph.prepare_accept(4096, 4140, positions, targets, values)
        assert calls and graph.bytes > before[0] and graph.binding == before[1]
        update.abort()
        assert (graph.bytes, graph.binding) == before
    finally:
        graph.close()


def test_dense_cpu_benchmark_compares_real_acceptance_and_releases_owners():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).parents[1]/'scripts/benchmark_graph_accept.py'
    spec = importlib.util.spec_from_file_location('dense_accept_benchmark', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.run(calls=3, active_tokens=128, chunk_tokens=32, block_tokens=4)
    assert report['bit_exact_equal'] and not report['gpu_used']
    assert report['shape'] == [8, 32]
    assert set(report['measurements']) == {'python_reference', 'native_dense'}


def test_native_buffer_boundary_rejects_malformed_partial_state():
    from types import SimpleNamespace
    partial = SimpleNamespace(source=4,targets=(0,4),sums=(1.,),samples=1,
                              prefill_samples=1,output_samples=0)
    with pytest.raises(ValueError, match='partial'):
        _native.prepare_block_graph([5], [0,4], np.array([[.5,.5]]),
                                   5,8,4,4,0.,True,partial)
