"""KVMEM_NATIVE_PUBLISH_THREADS: independent publication rows, same bytes as serial."""
import struct
import threading

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem import _prefill_state_native as native

THREADS = [1, 2, 3, 8, 64]


def threaded(monkeypatch, threads):
    if threads == 1:
        monkeypatch.delenv('KVMEM_NATIVE_PUBLISH_THREADS', raising=False)
    else:
        monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', str(threads))


def raw_pages(rng, records, dtype, layers=4, tokens=32, heads=2, dim=64):
    if dtype == 'fp8':
        pages = rng.integers(0, 256, (records, layers, tokens, heads, dim), dtype=np.uint8)
        pages[pages == 127] = 126
        pages[pages == 255] = 254
        return pages
    data = torch.from_numpy(rng.standard_normal((records, layers, tokens, heads, dim)).astype(np.float32))
    return data.bfloat16().view(torch.uint8).numpy()


@pytest.mark.parametrize('dtype', ['fp8', 'bf16'])
@pytest.mark.parametrize('vectorized', [False, True])
def test_finalize_mean_rows_match_serial_bitwise(monkeypatch, dtype, vectorized):
    rng = np.random.default_rng(1201)
    pages = raw_pages(rng, 37, dtype)
    results = []
    for threads in THREADS:
        threaded(monkeypatch, threads)
        sums = np.full((37, 3, 2, 64), -7., np.float32)
        jobs = [(pages[i], sums[i], 32 - i % 5, dtype == 'fp8') for i in range(37)]
        used = _native.finalize_mean_batch(jobs, vectorized=vectorized)
        results.append((used, sums.view(np.uint32).copy()))
    for used, sums in results[1:]:
        assert used == results[0][0]
        np.testing.assert_array_equal(sums, results[0][1])


def test_finalize_mean_aliased_sums_keep_serial_order(monkeypatch):
    rng = np.random.default_rng(1202)
    pages = raw_pages(rng, 6, 'fp8')
    outcomes = []
    for threads in (1, 8):
        threaded(monkeypatch, threads)
        shared = np.zeros((3, 2, 64), np.float32)
        own = np.zeros((4, 3, 2, 64), np.float32)
        # Two rows write the same sum: the later row must win, as in the loop.
        jobs = [(pages[0], shared, 32, True), (pages[1], own[0], 32, True),
                (pages[2], shared, 31, True)] + [(pages[3+i], own[1+i], 32, True) for i in range(3)]
        _native.finalize_mean_batch(jobs, vectorized=True)
        outcomes.append((shared.view(np.uint32).copy(), own.view(np.uint32).copy()))
    np.testing.assert_array_equal(outcomes[0][0], outcomes[1][0])
    np.testing.assert_array_equal(outcomes[0][1], outcomes[1][1])


def scatter_jobs(rng, records, *, fused, layers=4, heads=2, dim=64, block=32):
    source_tokens = records * 24 + 40
    raw = rng.integers(0, 256, (layers, source_tokens, heads, dim), dtype=np.uint8)
    values = rng.integers(0, 256, raw.shape, dtype=np.uint8)
    dst_raw = np.full((records, layers, block, heads, dim), 3, np.uint8)
    dst_values = np.full(dst_raw.shape, 5, np.uint8)
    sums = np.full((records, layers - 1, heads, dim), -1., np.float32)
    jobs = []
    for i in range(records):
        count = block if fused else 1 + (i * 7) % block
        offset = block - count
        jobs.append((raw, values, dst_raw[i], dst_values[i], None, sums[i], i * 24, offset, count, True)
                    + ((True,) if fused else ()))
    return jobs, (dst_raw, dst_values, sums)


@pytest.mark.parametrize('fused', [False, True])
def test_scatter_rows_match_serial_bitwise(monkeypatch, fused):
    outcomes = []
    for threads in THREADS:
        threaded(monkeypatch, threads)
        jobs, outputs = scatter_jobs(np.random.default_rng(1203), 41, fused=fused)
        result = _native.scatter_history_batch(jobs)
        outcomes.append((result, [o.view(np.uint8).copy() for o in outputs]))
    for result, outputs in outcomes[1:]:
        assert result == outcomes[0][0]
        for a, b in zip(outputs, outcomes[0][1]):
            np.testing.assert_array_equal(a, b)


def test_scatter_overlapping_destinations_keep_serial_last_write(monkeypatch):
    outcomes = []
    for threads in (1, 8):
        threaded(monkeypatch, threads)
        jobs, outputs = scatter_jobs(np.random.default_rng(1204), 9, fused=False)
        jobs = list(jobs)
        jobs[7] = (*jobs[7][:2], jobs[2][2], jobs[2][3], *jobs[7][4:])  # Same page, later write.
        _native.scatter_history_batch(jobs)
        outcomes.append([o.copy() for o in outputs])
    for a, b in zip(*outcomes):
        np.testing.assert_array_equal(a, b)


def state_bits(state):
    return [(key, struct.pack('<dd', *pair)) for key, pair in state.values().items()]


def fold_inputs(rng, layers=16, rows=6, pages=67):
    states = [native.State.from_values({9000: [-0., 1.], layer: [.123456789, .875]})
              for layer in range(layers)]
    logical = [rng.integers(0, 90, size=pages).tolist() for _ in range(layers)]
    masses = rng.random((layers, rows, pages))
    masses /= masses.sum(axis=2, keepdims=True)
    decays = [.5 + layer / 64 for layer in range(layers)]
    return states, decays, list(range(31, 32 * rows, 32)), list(range(0, pages * 32, 32)), logical, masses


def test_fold_layers_match_serial_bitwise(monkeypatch):
    outcomes = []
    for threads in THREADS:
        threaded(monkeypatch, threads)
        states, decays, rows, starts, logical, masses = fold_inputs(np.random.default_rng(1205))
        updated = native.committed_batch(states, decays, rows, starts, logical, 1792, masses)
        outcomes.append([state_bits(s) for s in updated])
    assert all(o == outcomes[0] for o in outcomes[1:])


@pytest.mark.parametrize('threads', THREADS)
@pytest.mark.parametrize('fault', ['nan', 'history', 'both'])
def test_fold_failure_matches_serial_and_returns_nothing(monkeypatch, threads, fault):
    states, decays, rows, starts, logical, masses = fold_inputs(np.random.default_rng(1206))
    before = [state_bits(s) for s in states]
    if fault in ('nan', 'both'):
        masses[5, 0, 0] = np.nan
    if fault in ('history', 'both'):
        logical[3] = logical[3][:-1]
    outcomes = []
    for count in (1, threads):
        threaded(monkeypatch, count)
        with pytest.raises((RuntimeError, ValueError)) as error:
            native.committed_batch(states, decays, rows, starts, logical, 1792, masses)
        outcomes.append((type(error.value), str(error.value)))
    assert outcomes[0] == outcomes[1]
    assert [state_bits(s) for s in states] == before


@pytest.mark.parametrize('value', ['0', '65', 'x', '2.5', '-1'])
def test_invalid_thread_count_is_rejected(monkeypatch, value):
    monkeypatch.setenv('KVMEM_NATIVE_PUBLISH_THREADS', value)
    states, decays, rows, starts, logical, masses = fold_inputs(np.random.default_rng(1207))
    with pytest.raises(ValueError, match='KVMEM_NATIVE_PUBLISH_THREADS'):
        native.committed_batch(states, decays, rows, starts, logical, 1792, masses)


def test_concurrent_callers_never_wait_for_the_busy_pool_and_stay_exact(monkeypatch):
    threaded(monkeypatch, 1)
    reference_jobs, reference = scatter_jobs(np.random.default_rng(1208), 33, fused=True)
    _native.scatter_history_batch(reference_jobs)
    states, decays, rows, starts, logical, masses = fold_inputs(np.random.default_rng(1209))
    expected_fold = [state_bits(s) for s in
                     native.committed_batch(states, decays, rows, starts, logical, 1792, masses)]
    threaded(monkeypatch, 4)
    failures = []

    def scatter_worker():
        for _ in range(20):
            jobs, outputs = scatter_jobs(np.random.default_rng(1208), 33, fused=True)
            _native.scatter_history_batch(jobs)
            if not all(np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in zip(outputs, reference)):
                failures.append('scatter')

    def fold_worker():
        for _ in range(20):
            updated = native.committed_batch(states, decays, rows, starts, logical, 1792, masses)
            if [state_bits(s) for s in updated] != expected_fold:
                failures.append('fold')

    workers = [threading.Thread(target=scatter_worker) for _ in range(3)] + \
              [threading.Thread(target=fold_worker) for _ in range(2)]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=120)
    assert not any(w.is_alive() for w in workers) and not failures
