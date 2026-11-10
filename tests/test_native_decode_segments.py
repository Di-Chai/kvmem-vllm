"""KVMEM_NATIVE_DECODE_SEGMENTS: native segment_identities equals the Python helper.

Results and failures (type and message) must be identical with the flag on and off.
"""
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem import layerwise_decode
from vllm_kvmem.action_attention import window_segments
from vllm_kvmem.layerwise_pressure import extend_records


def _native():
    from vllm_kvmem import _native
    if not hasattr(_native, 'decode_segment_identities'):
        pytest.skip('native extension lacks decode_segment_identities')
    return _native


def outcome(fn, *args):
    try:
        return 'ok', fn(*args)
    except Exception as error:  # noqa: BLE001 - compare the exact failure
        return type(error), str(error)


def both(monkeypatch, *args):
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_SEGMENTS', '0')
    want = outcome(layerwise_decode.segment_identities, *args)
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_SEGMENTS', '1')
    got = outcome(layerwise_decode.segment_identities, *args)
    return want, got


def window(rng, blocks, tail_tokens, partial=False):
    """A layer map of selected blocks (+ optional partial block), extended by
    accepted ordinary input exactly as LayerwisePrefillPressure.records does."""
    records, pos = [], 0
    chosen = sorted(rng.sample(range(blocks * 4), blocks))
    for k, b in enumerate(chosen):
        size = 32 if not (partial and k == len(chosen) - 1) else rng.randrange(1, 32)
        records.append((rng.randrange(64), rng.randrange(3), b * 32, size, pos))
        pos += size
    origin = chosen[-1] * 32 + records[-1][3] + rng.choice((0, 0, 32 * rng.randrange(1, 40)))
    if partial and origin % 32 == 0:
        origin += records[-1][3]
    return extend_records(records, origin, origin + tail_tokens), origin + tail_tokens


def columns(rng, records, begin, local, count, mode):
    ls, lc, lh = window_segments(records, begin, local, count, 32)
    starts, counts = [], []
    i = 0
    while i < len(ls):
        s, c = ls[i], lc[i]
        # Merge following segments of the same logical block (canonical partial tail).
        while mode != 'split' and i + 1 < len(ls) and lh[i + 1] == lh[i] and rng.random() < .7:
            i += 1
            c += lc[i]
        if mode == 'split' and c > 1 and rng.random() < .3:
            cut = rng.randrange(1, c)
            starts += [s, s + cut]; counts += [cut, c - cut]
        else:
            starts.append(s); counts.append(c)
        i += 1
    return starts, counts, lh


@pytest.mark.parametrize('seed', range(60))
def test_valid_windows_equal_python(seed, monkeypatch):
    _native()
    rng = random.Random(seed)
    records, end = window(rng, rng.randrange(1, 1300), rng.randrange(0, 5000), partial=seed % 3 == 0)
    local = sum(r[3] for r in records)
    count = rng.randrange(1, 6)
    begin = end
    starts, counts, _ = columns(rng, records, begin, local, count, 'split' if seed % 4 == 1 else 'merge')
    kind = seed % 5
    if kind == 1:
        starts, counts = np.asarray(starts, np.int64), np.asarray(counts, np.int64)
    elif kind == 2:
        starts, counts = np.asarray(starts, np.int32), np.asarray(counts, np.int32)
    elif kind == 3:
        starts, counts = tuple(starts), tuple(counts)
    elif kind == 4:
        starts, counts = np.asarray(starts, np.uint32), np.asarray(counts, np.uint32)
    want, got = both(monkeypatch, records, begin, local, count, starts, counts)
    assert want[0] == 'ok' and got == want
    assert all(type(x) is int for x in got[1])
    # The native helper itself takes the fast path on these inputs.
    assert _native().decode_segment_identities(records, begin, local, count, starts, counts, 32) == want[1]


@pytest.mark.parametrize('case', ['cross_block', 'gap', 'short_total', 'long_total', 'zero', 'negative',
                                  'length', 'gap_records', 'origin', 'uint64', 'numpy_scalars', 'float',
                                  'numpy_records', 'list_records_items', 'huge', 'two_d'])
def test_failures_and_fallbacks_equal_python(case, monkeypatch):
    _native()
    rng = random.Random(7)
    records, end = window(rng, 40, 300, partial=True)
    local = sum(r[3] for r in records)
    count, begin = 3, end
    starts, counts, lh = columns(rng, records, begin, local, count, 'merge')
    if case == 'cross_block':
        k = next(i for i in range(len(starts) - 1) if lh is not None)
        starts = starts[:k] + starts[k + 1:]
        counts = counts[:k] + [counts[k] + counts[k + 1]] + counts[k + 2:]
    elif case == 'gap':
        starts[3] += 1
    elif case == 'short_total':
        counts[-1] -= 1
    elif case == 'long_total':
        counts[-1] += 1
    elif case == 'zero':
        counts[2] = 0
    elif case == 'negative':
        counts[2] = -1
    elif case == 'length':
        counts = counts[:-1]
    elif case == 'gap_records':
        r = records[5]; records[5] = (r[0], r[1], r[2], r[3], r[4] + 1)
    elif case == 'origin':
        begin = local - 10 ** 6
    elif case == 'uint64':
        starts, counts = np.asarray(starts, np.uint64), np.asarray(counts, np.uint64)
    elif case == 'numpy_scalars':
        starts = [np.int64(x) for x in starts]
    elif case == 'float':
        counts = np.asarray(counts, np.float64)
    elif case == 'numpy_records':
        r = records[0]; records[0] = (r[0], r[1], np.int64(r[2]), r[3], r[4])
    elif case == 'list_records_items':
        records[0] = list(records[0])
    elif case == 'huge':
        begin = 1 << 70
    elif case == 'two_d':
        starts = np.asarray([starts], np.int64)
    want, got = both(monkeypatch, records, begin, local, count, starts, counts)
    assert got == want


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_SEGMENTS', 'yes')
    with pytest.raises(ValueError):
        layerwise_decode.native_segments_enabled()
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_SEGMENTS', '0')
    assert not layerwise_decode.native_segments_enabled()


# KVMEM_NATIVE_DECODE_SPARSE_MASS: prepare_decode_attention_sparse == prepare_decode_attention.

def _sparse():
    from vllm_kvmem import _native
    if not hasattr(_native, 'prepare_decode_attention_sparse'):
        pytest.skip('native extension lacks prepare_decode_attention_sparse')
    return _native


def _mass_case(rng, dtype, rows=5, cols=600, zero_frac=.3):
    targets = [32 * rng.randrange(0, cols) for _ in range(cols)]   # duplicates on purpose
    m = np.zeros((rows, cols), dtype)
    for r in range(rows):
        v = np.array([0. if rng.random() < zero_frac else rng.random() for _ in range(cols)])
        v[rng.randrange(cols)] += 1e-3
        m[r] = (v / v.sum()).astype(dtype)
    old = {}
    for _ in range(rng.randrange(0, 900)):
        old[32 * rng.randrange(0, 3 * cols)] = rng.random() * 3
    if rng.random() < .3 and old:
        k = next(iter(old)); old[k] = int(old[k])          # a non-float old value
    indices = sorted(rng.sample(range(rows), rng.randrange(0, rows + 1)))
    return old, indices, targets, m


def _call(fn, old, samples, indices, targets, m, cap=16777216):
    try:
        mass, n = fn(old, samples, indices, targets, m, cap)
        return 'ok', list(mass.items()), [type(v) for v in mass.values()], n, old
    except Exception as error:  # noqa: BLE001
        return type(error), str(error)


@pytest.mark.parametrize('seed', range(40))
def test_sparse_mass_equals_prepare(seed):
    native = _sparse()
    rng = random.Random(seed)
    old, indices, targets, m = _mass_case(rng, np.float32 if seed % 2 else np.float64)
    a = _call(native.prepare_decode_attention, dict(old), 3, indices, targets, m)
    b = _call(native.prepare_decode_attention_sparse, dict(old), 3, indices, targets, m)
    assert a[0] == 'ok' and a == b


@pytest.mark.parametrize('case', ['sum', 'negative', 'nan', 'order', 'duplicate', 'row_range', 'shape',
                                  'dtype', 'capacity'])
def test_sparse_mass_failures_equal_prepare(case):
    native = _sparse()
    rng = random.Random(11)
    old, _, targets, m = _mass_case(rng, np.float32)
    indices, cap = [0, 2, 4], 16777216
    if case == 'sum': m[2] *= 1.5
    elif case == 'negative': m[0, 3] = -1e-3
    elif case == 'nan': m[4, 7] = np.nan
    elif case == 'order': indices = [2, 0]
    elif case == 'duplicate': indices = [1, 1]
    elif case == 'row_range': indices = [0, 9]
    elif case == 'shape': targets = targets[:-1]
    elif case == 'dtype': m = m.astype(np.float16)
    elif case == 'capacity': cap = 4096
    a = _call(native.prepare_decode_attention, dict(old), 0, indices, targets, m, cap)
    b = _call(native.prepare_decode_attention_sparse, dict(old), 0, indices, targets, m, cap)
    assert a[0] != 'ok' and a == b


def test_sparse_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_SPARSE_MASS', 'on')
    with pytest.raises(ValueError):
        layerwise_decode.sparse_mass_enabled()


# KVMEM_NATIVE_DECODE_MASS_MAP: DecodeMassMap / NativeDecodeHistory == the dict path.

def _map():
    from vllm_kvmem import _native
    if not hasattr(_native, 'prepare_decode_attention_map'):
        pytest.skip('native extension lacks DecodeMassMap')
    return _native


def _as_items(mass):
    d = mass.to_dict() if hasattr(mass, 'to_dict') else mass
    return list(d.items()), [type(v) for v in d.values()]


@pytest.mark.parametrize('seed', range(30))
def test_mass_map_chain_equals_dict_chain(seed):
    native = _map()
    rng = random.Random(seed)
    old, _, _, _ = _mass_case(rng, np.float32)
    a_mass, a_n = dict(old), 0
    b_mass, b_n = dict(old), 0          # first map call converts the dict
    for step in range(6):
        _, indices, targets, m = _mass_case(rng, np.float32 if (seed + step) % 2 else np.float64)
        a_mass, a_n = native.prepare_decode_attention(a_mass, a_n, indices, targets, m, 16777216)
        b_mass, b_n = native.prepare_decode_attention_map(b_mass, b_n, indices, targets, m, 16777216)
        assert _as_items(a_mass) == _as_items(b_mass) and a_n == b_n
        assert len(b_mass) == len(a_mass)


@pytest.mark.parametrize('case', ['sum', 'negative', 'nan', 'order', 'duplicate', 'row_range', 'shape',
                                  'dtype', 'capacity'])
def test_mass_map_failures_equal_prepare(case):
    native = _map()
    rng = random.Random(13)
    old, _, targets, m = _mass_case(rng, np.float32)
    indices, cap = [0, 2, 4], 16777216
    if case == 'sum': m[2] *= 1.5
    elif case == 'negative': m[0, 3] = -1e-3
    elif case == 'nan': m[4, 7] = np.nan
    elif case == 'order': indices = [2, 0]
    elif case == 'duplicate': indices = [1, 1]
    elif case == 'row_range': indices = [0, 9]
    elif case == 'shape': targets = targets[:-1]
    elif case == 'dtype': m = m.astype(np.float16)
    elif case == 'capacity': cap = 4096
    base = native.DecodeMassMap.from_dict(old)
    a = _call(native.prepare_decode_attention, dict(old), 0, indices, targets, m, cap)
    for source in (dict(old), base):
        try:
            native.prepare_decode_attention_map(source, 0, indices, targets, m, cap)
            b = ('ok',)
        except Exception as error:  # noqa: BLE001
            b = (type(error), str(error))
        assert a[0] != 'ok' and b == a
    assert base.to_dict() == old          # a failed prepare never mutates its input


def _pending(rng, begin, end, count=5, interval=16):
    rows = list(range(count))
    return NS(candidate_count=count, history_begin=begin, rows=rows,
              state=NS(options=NS(action_interval=interval, action_commit_max_bytes=16777216)))


def test_native_decode_history_matches_dict_history(monkeypatch):
    _map()
    from vllm_kvmem.layerwise_decode import DecodeHistory, NativeDecodeHistory
    rng = random.Random(5)
    plain = DecodeHistory({32 * k: rng.random() for k in range(50)}, 2, 0, 1)
    chains = {}
    for flag in ('0', '1'):
        monkeypatch.setenv('KVMEM_NATIVE_DECODE_MASS_MAP', flag)
        h, begin, r = plain, 0, random.Random(9)
        for step in range(40):
            accepted = r.randrange(0, 6)
            cols = 300
            logical = [32 * r.randrange(0, 400) for _ in range(cols)]
            m = r.random() * np.ones((5, cols), np.float32)
            m = (m * np.asarray([[r.random() for _ in range(cols)] for _ in range(5)], np.float32))
            m /= m.sum(1, keepdims=True)
            pend = _pending(r, begin, begin + accepted)
            h = h.prepare(pend, accepted, logical, m)
            begin += accepted
            if step == 20:      # GC pruning / restore rebuild a plain dict history mid-chain
                h = DecodeHistory({k: v for k, v in h.mass.items() if k % 64}, h.samples, h.end, h.chunks)
        chains[flag] = h
    a, b = chains['0'], chains['1']
    assert type(a) is DecodeHistory and type(b) is NativeDecodeHistory
    assert list(a.mass.items()) == list(b.mass.items()) and all(type(v) is float for v in b.mass.values())
    assert (a.samples, a.end, a.chunks) == (b.samples, b.end, b.chunks)
    assert a.scores() == b.scores() and b == a and a == b


def test_mass_map_flag_is_validated(monkeypatch):
    from vllm_kvmem import layerwise_decode
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_MASS_MAP', 'true')
    with pytest.raises(ValueError):
        layerwise_decode.mass_map_enabled()


# KVMEM_NATIVE_DECODE_LAYER_MAP: fused identities from the frozen pressure frame.

def _controller(rng, layers=16, blocks=300, partial=True):
    from vllm_kvmem.layerwise_pressure import LayerwisePrefillPressure
    c = LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    c.names = [f'l{i}' for i in range(layers)]
    maps, total = [], None
    last_block = rng.randrange(blocks, blocks * 4)
    tail = rng.randrange(1, 32) if partial else 32
    for _ in range(layers):
        chosen = sorted(rng.sample(range(last_block), blocks - 1)) + [last_block]
        recs, pos = [], 0
        for k, b in enumerate(chosen):
            size = tail if k == len(chosen) - 1 else 32
            recs.append((rng.randrange(64), rng.randrange(3), b * 32, size, pos))
            pos += size
        maps.append(tuple(recs))
        total = pos
    origin = last_block * 32 + tail          # continues the (partial) last block
    c.maps = {'r': (origin, tuple(maps))}
    c.decode_histories = {}
    return c, origin, total


def _python_identities(c, begin, local, count, starts, counts):
    return [layerwise_decode._segment_identities_python(c.records('r', None, begin, local, layer),
                                                        begin, local, count, starts, counts)
            for layer in range(len(c.names))]


@pytest.mark.parametrize('seed', range(25))
def test_fused_layer_map_equals_per_layer_python(seed):
    native = _map()
    if not hasattr(native, 'decode_identities_from_map'):
        pytest.skip('native extension lacks decode_identities_from_map')
    rng = random.Random(seed)
    c, origin, total = _controller(rng, partial=seed % 3 != 0)
    begin = origin + rng.randrange(0, 4000)
    local = total + (begin - origin)
    count = rng.randrange(1, 6)
    recs0 = c.records('r', None, begin, local, 0)
    starts, counts, _ = columns(rng, recs0, begin, local, count, 'merge')
    # columns must partition every layer's window; use the shared tail-only split
    want = outcome(_python_identities, c, begin, local, count, starts, counts)
    got = layerwise_decode._fused_identities(c, 'r', begin, local, count, starts, counts)
    if want[0] == 'ok':
        assert got == want[1]
    else:
        assert got is None


@pytest.mark.parametrize('case', ['length', 'backwards', 'no_frame', 'numpy_ints', 'gap_column', 'list_records'])
def test_fused_layer_map_falls_back_with_identical_errors(case, monkeypatch):
    _map()
    rng = random.Random(3)
    c, origin, total = _controller(rng, layers=4, blocks=60)
    begin = origin + 100
    local, count = total + 100, 3
    recs0 = c.records('r', None, begin, local, 0)
    starts, counts, _ = columns(rng, recs0, begin, local, count, 'merge')
    if case == 'length': local += 1
    elif case == 'backwards': begin = origin - 1
    elif case == 'no_frame': c.maps = {}
    elif case == 'numpy_ints':
        o, layers = c.maps['r']; r = layers[1][0]
        c.maps['r'] = (o, layers[:1] + ((r[:2] + (np.int64(r[2]),) + r[3:],) + layers[1][1:],) + layers[2:])
    elif case == 'gap_column': starts = list(starts); starts[2] += 1
    elif case == 'list_records':
        o, layers = c.maps['r']; c.maps['r'] = (o, tuple(list(l) for l in layers))
    pend = NS(history_begin=begin, candidate_count=count)
    masses = np.zeros((4, 1, 1), np.float32)
    results = {}
    for flag in ('0', '1'):
        monkeypatch.setenv('KVMEM_NATIVE_DECODE_LAYER_MAP', flag)
        c.decode_histories = {}
        if case == 'no_frame':
            results[flag] = 'skip'
            continue
        results[flag] = outcome(layerwise_decode.attach_packet, c, pend, 'r', None, starts,
                                np.asarray([*counts[:-1], counts[-1]]), masses)
        if results[flag][0] == 'ok':
            results[flag] = ('ok', pend.layerwise_decode[2], pend.history)
    assert results['0'] == results['1']


def test_fused_layer_map_cache_follows_new_frames(monkeypatch):
    _map()
    monkeypatch.setenv('KVMEM_NATIVE_DECODE_LAYER_MAP', '1')
    rng = random.Random(21)
    c, origin, total = _controller(rng, layers=3, blocks=50)
    for _ in range(3):
        begin = origin + rng.randrange(0, 300)
        local, count = total + (begin - origin), 2
        recs0 = c.records('r', None, begin, local, 0)
        starts, counts, _ = columns(rng, recs0, begin, local, count, 'merge')
        assert layerwise_decode._fused_identities(c, 'r', begin, local, count, starts, counts) == \
            _python_identities(c, begin, local, count, starts, counts)
        c2, origin, total = _controller(rng, layers=3, blocks=50)   # a refresh installs a new frame
        c.maps = c2.maps
