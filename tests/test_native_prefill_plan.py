"""KVMEM_NATIVE_PREFILL_PLAN: native window segments / logical plan checks equal the Python helpers."""
import random
from dataclasses import replace

import pytest

from vllm_kvmem.action_attention import window_segments
from vllm_kvmem.logical_action import LogicalScorePlan, plan_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.prefill_planning import PrefillPlanFrame, native_plan_function
from vllm_kvmem.request_plan import RequestPlan
from types import SimpleNamespace as NS


def _native():
    try:
        from vllm_kvmem import _native
    except ImportError:
        pytest.skip('native extension not built')
    if not hasattr(_native, 'frame_window_segments'):
        pytest.skip('native extension lacks frame_window_segments')
    return _native


def outcome(function, *args):
    try:
        return 'ok', function(*args)
    except Exception as error:      # noqa: BLE001 - compare the exact failure
        return type(error), str(error)


def records_for(rng, n, block=32, partial_tail=True, gap=False):
    out, position, begin = [], 0, rng.randrange(0, 4) * block
    for i in range(n):
        size = block if (i < n - 1 or not partial_tail) else rng.randrange(1, block + 1)
        out.append((rng.randrange(100), rng.randrange(9), begin + rng.randrange(0, 3) * 0, size, position))
        position += size + (1 if gap and i == n // 2 else 0)
        begin += block * rng.randrange(1, 4)
    return tuple(out)


@pytest.mark.parametrize('seed', range(40))
def test_segments_equal_python(seed):
    native = _native()
    rng = random.Random(seed)
    block = rng.choice((16, 32, 64))
    records = records_for(rng, rng.randrange(0, 300), block, gap=seed % 7 == 3)
    end = records[-1][4] + records[-1][3] if records else 0
    local_begin = rng.choice((end, max(0, end - rng.randrange(0, 2 * block + 1)), end + 1, 0))
    history_begin = local_begin + rng.choice((0, rng.randrange(0, 1 << 20), -rng.randrange(1, 3 * block)))
    count = rng.choice((0, 1, 1024, rng.randrange(0, 5000), -3))
    want = outcome(window_segments, records, history_begin, local_begin, count, block)
    got = outcome(native.frame_window_segments, records, history_begin, local_begin, count, block)
    assert got == want
    if got[0] == 'ok':
        assert all(type(x) is int for part in got[1] for x in part) and all(type(p) is list for p in got[1])


def test_out_of_domain_inputs_fall_back():
    native = _native()
    records = ((0, 0, 0, 32, 0),)
    assert native.frame_window_segments(records, 1 << 70, 32, 8, 32) is None
    assert native.frame_window_segments(list(records), 0, 32, 8, 32) is None
    assert native.frame_window_segments(records, 0, 32, 8, 0) is None
    assert native.frame_window_segments(((0, 0, True, 32, 0),), 0, 32, 8, 32) is None
    assert native.frame_window_segments(([0, 0, 0, 32, 0],), 0, 32, 8, 32) is None
    # placeholder promise keys (1 << 60) stay inside the exact domain
    promised = tuple((-2, -2, (1 << 60) + i * 32, 32, i * 32) for i in range(4))
    assert native.frame_window_segments(promised, 500, 128, 64, 32) == window_segments(promised, 500, 128, 64, 32)
    # negative history begins use Python's floor modulo
    neg = ((0, 0, -45, 32, 0),)
    assert native.frame_window_segments(neg, 40, 32, 40, 32) == window_segments(neg, 40, 32, 40, 32)


def span(i, size=1024, base=9000, local=4096):
    return LogicalPrefillSpan(i, base + i * size, local + i * size, size, i * size)


@pytest.mark.parametrize('case', ['valid', 'rows-tuple', 'rows-bool', 'starts-list', 'block-bool', 'block-zero',
                                  'row-range', 'row-order', 'empty', 'lengths', 'gap', 'count0', 'count-big',
                                  'hist-neg', 'hist-mod', 'cover', 'huge'])
def test_plan_check_equals_python(case, monkeypatch):
    _native()
    s = span(1)
    rows = tuple(range(31, 1024, 32))
    starts = tuple(range(0, s.local_end, 32))
    counts = (32,) * len(starts)
    history = tuple(range(0, 32 * len(starts), 32))
    block = 32
    if case == 'rows-tuple': rows = list(rows)
    if case == 'rows-bool': rows = (True,) + rows[1:]
    if case == 'starts-list': starts = list(starts)
    if case == 'block-bool': block = True
    if case == 'block-zero': block = 0
    if case == 'row-range': rows = rows[:-1] + (1024,)
    if case == 'row-order': rows = (rows[1], rows[0]) + rows[2:]
    if case == 'empty': starts = counts = history = ()
    if case == 'lengths': history = history[:-1]
    if case == 'gap': starts = starts[:5] + (starts[5] + 1,) + starts[6:]
    if case == 'count0': counts = (0,) + counts[1:]
    if case == 'count-big': counts = (33,) + counts[1:]
    if case == 'hist-neg': history = (-32,) + history[1:]
    if case == 'hist-mod': history = (16,) + history[1:]
    if case == 'cover': starts, counts, history = starts[:-1], counts[:-1], history[:-1]
    if case == 'huge': history = (1 << 80,) + history[1:]
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '0')
    want = outcome(LogicalScorePlan, s, rows, starts, counts, history, block)
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '1')
    got = outcome(LogicalScorePlan, s, rows, starts, counts, history, block)
    assert (got[0], got[1] if got[0] != 'ok' else got[1]) == want


@pytest.mark.parametrize('chunks', [2, 3, 4])
@pytest.mark.parametrize('seed', range(5))
def test_logical_planning_end_to_end(chunks, seed, monkeypatch):
    _native()
    rng = random.Random(seed)
    B, C = 32, 1024
    local_begin = rng.randrange(100, 1100) * B + rng.choice((0, 0, 7))
    history_begin = local_begin + rng.randrange(0, 200000)
    n = local_begin // B
    records = tuple((1000 + i, 1, rng.randrange(0, 10000) * B, B, i * B) for i in range(n))
    if local_begin % B:
        records += ((7, 1, history_begin - local_begin % B, local_begin % B, n * B),)
    options = NS(prefill_attention_interval=1024, prefill_attention_stride=32,
                 memory=NS(select_budget=36864, gen_budget=4096, block_tokens=B))
    frame = PrefillPlanFrame('a', NS(options=options), (1, 0, 0, 0, None), history_begin, local_begin,
                             history_begin - rng.choice((0, 512, 2048)), records)
    spans = tuple(LogicalPrefillSpan(i, history_begin + i * C, local_begin + i * C, C, i * C) for i in range(chunks))
    group = LogicalPrefillGroup(request_id='r', owner_epoch=(1, 0, 0), scheduler_step=7,
        offset=history_begin - local_begin, prompt_tokens=262144, capacity=40960 + C * chunks, spans=spans,
        request_plan=RequestPlan(None, block_tokens=B, gen_budget=4096))
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '0')
    want = outcome(plan_logical_scores, group, frame)
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '1')
    got = outcome(plan_logical_scores, group, frame)
    assert got == want
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '0')
    assert frame.extend(history_begin, local_begin, C, B) == (
        monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '1') or frame.extend(history_begin, local_begin, C, B))


def test_flag_is_validated(monkeypatch):
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', 'yes')
    with pytest.raises(ValueError):
        native_plan_function('frame_window_segments')
    monkeypatch.setenv('KVMEM_NATIVE_PREFILL_PLAN', '0')
    assert native_plan_function('frame_window_segments') is None
