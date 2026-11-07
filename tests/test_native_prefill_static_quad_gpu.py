"""Actual quad static FA2: bitwise equal to the generic native statistics module.

Reuses the static-aligned GPU fixtures; only the static arm's capture variant
changes. The reference arm is the original generic KVMemAttention module.
"""
import pytest
import torch

import test_native_prefill_static_aligned_gpu as base

pytestmark = base.pytestmark


def quad_geometry(tmp_path, **kw):
    env = base.geometry(tmp_path, **kw)
    env.arms[True].bank.static_quad_enabled = True
    return env


def assert_quad_dispatched(env):
    stats = env.arms[True].bank.static_prefill_counts
    assert stats['static_dispatch_calls'] > 0
    keys = [key for key in env.arms[True].bank.prefill_modules if 'quad_store' in key]
    assert keys, 'Quad module was not selected'


@pytest.mark.parametrize('pattern', ['all', 'sparse', 'boundary', 'adjacent'])
def test_quad_partial_final_page_and_gqa_rows_match_bitwise(tmp_path, pattern):
    env = quad_geometry(tmp_path)
    boundary = env.wrapper._plan_info[3] // (env.heads // env.kvheads)
    # 'adjacent' samples both packed-row slots (r=0 and r=1) of many quads.
    rows = {'all': range(257), 'sparse': range(31, 257, 32),
            'boundary': [0, boundary - 1, boundary, boundary + 1, 256],
            'adjacent': [r for r in range(257) if r % 3 != 2]}[pattern]
    base.compare_pair(env, base.plans(env, [rows]))
    assert_quad_dispatched(env)


@pytest.mark.parametrize('heads,kvheads,qlen,page', [(24, 4, 2048, 16), (8, 8, 257, 32)])
def test_quad_2k_or_mha_and_larger_page_match_bitwise(tmp_path, heads, kvheads, qlen, page):
    env = quad_geometry(tmp_path, qlens=(qlen,), lengths=(4099,), heads=heads, kvheads=kvheads, page=page)
    base.compare_pair(env, base.plans(env, [range(31, qlen, 32)]))
    assert_quad_dispatched(env)


def test_quad_ragged_multiowner_with_unsampled_peer(tmp_path):
    env = quad_geometry(tmp_path, qlens=(129, 161, 1025), lengths=(2051, 3079, 5000))
    base.compare_pair(env, base.plans(env, [[0, 10, 31, 128], [], range(31, 1025, 32)]))
    assert_quad_dispatched(env)


def test_quad_actual_split_k_matches_bitwise(tmp_path):
    env = quad_geometry(tmp_path, qlens=(129,), lengths=(259,), page=16, split=1)
    assert env.wrapper._plan_info[14], 'Must exercise actual split-K attention'
    base.compare_pair(env, base.plans(env, [[0, 31, 63, 128]]))
    assert_quad_dispatched(env)


def test_quad_graph_capture_keeps_generic(tmp_path):
    env = quad_geometry(tmp_path)
    arm = env.arms[True]
    base.compare_pair(env, base.plans(env, [[0, 31, 256]]))
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): arm.execute()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): arm.execute()
    assert arm.bank.static_prefill_counts['generic_dispatch:cuda_graph_capture'] > 0
