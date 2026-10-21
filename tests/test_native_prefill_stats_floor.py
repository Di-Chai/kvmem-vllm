"""Pure stdlib checks of the offline component protocol, without importing CUDA."""
import ast
from collections import Counter
import itertools
from pathlib import Path
import statistics

import pytest


SOURCE = Path(__file__).resolve().parents[1] / 'scripts/benchmark_native_prefill_stats_floor.py'


def helpers():
    tree = ast.parse(SOURCE.read_text())
    names = {'order_for', 'validate_kernel_names', 'summarize'}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(itertools=itertools, statistics=statistics,
                     MODES=('native', 'generic', 'static_aligned'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


def test_balanced_six_orders_cover_positions_and_directed_predecessors():
    f = helpers()['order_for']
    orders = [f(i) for i in range(6)]
    assert len(set(orders)) == 6
    for position in range(3):
        assert set(Counter(order[position] for order in orders).values()) == {2}
    edges = Counter(pair for order in orders for pair in zip(order, order[1:]))
    assert len(edges) == 6 and set(edges.values()) == {2}
    assert [f(i) for i in range(6, 12)] == orders


def test_native_audit_requires_real_default_attention_not_empty_stats_gate():
    check = helpers()['validate_kernel_names']
    original = ['flashinfer::BatchPrefillWithPagedKVCacheKernel<flashinfer::DefaultAttention<false>>'] * 3
    check('native', original, original)
    for forbidden in ('KVMemAttention', 'KVMemStaticAlignedAttention', '_clear_lse', '_row_lse', '_segment_scores', '_map_pieces'):
        with pytest.raises(RuntimeError):
            check('native', original, original + [forbidden])
    with pytest.raises(RuntimeError):
        check('native', ['KVMemAttention'] * 3, ['KVMemAttention'] * 3)
    with pytest.raises(RuntimeError):
        check('native', original[:2], original[:2])


@pytest.mark.parametrize('mode,expected', [('generic', 'KVMemAttention'),
                                         ('static_aligned', 'KVMemStaticAlignedAttention')])
def test_statistics_profile_checks_actual_selected_specialization(mode, expected):
    check = helpers()['validate_kernel_names']
    check(mode, [expected] * 3, [expected] * 3)
    with pytest.raises(RuntimeError):
        check(mode, ['flashinfer::DefaultAttention<'] * 3, ['flashinfer::DefaultAttention<'] * 3)


def test_complete_component_and_kernel_overhead_are_separate():
    summarize = helpers()['summarize']
    samples = [dict(mode=mode, wall_ms=wall, cuda_interval_ms=wall - .5)
               for _ in range(2)
               for mode, wall in [('native', 10.), ('generic', 14.), ('static_aligned', 12.)]]
    profiles = {mode: [dict(fa2_kernel_ms=value)] for mode, value in
                [('native', 9.), ('generic', 12.), ('static_aligned', 10.)]}
    summary, kernels, delta = summarize(samples, profiles)
    assert summary['native']['wall_ms'] == 10.
    assert kernels['static_aligned'] == 10.
    assert delta['generic'] == dict(complete_wall_ms=4., complete_cuda_interval_ms=4., fa2_kernel_ms=3.)
    assert delta['static_aligned'] == dict(complete_wall_ms=2., complete_cuda_interval_ms=2., fa2_kernel_ms=1.)


def test_native_branch_does_not_arm_disarm_or_call_action():
    # The executable kernel audit supplements this narrow source guard. It
    # prevents accidentally redefining the native baseline as stats disabled
    # on the device while still paying the instrumented kernel's overhead.
    tree = ast.parse(SOURCE.read_text())
    setup = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'setup')
    run = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == 'run')
    branch = next(node for node in ast.walk(run) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "mode == 'native'")
    calls = [ast.unparse(node.func) for statement in branch.body for node in ast.walk(statement)
             if isinstance(node, ast.Call)]
    assert 'original_attention' in calls
    assert all(not name.startswith(('bank.', 'action.', 'layer.')) for name in calls)
