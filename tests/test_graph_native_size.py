"""KVMEM_GRAPH_NATIVE_SIZE: _native.graph_size equals block_graph._size exactly."""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytest

from vllm_kvmem import _native; from vllm_kvmem.serving import block_graph
from vllm_kvmem.serving.block_graph import GraphPartial

pytestmark = pytest.mark.skipif(not hasattr(_native, 'graph_size'), reason='Requires the rebuilt extension')


@dataclass
class Holder:
    a: object
    b: object


def python_size(value, seen=None):
    import os
    old = os.environ.pop('KVMEM_GRAPH_NATIVE_SIZE', None)
    try:
        return block_graph._size(value, seen)
    finally:
        if old is not None:
            os.environ['KVMEM_GRAPH_NATIVE_SIZE'] = old


def cases():
    rng = np.random.default_rng(3)
    targets = tuple(int(x) for x in rng.integers(0, 10**6, 3000))
    sums = tuple(float(x) for x in rng.random(3000))
    shared = 123456789
    point = namedtuple('P', 'x y')
    return [
        GraphPartial(5, targets, sums, 17, 3, 4),
        GraphPartial(5, (1, 2, 3, 1, 2), (0.5, 0.5, 0.5), 0),
        (shared, shared, [shared, {shared: shared}], (1.5,) * 10),
        {'a': [1, 2, (3, 4)], 'b': {'c': (5.0, 6.0)}, 7: Holder([1, 2], {'x': (9,)})},
        [point(1, 2), point(3, 4), [], (), {}],
        (('spans', ((0, 32), (32, 64))), ('ends', (True, False)), {'source': 'renderer', 'v': 1}),
    ]


@pytest.mark.parametrize('index', range(6))
def test_native_size_matches_python(index, monkeypatch):
    value = cases()[index]
    expected = python_size(value)
    assert _native.graph_size(value, set()) == expected
    monkeypatch.setenv('KVMEM_GRAPH_NATIVE_SIZE', '1')
    assert block_graph._size(value) == expected


def test_shared_seen_set_across_calls(monkeypatch):
    values = cases()
    seen_py, seen_native = set(), set()
    expected = [python_size(v, seen_py) for v in values + values]
    actual = [_native.graph_size(v, seen_native) for v in values + values]
    assert actual == expected and seen_py == seen_native


def test_flag_validation(monkeypatch):
    monkeypatch.setenv('KVMEM_GRAPH_NATIVE_SIZE', 'yes')
    with pytest.raises(ValueError, match='GRAPH_NATIVE_SIZE'):
        block_graph._size((1,))
