import importlib.util
from pathlib import Path
import sys


def load():
    path = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location('analyze_prefill_bottleneck', path/'analyze_prefill_bottleneck.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_graph_audit_retains_float_difference_and_rejects_discrete_change():
    compare = load().graph_comparison
    a = [{'sum':17.000005, 'mask':12, 'mode':'multi'}]
    b = [{'sum':17.0000052, 'mask':12, 'mode':'multi'}]
    result = compare(a, b)
    assert not result['exact_match'] and result['within_tolerance']
    assert result['differences'][0]['path'] == '/0/sum'
    b[0]['mask'] = 13
    assert not compare(a, b)['within_tolerance']
    assert not compare(a, [])['within_tolerance']


def test_graph_audit_rejects_large_or_nonfinite_float_difference():
    compare = load().graph_comparison
    assert compare([{'mass':1.}], [{'mass':1.}])['exact_match']
    assert not compare([{'mass':1.}], [{'mass':1.01}])['within_tolerance']
    assert not compare([{'mass':1.}], [{'mass':float('nan')}])['within_tolerance']
