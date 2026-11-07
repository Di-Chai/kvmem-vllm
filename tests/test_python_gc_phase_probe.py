import gc
import json
from types import SimpleNamespace

from python_gc_phase_probe import GCPhaseProbe


def test_gc_probe_balances_range_across_profiler_stop_and_preserves_policy(tmp_path):
    enabled = [False]
    calls = []
    scopes = SimpleNamespace(enabled=lambda: enabled[0], labels=set(),
                             push=lambda x: calls.append(x), pop=lambda: calls.append('pop'))
    before = gc.isenabled(), gc.get_threshold(), list(gc.callbacks)
    probe = GCPhaseProbe(scopes)
    try:
        gc.collect(0)
        assert not probe.rows and not calls
        enabled[0] = True
        gc.collect(0)
        assert probe.rows[-1]['generation'] == 0
        assert calls == ['prefill_phase::python.gc.generation0', 'pop']
        probe.collection('start', {'generation': 2})
        enabled[0] = False
        probe.collection('stop', {'generation': 2, 'collected': 3, 'uncollectable': 0})
        assert calls[-1] == 'pop'
        assert probe.rows[-1]['collected'] == 3
        assert probe.rows[-1]['automatic_collection'] == gc.isenabled()
        assert probe.rows[-1]['caller_stack'][0]['function'] == (
            'test_gc_probe_balances_range_across_profiler_stop_and_preserves_policy')
        assert all(isinstance(value, (str, int))
                   for frame in probe.rows[-1]['caller_stack'] for value in frame.values())
        assert len(probe.rows[-1]['caller_stack']) <= 32
        probe.save(tmp_path)
        assert len(list(tmp_path.glob('python-gc-*.json'))) == 1
        assert (gc.isenabled(), gc.get_threshold()) == before[:2]
    finally:
        probe.close()
    assert gc.callbacks == before[2]


def test_gc_rows_are_saved_once_after_capture_without_heap_scans(tmp_path, monkeypatch):
    enabled = [True]
    scopes = SimpleNamespace(enabled=lambda: enabled[0], labels=set(),
                             push=lambda _: None, pop=lambda: None)
    def no_heap_scan(*args, **kwargs):
        raise AssertionError('The diagnostic must not scan the heap')
    monkeypatch.setattr(gc, 'get_objects', no_heap_scan)
    monkeypatch.setattr(gc, 'get_freeze_count', no_heap_scan)
    probe = GCPhaseProbe(scopes)
    try:
        gc.collect(2)
        assert probe.rows and probe.rows[-1]['generation'] == 2
        with monkeypatch.context() as active:
            active.setattr(json, 'dumps', no_heap_scan)
            probe.save(tmp_path)
            probe.save(tmp_path)
        assert not list(tmp_path.iterdir()) and probe.saved == 0

        enabled[0] = False
        probe.save(tmp_path)
        output, = tmp_path.glob('python-gc-*.json')
        report = json.loads(output.read_text())
        assert report['rows'] == probe.rows
        assert report['diagnostic_only'] and report['policy_unchanged']
        assert probe.saved == len(probe.rows)
        with monkeypatch.context() as saved:
            saved.setattr(json, 'dumps', no_heap_scan)
            probe.save(tmp_path)
    finally:
        probe.close()
