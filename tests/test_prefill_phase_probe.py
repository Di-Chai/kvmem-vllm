"""CPU-only coverage for opt-in diagnostic wrappers, without importing vLLM."""
from argparse import Namespace
import json
from pathlib import Path
import sys
import threading
from types import MethodType, SimpleNamespace as NS
import weakref

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prefill_phase_probe as phases
import profile_prefill_nsight as profile


class Owner:
    def run(self, value, *, extra=0):
        return value + extra


def probe():
    events = []
    worker = NS(profiler=NS(is_running=False))
    scopes = phases.PhaseScopes(worker,
        push=lambda label: events.append(('push', label, threading.get_ident())),
        pop=lambda: events.append(('pop', None, threading.get_ident())))
    return scopes, worker, events


def test_gate_parameters_identity_and_exception_are_preserved():
    scopes, worker, events = probe()
    owner = Owner()
    assert scopes.wrap(owner, 'run', 'owner.run')
    wrapped = owner.run
    assert not scopes.wrap(owner, 'run', 'owner.run')
    assert owner.run is wrapped
    assert owner.run(4, extra=7) == 11 and not events
    worker.profiler.is_running = True
    assert owner.run(4, extra=7) == 11
    assert [e[:2] for e in events] == [('push', phases.PREFIX+'owner.run'), ('pop', None)]
    failure = ValueError('same failure')
    def fail():
        worker.profiler.is_running = False
        raise failure
    owner.fail = fail
    scopes.wrap(owner, 'fail', 'owner.fail')
    with pytest.raises(ValueError) as caught:
        owner.fail()
    assert caught.value is failure
    assert [e[0] for e in events][-2:] == ['push', 'pop']


def test_background_ranges_stay_on_calling_thread_and_never_cross():
    scopes, worker, events = probe()
    worker.profiler.is_running = True
    owner = Owner()
    scopes.wrap(owner, 'run', 'owner.run')
    output = []
    thread = threading.Thread(target=lambda: output.append(owner.run(9)))
    thread.start(); thread.join(timeout=2)
    assert not thread.is_alive() and output == [9]
    assert [e[2] for e in events] == [thread.ident, thread.ident]
    assert thread.ident != threading.get_ident()


def test_wrapper_does_not_retain_retired_authority_or_lose_dynamic_method():
    scopes, worker, events = probe()
    owner = Owner()
    owner.dynamic = MethodType(lambda self, value: value * 2, owner)
    scopes.wrap(owner, 'dynamic', 'owner.dynamic')
    saved = owner.dynamic
    assert saved(3) == 6
    reference = weakref.ref(owner)
    del owner
    assert reference() is None  # No cyclic-GC delay or owner registry.
    with pytest.raises(RuntimeError, match='released'):
        saved(3)


def test_lazy_installation_preserves_distinct_capture_phases_and_other_modes():
    scopes, worker, events = probe()
    runner = NS()
    scopes.install(runner)
    assert not scopes.labels
    owner = Owner()
    owner.prepare = owner.run
    owner.capture = owner.run
    owner._capture = owner.run
    runtime = NS(action=owner, authorities={})
    runner._kvmem_capture_runtime = runtime
    scopes.install(runner)
    assert {'action.prepare', 'action.capture', 'action._capture'} <= scopes.labels
    assert owner.capture(4) == 4
    before = owner.capture
    scopes.install(runner)
    assert owner.capture is before
    assert not events


def test_accept_subscopes_preserve_shallow_copy_transaction_owner():
    from copy import copy
    class History:
        def __init__(self):
            self.values = {'old':1}
        def accept(self, value):
            self.values = {'new':value}
            return True
    class Action:
        def _prefill_accept(self, history, value):
            return history.accept(value)
    class Graph:
        def prepare_accept(self):
            return self.prepare()
        def prepare(self):
            return 'reserved'
    scopes,worker,events=probe()
    worker.profiler.is_running=True
    live=History();action=Action();graph=Graph()
    authority=NS(action_history=NS(prefill=live),block_graph=graph)
    runner=NS(_kvmem_capture_runtime=NS(action=action,authorities={'owner':authority}))
    scopes.install(runner)
    assert 'accept' not in live.__dict__
    staged=copy(live)
    assert action._prefill_accept(staged,4)
    assert live.values=={'old':1} and staged.values=={'new':4}
    assert graph.prepare_accept()=='reserved'
    assert [label for operation,label,_ in events if operation=='push']==[
        phases.PREFIX+'action._prefill_accept',phases.PREFIX+'graph.prepare_accept',
        phases.PREFIX+'graph.prepare']


def test_worker_override_keeps_native_shape_probe_and_rejects_unknown_worker():
    native = ['serve', 'model', '--worker-cls', 'native_prefill_shape_probe.NativeProbeWorker']
    assert profile.phase_worker_argv(native, 'native')[-1] == 'prefill_phase_probe.NativePhaseWorker'
    assert native[-1] == 'native_prefill_shape_probe.NativeProbeWorker'
    assert profile.phase_worker_argv(['serve','model'], 'kvmem')[-2:] == [
        '--worker-cls', 'prefill_phase_probe.KVMemPhaseWorker']
    with pytest.raises(ValueError, match='unknown'):
        profile.phase_worker_argv(['--worker-cls', 'unrelated.Worker'], 'kvmem')


def test_prepare_freezes_phase_wrapper_and_detects_mutation(tmp_path):
    source = tmp_path / 'source'
    runtime = source / 'runtime'
    runtime.mkdir(parents=True)
    module = runtime / 'sample.py'
    module.write_text('frozen = True\n')
    profile.save(source / 'runtime-proof.json', {'files':{'sample.py':profile.sha(module)}})
    profile.save(source / 'profile.json', dict(prefill_chunk_tokens=2048,
        max_num_batched_tokens=16424, kvmem={}))
    tokens = tmp_path / 'tokens.i32'
    tokens.write_bytes(bytes(32768*4))
    case = tmp_path / 'case.json'
    profile.save(case, dict(input_path=str(tokens), row=dict(served_input_tokens=32768,
        served_sha256=profile.sha(tokens)), sampling={'max_tokens':128}))
    args = Namespace(source=source, case=case, root=tmp_path / 'probe', mode='kvmem',
        port=18866, delay_iterations=8, max_iterations=4, startup_timeout=1200,
        request_timeout=300, nsys='/usr/local/bin/nsys', phase_scopes=True)
    root = profile.prepare(args)
    assert profile.assert_frozen(root)['phase_scopes']
    (root / 'prefill_phase_probe.py').write_text('changed\n')
    with pytest.raises(ValueError, match='prefill_phase_probe.py'):
        profile.assert_frozen(root)


def test_manifest_records_gate_and_nonadditive_scope(tmp_path, monkeypatch):
    monkeypatch.setenv('PREFILL_PHASE_PROBE_OUTPUT', str(tmp_path))
    scopes, worker, events = probe()
    scopes.wrap(Owner(), 'run', 'owner.run')
    scopes.save_manifest()
    report = json.loads(next(tmp_path.glob('phase-scopes-*.json')).read_text())
    assert report['diagnostic_only'] and report['gate'] == 'worker.profiler.is_running'
    assert report['labels'] == ['owner.run'] and 'nonadditive' in report['scope']
