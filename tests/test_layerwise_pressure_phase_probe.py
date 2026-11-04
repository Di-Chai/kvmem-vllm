"""Diagnostic scopes must preserve factories and staged history ownership."""
from copy import copy
from types import SimpleNamespace as NS

import pytest

from layerwise_pressure_phase_probe import install_layerwise_scopes
from prefill_phase_probe import PhaseScopes, PREFIX


def setup_probe():
    class History:
        def __init__(self):
            self.value = 1

        def accept(self, value):
            self.value = value

    class Pressure:
        def __init__(self):
            self.history = History()

        def prepare_accept(self, value):
            staged = copy(self.history)
            staged.accept(value)
            return staged

    class Assembler:
        def __init__(self, value, *, extra=0):
            if value < 0:
                raise ValueError('bad geometry')
            self.value = value + extra

        @classmethod
        def from_runner(cls, value, *, extra=0):
            return cls(value, extra=extra)

    events = []
    pressure = Pressure()
    runner = NS(_kvmem_serving=True, _kvmem_layerwise_query=NS(),
                _kvmem_layerwise_pressure=pressure)
    worker = NS(model_runner=runner, profiler=NS(is_running=False))
    worker._prefill_phase_probe = PhaseScopes(worker,
        push=lambda label: events.append(('push', label)),
        pop=lambda: events.append(('pop', None)))
    modules = dict(window_assembly=NS(NativeWindowAssembler=Assembler),
                   native_pages=NS(), cuda_scoring=NS())
    return worker, pressure, Assembler, modules, events


def test_class_factory_constructor_and_exception_keep_their_semantics():
    worker, _, assembler, modules, events = setup_probe()
    assert install_layerwise_scopes(worker, modules=modules)
    factory = assembler.from_runner
    assert install_layerwise_scopes(worker, modules=modules)
    assert assembler.from_runner is factory
    result = factory(3, extra=4)
    assert isinstance(result, assembler) and result.value == 7 and not events
    worker.profiler.is_running = True
    assert factory(5, extra=2).value == 7
    assert events == [('push', PREFIX+'assembly.from_runner'),
                      ('push', PREFIX+'assembly.__init__'),
                      ('pop', None), ('pop', None)]
    events.clear()
    with pytest.raises(ValueError, match='bad geometry'):
        factory(-1)
    assert [op for op, _ in events] == ['push', 'push', 'pop', 'pop']


def test_shallow_copy_acceptance_stays_on_staged_history():
    worker, pressure, _, modules, events = setup_probe()
    worker.profiler.is_running = True
    assert install_layerwise_scopes(worker, modules=modules)
    staged = pressure.prepare_accept(7)
    assert staged.value == 7 and pressure.history.value == 1
    assert 'accept' not in pressure.history.__dict__
    assert events == [('push', PREFIX+'layerwise_pressure.prepare_accept'),
                      ('pop', None)]


def test_lazy_runtime_and_missing_controller_do_not_mislabel_shared_pressure():
    worker, _, _, modules, events = setup_probe()
    worker.model_runner = NS()
    assert not install_layerwise_scopes(worker, modules=modules)
    worker.model_runner._kvmem_serving = True
    assert not install_layerwise_scopes(worker, modules=modules)
    worker.model_runner._kvmem_layerwise_query = NS()
    with pytest.raises(ValueError, match='actual controller'):
        install_layerwise_scopes(worker, modules=modules)
    assert not events
