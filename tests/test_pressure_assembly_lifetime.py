"""Completed pressure jobs must release without a process-wide cyclic GC."""
import gc
from types import SimpleNamespace as NS
import weakref

import pytest

from vllm_kvmem.layerwise_pressure_efficiency import install_single_assembly
from vllm_kvmem.window_assembly import NativeWindowAssembler


class Owner:
    pass


@pytest.mark.parametrize('stage', ['prefill', 'decode'])
@pytest.mark.parametrize('blocked', ['owner', 'event', 'table', 'staging'])
def test_close_drops_pressure_hook_only_after_all_ownership_fences(stage, blocked):
    automatic = gc.isenabled()
    gc.disable()
    try:
        gates = dict(event=True, table=True, staging=True)
        assembler = NativeWindowAssembler.__new__(NativeWindowAssembler)
        assembler.staging = NS(owner=None,
            poll_fused_table=lambda: gates['table'], close=lambda: gates['staging'])
        assembler._shared_staging = False
        assembler._completion = NS(query=lambda: gates['event'])
        assembler._metadata_owners = []
        assembler.closed = False
        job, controller = Owner(), Owner()
        job.command = NS(pressure_stage=stage)
        controller.names = ('target',)
        controller.rolling_decode = stage == 'decode'
        install_single_assembly(assembler, controller, job, ('target',))
        weak_assembler, weak_job, weak_controller = map(weakref.ref,
                                                       (assembler, job, controller))
        del job, controller
        if blocked == 'owner':
            assembler.staging.owner = assembler
        else:
            gates[blocked] = False
        assert not assembler.close()
        assert weak_job() is not None and weak_controller() is not None
        assert '_assemble' in assembler.__dict__ and not assembler.closed
        assembler.staging.owner = None
        gates.update(event=True, table=True, staging=True)
        assert assembler.close() and assembler.close()
        assert weak_job() is None and weak_controller() is None
        assert '_assemble' not in assembler.__dict__
        del assembler
        # No gc.collect(): completion must break the bound-method self-cycle.
        assert weak_assembler() is None
    finally:
        if automatic:
            gc.enable()
