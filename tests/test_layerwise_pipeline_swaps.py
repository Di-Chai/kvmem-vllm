"""Stage B opt-ins: the layerwise pipeline with incremental pages, exact swaps and historical GC."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.serving import serve
from test_dynamic_layerwise_config import profile


def gc_profile(**changes):
    cfg = profile()
    cfg['kvmem'].update(archive_backend='cpu', cpu_read_cache_bytes=8*2**30, async_seal=True,
        async_archive_retirement=True, cpu_history_coordination=True, session_capacity=2,
        request_planning=True, graph_mode='off', cpu_history_cache_bytes_per_request=2**30,
        historical_gc={'mode': 'delete', 'policy': 'lru', 'ready_low': 256, 'ready_high': 512})
    cfg['kvmem'].update(changes)
    return cfg


def launch(cfg, tmp_path):
    return serve.build_launch(cfg, tmp_path, tmp_path/'run', check_paths=False)


def test_gc_with_incremental_admission_needs_the_explicit_opt_in(tmp_path):
    with pytest.raises(ValueError, match='Incremental GPU admission'):
        launch(gc_profile(), tmp_path)
    plan = launch(gc_profile(request_swap_historical_gc=True), tmp_path)
    assert plan['profile']['kvmem']['request_swap_historical_gc'] is True


@pytest.mark.parametrize('changes', [
    {'request_swap_historical_gc': True, 'historical_gc': None},
    {'request_swap_historical_gc': True, 'request_swap_layerwise': False},
    {'request_swap_historical_gc': 1},
])
def test_gc_swap_opt_in_requires_layerwise_swaps_and_gc(tmp_path, changes):
    cfg = gc_profile(**changes)
    if cfg['kvmem']['historical_gc'] is None:
        del cfg['kvmem']['historical_gc']
    with pytest.raises(ValueError):
        launch(cfg, tmp_path)


class Pipeline:
    installed = False

    def install(self):
        self.installed = True


def refresh(admission, swap_layerwise):
    from vllm_kvmem.serving.async_refresh import AsyncRefresh
    r = AsyncRefresh.__new__(AsyncRefresh)
    r.options = NS(maintenance_max_steps=0, gpu_memory_admission=admission,
                   request_swap_layerwise=swap_layerwise)
    r.maintenance = None
    r.runner = NS(_kvmem_layerwise_pressure=NS(pipeline=Pipeline()))
    return r


@pytest.mark.parametrize('flag', [None, '0', '1'])
def test_static_admission_installs_the_pipeline_regardless_of_the_flag(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv('KVMEM_LAYERWISE_PIPELINE_SWAPS', raising=False)
    else:
        monkeypatch.setenv('KVMEM_LAYERWISE_PIPELINE_SWAPS', flag)
    r = refresh('static', False)
    r.initialize_cache()
    assert r.runner._kvmem_layerwise_pressure.pipeline.installed


def test_incremental_admission_needs_the_flag_and_layerwise_swaps(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_PIPELINE_SWAPS', raising=False)
    with pytest.raises(ValueError, match='static page allocation'):
        refresh('incremental', True).initialize_cache()
    monkeypatch.setenv('KVMEM_LAYERWISE_PIPELINE_SWAPS', '1')
    with pytest.raises(ValueError, match='static page allocation'):
        refresh('incremental', False).initialize_cache()
    r = refresh('incremental', True)
    r.initialize_cache()
    assert r.runner._kvmem_layerwise_pressure.pipeline.installed
    monkeypatch.setenv('KVMEM_LAYERWISE_PIPELINE_SWAPS', 'yes')
    with pytest.raises(ValueError, match='KVMEM_LAYERWISE_PIPELINE_SWAPS'):
        refresh('incremental', True).initialize_cache()


def test_swap_witness_resolves_deferred_maps_before_identifying_state(monkeypatch):
    from vllm_kvmem.serving import layerwise_pressure as lp, layerwise_swap
    pressure = lp.LayerwisePrefillPressure.__new__(lp.LayerwisePrefillPressure)
    built = (7, (('layer0',), ('layer1',)))

    class Deferred:
        def value(self):
            return NS(maps=list(built[1]), validate=lambda job: None)
    pressure.maps = {'r': lp.PendingMaps(Deferred(), NS(history=7))}
    pressure.histories, pressure.decode_histories, pressure.names = {}, {}, ('a', 'b')
    query = NS(finished=set(), names=('a', 'b'))
    monkeypatch.setattr(layerwise_swap, '_controllers', lambda runner: (query, pressure, NS()))
    memory = NS(_store=NS(working_records=lambda: ()))
    runner = NS(_kvmem_capture_runtime=NS(authorities={'r': NS(memory=memory, history_end=64)}),
                _kvmem_window_adapter=None)
    owners, signature = layerwise_swap._state(runner, 'r', deep=True)
    assert pressure.maps['r'] == built and signature[0] == layerwise_swap._freeze(built)
    # A later use (e.g. after the restore) sees the same frame object, so the witness holds.
    assert pressure.resolved_maps('r') is owners[6]


def test_request_swaps_do_not_force_an_eager_graph_batch():
    from vllm_kvmem.serving.connector import ControlMetadata, RefreshCommand, RestoreCommand
    from vllm_kvmem.serving.graph_runtime import needs_refresh_batch
    from vllm_kvmem.serving.request_swap_worker import SwapCommand
    swap = SwapCommand('r', 1, 1, 'capture', 64, block_ids=((1,),), checkpoint_bytes=4096)
    control = ControlMetadata(commands=[swap])
    assert not needs_refresh_batch(NS(kv_connector_metadata=control))
    control.commands.append(RefreshCommand('q', 2, ()))
    assert needs_refresh_batch(NS(kv_connector_metadata=control))


def test_verification_oracle_keeps_the_job_on_the_whole_window_path():
    # KVMEM_VERIFY_SESSION_RESTORE attaches a whole-window pressure oracle; the
    # pipeline must decline (not crash) so the job completes and verifies whole.
    from vllm_kvmem.serving.pressure_pipeline import PressurePipeline
    pipeline = PressurePipeline.__new__(PressurePipeline)
    pipeline.active = {}
    lease = NS(dispatchable=lambda: True)
    job = NS(layerwise_pipeline=lease, cancelled=False, pressure_oracle=object(),
             command=NS(request_id='r'), authority=None)
    assert pipeline.dispatch(None, job) is False and pipeline.active == {}
