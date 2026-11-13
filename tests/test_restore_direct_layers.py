"""Opt-in KVMEM_RESTORE_DIRECT_LAYERS: async session swap-ins write every layer once."""
from types import SimpleNamespace as NS

import pytest

from test_layerwise_restore_efficiency import fixture
from vllm_kvmem.serving.layerwise_restore_efficiency import (direct_session_restore,
    install_direct_session_restore, finish_direct_session_restore)
from vllm_kvmem.core.memory import Phase


class Staging:
    def __init__(self, calls):
        self.owner, self.calls = None, calls

    def claim(self, owner):
        assert self.owner is None
        self.owner = owner
        self.calls.append(('claim', owner))

    def release(self, owner):
        assert self.owner is owner
        self.owner = None
        self.calls.append(('release', owner))


def session_fixture(monkeypatch, *, flag='1'):
    monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '1')
    monkeypatch.setenv('KVMEM_RESTORE_DIRECT_LAYERS', flag)
    worker, checkpoint, assembler, plan, event, calls = fixture(monkeypatch)
    controller = worker.runner._kvmem_layerwise_pressure
    assembler.staging = Staging(calls)
    assembler.staging.claim(assembler)        # the restore job's assemble() holds it
    calls.clear()

    def assemble(state, index, records, *, layer_name):
        assert assembler.staging.owner is None, 'the per-layer assembly owns the job scratch'
        assert state['assembly_arena'] == 'refresh' and state['borrowed_staging'] is assembler.staging
        assert 'new' not in controller.maps and 'new' not in controller.histories
        calls.append((index, layer_name, [r.history_begin for r in records]))
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = assemble
    return worker, checkpoint, assembler, plan, event, calls


def test_flag_parsing(monkeypatch):
    monkeypatch.delenv('KVMEM_RESTORE_DIRECT_LAYERS', raising=False)
    assert not direct_session_restore()
    monkeypatch.setenv('KVMEM_RESTORE_DIRECT_LAYERS', '1')
    assert direct_session_restore()
    monkeypatch.setenv('KVMEM_RESTORE_DIRECT_LAYERS', 'yes')
    with pytest.raises(ValueError):
        direct_session_restore()


def test_off_by_default_keeps_restore_sync(monkeypatch):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch, flag='0')
    assert not install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assert '_assemble' not in vars(assembler) and not calls
    assert not finish_direct_session_restore(worker, 'new', checkpoint, assembler)


def test_job_writes_targets_and_draft_once_then_engine_installs(monkeypatch):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    controller = worker.runner._kvmem_layerwise_pressure
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assert not calls and 'new' not in controller.maps
    result = assembler._assemble(checkpoint.authority, plan)
    # Scratch handed over, every target layer from its own map, the draft from
    # layer zero's, scratch taken back, then one fence on the job's stream.
    assert calls[0] == ('release', assembler)
    assert [c[2] for c in calls[1:4]] == [[0, 64, 256], [0, 128, 256], [0, 64, 256]]
    assert [c[1] for c in calls[1:4]] == list(worker.runtime.names)
    assert calls[4] == ('claim', assembler) and calls[5] == ('fence', 'consumer')
    assert assembler.staging.owner is assembler
    assert result.uploaded_bytes == 3 * 768 and result.rebuilt_tokens == 96 and result.event is event
    assert '_assemble' not in vars(assembler), 'no self-cycle keeps the plan pinned'
    with pytest.raises(ValueError):
        finish_direct_session_restore(worker, 'new', checkpoint, assembler)   # fence pending
    event.ready = True
    with pytest.raises(ValueError):
        finish_direct_session_restore(worker, 'new', checkpoint, assembler)   # not committed
    checkpoint.authority.memory.phase = Phase.READY
    assert finish_direct_session_restore(worker, 'new', checkpoint, assembler)
    assert controller.maps['new'] == (288, checkpoint.layerwise_pressure.maps)
    assert len(controller.histories['new']) == len(controller.names)
    assert calls[-1][0] == 'published' and calls[-1][1]['restore_backend'] == 'direct_all_layers_async'
    assert not finish_direct_session_restore(worker, 'new', checkpoint, assembler)


def test_failed_layer_returns_scratch_and_publishes_nothing(monkeypatch):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    controller = worker.runner._kvmem_layerwise_pressure
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)

    def failing(state, index, records, *, layer_name):
        if index == 1:
            raise OSError('archive read failed')
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = failing
    with pytest.raises(OSError):
        assembler._assemble(checkpoint.authority, plan)
    assert assembler.staging.owner is assembler     # assemble() can fence and release it
    assert '_assemble' not in vars(assembler)
    assert not finish_direct_session_restore(worker, 'new', checkpoint, assembler)
    assert 'new' not in controller.maps


@pytest.mark.parametrize('case', ['sessions_off', 'pressure', 'resident', 'controller'])
def test_other_restores_keep_their_path(monkeypatch, case):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    if case == 'sessions_off':
        monkeypatch.setenv('KVMEM_LAYERWISE_SESSIONS', '0')
    if case == 'pressure':
        worker.bindings['new'].pressure_restore = True
    if case == 'resident':
        assembler = None
    if case == 'controller':
        worker.runner._kvmem_layerwise_pressure = None
    assert not install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assert assembler is None or '_assemble' not in vars(assembler)
    assert not calls


def test_finish_async_restore_skips_restore_sync(monkeypatch):
    from vllm_kvmem.serving import layerwise_checkpoints as module
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    worker.bindings['new'].pressure_restore = False
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assembler._assemble(checkpoint.authority, plan)
    event.ready = True
    checkpoint.authority.memory.phase = Phase.READY
    monkeypatch.setattr(module, 'restore_sync', lambda *args: pytest.fail('rebuilt twice'))
    module.finish_async_restore(worker, 'new', checkpoint, resident=False, assembler=assembler)
    assert 'new' in worker.runner._kvmem_layerwise_pressure.maps
    plain = NS()
    seen = []
    monkeypatch.setattr(module, 'restore_sync', lambda *args: seen.append(args[1]))
    module.finish_async_restore(worker, 'other', checkpoint, resident=False, pressure=False, assembler=plain)
    assert seen == ['other']


def _layer():
    import torch
    from vllm_kvmem.serving.window_assembly import LayerWindow
    cache = torch.empty((5, 32, 2, 2, 64), dtype=torch.float8_e4m3fn)
    return LayerWindow(cache[:, :, :, 0], cache[:, :, :, 1], torch.zeros(128, 32),
                       torch.tensor(1.), torch.tensor([1, 3, 4]))


def test_per_layer_assembly_borrows_a_larger_transaction_scratch():
    """The restore job's all-layer scratch serves its one-layer assemblies."""
    import torch
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    layers = tuple(_layer() for _ in range(3))
    parent = NativeWindowAssembler(layers, 32, 32, raw_dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match='geometry changed'):   # the strict shared rule
        NativeWindowAssembler((layers[1],), 32, 64, raw_dtype=torch.float8_e4m3fn, staging=parent.staging)
    child = NativeWindowAssembler((layers[1],), 32, 64, raw_dtype=torch.float8_e4m3fn,
                                  staging=parent.staging, borrowed_staging=True)
    assert child.staging is parent.staging and child._shared_staging
    child.staging.claim(child)
    child.staging.release(child)
    assert child.close() and parent.staging.owner is None


@pytest.mark.parametrize('fault', ['missing', 'small', 'resident_v'])
def test_borrowed_scratch_rejects_unsafe_geometry(fault):
    import torch
    from vllm_kvmem.serving.window_assembly import NativeWindowAssembler
    layers = tuple(_layer() for _ in range(2))
    parent = NativeWindowAssembler(layers, 32, 8, raw_dtype=torch.float8_e4m3fn)
    kwargs = dict(raw_dtype=torch.float8_e4m3fn, staging=parent.staging, borrowed_staging=True)
    chunk = 8
    if fault == 'missing':
        kwargs['staging'] = None
    if fault == 'small':
        chunk = 64                     # one layer x 64 blocks > two layers x 8 blocks
    if fault == 'resident_v':
        kwargs.update(packed=True, direct_pinned_reads=True, reuse_resident_v=True)
    with pytest.raises(ValueError, match='Borrowed'):
        NativeWindowAssembler((layers[0],), 32, chunk, **kwargs)


@pytest.mark.parametrize('available', [True, False])
def test_prepared_reads_share_one_reader_across_layers(monkeypatch, available):
    from vllm_kvmem.serving import pressure_reader
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    monkeypatch.setenv('KVMEM_RESTORE_PREPARED_READS', '1')
    controller = worker.runner._kvmem_layerwise_pressure
    controller.query.paired_reads = True
    log = []

    class Reader:
        def __enter__(self):
            log.append('open')
            return self

        def __exit__(self, *exc):
            log.append('close')
            return False

    reader = Reader()

    def prepare(authority, actual_plan, chosen):
        assert authority is checkpoint.authority and actual_plan is plan
        assert len(chosen) == len(controller.names)
        log.append('prepare')
        return reader if available else None
    monkeypatch.setattr(pressure_reader, 'prepare_pressure_reader', prepare)

    def assemble(state, index, records, *, layer_name):
        if available:
            assert state['prepared_pressure_reader'] is reader and log[-1] == 'open'
            assert state['restore_inline_reads'] and state['restore_parallel_reads']
        else:
            assert 'prepared_pressure_reader' not in state and 'restore_inline_reads' not in state
        calls.append((index, layer_name))
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = assemble
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assembler._assemble(checkpoint.authority, plan)
    assert log == (['prepare', 'open', 'close'] if available else ['prepare'])
    assert [c[0] for c in calls if isinstance(c[0], int)] == [0, 1, 2]


def test_prepared_reads_flag_parsing(monkeypatch):
    from vllm_kvmem.serving.layerwise_restore_efficiency import prepared_restore_reads
    monkeypatch.delenv('KVMEM_RESTORE_PREPARED_READS', raising=False)
    assert not prepared_restore_reads()
    monkeypatch.setenv('KVMEM_RESTORE_PREPARED_READS', '2')
    with pytest.raises(ValueError):
        prepared_restore_reads()


def test_stage_profile_is_written_once_at_finish(monkeypatch, tmp_path):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    monkeypatch.setenv('KVMEM_RESTORE_STAGE_PROFILE', '1')
    worker.runner._kvmem_probe_output = tmp_path
    controller = worker.runner._kvmem_layerwise_pressure
    seen = []

    def assemble(state, index, records, *, layer_name):
        seen.append(state.get('query_stage_profile'))
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = assemble
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    import torch
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False, raising=False)
    assembler._assemble(checkpoint.authority, plan)
    assert seen and all(p is seen[0] and p is not None for p in seen)
    assert not (tmp_path / 'restore_stage_profile.jsonl').exists()
    event.ready = True
    checkpoint.authority.memory.phase = Phase.READY
    assert finish_direct_session_restore(worker, 'new', checkpoint, assembler)
    rows = (tmp_path / 'restore_stage_profile.jsonl').read_text().splitlines()
    assert len(rows) == 1 and '"request_id": "new"' in rows[0]
    assert '_direct_session_profile' not in vars(assembler)


def test_job_lends_its_validated_views_and_a_partial_record_cache(monkeypatch):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    assembler._restore_lease = lease = object()
    controller = worker.runner._kvmem_layerwise_pressure
    seen = []

    def assemble(state, index, records, *, layer_name):
        seen.append((state['borrowed_layers'], state['borrowed_lease'], id(state['restore_partial_operands'])))
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = assemble
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assembler._assemble(checkpoint.authority, plan)
    assert len(seen) == 3 and all(s == seen[0] for s in seen)
    assert seen[0][0] is assembler.layers and seen[0][1] is lease


def test_partial_record_operands_are_read_once_per_restore():
    """assemble_layer_pinned's slow path: one read_records per staged record per restore."""
    import inspect
    from vllm_kvmem.serving import layerwise_query
    source = inspect.getsource(layerwise_query.LayerwiseQuery.assemble_layer_pinned)
    assert "state.get('restore_partial_operands')" in source
    assert "missing = [r for r in group if id(r) not in cache]" in source


@pytest.mark.parametrize('flag', ['0', '1'])
def test_gate_exemption_is_opt_in(monkeypatch, flag):
    worker, checkpoint, assembler, plan, event, calls = session_fixture(monkeypatch)
    monkeypatch.setenv('KVMEM_RESTORE_GATE_EXEMPT', flag)
    controller = worker.runner._kvmem_layerwise_pressure
    seen = []

    def assemble(state, index, records, *, layer_name):
        seen.append(state.get('restore_gate_exempt', False))
        return dict(uploaded_bytes=768, chunks=1)
    controller.query.assemble_layer_pinned = assemble
    assert install_direct_session_restore(worker, 'new', checkpoint, assembler, plan)
    assembler._assemble(checkpoint.authority, plan)
    assert seen == [flag == '1'] * 3
