"""KVMEM_DEFERRED_P_CAPTURE: begin_capture() + finish_capture() install the same P checkpoint
as capture(), with the same admission decisions; the worker defers only the CPU half."""
from types import SimpleNamespace

import pytest

from vllm_kvmem.serving.prompt_checkpoints import PromptCheckpointPool
from vllm_kvmem.serving import session_worker as sw
from test_authority import setup, payload, drain
from test_session_state import fixture


def authority_with(tmp_path, name, archive=None):
    authority, archive = setup(tmp_path, name, archive)
    assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(authority)
    return authority, archive


def fields(checkpoint):
    return (checkpoint.kind, checkpoint.position, checkpoint.tokens, checkpoint.payload_bytes,
            tuple(checkpoint.window.records), checkpoint.model.bytes, checkpoint.action_history,
            checkpoint.layerwise_pressure, checkpoint.guided_seed, checkpoint.request_id, checkpoint.namespace)


def test_deferred_capture_installs_the_same_checkpoint(tmp_path):
    a, archive_a = authority_with(tmp_path / 'x', 'a')
    adapter_a, *_ = fixture()
    pool_a = PromptCheckpointPool(2, 4096)
    direct = pool_a.capture(('t', 's'), 'a', a, list(range(11)), adapter_a)

    b, archive_b = authority_with(tmp_path / 'y', 'a')
    adapter_b, *_ = fixture()
    pool_b = PromptCheckpointPool(2, 4096)
    pending = pool_b.begin_capture(('t', 's'), 'a', b, list(range(11)), adapter_b)
    assert not pool_b.entries and pool_b.payload_bytes == 0       # nothing installed before finish
    deferred = pool_b.finish_capture(pending)
    assert fields(deferred)[:6] == fields(direct)[:6] and fields(deferred)[6:] == fields(direct)[6:]
    assert pool_b.entries[('t', 's')] is deferred and pool_b.payload_bytes == pool_a.payload_bytes
    for pool, checkpoint, authority, archive in ((pool_a, direct, a, archive_a), (pool_b, deferred, b, archive_b)):
        pool.drop_live(checkpoint)
        assert authority.close() and archive.close()


def test_deferred_capture_keeps_capture_admission(tmp_path):
    """capture() rejects a model larger than the room left after the CPU snapshots;
    the deferred path rejects the same capture at finish and leaves the pool unchanged."""
    a, archive = authority_with(tmp_path, 'a')
    adapter, *_ = fixture()
    pool = PromptCheckpointPool(2, 600)
    first = pool.capture(('t', 'a'), 'a', a, list(range(11)), adapter)
    b, _ = setup(tmp_path, 'b', archive)
    assert b.append(*payload(20, 8), history_begin=0, local_rope_begin=0)
    drain(b)
    with pytest.raises(BufferError):
        pool.capture(('t', 'b'), 'b', b, list(range(11)), adapter)
    with pytest.raises(BufferError):
        pending = pool.begin_capture(('t', 'b'), 'b', b, list(range(11)), adapter)
        pool.finish_capture(pending)
    assert len(pool.entries) == 1 and pool.payload_bytes == first.payload_bytes
    assert b.close()
    pool.drop_live(first)
    assert a.close() and archive.close()


def test_deferred_replacement_reuses_the_byte_credit(tmp_path):
    a, archive = authority_with(tmp_path, 'a')
    adapter, *_ = fixture()
    pool = PromptCheckpointPool(2, 4096)
    old = pool.capture(('t', 's'), 'a', a, list(range(11)), adapter)
    pool.max_payload_bytes = pool.payload_bytes
    pending = pool.begin_capture(('t', 's'), 'a', a, list(range(11)), adapter, replace=True)
    assert pool.entries[('t', 's')] is old                      # the old P stays until finish
    new = pool.finish_capture(pending)
    assert new is not old and pool.payload_bytes == new.payload_bytes and old.authority is None
    pool.drop_live(new)
    assert a.close() and archive.close()


def test_moved_authority_is_rejected(tmp_path):
    a, archive = authority_with(tmp_path, 'a')
    adapter, *_ = fixture()
    pool = PromptCheckpointPool(2, 4096)
    pending = pool.begin_capture(('t', 's'), 'a', a, list(range(11)), adapter)
    pending.position += 1
    with pytest.raises(RuntimeError, match='moved'):
        pool.finish_capture(pending)
    assert not pool.entries and pool.payload_bytes == 0
    assert a.close() and archive.close()


class _Worker:
    """The SessionWorker methods under test, on a minimal stand-in."""
    selection_committed = sw.SessionWorker.selection_committed
    finish_deferred_captures = sw.SessionWorker.finish_deferred_captures
    _capture_installed = sw.SessionWorker._capture_installed
    _capture_not_admitted = sw.SessionWorker._capture_not_admitted

    def __init__(self, witness):
        self.calls, self.completed, self.checkpoint_deferrals = [], [], 0
        self.options = SimpleNamespace(session_capture_decode=False, session_freeze_prompt=False)
        self.witness = witness
        self.bindings = {'r': SimpleNamespace(namespace=('t', 's'), generation=1)}
        self.pool = SimpleNamespace(retain_prompt=True, entries={},
            finish_capture=lambda pending, committed=False: self.calls.append(committed) or
                SimpleNamespace(layerwise_pressure=None))

    def _require_not_swapped(self, ids):
        pass

    def _capture_witness(self, request_id, authority):
        return self.witness


def pending_worker(witness='w'):
    worker = _Worker(witness)
    capture = SimpleNamespace(position=64, authority=object())
    worker.deferred_captures = {'r': [capture, worker.bindings['r'], SimpleNamespace(position=64), False, 'w']}
    return worker


def test_commit_at_the_p_position_is_folded_into_the_deferred_capture():
    worker = pending_worker()
    worker.selection_committed('r', 64)
    worker.finish_deferred_captures()
    assert worker.calls == [True] and worker.deferred_captures == {}
    assert [n.kind for n in worker.completed] == ['captured']


def test_uncommitted_capture_and_a_changed_witness():
    worker = pending_worker()
    worker.selection_committed('r', 32)                         # another position: not this P
    worker.finish_deferred_captures()
    assert worker.calls == [False]
    worker = pending_worker(witness='changed')
    with pytest.raises(RuntimeError, match='changed'):
        worker.finish_deferred_captures()


def test_flag_is_opt_in_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_DEFERRED_P_CAPTURE', raising=False)
    assert sw.deferred_p_capture() is False
    monkeypatch.setenv('KVMEM_DEFERRED_P_CAPTURE', '2')
    with pytest.raises(ValueError):
        sw.deferred_p_capture()


def test_private_forward_calls_never_defer(monkeypatch):
    """Maintenance forwards call before_attention without deferrable: a P capture there is
    synchronous, and a pending deferral from the public call does not trip the guard."""
    import inspect
    source = inspect.getsource(sw)          # the module text (a test shim may wrap the method)
    assert 'def before_attention(self, batch, *, deferrable=False):' in source
    assert 'if deferrable and deferred_p_capture()' in source
    assert "if deferrable and getattr(self, 'deferred_captures', None):" in source
