"""Real native page ownership and connector lifecycle; no model/GPU work."""
from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip('vllm')
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec, MambaSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from vllm_kvmem.active_residency import ActiveResidency
from vllm_kvmem.capacity import ReservedKVCacheManager
from vllm_kvmem.connector import ControlCompletions, RestoreCommand
from vllm_kvmem.prompt_checkpoints import token_bytes
from vllm_kvmem.restore_admission import RestoreAdmission
from vllm_kvmem.scheduler import KVMemScheduler
from vllm_kvmem.session_control import SessionLedger, SessionNotice, _Entry
from vllm_kvmem.window_manager import WorkingSetManager
from test_connector import pair


def setup(*, hot=True):
    specs = [FullAttentionSpec(block_size=16, num_kv_heads=2, head_size=64, dtype=torch.uint8),
             MambaSpec(block_size=1024, shapes=((1024,),), dtypes=(torch.float32,))]
    config = KVCacheConfig(num_blocks=33, kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec([str(i)], spec) for i, spec in enumerate(specs)])
    native = KVCacheManager(config, 1024, 1024, 16, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    managers[0] = WorkingSetManager(managers[0], 32, 32)
    native.coordinator.single_type_managers = tuple(managers)
    cfg = NS(speculative_config=None, max_concurrent_batches=1,
             cache_config=NS(enable_prefix_caching=False, mamba_cache_mode='none'))
    capacity = ReservedKVCacheManager(native, cfg)
    manager = managers[0]
    now = [10.]
    residency = ActiveResidency({0: manager}, native.block_pool, 1000, .5, clock=lambda: now[0])
    ledger = SessionLedger(8, 16, manager.prepare_import, manager.abort_import_preparation,
                           lambda *args: None)
    connector, _ = pair()
    connector.sessions, connector.active_residency = ledger, residency
    admission = RestoreAdmission(residency=residency)
    connector.restore_admission = capacity.restore_admission = admission
    capacity.active_residency, capacity.resident_entry = residency, ledger.requests.get
    namespace = ('tenant', 'session')
    if hot:
        manager.offsets['old'] = 16
        manager.allocate_new_blocks('old', 64, 64)
        residency.consume(SessionNotice('retired', namespace, 1, 'old', 64, 48, resident_ready=True))
        manager.free('old')
    ledger.entries[namespace] = _Entry(namespace, 1, None, phase='idle', history=64,
                                       window=48, tokens=token_bytes([1]*80, 64))
    request = Request('new', [1]*80, SamplingParams(max_tokens=16), None)
    request.session_id, request.cache_salt = 'session', 'tenant'
    return connector, capacity, admission, request, now


def allocate(capacity, request):
    return capacity.allocate_slots(request, 16, num_external_computed_tokens=64)


def test_exact_hot_view_cannot_turn_cold_between_prefix_lookup_and_native_allocation():
    connector, capacity, admission, request, now = setup()
    pool = capacity.block_pool
    free_before = pool.get_num_free_blocks()
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    permit = admission.permits['new']
    assert permit.view is not None and not permit.allocated
    assert not connector.active_residency.entries
    assert not capacity.reservations and 'new' not in capacity.groups[1].req_to_blocks
    now[0] += 2  # TTL crossing inside the same synchronous admission transaction.
    result = allocate(capacity, request)
    assert result is not None and admission.permits['new'].allocated
    assert 'new' in capacity.resident_restores
    assert tuple(capacity.groups[0].req_to_blocks['new'][:3]) == permit.view.groups[0][1]
    assert len(capacity.groups[1].req_to_blocks['new']) == 1
    assert pool.get_num_free_blocks() == free_before - 2  # one suffix + one GDN row
    assert admission.snapshot()['admitted'] == {'resident': 1, 'assembly': 0}


def test_failed_native_allocation_returns_hot_pages_and_releases_lane(monkeypatch):
    connector, capacity, admission, request, _ = setup()
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    view = admission.permits['new'].view
    free_before = capacity.block_pool.get_num_free_blocks()
    monkeypatch.setattr(capacity.native, 'allocate_slots', lambda *args, **kwargs: None)
    assert allocate(capacity, request) is None
    assert not admission.permits and not capacity.reservations
    assert connector.active_residency.entries[view.namespace] is view
    assert not capacity.groups[0].resident_imports
    assert capacity.block_pool.get_num_free_blocks() == free_before


def test_unused_lookup_and_cancel_roll_back_before_import_preparation_is_aborted():
    connector, capacity, admission, request, _ = setup()
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    view = admission.permits['new'].view
    admission.end_step()  # Native token/phase gate returned zero before allocate.
    assert connector.active_residency.entries[view.namespace] is view
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    connector.request_finished_all_groups(request, ())
    assert not admission.permits and not admission.waiting
    assert not capacity.groups[0].resident_imports and not capacity.groups[0].imports
    assert connector.active_residency.entries[view.namespace] is view
    assert connector.sessions.entries[view.namespace].phase == 'idle'


def test_partial_native_allocation_exception_never_returns_modified_pages_to_idle(monkeypatch):
    connector, capacity, admission, request, _ = setup()
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    original = capacity.native.allocate_slots
    def partial(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError('partial native failure')
    monkeypatch.setattr(capacity.native, 'allocate_slots', partial)
    with pytest.raises(RuntimeError, match='partial native'):
        allocate(capacity, request)
    admission.end_step()
    assert admission.permits['new'].allocated
    assert capacity.groups[0].req_to_blocks['new']
    assert not connector.active_residency.entries
    assert not any(admission.admitted.values())


def test_cancelled_allocated_restore_releases_lane_only_after_real_connector_ack():
    connector, capacity, admission, request, _ = setup()
    assert connector.get_num_new_matched_tokens(request, 0) == (64, False)
    assert allocate(capacity, request) is not None
    connector.update_state_after_alloc(request, capacity.get_blocks('new'), 64)
    command = RestoreCommand('new', -1, 1, 64, 48,
        tuple(tuple(b.block_id for b in m.req_to_blocks['new']) for m in capacity.groups), resident=True)
    admission.dispatch(command)
    connector.queue(command)
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler.connector, scheduler.kv_cache_manager = connector, capacity
    scheduler.window_managers = {0: capacity.groups[0]}
    scheduler.requests, scheduler.transactions = {'new': request}, {'new': (command, 7)}
    connector.scheduler_completion = scheduler._complete_refresh
    manager = capacity.groups[0]
    connector.sessions.commit_import = lambda rid, history: manager.commit_import(
        rid, history, NS(query=lambda: True))
    request.status = RequestStatus.FINISHED_ABORTED
    assert connector.request_finished_all_groups(request, ()) == (True, None)
    assert admission.permits['new'].allocated and 'new' in connector.pending
    with pytest.raises(ValueError, match='Stale'):
        connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions([('new', -2)])))
    assert 'new' in admission.permits
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        [('new', -1)], sessions=[SessionNotice('restored', ('tenant', 'session'), 1, 'new', 64, 48, retained=True)])))
    assert not admission.permits and not scheduler.transactions and not connector.pending
    assert not manager.imports and request.next_decode_eligible_step == 7
    # Completion releases the maintenance permit, not native allocation/pages.
    assert 'new' in capacity.reservations and manager.req_to_blocks['new']


def test_publication_feedback_clears_with_empty_snapshot_and_keeps_idle_control_alive():
    scheduler, worker = pair()
    blocked = {'prefill'}
    worker.publication_snapshot = lambda: blocked
    scheduler.publication_live = lambda: {'prefill', 'decode'}
    metadata = worker.build_connector_worker_meta()
    assert metadata.publication_sequence == 1
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=metadata))
    assert scheduler.publication_blocked == {'prefill'}
    assert scheduler.has_pending_push_work()
    blocked.clear()
    metadata = worker.build_connector_worker_meta()
    assert metadata.publication_sequence == 2 and not metadata.publication_blocked
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=metadata))
    assert not scheduler.publication_blocked and not scheduler.has_pending_push_work()
    with pytest.raises(ValueError, match='Stale publication'):
        scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=metadata))


def test_publication_feedback_discards_finished_or_released_request_ids():
    scheduler, _ = pair()
    scheduler.publication_live = lambda: {'live'}
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        [], publication_sequence=1, publication_blocked={'finished', 'released', 'live'})))
    assert scheduler.publication_blocked == {'live'}
    scheduler.request_finished_all_groups(NS(request_id='live'), ())
    assert not scheduler.publication_blocked
