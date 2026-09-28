import time

import pytest
import torch

from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool, CheckpointPhase, token_bytes
from test_authority import setup, payload, drain
from test_core import Event
from test_session_state import fixture


def saved(tmp_path, *, name='a', namespace=('tenant', 'session'), pool=None, archive=None):
    authority, archive = setup(tmp_path, name, archive)
    assert authority.append(*payload(0, 8), history_begin=0, local_rope_begin=0)
    drain(authority)
    adapter, conv, ssm, cache, _ = fixture()
    pool = pool or PromptCheckpointPool(2, 4096)
    checkpoint = pool.capture(namespace, name, authority, list(range(11)), adapter)
    return pool, checkpoint, authority, archive, adapter, (conv, ssm, cache)


def ready(pool, checkpoint):
    deadline = time.monotonic() + 5
    while not pool.idle(checkpoint):
        if time.monotonic() >= deadline:
            pytest.fail('Checkpoint retirement did not drain')
        time.sleep(.001)


def test_exact_prompt_reuse_rewinds_suffix_and_restores_fresh_native_state(tmp_path):
    pool, checkpoint, authority, archive, adapter, (conv, ssm, cache) = saved(tmp_path)
    assert checkpoint.payload_bytes == 32 + 80 + 248
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    fence = Event()
    pool.retire(checkpoint, fence)
    assert not pool.idle(checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(20))) is None
    fence.done = True
    ready(pool, checkpoint)
    assert authority.history_end == 8
    assert len(authority.records) == 2
    # Rendered output after the saved prefix can differ; the prefix itself cannot.
    changed = list(range(20)); changed[2] = 100
    assert pool.claim(checkpoint.namespace, 'wrong', changed) is None
    assert pool.claim(('other-tenant', 'session'), 'wrong', list(range(20))) is None
    assert pool.claim(checkpoint.namespace, 'new', list(range(8)) + [400, 500]) is checkpoint
    assert pool.claim(checkpoint.namespace, 'racing', list(range(20))) is None
    before = conv.clone(), ssm.clone(), cache.clone()
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    conv[1:3].fill_(-11); ssm[1:3].fill_(-11); cache[1].fill_(-11)
    completion = Event(True)
    with pytest.raises(ValueError, match='did not commit'):
        pool.finish_claim(checkpoint, completion)
    plan = pool.prepare_claim_window(checkpoint)
    assert plan.total_window_tokens == 8
    assert all(not row.skip and not row.working_k_resident for row in plan.remaps)
    authority.memory.attach_completion(completion)
    assert authority.memory.commit_selection()
    assert adapter.restore(checkpoint.model, 'new') is None
    assert pool.finish_claim(checkpoint, completion) is authority
    del plan
    assert not pool.entries and pool.payload_bytes == 0
    assert checkpoint.authority is checkpoint.window is checkpoint.model is None
    assert torch.equal(conv[5], before[0][1]) and torch.equal(ssm[5], before[1][1])
    assert torch.equal(cache[8], before[2][1])
    assert torch.equal(conv[3:5], before[0][3:5])
    assert authority.append(*payload(100, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    assert authority.close() and archive.close()


def test_checkpoint_payload_and_slots_remain_reserved_through_fenced_eviction(tmp_path):
    pool, checkpoint, authority, archive, adapter, _ = saved(tmp_path)
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    retained = next(iter(authority.records.values())).handle
    reader = authority.memory.repository.pin(retained)
    fence = Event()
    pool.retire(checkpoint, fence)
    payload_bytes = pool.payload_bytes
    assert not pool.evict(checkpoint)
    assert checkpoint.window is not None
    assert pool.payload_bytes == payload_bytes
    fence.done = True
    assert not pool.poll_eviction(checkpoint)
    assert checkpoint.window is None and checkpoint.phase is CheckpointPhase.CLOSING
    assert pool.payload_bytes == payload_bytes and len(pool.entries) == 1
    reader.reset()
    assert pool.poll_eviction(checkpoint)
    assert not pool.entries and pool.payload_bytes == 0 and archive.close()


def test_checkpoint_admission_never_overallocates_or_closes_live_authority(tmp_path):
    pool = PromptCheckpointPool(2, 600)
    pool, a, authority_a, archive, adapter, _ = saved(tmp_path, pool=pool)
    b, _ = setup(tmp_path, 'b', archive)
    assert b.append(*payload(20, 8), history_begin=0, local_rope_begin=0)
    drain(b)
    with pytest.raises(BufferError, match='byte reservation'):
        pool.capture(('tenant', 'b'), 'b', b, list(range(11)), adapter)
    assert len(pool.entries) == 1 and pool.payload_bytes == a.payload_bytes
    # Failed capture must not leak its temporary native prefix pin.
    assert b.close()
    with pytest.raises(ValueError, match='serialize'):
        pool.capture(a.namespace, 'a', authority_a, list(range(11)), adapter)
    with pytest.raises(ValueError, match='Live capture'):
        pool.evict(a)
    pool.drop_live(a)
    assert authority_a.history_end == 8 and authority_a.close()
    assert pool.payload_bytes == 0 and archive.close()


def test_import_cancellation_waits_for_restore_cleanup_and_rejects_stale_leases(tmp_path):
    pool, checkpoint, authority, archive, _, _ = saved(tmp_path)
    pool.retire(checkpoint, Event(True)); ready(pool, checkpoint)
    pool.claim(checkpoint.namespace, 'new', list(range(20)))
    with pytest.raises(ValueError, match='cleanup fence'):
        pool.evict(checkpoint)
    plan = pool.prepare_claim_window(checkpoint)
    cleanup = Event()
    authority.memory.fail_transaction(cleanup)
    assert not pool.evict(checkpoint, cleanup)
    cleanup.done = True
    del plan
    assert pool.poll_eviction(checkpoint)
    with pytest.raises(ValueError, match='Stale'):
        pool.finish_claim(checkpoint, cleanup)
    assert archive.close()


def test_retirement_is_unavailable_until_discarded_readers_and_pool_credits_drain(tmp_path):
    pool = PromptCheckpointPool(1, 4096)
    pool, checkpoint, authority, archive, adapter, _ = saved(tmp_path, pool=pool)
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    last = list(authority.records.values())[-1].handle
    reader = authority.memory.repository.pin(last)
    pool.retire(checkpoint, Event(True))
    assert not pool.idle(checkpoint)
    assert authority.history_end == 8
    assert not pool.idle(checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(20))) is None
    with pytest.raises(BufferError, match='entry capacity'):
        pool.capture(('tenant', 'other'), 'b', authority, list(range(20)), adapter)
    reader.reset()
    ready(pool, checkpoint)
    assert pool.evict(checkpoint)
    assert archive.close() and pool.payload_bytes == 0


def test_prompt_identity_has_no_hash_collision_or_generated_suffix_assumption():
    assert token_bytes([0, 2**32-1, 9], 2) == token_bytes([0, 2**32-1, 10], 2)
    assert token_bytes([0, 2, 9], 2) != token_bytes([0, 3, 9], 2)
    for tokens, count in (([0, 1], 2), ([0, 1], 0), ([True, 1], 1), ([-1, 1], 1), ([2**32, 1], 1)):
        with pytest.raises(ValueError):
            token_bytes(tokens, count)
