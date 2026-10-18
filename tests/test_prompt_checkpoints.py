import time
from dataclasses import replace
from types import SimpleNamespace

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


def test_terminal_restore_is_not_advertised_before_sealed_publication(tmp_path,monkeypatch):
    pool,old,authority,archive,adapter,_=saved(tmp_path)
    write=archive.write
    monkeypatch.setattr(archive,'write',lambda *args,**kwargs:None)
    assert authority.append(*payload(8,4),history_begin=8,local_rope_begin=8)
    adapter.runner.req_states.num_computed_tokens_np[0]=12
    adapter.runner.req_states.prefill_len.np[0]=16
    adapter.runner._kvmem_capture_runtime = type('Runtime', (), {
        'last_steps': {'a': [{'history_begin': 8, 'accepted': 4, 'scheduled': 4}]},
        'authorities': {'a': authority}})()
    adapter.recurrent.conv_dim_first=False;adapter.recurrent.conv_widths={'gdn':4}
    checkpoint=pool.capture(old.namespace,'a',authority,list(range(16)),adapter,replace=True,terminal=True)
    pool.retire(checkpoint,Event(True))
    assert not pool.idle(checkpoint)
    assert checkpoint.phase is CheckpointPhase.RETIRING
    assert pool.claim(checkpoint.namespace,'new',list(range(16))) is None
    assert authority._unsubmitted
    monkeypatch.setattr(archive,'write',write)
    ready(pool,checkpoint)
    assert all(r.archived for r in authority.records.values())
    assert pool.claim(checkpoint.namespace,'new',list(range(16))) is checkpoint
    plan=pool.prepare_claim_window(checkpoint)
    authority.memory.attach_completion(Event(True));assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint,Event(True)) is authority
    del plan
    assert authority.close() and archive.close()


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


@pytest.mark.parametrize('action_enabled', [False, True])
def test_pressure_import_reselects_without_reusing_retired_gpu_pages(tmp_path, action_enabled):
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    adapter.runner.req_states.num_computed_tokens_np[0] = 12
    adapter.runner.req_states.prefill_len.np[0] = 16
    if action_enabled:
        from test_prefill_attention import options
        from vllm_kvmem.action_attention import ActionHistory
        authority.action_history=ActionHistory(options(),[10,11,12,13,14])
        authority.action_history.prefill.values={4:[1.,1.]}
    checkpoint = pool.capture(old.namespace, 'a', authority, list(range(16)), adapter, replace=True)
    assert checkpoint.window.window_tokens == 12
    pool.retire(checkpoint, Event(True)); ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(20))) is checkpoint
    plan = pool.prepare_claim_window(checkpoint, pressure=True)
    assert plan.total_window_tokens == 8
    assert all(not r.skip and not r.working_k_resident for r in plan.remaps)
    if action_enabled:
        # Retiring residency must not erase logical pressure candidates, and
        # a scored old page must beat an unscored newer page.
        assert [r.to_base for r in plan.remaps]==[0,4]
        assert [r.block_id for r in plan.remaps]==[authority.records[k].handle.slot
            for k in sorted(authority.records)[:2]]
    assert checkpoint.restore_records != checkpoint.window.records
    authority.memory.attach_completion(Event(True)); assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    assert authority.history_end == 12
    del plan
    assert authority.close() and archive.close()


def test_short_turn_checkpoint_replays_the_committed_query_frame(tmp_path):
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    assert authority.append(*payload(8,4),history_begin=8,local_rope_begin=8)
    drain(authority)
    adapter.runner.req_states.num_computed_tokens_np[0]=12
    adapter.runner.req_states.prefill_len.np[0]=20
    checkpoint=pool.capture(old.namespace,'a',authority,list(range(20)),adapter,replace=True)
    model=checkpoint.model;old_bytes=pool.payload_bytes
    records=list(authority.records.values())
    plan=authority.memory.select({r.handle.slot:float(i) for i,r in enumerate(records)})
    authority.memory.attach_completion(Event(True));assert authority.memory.commit_selection()
    expected=tuple(authority.memory._store.working_records())
    assert checkpoint.window.records!=expected
    pool.update_live_window(checkpoint)
    assert checkpoint.window.records==expected and checkpoint.model is model
    assert pool.payload_bytes==old_bytes-40
    assert authority.append(*payload(12,1),history_begin=12,local_rope_begin=8)
    pool.retire(checkpoint,Event(True));ready(pool,checkpoint)
    assert pool.claim(checkpoint.namespace,'new',list(range(20))) is checkpoint
    restored=pool.prepare_claim_window(checkpoint)
    authority.memory.attach_completion(Event(True));assert authority.memory.commit_selection()
    assert tuple(authority.memory._store.working_records())==expected
    assert pool.finish_claim(checkpoint,Event(True)) is authority
    del restored,plan
    assert authority.close() and archive.close()


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


def test_live_checkpoint_replacement_is_atomic_and_reuses_its_byte_credit(tmp_path):
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    pool.max_payload_bytes = pool.payload_bytes
    replacement = pool.capture(old.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert replacement is not old and pool.payload_bytes == replacement.payload_bytes
    assert old.authority is None
    pool.max_payload_bytes = 1
    with pytest.raises(BufferError):
        pool.capture(replacement.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert pool.entries[replacement.namespace] is replacement and replacement.authority is authority
    pool.drop_live(replacement)
    assert authority.close() and archive.close()


@pytest.mark.parametrize('pressure', [False, True])
def test_terminal_checkpoint_keeps_partial_accepted_raw_kv_tail_and_fences_import(tmp_path, pressure):
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    raw, values = payload(8, 2)
    assert authority.append(raw, values, history_begin=8, local_rope_begin=8)
    adapter.runner._kvmem_capture_runtime = type('Runtime', (), {
        'last_steps': {'a': [{'history_begin': 8, 'accepted': 2, 'scheduled': 2}]},
        'authorities': {'a': authority}})()
    adapter.recurrent.conv_dim_first = False; adapter.recurrent.conv_widths = {'gdn': 4}
    terminal = pool.capture(old.namespace, 'a', authority, list(range(12)), adapter, replace=True, terminal=True)
    assert terminal.position == 10 and terminal.window.window_tokens == 10
    fence = Event()
    pool.retire(terminal, fence)
    assert not pool.idle(terminal)
    fence.done = True
    assert pool.idle(terminal) and authority.history_end == 10
    tail = max(authority.records.values(), key=lambda r: r.history_begin)
    assert tail.count == 2 and not tail.archived
    assert torch.equal(tail.raw[:, :2], raw) and torch.equal(tail.values[:, :2], values)
    assert pool.claim(terminal.namespace, 'new', list(range(14))) is terminal
    plan = pool.prepare_claim_window(terminal, pressure=pressure,
        pressure_budget=8 if pressure else None,
        mandatory_spans=((0,4,'root'),) if pressure else ())
    assert plan.total_window_tokens == (6 if pressure else 10)
    assert plan.remaps[0].block_id == 0
    assert plan.remaps[-1].n_tokens == 2
    authority.memory.attach_completion(Event(True)); assert authority.memory.commit_selection()
    assert pool.finish_claim(terminal, Event(True)) is authority
    del plan
    assert authority.append(*payload(10, 2), history_begin=10, local_rope_begin=6 if pressure else 10)
    drain(authority)
    sealed_raw, sealed_values = authority.read_record(tail.handle)
    assert torch.equal(sealed_raw[:, :2], raw) and torch.equal(sealed_values[:, :2], values)
    assert authority.close() and archive.close()


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


@pytest.mark.parametrize('suffix', [1, 6])
def test_cancelled_terminal_successor_rebases_partial_tail_without_recompute(tmp_path, suffix):
    pool, old, authority, archive, adapter, _ = saved(tmp_path)
    assert authority.append(*payload(8, 2), history_begin=8, local_rope_begin=8)
    adapter.runner._kvmem_capture_runtime = type('Runtime', (), {
        'last_steps': {'a': [{'history_begin': 8, 'accepted': 2, 'scheduled': 2}]},
        'authorities': {'a': authority}})()
    adapter.recurrent.conv_dim_first = False; adapter.recurrent.conv_widths = {'gdn': 4}
    checkpoint = pool.capture(old.namespace, 'a', authority, list(range(20)), adapter,
                              replace=True, terminal=True)
    model, identity, size = checkpoint.model, checkpoint.tokens, pool.payload_bytes
    pool.retire(checkpoint, Event(True)); ready(pool, checkpoint)
    originals = tuple(checkpoint.window.records)
    peer, _ = setup(tmp_path, 'peer', archive)
    assert peer.append(*payload(50, 4), history_begin=0, local_rope_begin=0)
    drain(peer)
    generations = set()
    for branch in range(5):
        assert pool.claim(checkpoint.namespace, 'branch-' + str(branch), list(range(20))) is checkpoint
        plan = pool.prepare_claim_window(checkpoint)
        authority.memory.attach_completion(Event(True)); assert authority.memory.commit_selection()
        assert pool.finish_claim(checkpoint, Event(True), retain=True) is authority
        del plan
        assert authority.append(*payload(100 + branch * 10, suffix), history_begin=10, local_rope_begin=10)
        drain(authority)
        tail = next(r for r in authority.records.values() if r.history_begin == 8)
        old_key = (tail.handle.slot, tail.handle.generation)
        generations.add(old_key)
        reader = authority.memory.repository.pin(tail.handle)
        # An independent selection must not replace the checkpoint's frame.
        if suffix == 6:
            selection = authority.memory.select({r.handle.slot: float(r.history_begin)
                                                  for r in authority.records.values()})
            authority.memory.attach_completion(Event(True)); assert authority.memory.commit_selection()
            del selection
        fence = Event()
        pool.retire(checkpoint, fence)
        assert not pool.idle(checkpoint) and authority.history_end == 10 + suffix
        fence.done = True
        assert not pool.idle(checkpoint) and authority.history_end == 8
        assert pool.claim(checkpoint.namespace, 'racing', list(range(20))) is None
        reader.reset()
        ready(pool, checkpoint)
        assert authority.history_end == checkpoint.position == 10
        assert checkpoint.model is model and checkpoint.tokens == identity and pool.payload_bytes == size
        assert checkpoint.window.records[:-1] == originals[:-1]
        tail = next(r for r in authority.records.values() if r.history_begin == 8)
        assert (tail.handle.slot, tail.handle.generation) not in generations
        assert tail.count == 2 and not tail.archived and not tail.submitted
        raw, values = authority.read_record(tail.handle)
        assert torch.equal(raw, payload(8, 2)[0]) and torch.equal(values, payload(8, 2)[1])
        assert torch.equal(tail.key_sum, raw.float().sum(dim=1))
        assert len(authority.records) == 3 and not authority.discarded_records
        assert torch.equal(peer.read_record(next(iter(peer.records.values())).handle)[0], payload(50, 4)[0])
    # Eviction during a subsequent rollback must release the rebased frame's
    # explicit prefix pins while an external reader still owns the old tail.
    assert pool.claim(checkpoint.namespace, 'closing-branch', list(range(20))) is checkpoint
    plan = pool.prepare_claim_window(checkpoint)
    authority.memory.attach_completion(Event(True)); assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True), retain=True) is authority
    del plan
    assert authority.append(*payload(200, suffix), history_begin=10, local_rope_begin=10)
    drain(authority)
    reader = authority.memory.repository.pin(next(r for r in authority.records.values() if r.history_begin == 8).handle)
    pool.retire(checkpoint, Event(True))
    assert not pool.idle(checkpoint)
    assert not pool.evict(checkpoint) and pool.payload_bytes == size
    reader.reset()
    assert pool.poll_eviction(checkpoint) and pool.payload_bytes == 0
    assert peer.close() and archive.close()


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


def terminal_with_prompt(tmp_path):
    pool = PromptCheckpointPool(2, 8192, retain_prompt=True)
    pool, prompt, authority, archive, adapter, state = saved(tmp_path, pool=pool)
    assert authority.append(*payload(8, 4), history_begin=8, local_rope_begin=8)
    drain(authority)
    adapter.runner.req_states.num_computed_tokens_np[0] = 12
    adapter.runner.req_states.prefill_len.np[0] = 16
    adapter.runner._kvmem_capture_runtime = type('Runtime', (), {
        'last_steps': {'a': [{'history_begin': 8, 'accepted': 4, 'scheduled': 4}]},
        'authorities': {'a': authority}})()
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    terminal = pool.capture(prompt.namespace, 'a', authority, list(range(16)),
                            adapter, replace=True, terminal=True)
    pool.retire(terminal, Event(True))
    ready(pool, terminal)
    return pool, prompt, terminal, authority, archive, adapter


def test_nonblocking_prompt_rewind_waits_only_its_reader_without_archive_wait(tmp_path, monkeypatch):
    pool, prompt, terminal, authority, archive, _ = terminal_with_prompt(tmp_path)
    suffix = max(authority.records.values(), key=lambda record: record.history_begin)
    reader = authority.memory.repository.pin(suffix.handle)
    checkpoint = pool.claim(prompt.namespace, 'new', list(range(16)), position=8)
    assert checkpoint is prompt and terminal.authority is None
    monkeypatch.setattr(archive, 'wait_for_idle',
                        lambda *_: pytest.fail('A cooperative rewind must not wait on the archive'))
    assert pool.prepare_claim_window(checkpoint, wait=False) is None
    assert checkpoint.rewind_started and not checkpoint.restore_started
    assert checkpoint.phase is CheckpointPhase.CLAIMED
    deadline = checkpoint.restore_deadline
    assert pool.prepare_claim_window(checkpoint, wait=False) is None
    assert checkpoint.restore_deadline == deadline
    checkpoint.restore_deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError, match='rewind did not drain'):
        pool.prepare_claim_window(checkpoint, wait=False)
    checkpoint.restore_deadline = deadline
    reader.reset()
    plan = pool.prepare_claim_window(checkpoint, wait=False)
    assert plan is not None and plan.total_window_tokens == 8
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    del plan
    assert authority.close() and archive.close()


def test_retained_import_can_retire_and_restore_again_without_losing_cpu_history(tmp_path):
    pool, prompt, terminal, authority, archive, _ = terminal_with_prompt(tmp_path)
    retained_bytes = pool.payload_bytes
    checkpoint = pool.claim(terminal.namespace, 'new', list(range(20)), position=12)
    plan = pool.prepare_claim_window(checkpoint, wait=False)
    fence = Event()
    authority.memory.attach_completion(fence)
    assert not authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, fence, retain=True) is None
    assert checkpoint.phase is CheckpointPhase.CLAIMED
    assert pool.payload_bytes == retained_bytes
    fence.done = True
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, fence, retain=True) is authority
    assert checkpoint.phase is prompt.phase is CheckpointPhase.LIVE
    assert checkpoint.request_id == prompt.request_id == 'new'
    assert checkpoint.model is not None and pool.payload_bytes == retained_bytes
    # Cancellation before any new decode capture still has an exact CPU-owned
    # checkpoint for the next request; it must not force recomputation.
    pool.retire(checkpoint, Event(True))
    ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'again', list(range(20)), position=12) is checkpoint
    second_plan = pool.prepare_claim_window(checkpoint, wait=False)
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True), retain=True) is authority
    assert authority.history_end == 12 and checkpoint.request_id == 'again'
    pool.drop_live(checkpoint)
    del plan, second_plan
    assert pool.payload_bytes == 0 and not pool.prompts
    assert authority.close() and archive.close()


def test_new_prompt_capture_replaces_retained_import_and_releases_old_prompt_pins(tmp_path):
    pool, old_prompt, terminal, authority, archive, adapter = terminal_with_prompt(tmp_path)
    checkpoint = pool.claim(terminal.namespace, 'new', list(range(20)), position=12)
    plan = pool.prepare_claim_window(checkpoint, wait=False)
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(checkpoint, Event(True), retain=True) is authority
    adapter.runner.req_states.req_id_to_index['new'] = 2
    adapter.runner._kvmem_request_generations['new'] = 3
    adapter.runner.req_states.num_computed_tokens_np[2] = 12
    adapter.runner.req_states.prefill_len.np[2] = 20
    adapter.runner.model_state.num_accepted_tokens_gpu[2] = 1
    replacement = pool.capture(checkpoint.namespace, 'new', authority,
                               list(range(20)), adapter, replace=True)
    assert replacement.kind == 'P' and replacement.position == 12
    assert not pool.prompts and pool.payload_bytes == replacement.payload_bytes
    assert old_prompt.authority is checkpoint.authority is None
    assert old_prompt.window is checkpoint.window is None
    pool.drop_live(replacement)
    del plan
    assert authority.close() and archive.close()


def test_pending_model_capture_blocks_retirement_claim_and_eviction_without_waiting(tmp_path):
    pool, checkpoint, authority, archive, _, _ = saved(tmp_path)
    fence = Event()
    checkpoint.model = replace(checkpoint.model, completion=fence, source_owners=(object(),))
    amount = pool.payload_bytes
    pool.retire(checkpoint, Event(True))
    assert not pool.idle(checkpoint)
    assert checkpoint.phase is CheckpointPhase.RETIRING
    assert pool.claim(checkpoint.namespace, 'new', list(range(11))) is None
    assert not pool.evict(checkpoint)
    assert checkpoint.model is not None and checkpoint.window is not None
    assert pool.payload_bytes == amount and checkpoint.phase is CheckpointPhase.CLOSING
    fence.done = True
    assert pool.poll_eviction(checkpoint)
    assert pool.payload_bytes == 0 and not pool.entries and archive.close()


def test_replaced_pending_capture_retains_its_payload_credit_and_source_until_ready(tmp_path):
    pool, checkpoint, authority, archive, adapter, _ = saved(tmp_path)
    fence, owner = Event(), object()
    checkpoint.model = replace(checkpoint.model, completion=fence, source_owners=(owner,))
    amount = pool.payload_bytes
    pool.max_payload_bytes = amount
    with pytest.raises(BufferError, match='payload capacity'):
        pool.capture(checkpoint.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert pool.entries[checkpoint.namespace] is checkpoint
    assert pool.payload_bytes == amount and not pool._discarded_captures
    pool.max_payload_bytes = 2 * amount
    successor = pool.capture(checkpoint.namespace, 'a', authority, list(range(11)), adapter, replace=True)
    assert pool.entries[checkpoint.namespace] is successor
    assert pool.payload_bytes == 2 * amount and pool._discarded_captures == [checkpoint]
    assert checkpoint.model.source_owners == (owner,) and checkpoint.window is not None
    pool.drop_live(successor)
    assert not pool.entries and pool.payload_bytes == amount
    assert not pool.poll_captures()
    fence.done = True
    assert pool.poll_captures() and pool.payload_bytes == 0
    assert checkpoint.model is checkpoint.window is checkpoint.authority is None
    assert authority.close() and archive.close()


def test_terminal_m_waits_for_retained_prompt_p_payload_before_advertising_either(tmp_path):
    pool, prompt, authority, archive, adapter, _ = saved(tmp_path)
    pool.retain_prompt = True
    fence = Event()
    prompt.model = replace(prompt.model, completion=fence, source_owners=(object(),))
    terminal_model = adapter.capture('a', 8, max_bytes=4096)
    terminal = pool.capture(prompt.namespace, 'a', authority, list(range(11)),
        SimpleNamespace(capture=lambda *args, **kwargs: terminal_model), replace=True, terminal=True)
    assert pool.prompts[prompt.namespace] is prompt and terminal.kind == 'M'
    pool.retire(terminal, Event(True))
    assert not pool.idle(terminal)
    assert prompt.phase is CheckpointPhase.LIVE and terminal.phase is CheckpointPhase.RETIRING
    fence.done = True
    ready(pool, terminal)
    assert prompt.phase is terminal.phase is CheckpointPhase.IDLE
    assert pool.evict(terminal) and pool.payload_bytes == 0 and archive.close()


def test_retained_prompt_reimport_cannot_reuse_prior_claims_different_suffix_scores(tmp_path):
    pool, prompt, terminal, authority, archive, _ = terminal_with_prompt(tmp_path)
    terminal.action_history = ({0: .9}, 12)
    original = list(range(20))
    assert pool.claim(prompt.namespace, 'new', original, position=8) is prompt
    assert prompt.claim_action_history == ({0: .9}, 12) and prompt.claim_action_end == 12
    plan = pool.prepare_claim_window(prompt, wait=False)
    assert plan is not None
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()
    assert pool.finish_claim(prompt, Event(True), retain=True) is authority
    pool.retire(prompt, Event(True))
    ready(pool, prompt)
    changed = original.copy()
    changed[8:12] = [900, 901, 902, 903]
    assert pool.claim(prompt.namespace, 'another', changed, position=8) is prompt
    assert prompt.claim_action_history is None and prompt.claim_action_end == 0
    del plan
    assert pool.evict(prompt, Event(True)) and archive.close()
