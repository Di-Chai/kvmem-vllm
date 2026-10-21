"""Offline cursor qualification using the existing CPU archive and graph code."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.action_attention import ActionRuntime
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.capture_runtime import CapturedPacket
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.logical_action import LogicalScorePlan, project_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.logical_publication import LogicalPublicationCursor
from vllm_kvmem.memory_storage import MemoryArchive
from vllm_kvmem.prefill_attention import PrefillHistory
from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.terminal import TerminalPrefix


class Lease:
    pass


@contextmanager
def fixture(tmp_path, *, native_history=False, raw_dtype=torch.bfloat16, chunks=2):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = KVMemConfig(block_tokens=32, select_budget=8192, gen_budget=8192,
                        repository_blocks=1024, sink_blocks=0, recent_blocks=0)
    archive = MemoryArchive(str(tmp_path), 64*(raw_dtype.itemsize+1), 1024, native_payloads=native_history)
    if native_history:
        gc = _native.HistoricalGcConfig()
        gc.background, gc.ready_low, gc.ready_high = False, 2, 4
        gc.scan_blocks, gc.delete_blocks, gc.poll_ms = 16, 4, 1
        archive.enable_history_gc(gc)
    authority = RequestAuthority('r', config, archive, 1, 1, 2, resident_records=256,
                                 raw_dtype=raw_dtype,mean_k_backend='qw3_cuda')
    graph = authority.block_graph = BlockGraph('logical-owner', 32, topk=4)
    state = authority.action_history = NS(prefill=PrefillHistory(NS(
        prefill_attention_half_life=8192, native_prefill_commit=True)), prefill_replay=None)
    action = ActionRuntime.__new__(ActionRuntime)
    action.runtime = NS(output=tmp_path)
    spans = tuple(LogicalPrefillSpan(i, i*2048, i*2048, 2048, i*2048) for i in range(chunks))
    group = LogicalPrefillGroup('r', (1, 0, 0), 7, 0, max(16384,(chunks+1)*2048), 16384, spans,
                               RequestPlan(None, block_tokens=32, gen_budget=8192))
    plans = tuple(LogicalScorePlan(span, tuple(range(31, 2048, 32)),
        tuple(range(0, span.local_end, 32)), (32,)*(span.local_end//32),
        tuple(range(0, span.history_end, 32))) for span in spans)
    masses = np.eye(chunks*64, dtype=np.float32)
    physical_rows = tuple(span.physical_offset+row for span, plan in zip(spans, plans) for row in plan.rows)
    parts = project_logical_scores(group, plans, masses, physical_rows=physical_rows)
    raw = ((torch.arange(group.count*2, dtype=torch.float32)%99)/16).reshape(1,group.count,1,2).to(raw_dtype)
    values = raw.float().to(torch.uint8)
    packet = CapturedPacket(('r',),group.count,NS(owners=(Lease(),)),confirmed=False,
                            request_tokens=(group.count,),source_generation=9)
    cursor = LogicalPublicationCursor(group,packet,authority,action,parts,
                                     owner_epoch=group.owner_epoch,source_generation=9)
    env = NS(authority=authority,archive=archive,graph=graph,state=state,action=action,
             group=group,packet=packet,cursor=cursor,parts=parts,raw=raw,values=values)
    del packet
    try:
        yield env
    finally:
        graph.close()
        assert authority.flush()
        for job,_ in tuple(authority.writes): job.result(5)
        authority.poll()
        # Native retirement is an explicit bounded multi-pass worker protocol.
        for _ in range(256):
            if authority.close():
                break
        else:
            pytest.fail('CPU authority retirement did not finish')
        for _ in range(256):
            if archive.close():
                break
        else:
            pytest.fail('CPU archive retirement did not finish')


def confirm(env, retained=None, *, epoch=8):
    env.packet.confirmed = True
    retained = env.group.count if retained is None else retained
    terminal = None if retained == env.group.count else TerminalPrefix(retained,env.group.count,0,'FINISHED_ABORTED')
    env.cursor.confirm(epoch, owner_epoch=env.group.owner_epoch, terminal_prefix=terminal)


def advance(env):
    receipt = env.cursor.advance(env.raw,env.values,owner_epoch=env.group.owner_epoch)
    if receipt is not None:
        env.cursor.ack(receipt)
    return receipt


def snapshot(env):
    return (env.authority.history_end,env.state.prefill.snapshot(),env.graph.rows,
            env.graph.partial,env.graph.accepted_through,env.graph.revision)


@pytest.mark.parametrize('retained',[0,1,2048,2049,4095,4096])
@pytest.mark.parametrize('raw_dtype',[torch.bfloat16,torch.float8_e4m3fn])
def test_each_logical_commit_matches_original_transactions_and_retained_bytes(tmp_path,retained,raw_dtype):
    with fixture(tmp_path/'new',raw_dtype=raw_dtype) as new, fixture(tmp_path/'old',raw_dtype=raw_dtype) as old:
        confirm(new,retained)
        counts = new.group.accepted_counts(retained)
        for index,(span,part,count) in enumerate(zip(old.group.spans,old.parts,counts)):
            child = NS(action_inputs={'r':part.action_packet(old.state)})
            prepared = old.action.prepare_accept(child,'r',count,graph=old.graph,history_begin=span.history_begin)
            prepared.graph_update.detach()
            assert append_authority_batch([(old.authority,old.raw[:,index*2048:(index+1)*2048],
                old.values[:,index*2048:(index+1)*2048],dict(history_begin=span.history_begin,
                local_rope_begin=span.local_begin,accepted=count))]) == 1
            prepared.commit()
            receipt = advance(new)
            assert snapshot(new) == snapshot(old)
            assert (receipt.logical_index,receipt.scheduled,receipt.accepted,receipt.rejected,
                    receipt.terminal_discarded,receipt.owner_done) == (index,2048,count,0,2048-count,index==1)
            assert receipt.published_history_end == new.authority.history_end
            assert receipt.ready_records == new.authority.memory.repository.ready_slots
            assert (receipt.append_calls,receipt.append_requests,receipt.logical_commits) == (1,1,1)
        assert new.cursor.done and new.cursor.next_index == 2
        assert new.state.prefill.samples == 64*(retained//2048)
        assert new.graph.accepted_through == retained
        for record in new.authority.records.values():
            raw,values = new.authority.read_record(record.handle)
            begin,end = record.history_begin,record.history_begin+record.count
            assert torch.equal(raw.view(torch.uint8),new.raw[:,begin:end].view(torch.uint8))
            assert torch.equal(values,new.values[:,begin:end])


def test_no_confirmation_no_commit_and_receipts_are_monotonic(tmp_path):
    with fixture(tmp_path) as env:
        before = snapshot(env)
        with pytest.raises(RuntimeError,match='not scheduler-confirmed'): advance(env)
        with pytest.raises(RuntimeError,match='confirm the physical'):
            env.cursor.confirm(8,owner_epoch=env.group.owner_epoch)
        assert snapshot(env) == before and env.cursor.error is None
        confirm(env)
        env.cursor.confirm(8,owner_epoch=env.group.owner_epoch)
        with pytest.raises(ValueError,match='Conflicting'):
            env.cursor.confirm(8,owner_epoch=env.group.owner_epoch,terminal_prefix=TerminalPrefix(2049,4096,0,'ABORT'))
        with pytest.raises(ValueError,match='follow|moved backwards'):
            env.cursor.confirm(7,owner_epoch=env.group.owner_epoch)
        advance(env)
        confirm(env,2049,epoch=9)
        with pytest.raises(ValueError,match='retract'):
            env.cursor.confirm(10,owner_epoch=env.group.owner_epoch,terminal_prefix=TerminalPrefix(2047,4096,0,'ABORT'))
        with pytest.raises(ValueError,match='resurrect'):
            env.cursor.confirm(10,owner_epoch=env.group.owner_epoch)
        receipt = advance(env)
        assert receipt.accepted == 1 and env.authority.history_end == 2049


def test_second_graph_budget_failure_keeps_first_prefix_and_whole_lease(tmp_path):
    with fixture(tmp_path) as env:
        confirm(env)
        advance(env)
        before = snapshot(env)
        env.graph.max_bytes_per_session = env.graph.bytes
        with pytest.raises(BufferError) as failure: advance(env)
        assert snapshot(env) == before and env.cursor.next_index == 1
        assert env.cursor.failure_stage == 'preparing' and not env.cursor.archive_done
        assert env.cursor.packet is env.packet and env.packet.download.owners
        assert env.graph._pending_update is None
        with pytest.raises(BufferError) as repeated: advance(env)
        assert repeated.value is failure.value
        with pytest.raises(BufferError): env.cursor.release()


def test_second_archive_capacity_retry_starts_at_second_logical_span(tmp_path,monkeypatch):
    with fixture(tmp_path) as env:
        confirm(env)
        first = advance(env)
        before = snapshot(env)
        original = env.authority._prepare_append
        monkeypatch.setattr(env.authority,'_prepare_append',lambda *a,**kw: None)
        assert advance(env) is None
        assert snapshot(env) == before and env.cursor.next_index == 1 and env.cursor.phase == 'ready'
        assert env.graph._pending_update is None
        assert env.cursor.packet.download.owners
        monkeypatch.setattr(env.authority,'_prepare_append',original)
        second = advance(env)
        assert (first.logical_index,second.logical_index) == (0,1)
        assert env.state.prefill.chunks == 2 and env.state.prefill.samples == 128
        assert env.graph.revision == 2 and env.authority.history_end == 4096


@pytest.mark.parametrize('stage',['append_after_commit','action','report'])
def test_error_after_second_archive_commit_is_permanent_and_never_replays(tmp_path,monkeypatch,stage):
    with fixture(tmp_path) as env:
        confirm(env)
        advance(env)
        failure = OSError('injected '+stage)
        if stage == 'append_after_commit':
            original = env.authority._commit_prepared_append
            def broken(prepared):
                original(prepared)
                raise failure
            monkeypatch.setattr(env.authority,'_commit_prepared_append',broken)
        elif stage == 'action':
            original = env.action.prepare_accept
            def broken(*args,**kwargs):
                prepared = original(*args,**kwargs)
                def commit(): raise failure
                prepared.commit = commit
                return prepared
            monkeypatch.setattr(env.action,'prepare_accept',broken)
        else:
            def report(*args): raise failure
            monkeypatch.setattr(env.action,'_report_prefill',report)
        with pytest.raises(OSError) as caught: advance(env)
        assert caught.value is failure and env.cursor.phase == 'failed'
        assert env.authority.history_end == 4096 and env.cursor.published_history_end == 2048
        assert env.cursor.next_index == 1 and env.cursor.packet.download.owners
        # A throwing append may already have committed internally. Its stage
        # is deliberately not interpreted as permission to retry.
        assert env.cursor.archive_done == (stage != 'append_after_commit')
        assert env.graph._pending_update is None
        after = snapshot(env)
        with pytest.raises(OSError) as again: advance(env)
        assert again.value is failure and snapshot(env) == after


@pytest.mark.parametrize('change',['source','history_state','graph_epoch','owner_epoch','payload'])
def test_next_span_rechecks_owner_generation_state_and_source(tmp_path,change):
    with fixture(tmp_path) as env:
        confirm(env)
        advance(env)
        if change == 'source': env.packet.source_generation += 1
        elif change == 'history_state': env.authority.action_history = NS(prefill=env.state.prefill)
        elif change == 'graph_epoch': env.graph.invalidate()
        elif change == 'payload': env.raw = env.raw.clone()
        epoch = (999,) if change == 'owner_epoch' else env.group.owner_epoch
        with pytest.raises((ValueError,RuntimeError)):
            env.cursor.advance(env.raw,env.values,owner_epoch=epoch)
        assert env.authority.history_end == 2048 and env.cursor.next_index == 1
        assert env.cursor.phase == 'failed'


def test_one_physical_lease_survives_first_span_and_is_only_released_explicitly(tmp_path):
    with fixture(tmp_path) as env:
        reference = weakref.ref(env.packet.download.owners[0])
        confirm(env)
        advance(env)
        del env.packet
        assert reference() is not None
        with pytest.raises(RuntimeError,match='unpublished'): env.cursor.release()
        advance(env)
        assert reference() is not None
        env.cursor.release()
        assert reference() is None


def test_runtime_counter_failure_after_receipt_also_quarantines_completed_native_state(tmp_path):
    with fixture(tmp_path) as env:
        confirm(env)
        receipt = advance(env)
        error = OSError('published counter failed')
        env.cursor.fail(error)
        assert receipt.published_history_end == 2048 and env.cursor.next_index == 1
        with pytest.raises(OSError) as caught: advance(env)
        assert caught.value is error and env.authority.history_end == 2048
        with pytest.raises(OSError): env.cursor.release()


def test_native_history_gc_boundary_is_called_once_per_nonzero_logical_append(tmp_path):
    with fixture(tmp_path,native_history=True) as env:
        assert env.authority.gc.native_history
        original = env.authority.memory._store
        calls = []
        class StoreProxy:
            def __getattr__(self,name): return getattr(original,name)
            def gc_begin_step(self):
                calls.append(env.authority.history_end)
                return original.gc_begin_step()
        env.authority.memory._store = StoreProxy()
        try:
            confirm(env)
            advance(env)
            assert calls == [2048]
            advance(env)
            assert calls == [2048,4096]
        finally:
            env.authority.memory._store = original
            del original


def test_runtime_receipt_must_be_acknowledged_before_next_span_or_source_release(tmp_path):
    with fixture(tmp_path) as env:
        confirm(env)
        receipt = env.cursor.advance(env.raw,env.values,owner_epoch=env.group.owner_epoch)
        assert env.cursor.phase == 'awaiting_ack' and not env.cursor.done
        with pytest.raises(RuntimeError,match='Acknowledge'): advance(env)
        with pytest.raises(ValueError,match='does not own'): env.cursor.ack(NS())
        with pytest.raises(RuntimeError,match='unpublished'): env.cursor.release()
        env.cursor.ack(receipt)
        env.cursor.ack(receipt)  # The same bookkeeping receipt is idempotent.
        second = env.cursor.advance(env.raw,env.values,owner_epoch=env.group.owner_epoch)
        assert second.owner_done and not env.cursor.done
        with pytest.raises(RuntimeError,match='unpublished'): env.cursor.release()
        env.cursor.ack(second)
        assert env.cursor.done


@pytest.mark.parametrize('corruption',['raw_rows','value_rows','value_dtype'])
def test_complete_payload_is_validated_before_first_graph_reservation(tmp_path,corruption):
    with fixture(tmp_path) as env:
        confirm(env)
        if corruption == 'raw_rows': env.raw = env.raw[:,:2048]
        elif corruption == 'value_rows': env.values = env.values[:,:2048]
        else: env.values = env.values.to(torch.int8)
        before = snapshot(env)
        with pytest.raises(ValueError,match='codec'): advance(env)
        assert snapshot(env) == before and env.graph._pending_update is None
        assert env.cursor.phase == 'failed' and env.cursor.next_index == 0


def test_concurrent_terminal_receipt_must_be_deferred_instead_of_reclipping_an_append(tmp_path,monkeypatch):
    with fixture(tmp_path) as env:
        confirm(env)
        entered,release = Event(),Event()
        original = env.authority._commit_prepared_append
        def blocked(prepared):
            entered.set()
            assert release.wait(5)
            return original(prepared)
        monkeypatch.setattr(env.authority,'_commit_prepared_append',blocked)
        with ThreadPoolExecutor(max_workers=1) as executor:
            running = executor.submit(advance,env)
            assert entered.wait(2)
            try:
                with pytest.raises(BlockingIOError):
                    env.cursor.confirm(9,owner_epoch=env.group.owner_epoch,
                        terminal_prefix=TerminalPrefix(0,4096,0,'ABORT'))
                assert env.cursor.confirmation_epoch == 8
            finally:
                release.set()
            assert running.result(5).accepted == 2048
        with pytest.raises(ValueError,match='retract'):
            env.cursor.confirm(9,owner_epoch=env.group.owner_epoch,
                terminal_prefix=TerminalPrefix(0,4096,0,'ABORT'))
        assert env.authority.history_end == 2048
