"""Real native graph transactions at refresh boundaries; GPU copying is isolated."""
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
import sys
import time
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native as n
from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
from vllm_kvmem.historical_graph import compact_retired, retry_selection, synchronous_selection
from vllm_kvmem.memory import Phase
from test_historical_graph import fixture, retire
from test_historical_authority import close, Ready
from test_graph_lifecycle import complete


def manager(authority, monkeypatch):
    # Message typing and device-copy code are not part of these host protocol
    # tests. Keep the actual AsyncRefresh methods, native store and graph owner.
    monkeypatch.setitem(sys.modules, 'vllm_kvmem.connector',
                        NS(RestoreCommand=type('RestoreCommand', (), {})))
    obj = AsyncRefresh.__new__(AsyncRefresh)
    obj.options = NS(memory=authority.memory.config, pressure_policy='graph',
                     packed_assembly=False, direct_pinned_reads=False)
    obj.runner = NS(device='cpu', _kvmem_capture_runtime=NS(
        names=(), authorities={authority.memory.request_id: authority}))
    obj.stream = NS(wait_event=lambda event: None)
    obj.maintenance = None
    obj.events = []
    obj.emit = lambda job, stage, **fields: obj.events.append(stage)
    command = NS(request_id=authority.memory.request_id, command_id=1,
        selection='pressure', guided_reason='', input_tokens=(), query_span=None,
        mandatory_spans=(), groups=((0, NS(history_end=authority.history_end, window_tokens=12)),))
    job = RefreshJob(command)
    job.authority, job.graph_binding = authority, authority.block_graph.binding
    return obj, job


def test_actual_async_selection_uses_its_cow_revision_at_assembly_commit(tmp_path, monkeypatch):
    a, archive, worker, graph, handles = fixture(tmp_path)
    obj, job = manager(a, monkeypatch)
    old = job.graph_binding
    plan = result = None
    class Assembler:
        @classmethod
        def from_runner(cls, *args, **kwargs):
            return cls()
        def assemble(self, authority, selected):
            assert authority is a and authority.memory.phase is Phase.ASSEMBLING
            assert job.graph_binding == graph.binding == a._graph_selection_binding
            return NS(event=Ready(), uploaded_bytes=0)
    monkeypatch.setitem(sys.modules, 'vllm_kvmem.window_assembly',
                        NS(NativeWindowAssembler=Assembler))
    monkeypatch.setattr(torch.cuda, 'device', lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, 'stream', lambda *args: nullcontext())
    try:
        retire(a, worker, handles[1])
        result = obj._assemble(job, None, object())
        plan = result[2]
        assert graph.binding != old and job.graph_binding == graph.binding
        assert graph.rebind_after_gc(old) == graph.binding
        with pytest.raises(RuntimeError, match='Stale'):
            graph.validate_binding(old)
        assert handles[1].slot not in {row.block_id for row in plan.remaps}
        complete(a)
        result = plan = None
    finally:
        result = plan = None
        close(a, archive)


def test_async_retry_yields_credit_keeps_queries_and_can_cancel_without_resubmission(tmp_path, monkeypatch):
    a, archive, worker, graph, handles = fixture(tmp_path)
    obj, job = manager(a, monkeypatch)
    query, ready = torch.zeros(2, 1, 1, 2), Ready()
    releases = []
    obj.maintenance = NS(release=lambda value: releases.append(value.command.request_id))
    try:
        retire(a, worker, handles[1])
        compact_retired(a)  # A failed attempt may already have compacted its graph.
        job.stage = 'assembly'
        job.selection_inputs = (query, ready)
        job.future = Future()
        job.future.set_exception(n.HistoryAdmissionRetry('busy selection'))
        a._refresh_background = True
        assert not obj._poll_assembly_result(job)
        assert job.stage == 'assembly_wait' and job.future is None
        assert job.assembly_inputs == (query, ready) and job.selection_inputs is None
        assert job.graph_binding == graph.binding
        assert not a._refresh_background and releases == [a.memory.request_id]
        assert obj.events == ['selection_retry']
        # Cancellation uses the normal restored-state path and drops retained
        # query tensors. It neither resubmits nor returns transfer credit twice.
        job.cancelled, job.restore_event = True, Ready()
        job.query_rows.append(query)
        assert obj.step(job)
        assert job.done and not job.query_rows
        assert job.assembly_inputs is job.selection_inputs is None
        assert releases == [a.memory.request_id] and obj.events[-1] == 'cancelled'
    finally:
        close(a, archive)


def test_retry_budget_is_constant_and_synchronous_backoff_is_refresh_only(tmp_path, monkeypatch):
    a, archive, worker, graph, handles = fixture(tmp_path)
    calls, waits = [], []
    monkeypatch.setattr('vllm_kvmem.historical_graph._SYNC_RETRY_PAUSE',
                        NS(wait=lambda seconds: waits.append(seconds)))
    def busy():
        calls.append(True)
        raise n.HistoryAdmissionRetry('policy gate busy')
    try:
        with pytest.raises(n.HistoryAdmissionRetry):
            retry_selection(a, busy)
        assert len(calls) == 3 and not waits
        calls.clear()
        with pytest.raises(n.HistoryAdmissionRetry):
            synchronous_selection(a, lambda: retry_selection(a, busy))
        assert len(calls) == 9 and waits == [.001, .001]
        assert a.memory.phase is Phase.READY
    finally:
        close(a, archive)

def test_async_checkpoint_pressure_yields_before_model_copy_and_retries(tmp_path, monkeypatch):
    from test_graph_lifecycle import graph_checkpoints
    from test_graph_pressure_v2 import configure
    from vllm_kvmem import prefill_attention
    pool, prompt, terminal, a, archive, adapter, graph = graph_checkpoints(tmp_path)
    configure(a)
    checkpoint = pool.claim(prompt.namespace, 'new', list(range(24)), position=8)
    original = prefill_attention.select_pressure
    attempts = []
    plan = None
    def choose(*args, **kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise n.HistoryAdmissionRetry('injected checkpoint selection contention')
        return original(*args, **kwargs)
    monkeypatch.setattr(prefill_attention, 'select_pressure', choose)
    try:
        assert pool.prepare_claim_window(checkpoint, pressure=True, wait=False) is None
        assert a.memory.phase is Phase.READY and not checkpoint.restore_started
        plan = pool.prepare_claim_window(checkpoint, pressure=True, wait=False)
        assert plan is not None and len(attempts) == 2
        assert checkpoint.restore_started and a._graph_selection_binding == graph.binding
        complete(a)
        assert pool.finish_claim(checkpoint, Ready()) is a
        plan = None
    finally:
        plan = None
        assert a.close() and archive.close()



@pytest.mark.parametrize('wait', [False, True])
def test_exact_prefix_pressure_retry_obeys_original_claim_deadline(tmp_path, monkeypatch, wait):
    from test_prompt_checkpoints import saved, ready
    from vllm_kvmem import prefill_attention
    pool, checkpoint, a, archive, adapter, _ = saved(tmp_path)
    pool.retire(checkpoint, Ready())
    ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(11))) is checkpoint
    assert a.history_end == checkpoint.position and not checkpoint.rewind_started
    def forbidden(*args, **kwargs):
        pytest.fail('Expired restore must fail before attempting selection')
    monkeypatch.setattr(prefill_attention, 'select_pressure', forbidden)
    checkpoint.restore_deadline = time.monotonic() - 1
    try:
        with pytest.raises(TimeoutError, match='restore deadline'):
            pool.prepare_claim_window(checkpoint, pressure=True, wait=wait)
        assert a.memory.phase is Phase.READY and not checkpoint.restore_started
    finally:
        assert pool.evict(checkpoint, Ready()) and archive.close()


def test_cancelled_restore_after_selection_busy_finishes_fence_and_terminal_release(tmp_path, monkeypatch):
    from test_graph_lifecycle import graph_checkpoints
    from test_graph_pressure_v2 import configure
    from test_prompt_checkpoints import ready
    from test_core import Event
    from vllm_kvmem import prefill_attention
    from vllm_kvmem.prompt_checkpoints import CheckpointPhase
    pool, prompt, checkpoint, a, archive, adapter, graph = graph_checkpoints(tmp_path)
    configure(a)
    obj, _ = manager(a, monkeypatch)
    command = sys.modules['vllm_kvmem.connector'].RestoreCommand()
    command.__dict__.update(request_id='new', command_id=-1, generation=3,
        history_end=12, window_tokens=8, block_ids=((1, 2),), resident=False,
        asynchronous=True)
    job = RefreshJob(command, dispatched=True)
    obj.jobs, obj.deferred = {'new': job}, {}
    trace, submitted = [], []
    binding = NS(namespace=checkpoint.namespace, generation=3, history=12,
        window=8, tokens=list(range(24)), pressure_restore=True, mandatory_spans=())
    obj.runner.req_states = NS(req_id_to_index={'new': 0})
    obj.runner._kvmem_probe_output = tmp_path
    obj.runner._kvmem_restore_allocations = {'new': NS(validate=lambda *args, **kw: None)}
    worker = NS(pool=pool, bindings={'new': binding}, imports={'new'},
        model_adapter=NS(
            prepare_restore=lambda *args, **kw: trace.append('model') or object(),
            validate_restore=lambda model: trace.append('validate')),
        _verify_restored=lambda *args: trace.append('verify'),
        publish_restore=lambda *args, **kw: trace.append('publish'))
    obj.runner._kvmem_sessions = worker
    obj.maintenance = NS(transfer_budget=None,
        acquire=lambda job: trace.append('acquire') or True,
        observe=lambda job, size: trace.append('observe'),
        release=lambda job: trace.append('release'))
    class Assembler:
        @classmethod
        def from_runner(cls, *args, **kw):
            trace.append('assembler')
            return cls()
        def close(self):
            trace.append('close')
            return True
    monkeypatch.setitem(sys.modules, 'vllm_kvmem.window_assembly',
                        NS(NativeWindowAssembler=Assembler))
    monkeypatch.setattr(torch.cuda, 'Event', lambda: NS(record=lambda: None))
    def submit(*args):
        trace.append('submit')
        future = Future()
        submitted.append(future)
        return future
    obj.executor = NS(submit=submit)
    original = prefill_attention.select_pressure
    attempts = []
    def choose(*args, **kw):
        attempts.append(True)
        if len(attempts) == 1:
            raise n.HistoryAdmissionRetry('injected cancelled restore contention')
        return original(*args, **kw)
    monkeypatch.setattr(prefill_attention, 'select_pressure', choose)
    @dataclass
    class Control:
        commands: tuple = ()
        cancelled: frozenset = frozenset()
        terminal_prefixes: dict = field(default_factory=dict)
    @dataclass
    class Output:
        finished_req_ids: set
        kv_connector_metadata: Control
    sys.modules['vllm_kvmem.connector'].ControlMetadata = Control
    try:
        assert not obj.step(job)
        assert job.checkpoint is checkpoint and checkpoint.phase is CheckpointPhase.CLAIMED
        assert a.memory.phase is Phase.READY and not checkpoint.restore_started
        assert job.future is job.restore_context is None and not trace and not submitted
        held = obj.defer_terminals(Output({'new'}, Control(
            cancelled=frozenset({'new'}), terminal_prefixes={'new': 'owned-prefix'})))
        assert job.cancelled and not held.finished_req_ids and obj.deferred == {'new': 'owned-prefix'}
        assert obj.step(job) and len(attempts) == 2
        assert trace == ['model', 'assembler', 'acquire', 'submit']
        assert a.memory.phase is Phase.ASSEMBLING and checkpoint.restore_started
        fence = Event()
        assembly = NS(query=fence.query, uploaded_bytes=128, host_prepare_seconds=.002)
        submitted[0].set_result((assembly, None, fence))
        assert not obj.step(job) and not job.done
        assert 'new' in worker.imports and 'new' in obj.runner._kvmem_restore_allocations
        assert 'close' not in trace and 'release' not in trace
        fence.done = True
        assert obj.step(job) and job.done
        assert job.restore_context is job.future is job.checkpoint is None
        assert checkpoint.phase is CheckpointPhase.LIVE and checkpoint.request_id == 'new'
        assert not worker.imports and not obj.runner._kvmem_restore_allocations
        assert not a._refresh_background and a.memory.phase is Phase.READY
        assert trace[-6:] == ['validate', 'verify', 'publish', 'close', 'observe', 'release']
        released = obj.defer_terminals(Output(set(), Control()))
        assert released.finished_req_ids == {'new'} and not obj.deferred
        assert released.kv_connector_metadata.terminal_prefixes == {'new': 'owned-prefix'}
        pool.retire(checkpoint, Ready())
        ready(pool, checkpoint)
        assert pool.evict(checkpoint) and archive.close()
    finally:
        # Successful path evicts through the real pool, releasing graph snapshots,
        # native checkpoint pins and the archived payload after the copy fence.
        if checkpoint.authority is not None and job.done:
            pool.retire(checkpoint, Ready())
            ready(pool, checkpoint)
            assert pool.evict(checkpoint)
        assert a.close() and archive.close()
