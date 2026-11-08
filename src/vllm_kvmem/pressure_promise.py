"""Opt-in KVMEM_LAYERWISE_PRESSURE_PROMISE: acknowledge a prefill refresh before its plan.

A layerwise prefill pressure refresh keeps exactly select_budget block-aligned
tokens (every layer has the same causal length). The next forward's block
table is a prefix of the current pages, its window offset is
history - selected_tokens, and its native score layout is uniform pages. None
of these depend on *which* blocks are selected, so the refresh can be
acknowledged right after the forward that filled the window.

The acknowledgement registers a plan-less LayerMaterialization. Every KV writer
and reader of every attention layer of the next forward already waits for its
layer's event (pressure_pipeline guards). The producer, on the original refresh
executor, then waits for the chunk's publication, runs the unchanged selection,
commits host maps and uploads layer by layer into that lease. Selection inputs,
selected blocks and KV bytes are identical to the acknowledged-after-plan path;
only the time at which the GPU may start the next forward changes.

The next forward's Action score plan is built from geometry and verified
against the exactly frozen plan after the refresh commits (action_attention).
"""
import os
import threading
import time

import torch


def enabled():
    value = os.getenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_PROMISE must be 0 or 1')
    return value == '1'


def predicted_extent(refresh, job):
    """The selected extent the unchanged picker always produces, or None."""
    memory = refresh.options.memory
    block = memory.block_tokens
    window = job.command.groups[0][1].window_tokens
    history = job.history
    if (window != memory.select_budget or history < window or history % block or
            window % block or len(job.command.groups) != 1 and
            any(t.window_tokens != window for _, t in job.command.groups)):
        return None
    return window


def ineligible(pipeline, refresh, job, capture):
    """Reason this job must take the original acknowledged-after-plan path."""
    command = job.command
    rid = command.request_id
    if (job.done or job.cancelled or job.stage != 'prepare' or
            getattr(command, 'selection', None) != 'pressure' or
            getattr(command, 'pressure_stage', None) != 'prefill' or job.guided_pressure or
            command.query_span is not None or command.input_tokens or
            command.groups[0][1].replay_tokens):
        return 'command'
    from .async_restore import lane_busy
    if (pipeline.active or getattr(pipeline.pressure, 'lanes', 1) != 1 or not pipeline.installed or
            lane_busy(refresh)):
        return 'lane_busy'
    if not getattr(pipeline.pressure, 'single_assembly', False):
        return 'assembly_mode'
    if os.getenv('KVMEM_VERIFY_SESSION_RESTORE') == '1':
        return 'oracle'
    if rid not in refresh.runner.req_states.req_id_to_index:
        return 'native_state'
    authority = capture.authorities.get(rid)
    # A block graph is fine: pressure never consults it, the next forward's
    # graph sources are its sampled positions (asserted equal at capture) and
    # the producer installs the graph binding before _assemble validates it.
    if (authority is None or getattr(authority, 'query_history', None) is not None or
            authority.history_end > job.history):
        return 'authority'
    sessions = refresh.runner._kvmem_sessions
    if sessions is not None and (not getattr(sessions.options, 'session_freeze_prompt', False) or
                                 rid in sessions.captures or rid in sessions.terminal_captures):
        return 'session_checkpoint'
    if predicted_extent(refresh, job) is None:
        return 'extent'
    return None


def promise(pipeline, refresh, job):
    """Engine thread, before the next schedule: ACK with the fixed extent.

    Returns True when the job is acknowledged (job.done, selected_tokens set).
    """
    from .async_refresh import stored_expansions
    from .pressure_pipeline import LayerMaterialization
    capture = refresh.runner._kvmem_capture_runtime
    reason = ineligible(pipeline, refresh, job, capture)
    counts = pipeline.__dict__.setdefault('promise_counts', {})
    if reason is not None:
        counts['declined:' + reason] = counts.get('declined:' + reason, 0) + 1
        from .publication_trace import emit as _trace
        _trace('promise_declined', rid=job.command.request_id, reason=reason, stage=job.stage)
        return False
    rid = job.command.request_id
    # Engine-thread half of the original 'prepare' stage. The page check reads
    # the native table before the scheduler truncates it on this ACK.
    refresh._ensure_window_adapter(job.command)
    sessions = refresh.runner._kvmem_sessions
    if sessions is not None:
        from types import SimpleNamespace
        sessions.before_attention(SimpleNamespace(req_ids=[rid]))
    index = refresh.runner.req_states.req_id_to_index[rid]
    for group, ticket in job.command.groups:
        expansion = stored_expansions(refresh.runner.block_tables)[group]
        expected = [b*expansion+p for b in ticket.original_block_ids for p in range(expansion)]
        count = int(refresh.runner.block_tables.num_blocks.np[group, index])
        actual = refresh.runner.block_tables.block_tables[group].gpu[index, :count].cpu().tolist()
        if actual != expected:
            raise ValueError('Private refresh page reservation mismatch')
    policy = getattr(refresh, 'maintenance', None)
    if policy is not None and not policy.acquire(job):
        counts['declined:maintenance'] = counts.get('declined:maintenance', 0) + 1
        return False
    extent = predicted_extent(refresh, job)
    authority = capture.authorities[rid]
    lease = LayerMaterialization(pipeline, job, None, pipeline.names)
    lease.promised = True
    lease.promise_extent = extent
    lease.committed = threading.Event()
    lease.dispatched = True
    lease.dispatch_at = time.perf_counter()
    job.promised_lease = lease
    job.layerwise_pipeline = lease
    job.authority = authority
    pipeline.active[rid] = lease
    if refresh.stream is None:
        refresh.stream = torch.cuda.Stream(device=refresh.runner.device, priority=0)
    ready = torch.cuda.Event()
    ready.record()
    authority._refresh_background = True
    job.stage = 'assembly'
    job.lane = 0
    job.selected_tokens = extent
    adapter = refresh.runner._kvmem_window_adapter
    action = getattr(capture, 'action', None)
    if action is not None:
        # The chunk that filled the old window is still unpublished: keep its
        # planning identity valid for its logical cursor (_publication_identity).
        action.__dict__.setdefault('promise_epochs', {})[rid] = (
            job.history, getattr(adapter, 'generations', {}).get(rid, 0),
            getattr(adapter, 'offsets', {}).get(rid, 0))
    adapter.commit(rid, job.history, extent)
    if sessions is not None:
        sessions.selection_committed(rid, job.history)
    from .layer0_fast import install
    fast = install(refresh.runner)
    # Opt-in KVMEM_LAYER0_FAST: layer 0 from the GPU shadow, before publication.
    job.layer0_fast = fast.launch(pipeline, refresh, job, lease, capture) if fast is not None else None
    job.future = refresh.executor.submit(producer, pipeline, refresh, job, ready)
    job.done = True
    counts['promised'] = counts.get('promised', 0) + 1
    refresh.emit(job, 'pipeline_promised', selected_tokens=extent)
    return True


def _wait_published(refresh, job, capture, authority):
    rid = job.command.request_id
    condition = capture._publication_lock()
    deadline = time.monotonic() + 600
    waited = time.perf_counter()
    while True:
        with condition:
            capture._raise_publication_error()
            ready = not any(any(r == rid and i not in packet.published
                                for i, r in enumerate(packet.request_ids)) for packet in capture.queue)
            if ready and authority.history_end == job.history:
                return time.perf_counter() - waited
            if authority.history_end > job.history:
                raise RuntimeError('Promised refresh crossed accepted history')
            if time.monotonic() > deadline:
                raise TimeoutError('Promised refresh publication did not complete')
            condition.wait(.002)


def producer(pipeline, refresh, job, ready):
    """Refresh executor: publication wait, unchanged selection and assembly."""
    lease = job.promised_lease
    capture = refresh.runner._kvmem_capture_runtime
    authority = job.authority
    try:
        published = _wait_published(refresh, job, capture, authority)
        refresh.emit(job, 'pipeline_promise_published', publication_wait_seconds=published)
        # Authority half of the original 'prepare' stage.
        from .prompt_spans import install_selection_metadata, selection_metadata
        _m0 = time.perf_counter()
        install_selection_metadata(authority, job.command, validated_prefix=job.history)
        graph = selection_metadata(authority)
        if graph is not None:
            job.graph_binding = graph.binding
        job.assembly_inputs = None
        from .publication_trace import emit as _trace
        _trace('promise_handoff', rid=job.command.request_id, metadata=time.perf_counter()-_m0)
        return refresh._assemble(job, None, ready)
    except BaseException as error:
        if lease.error is None:
            fence = torch.cuda.Event()
            fence.record(torch.cuda.current_stream(refresh.runner.device))
            lease.fail(error, fence)
        lease.committed.set()
        raise


def commit(pipeline, job, lease):
    """Producer thread, once the immutable plan exists: the plan half of dispatch."""
    rid, authority = job.command.request_id, job.authority
    plan = lease.plan
    if plan is None or plan.total_window_tokens != lease.promise_extent:
        raise RuntimeError('Promised refresh selected a different extent than acknowledged')
    authority.memory.commit_layerwise_selection(lease)
    job.layerwise_pipeline_published = True
    job.pipeline_report_records = tuple(authority.memory._store.working_records())
    pipeline.pressure.committed(job)
    lease.dispatch_at = time.perf_counter()
    pipeline.runner._kvmem_async_refresh.emit(job, 'pipeline_dispatch', promised=True,
        prepared_at=lease.created_at, handoff='promised_plan')
    lease.committed.set()


def wait_committed(pipeline, rid, timeout=600.):
    """Engine/scatter threads: block until a promised refresh has committed maps."""
    lease = pipeline.active.get(rid) if pipeline is not None else None
    if lease is None or not getattr(lease, 'promised', False):
        return None
    if not lease.committed.wait(timeout):
        raise TimeoutError('Promised refresh did not commit its host maps')
    if lease.error is not None:
        raise RuntimeError('Promised pressure refresh failed') from lease.error
    return lease
