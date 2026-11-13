"""Opt-in per-layer materialization of an unchanged pressure selection.

An immutable plan (early handoff), or the first ready layer (legacy handoff),
permits *guarded* forwarding, not completion of all KV writes.
Every cache writer/reader waits for its layer. The original producer, page-table
views, source leases and maintenance credit survive through the actual final
fence. Only host selection metadata is published early.
"""
from functools import wraps
import os
from threading import Condition
import time
from . import settings as _settings


class LayerMaterialization:
    def __init__(self, controller, job, plan, names):
        self.controller, self.job, self.plan = controller, job, plan
        self.names = tuple(names)
        self.condition = Condition()
        self.events = [None] * len(names)
        self.submitted_at = [None] * len(names)
        self.waited = set()
        self.consumer_stream = None
        self.error = None
        self.final_event = None
        self.dispatched = False
        self.first_consumer_at = None
        self.host_wait_seconds = 0.
        self.layer_wait_seconds = [0.] * len(names)
        self.created_at = time.perf_counter()
        self.dispatch_at = None
        # Opt-in promised ACK (pressure_promise): acknowledged before its plan.
        self.promised = False
        self.committed = None

    def publish_layer(self, index, event):
        with self.condition:
            if (self.error is not None or self.events[index] is not None or
                    any(e is None for e in self.events[:index])):
                raise RuntimeError('Pressure pipeline layer submission order changed')
            self.events[index] = event
            self.submitted_at[index] = time.perf_counter()
            self.condition.notify_all()

    def fail(self, error, cleanup_event):
        with self.condition:
            self.error, self.final_event = error, cleanup_event
            self.condition.notify_all()

    def first_ready(self):
        if self.error is not None:
            raise RuntimeError('Pressure pipeline producer failed') from self.error
        return self.events[0] is not None and self.events[0].query()

    def dispatchable(self):
        if self.error is not None:
            raise RuntimeError('Pressure pipeline producer failed') from self.error
        # begin() publishes only after the immutable plan/reader and all
        # destinations are validated. The per-layer guards also handle events
        # not yet recorded, so host mapping need not wait for any GPU copy.
        return getattr(self.controller, 'early_handoff', False) or self.first_ready()

    def query(self):
        # A never-recorded CUDA event must never be mistaken for completion.
        return self.final_event is not None and self.final_event.query()

    def validate_dispatch(self, memory):
        # A promised lease is acknowledged (dispatched) before its plan exists;
        # its host maps are committed exactly once, by its producer.
        published = (self.committed.is_set() if self.promised else self.dispatched)
        if (not self.controller.installed or memory is not self.job.authority.memory or
                self.plan is None or memory._plan is not self.plan or published or
                not self.dispatchable()):
            raise RuntimeError('Invalid guarded pressure materialization')

    def wait_layer(self, index, stream):
        started = time.perf_counter()
        from .gil_gate import engine_waiting
        with self.condition, engine_waiting():
            while self.events[index] is None and self.error is None:
                # Releases the GIL; the sole existing producer supplies the
                # event. No wait on an unrecorded event is ever submitted.
                self.condition.wait(.1)
            if self.error is not None:
                raise RuntimeError('Pressure pipeline producer failed') from self.error
            event = self.events[index]
        key = stream.cuda_stream
        if self.consumer_stream is not None and self.consumer_stream != key:
            raise RuntimeError('Pressure consumer stream changed before completion')
        self.consumer_stream = key
        if index not in self.waited:
            stream.wait_event(event)
            self.waited.add(index)
            now = time.perf_counter()
            self.host_wait_seconds += now-started
            self.layer_wait_seconds[index] = now-started
            if self.first_consumer_at is None:
                self.first_consumer_at = now

    def evidence(self):
        return dict(layerwise_pipeline=True, pipeline_layers=len(self.names),
            pipeline_prepared_at=self.created_at,
            pipeline_layer_submitted_at=list(self.submitted_at),
            pipeline_dispatch_at=self.dispatch_at,
            pipeline_waited_layers=len(self.waited),
            pipeline_host_event_wait_seconds=self.host_wait_seconds,
            pipeline_layer_wait_ms=[round(1e3*w, 2) for w in self.layer_wait_seconds],
            pipeline_first_consumer_at=self.first_consumer_at,
            pipeline_last_layer_submitted_at=self.submitted_at[-1],
            pipeline_consumer_before_last_submit=(self.first_consumer_at is not None and
                self.submitted_at[-1] is not None and
                self.first_consumer_at < self.submitted_at[-1]))


class PressurePipeline:
    def __init__(self, pressure):
        self.pressure, self.runner = pressure, pressure.runner
        self.active = {}
        self.installed = False
        self.names = ()
        self.early_handoff = getattr(pressure, 'pipeline_handoff', False)
        self.preforward = getattr(pressure, 'pipeline_preforward', False)
        self.progress = getattr(pressure, 'pipeline_progress', False)
        self.progress_requests = None
        self.dispatch_members = dispatch_batch_members()
        self.fresh_handoff_wait = fresh_handoff_wait()
        self.prepared_condition = Condition()

    def install(self):
        if self.installed:
            return
        from .qsa_adapter import attention_owner
        modules = dict(self.runner.model.named_modules())
        modules.update(getattr(self.runner, '_kvmem_mtp_modules', {}))
        self.names = tuple(getattr(self.runner, '_kvmem_archive_layers', self.pressure.names))
        if self.names[:len(self.pressure.names)] != tuple(self.pressure.names):
            raise ValueError('Pipeline archive order differs from pressure selection')
        implementations = [attention_owner(modules[name]).impl for name in self.names]
        if (len({id(impl) for impl in implementations}) != len(implementations) or
                any(not callable(getattr(impl, 'do_kv_cache_update', None)) for impl in implementations)):
            raise ValueError('Pipeline requires distinct guarded native KV writers')
        for index, impl in enumerate(implementations):
            for method in ('do_kv_cache_update', 'forward'):
                original = getattr(impl, method)
                if method == 'forward':
                    @wraps(original)
                    def guarded(*args, _original=original, _index=index, **kwargs):
                        self.wait_layer(_index)
                        result = _original(*args, **kwargs)
                        # Opt-in streaming publication: this layer's projected
                        # K/V is complete; enqueue its pack/D2H right away.
                        session = getattr(getattr(self.runner, '_kvmem_capture_runtime', None),
                                          '_stream_active', None)
                        if session is not None and _index < session.target_layers:
                            session.on_layer(_index)
                        return result
                else:
                    @wraps(original)
                    def guarded(*args, _original=original, _index=index, **kwargs):
                        self.wait_layer(_index)
                        return _original(*args, **kwargs)
                setattr(impl, method, guarded)
        self.installed = True
        if self.preforward:
            self.install_preforward()
        if self.progress:
            self.install_progress()

    def install_progress(self):
        """Poll completed leases at native boundaries, before the next batch.

        All mutations stay on the original engine thread. These polls never
        consume a new load receipt, wait for preparation, synchronize CUDA or
        enter a private model forward. The old post-forward bounded handoff
        remains available to include a newly prepared plan in this feedback.
        """
        native = self.runner.kv_connector
        original_pre, original_finish = native.pre_forward, native.finish_forward

        @wraps(original_pre)
        def before(scheduler_output, **kwargs):
            output = scheduler_output
            self.progress_requests = None
            result = original_pre(output, **kwargs)
            from .connector import ControlMetadata, RefreshCommand
            control = output.kv_connector_metadata
            if (getattr(native, '_disabled', False) or not isinstance(control, ControlMetadata) or
                    control.cancelled or control.sessions.evictions or output.finished_req_ids or
                    any(not isinstance(c, RefreshCommand) or not c.asynchronous or
                        c.selection != 'pressure' or c.pressure_stage != 'prefill' or
                        c.query_span is not None or c.guided_reason or c.guided_private
                        for c in control.commands)):
                return result
            self.progress_requests = set(output.num_scheduled_tokens)
            self.progress_boundary('native_pre_forward')
            return result

        @wraps(original_finish)
        def finished(*args, **kwargs):
            result = original_finish(*args, **kwargs)
            if not getattr(native, '_disabled', False):
                self.progress_boundary('target_finished')
            self.progress_requests = None
            return result

        native.pre_forward, native.finish_forward = before, finished

    def progress_boundary(self, boundary):
        if self.progress_requests is None:
            return
        refresh = self.runner._kvmem_async_refresh
        # reap checks both the producer future and final GPU event before
        # closing scratch/readers or returning the existing transfer credit.
        self.reap(refresh, boundary=boundary)
        self.prime_post_forward(refresh, max_wait_seconds=0,
            current_requests=self.progress_requests, boundary=boundary)

    def schedule_boundary(self, wait_seconds=0., idle=False):
        """Dispatch already-prepared plans just before the next schedule.

        Opt-in (KVMEM_LAYERWISE_PRESSURE_SCHEDULE_ACK). The engine is
        synchronous and single-process, so this runs on the engine thread after
        the previous batch's feedback and before the next schedule: the same
        worker state as the end of post-forward. It never prepares, waits for a
        plan or synchronizes CUDA; it only reaps finished leases, admits an
        already-prepared waiter to the lane and dispatches prepared plans.
        Returns the request IDs whose guarded handoff was acknowledged here.
        """
        refresh = getattr(self.runner, '_kvmem_async_refresh', None)
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        if refresh is None or capture is None or not self.early_handoff:
            return ()
        from .pressure_promise import enabled as promise_enabled, promise
        if idle and promise_enabled() and self.active:
            # A promised producer normally finishes inside the forward it
            # guarded. Nothing else can run now, so finish reaping it here
            # (bounded) instead of losing the next promise to a busy lane.
            import concurrent.futures
            deadline = time.perf_counter() + .2
            for lease in tuple(self.active.values()):
                future = getattr(lease.job, 'future', None)
                if not getattr(lease, 'promised', False) or future is None:
                    continue
                try:
                    future.exception(timeout=max(0., deadline - time.perf_counter()))
                except concurrent.futures.TimeoutError:
                    continue
                if lease.final_event is not None:
                    lease.final_event.synchronize()
        self.reap(refresh, boundary='schedule')
        lanes = getattr(self.pressure, 'lanes', 1)
        dispatched = []
        if idle and promise_enabled():
            # Opt-in: nothing else can run, so acknowledge with the fixed extent
            # now and let the next forward wait per layer instead of idling.
            for job in tuple(refresh.jobs.values()):
                if promise(self, refresh, job):
                    dispatched.append(job.command.request_id)
            if dispatched:
                return tuple(dispatched)
        for job in tuple(refresh.jobs.values()):
            command = job.command
            rid = command.request_id
            if (job.done or job.cancelled or getattr(command, 'selection', None) != 'pressure' or
                    getattr(command, 'pressure_stage', None) != 'prefill' or job.guided_pressure or
                    not capture.publication_ready((rid,))):
                continue
            authority = capture.authorities.get(rid)
            if authority is None or authority.history_end != job.history:
                continue
            if job.stage == 'assembly_wait' and len(self.active) < lanes:
                # Lane/byte admission only: submits the existing producer.
                refresh.step(job)
            if job.stage != 'assembly' or job.future is None:
                continue
            if getattr(job, 'layerwise_pipeline', None) is None and wait_seconds > 0:
                # Only called when no other request could run this step: the GPU
                # is idle anyway, so wait (GIL released) for the producer's plan.
                deadline = time.perf_counter() + wait_seconds
                with self.prepared_condition:
                    while (getattr(job, 'layerwise_pipeline', None) is None and
                           not job.future.done() and time.perf_counter() < deadline):
                        self.prepared_condition.wait(min(.002, max(0., deadline-time.perf_counter())))
            if getattr(job, 'layerwise_pipeline', None) is None:
                if job.future.done():
                    job.future.result()  # Do not swallow preparation failures.
                continue
            started = time.perf_counter()
            if self.dispatch(refresh, job):
                refresh.emit(job, 'pipeline_handoff_attempt', dispatched=True,
                    host_seconds=time.perf_counter()-started, prepared=True, boundary='schedule')
                dispatched.append(rid)
        return tuple(dispatched)

    def install_preforward(self):
        native = self.runner.kv_connector
        if not callable(getattr(native, '_start_load_kv', None)):
            raise ValueError('Early pressure preparation needs native deferred-load support')
        original = native.pre_forward

        @wraps(original)
        def before(scheduler_output, **kwargs):
            original(scheduler_output, **kwargs)
            self.start_preforward(native, scheduler_output)
        native.pre_forward = before

    def start_preforward(self, native, output):
        """Queue pure pressure control at the native pre-forward boundary.

        Native request states, block tables and peer attention metadata have
        already been prepared here. Mixed control and retirement keep their
        original lifecycle ordering. _start_load_kv clears its pending receipt,
        so native post-forward cannot enqueue the commands a second time.
        """
        from .connector import ControlMetadata, RefreshCommand
        control = output.kv_connector_metadata
        if (not self.preforward or getattr(native, '_disabled', False) or
                not isinstance(control, ControlMetadata) or not control.commands or
                control.cancelled or control.sessions.evictions or output.finished_req_ids):
            return
        scheduled = set(output.num_scheduled_tokens)
        if any(not isinstance(c, RefreshCommand) or not c.asynchronous or
               c.selection != 'pressure' or c.pressure_stage != 'prefill' or
               c.query_span is not None or c.guided_reason or c.guided_private or c.request_id in scheduled
               for c in control.commands):
            return
        refresh = self.runner._kvmem_async_refresh
        if native._pending_load_kwargs is not None:
            native._start_load_kv()
            for c in control.commands:
                refresh.emit(refresh.jobs[c.request_id], 'pipeline_preforward_start')
        # No plan wait or extra scheduling here. The existing executor can
        # prepare/upload while the native peer batch runs; ACK is still sent
        # only through the native completion feedback after that forward.
        self.prime_post_forward(refresh, max_wait_seconds=0,
            current_requests=scheduled, boundary='pre_forward')

    def begin(self, job, plan, names):
        if not self.installed or tuple(names) != self.names:
            raise RuntimeError('Pipeline cache guards are not installed for every layer')
        promised = getattr(job, 'promised_lease', None)
        if promised is not None:
            # Producer thread: bind the immutable plan to the already
            # acknowledged lease, then commit host maps before any upload.
            if promised.plan is not None or self.active.get(job.command.request_id) is not promised:
                raise RuntimeError('Promised pressure lease was replaced or bound twice')
            with promised.condition:
                promised.plan = plan
            # Host maps commit after layer 0's upload is submitted (or at once
            # when the assembler has no per-layer hook): uploads need only the
            # immutable plan; layer-1 selection and capture wait for the commit.
            from .pressure_promise import commit
            if not getattr(job, 'promise_commit_after_layer0', False):
                commit(self, job, promised)
            return promised
        lease = LayerMaterialization(self, job, plan, names)
        with self.prepared_condition:
            job.layerwise_pipeline = lease
            self.prepared_condition.notify_all()
        return lease

    def prime_post_forward(self, refresh, *, max_wait_seconds=.064,
                           current_requests=None, boundary='post_forward'):
        """Hand off one accepted pressure plan in the current feedback turn.

        Native async commands arrive only at post-forward. Waiting until the
        next pre-forward pump adds a complete peer batch before their ACK.
        Start the existing producer now and wait a bounded interval for its
        immutable plan, never for all KV uploads. Only this engine thread
        commits maps; the producer keeps every original lane/byte lease.
        """
        if not self.early_handoff:
            return
        capture = getattr(self.runner, '_kvmem_capture_runtime', None)
        if capture is None:
            return
        current = (set(getattr(getattr(self.runner, '_kvmem_last_batch', None), 'req_ids', ()))
                   if current_requests is None else set(current_requests))
        if self.preforward:
            max_wait_seconds = 0
        started = time.perf_counter()
        deadline = started + max_wait_seconds
        acknowledged = False
        for job in tuple(refresh.jobs.values()):
            command = job.command
            rid = command.request_id
            if (job.done or job.cancelled or getattr(command, 'selection', None) != 'pressure' or
                    getattr(command, 'pressure_stage', None) != 'prefill' or job.guided_pressure or
                    rid in current or job.stage not in ('prepare', 'assembly_wait', 'assembly') or
                    not capture.publication_ready((rid,))):
                continue
            authority = capture.authorities.get(rid)
            if authority is None or authority.history_end != job.history:
                continue
            if (self.progress and len(self.active) >= getattr(self.pressure, 'lanes', 1) and
                    job.stage in ('prepare', 'assembly_wait')):
                # An early ACK leaves its scratch lease in active even when
                # the original jobs map no longer contains that done job.
                # Preserve one copy lane also without a maintenance policy.
                continue
            fresh = False
            if job.stage in ('prepare', 'assembly_wait'):
                before = job.stage
                refresh.step(job)  # Existing publication, page, lane and byte checks.
                refresh.emit(job, 'pipeline_handoff_prime', previous_stage=before, boundary=boundary)
                fresh = job.stage == 'assembly'
            if job.stage != 'assembly' or job.future is None:
                continue
            if not self.progress:
                deadline = time.perf_counter() + max_wait_seconds
            if fresh and not self.fresh_handoff_wait:
                # The producer was submitted in this very call: its plan takes
                # longer than the bounded wait about half the time, and nearly
                # always other requests can run meanwhile. Do not hold the
                # engine; a later boundary (or the idle schedule wait, when
                # nothing else is runnable) acknowledges the prepared plan.
                if job.future.done():
                    job.future.result()  # Do not swallow preparation failures.
            else:
                from .gil_gate import engine_waiting
                with self.prepared_condition, engine_waiting():
                    while getattr(job, 'layerwise_pipeline', None) is None:
                        if job.future.done():
                            job.future.result()  # Do not swallow preparation failures.
                            break
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self.prepared_condition.wait(min(.001, remaining))
            dispatched = self.dispatch(refresh, job)
            refresh.emit(job, 'pipeline_handoff_attempt', dispatched=dispatched,
                host_seconds=time.perf_counter()-started,
                prepared=getattr(job, 'layerwise_pipeline', None) is not None, boundary=boundary)
            if not self.progress or not dispatched or acknowledged:
                break
            # A ready ACK must not consume the whole feedback opportunity.
            # Reap only an actually completed copy, then let the next waiter
            # try the original single-lane/byte admission once. The bounded
            # preparation wait above is shared across the entire call.
            acknowledged = True
            self.reap(refresh, boundary=boundary)
            if len(self.active) >= getattr(self.pressure, 'lanes', 1):
                break

    @property
    def needs_dispatch_boundary(self):
        """Whether a whole-model FULL graph replay must give way to Python.

        The per-layer guards wait only for leases whose owner is in the batch
        being run. With KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS=1 a batch that
        carries no lease owner therefore keeps its graph; without the flag, or
        when the runner did not announce the batch, any lease forces Python.
        """
        if not self.active:
            return False
        requests = getattr(self.runner, '_kvmem_dispatch_requests', None)
        if not self.dispatch_members or requests is None:
            return True
        if any(rid in requests for rid in self.active):
            return True
        # prepare_attn re-checks the prepared batch before the graph replays.
        self.runner._kvmem_dispatch_relaxed = True
        native = getattr(self.runner, '_kvmem_native_attention', None)
        if native is not None:
            native.counts['dispatch:full_graph_beside_lease'] += 1
        return False

    def verify_relaxed_dispatch(self, batch):
        """Fail closed if a kept FULL graph would bypass a lease of its batch."""
        if any(rid in self.active for rid in getattr(batch, 'req_ids', ())):
            raise RuntimeError('A FULL graph batch contains an outstanding pressure pipeline owner')

    def wait_layer(self, index):
        if not self.active:
            return
        batch = getattr(self.runner, '_kvmem_last_batch', None)
        requests = set(getattr(batch, 'req_ids', ()))
        leases = [lease for rid, lease in self.active.items() if rid in requests]
        if not leases:
            return
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('An outstanding pressure pipeline entered CUDA graph capture')
        stream = torch.cuda.current_stream(self.runner.device)
        for lease in leases:
            if lease.error is not None:
                raise RuntimeError('Pressure pipeline producer failed') from lease.error
            if not lease.query():
                lease.wait_layer(index, stream)

    def pending(self, rid):
        lease = self.active.get(rid)
        return lease is not None and not lease.query()

    def dispatch(self, refresh, job):
        lease = getattr(job, 'layerwise_pipeline', None)
        if lease is None or job.cancelled or not lease.dispatchable():
            return False
        if job.pressure_oracle is not None:
            # KVMEM_VERIFY_SESSION_RESTORE: the whole-window byte oracle needs
            # completed materialization, so keep this job on the original
            # non-pipelined completion path (the promise path declines too).
            return False
        rid, authority = job.command.request_id, job.authority
        if rid in self.active:
            raise RuntimeError('Overlapping materializations for one owner')
        sessions = self.runner._kvmem_sessions
        if sessions is not None and not getattr(sessions.options, 'session_freeze_prompt', False):
            from .layerwise_checkpoints import sessions_enabled
            if not sessions_enabled():
                raise ValueError('Pipeline qualification requires immutable prompt checkpoints')
        if sessions is not None and (rid in sessions.captures or rid in sessions.terminal_captures or
                                     updates_live_window(sessions, rid, job.history)):
            # A checkpoint copies the whole window before entering any model
            # layer. It cannot use per-layer attention guards. The same holds
            # when selection_committed below refreshes a live P/M checkpoint at
            # this exact position (update_live_window -> checkpoint_window).
            if not lease.query():
                if not getattr(job, 'pipeline_checkpoint_wait_reported', False):
                    refresh.emit(job, 'pipeline_checkpoint_boundary_wait')
                    job.pipeline_checkpoint_wait_reported = True
                return False
        from . import publication_trace
        marks = [] if publication_trace.enabled() else None
        def mark(name):
            if marks is not None:
                marks.append((name, time.perf_counter(), time.thread_time()))
        mark('start')
        authority.memory.commit_layerwise_selection(lease)
        mark('commit_selection')
        lease.dispatched = True
        self.active[rid] = lease
        job.layerwise_pipeline_published = True
        job.selected_tokens = lease.plan.total_window_tokens
        job.pipeline_report_records = tuple(authority.memory._store.working_records())
        mark('working_records')
        self.pressure.committed(job)
        mark('pressure_committed')
        self.runner._kvmem_window_adapter.commit(rid, job.history, job.selected_tokens)
        mark('window_commit')
        if sessions is not None:
            sessions.selection_committed(rid, job.history)
        mark('sessions')
        refresh._release(job)
        mark('release')
        if marks is not None:
            publication_trace.emit('dispatch_sections', request_id=rid, sections={
                b[0]: [round(b[1]-a[1], 6), round(b[2]-a[2], 6)] for a, b in zip(marks, marks[1:])})
        # ACK means guarded execution is now legal. It does not release the
        # assembler, source plan, scratch or maintenance credit.
        job.done = True
        lease.dispatch_at = time.perf_counter()
        refresh.emit(job, 'pipeline_dispatch', layers_ready=sum(
            e is not None and e.query() for e in lease.events),
            layers_submitted=sum(e is not None for e in lease.events),
            prepared_at=lease.created_at, first_layer_submitted_at=lease.submitted_at[0],
            handoff='prepared_plan' if self.early_handoff else 'first_layer_ready')
        return True

    def reap(self, refresh, *, boundary=None):
        for rid, lease in tuple(self.active.items()):
            job = lease.job
            if not job.future.done():
                continue
            if job.result is None:
                job.result = job.future.result()  # Producer errors remain fatal.
            assembler, assembly, plan, fit, seconds = job.result
            if not assembly.event.query():
                continue
            if plan is not lease.plan or not lease.query():
                raise RuntimeError('Pipeline lost its final materialization fence')
            job.authority._refresh_background = False
            job.authority.memory.finish_layerwise_selection(lease)
            assembly.read_stats.update(lease.evidence())
            if not assembler.close():
                raise RuntimeError('Completed pipeline retained its assembly scratch')
            if refresh.maintenance is not None:
                refresh.maintenance.observe(job, assembly.uploaded_bytes)
                refresh.maintenance.release(job)
            refresh.report(job, assembly, fit, seconds)
            if lease.promised:
                refresh._release(job)
                job.promised_lease = None
                action = getattr(getattr(self.runner, '_kvmem_capture_runtime', None), 'action', None)
                epochs = getattr(action, 'promise_epochs', {})
                if epochs.get(rid, (None,))[0] == job.history:
                    del epochs[rid]
            refresh.emit(job, 'pipeline_completed', completion_poll=boundary or 'original', **lease.evidence())
            job.future = job.result = None
            job.layerwise_pipeline = None
            # Every layer is selected and published: the lazy selection closure
            # (which references this job, its histories and source records)
            # is no longer needed. Dropping it breaks a job <-> closure cycle.
            job.layerwise_select_layer = None
            lease.job = lease.plan = None  # Break cross-thread ownership cycles.
            del self.active[rid]


def updates_live_window(sessions, request_id, history):
    """True when selection_committed(request_id, history) will snapshot the whole
    selected window into the live P/M checkpoint at this exact position
    (SessionWorker.selection_committed -> update_live_window -> checkpoint_window).
    That snapshot requires every layer to be materialized, so an early ACK must
    wait for the whole lease, exactly like a checkpoint capture."""
    if getattr(getattr(sessions, 'options', None), 'session_freeze_prompt', False):
        return False   # frozen prompts never adopt a later window
    from .pressure_promise import _checkpoint_at
    return _checkpoint_at(sessions, request_id, history)


def dispatch_batch_members():
    """KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS=1: an outstanding materialization
    forces Python execution only for batches that contain its owner. Default
    off: any outstanding materialization disables FULL graph replay."""
    value = _settings.get('KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PIPELINE_DISPATCH_BATCH_MEMBERS must be 0 or 1')
    return value == '1'


def fresh_handoff_wait():
    """KVMEM_PRESSURE_HANDOFF_FRESH_WAIT=0: post-forward does not wait for the
    plan of a producer it has just submitted (results are identical; only the
    boundary at which the handoff is acknowledged changes). Default 1."""
    value = _settings.get('KVMEM_PRESSURE_HANDOFF_FRESH_WAIT', '1')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_HANDOFF_FRESH_WAIT must be 0 or 1')
    return value == '1'


def swaps_enabled():
    """KVMEM_LAYERWISE_PIPELINE_SWAPS: run the pipeline with incremental GPU
    admission and exact layerwise request swaps (default off: static only)."""
    value = _settings.get('KVMEM_LAYERWISE_PIPELINE_SWAPS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_PIPELINE_SWAPS must be 0 or 1')
    return value == '1'


def pipeline_for(refresh):
    return getattr(getattr(getattr(refresh, 'runner', None), '_kvmem_layerwise_pressure', None), 'pipeline', None)
