"""Continuous accepted-prefix archive runtime for the native target runner.

Target layers share bounded packed D2H storage. Small decode requests can use an
independent pool/event from bulk prefill; rejection counts share their owner's
fenced packet. This stage provides
the immutable authority for refresh, but does not yet trigger retrieval/replay.
Finalization always drains I/O; diagnostic checksum readback is separately gated.
"""
from dataclasses import asdict, dataclass, field, replace
from contextlib import nullcontext
import hashlib
import json
import os
import time
from threading import Condition, Event, RLock, get_ident
from concurrent.futures import ThreadPoolExecutor

import torch

from .authority import RequestAuthority
from .diagnostics import CompletionHistory, completion_counts
from .options import RuntimeOptions
from .storage import make_shared_archive
from .transfer import PackedDownloads
from .capture_feedback import EarlyCaptureFeedback


@dataclass
class CapturedPacket:
    request_ids: tuple[str, ...]
    tokens: int
    download: object
    next_request: int = 0
    query_inputs: dict = field(default_factory=dict)
    action_inputs: dict = field(default_factory=dict)
    # Hand-built diagnostic packets are already scheduler-confirmed. Serving
    # submit explicitly sets this false until the next scheduler decision.
    confirmed: bool = True
    request_tokens: tuple[int, ...] = ()
    published: set[int] = field(default_factory=set)
    consuming: set[int] = field(default_factory=set)
    lane: str = 'bulk'
    source_copied: dict = field(default_factory=dict)
    source_feedback_bytes: dict = field(default_factory=dict)
    source_released: bool = False
    source_generation: int = 0
    # Fully validated, accounted graph/Action reservations survive source-copy
    # handoff; their owner locks are detached before another thread publishes.
    graph_updates: dict | None = None


class CaptureRuntime(EarlyCaptureFeedback):
    def __init__(self, runner, output):
        self.runner = runner
        self.output = output
        self.options = RuntimeOptions.from_vllm(runner.vllm_config)
        self.config = self.options.memory
        modules = dict(runner.model.named_modules())
        modules.update(getattr(runner, '_kvmem_mtp_modules', {}))
        self.names = tuple(getattr(runner, '_kvmem_archive_layers', runner._kvmem_layers))
        self.layers = [modules[name] for name in self.names]
        heads = {module.num_kv_heads for module in self.layers}
        dims = {module.head_dim for module in self.layers}
        if len(heads) != 1 or len(dims) != 1:
            raise ValueError("First continuous archive requires uniform target KV geometry")
        if any(module.attn.kv_cache_dtype != "fp8_e4m3" for module in self.layers):
            raise ValueError("Continuous target archive currently qualifies FP8 e4m3")
        self.heads, self.dim = heads.pop(), dims.pop()
        self.scales = torch.stack([module.attn._v_scale.reshape(()) for module in self.layers])
        if not torch.all(torch.isfinite(self.scales) & (self.scales > 0)).item():
            raise ValueError("Invalid target V scales")
        self.row_elements = len(self.layers) * self.heads * self.dim
        self.raw_dtype = torch.float8_e4m3fn
        record_bytes = self.row_elements * self.config.block_tokens * 2
        if self.options.cpu_history_coordination:
            from .cache_manifest import archive_bytes_per_token
            if archive_bytes_per_token(runner.kv_cache_config) != self.row_elements * 2:
                raise ValueError('Worker CPU authority differs from scheduler credit geometry')
        output.mkdir(parents=True, exist_ok=True)
        max_tokens = runner.vllm_config.scheduler_config.max_num_batched_tokens
        max_reqs = runner.vllm_config.scheduler_config.max_num_seqs
        self._early_owner_limit = max_reqs
        self.gc_config = None
        self.gc_progress = Event()
        self.gc_wait_seconds = 0.0
        self.gc_wait_count = 0
        self._archive_wait_deadline = None
        self._cpu_scatter_executor = None
        self._cpu_scatter_future = None
        self._publication_condition = Condition(RLock())
        self._publication_error = None
        self._published_capacity = {}
        self._progress_lock = RLock()
        self.cpu_scatter_batches = 0
        self.cpu_scatter_seconds = self.cpu_scatter_join_seconds = 0.0
        if self.options.async_cpu_scatter and not getattr(runner,'_kvmem_bounded_target',False):
            raise ValueError('Background CPU scatter requires native bounded terminal-prefix control')
        if self.options.repository_gc:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Serving repository GC requires bounded execution')
            from .repository_gc import serving_gc_config
            self.gc_config = serving_gc_config(self.config, max_tokens,
                checkpoint_frames=int(bool(self.options.session_capacity)))
        self.archive = getattr(runner, '_kvmem_prepared_archive', None)
        if self.archive is None:
            self.archive = make_shared_archive(self.options, output / 'continuous_archive', record_bytes,
                max_reqs if getattr(runner, '_kvmem_bounded_target', False) else 1,
                max_query_tokens=max_tokens)
        self.downloads = PackedDownloads(max_tokens * self.row_elements * 2 + (4 * max_reqs + 1) * 8,
                                         2, runner.device)
        split_flag = os.getenv('KVMEM_SPLIT_CAPTURE_DMA',
            '1' if getattr(self.options, 'split_capture_dma', True) else '0')
        if split_flag not in ('0', '1'):
            raise ValueError('KVMEM_SPLIT_CAPTURE_DMA must be 0 or 1')
        self.split_capture_dma = split_flag == '1'
        if self.early_capture_enabled and not self.split_capture_dma:
            raise ValueError('Early capture slot release requires split capture DMA')
        self.decode_downloads = None
        self.decode_capture_limit = min(127, max(1, int(getattr(runner, 'decode_query_len', 1))))
        if self.split_capture_dma:
            # Two fixed small slots, not another full-batch pool. The original
            # bulk capacity remains sufficient for an all-prefill batch.
            small_tokens = min(max_tokens, max_reqs * self.decode_capture_limit)
            self.decode_downloads = PackedDownloads(
                small_tokens * self.row_elements * 2 + (4 * max_reqs + 1) * 8,
                2, runner.device)
        self.resident_records = max(16, (max_tokens + self.config.block_tokens - 1) // self.config.block_tokens + 1)
        self.authorities = {}
        self.queue = []
        self.accepted = {}
        self.rejected = {}
        self.last_steps = {}
        self.terminal_prefixes = {}
        self.terminal_discarded = {}
        self.completed = CompletionHistory(serving=bool(getattr(runner, '_kvmem_serving', False)))
        self.batches = 0
        self._progress_batch = -1
        self.retiring = {}
        self.retirement = None
        if self.options.async_archive_retirement:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Asynchronous retirement requires native bounded authority admission')
            from .retirement import RetirementQueue
            self.retirement = RetirementQueue(max_reqs, self.options.verify_archive_on_finish)
        self.action = None
        if self.config.action_attention:
            from .action_attention import ActionRuntime
            self.action = ActionRuntime(self)
        self.block_graph_budget = None
        if getattr(self.options, 'graph_mode', 'off') != 'off':
            from .block_graph import GraphBudget
            self.block_graph_budget = GraphBudget(self.options.graph_max_total_bytes)

    def submit(self, captured, batch, rejected):
        self._raise_publication_error()
        profile = getattr(getattr(self, 'action', None), 'profile', None)
        trace = profile.begin_batch(batch) if profile is not None else None
        try:
            with trace.stage('publication.batch_drain', kind='batch',
                             request_ids=list(batch.req_ids)) if trace else nullcontext():
                if not self.drain_requests(batch.req_ids, wait=True):
                    raise RuntimeError('Public capture preceded scheduler confirmation')
        except BaseException:
            if profile is not None:
                profile.finish_batch(status='error')
            raise
        if tuple(c.name for c in captured) != self.names:
            raise ValueError("Continuous capture missed or reordered target layers")
        if any(c.key.dtype != torch.bfloat16 for c in captured):
            raise ValueError("Continuous Qwen target capture currently qualifies BF16 activations")
        groups = self._capture_groups(batch)
        if not getattr(self, 'split_capture_dma', False) and len(self.downloads.pending) == len(self.downloads.buffers):
            # Keep the original pre-Action backpressure order in the disabled
            # control arm. Split mode must enqueue decode before a bulk wait.
            self.ensure_download_slot(batch.req_ids)
        action_inputs = self.action.capture(captured,batch) if getattr(self,'action',None) is not None else {}
        packets = {}
        for lane, indices in groups:
            pool = self._download_pool(lane)
            if len(pool.pending) == len(pool.buffers):
                self.ensure_download_slot((batch.req_ids[i] for i in indices), lane=lane)
            payload, t = self._capture_payload(captured, batch, rejected, indices)
            download = pool.download(payload)
            if download is None:
                raise RuntimeError("Capture queue failed to provide bounded staging after drain")
            packet = CapturedPacket(tuple(batch.req_ids[i] for i in indices), t, download, confirmed=False,
                request_tokens=tuple(int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i])
                                     for i in indices), lane=lane)
            packet.action_inputs = {rid: action_inputs[rid] for rid in packet.request_ids if rid in action_inputs}
            with self._publication_lock():
                self.queue.append(packet)
                if self.early_capture_packet(packet):
                    self._early_source_submitted(packet)
            packets.update((rid, packet) for rid in packet.request_ids)
            self._publication_metric('submitted_bytes', payload.numel(), lane=lane)
            self._publication_metric('submitted_packets', 1, lane=lane)
        with self._publication_lock():
            self.batches += 1
        n = batch.num_reqs
        relevance_rows = []
        if self.options.query_history and batch.has_prefill:
            sessions = getattr(self.runner,'_kvmem_sessions',None)
            for i, request_id in enumerate(batch.req_ids):
                binding = sessions.bindings.get(request_id) if sessions is not None else None
                if binding is None or not batch.is_prefilling_np[i]:
                    continue
                start = int(batch.num_computed_tokens_np[i])
                end = start + int(batch.num_scheduled_tokens[i])
                q = binding.query_span
                if end >= len(binding.tokens) or (q is not None and start <= q[0] < q[1] <= end):
                    relevance_rows.append((i,request_id))
        if relevance_rows:
            # Preserve Q, but never drain the just-sampled batch: neighbouring
            # MTP rows may still be trimmed by the next scheduler terminal notice.
            for i, request_id in relevance_rows:
                binding = sessions.bindings[request_id]
                queries = None
                history = getattr(self.authorities.get(request_id),'query_history',None)
                if binding.query_span is not None and (history is None or
                        history.request_id != request_id or history.observation is None):
                    qb, qe = binding.query_span
                    start = int(batch.num_computed_tokens_np[i])
                    end = start + int(batch.num_scheduled_tokens[i])
                    if start <= qb < qe <= end:
                        at = int(batch.query_start_loc_np[i])
                        queries = torch.stack([c.query[at+qb-start:at+qe-start]
                            for c in captured[:len(self.runner._kvmem_layers)]]).clone()
                packets[request_id].query_inputs[request_id] = (binding,queries)

    def _capture_groups(self, batch):
        all_indices = tuple(range(batch.num_reqs))
        if not getattr(self, 'split_capture_dma', False):
            return [('bulk', all_indices)]
        small = tuple(i for i in all_indices if not batch.is_prefilling_np[i] and
            0 < int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i]) <= self.decode_capture_limit)
        large = tuple(i for i in all_indices if i not in small)
        # Enqueue decode's compact payload and ready event before doing any of
        # the bulk quantization/packing. Independent streams avoid a software
        # FIFO dependency; the physical copy engine/bandwidth is still shared.
        return ([('decode', small)] if small else []) + ([('bulk', large)] if large else [])

    def _capture_payload(self, captured, batch, rejected, indices):
        n = batch.num_reqs
        starts = batch.query_start_loc[:n+1].to(torch.int64)
        positions = captured[0].positions
        if positions.ndim == 2:
            positions = positions[0]
        if len(indices) == n:
            t = batch.num_tokens
            header = torch.cat((starts, batch.positions[starts[:-1]], positions[starts[:-1]],
                                rejected[:n].to(torch.int64)))
            keys = [c.key[:t] for c in captured]
            vals = [c.value[:t] for c in captured]
        else:
            # Stack existing device scalars: constructing a CUDA tensor from a
            # Python list here would add a blocking H2D copy after the model.
            original = torch.stack([starts[i] for i in indices])
            lengths = torch.stack([starts[i+1] for i in indices]) - original
            compact = torch.cat((starts.new_zeros(1), lengths.cumsum(0)))
            # CPU counts are upper bounds under adaptive MTP. Allocate that
            # bounded size, but gather/describe actual rows using GPU starts.
            # Padding after compact[-1] is never accepted or published.
            t = sum(int(batch.query_start_loc_np[i+1]-batch.query_start_loc_np[i]) for i in indices)
            offsets = torch.arange(t, device=starts.device, dtype=torch.int64)
            owner = torch.searchsorted(compact[1:], offsets, right=True).clamp_max(len(indices)-1)
            rows = (original[owner] + offsets - compact[owner]).clamp(0, batch.num_tokens-1)
            header = torch.cat((compact, batch.positions[original], positions[original],
                                torch.stack([rejected[i] for i in indices]).to(torch.int64)))
            keys = [c.key.index_select(0, rows) for c in captured]
            vals = [c.value.index_select(0, rows) for c in captured]
        raw = torch.stack(keys).float().clamp(-448, 448).to(self.raw_dtype)
        values = (torch.stack(vals).float() / self.scales[:, None, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return torch.cat((header.view(torch.uint8), raw.view(torch.uint8).flatten(),
                          values.view(torch.uint8).flatten())), t

    def _download_pool(self, lane='bulk'):
        return self.decode_downloads if lane == 'decode' else self.downloads

    def _pending_download_count(self):
        return len(self.downloads.pending) + (len(self.decode_downloads.pending)
            if getattr(self, 'decode_downloads', None) is not None else 0)

    def _publication_metric(self, name, value, *, lane=None):
        thread = 'background' if getattr(self, '_cpu_scatter_thread', None) == get_ident() else 'main'
        key = thread if lane is None else thread + '/' + lane
        with self._publication_lock():
            counters = self.__dict__.setdefault('_publication_metrics', {}).setdefault(key, {})
            counters[name] = counters.get(name, 0) + value

    def _publication_metrics_snapshot(self):
        # Caller holds the publication lock. Durations overlap GPU execution;
        # these host counters are never an additive GPU-idle measurement.
        return {key: dict(value) for key, value in getattr(self, '_publication_metrics', {}).items()}

    def _accept_query_input(self, packet, request_id):
        value = packet.query_inputs.pop(request_id,None)
        if value is None:
            return
        binding, queries = value
        authority = self.authorities[request_id]
        history = authority.query_history
        history.begin(request_id,binding.tokens)
        if queries is not None and history.observation is None:
            history.query = queries
        if authority.history_end >= len(binding.tokens) and history.policy.open:
            if history.observation is None and history.query is not None:
                from .request_plan import RequestPlan
                fit = RequestPlan(None,block_tokens=self.config.block_tokens,
                    gen_budget=self.config.gen_budget,mandatory=binding.mandatory_spans).mandatory_fit(
                        authority.history_end,self.config)
                required = set(fit['hard_blocks'])
                records = [r for r in authority.records.values()
                           if r.history_begin//self.config.block_tokens in required]
                if {r.history_begin//self.config.block_tokens for r in records} != required:
                    raise ValueError('Required Query-history evidence is absent from the durable source')
                authority.select_for_queries(history.query, [r.handle for r in records], observe_only=True)
            observed = history.policy.has_observation
            history.commit()
            with (self.output / 'query_history.jsonl').open('a') as stream:
                stream.write(json.dumps({'request_id': request_id,
                    'round': history.policy.round_count, 'input_end': history.policy.input_end,
                    'observed': observed, 'scored_pages': len(history.policy.scores),
                    'score_sha256':hashlib.sha256(json.dumps(sorted(history.policy.scores.items())).encode()).hexdigest()}) + '\n')

    def _publication_lock(self):
        # __new__ based CPU fixtures use the same locking protocol.
        self.__dict__.setdefault('_published_capacity', {})
        return self.__dict__.setdefault('_publication_condition', Condition(RLock()))

    def _raise_publication_error(self):
        error = getattr(self, '_publication_error', None)
        if error is not None:
            raise error

    def _busy_requests(self):
        return {packet.request_ids[i] for packet in self.queue for i in
                (packet.consuming | (set(packet.source_copied)-packet.published
                 if self.early_capture_packet(packet) else set()))}

    def _install_terminals(self, request_ids, terminal_prefixes):
        if terminal_prefixes is None:
            return
        for request_id in request_ids:
            known = request_id in self.authorities or any(
                rid == request_id and i not in packet.published
                for packet in self.queue for i, rid in enumerate(packet.request_ids))
            prefix = terminal_prefixes.get(request_id)
            if known and prefix is None:
                raise RuntimeError('Missing scheduler terminal prefix for captured request')
            if prefix is None or not known:
                continue
            authority = self.authorities.get(request_id)
            # history_end is published by native commit, never by reservation.
            if authority is not None and authority.history_end > prefix.history_end:
                raise RuntimeError('Terminal prefix arrived after excess history was published')
            self.terminal_prefixes[request_id] = prefix

    def begin_step(self, request_ids=(), terminal_prefixes=None):
        """Apply terminal clipping before authorizing earlier sample packets.

        Does not finish sessions or wait for CPU publication. The runner calls
        this once per scheduler decision, then receives session metadata and
        calls finish for the ended requests. Newly submitted packets remain
        unconfirmed until the following decision.
        """
        with self._publication_lock():
            self._raise_publication_error()
            self._install_terminals(request_ids, terminal_prefixes)
            for packet in self.queue:
                packet.confirmed = True
            self._publication_condition.notify_all()

    def defer_terminals(self, output):
        """Keep ended owners' native indices until their CPU prefix is ready.

        begin_step must install terminal clipping first. Finished IDs are sent
        once by the scheduler, so save and replay them on a later worker step.
        This composes after the refresh runtime's own terminal deferral.
        """
        from .connector import ControlMetadata
        control = output.kv_connector_metadata
        if not isinstance(control, ControlMetadata):
            return output
        deferred = self.__dict__.setdefault('_deferred_terminals', {})
        finished, prefixes = set(output.finished_req_ids), dict(control.terminal_prefixes)
        with self._publication_lock():
            # Refresh may replay a previously hidden finished ID this step.
            self._install_terminals(finished, prefixes)
        # A small completed decode needs no worker handoff, but must never wait
        # for DMA or for an unrelated in-flight owner on this control path.
        small = (finished | set(deferred)) - self.publication_blocked_requests()
        self.drain_requests(small, wait=False)
        with self._publication_lock():
            for request_id in list(finished):
                if not self.publication_ready((request_id,)):
                    deferred[request_id] = prefixes.pop(request_id, None)
                    finished.remove(request_id)
            for request_id, prefix in list(deferred.items()):
                if self.publication_ready((request_id,)):
                    finished.add(request_id)
                    if prefix is not None:
                        prefixes[request_id] = prefix
                    del deferred[request_id]
        return replace(output, finished_req_ids=finished,
                       kv_connector_metadata=replace(control, terminal_prefixes=prefixes))

    def publication_ready(self, request_ids):
        """Whether every queued publication for these requests is complete."""
        wanted = set(request_ids)
        with self._publication_lock():
            self._raise_publication_error()
            return not any(any(request_id in wanted and i not in packet.published
                for i, request_id in enumerate(packet.request_ids)) for packet in self.queue)

    def publication_blocked_requests(self):
        """Large pending owners for scheduler admission, including newest DMA.

        A scheduler must still send confirmation/poll iterations when all such
        owners are blocked. Tiny decode prefixes can use request-local fences.
        """
        with self._publication_lock():
            self._raise_publication_error()
            threshold = self.options.cpu_scatter_min_tokens
            pending = {}
            for packet in self.queue:
                sizes = packet.request_tokens or (packet.tokens,) * len(packet.request_ids)
                for i, request_id in enumerate(packet.request_ids):
                    if i not in packet.published:
                        pending[request_id] = pending.get(request_id, 0) + sizes[i]
            blocked = {request_id for request_id, count in pending.items() if count >= threshold}
            if self.early_capture_enabled:
                blocked.update(packet.request_ids[i] for packet in self.queue
                               if self.early_capture_packet(packet)
                               for i in set(packet.source_copied)-packet.published)
            return blocked

    def ensure_download_slot(self, request_ids, *, lane='bulk'):
        """Wait for one reusable bounded slot, never for the entire queue."""
        if self.early_capture_enabled and lane == 'bulk':
            return self._ensure_early_download_slot(lane)
        self.drain_requests(request_ids, wait=True)
        pool = self._download_pool(lane)
        condition = self._publication_lock()
        started = time.perf_counter()
        try:
            while True:
                with condition:
                    self._raise_publication_error()
                    if len(pool.pending) < len(pool.buffers):
                        return
                    owned = [p for p in self.queue if p.lane == lane]
                    available = [p for p in owned if p.confirmed and not p.consuming]
                    if available:
                        packet = min(available, key=lambda p: sum(
                            (p.request_tokens or (p.tokens,) * len(p.request_ids))[i]
                            for i in range(len(p.request_ids)) if i not in p.published))
                        needed = {request_id for i, request_id in enumerate(packet.request_ids)
                                  if i not in packet.published}
                    elif any(p.consuming for p in owned):
                        waited = time.perf_counter()
                        condition.wait()
                        self._publication_metric('slot_condition_wait_seconds', time.perf_counter()-waited, lane=lane)
                        continue
                    else:
                        raise RuntimeError('Bounded capture slots contain only unconfirmed samples')
                reclaimed = time.perf_counter()
                published = self.drain_requests(needed, wait=True)
                self._publication_metric('slot_reclaim_seconds', time.perf_counter()-reclaimed, lane=lane)
                if not published:
                    raise RuntimeError('Cannot reclaim an unconfirmed capture slot')
        finally:
            self._publication_metric('slot_wait_seconds', time.perf_counter()-started, lane=lane)
            self._publication_metric('slot_wait_calls', 1, lane=lane)

    def advance_publication(self, scheduled_tokens, refresh=None):
        """Give confirmed packets to the scatter worker before public compute.

        Large owners advance independently in the background. Tiny owners can
        publish through their foreground request fence. A no-forward iteration
        must also advance those tiny owners, since it has no later submit.
        Maintenance polls only jobs whose own publication fence is complete.
        """
        if self.options.async_cpu_scatter:
            self.start_cpu_scatter()
            # An empty scheduler step must advance tiny confirmed packets too;
            # large packets remain on their owning worker. Lifecycle readers
            # use request fences, not a no-forward packet-wide barrier.
            if not scheduled_tokens:
                self.drain_requests(self._small_pending_requests(), wait=False)
        else:
            self.drain(wait=False)
        if refresh is not None and refresh.jobs:
            self.drain_requests(set(refresh.jobs) & self._small_pending_requests(), wait=False)
            # pump checks each owner's publication fence before it reads any
            # mutable authority; an unrelated ready job can still advance.
            refresh.pump(tuple(scheduled_tokens))

    def _small_pending_requests(self):
        with self._publication_lock():
            return {request_id for packet in self.queue if packet.confirmed
                    for request_id in packet.request_ids} - self.publication_blocked_requests()

    def start_cpu_scatter(self):
        """Called only after applying the next scheduler's EOS/cancel decision.

        The worker claims one confirmed request at a time. Foreground callers
        can publish unrelated owners, including those sharing its DMA slot.
        Newly sampled packets remain unconfirmed until the next begin_step.
        """
        self._raise_publication_error()
        future = getattr(self, '_cpu_scatter_future', None)
        if future is not None:
            if not future.done():
                return False
            self.join_cpu_scatter()
        with self._publication_lock():
            queued = [packet for packet in self.queue if getattr(packet, 'confirmed', True)]
        if not self.options.async_cpu_scatter or not queued:
            return False
        feedback = any(self.early_capture_packet(packet) and
                       (packet.source_copied or packet.query_inputs) for packet in queued)
        if sum(packet.tokens for packet in queued)<self.options.cpu_scatter_min_tokens and not feedback:
            # Tiny decode packets cost less than a worker handoff/GIL contention.
            # submit drains them after current GPU work has been enqueued; keep
            # the same bounded DMA slots and next-scheduler terminal barrier.
            return False
        if getattr(self,'_cpu_scatter_executor',None) is None:
            self._cpu_scatter_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='kvmem-scatter')
        self.cpu_scatter_batches=getattr(self,'cpu_scatter_batches',0)+len(queued)
        def work():
            started=time.perf_counter()
            self._cpu_scatter_thread=get_ident()
            try:
                return self.drain(wait=True)
            except BaseException as error:
                with self._publication_lock():
                    self._publication_error = error
                    self._publication_condition.notify_all()
                raise
            finally:
                self._cpu_scatter_thread=None
                self.cpu_scatter_seconds=getattr(self,'cpu_scatter_seconds',0.0)+time.perf_counter()-started
        self._cpu_scatter_future=self._cpu_scatter_executor.submit(work)
        return True

    def join_cpu_scatter(self):
        future=getattr(self,'_cpu_scatter_future',None)
        if future is None:
            return
        started=time.perf_counter()
        try:
            future.result()
            self._cpu_scatter_future=None
        finally:
            self.cpu_scatter_join_seconds=getattr(self,'cpu_scatter_join_seconds',0.0)+time.perf_counter()-started

    def ensure_authority(self, request_id):
        with self._publication_lock():
            return self._ensure_authority_locked(request_id)

    def _ensure_authority_locked(self, request_id):
        authority = self.authorities.get(request_id)
        if authority is None:
            sessions = getattr(self.runner,'_kvmem_sessions',None)
            graph_enabled = (getattr(self.options, 'graph_mode', 'off') != 'off' and
                             request_id not in getattr(sessions, 'release_requests', ()))
            binding = sessions.bindings.get(request_id) if sessions is not None else None
            if graph_enabled and binding is None:
                # Never allocate authority state for a graph with temporary
                # request ownership; its logical identities would be unstable.
                raise ValueError('Block graph requires an owned session binding')
            authority = RequestAuthority(request_id, self.config, self.archive,
                len(self.layers), self.heads, self.dim, raw_dtype=self.raw_dtype,
                resident_records=self.resident_records,
                gc_config=self.gc_config, gc_progress=self.gc_progress,
                scoring_layers=len(self.runner._kvmem_layers),mean_k_backend=self.options.mean_k_backend)
            if getattr(self,'action',None) is not None:
                from .action_attention import ActionHistory
                authority.action_history = ActionHistory(self.options,self.action.ids)
            if graph_enabled:
                from .block_graph import BlockGraph
                try:
                    authority.block_graph = BlockGraph((binding.namespace,binding.generation),
                        self.config.block_tokens,topk=self.options.graph_attention_topk,
                        min_mass=self.options.graph_attention_min_mass,
                        max_bytes_per_session=self.options.graph_max_bytes_per_session,
                        budget=self.block_graph_budget)
                    authority.graph_options = self.options
                    authority.graph_report_path = self.output/'block_graph.jsonl'
                    authority.block_graph.update_message_spans(
                        getattr(binding,'message_spans',()),validated_prefix=0)
                except BaseException:
                    graph = getattr(authority,'block_graph',None)
                    if graph is not None:
                        graph.close()
                    authority.close()
                    raise
            if self.options.query_history and sessions is not None and request_id in sessions.bindings:
                from .query_history import QueryRelevance
                authority.query_history = QueryRelevance(self.options.query_history_decay)
            self.authorities[request_id] = authority
            self.accepted[request_id] = self.rejected[request_id] = 0
            self.last_steps[request_id] = []
            self._snapshot_capacity(request_id, authority)
        return authority

    def _snapshot_capacity(self, request_id, authority):
        from .archive_admission import ArchiveCapacity
        self._published_capacity[request_id] = ArchiveCapacity(
            authority.history_end, authority.memory.repository.ready_slots)

    def drain(self, wait=False):
        """Explicit all-request barrier; public paths use drain_requests."""
        return self.drain_requests(None, wait=wait)

    def _claim_publications(self, wanted, wait):
        """Claim independent owners; never hold the condition during CPU copy."""
        blocked = set()
        pending = False
        unconfirmed = False
        background = getattr(self, '_cpu_scatter_thread', None) == get_ident()
        for packet in self.queue:
            indices = []
            for i, request_id in enumerate(packet.request_ids):
                if i in packet.published or (wanted is not None and request_id not in wanted):
                    continue
                pending = True
                earlier = request_id in blocked
                blocked.add(request_id)
                if not packet.confirmed:
                    unconfirmed = True
                    continue
                if earlier or i in packet.consuming:
                    continue
                if not wait and not packet.source_released and not packet.download.query():
                    continue
                indices.append(i)
                # Large background requests have independent publication
                # fences. A short foreground decode may still batch its peers.
                if background or not self.options.batched_cpu_append:
                    break
            if indices:
                packet.consuming.update(indices)
                return (packet, indices), pending, unconfirmed
        return None, pending, unconfirmed

    def drain_requests(self, request_ids, wait=False):
        """Publish only the selected requests, preserving each owner's FIFO.

        Another thread may publish an unrelated owner from the same DMA packet.
        Bulk source release can precede scores only after every owner is copied;
        owner publication still waits for its scores. A false result means
        an unready DMA/capacity/in-flight owner, or an unconfirmed sample; even
        wait=True never bypasses the scheduler-confirmation boundary.
        """
        wanted = None if request_ids is None else set(request_ids)
        condition = self._publication_lock()
        while True:
            with condition:
                self._raise_publication_error()
                claim, pending, unconfirmed = self._claim_publications(wanted, wait)
                if claim is None:
                    if not pending:
                        return True
                    if not wait or unconfirmed:
                        return False
                    condition.wait()
                    continue
            packet, indices = claim
            started = time.perf_counter()
            try:
                consumed = self._publish_claim(packet, indices, wait)
            except BaseException as error:
                with condition:
                    self._publication_error = error
                    # Failed owners keep their DMA source alive. Every waiter
                    # receives the same fatal publication failure.
                    condition.notify_all()
                raise
            finally:
                self._publication_metric('claim_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('claim_calls', 1, lane=packet.lane)
            with condition:
                packet.published.update(indices[:consumed])
                packet.consuming.difference_update(indices)
                while packet.next_request in packet.published:
                    packet.next_request += 1
                if len(packet.published) == len(packet.request_ids):
                    assert not packet.consuming
                    if self.early_capture_packet(packet):
                        self._finish_early_packet(packet)
                    else:
                        assert self._download_pool(packet.lane).release(packet.download)
                    self.queue[:] = [queued for queued in self.queue if queued is not packet]
                condition.notify_all()
            if not consumed and not wait:
                return False
            # Only the engine thread writes diagnostics. It snapshots published
            # counters and never traverses an in-flight authority's records.
            if (getattr(self, '_cpu_scatter_thread', None) != get_ident() and
                    self.batches // 8 != self._progress_batch):
                self._progress_batch = self.batches // 8
                self._write_progress()

    @staticmethod
    def _graph_reservation(prepared):
        return getattr(prepared, 'graph_update', prepared)

    def _prepare_graph_acceptance(self, packet, index, accepted, history_begin):
        request_id = packet.request_ids[index]
        graph = getattr(self.authorities[request_id], 'block_graph', None)
        if graph is None:
            return None
        action = getattr(self, 'action', None)
        if action is not None:
            return action.prepare_accept(packet, request_id, accepted,
                                         graph=graph, history_begin=history_begin)
        begin = min(history_begin, graph.accepted_through) if not accepted else history_begin
        return graph.prepare(begin, begin + accepted)

    def _abort_graph_updates(self, packet, indices):
        updates = getattr(packet, 'graph_updates', None)
        if updates is None:
            return
        for index in indices:
            prepared = updates.pop(index, None)
            if prepared is not None:
                prepared.abort()
        if not updates:
            packet.graph_updates = None

    def _publish_claim(self, packet, indices, wait):
        try:
            return self._publish_reserved_claim(packet, indices, wait)
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _publish_reserved_claim(self, packet, indices, wait):
        metadata = self._copy_claim_metadata(packet, indices, wait)
        for i, accepted, discarded, scheduled, history_begin, rejected in metadata:
            request_id = packet.request_ids[i]
            if self.early_capture_packet(packet):
                self._early_event('semantic_owner_begin', packet, owner=request_id)
            action = getattr(self, 'action', None)
            updates = getattr(packet, 'graph_updates', None)
            prepared = updates.pop(i, None) if updates is not None else None
            if updates is not None and not updates:
                packet.graph_updates = None
            if prepared is None:
                # Compatibility for diagnostic/test metadata sources which
                # bypass raw archive copying. Production reserves before append.
                prepared = self._prepare_graph_acceptance(packet, i, accepted, history_begin)
            try:
                if prepared is not None:
                    # Early source copy can run on a different thread. Resume
                    # and validate ownership before Query/parser/legacy mutation.
                    self._graph_reservation(prepared).resume()
                started = time.perf_counter()
                self._accept_query_input(packet, request_id)
                self._publication_metric('owner_query_seconds', time.perf_counter()-started, lane=packet.lane)
                if prepared is not None:
                    started = time.perf_counter()
                    prepared.commit()
                    self._publication_metric('owner_action_seconds', time.perf_counter()-started, lane=packet.lane)
                elif action is not None:
                    started = time.perf_counter()
                    action.accept(packet, request_id, accepted)
                    self._publication_metric('owner_action_seconds', time.perf_counter()-started, lane=packet.lane)
            except BaseException:
                if prepared is not None:
                    prepared.abort()
                raise
            self._publication_metric('published_owners', 1, lane=packet.lane)
            self._publication_metric('published_tokens', accepted, lane=packet.lane)
            with self._publication_lock():
                self.accepted[request_id] += accepted
                self.rejected[request_id] += rejected
                self.terminal_discarded[request_id] = self.terminal_discarded.get(request_id, 0) + discarded
                steps = self.last_steps[request_id]
                steps.append(dict(history_begin=history_begin, scheduled=scheduled, accepted=accepted,
                                  rejected=rejected, terminal_discarded=discarded))
                del steps[:max(0, len(steps)-self.runner.vllm_config.max_concurrent_batches-1)]
                self._snapshot_capacity(request_id, self.authorities[request_id])
                if self.early_capture_packet(packet):
                    self._early_event('semantic_owner_committed', packet, owner=request_id)
        if metadata:
            with self._publication_lock():
                getattr(self, '_archive_request_deadlines', {}).pop(
                    frozenset(packet.request_ids[i] for i in indices), None)
        return len(metadata)

    def _copy_claim_metadata(self, packet, indices, wait):
        try:
            return self._copy_reserved_metadata(packet, indices, wait)
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _copy_reserved_metadata(self, packet, indices, wait):
        if not self.early_capture_packet(packet):
            return self._copy_source_owners(packet, indices, wait)
        missing = [i for i in indices if i not in packet.source_copied]
        if missing:
            if packet.source_released:
                raise RuntimeError('Released DMA source still has uncopied owners')
            started = time.perf_counter()
            metadata = self._copy_source_owners(packet, missing, wait)
            # _copy_source_owners has returned: no local tensor view of staging
            # survives in these integer tuples. Async seal owns record copies.
            with self._publication_lock():
                for row in metadata:
                    i = row[0]
                    packet.source_feedback_bytes[i] = self._feedback_bytes(packet, i)
                    packet.source_copied[i] = row
                self._publication_metric('source_copy_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('source_copied_owners', len(metadata), lane=packet.lane)
                self._early_event('source_copied', packet,
                                  copied_indices=[row[0] for row in metadata],
                                  feedback_bytes=sum(packet.source_feedback_bytes.values()),
                                  owner_storage_bytes={packet.request_ids[row[0]]:
                                      self._feedback_storage_bytes(packet, row[0]) for row in metadata})
                self._try_release_source(packet)
        result = []
        for i in indices:
            if i not in packet.source_copied:
                break
            result.append(packet.source_copied[i])
        return result

    def _copy_source_owners(self, packet, indices, wait):
        if not packet.download.query():
            started = time.perf_counter()
            try:
                packet.download.event.synchronize()
            finally:
                self._publication_metric('dma_wait_seconds', time.perf_counter()-started, lane=packet.lane)
                self._publication_metric('dma_wait_calls', 1, lane=packet.lane)
        n, t = len(packet.request_ids), packet.tokens
        data = packet.download.payload
        header_bytes = (4 * n + 1) * 8
        header = data[:header_bytes].view(torch.int64).tolist()
        starts, histories = header[:n+1], header[n+1:2*n+1]
        local, rejected = header[2*n+1:3*n+1], header[3*n+1:]
        shape = (len(self.layers), t, self.heads, self.dim)
        raw_bytes = self.row_elements * t * self.raw_dtype.itemsize
        raw = data[header_bytes:header_bytes+raw_bytes].view(self.raw_dtype).reshape(shape)
        values = data[header_bytes+raw_bytes:].reshape(shape)
        entries, metadata = [], []
        for i in indices:
            request_id = packet.request_ids[i]
            authority = self.ensure_authority(request_id)
            begin, end = starts[i:i+2]
            accepted = end - begin - rejected[i]
            if not 0 <= accepted <= end - begin:
                raise ValueError('Native sampling rejection count exceeds captured rows')
            terminal = self.terminal_prefixes.get(request_id)
            retained = accepted if terminal is None else max(0, min(accepted, terminal.history_end-histories[i]))
            entries.append((authority, raw[:, begin:end], values[:, begin:end],
                dict(history_begin=histories[i], local_rope_begin=local[i], accepted=retained)))
            metadata.append((i, retained, accepted-retained, end-begin, histories[i], rejected[i]))
        try:
            # Preflight every owner before native append can reserve/commit any
            # raw KV. Failed graph capacity leaves all history and parser state
            # unchanged. This is deliberately before append_authority_batch.
            for i, retained, _discarded, _scheduled, history_begin, _rejected in metadata:
                if getattr(packet, 'graph_updates', None) is not None and i in packet.graph_updates:
                    raise RuntimeError('Capture owner already has a graph reservation')
                prepared = self._prepare_graph_acceptance(packet, i, retained, history_begin)
                if prepared is not None:
                    self._graph_reservation(prepared).detach()
                    if getattr(packet, 'graph_updates', None) is None:
                        packet.graph_updates = {}
                    packet.graph_updates[i] = prepared
            while True:
                started = time.perf_counter()
                try:
                    if self.options.batched_cpu_append:
                        from .authority import append_authority_batch
                        consumed = append_authority_batch(entries)
                        if consumed:
                            self.batched_append_calls = getattr(self, 'batched_append_calls', 0) + 1
                            self.batched_append_requests = getattr(self, 'batched_append_requests', 0) + consumed
                    else:
                        authority, request_raw, request_values, kwargs = entries[0]
                        consumed = int(not kwargs['accepted'] or authority.append(request_raw, request_values, **kwargs))
                finally:
                    self._publication_metric('append_seconds', time.perf_counter()-started, lane=packet.lane)
                if consumed or not wait:
                    break
                self._wait_archive_progress({packet.request_ids[i] for i in indices})
            self._abort_graph_updates(packet, indices[consumed:])
            return metadata[:consumed]
        except BaseException:
            self._abort_graph_updates(packet, indices)
            raise

    def _write_progress(self):
        lock = self.__dict__.setdefault('_progress_lock', RLock())
        with lock, self._publication_lock():
            self._write_progress_locked()

    def _write_progress_locked(self):
        busy = self._busy_requests()
        progress = {"batches_submitted": self.batches,
            'batched_append_calls': getattr(self, 'batched_append_calls', 0),
            'batched_append_requests': getattr(self, 'batched_append_requests', 0),
            'full_graph_decode_calls': getattr(self.runner, '_kvmem_graph_decode_calls', 0),
            "accepted_history": {req: (self._published_capacity[req].history_end
                if req in busy and req in self._published_capacity else owner.history_end)
                for req, owner in self.authorities.items()},
            "archive_used_slots": self.archive.tier.used_slots,
            "pending_downloads": self._pending_download_count(),
            'split_capture_dma': getattr(self, 'split_capture_dma', False),
            'publication_metrics': self._publication_metrics_snapshot(),
            'publication_metrics_scope': 'Cumulative host durations by thread/lane; overlapping, not GPU idle. claim includes append and owner scores; slot wait may include claims.',
            'capture_staging_capacity_bytes': sum(getattr(pool, 'capacity_bytes', 0)
                for pool in (self.downloads, getattr(self, 'decode_downloads', None)) if pool is not None),
            'cpu_scatter_batches':getattr(self,'cpu_scatter_batches',0),
            'cpu_scatter_seconds':getattr(self,'cpu_scatter_seconds',0.0),
            'cpu_scatter_join_seconds':getattr(self,'cpu_scatter_join_seconds',0.0),
            **completion_counts(self.completed),
            "retiring_requests": len(getattr(self, 'retiring', {}))}
        if self.early_capture_enabled:
            self._sample_early_publication(progress)
        temporary = self.output / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(self.output / "progress.json")

    def archive_capacity(self):
        """Nonblocking worker feedback, including DMA not yet incorporated.

        The scheduler debits any accepted/computing suffix beyond history_end;
        snapshotting an unfinished download cannot invent free archive space.
        """
        from .archive_admission import ArchiveCapacity
        # Do not drain the latest sample here: native EOS clipping is decided
        # by the scheduler after this output. execute_model applies that boundary
        # before draining on the following iteration.
        with self._publication_lock():
            self._raise_publication_error()
            busy = self._busy_requests()
            result = {}
            for request_id, authority in self.authorities.items():
                if request_id not in busy and not getattr(authority, '_refresh_background', False):
                    authority.poll()
                    self._snapshot_capacity(request_id, authority)
                # An in-flight append may have reserved rows but cannot make
                # its history/capacity visible before Action and Query commit.
                # The scheduler debits its suffix against this older prefix.
                value = self._published_capacity.get(request_id)
                result[request_id] = value if value is not None else ArchiveCapacity(0, 0)
            return result

    def _wait_archive_progress(self, request_ids=None):
        # Shared queue pressure may come from another request, even when this
        # authority has not admitted its first write yet.
        with self._publication_lock():
            owners = [a for request_id, a in self.authorities.items()
                if (request_ids is None or request_id in request_ids)
                and not getattr(a, '_refresh_background', False)]
        progressed = sum(authority.poll() for authority in owners)
        jobs = [job for authority in owners for job, _ in authority.writes]
        if not jobs:
            if progressed:
                if request_ids is None:
                    self._archive_wait_deadline = None
                else:
                    getattr(self, '_archive_request_deadlines', {}).pop(frozenset(request_ids), None)
                return
            if (request_ids is not None or getattr(self, 'retiring', {}) or
                    getattr(getattr(getattr(self, 'runner', None), '_kvmem_sessions', None), 'retiring', {})):
                # Detached finalizers share the same count/byte queue. Their
                # I/O is absent from active authority.writes; do not report a
                # false repository-capacity failure while they still own it.
                if self.archive.io_usage()['archive_io_pending_bytes']:
                    if not self.archive.wait_for_idle(30):
                        raise TimeoutError('Detached archive I/O did not release capacity')
                    return
            if self.options.repository_gc:
                # The CPU policy worker may be reclaiming durable generations
                # even with no outstanding archive write. Never run policy on
                # this engine thread or silently discard captured accepted rows.
                # This synchronous fallback is measured; scheduler-side capacity
                # feedback is still required to remove such serving stalls.
                started = time.monotonic()
                if request_ids is None:
                    self._archive_wait_deadline = self._archive_wait_deadline or started + 30
                    deadline = self._archive_wait_deadline
                else:
                    with self._publication_lock():
                        deadlines = self.__dict__.setdefault('_archive_request_deadlines', {})
                        deadline = deadlines.setdefault(frozenset(request_ids), started+30)
                if started >= deadline:
                    raise TimeoutError('Repository GC made no append progress for 30 seconds')
                self.gc_progress.wait(min(.05, deadline - started))
                self.gc_progress.clear()
                self.gc_wait_seconds += time.monotonic() - started
                self.gc_wait_count += 1
                return
            raise BufferError("Authority capacity exhausted; scheduler admission/GC must yield")
        for job in jobs:
            job.result()

    def finish(self, request_ids, terminal_prefixes=None):
        """Drain publication, optionally checksum records, then reclaim the request."""
        if not request_ids:
            return
        with self._publication_lock():
            self._install_terminals(request_ids, terminal_prefixes)
        if not self.drain_requests(request_ids, wait=True):
            raise RuntimeError('Terminal capture was not confirmed by begin_step')
        for request_id in request_ids:
            authority = self.authorities.get(request_id)
            if authority is None:
                continue
            terminal = getattr(self, 'terminal_prefixes', {}).pop(request_id, None)
            if terminal is not None and authority.history_end != terminal.history_end:
                raise RuntimeError('Final capture does not match scheduler terminal prefix')
            terminal_report = {
                'terminal_prefix': asdict(terminal) if terminal is not None else None,
                'terminal_discarded_tokens': getattr(self, 'terminal_discarded', {}).pop(request_id, 0)}
            if getattr(self,'action',None) is not None:
                terminal_report['action_attention'] = self.action.finish(request_id,authority)
            sessions = getattr(self.runner, '_kvmem_sessions', None)
            if sessions is not None and sessions.detach(request_id, authority, {
                    'request_id': request_id, 'accepted_tokens': self.accepted[request_id],
                    'rejected_tokens': self.rejected[request_id],
                    'last_steps': self.last_steps[request_id], **terminal_report}):
                with self._publication_lock():
                    del self.authorities[request_id], self.accepted[request_id], self.rejected[request_id]
                    del self.last_steps[request_id]
                    self._published_capacity.pop(request_id, None)
                    getattr(self, '_deferred_terminals', {}).pop(request_id, None)
                continue
            from .gpu_raw_cache import retire_raw_cache
            retire_raw_cache(self.runner, authority)
            if self.options.async_archive_retirement:
                # All GPU capture packets have drained above. Detach ownership
                # before another native batch; this worker touches CPU data only.
                enqueued_at = time.perf_counter()
                self.retirement.submit(request_id, authority)
                self.retiring[request_id] = {"request_id": request_id,
                    "retirement_enqueued_at": enqueued_at,
                    "accepted_tokens": self.accepted.pop(request_id),
                    "rejected_tokens": self.rejected.pop(request_id),
                    "last_steps": self.last_steps.pop(request_id), **terminal_report}
                with self._publication_lock():
                    del self.authorities[request_id]
                    self._published_capacity.pop(request_id, None)
                    getattr(self, '_deferred_terminals', {}).pop(request_id, None)
                continue
            authority.stop_gc()
            if not authority.flush():
                self._wait_archive_progress({request_id})
                if not authority.flush():
                    raise BufferError("Archive cannot finalize complete authority records")
            for job, _ in authority.writes:
                job.result()
            authority.poll()
            published = sum(record.archived for record in authority.records.values())
            if published + authority.gc_reclaimed_records != authority.history_end // self.config.block_tokens:
                raise RuntimeError("Finalized authority has unpublished complete archive records")
            verified = 0
            for record in authority.records.values():
                if record.archived and self.options.verify_archive_on_finish:
                    raw, values = authority.read_record(record.handle)
                    digest = hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()
                                           + values.numpy().tobytes()).hexdigest()
                    if digest != record.digest:
                        raise RuntimeError("Archived raw K/V checksum differs from accepted capture")
                    verified += 1
            self.completed.append({"request_id": request_id, "history_end": authority.history_end,
                "accepted_tokens": self.accepted.pop(request_id),
                "rejected_tokens": self.rejected.pop(request_id),
                "last_steps": self.last_steps.pop(request_id),
                "records": len(authority.records), "sealed_records_archived": published,
                "sealed_records_verified": verified,
                "gc_reclaimed_records": authority.gc_reclaimed_records, **terminal_report})
            if not authority.close():
                raise RuntimeError("Finalized authority still has unexpected live storage leases")
            with self._publication_lock():
                del self.authorities[request_id]
                self._published_capacity.pop(request_id, None)
                getattr(self, '_deferred_terminals', {}).pop(request_id, None)
        self._write_archive_report()

    def retirement_done(self, request_id):
        """Called on the engine thread before the native connector releases pages."""
        if request_id in getattr(self, '_deferred_terminals', {}) or not self.publication_ready((request_id,)):
            return False
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if sessions is not None:
            retained = sessions.retirement_done(request_id)
            if retained is not None:
                return retained
        if request_id in self.retiring:
            result = self.retirement.poll(request_id)
            if result is None:
                return False
            self.completed.append({**self.retiring.pop(request_id), **result,
                                   "retirement_ack_ready_at": time.perf_counter()})
            self._write_archive_report()
        done = request_id not in self.authorities
        if done and sessions is not None:
            sessions.ordinary_retired(request_id)
        return done

    def shutdown_retirement(self):
        """Graceful engine shutdown drains detached work before releasing storage."""
        self.join_cpu_scatter()
        executor=getattr(self,'_cpu_scatter_executor',None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._cpu_scatter_executor=None
        if not self.options.async_archive_retirement:
            return
        self.drain(wait=True)
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        if sessions is not None:
            deadline = time.monotonic() + 180
            while not sessions.close():
                if time.monotonic() >= deadline:
                    raise TimeoutError('Session checkpoint cleanup did not drain during shutdown')
                self.archive.wait_for_idle(.05)
        self.finish(set(self.authorities))
        for request_id in list(self.retiring):
            self.retirement.jobs[request_id].result(timeout=180)
            assert self.retirement_done(request_id)
        if not self.retirement.close() or not self.archive.close():
            raise RuntimeError('Shutdown retained archive I/O or retirement jobs')

    def _write_archive_report(self):
        sessions = getattr(self.runner, '_kvmem_sessions', None)
        report = {"scope": "continuous target authority; refresh qualification is recorded separately",
                  "batches": self.batches, "completed": list(self.completed),
                  **completion_counts(self.completed),
                  "block_tokens": self.config.block_tokens,
                  "memory_config": asdict(self.config),
                  "cpu_cache_capacity_bytes": self.options.cpu_read_cache_bytes,
                  "cpu_cache_used_bytes": getattr(getattr(self.archive, 'read_cache', None), 'used_bytes', 0),
                  "packed_assembly": self.options.packed_assembly,
                  "direct_pinned_reads": self.options.direct_pinned_reads,
                  "async_archive_retirement": self.options.async_archive_retirement,
                  "terminal_prefix_required": bool(getattr(self.runner, '_kvmem_bounded_target', False)),
                  "repository_gc": self.options.repository_gc,
                  "gc_wait_seconds": getattr(self, 'gc_wait_seconds', 0.0),
                  "gc_wait_count": getattr(self, 'gc_wait_count', 0),
                  "retiring_requests": len(getattr(self, 'retiring', {})),
                  "session_capacity": self.options.session_capacity,
                  "session_checkpoint_payload_bytes": sessions.pool.payload_bytes if sessions else 0,
                  "session_checkpoints": len(sessions.pool.entries) if sessions else 0,
                  "session_retiring_requests": len(sessions.retiring) if sessions else 0,
                  "session_restored_tokens": sessions.restored_tokens if sessions else 0,
                  "session_checkpoint_deferrals": sessions.checkpoint_deferrals if sessions else 0,
                  "archive_verification": "full_checksum" if self.options.verify_archive_on_finish else "publication_only",
                  "max_concurrent_batches": self.runner.vllm_config.max_concurrent_batches,
                  "decode_query_len": self.runner.decode_query_len,
                  "active_requests": len(self.authorities),
                  "block_graph_bytes": getattr(getattr(self, 'block_graph_budget', None), 'used', 0),
                  "block_graph_limit_bytes": getattr(getattr(self, 'block_graph_budget', None), 'maximum', 0),
                  "archive_used_slots": self.archive.tier.used_slots,
                  "nvme_capacity_bytes": 0 if self.options.archive_backend=='cpu' else self.archive.capacity_bytes,
                  "nvme_preallocated": self.archive.preallocated,
                  **self.archive.io_usage(),
                  "pending_downloads": self._pending_download_count()}
        temporary = self.output / 'archive.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(self.output / 'archive.json')
        self._write_progress()
