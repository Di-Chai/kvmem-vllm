"""Continuous accepted-prefix archive runtime for the native target runner.

All target layers and requests share one packed D2H operation per batch. Native
sampling rejection counts travel in that same fenced packet. This stage provides
the immutable authority for refresh, but does not yet trigger retrieval/replay.
Finalization always drains I/O; diagnostic checksum readback is separately gated.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import time
from threading import Event

import torch

from .authority import RequestAuthority
from .options import RuntimeOptions
from .storage import make_shared_archive
from .transfer import PackedDownloads


@dataclass
class CapturedPacket:
    request_ids: tuple[str, ...]
    tokens: int
    download: object
    next_request: int = 0


class CaptureRuntime:
    def __init__(self, runner, output):
        self.runner = runner
        self.output = output
        self.options = RuntimeOptions.from_vllm(runner.vllm_config)
        self.config = self.options.memory
        modules = dict(runner.model.named_modules())
        self.names = tuple(runner._kvmem_layers)
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
        record_bytes = self.row_elements * self.config.block_tokens * 3
        output.mkdir(parents=True, exist_ok=True)
        max_tokens = runner.vllm_config.scheduler_config.max_num_batched_tokens
        max_reqs = runner.vllm_config.scheduler_config.max_num_seqs
        self.gc_config = None
        self.gc_progress = Event()
        self.gc_wait_seconds = 0.0
        self.gc_wait_count = 0
        self._archive_wait_deadline = None
        if self.options.repository_gc:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Serving repository GC requires bounded execution')
            from .repository_gc import serving_gc_config
            self.gc_config = serving_gc_config(self.config, max_tokens)
        self.archive = getattr(runner, '_kvmem_prepared_archive', None)
        if self.archive is None:
            self.archive = make_shared_archive(self.options, output / 'continuous_archive', record_bytes,
                max_reqs if getattr(runner, '_kvmem_bounded_target', False) else 1)
        self.downloads = PackedDownloads(max_tokens * self.row_elements * 3 + (4 * max_reqs + 1) * 8,
                                         2, runner.device)
        self.resident_records = max(16, (max_tokens + self.config.block_tokens - 1) // self.config.block_tokens + 1)
        self.authorities = {}
        self.queue = []
        self.accepted = {}
        self.rejected = {}
        self.last_steps = {}
        self.terminal_prefixes = {}
        self.terminal_discarded = {}
        self.completed = []
        self.batches = 0
        self._progress_batch = -1
        self.retiring = {}
        self.retirement = None
        if self.options.async_archive_retirement:
            if not getattr(runner, '_kvmem_bounded_target', False):
                raise ValueError('Asynchronous retirement requires native bounded authority admission')
            from .retirement import RetirementQueue
            self.retirement = RetirementQueue(max_reqs, self.options.verify_archive_on_finish)

    def submit(self, captured, batch, rejected):
        if tuple(c.name for c in captured) != self.names:
            raise ValueError("Continuous capture missed or reordered target layers")
        if any(c.key.dtype != torch.bfloat16 for c in captured):
            raise ValueError("Continuous Qwen target capture currently qualifies BF16 activations")
        self.drain(wait=False)
        if len(self.downloads.pending) == len(self.downloads.buffers):
            self.drain(wait=True)
        n, t = batch.num_reqs, batch.num_tokens
        starts = batch.query_start_loc[:n+1].to(torch.int64)
        global_begin = batch.positions[starts[:-1]]
        positions = captured[0].positions
        if positions.ndim == 2:
            positions = positions[0]  # Text-only model adapter.
        local_begin = positions[starts[:-1]]
        header = torch.cat((starts, global_begin, local_begin, rejected[:n].to(torch.int64)))
        raw = torch.stack([c.key[:t] for c in captured])
        values = torch.stack([c.value[:t] for c in captured]).float()
        values = (values / self.scales[:, None, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        payload = torch.cat((header.view(torch.uint8), raw.view(torch.uint8).flatten(),
                             values.view(torch.uint8).flatten()))
        download = self.downloads.download(payload)
        if download is None:
            raise RuntimeError("Capture queue failed to provide bounded staging after drain")
        self.queue.append(CapturedPacket(tuple(batch.req_ids), t, download))
        self.batches += 1

    def drain(self, wait=False):
        while self.queue:
            packet = self.queue[0]
            if not packet.download.query():
                if not wait:
                    return False
                packet.download.event.synchronize()
            n, t = len(packet.request_ids), packet.tokens
            data = packet.download.payload
            header_bytes = (4 * n + 1) * 8
            header = data[:header_bytes].view(torch.int64).tolist()
            starts = header[:n+1]
            histories = header[n+1:2*n+1]
            local = header[2*n+1:3*n+1]
            rejected = header[3*n+1:]
            elements = self.row_elements * t
            shape = (len(self.layers), t, self.heads, self.dim)
            raw = data[header_bytes:header_bytes+elements*2].view(torch.bfloat16).reshape(shape)
            values = data[header_bytes+elements*2:].reshape(shape)
            while packet.next_request < n:
                i = packet.next_request
                request_id = packet.request_ids[i]
                authority = self.authorities.get(request_id)
                if authority is None:
                    authority = RequestAuthority(request_id, self.config, self.archive,
                        len(self.layers), self.heads, self.dim, resident_records=self.resident_records,
                        gc_config=self.gc_config, gc_progress=self.gc_progress)
                    self.authorities[request_id] = authority
                    self.accepted[request_id] = self.rejected[request_id] = 0
                    self.last_steps[request_id] = []
                begin, end = starts[i:i+2]
                accepted = end - begin - rejected[i]
                if not 0 <= accepted <= end - begin:
                    raise ValueError('Native sampling rejection count exceeds captured rows')
                discarded = 0
                terminal = self.terminal_prefixes.get(request_id)
                if terminal is not None:
                    retained = max(0, min(accepted, terminal.history_end - histories[i]))
                    discarded = accepted - retained
                    accepted = retained
                success = not accepted or authority.append(raw[:, begin:end], values[:, begin:end],
                    history_begin=histories[i], local_rope_begin=local[i], accepted=accepted)
                if not success:
                    if not wait:
                        return False
                    self._wait_archive_progress()
                    continue
                self.accepted[request_id] += accepted
                self._archive_wait_deadline = None
                self.rejected[request_id] += rejected[i]
                self.terminal_discarded[request_id] = self.terminal_discarded.get(request_id, 0) + discarded
                steps = self.last_steps[request_id]
                steps.append({"history_begin": histories[i], "scheduled": end - begin,
                              "accepted": accepted, "rejected": rejected[i],
                              "terminal_discarded": discarded})
                del steps[:max(0, len(steps) - self.runner.vllm_config.max_concurrent_batches - 1)]
                packet.next_request += 1
            assert self.downloads.release(packet.download)
            self.queue.pop(0)
            if self.batches // 8 != self._progress_batch:
                self._progress_batch = self.batches // 8
                self._write_progress()
        return True

    def _write_progress(self):
        progress = {"batches_submitted": self.batches,
            "accepted_history": {req: owner.history_end for req, owner in self.authorities.items()},
            "archive_used_slots": self.archive.tier.used_slots,
            "pending_downloads": len(self.downloads.pending),
                    "completed_requests": len(self.completed),
                    "retiring_requests": len(getattr(self, 'retiring', {}))}
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
        result = {}
        for request_id, authority in self.authorities.items():
            authority.poll()
            result[request_id] = ArchiveCapacity(authority.history_end,
                                                 authority.memory.repository.ready_slots)
        return result

    def _wait_archive_progress(self):
        # Shared queue pressure may come from another request, even when this
        # authority has not admitted its first write yet.
        progressed = sum(authority.poll() for authority in self.authorities.values())
        jobs = [job for authority in self.authorities.values() for job, _ in authority.writes]
        if not jobs:
            if progressed:
                self._archive_wait_deadline = None
                return
            if self.options.repository_gc:
                # The CPU policy worker may be reclaiming durable generations
                # even with no outstanding archive write. Never run policy on
                # this engine thread or silently discard captured accepted rows.
                # This synchronous fallback is measured; scheduler-side capacity
                # feedback is still required to remove such serving stalls.
                started = time.monotonic()
                self._archive_wait_deadline = self._archive_wait_deadline or started + 30
                if started >= self._archive_wait_deadline:
                    raise TimeoutError('Repository GC made no append progress for 30 seconds')
                self.gc_progress.wait(min(.05, self._archive_wait_deadline - started))
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
        if terminal_prefixes is not None:
            for request_id in request_ids:
                known = request_id in self.authorities or any(request_id in p.request_ids for p in self.queue)
                prefix = terminal_prefixes.get(request_id)
                if known and prefix is None:
                    raise RuntimeError('Missing scheduler terminal prefix for captured request')
                if prefix is None or not known:
                    continue
                authority = self.authorities.get(request_id)
                if authority is not None and authority.history_end > prefix.history_end:
                    raise RuntimeError('Terminal prefix arrived after excess history was published')
                self.terminal_prefixes[request_id] = prefix
        self.drain(wait=True)
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
                del self.authorities[request_id]
                continue
            authority.stop_gc()
            if not authority.flush():
                self._wait_archive_progress()
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
            del self.authorities[request_id]
        self._write_archive_report()

    def retirement_done(self, request_id):
        """Called on the engine thread before the native connector releases pages."""
        if request_id in self.retiring:
            result = self.retirement.poll(request_id)
            if result is None:
                return False
            self.completed.append({**self.retiring.pop(request_id), **result,
                                   "retirement_ack_ready_at": time.perf_counter()})
            self._write_archive_report()
        return (request_id not in self.authorities and
                all(request_id not in packet.request_ids for packet in self.queue))

    def shutdown_retirement(self):
        """Graceful engine shutdown drains detached work before releasing storage."""
        if not self.options.async_archive_retirement:
            return
        self.drain(wait=True)
        self.finish(set(self.authorities))
        for request_id in list(self.retiring):
            self.retirement.jobs[request_id].result(timeout=180)
            assert self.retirement_done(request_id)
        if not self.retirement.close() or not self.archive.close():
            raise RuntimeError('Shutdown retained archive I/O or retirement jobs')

    def _write_archive_report(self):
        report = {"scope": "continuous target authority; refresh qualification is recorded separately",
                  "batches": self.batches, "completed": self.completed,
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
                  "archive_verification": "full_checksum" if self.options.verify_archive_on_finish else "publication_only",
                  "max_concurrent_batches": self.runner.vllm_config.max_concurrent_batches,
                  "decode_query_len": self.runner.decode_query_len,
                  "active_requests": len(self.authorities),
                  "archive_used_slots": self.archive.tier.used_slots,
                  "nvme_capacity_bytes": self.archive.capacity_bytes,
                  "nvme_preallocated": self.archive.preallocated,
                  "pending_downloads": len(self.downloads.pending)}
        temporary = self.output / 'archive.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(self.output / 'archive.json')
        self._write_progress()
