"""Continuous accepted-prefix archive runtime for the native target runner.

All target layers and requests share one packed D2H operation per batch. Native
sampling rejection counts travel in that same fenced packet. This stage provides
the immutable authority for refresh, but does not yet trigger retrieval/replay.
Finalization always drains I/O; diagnostic checksum readback is separately gated.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch

from .authority import RequestAuthority
from .options import RuntimeOptions
from .storage import AsyncArchive
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
        archive_records = self.config.repository_blocks * (max_reqs if getattr(runner, "_kvmem_bounded_target", False) else 1)
        archive_type, archive_options = AsyncArchive, {}
        if self.options.cpu_read_cache_bytes:
            from .cached_storage import CachedArchive
            archive_type = CachedArchive
            archive_options = dict(cache_bytes=self.options.cpu_read_cache_bytes,
                namespace_bytes=(self.config.select_budget + self.config.gen_budget) * self.row_elements * 3)
        self.archive = archive_type(str(output / "continuous_archive"), record_bytes, archive_records, **archive_options)
        self.downloads = PackedDownloads(max_tokens * self.row_elements * 3 + (4 * max_reqs + 1) * 8,
                                         2, runner.device)
        self.resident_records = max(16, (max_tokens + self.config.block_tokens - 1) // self.config.block_tokens + 1)
        self.authorities = {}
        self.queue = []
        self.accepted = {}
        self.rejected = {}
        self.last_steps = {}
        self.completed = []
        self.batches = 0
        self._progress_batch = -1

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
                        len(self.layers), self.heads, self.dim, resident_records=self.resident_records)
                    self.authorities[request_id] = authority
                    self.accepted[request_id] = self.rejected[request_id] = 0
                    self.last_steps[request_id] = []
                begin, end = starts[i:i+2]
                accepted = end - begin - rejected[i]
                success = authority.append(raw[:, begin:end], values[:, begin:end],
                    history_begin=histories[i], local_rope_begin=local[i], accepted=accepted)
                if not success:
                    if not wait:
                        return False
                    self._wait_archive_progress()
                    continue
                self.accepted[request_id] += accepted
                self.rejected[request_id] += rejected[i]
                steps = self.last_steps[request_id]
                steps.append({"history_begin": histories[i], "scheduled": end - begin,
                              "accepted": accepted, "rejected": rejected[i]})
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
            "completed_requests": len(self.completed)}
        temporary = self.output / "progress.tmp"
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(self.output / "progress.json")

    def _wait_archive_progress(self):
        # Shared queue pressure may come from another request, even when this
        # authority has not admitted its first write yet.
        progressed = sum(authority.poll() for authority in self.authorities.values())
        jobs = [job for authority in self.authorities.values() for job, _ in authority.writes]
        if not jobs:
            if progressed:
                return
            raise BufferError("Authority capacity exhausted; scheduler admission/GC must yield")
        for job in jobs:
            job.result()

    def finish(self, request_ids):
        """Drain publication, optionally checksum records, then reclaim the request."""
        if not request_ids:
            return
        self.drain(wait=True)
        for request_id in request_ids:
            authority = self.authorities.get(request_id)
            if authority is None:
                continue
            if not authority.flush():
                self._wait_archive_progress()
                if not authority.flush():
                    raise BufferError("Archive cannot finalize complete authority records")
            for job, _ in authority.writes:
                job.result()
            authority.poll()
            published = sum(record.archived for record in authority.records.values())
            if published != authority.history_end // self.config.block_tokens:
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
                "sealed_records_verified": verified})
            if not authority.close():
                raise RuntimeError("Finalized authority still has unexpected live storage leases")
            del self.authorities[request_id]
        report = {"scope": "continuous target authority; refresh qualification is recorded separately",
                  "batches": self.batches, "completed": self.completed,
                  "block_tokens": self.config.block_tokens,
                  "memory_config": asdict(self.config),
                  "cpu_cache_capacity_bytes": self.options.cpu_read_cache_bytes,
                  "cpu_cache_used_bytes": getattr(getattr(self.archive, 'read_cache', None), 'used_bytes', 0),
                  "packed_assembly": self.options.packed_assembly,
                  "archive_verification": "full_checksum" if self.options.verify_archive_on_finish else "publication_only",
                  "max_concurrent_batches": self.runner.vllm_config.max_concurrent_batches,
                  "decode_query_len": self.runner.decode_query_len,
                  "active_requests": len(self.authorities),
                  "archive_used_slots": self.archive.tier.used_slots,
                  "pending_downloads": len(self.downloads.pending)}
        (self.output / "archive.json").write_text(json.dumps(report, indent=2))
        self._write_progress()
