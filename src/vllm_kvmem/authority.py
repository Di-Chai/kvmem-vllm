"""Accepted-token raw-K / FP8-V authority with immutable sealed records.

CPU record buffers and admitted archive writes have a shared per-request bound.
The caller retries after poll() on backpressure. An incomplete tail is never
published to the immutable archive; repeated decode appends extend it in memory.
"""
from dataclasses import dataclass
import hashlib

import torch

from . import _native
from .memory import RequestMemory


class _Ready:
    def query(self):
        return True


@dataclass
class Record:
    handle: object
    history_begin: int
    count: int
    raw: torch.Tensor | None
    values: torch.Tensor | None
    key_sum: torch.Tensor
    archived: bool = False
    submitted: bool = False
    digest: str | None = None


def record_key(handle):
    return handle.slot, handle.generation


class RequestAuthority:
    def __init__(self, request_id, config, archive, layers, heads, head_dim,
                 raw_dtype=torch.bfloat16, resident_records=16):
        if min(layers, heads, head_dim, resident_records) <= 0 or raw_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Invalid authority geometry or staging budget")
        self.memory = RequestMemory(request_id, config)
        self.archive = archive
        self.shape = (layers, config.block_tokens, heads, head_dim)
        self.raw_dtype = raw_dtype
        self.record_bytes = layers * config.block_tokens * heads * head_dim * 3
        if archive.record_bytes != self.record_bytes:
            raise ValueError("Archive record format does not match model geometry")
        self.resident_records = resident_records
        self.records: dict[tuple[int, int], Record] = {}
        self.writes = []
        self.closing = False

    @property
    def history_end(self):
        return self.memory.history_end

    def poll(self):
        remaining = []
        completed = 0
        for job, keys in self.writes:
            if not job.done():
                remaining.append((job, keys))
                continue
            job.result()  # Failed I/O is fatal, never a successfully archived record.
            for key in keys:
                self.records[key].archived = True
            completed += 1
        self.writes = remaining
        return completed

    def flush(self):
        """Submit complete blocks only; false means archive queue/capacity pressure."""
        self.poll()
        records = [r for r in self.records.values()
                   if r.count == self.shape[1] and not r.submitted]
        if not records:
            return True
        leases = [self.memory.repository.pin(r.handle) for r in records]
        if not all(leases):
            raise RuntimeError("Authority record lost its repository generation")
        try:
            encoded = [r.raw.view(torch.uint8).numpy().tobytes()
                       + r.values.numpy().tobytes() for r in records]
            for record, data in zip(records, encoded):
                record.digest = hashlib.sha256(data).hexdigest()
            payload = b"".join(encoded)
            job = self.archive.write(self.memory.request_id, leases, payload)
        finally:
            for lease in leases:
                lease.reset()
        if job is None:
            return False
        keys = []
        for record in records:
            record.submitted = True
            record.raw = record.values = None
            keys.append(record_key(record.handle))
        self.writes.append((job, keys))
        return True

    def append(self, raw, values, *, history_begin, local_rope_begin, accepted=None):
        """Publish a CPU-ready accepted prefix; rejected candidate rows are ignored.

        raw/values: [layers, candidate tokens, KV heads, head_dim]. The caller
        must have observed its DMA fence before passing these CPU tensors.
        """
        if self.closing:
            raise RuntimeError("Authority is closing")
        if (raw.device.type != "cpu" or values.device.type != "cpu" or
                raw.dtype != self.raw_dtype or values.dtype != torch.uint8 or
                raw.ndim != 4 or raw.shape != values.shape or
                (raw.shape[0], raw.shape[2], raw.shape[3]) != (self.shape[0], self.shape[2], self.shape[3])):
            raise ValueError("Captured payload does not match the authority codec")
        count = raw.shape[1] if accepted is None else accepted
        if type(count) is not int or not 0 <= count <= raw.shape[1]:
            raise ValueError("Accepted prefix outside candidate rows")
        if history_begin != self.history_end:
            raise ValueError("Capture skipped or repeated logical history")
        if count == 0:
            return True
        self.flush()
        block = self.shape[1]
        # Includes I/O-owned byte snapshots after torch staging was released.
        resident = sum(not r.archived for r in self.records.values())
        tail = self.history_end % block
        new_records = (tail + count + block - 1) // block - bool(tail)
        if resident + new_records > self.resident_records:
            return False
        spans = self.memory.begin_append(count, local_rope_begin)
        if spans is None:
            return False
        # Validate the complete write map before mutating any existing tail.
        for span in spans:
            key = record_key(span.handle)
            offset = span.storage_begin % block
            previous = self.records.get(key)
            if ((offset and previous is None) or
                    (previous is not None and (previous.count != offset or previous.submitted))):
                raise RuntimeError("Repository append no longer matches the authoritative tail")
        for span in spans:
            key = record_key(span.handle)
            offset = span.storage_begin % block
            start = span.history_begin - history_begin
            n = span.token_count
            record = self.records.get(key)
            if record is None:
                # pybind field views retain their Append/Store owner. Persist a
                # value handle, otherwise an index entry pins the entire store
                # and prevents request cancellation from reclaiming any record.
                handle = _native.Handle(span.handle.slot, span.handle.generation)
                record = Record(handle, span.history_begin, 0,
                    torch.empty(self.shape, dtype=self.raw_dtype),
                    torch.empty(self.shape, dtype=torch.uint8),
                    torch.zeros((self.shape[0], self.shape[2], self.shape[3]), dtype=torch.float32))
                self.records[key] = record
            record.raw[:, offset:offset+n].copy_(raw[:, start:start+n])
            record.values[:, offset:offset+n].copy_(values[:, start:start+n])
            record.key_sum.add_(raw[:, start:start+n].float().sum(dim=1))
            record.count += n
        self.memory.attach_completion(_Ready())
        assert self.memory.commit_append(count)
        self.flush()
        return True

    def mean_keys(self):
        """Bounded CPU index snapshot in logical history order, with generation IDs."""
        self.poll()
        ordered = sorted(self.records.values(), key=lambda r: r.history_begin)
        if not ordered:
            raise ValueError("No accepted tokens in the index")
        return ([r.handle for r in ordered],
                torch.stack([r.key_sum / r.count for r in ordered], dim=1))

    def read_record(self, handle):
        return self.read_records([handle])[0]

    def read_records(self, handles):
        """Read a bounded batch through one coalescible native archive I/O job."""
        self.poll()
        records = [self.records[record_key(handle)] for handle in handles]
        leases = [self.memory.repository.pin(handle) for handle in handles]
        try:
            if not all(leases):
                raise ValueError("Stale authority handle")
            disk_indices = [i for i, record in enumerate(records) if record.raw is None]
            if any(not records[i].archived for i in disk_indices):
                raise BlockingIOError("Immutable record write is still in flight")
            result = [None] * len(records)
            for i, record in enumerate(records):
                if record.raw is not None:
                    result[i] = (record.raw[:, :record.count].clone(), record.values[:, :record.count].clone())
            if not disk_indices:
                return result
            job = self.archive.read(self.memory.request_id, [leases[i] for i in disk_indices])
            if job is None:
                raise BlockingIOError("Archive read queue is full")
            data, _ = job.result()
            elements = self.record_bytes // 3
            for packet_index, i in enumerate(disk_indices):
                offset = packet_index * self.record_bytes
                raw = torch.frombuffer(data, dtype=self.raw_dtype, count=elements, offset=offset).reshape(self.shape)
                values = torch.frombuffer(data, dtype=torch.uint8, count=elements,
                                          offset=offset + elements * 2).reshape(self.shape)
                count = records[i].count
                result[i] = (raw[:, :count].clone(), values[:, :count].clone())
            return result
        finally:
            for lease in leases:
                lease.reset()

    def select_for_queries(self, queries, mandatory_handles=()):
        """Reference MeanK-to-native-selection bridge over accepted history only.

        The returned plan still needs fenced GPU assembly and commit_selection.
        Queries must come from the model's content-frame query probe/replay.
        """
        if self.closing:
            raise RuntimeError("Authority is closing")
        from .retrieval import mean_key_scores
        handles, means = self.mean_keys()
        keys = [record_key(h) for h in handles]
        mandatory = {record_key(h) for h in mandatory_handles}
        if not mandatory.issubset(self.records):
            raise ValueError("Mandatory selection uses a stale or foreign record")
        cfg = self.memory.config
        eligible = torch.tensor([
            self.records[key].history_begin >= cfg.sink_blocks * cfg.block_tokens
            and i < len(keys) - cfg.recent_blocks and key not in mandatory
            for i, key in enumerate(keys)], dtype=torch.bool, device=queries.device)
        scores = mean_key_scores(queries, means.to(queries.device), eligible).cpu().tolist()
        return self.memory.select({h.slot: score for h, score in zip(handles, scores)},
                                  tuple(slot for slot, _ in mandatory))

    def close(self):
        self.closing = True
        self.poll()
        if self.writes or not self.memory.close():
            return False
        all_reclaimed = True
        for key, record in list(self.records.items()):
            result = self.memory.repository.reclaim(record.handle,
                lambda entry: self.archive.reclaim(self.memory.request_id, entry))
            if result == _native.ReclaimResult.RECLAIMED:
                del self.records[key]
            else:
                all_reclaimed = False
        return all_reclaimed
