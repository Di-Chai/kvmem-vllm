"""Accepted-token raw-K / FP8-V authority with immutable sealed records.

CPU record buffers and admitted archive writes have a shared per-request bound.
The caller retries after poll() on backpressure. An incomplete tail is never
published to the immutable archive; repeated decode appends extend it in memory.
"""
from dataclasses import dataclass, field
from contextlib import contextmanager
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
    scatter_views: tuple | None = field(default=None, repr=False)


def record_key(handle):
    return handle.slot, handle.generation


class RequestAuthority:
    def __init__(self, request_id, config, archive, layers, heads, head_dim,
                 raw_dtype=torch.bfloat16, resident_records=16, gc_config=None, gc_progress=None,
                 scoring_layers=None):
        if min(layers, heads, head_dim, resident_records) <= 0 or raw_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Invalid authority geometry or staging budget")
        self.memory = RequestMemory(request_id, config)
        self.archive = archive
        self.shape = (layers, config.block_tokens, heads, head_dim)
        self.scoring_layers = layers if scoring_layers is None else scoring_layers
        if type(self.scoring_layers) is not int or not 0 < self.scoring_layers <= layers:
            raise ValueError('MeanK scoring must cover a positive target layer prefix')
        self.raw_dtype = raw_dtype
        self.record_bytes = layers * config.block_tokens * heads * head_dim * 3
        if archive.record_bytes != self.record_bytes:
            raise ValueError("Archive record format does not match model geometry")
        self.resident_records = resident_records
        self.records: dict[tuple[int, int], Record] = {}
        # These contain only live, not-yet-published generations, so ordinary
        # decode bookkeeping does not grow with archived history. GC can only
        # remove archived records; rewind explicitly removes its live suffix.
        self._unsubmitted: dict[tuple[int, int], Record] = {}
        self._resident_count = 0
        self.writes = []
        self.closing = False
        self.discarded_records: dict[tuple[int, int], Record] = {}
        self._rewind_position = None
        self._rewind_committed = False
        self.gc = None
        if gc_config is not None:
            from .repository_gc import AuthorityGc
            self.gc = AuthorityGc(self.memory, archive, gc_config, gc_progress)

    @property
    def history_end(self):
        return self.memory.history_end

    def poll(self):
        if self.gc is not None:
            self.gc.drain(self.records, self.discarded_records)
        remaining = []
        completed = 0
        for job, keys in self.writes:
            if not job.done():
                remaining.append((job, keys))
                continue
            result = job.result()  # Failed I/O is fatal, never a successfully archived record.
            digests = result.get('record_digests') if isinstance(result, dict) else None
            if digests is not None and len(digests) != len(keys):
                raise RuntimeError('Archived checksum count differs from sealed records')
            for index, key in enumerate(keys):
                record = self.records[key]
                if digests is not None:
                    record.digest = digests[index]
                if not record.archived:
                    self._resident_count -= 1
                record.archived = True
                # Publish only after durable write completion, never on CPU
                # cache admission or asynchronous I/O submission.
                flags = _native.GC_RAW_BACKED | _native.GC_VALUE_BACKED | _native.GC_INDEXED
                self.memory._store.gc_flags(record.handle, flags, flags)
            completed += 1
        self.writes = remaining
        for key, record in list(self.discarded_records.items()):
            result = self.memory.repository.reclaim(record.handle,
                lambda entry: self.archive.reclaim(self.memory.request_id, entry))
            if result == _native.ReclaimResult.RECLAIMED:
                del self.discarded_records[key]
        if self.gc is not None:
            # The background reclaimer may have finished a discarded generation
            # between our first drain and reclaim attempt. Its tombstone owns
            # index removal; neither worker fabricates a second release.
            self.gc.drain(self.records, self.discarded_records)
        return completed

    def _require_open(self):
        if self.closing:
            raise RuntimeError('Authority is closing')
        if self._rewind_position is not None:
            raise RuntimeError('Authority rewind has not finished suffix cleanup')

    def flush(self):
        """Submit complete blocks only; false means archive queue/capacity pressure."""
        self._require_open()
        self.poll()
        records = list(self._unsubmitted.values())
        if not records:
            return True
        leases = [self.memory.repository.pin(r.handle) for r in records]
        try:
            if not all(leases):
                raise RuntimeError("Authority record lost its repository generation")
            # Join tensor buffer views directly into the immutable I/O snapshot.
            # Avoid raw/V tobytes, per-record concatenations and their copies;
            # row layout and per-record checksums remain byte-for-byte identical.
            views = []
            for record in records:
                views.extend((memoryview(record.raw.view(torch.uint8).numpy()).cast('B'),
                              memoryview(record.values.numpy()).cast('B')))
            if getattr(self.archive, 'async_seal', False):
                job = self.archive.write_from_views(self.memory.request_id, leases, views)
            else:
                payload = b"".join(views)
                view = memoryview(payload)
                for i, record in enumerate(records):
                    begin = i * self.record_bytes
                    record.digest = hashlib.sha256(view[begin:begin+self.record_bytes]).hexdigest()
                job = self.archive.write(self.memory.request_id, leases, payload)
        finally:
            for lease in leases:
                lease.reset()
        if job is None:
            return False
        keys = []
        for record in records:
            record.submitted = True
            record.scatter_views = None
            record.raw = record.values = None
            key = record_key(record.handle)
            del self._unsubmitted[key]
            keys.append(key)
        self.writes.append((job, keys))
        return True

    def _prepare_append(self, raw, values, *, history_begin, local_rope_begin, accepted=None):
        """Reserve a CPU-ready accepted prefix; no copied rows are published here.

        raw/values: [layers, candidate tokens, KV heads, head_dim]. The caller
        must have observed its DMA fence before passing these CPU tensors.
        """
        self._require_open()
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
            return (raw, values, count, [])
        self.flush()
        block = self.shape[1]
        # Includes I/O-owned byte snapshots after torch staging was released.
        tail = self.history_end % block
        new_records = (tail + count + block - 1) // block - bool(tail)
        if self._resident_count + new_records > self.resident_records:
            return None
        spans = self.memory.begin_append(count, local_rope_begin)
        if spans is None:
            return None
        if self.gc is not None:
            # Recycled reservations become ready after their release callback
            # queued the old generation. Drop it before installing a new index.
            self.gc.drain(self.records, self.discarded_records)
        # Validate the complete write map before mutating any existing tail.
        for span in spans:
            key = record_key(span.handle)
            offset = span.storage_begin % block
            previous = self.records.get(key)
            if ((offset and previous is None) or
                    (previous is not None and (previous.count != offset or previous.submitted))):
                raise RuntimeError("Repository append no longer matches the authoritative tail")
        rows = []
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
                    torch.zeros((self.scoring_layers, self.shape[2], self.shape[3]), dtype=torch.float32))
                self.records[key] = record
                self._resident_count += 1
            rows.append((key, record, offset, start, n))
        return raw, values, count, rows

    def _commit_prepared_append(self, prepared):
        raw, values, count, rows = prepared
        if not count:
            return
        for key, record, offset, start, n in rows:
            record.count += n
            if record.count == self.shape[1]:
                self._unsubmitted[key] = record
        self.memory.attach_completion(_Ready())
        assert self.memory.commit_append(count)
        self.flush()

    def append(self, raw, values, *, history_begin, local_rope_begin, accepted=None):
        prepared = self._prepare_append(raw, values, history_begin=history_begin,
            local_rope_begin=local_rope_begin, accepted=accepted)
        if prepared is None:
            return False
        raw, values, count, rows = prepared
        for key, record, offset, start, n in rows:
            record.raw[:, offset:offset+n].copy_(raw[:, start:start+n])
            record.values[:, offset:offset+n].copy_(values[:, start:start+n])
            record.key_sum.add_(raw[:record.key_sum.shape[0], start:start+n].float().sum(dim=1))
        self._commit_prepared_append(prepared)
        return True

    def publish_sealed(self, wait_for_progress):
        """Complete the read barrier, including writes refused by admission.

        Append may keep sealed CPU records when another namespace fills the
        shared queue. Waiting only for this authority's admitted jobs misses
        those records; retry admission before any published-only packed read.
        """
        while not self.flush():
            wait_for_progress()
        for job, _ in tuple(self.writes):
            job.result()
        self.poll()

    @contextmanager
    def index_snapshot(self):
        """Pin every scored generation until selection has acquired its leases."""
        self._require_open()
        self.poll()
        leases, ordered = [], []
        try:
            for record in sorted(self.records.values(), key=lambda r: r.history_begin):
                lease = self.memory.repository.pin(record.handle)
                if lease:
                    leases.append(lease)
                    ordered.append(record)
            if not ordered:
                raise ValueError("No accepted tokens in the index")
            yield ordered, torch.stack([r.key_sum / r.count for r in ordered], dim=1)
        finally:
            for lease in leases:
                lease.reset()
            if self.gc is not None:
                self.gc.worker.notify_progress()

    def mean_keys(self):
        """Detached diagnostic copy; use index_snapshot() for retrieval/selection."""
        with self.index_snapshot() as (records, means):
            return [r.handle for r in records], means

    def read_record(self, handle):
        return self.read_records([handle])[0]

    def read_records(self, handles):
        """Read a bounded batch through one coalescible native archive I/O job."""
        self._require_open()
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
        self._require_open()
        from .retrieval import mean_key_scores
        with self.index_snapshot() as (records, means):
            # MTP records travel with target KV but never alter MeanK scoring.
            means = means[:getattr(self, 'scoring_layers', self.shape[0])]
            handles = [r.handle for r in records]
            keys = [record_key(h) for h in handles]
            mandatory = {record_key(h) for h in mandatory_handles}
            if not mandatory.issubset(keys):
                raise ValueError("Mandatory selection uses a stale or foreign record")
            cfg = self.memory.config
            eligible = torch.tensor([
                record.history_begin >= cfg.sink_blocks * cfg.block_tokens
                and i < len(keys) - cfg.recent_blocks and key not in mandatory
                for i, (key, record) in enumerate(zip(keys, records))],
                dtype=torch.bool, device=queries.device)
            scores = mean_key_scores(queries, means.to(queries.device), eligible).cpu().tolist()
            return self.memory.select({h.slot: score for h, score in zip(handles, scores)},
                                      tuple(slot for slot, _ in mandatory))

    def begin_rewind(self, checkpoint):
        """Quiesce publication and prepare a block-aligned branch rollback.

        None means existing archive publication must drain first; an empty tuple
        is a valid no-suffix transaction. The caller must also quiesce GPU/model
        users and supply their cleanup fence before poll_rewind can commit.
        """
        self._require_open()
        if not self.memory.checkpoint_valid(checkpoint):
            raise ValueError('Stale or foreign authority checkpoint')
        if checkpoint.position % self.shape[1]:
            raise ValueError('Rewind requires a sealed-record boundary')
        if not self.flush() or self.writes:
            return None
        spans = self.memory.begin_rewind(checkpoint)
        self._rewind_position = checkpoint.position
        self._rewind_committed = False
        return spans

    def attach_rewind_completion(self, fence):
        if self._rewind_position is None or self._rewind_committed:
            raise RuntimeError('No uncommitted authority rewind')
        self.memory.attach_completion(fence)

    def poll_rewind(self):
        """Retire the old index, then yield until all suffix I/O readers release."""
        if self._rewind_position is None or self.closing:
            raise RuntimeError('No live authority rewind')
        if not self._rewind_committed:
            if not self.memory.commit_rewind():
                return False
            self._rewind_committed = True
            for key, record in list(self.records.items()):
                if record.history_begin >= self._rewind_position:
                    self.discarded_records[key] = self.records.pop(key)
                    self._unsubmitted.pop(key, None)
                    if not record.archived:
                        self._resident_count -= 1
            if self.gc is not None:
                # GC accounting is branch-relative. Previously collected suffix
                # records are not holes in the now-shorter accepted prefix.
                self.gc.reclaimed_records = self.history_end // self.shape[1] - len(self.records)
                if self.gc.reclaimed_records < 0:
                    raise RuntimeError('Rewound index exceeds its sealed prefix')
        self.poll()
        if self.discarded_records:
            return False
        self._rewind_position = None
        self._rewind_committed = False
        return True

    def close(self):
        self.closing = True
        self.stop_gc()
        self.poll()
        if self.writes or not self.memory.close():
            return False
        all_reclaimed = True
        for records in (self.records, self.discarded_records):
            for key, record in list(records.items()):
                result = self.memory.repository.reclaim(record.handle,
                    lambda entry: self.archive.reclaim(self.memory.request_id, entry))
                if result == _native.ReclaimResult.RECLAIMED:
                    del records[key]
                    if records is self.records:
                        self._unsubmitted.pop(key, None)
                        if not record.archived:
                            self._resident_count -= 1
                else:
                    all_reclaimed = False
        return all_reclaimed

    def stop_gc(self):
        """Freeze the live index before final checksum iteration and teardown."""
        if self.gc is not None:
            self.gc.stop()
            self.gc.drain(self.records, self.discarded_records)

    @property
    def gc_reclaimed_records(self):
        return self.gc.reclaimed_records if self.gc is not None else 0


# The fast reduction is qualified against the pinned ATen outer-reduction
# kernel on x86; other Torch versions/CPUs use the reference operation.
_FUSED_DECODE_SUM = (torch.__version__.split('+')[0].startswith('2.13.') and
                     torch.backends.cpu.get_cpu_capability() in ('AVX2', 'AVX512'))


def append_authority_batch(entries):
    """Copy a capacity-admitted prefix of one packet in one GIL-free call.

    Each entry is (authority, raw, values, append keyword arguments). The caller
    has already clipped rejected/EOS/cancelled rows and observed the DMA fence.
    Small qualified BF16 decode reductions use the same FP32 addition order;
    other geometries and larger prefixes retain the reference Torch reduction. No prefix is published before
    all admitted copies complete. Backpressure consumes only the returned prefix.
    """
    if len({id(entry[0]) for entry in entries}) != len(entries):
        raise ValueError('A history batch must contain distinct authority owners')
    plans, jobs = [], []
    for authority, raw, values, kwargs in entries:
        # Later queued packets can start beyond an EOS/cancelled prefix. They
        # are fully discarded, as on the reference drain path, without opening
        # a transaction or requiring authority.history_end to reach those rows.
        if kwargs.get('accepted') == 0:
            plans.append((authority, (raw, values, 0, [])))
            continue
        prepared = authority._prepare_append(raw, values, **kwargs)
        if prepared is None:
            break
        plans.append((authority, prepared))
        source_raw = raw.view(torch.uint8).numpy()
        source_values = values.numpy()
        for key, record, offset, start, n in prepared[3]:
            fused = (raw.dtype == torch.bfloat16 and n <= 4 and
                     raw.shape[2] * raw.shape[3] % 64 == 0 and _FUSED_DECODE_SUM)
            delta = (None if fused else
                     raw[:record.key_sum.shape[0], start:start+n].float().sum(dim=1).numpy())
            if record.scatter_views is None:
                record.scatter_views = (record.raw.view(torch.uint8).numpy(),
                                        record.values.numpy(), record.key_sum.numpy())
            destination_raw, destination_values, sums = record.scatter_views
            jobs.append((source_raw, source_values, destination_raw,
                         destination_values, delta, sums, start, offset, n))
    _native.scatter_history_batch(jobs)
    for authority, prepared in plans:
        authority._commit_prepared_append(prepared)
    return len(plans)
