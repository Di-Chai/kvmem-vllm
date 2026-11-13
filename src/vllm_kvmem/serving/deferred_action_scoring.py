"""Bounded, private inputs for delayed native-score validation and fallback.

The publication packet, not this module, protects the original FP8 K prefix
against retirement and in-place assembly. Q, tables and tensor scales are copied
because the next forward may overwrite them. A pool has exactly two generations
and one serial fallback stream; it never borrows a live Action/native bank.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from threading import RLock

import torch


@dataclass(frozen=True)
class FrozenScoreLimits:
    layers: int
    groups: int
    heads: int
    dim: int
    page_tokens: int
    block_tokens: int
    max_rows: int
    max_owners: int
    max_segments: int
    max_length: int

    def __post_init__(self):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
               for x in vars(self).values()):
            raise ValueError('Frozen scoring capacities must be positive integers')

    @property
    def max_pages(self):
        return (self.max_length + self.page_tokens - 1) // self.page_tokens

    def budget(self, *, fused):
        # Tensor payload only. Events/allocator bookkeeping and external packet
        # masses/K/cos_sin retain their original accounting and are not duplicated.
        q = self.layers * self.max_rows * self.heads * self.dim * 2
        coordinates = 2 * self.max_rows * 8
        segments = 2 * self.max_owners * self.max_segments * 4
        tables = self.max_owners * self.groups * self.max_pages * 4
        scales = self.layers * 2 * 4
        gpu = 2 * (q + coordinates + segments + tables + scales)
        host = 2 * (coordinates + segments + self.layers * 4)
        rows = min(self.max_rows, 64)
        width = max(32, 1 << (self.max_length - 1).bit_length())
        # page_attention tiles independent Q rows at 64. Reserve simultaneous
        # logits+softmax or fused max/sum, RoPE/codec temporaries, recursive row
        # concatenation, and retained mass/add/mean outputs. One worker holds
        # this reservation until its final D2H completes, shared by both slots.
        reduction = (2 * rows * self.heads * self.max_segments * 4
                     if fused and self.block_tokens <= 128 else
                     2 * rows * self.heads * width * 4)
        scratch = (reduction + 16 * rows * self.heads * self.dim * 4 +
                   8 * self.max_rows * self.max_segments * 4)
        return dict(gpu_bytes=gpu, host_bytes=host, resident_bytes=gpu + host,
                    scratch_bytes=scratch, reserved_bytes=gpu + host + scratch)


@dataclass(frozen=True)
class FrozenLayerInput:
    name: str
    key: object
    cos_sin: object
    key_scale: object
    q_scale: object
    scale: float
    q_dtype: object
    group: int = 0


@dataclass(frozen=True)
class FrozenOwnerInput:
    request_id: str
    queries: tuple
    query_rows: tuple
    positions: tuple
    starts: tuple
    counts: tuple
    length: int
    tables: tuple
    masses: object


@dataclass(frozen=True)
class _FrozenOwner:
    request_id: str
    begin: int
    rows: int
    segments: int
    length: int
    masses: object


class FrozenScoreInputs:
    """Single-use handle. Storage can be recycled; a stale handle cannot."""
    def __init__(self, storage, generation, layers, owners, packet_lease):
        self.storage = storage
        self.generation = generation
        self.layers = layers
        self.owners = owners
        self.packet_lease = packet_lease
        self.sources = ()
        self.state = 'private'
        self.error = None
        self.host_flags = storage.host_flags
        self.event = storage.event
        self.backend_counts = {}


class _CUDA:
    def __init__(self, device):
        self.device = device

    def empty(self, shape, dtype, *, host=False):
        return torch.empty(shape, dtype=dtype, device='cpu' if host else self.device,
                           pin_memory=host)

    def event(self):
        return torch.cuda.Event()

    def stream(self):
        return torch.cuda.Stream(device=self.device)

    def current(self):
        return torch.cuda.current_stream(self.device)

    @contextmanager
    def context(self, stream):
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            yield

    def track(self, tensor, stream):
        tensor.record_stream(stream)


class _Storage:
    def __init__(self, limits, backend):
        l = limits
        self.q = backend.empty((l.layers, l.max_rows, l.heads, l.dim), torch.bfloat16)
        self.rows = backend.empty((l.max_rows,), torch.int64)
        self.positions = backend.empty((l.max_rows,), torch.int64)
        self.starts = backend.empty((l.max_owners, l.max_segments), torch.int32)
        self.counts = backend.empty((l.max_owners, l.max_segments), torch.int32)
        self.tables = backend.empty((l.max_owners, l.groups, l.max_pages), torch.int32)
        self.scales = backend.empty((l.layers, 2), torch.float32)
        self.host_rows = backend.empty((l.max_rows,), torch.int64, host=True)
        self.host_positions = backend.empty((l.max_rows,), torch.int64, host=True)
        self.host_starts = backend.empty((l.max_owners, l.max_segments), torch.int32, host=True)
        self.host_counts = backend.empty((l.max_owners, l.max_segments), torch.int32, host=True)
        self.host_flags = backend.empty((l.layers,), torch.int32, host=True)
        self.event, self.fallback_event = backend.event(), backend.event()
        self.active = None
        self.generation = 0


class FrozenScorePool:
    SLOTS = 2

    def __init__(self, limits, device, *, fused, max_bytes, max_scratch_bytes,
                 _backend=None, _scorer=None):
        if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
               for x in (max_bytes, max_scratch_bytes)):
            raise ValueError('Frozen scoring budgets must be positive integer bytes')
        self.limits, self.device, self.fused = limits, torch.device(device), bool(fused)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.budget = limits.budget(fused=self.fused)
        self.max_bytes, self.max_scratch_bytes = max_bytes, max_scratch_bytes
        self.disabled_reason = ('resident_budget' if self.budget['resident_bytes'] > max_bytes else
                                'scratch_budget' if self.budget['scratch_bytes'] > max_scratch_bytes else None)
        self.lock, self.fallback_lock = RLock(), RLock()
        self.counts = Counter()
        self.backends = Counter()
        self.storages = []
        self.error = None
        self._test_backend = _backend is not None
        self.backend = _backend if _backend is not None else _CUDA(self.device)
        self._scorer = _scorer
        self.stream = None
        if self.disabled_reason is None:
            if self.device.type != 'cuda' and not self._test_backend:
                self.disabled_reason = 'unsupported_device'
            else:
                # All capacity checks precede allocation. No resize/JIT-specific
                # scratch allocation or additional slot is ever added later.
                self.storages = [_Storage(limits, self.backend) for _ in range(self.SLOTS)]
                self.stream = self.backend.stream()

    def _reason(self, layers, owners):
        l = self.limits
        if not isinstance(layers, tuple) or not isinstance(owners, tuple):
            return 'unordered_inputs'
        if len(layers) != l.layers or not 0 < len(owners) <= l.max_owners:
            return 'owner_layer_capacity'
        if len({x.name for x in layers}) != len(layers) or len({x.request_id for x in owners}) != len(owners):
            return 'duplicate_identity'
        if {x.group for x in layers} != set(range(l.groups)):
            return 'group_geometry'
        for layer in layers:
            key, trig = layer.key, layer.cos_sin
            if (not isinstance(key, torch.Tensor) or key.device != self.device or key.ndim != 4 or
                    key.dtype != torch.float8_e4m3fn or key.shape[1] != l.page_tokens or
                    key.shape[3] != l.dim or key.stride(-1) != 1 or key.shape[2] <= 0 or
                    l.heads % key.shape[2] or not isinstance(trig, torch.Tensor) or
                    trig.device != self.device or trig.ndim != 2 or trig.shape[1] % 2 or
                    not 0 < trig.shape[1] <= l.dim or
                    layer.q_dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)):
                return 'layer_geometry'
            for scale in (layer.key_scale, layer.q_scale):
                if (not isinstance(scale, torch.Tensor) or scale.device != self.device or
                        scale.dtype != torch.float32 or scale.numel() != 1):
                    return 'scale_geometry'
        total_rows = 0
        for owner in owners:
            if not all(isinstance(x, tuple) for x in (owner.queries, owner.query_rows, owner.positions,
                                                       owner.starts, owner.counts, owner.tables)):
                return 'unordered_inputs'
            rows, segments = len(owner.query_rows), len(owner.starts)
            total_rows += rows
            if (not rows or total_rows > l.max_rows or not 0 < segments <= l.max_segments or
                    not _integer(owner.length) or not 0 < owner.length <= l.max_length or
                    len(owner.positions) != rows or len(owner.counts) != segments or
                    len(owner.queries) != l.layers or len(owner.tables) != l.groups):
                return 'owner_geometry'
            if (not all(_integer(x) and x >= 0 for x in owner.query_rows) or
                    not all(_integer(x) and 0 <= x < owner.length for x in owner.positions)):
                return 'row_geometry'
            end = 0
            for start, count in zip(owner.starts, owner.counts):
                if not _integer(start) or not _integer(count) or start != end or not 0 < count <= l.block_tokens:
                    return 'incomplete_partition'
                end += count
            if end != owner.length:
                return 'incomplete_partition'
            for layer, query in zip(layers, owner.queries):
                if (not isinstance(query, torch.Tensor) or query.device != self.device or
                        query.dtype != torch.bfloat16 or query.ndim != 3 or
                        tuple(query.shape[1:]) != (l.heads, l.dim) or max(owner.query_rows) >= query.shape[0] or
                        max(owner.positions) >= layer.cos_sin.shape[0]):
                    return 'query_geometry'
            pages = (owner.length + l.page_tokens - 1) // l.page_tokens
            for table in owner.tables:
                if (not isinstance(table, torch.Tensor) or table.device != self.device or
                        table.dtype != torch.int32 or table.ndim != 1 or table.numel() < pages):
                    return 'table_geometry'
            mass = owner.masses
            if (not isinstance(mass, torch.Tensor) or mass.device.type != 'cpu' or
                    mass.dtype != torch.float32 or tuple(mass.shape) != (rows, segments) or
                    not mass.is_contiguous() or (not self._test_backend and not mass.is_pinned())):
                return 'mass_geometry'
        return None

    @torch.inference_mode()
    def freeze(self, layers, owners, *, packet_lease):
        with self.lock:
            if self.error is not None:
                raise RuntimeError('Frozen score pool is quarantined') from self.error
            reason = self.disabled_reason or self._reason(layers, owners)
            if reason:
                self.counts['declined:' + reason] += 1
                return None
            storage = next((x for x in self.storages if x.active is None), None)
            if storage is None:
                self.counts['declined:full'] += 1
                return None
            storage.generation += 1
            frozen = []
            offset = 0
            for owner in owners:
                frozen.append(_FrozenOwner(owner.request_id, offset, len(owner.query_rows),
                                           len(owner.starts), int(owner.length), owner.masses))
                offset += len(owner.query_rows)
            handle = FrozenScoreInputs(storage, storage.generation, layers, tuple(frozen), packet_lease)
            storage.active = handle
            # Retain every source until the explicit release. K/cos_sin remain
            # externally immutable under packet ownership; the rest is copied.
            handle.sources = owners
            try:
                stream = self.backend.current()
                storage.host_flags.fill_(-1)
                for index, (owner, fixed) in enumerate(zip(owners, frozen)):
                    begin, end = fixed.begin, fixed.begin + fixed.rows
                    storage.host_rows[begin:end].numpy()[:] = owner.query_rows
                    storage.host_positions[begin:end].numpy()[:] = owner.positions
                    storage.host_starts[index, :fixed.segments].numpy()[:] = owner.starts
                    storage.host_counts[index, :fixed.segments].numpy()[:] = owner.counts
                storage.rows[:offset].copy_(storage.host_rows[:offset], non_blocking=True)
                storage.positions[:offset].copy_(storage.host_positions[:offset], non_blocking=True)
                for index, (owner, fixed) in enumerate(zip(owners, frozen)):
                    begin, end = fixed.begin, fixed.begin + fixed.rows
                    storage.starts[index, :fixed.segments].copy_(storage.host_starts[index, :fixed.segments], non_blocking=True)
                    storage.counts[index, :fixed.segments].copy_(storage.host_counts[index, :fixed.segments], non_blocking=True)
                    pages = (fixed.length + self.limits.page_tokens - 1) // self.limits.page_tokens
                    for group, table in enumerate(owner.tables):
                        storage.tables[index, group, :pages].copy_(table[:pages], non_blocking=True)
                        self.backend.track(table, stream)
                    for layer, query in enumerate(owner.queries):
                        torch.index_select(query, 0, storage.rows[begin:end], out=storage.q[layer, begin:end])
                        self.backend.track(query, stream)
                for index, layer in enumerate(layers):
                    storage.scales[index, 0].copy_(layer.key_scale.reshape(()), non_blocking=True)
                    storage.scales[index, 1].copy_(layer.q_scale.reshape(()), non_blocking=True)
                    self.backend.track(layer.key_scale, stream)
                    self.backend.track(layer.q_scale, stream)
                self.counts['frozen_frames'] += 1
                self.counts['frozen_rows'] += offset
                return handle
            except BaseException as error:
                self.quarantine(handle, error)
                raise

    def _check(self, handle):
        if (handle.storage not in self.storages or handle.storage.active is not handle or
                handle.generation != handle.storage.generation):
            raise RuntimeError('Stale or released frozen score handle')
        if handle.error is not None or self.error is not None:
            raise RuntimeError('Frozen score inputs are quarantined') from (handle.error or self.error)

    def seal(self, handle):
        with self.lock:
            self._check(handle)
            if handle.state != 'private':
                raise RuntimeError('Frozen score snapshot already sealed')
            # The caller has just recorded this event through detach_validation.
            # Never query an unrecorded event: CUDA considers it complete.
            handle.state = 'sealed'

    def quarantine(self, handle, error):
        with self.lock:
            if handle.error is None:
                handle.error = error
                handle.state = 'failed'
                self.counts['quarantined_frames'] += 1
            if self.error is None:
                self.error = error

    @torch.inference_mode()
    def fallback(self, handle):
        # Reserve all scratch across scoring AND the last output DMA; different
        # owners/slots cannot overlap or borrow the foreground download pool.
        with self.fallback_lock:
            with self.lock:
                self._check(handle)
                if handle.state != 'sealed':
                    raise RuntimeError('Frozen fallback requires one sealed generation')
                handle.state = 'fallback'
            storage = handle.storage
            deltas = Counter()
            try:
                scorer = self._scorer
                if scorer is None:
                    from .action_scoring import page_attention
                    scorer = page_attention
                with self.backend.context(self.stream):
                    self.stream.wait_event(handle.event)
                    for index, owner in enumerate(handle.owners):
                        begin, end = owner.begin, owner.begin + owner.rows
                        pages = (owner.length + self.limits.page_tokens - 1) // self.limits.page_tokens
                        total = None
                        for layer_index, layer in enumerate(handle.layers):
                            # Opaque paged K and immutable RoPE tables are leased
                            # by the packet, and recorded on this read stream.
                            self.backend.track(layer.key, self.stream)
                            self.backend.track(layer.cos_sin, self.stream)
                            mass = scorer(storage.q[layer_index, begin:end], storage.positions[begin:end],
                                layer.key, storage.tables[index, layer.group, :pages], layer.cos_sin,
                                storage.scales[layer_index, 0], storage.starts[index, :owner.segments],
                                storage.counts[index, :owner.segments], owner.length,
                                scale=layer.scale, q_dtype=layer.q_dtype,
                                q_scale=storage.scales[layer_index, 1], block_tokens=self.limits.block_tokens,
                                fused=self.fused, complete_partition=True)
                            deltas[getattr(mass, '_action_scoring_backend', 'reference')] += 1
                            total = mass if total is None else total + mass
                        total = total / len(handle.layers)
                        owner.masses.copy_(total, non_blocking=True)
                    storage.fallback_event.record(self.stream)
                storage.fallback_event.synchronize()
                with self.lock:
                    handle.state = 'fallback_complete'
                    handle.backend_counts = dict(deltas)
                    self.backends.update(deltas)
                    self.counts['fallback_frames'] += 1
                return dict(deltas)
            except BaseException as error:
                self.quarantine(handle, error)
                raise

    def release(self, handle):
        with self.lock:
            self._check(handle)
            if handle.state not in ('sealed', 'fallback_complete'):
                raise RuntimeError('Frozen snapshot is not ready for release')
            try:
                event = handle.storage.fallback_event if handle.state == 'fallback_complete' else handle.event
                if not event.query():
                    raise _NotReady('Frozen snapshot is still in use')
            except _NotReady:
                raise
            except BaseException as error:
                self.quarantine(handle, error)
                raise
            handle.state = 'released'
            handle.layers = handle.owners = handle.sources = ()
            handle.packet_lease = None
            handle.storage.active = None
            self.counts['released_frames'] += 1

    def snapshot(self):
        with self.lock:
            return dict(self.budget, capacity=self.SLOTS, allocated_slots=len(self.storages),
                        active_slots=sum(x.active is not None for x in self.storages),
                        max_bytes=self.max_bytes, max_scratch_bytes=self.max_scratch_bytes,
                        disabled_reason=self.disabled_reason, quarantined=self.error is not None,
                        counts=dict(self.counts), backend_counts=dict(self.backends))


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


class _NotReady(RuntimeError):
    pass
