"""Exact, bounded CPU snapshots of native request cache rows.

The request retains its runner slot, sampler state and authority throughout a
swap. These snapshots copy the current attention/recurrent bytes without raw-K
reconstruction or accepted-state canonicalization. The caller owns quiescence,
page leases and a complete multi-stream fence until capture has completed.
"""
from dataclasses import dataclass
import hashlib

import torch


@dataclass(frozen=True)
class CacheRows:
    name: str
    tensor: torch.Tensor
    rows: tuple[int, ...]

    @property
    def row_shape(self):
        return tuple(self.tensor.shape[1:])

    @property
    def row_bytes(self):
        return self.tensor[0].numel() * self.tensor.element_size()


@dataclass(frozen=True)
class TensorSchema:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    rows: int
    offset: int
    row_bytes: int


class CopyCompletion:
    def __init__(self, event=None, owners=()):
        self.event = event
        self.owners = owners

    def query(self):
        if self.event is not None and not self.event.query():
            return False
        self.owners = ()
        return True

    def synchronize(self):
        if self.event is not None:
            self.event.synchronize()
        self.query()


@dataclass
class ExactRequestCheckpoint:
    identity: tuple[str, int, int]
    schemas: tuple[TensorSchema, ...]
    payload: torch.Tensor
    completion: CopyCompletion

    @property
    def bytes(self):
        return self.payload.numel()

    def query(self):
        return self.completion.query()


def _identity(identity):
    if (not isinstance(identity, tuple) or len(identity) != 3 or
            not isinstance(identity[0], str) or not identity[0] or
            any(type(value) is not int or value <= 0 for value in identity[1:])):
        raise ValueError('Checkpoint requires request, generation and operation identity')


def _row(tensor, row):
    view = tensor[row]
    return view.reshape(1) if view.ndim == 0 else view


def _bindings(bindings):
    bindings = tuple(bindings)
    if not bindings or len({b.name for b in bindings}) != len(bindings):
        raise ValueError('Checkpoint needs unique named cache bindings')
    devices = set()
    seen = set()
    for binding in bindings:
        tensor = binding.tensor
        if (not isinstance(binding.name, str) or not binding.name or
                not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or
                tensor.layout != torch.strided or tensor.requires_grad or
                not tensor.shape[0] or not binding.rows or
                any(type(row) is not int or not 0 <= row < tensor.shape[0] for row in binding.rows) or
                len(set(binding.rows)) != len(binding.rows) or
                any(stride <= 0 for size, stride in zip(tensor.shape, tensor.stride()) if size > 1)):
            raise ValueError('Invalid native cache rows')
        for row in binding.rows:
            view = _row(tensor, row)
            if view.numel() == 0 or view.stride(-1) != 1 or torch._debug_has_internal_overlap(view) == 1:
                raise ValueError('Cache rows require a nonempty contiguous element axis')
            key = (view.data_ptr(), view.dtype, view.shape, view.stride())
            if key in seen:
                raise ValueError('Aliased native cache row binding')
            seen.add(key)
        devices.add(tensor.device)
    if len(devices) != 1 or next(iter(devices)).type not in ('cpu', 'cuda'):
        raise ValueError('Checkpoint rows must occupy one CPU or CUDA device')
    return bindings, next(iter(devices))


def checkpoint_size(bindings):
    bindings, _ = _bindings(bindings)
    return sum(len(binding.rows) * binding.row_bytes for binding in bindings)


def capture_exact(identity, bindings, *, max_bytes):
    """Capture admitted rows; the source pages stay leased until ``query``.

    Allocates exactly one owned CPU payload. Copy each noncontiguous row through
    a byte view, retaining borrowed tensor owners through the returned event.
    This implementation makes no explicit full-size GPU snapshot allocation.
    PyTorch may allocate internal staging for noncontiguous device/host copies.
    """
    _identity(identity)
    bindings, device = _bindings(bindings)
    size = sum(len(binding.rows) * binding.row_bytes for binding in bindings)
    if type(max_bytes) is not int or max_bytes <= 0 or size > max_bytes:
        raise BufferError('Exact request checkpoint exceeds its byte reservation')
    payload = torch.empty(size, dtype=torch.uint8, device='cpu', pin_memory=device.type == 'cuda')
    schemas, offset = [], 0
    event = torch.cuda.Event() if device.type == 'cuda' else None
    try:
        for binding in bindings:
            schemas.append(TensorSchema(binding.name, binding.tensor.dtype, binding.row_shape,
                                        len(binding.rows), offset, binding.row_bytes))
            for row in binding.rows:
                source = _row(binding.tensor, row).view(torch.uint8)
                destination = payload[offset:offset + binding.row_bytes].view(source.shape)
                destination.copy_(source, non_blocking=device.type == 'cuda')
                offset += binding.row_bytes
        if event is not None:
            event.record(torch.cuda.current_stream(device))
    except BaseException:
        if device.type == 'cuda':
            # Submission failure cannot release a source/destination still in
            # use by an earlier accepted copy on this stream.
            torch.cuda.current_stream(device).synchronize()
        raise
    return ExactRequestCheckpoint(identity, tuple(schemas), payload,
                                  CopyCompletion(event, (bindings, payload)))


def _validate_exact_restore(snapshot, identity, bindings):
    _identity(identity)
    if not isinstance(snapshot, ExactRequestCheckpoint) or snapshot.identity != identity:
        raise ValueError('Stale or foreign exact request checkpoint')
    if not snapshot.query():
        raise RuntimeError('Exact request checkpoint is not ready')
    bindings, device = _bindings(bindings)
    if len(bindings) != len(snapshot.schemas):
        raise ValueError('Exact restore omitted native cache bindings')
    offset = 0
    for binding, schema in zip(bindings, snapshot.schemas):
        if (binding.name != schema.name or binding.tensor.dtype != schema.dtype or
                binding.row_shape != schema.shape or len(binding.rows) != schema.rows or
                binding.row_bytes != schema.row_bytes or schema.offset != offset):
            raise ValueError('Exact restore schema differs from checkpoint')
        offset += schema.rows * schema.row_bytes
    if (snapshot.payload.device.type != 'cpu' or snapshot.payload.dtype != torch.uint8 or
            snapshot.payload.ndim != 1 or not snapshot.payload.is_contiguous() or
            snapshot.bytes != offset or (device.type == 'cuda' and not snapshot.payload.is_pinned())):
        raise ValueError('Exact checkpoint payload extent or pinning changed')
    return bindings, device


def restore_exact(snapshot, identity, bindings):
    """Validate every destination before enqueuing the first byte write."""
    bindings, device = _validate_exact_restore(snapshot, identity, bindings)
    event = torch.cuda.Event() if device.type == 'cuda' else None
    try:
        for binding, schema in zip(bindings, snapshot.schemas):
            offset = schema.offset
            for row in binding.rows:
                destination = _row(binding.tensor, row).view(torch.uint8)
                source = snapshot.payload[offset:offset + schema.row_bytes].view(destination.shape)
                destination.copy_(source, non_blocking=device.type == 'cuda')
                offset += schema.row_bytes
        if event is not None:
            event.record(torch.cuda.current_stream(device))
    except BaseException:
        if device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
        raise
    return CopyCompletion(event, (snapshot, bindings))


def _bounded_byte_views(view, maximum):
    """Yield logical-order rectangular views, without flattening GPU storage."""
    if view.numel() <= maximum:
        yield view
        return
    inner = view[0].numel() if view.ndim > 1 else 1
    if inner > maximum:
        for index in range(view.shape[0]):
            yield from _bounded_byte_views(view[index], maximum)
    else:
        count = max(1, maximum//inner)
        for start in range(0, view.shape[0], count):
            yield view[start:start+count]


def verify_exact(snapshot, identity, bindings, *, chunk_bytes=1024**2):
    """Diagnostic byte readback after the caller's H2D completion fence.

    Adds one bounded CPU slab (pinned for CUDA), outside the checkpoint payload
    quota. Every GPU source remains a strided view. PyTorch's internal staging
    for a noncontiguous copy is bounded by that chunk, never the full snapshot.
    This deliberately blocking verification is opt-in, not a performance path.
    """
    if type(chunk_bytes) is not int or not 0 < chunk_bytes <= 4*1024**2:
        raise ValueError('Exact verification chunk must be in [1, 4 MiB]')
    bindings, device = _validate_exact_restore(snapshot, identity, bindings)
    scratch = torch.empty(min(chunk_bytes, snapshot.bytes), dtype=torch.uint8,
                          device='cpu', pin_memory=device.type == 'cuda')
    digest, offset, chunks = hashlib.sha256(), 0, 0
    for binding in bindings:
        for row in binding.rows:
            raw = _row(binding.tensor, row).view(torch.uint8)
            for source in _bounded_byte_views(raw, scratch.numel()):
                count = source.numel()
                actual = scratch[:count]
                expected = snapshot.payload[offset:offset+count]
                # Blocking D2H: a slab is never read/overwritten before its DMA
                # completes, including when the source has nontrivial strides.
                try:
                    actual.view(source.shape).copy_(source, non_blocking=False)
                except BaseException:
                    if device.type == 'cuda':
                        torch.cuda.current_stream(device).synchronize()
                    raise
                view = memoryview(actual.numpy()).cast('B')
                reference = None
                try:
                    if not torch.equal(actual, expected):
                        reference = memoryview(expected.numpy()).cast('B')
                        first = next(i for i, (a, b) in enumerate(zip(view, reference)) if a != b)
                        raise RuntimeError('Exact restore verification mismatch: '
                            f'identity={identity!r}, binding={binding.name!r}, '
                            f'row={row}, payload_byte={offset+first}')
                    digest.update(view)
                finally:
                    view.release()
                    if reference is not None:
                        reference.release()
                offset += count
                chunks += 1
    if offset != snapshot.bytes:
        raise RuntimeError('Exact verification did not cover the complete checkpoint')
    return dict(identity=identity, bytes=offset, sha256=digest.hexdigest(),
        schemas=[dict(name=s.name, dtype=str(s.dtype), shape=s.shape, rows=s.rows,
                      offset=s.offset, row_bytes=s.row_bytes) for s in snapshot.schemas],
        chunks=chunks, readback_peak_bytes=scratch.numel(), chunk_bytes=chunk_bytes)


class NativeRequestCheckpointAdapter:
    """Bind exact target/MTP/GDN rows while keeping native slot state live.

    The caller supplies scheduler-authorized allocator page IDs and keeps the
    worker request index/generation unchanged. This adapter deliberately does
    not serialize a sampler or remove/recreate a runner request.
    """

    def __init__(self, runner):
        from .native_state import NativeRecurrentState
        config = runner.vllm_config
        if (config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none' or
                getattr(config, 'max_concurrent_batches', 1) != 1 or
                any(getattr(config.parallel_config, name, 1) != 1 for name in
                    ('tensor_parallel_size', 'pipeline_parallel_size',
                     'decode_context_parallel_size', 'prefill_context_parallel_size')) or
                (runner.speculative_config is not None and runner.speculative_config.method != 'mtp') or
                getattr(runner, '_kvmem_qsa_layers', None)):
            raise ValueError('Exact swaps require single-rank target/MTP APC-off native state')
        self.runner = runner
        self.recurrent = NativeRecurrentState(runner)
        self._slots = {}

    def bindings(self, request_id, block_ids, *, capture=False):
        from .codec import native_vllm_fp8_views
        from .native_pages import kernel_expansions, kernel_table
        runner = self.runner
        index = runner.req_states.req_id_to_index[request_id]
        generation = runner._kvmem_request_generations[request_id]
        identity = index, generation
        if capture:
            if request_id in self._slots:
                raise ValueError('Duplicate native request checkpoint owner')
        elif self._slots.get(request_id) != identity:
            raise ValueError('Paused request index or generation was recycled')
        groups = runner.kv_cache_config.kv_cache_groups
        expansion = kernel_expansions(runner)
        if len(block_ids) != len(groups) or len(expansion) != len(groups):
            raise ValueError('Native checkpoint omitted cache groups')
        modules = dict(runner.model.named_modules())
        modules.update(getattr(runner, '_kvmem_mtp_modules', {}))
        names = getattr(runner, '_kvmem_archive_layers', runner._kvmem_layers)
        by_native_name = {modules[name].attn.layer_name: modules[name] for name in names}
        result = []
        allocated = []
        for group_id, (group, pages) in enumerate(zip(groups, block_ids)):
            pages = tuple(pages)
            if (not pages or any(type(page) is not int or page <= 0 for page in pages) or
                    len(set(pages)) != len(pages)):
                raise ValueError('Native checkpoint requires exact non-null allocator pages')
            allocated.extend(pages)
            rows = tuple(page * expansion[group_id] + sub for page in pages
                         for sub in range(expansion[group_id]))
            if group_id in self.recurrent.groups:
                _, actual = self.recurrent._rows(request_id, group_id)
                if tuple(actual) != rows:
                    raise ValueError('Native recurrent table differs from swap allocation')
                for layer_name, tensors in self.recurrent.groups[group_id].items():
                    for kind, tensor in zip(('conv', 'ssm'), tensors):
                        result.append(CacheRows(f'{group_id}/{layer_name}/{kind}', tensor, rows))
            else:
                actual = tuple(kernel_table(runner, group_id, index).cpu().tolist())
                if actual != rows:
                    raise ValueError('Native attention table differs from swap allocation')
                for layer_name in group.layer_names:
                    module = by_native_name.get(layer_name)
                    if module is None or module.attn.kv_cache_dtype != 'fp8_e4m3':
                        raise ValueError('Native swap encountered an unqualified attention layer')
                    for kind, tensor in zip(('key', 'value'), native_vllm_fp8_views(
                            module.attn.kv_cache, module.num_kv_heads, module.head_dim)):
                        result.append(CacheRows(f'{group_id}/{layer_name}/{kind}', tensor, rows))
        if len(set(allocated)) != len(allocated):
            raise ValueError('Native groups alias allocator pages')
        # Preserve every speculative row and the exact native counter. Unlike
        # PromptStateAdapter, no accepted-row canonicalization is performed.
        result.append(CacheRows('num_accepted', runner.model_state.num_accepted_tokens_gpu, (index,)))
        _bindings(result)
        if capture:
            self._slots[request_id] = identity
        return tuple(result)

    def capture(self, request_id, operation_id, block_ids, *, max_bytes):
        bindings = self.bindings(request_id, block_ids, capture=True)
        identity = request_id, self.runner._kvmem_request_generations[request_id], operation_id
        try:
            return capture_exact(identity, bindings, max_bytes=max_bytes)
        except BaseException:
            self._slots.pop(request_id, None)
            raise

    def restore(self, snapshot, request_id, operation_id, block_ids):
        identity = request_id, self.runner._kvmem_request_generations[request_id], operation_id
        return restore_exact(snapshot, identity, self.bindings(request_id, block_ids))

    def verify_restore(self, snapshot, request_id, operation_id, block_ids):
        identity = request_id, self.runner._kvmem_request_generations[request_id], operation_id
        return verify_exact(snapshot, identity, self.bindings(request_id, block_ids))

    def release(self, request_id):
        """Call only after restore completion, or fenced cancellation cleanup."""
        del self._slots[request_id]
