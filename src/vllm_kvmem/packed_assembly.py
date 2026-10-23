"""Archive-layout assembly without per-record CPU tensor repacking.

Immutable records remain in their file layout:
[record][all raw-K layers][all FP8-V layers].
"""
import torch
import time
from dataclasses import dataclass
from vllm.triton_utils import tl, triton
from .fp8_compat import fp8_storage, fp8_to_float, float_to_fp8


@dataclass
class PackedRead:
    job: object
    leases: list
    owners: tuple
    expected_bytes: int
    destination: object
    admission_wait_seconds: float

    def result(self):
        """Join an admitted read before releasing its immutable-source leases."""
        try:
            data, stats = self.job.result()
            stats['admission_wait_seconds'] = self.admission_wait_seconds
            if len(data) != self.expected_bytes:
                raise ValueError('Archive returned an incomplete packed record batch')
            result = torch.frombuffer(data, dtype=torch.uint8)
            if self.destination is not None and result.data_ptr() != self.destination.data_ptr():
                raise ValueError('Archive did not fill the requested staging buffer')
            return result, stats
        finally:
            for lease in self.leases:
                lease.reset()
            self.leases.clear()
            self.owners = ()


def begin_read_packed_records(authority, handles, *, destination=None, region=None, wait=True):
    """Submit through the existing archive executor; optionally decline busy admission."""
    from .authority import record_key
    if not handles:
        raise ValueError("No immutable records requested")
    authority.poll()
    records = [authority.records[record_key(handle)] for handle in handles]
    if any(not record.archived or record.count != authority.shape[1] for record in records):
        raise ValueError("Packed assembly requires fully sealed, published records")
    admitted = [authority.memory.admitted_read(handle) for handle in handles]
    leases = [entry[0] for entry in admitted]
    transferred = False
    job = None
    try:
        if not all(leases):
            raise ValueError("Stale archive generation")
        kwargs = {}
        sources = [entry[1] for entry in admitted]
        if sources and all(sources):
            kwargs['_sources'] = sources
        if destination is not None:
            if (destination.device.type != 'cpu' or destination.dtype != torch.uint8 or
                    destination.ndim != 1 or not destination.is_contiguous()):
                raise ValueError("Packed read destination must be contiguous CPU bytes")
            kwargs['destination'] = destination.numpy()
        deadline = time.monotonic() + 180
        waited = 0.0
        while True:
            job = (authority.archive.read(authority.memory.request_id, leases, **kwargs) if region is None else
                   authority.archive.read_region(authority.memory.request_id, leases, *region, **kwargs))
            if job is not None:
                break
            if not wait:
                return None
            # Background seal/writeback can fill the shared queue during a
            # restore. Drain it with the condition lock released; retain the
            # exact generation leases and exclusive staging buffer meanwhile.
            start = time.monotonic()
            remaining = deadline - start
            if remaining <= 0 or not authority.archive.wait_for_idle(remaining):
                raise TimeoutError('Archive read admission did not drain')
            waited += time.monotonic() - start
        size = authority.record_bytes if region is None else region[1]
        result = PackedRead(job, leases, (authority, *sources), len(handles)*size, destination, waited)
        transferred = True
        return result
    except BaseException as error:
        # Even failure constructing the completion holder must not return a
        # reserved pinned slot while an already admitted writer still uses it.
        if job is not None:
            try:
                job.result()
            except BaseException as cleanup:
                error.add_note('Archive read also failed while draining: '+repr(cleanup))
        raise
    finally:
        if not transferred:
            for lease in leases:
                lease.reset()


def read_packed_records(authority, handles, *, destination=None, region=None):
    """One bounded read; optionally fill an exclusively owned CPU staging tensor."""
    return begin_read_packed_records(authority, handles, destination=destination, region=region).result()


class PackedReadAhead:
    """One lookahead within one immutable assembly plan, using existing slots."""
    def __init__(self, authority, stats):
        self.authority = authority
        self.plan = authority.memory._plan
        self.namespace = authority.memory.request_id
        self.stats = stats
        self.pending = None

    def _key(self, handles, region):
        if (self.authority.memory._plan is not self.plan or
                self.authority.memory.request_id != self.namespace):
            raise ValueError('Prefetched read belongs to a different assembly plan or owner')
        return tuple((int(h.slot), int(h.generation)) for h in handles), region

    def start(self, transfers, handles, region, size):
        if self.pending is not None:
            raise RuntimeError('Assembly permits only one prefetched read')
        key = self._key(handles, region)
        attempted = False
        def begin(buffer):
            nonlocal attempted
            attempted = True
            return begin_read_packed_records(self.authority, handles, destination=buffer,
                                             region=region, wait=False)
        ticket = transfers.prepare_from_reader(size, begin)
        if ticket is None:
            name = 'admission_busy' if attempted else 'slot_busy'
            self.stats['assembly_prefetch_'+name] += 1
            return
        self.pending = key, transfers, ticket
        self.stats['assembly_prefetch_submitted'] += 1

    def take(self, transfers, handles, region, destination, ready):
        if self.pending is None:
            return None
        key, pool, ticket = self.pending
        if pool is not transfers or key != self._key(handles, region):
            raise ValueError('Prefetched read does not match the next immutable chunk')
        try:
            transfer = pool.upload_prepared(ticket, destination, ready=ready)
        except BaseException:
            self.stats['assembly_prefetch_abandoned'] += 1
            raise
        else:
            self.stats['assembly_prefetch_completed'] += 1
            return transfer, ticket.filled[1]
        finally:
            self.pending = None

    def close(self):
        if self.pending is not None:
            _, pool, ticket = self.pending
            self.pending = None
            try:
                # ArchiveJob.cancel() intentionally cannot cancel an admitted
                # writer. Its pinned storage remains exclusive until it drains.
                pool.discard_prepared(ticket)
            finally:
                self.stats['assembly_prefetch_abandoned'] += 1

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if error is None:
                raise
            error.add_note('Prefetched read also failed while draining: '+repr(cleanup))
        return False


@triton.jit
def _assemble_packed(payload, bases, page_table, trig, kcache, vcache, scale,
                     k0: tl.constexpr, k1: tl.constexpr, k2: tl.constexpr,
                     v0: tl.constexpr, v1: tl.constexpr, v2: tl.constexpr,
                     TRIG_STRIDE: tl.constexpr, RECORD_TOKENS: tl.constexpr,
                     LAYERS: tl.constexpr, LAYER: tl.constexpr, HEADS: tl.constexpr,
                     DIM: tl.constexpr, HALF: tl.constexpr, PAGE: tl.constexpr,
                     BLOCK: tl.constexpr, RAW_TYPE: tl.constexpr,
                     RAW_BYTES: tl.constexpr, ROPE_TYPE: tl.constexpr,
                     SOFTWARE_FP8: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    record, offset = token // RECORD_TOKENS, token % RECORD_TOKENS
    position = tl.load(bases + record).to(tl.int64) + offset
    page = tl.load(page_table + position // PAGE).to(tl.int64)
    cache_offset = position % PAGE
    layer_elements: tl.constexpr = RECORD_TOKENS * HEADS * DIM
    raw_elements: tl.constexpr = LAYERS * layer_elements
    d = tl.arange(0, BLOCK)
    mask = d < DIM
    raw = payload.to(tl.pointer_type(RAW_TYPE))
    raw_base = record * (raw_elements * (RAW_BYTES + 1) // RAW_BYTES) + LAYER * layer_elements + offset * HEADS * DIM + head * DIM
    x = fp8_to_float(tl.load(raw + raw_base + d, mask=mask, other=0.0), SOFTWARE_FP8 and RAW_BYTES == 1)
    other_d = tl.where(d < HALF, d + HALF, d - HALF)
    mate = fp8_to_float(tl.load(raw + raw_base + other_d, mask=d < 2*HALF, other=0.0), SOFTWARE_FP8 and RAW_BYTES == 1)
    freq = d % HALF
    cos = tl.load(trig + position * TRIG_STRIDE + freq).to(tl.float32)
    sin = tl.load(trig + position * TRIG_STRIDE + HALF + freq).to(tl.float32)
    rotated = tl.where(d < HALF, x*cos - mate*sin, x*cos + mate*sin)
    rotated = tl.where(d < 2*HALF, rotated, x).to(ROPE_TYPE).to(tl.float32)
    # FP8 source reconstruction is byte-qualified against the CPU oracle.
    # Approximate reciprocal division can cross an FP8 midpoint at real model
    # scales such as 0.04052734375 even when the rotation is identical.
    if RAW_BYTES == 1:
        scaled = tl.div_rn(rotated, tl.load(scale))
    else:
        scaled = rotated / tl.load(scale)
    quantized = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    if SOFTWARE_FP8:
        quantized = float_to_fp8(quantized)
    tl.store(kcache + page*k0 + cache_offset*k1 + head*k2 + d, quantized, mask=mask)
    value_base = record * raw_elements * (RAW_BYTES + 1) + raw_elements * RAW_BYTES + LAYER * layer_elements + offset * HEADS * DIM + head * DIM
    value = tl.load(payload + value_base + d, mask=mask, other=0)
    dst = vcache.to(tl.pointer_type(tl.uint8))
    tl.store(dst + page*v0 + cache_offset*v1 + head*v2 + d, value, mask=mask)


def assemble_packed_layer(payload, bases, layer, *, layer_index, layers, block_tokens, raw_dtype):
    """Caller validates destination page ownership, uniqueness and position bounds."""
    key, value = layer.key, layer.value
    if key.ndim != 4 or value.shape != key.shape or key.dtype != torch.float8_e4m3fn or value.dtype != key.dtype:
        raise ValueError("Packed assembly needs matching native FP8 cache views")
    heads, dim = key.shape[-2:]
    if (min(layers, block_tokens) <= 0 or (layers * block_tokens * heads * dim) % 2 or
            not 0 <= layer_index < layers or
            raw_dtype not in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)):
        raise ValueError("Invalid packed source geometry")
    if (bases.ndim != 1 or bases.dtype not in (torch.int32, torch.int64) or not bases.is_contiguous() or
            payload.ndim != 1 or payload.dtype != torch.uint8 or not payload.is_contiguous() or
            payload.numel() != bases.numel() * layers * block_tokens * heads * dim *
                (raw_dtype.itemsize+1)):
        raise ValueError("Packed record extent does not match destination bases")
    tensors = (payload, bases, key, value, layer.kernel_page_table, layer.cos_sin, layer.key_scale)
    if payload.device.type != 'cuda' or any(t.device != payload.device for t in tensors):
        raise ValueError("Packed assembly requires one CUDA device")
    if (layer.cos_sin.ndim != 2 or not 0 < layer.cos_sin.shape[1] <= dim or
            layer.cos_sin.shape[1] % 2 or layer.key_scale.numel() != 1 or
            any(t.ndim and t.stride(-1) != 1 for t in tensors)):
        raise ValueError("Invalid native scale, rotary geometry or inner stride")
    if not bases.numel():
        return
    key_input,software = fp8_storage(key)
    value_input,_ = fp8_storage(value)
    _assemble_packed[(bases.numel() * block_tokens, heads)](
        payload, bases, layer.kernel_page_table, layer.cos_sin, key_input, value_input, layer.key_scale,
        *key.stride()[:3], *value.stride()[:3], layer.cos_sin.stride(0), block_tokens,
        layers, layer_index, heads, dim, layer.cos_sin.shape[1] // 2, key.shape[1],
        triton.next_power_of_2(dim),
        ((tl.uint8 if software else tl.float8e4nv) if raw_dtype == torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype == torch.bfloat16 else tl.float16),
        raw_dtype.itemsize,
        (tl.float32 if raw_dtype == torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype == torch.bfloat16 else tl.float16),
        software,
        num_warps=4, enable_fp_fusion=raw_dtype != torch.float8_e4m3fn)
