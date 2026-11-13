"""Opt-in KVMEM_LAYERWISE_PRESSURE_GPU_REUSE: reuse resident K across pressure refreshes.

A prefill pressure refresh keeps a subset of each layer's current window, in
history order, and compacts it to dense positions. The original path re-reads
every kept block's raw K from the CPU archive and re-applies RoPE at the new
position. Here a kept block that moved is rebuilt on the GPU instead:

- V and the position-free K dims (beyond the rotary prefix) are byte copies;
- the rotary K dims of the stored, already scaled FP8 values are rotated by the
  position change (-delta) and re-quantized. This is not byte-exact: each reuse
  adds about one FP8 rounding. A per-block counter forces a CPU reload (the
  original raw-K path, byte-identical to it) once a block has been reused
  KVMEM_LAYERWISE_PRESSURE_REUSE_LIMIT times (default 32);
- blocks that did not move are left untouched (always a prefix: the shift
  delta is non-decreasing in history order).

All moved V rows, then all reused K rows, are snapshotted in the existing
one-window resident-V buffer before any of them is written, so overlapping
in-place moves need no ordering. Requires the resident-V reuse sources
(KVMEM_LAYERWISE_PRESSURE_RESIDENT_VALUES) and compact positions.
"""
import os
from ..core import settings as _settings

REUSE, RELOAD = 0, 1


def enabled():
    value = _settings.get('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_GPU_REUSE must be 0 or 1')
    # Byte-verification runs (KVMEM_VERIFY_SESSION_RESTORE) keep the exact path.
    return value == '1' and _settings.get('KVMEM_VERIFY_SESSION_RESTORE') != '1'


def reuse_limit():
    value = _settings.get('KVMEM_LAYERWISE_PRESSURE_REUSE_LIMIT', '32')
    if not value.isdigit() or not 0 <= int(value) <= 1 << 20:
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_REUSE_LIMIT must be a nonnegative integer')
    return int(value)


def restored(controller, request_id, *, resident):
    """Resident pages keep their reuse history (unknown here); copies are exact."""
    if not enabled():
        return
    if resident:
        mark_untrusted(controller, request_id)
    else:
        forget(controller, request_id)


def cancelled(runner, job):
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is not None and enabled() and getattr(job.command, 'selection', None) == 'pressure':
        mark_untrusted(controller, job.command.request_id)


def record_stats(controller, **amounts):
    """Cumulative reuse counters; one summary event per 1,000 layer assemblies."""
    stats = controller.__dict__.setdefault('reuse_stats', dict(layers=0, reused=0, reloaded=0, skipped=0,
                                                                uploaded=0, fallback=0))
    stats['layers'] += 1
    for key, value in amounts.items():
        stats[key] += value
    if stats['layers'] % 1000 == 0:
        emit = getattr(getattr(controller, 'query', None), 'emit', None)
        if callable(emit):
            emit('gpu_reuse_summary', **stats)


def plan_layer(chosen, sources, counts, limit, *, untrusted=False):
    """Classify one layer's kept blocks.

    chosen: [(history_begin, tokens)] in history order; sources: their old dense
    bases. Returns (moves, new_counts, skipped): moves are
    (index, begin, source, destination, delta, mode) for blocks that moved.
    `untrusted` (pages restored or of unknown history) forces a reload of every
    block that moves.
    """
    if len(chosen) != len(sources):
        raise ValueError('Reuse plan needs one source per kept block')
    moves, new_counts = [], {}
    cursor = previous = skipped = 0
    for index, ((begin, tokens), source) in enumerate(zip(chosen, sources)):
        destination, cursor = cursor, cursor + tokens
        delta = source - destination
        if delta < 0 or delta < previous:
            raise ValueError('Pressure compaction must shift kept blocks left, non-decreasingly')
        previous = delta
        count = limit if untrusted else counts.get(begin, 0)
        if delta == 0:
            new_counts[begin] = count
            skipped += 1
            continue
        mode = REUSE if count < limit else RELOAD
        new_counts[begin] = count + 1 if mode == REUSE else 0
        moves.append((index, begin, source, destination, delta, mode))
    return moves, new_counts, skipped


def counters(controller, request_id, layer):
    """Per (request, layer) reuse counts and whether that layer's pages are untrusted.

    An untrusted owner (resident restore, cancelled refresh) reloads every moving
    block of a layer once; that layer's counts are then exact again.
    """
    table = controller.__dict__.setdefault('reuse_counts', {})
    refreshed = controller.__dict__.setdefault('reuse_untrusted', {}).get(request_id)
    return (table.setdefault(request_id, {}).setdefault(layer, {}),
            refreshed is not None and layer not in refreshed)


def mark_untrusted(controller, request_id):
    controller.__dict__.setdefault('reuse_counts', {}).pop(request_id, None)
    controller.__dict__.setdefault('reuse_untrusted', {})[request_id] = set()


def forget(controller, request_id):
    controller.__dict__.get('reuse_counts', {}).pop(request_id, None)
    controller.__dict__.get('reuse_untrusted', {}).pop(request_id, None)


def commit_counts(controller, request_id, layer, new_counts):
    controller.__dict__.setdefault('reuse_counts', {}).setdefault(request_id, {})[layer] = new_counts
    refreshed = controller.__dict__.setdefault('reuse_untrusted', {}).get(request_id)
    if refreshed is not None:
        refreshed.add(layer)


try:
    from vllm.triton_utils import tl, triton
    from ..adapters.fp8_compat import fp8_to_float, float_to_fp8

    @triton.jit
    def _scatter_rows_tiled(source, bases, pages, output,
                            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
                            PAGE: tl.constexpr, TOKENS: tl.constexpr, HEADS: tl.constexpr,
                            DIM: tl.constexpr, BLOCK: tl.constexpr):
        # Inverse of split_assembly._gather_v_tiled: record-major rows -> pages.
        record = tl.program_id(0).to(tl.int64)
        x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = x < TOKENS * HEADS * DIM
        token = x // (HEADS * DIM)
        head = (x // DIM) % HEADS
        d = x % DIM
        position = tl.load(bases + record).to(tl.int64) + token
        page = tl.load(pages + position // PAGE, mask=valid, other=0).to(tl.int64)
        value = tl.load(source + record * TOKENS * HEADS * DIM + x, mask=valid, other=0)
        tl.store(output + page * S0 + (position % PAGE) * S1 + head * S2 + d, value, mask=valid)

    @triton.jit
    def _reuse_keys(payload, k_indices, modes, deltas, bases, pages, trig, key, scale,
                    K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
                    ROW: tl.constexpr, TRIG: tl.constexpr, TOKENS: tl.constexpr,
                    HEADS: tl.constexpr, DIM: tl.constexpr, HALF: tl.constexpr,
                    PAGE: tl.constexpr, BLOCK: tl.constexpr,
                    SOFTWARE_FP8: tl.constexpr, SOFTWARE_WRITE: tl.constexpr):
        token, head = tl.program_id(0), tl.program_id(1)
        record, offset = token // TOKENS, token % TOKENS
        position = tl.load(bases + record).to(tl.int64) + offset
        page = tl.load(pages + position // PAGE).to(tl.int64)
        kidx = tl.load(k_indices + record).to(tl.int64)
        mode = tl.load(modes + record)
        delta = tl.load(deltas + record).to(tl.int64)
        inner = offset * HEADS * DIM + head * DIM
        d = tl.arange(0, BLOCK)
        packed = payload.to(tl.pointer_type(tl.uint8 if SOFTWARE_FP8 else tl.float8e4nv))
        source = kidx * ROW + inner
        x = fp8_to_float(tl.load(packed + source + d, mask=d < DIM, other=0.0), SOFTWARE_FP8)
        mate_d = tl.where(d < HALF, d + HALF, d - HALF)
        mate = fp8_to_float(tl.load(packed + source + mate_d, mask=d < 2 * HALF, other=0.0), SOFTWARE_FP8)
        freq = d % HALF
        reload = mode == 1
        # Reload: raw K rotated to its position (the original split kernel).
        # Reuse: stored scaled K rotated by -delta (position moved left by delta).
        angle = tl.where(reload, position, delta)
        cos = tl.load(trig + angle * TRIG + freq).to(tl.float32)
        sin = tl.load(trig + angle * TRIG + HALF + freq).to(tl.float32)
        sign = tl.where(reload, -1.0, 1.0)
        rotated = tl.where(d < HALF, x * cos + sign * (mate * sin), x * cos - sign * (mate * sin))
        rotated = tl.where(d < 2 * HALF, rotated, x)
        scaled = tl.where(reload, tl.div_rn(rotated, tl.load(scale)), rotated)
        quantized = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        if SOFTWARE_WRITE:
            quantized = float_to_fp8(quantized)
        tl.store(key + page * K0 + (position % PAGE) * K1 + head * K2 + d, quantized, mask=d < DIM)
except ImportError:          # CPU-only environments (unit tests of the plan)
    _reuse_keys = None


def gather_rows(cache, bases, pages, output, block_tokens=32):
    """Byte-copy whole blocks (all heads) at old dense bases into record-major rows."""
    import torch
    from .split_assembly import _gather_v_tiled
    heads, dim = cache.shape[-2:]
    _gather_v_tiled[(bases.numel(), triton.cdiv(block_tokens * heads * dim, 4096))](
        cache.view(torch.uint8), bases, pages, output, *cache.stride()[:3], cache.shape[1],
        block_tokens, 1, 0, heads, dim, 4096, num_warps=4)


def scatter_rows(source, bases, pages, cache, block_tokens=32):
    """Byte-copy record-major rows to whole blocks at new dense bases."""
    import torch
    heads, dim = cache.shape[-2:]
    _scatter_rows_tiled[(bases.numel(), triton.cdiv(block_tokens * heads * dim, 4096))](
        source, bases, pages, cache.view(torch.uint8), *cache.stride()[:3], cache.shape[1],
        block_tokens, heads, dim, 4096, num_warps=4)


def write_keys(payload, k_indices, modes, deltas, bases, layer, *, rows, block_tokens=32):
    from ..adapters.fp8_compat import fp8_storage
    key = layer.key
    heads, dim = key.shape[-2:]
    key_input, software_write = fp8_storage(key, for_write=True)
    _, software_read = fp8_storage(key)
    _reuse_keys[(bases.numel() * block_tokens, heads)](
        payload, k_indices, modes, deltas, bases, layer.kernel_page_table, layer.cos_sin,
        key_input, layer.key_scale, *key.stride()[:3], rows, layer.cos_sin.stride(0),
        block_tokens, heads, dim, layer.cos_sin.shape[1] // 2, key.shape[1],
        triton.next_power_of_2(dim), software_read, software_write, num_warps=4,
        enable_fp_fusion=False)


def _native(name):
    try:
        from .. import _native as native
    except ImportError:       # CPU-only test environments without the extension
        return None
    return getattr(native, name, None)


def layer_plan(chosen, sources, counts, limit, untrusted):
    """plan_layer over chosen records plus the metadata rows assemble_reused uploads.

    Returns (moved, reused, reloads [(index, destination)], new_counts, skipped,
    rows int64 [V sources, V destinations, K sources, K destinations, K deltas,
    K payload indices, K modes]). Native when the extension provides it; the
    Python reference otherwise (same results).
    """
    import numpy as np
    native = _native('reuse_layer_plan')
    if native is not None:
        rows, m, r, reload, new_counts, skipped = native(chosen, sources, counts, limit, untrusted)
        return m, r, reload, new_counts, skipped, rows
    moves, new_counts, skipped = plan_layer([(c.history_begin, c.count) for c in chosen], sources,
                                            counts, limit, untrusted=untrusted)
    reuse = [x for x in moves if x[5] == REUSE]
    reload = [(x[0], x[3]) for x in moves if x[5] != REUSE]
    rows = np.array([x[2] for x in moves] + [x[3] for x in moves] +
                    [x[2] for x in reuse] + [x[3] for x in reuse] + [x[4] for x in reuse] +
                    list(range(len(reuse))) + [REUSE] * len(reuse), dtype=np.int64)
    return len(moves), len(reuse), reload, new_counts, skipped, rows


def upload_rows(assembler, rows):
    """NativeWindowAssembler._upload_metadata for an int64 array: pinned, fenced owners."""
    import torch
    host = torch.from_numpy(rows).pin_memory()
    assembler._metadata_owners.append(host)
    device = host.to(assembler.device, non_blocking=True)
    assembler._metadata_owners.append(device)
    return device


def resident_bases(chosen, mapping, local_tokens):
    """pressure_resident_values.resident_bases, native when available (same results)."""
    native = _native('reuse_resident_bases')
    if native is not None:
        return native(mapping, chosen, local_tokens, 32)
    from .pressure_resident_values import resident_bases as reference
    return reference(chosen, mapping, local_tokens)


def assemble_reused(query, state, index, chosen, sources, assembler, row_bytes, chunk_blocks):
    """GPU reuse for one layer of a prefill pressure refresh; returns assembly stats.

    Whole-layer snapshots in the existing resident-V buffer (one window): all
    moved V rows are gathered then scattered; then all reused K rows are
    gathered then rotated into place. Only blocks at the reuse limit read raw K
    from the CPU, through the bounded staging slots.
    """
    import torch
    from .packed_assembly import read_packed_records
    authority, staging, layer = state['authority'], assembler.staging, assembler.layers[0]
    controller = query.pressure
    rid = state['request_id']
    counts, untrusted = counters(controller, rid, index)
    if max(sources, default=0) + 32 > layer.cos_sin.shape[0]:
        raise ValueError('Reuse shift exceeds the layer RoPE table')
    m, r, reload, new_counts, skipped, rows = layer_plan(chosen, sources, counts, reuse_limit(), untrusted)
    stream = torch.cuda.current_stream(query.runner.device)
    uploaded = chunks = inline = 0
    if m:
        meta = upload_rows(assembler, rows)
        pages = layer.kernel_page_table
        snapshot = staging.value_destination(m * row_bytes)                   # waits for its last user
        gather_rows(layer.value, meta[:m], pages, snapshot[:m * row_bytes])
        scatter_rows(snapshot, meta[m:2 * m], pages, layer.value)
        if r:
            o = 2 * m
            gather_rows(layer.key, meta[o:o + r], pages, snapshot[:r * row_bytes])
            write_keys(snapshot, meta[o + 3 * r:o + 4 * r], meta[o + 4 * r:o + 5 * r], meta[o + 2 * r:o + 3 * r],
                       meta[o + r:o + 2 * r], layer, rows=row_bytes)
        for start in range(0, len(reload), chunk_blocks):
            part = reload[start:start + chunk_blocks]                         # (index, destination)
            n, size = len(part), len(part) * row_bytes
            records = [chosen[x[0]] for x in part]
            gpu, ready, slot = staging.destination(size)
            def fill(host):
                nonlocal inline
                _, stats = read_packed_records(authority, [x.handle for x in records], destination=host,
                    region=(index * row_bytes, row_bytes),
                    inline_cpu=bool(getattr(authority.archive, 'memory_only', False)))
                inline += int(stats.get('inline_cpu_region_read', False))
            transfer = assembler.transfers.upload_from_reader(size, gpu, fill, ready=ready)
            if transfer is None:
                next(iter(assembler.transfers.pending.values())).event.synchronize()
                transfer = assembler.transfers.upload_from_reader(size, gpu, fill, ready=ready)
            if transfer is None:
                raise RuntimeError('Reuse reload pinned transfer made no progress')
            stream.wait_event(transfer.event)
            meta_reload = assembler._upload_metadata(list(range(n)) + [1] * n + [0] * n + [x[1] for x in part])
            write_keys(gpu, meta_reload[:n], meta_reload[n:2 * n], meta_reload[2 * n:3 * n],
                       meta_reload[3 * n:], layer, rows=row_bytes)
            staging.consumed(slot)
            uploaded += size; chunks += 1
        done = torch.cuda.Event(); done.record(stream)
        staging.value_ready = done
    commit_counts(controller, rid, index, new_counts)
    record_stats(controller, reused=r, reloaded=len(reload), skipped=skipped, uploaded=uploaded)
    return dict(uploaded_bytes=uploaded, chunks=chunks, packed_chunks=chunks, chunk_blocks=chunk_blocks,
                staging_slot_bytes=staging.slot_bytes, paired_read_chunks=0, inline_read_chunks=inline,
                prepared_read_chunks=0, resident_v_reused_tokens=32 * m,
                gpu_reuse_blocks=r, gpu_reload_blocks=len(reload), gpu_skip_blocks=skipped)
