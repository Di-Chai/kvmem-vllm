"""16/8 content-Key windows, reduced to original 32-token blocks before softmax.

The cache owns only derived CPU means. It never pins archive records, changes
logical positions, adds RoPE, or changes the native picker/transport domain.
"""
from dataclasses import dataclass
from contextlib import contextmanager
import math
import time
import weakref
from threading import RLock

import torch


MODE = 'overlap_16_8'


@dataclass(frozen=True)
class Window:
    slot: int
    begin: int
    end: int
    # (original block index, offset inside that block, number of tokens)
    parts: tuple


def window_layout(starts, counts, history_end):
    """All geometrically valid windows; masks are deliberately not cached.

    Only the actual history tail can shorten a window. An interior hole or a
    partial interior block invalidates the whole window, rather than joining
    the two sides of the gap or treating it as a shorter window.
    """
    if (len(starts) != len(counts) or type(history_end) is not int or history_end < 0 or
            any(type(s) is not int or s < 0 or s % 32 for s in starts) or
            any(type(n) is not int or not 0 < n <= 32 for n in counts) or
            any(a >= b for a, b in zip(starts, starts[1:])) or
            any(s+n > history_end for s, n in zip(starts, counts))):
        raise ValueError('Invalid original 32-token history index')
    lookup = {start: i for i, start in enumerate(starts)}
    result = []
    for i, (start, count) in enumerate(zip(starts, counts)):
        for offset in range(0, count, 8):
            begin, end = start+offset, min(start+offset+16, history_end)
            cursor, parts = begin, []
            while cursor < end:
                block = lookup.get(cursor//32*32)
                if block is None or cursor >= starts[block]+counts[block]:
                    break
                length = min(end, starts[block]+counts[block])-cursor
                parts.append((block, cursor-starts[block], length))
                cursor += length
            if cursor == end:
                result.append(Window(i*4+offset//8, begin, end, tuple(parts)))
    return tuple(result)


def window_targets(windows, blocks, score_mask, *, device='cpu'):
    """Score exclusion invalidates entire windows; transport is not an input."""
    if len(score_mask) != blocks:
        raise ValueError('Window score mask differs from the original block index')
    targets = torch.full((blocks*4, 2), -1, dtype=torch.long, device=device)
    valid = torch.zeros(blocks*4, dtype=torch.bool, device=device)
    # Build on CPU to avoid thousands of tiny CUDA assignments.
    pairs, good = [[-1, -1] for _ in range(blocks*4)], [False]*(blocks*4)
    for window in windows:
        owners = tuple(part[0] for part in window.parts)
        if all(score_mask[i] for i in owners):
            good[window.slot] = True
            pairs[window.slot][:len(owners)] = owners
    if pairs:
        targets.copy_(torch.tensor(pairs, dtype=torch.long))
        valid.copy_(torch.tensor(good, dtype=torch.bool))
    return targets, valid


class WindowMeanCache:
    """Owner-local, capacity-bounded means, qualified by archive generations.

    One slab avoids one extra native allocation owner per historical block.
    The slab and metadata allowance share a globally reserved archive cache
    budget. Inactive owner caches can be evicted; physical KV slots are unchanged.
    Missing/changed entries are published only after complete archive reads.
    """
    def __init__(self, archive, layers, heads, dim, *, dtype, max_bytes):
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('Window cache capacity must be positive')
        self.archive, self.geometry = archive, (layers, heads, dim)
        self.dtype, self.max_bytes = dtype, max_bytes
        self.row_bytes = layers*4*heads*dim*torch.empty((), dtype=dtype).element_size()+2048
        self.tensor, self.signatures, self.capacity, self.bytes = None, [], 0, 0
        self.active, self.touched = 0, time.monotonic()
        with archive._lock:
            if not hasattr(archive, '_query_window_caches'):
                archive._query_window_caches = weakref.WeakSet()
                archive._query_window_cache_lock = RLock()
                archive._query_window_cache_bytes = 0
                archive._query_window_cache_limit = max_bytes
            self._lock = archive._query_window_cache_lock
            if archive._query_window_cache_limit != max_bytes:
                raise ValueError('Window caches sharing an archive must share one capacity')
            with self._lock:
                archive._query_window_caches.add(self)

    @contextmanager
    def use(self):
        with self._lock:
            self.active += 1
            self.touched = time.monotonic()
        try:
            yield self
        finally:
            with self._lock:
                self.active -= 1

    def clear(self):
        with self._lock:
            self.tensor, self.signatures, self.capacity = None, [], 0
            if self.bytes:
                self.archive._query_window_cache_bytes -= self.bytes
                self.bytes = 0

    def __del__(self):
        if hasattr(self, '_lock'):
            self.clear()

    def _reserve(self, records):
        maximum = self.max_bytes//self.row_bytes
        if records > maximum:
            raise MemoryError('16/8 window index exceeds query_window_cache_bytes')
        if records <= self.capacity:
            return
        capacity = min(maximum, 1 << max(0, records-1).bit_length())
        # Resizing discards the derived cache before allocating its replacement,
        # keeping the capacity bound true even at the high-water mark.
        self.clear()
        layers, heads, dim = self.geometry
        with self._lock:
            amount = capacity*self.row_bytes
            for other in sorted(self.archive._query_window_caches, key=lambda c: c.touched):
                if self.archive._query_window_cache_bytes+amount <= self.max_bytes:
                    break
                if other is not self and not other.active:
                    other.clear()
            if self.archive._query_window_cache_bytes+amount > self.max_bytes:
                raise MemoryError('16/8 window cache is occupied by active scoring owners')
            tensor = torch.empty((layers, capacity, 4, heads, dim), dtype=self.dtype)
            self.tensor, self.capacity = tensor, capacity
            self.signatures = [None]*capacity
            self.bytes = amount
            self.archive._query_window_cache_bytes += amount

    def snapshot(self, authority, records):
        starts = [r.history_begin for r in records]
        counts = [r.count for r in records]
        windows = window_layout(starts, counts, authority.history_end)
        self._reserve(len(records))
        by_block = [[] for _ in records]
        for window in windows:
            by_block[window.slot//4].append(window)
        signatures = []
        for i, group in enumerate(by_block):
            r = records[i]
            signatures.append((r.history_begin, r.handle.slot, r.handle.generation, r.count,
                tuple((w.begin, w.end, tuple((records[b].handle.slot,
                    records[b].handle.generation, offset, length) for b, offset, length in w.parts))
                    for w in group)))
        dirty = [i for i, sig in enumerate(signatures) if self.signatures[i] != sig]
        # Bound temporary raw reads; next-block dependencies are read together
        # with the source. Never read a contiguous span across logical holes.
        for offset in range(0, len(dirty), 32):
            batch = dirty[offset:offset+32]
            needed = sorted({part[0] for i in batch for w in by_block[i] for part in w.parts})
            payloads = authority.read_records([records[i].handle for i in needed]) if needed else []
            raw = {i: value[0][:self.geometry[0]].float() for i, value in zip(needed, payloads)}
            for i in batch:
                self.signatures[i] = None
                self.tensor[:, i].fill_(-torch.inf)
                for window in by_block[i]:
                    pieces = [raw[b][:, begin:begin+length] for b, begin, length in window.parts]
                    keys = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)
                    # FP32 reduction of the original archived content K; the
                    # original backend controls the stored mean precision.
                    mean = keys.mean(dim=1)
                    if not bool(torch.isfinite(mean).all()):
                        raise ValueError('Nonfinite original Key in window index')
                    self.tensor[:, i, window.slot % 4].copy_(mean)
                self.signatures[i] = signatures[i]
            del payloads, raw
        # Extra retained slots are invalidated on shrinking/rewinding history.
        self.signatures[len(records):] = [None]*(self.capacity-len(records))
        view = self.tensor[:, :len(records)].flatten(1, 2)
        return view, windows, dict(cache_bytes=self.bytes, cache_capacity_bytes=self.max_bytes,
            cache_hit_blocks=len(records)-len(dirty), cache_rebuilt_blocks=len(dirty))


def overlap_window_scores(queries, means, targets, valid, block_means, score_mask,
                          *, scale=None, query_half=False, query_chunk=8):
    """Per L/T/head window dot -> block max -> block softmax -> mean/sum.

    Layers and Query rows are tiled. Scratch is bounded by one layer and at
    most eight Query rows; no complete [L,T,H,windows] tensor is materialized.
    """
    if queries.ndim != 4 or means.ndim != 4 or block_means.ndim != 4:
        raise ValueError('Invalid window scoring tensor ranks')
    layers, rows, heads, dim = queries.shape
    ml, windows, kvheads, md = means.shape
    blocks = block_means.shape[1]
    if (ml != layers or md != dim or not kvheads or heads % kvheads or
            block_means.shape != (layers, blocks, kvheads, dim) or
            targets.shape != (windows, 2) or valid.shape != (windows,) or
            score_mask.shape != (blocks,) or score_mask.dtype != torch.bool or
            valid.dtype != torch.bool or targets.dtype != torch.long or
            not layers or not heads or not dim or not 0 < query_chunk <= 8):
        raise ValueError('Invalid GQA/window/block score geometry')
    scale = dim**-.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError('Invalid Query scale')
    device = queries.device
    targets, valid, score_mask = targets.to(device), valid.to(device), score_mask.to(device)
    if bool(((targets < -1) | (targets >= blocks)).any()):
        raise ValueError('Window maps outside original blocks')
    covered = torch.zeros(blocks, dtype=torch.bool, device=device)
    for side in range(2):
        indices = targets[valid & (targets[:, side] >= 0), side]
        covered[indices] = True
    fallback = ~covered & score_mask
    scores = torch.zeros(blocks, device=device, dtype=torch.float32)
    if not bool(score_mask.any()):
        return scores
    # Explicit bounds also cover unusual head geometry, rather than assuming
    # the current model's 24 Q heads / 4 KV heads.
    scratch = windows*kvheads*dim*4 + blocks*kvheads*dim*4 + query_chunk*heads*(windows+blocks)*8
    if scratch > 512*1024**2:
        raise MemoryError('16/8 scoring scratch exceeds 512 MiB per layer')
    for layer in range(layers):
        wm = means[layer].to(device=device, dtype=torch.float32).permute(1, 2, 0)
        bm = block_means[layer].to(device=device, dtype=torch.float32).permute(1, 2, 0)
        for begin in range(0, rows, query_chunk):
            q = queries[layer, begin:begin+query_chunk]
            if query_half:
                q = q.half()
            n = q.shape[0]
            q = q.float().reshape(n, kvheads, heads//kvheads, dim)
            logits = torch.einsum('tkgd,kdw->tkgw', q, wm)*scale
            logits.masked_fill_(~valid, -torch.inf)
            pooled = torch.full((*logits.shape[:-1], blocks), -torch.inf, device=device)
            for side in range(2):
                index = targets[:, side]
                contribution = logits.masked_fill(index < 0, -torch.inf)
                pooled.scatter_reduce_(-1, index.clamp_min(0).expand_as(logits),
                    contribution, reduce='amax', include_self=True)
            if bool(fallback.any()):
                original = torch.einsum('tkgd,kdb->tkgb', q, bm)*scale
                pooled = torch.where(fallback, original, pooled)
            pooled.masked_fill_(~score_mask, -torch.inf)
            probability = pooled.softmax(-1)
            scores.add_(probability.sum((0, 1, 2)), alpha=1/(layers*heads))
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite 16/8 block probability')
    return scores


def score_authority_windows(authority, records, queries, block_means, score_mask):
    started = time.perf_counter()
    cache = authority.query_window_cache
    with cache.use():
        means, windows, stats = cache.snapshot(authority, records)
        targets, valid = window_targets(windows, len(records), score_mask)
        eligible = torch.tensor(score_mask, dtype=torch.bool)
        scores = overlap_window_scores(queries, means, targets, valid, block_means, eligible,
            query_half=authority.mean_k_backend == 'qw3_cuda')
        del means
    covered = set(targets[valid].flatten().tolist()) - {-1}
    stats.update(method=MODE, window_tokens=16, window_stride=8, block_tokens=32,
        valid_windows=int(valid.sum()), geometry_windows=len(windows),
        fallback_blocks=sum(bool(keep) and i not in covered for i, keep in enumerate(score_mask)),
        seconds=time.perf_counter()-started,
        normalization='block softmax after per-layer/per-Query/per-head window max',
        cache_mean_dtype=str(cache.dtype), rope='existing content frame; no additional RoPE')
    return scores, stats
