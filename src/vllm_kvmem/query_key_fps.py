"""Four real content Keys per physical block, selected by raw-Key k-center.

Distances choose representatives only. Retrieval uses the original Keys and
per-Query/head maximum dot products before the native block softmax domain.
The derived CPU cache shares the bounded archive pool with window scoring.
"""
import math
import time

import torch

from .query_window_scoring import WindowMeanCache


MODE = 'key_fps_4'
REPRESENTATIVES = 4
BLOCK_TOKENS = 32
SCRATCH_LIMIT = 512 * 1024**2


def _select_key_representatives(keys, valid_counts, *, track_ambiguity=False):
    """Return real Keys and original token offsets for each layer/block/head.

    Input is [layers, blocks, tokens <= 32, KV heads, dim]. Norms and direct
    squared differences use FP32. Argmax selects the lowest original offset
    on ties; already selected offsets are excluded even for identical Keys.
    Output slots beyond min(4, count) contain zero and index -1, which callers
    must exclude from maximum logits. Zero-norm *valid* Keys remain candidates.
    """
    if keys.ndim != 5 or not keys.is_floating_point():
        raise ValueError('Invalid raw-Key representative tensor')
    layers, blocks, tokens, heads, dim = keys.shape
    if not (layers > 0 and heads > 0 and dim > 0 and 0 < tokens <= BLOCK_TOKENS):
        raise ValueError('Invalid raw-Key representative geometry')
    counts = torch.as_tensor(valid_counts, device=keys.device)
    if (counts.shape != (blocks,) or counts.dtype not in
            (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError('Invalid raw-Key valid counts')
    if bool(((counts < 0) | (counts > tokens)).any()):
        raise ValueError('Raw-Key valid counts exceed the token dimension')
    # A cache builder calls this on at most 32 records. Reject unexpectedly
    # large GPU batches before creating the FP32/distance temporaries.
    if keys.device.type == 'cuda' and keys.numel()*4*5 > SCRATCH_LIMIT:
        raise MemoryError('Raw-Key selection scratch exceeds 512 MiB')
    available = (torch.arange(tokens, device=keys.device)[None, None, :, None]
        < counts[None, :, None, None]).expand(layers, blocks, tokens, heads).clone()
    raw = torch.where(available[..., None], keys.float(), 0.)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite original Key in representative index')
    criterion = raw.square().sum(-1)
    if not bool(torch.isfinite(criterion).all()):
        raise ValueError('Nonfinite FP32 raw-Key norm')
    nearest = None
    indices = torch.full((layers, blocks, REPRESENTATIVES, heads), -1,
        dtype=torch.long, device=keys.device)
    uncertain = torch.zeros((layers, blocks, heads), dtype=torch.bool, device=keys.device)
    for slot in range(REPRESENTATIVES):
        candidates = criterion.masked_fill(~available, -torch.inf)
        chosen = candidates.argmax(2)
        if track_ambiguity and tokens > 1:
            best = candidates.topk(2, dim=2).values
            # FP8 inputs give identical elementwise FP32 subtraction/squaring
            # on CPU and GPU implementations. Only summation order may
            # differ. This conservative bound exceeds the two sequential-sum
            # gamma_(dim-1) error bounds; taking min over distances preserves
            # their relative intervals. Ties/near ties use the CPU reference.
            tolerance = 4*dim*torch.finfo(torch.float32).eps*best[:, :, 0].abs()
            uncertain |= torch.isfinite(best[:, :, 1]) & (
                best[:, :, 0]-best[:, :, 1] <= tolerance)
        live = (counts > slot)[None, :, None].expand(layers, blocks, heads)
        indices[:, :, slot] = torch.where(live, chosen, -1)
        available.scatter_(2, chosen[:, :, None, :], False)
        if slot + 1 == REPRESENTATIVES:
            break
        selected = raw.gather(2, chosen[:, :, None, :, None].expand(
            layers, blocks, 1, heads, dim))
        # Direct differences avoid cancellation in ||k||²+||s||²-2k·s.
        distance = (raw-selected).square_().sum(-1)
        if not bool(torch.isfinite(distance).all()):
            raise ValueError('Nonfinite FP32 raw-Key distance')
        nearest = distance if nearest is None else torch.minimum(nearest, distance)
        criterion = nearest
    gather_indices = indices.clamp_min(0)[..., None].expand(
        layers, blocks, REPRESENTATIVES, heads, dim)
    # CPU gather does not support FP8. FP32 represents every finite FP8 value
    # exactly; casting the selected values back preserves the original Keys.
    source = raw if keys.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else keys
    result = source.gather(2, gather_indices)
    result = torch.where(indices[..., None] >= 0, result, 0.).to(keys.dtype)
    return result, indices, uncertain


def select_key_representatives(keys, valid_counts):
    """Reference raw-Key FPS; preserve CPU reduction and deterministic ties."""
    representatives, indices, _ = _select_key_representatives(keys, valid_counts)
    return representatives, indices


def select_key_representatives_optimized(keys, valid_counts, *, device, cache_dtype=None):
    """FP8 GPU construction with CPU reference only for ambiguous groups.

    Returned real Keys and offsets live on CPU. ``cache_dtype`` optionally moves
    the finite FP8 -> cache conversion onto the GPU; its exact embedding is
    qualified only for FP16, BF16 and FP32. A group is one layer, original
    block and KV head. Near-tie groups are recomputed from their original Keys,
    never from an already-selected subset. Unsupported geometry/dtypes retain
    the reference path and report that fallback explicitly.
    """
    device = torch.device(device)
    if cache_dtype not in (None, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('Invalid exact FP8 representative cache dtype')
    groups = keys.shape[0]*keys.shape[1]*keys.shape[3]
    stats = dict(fps_gpu_groups=0, fps_cpu_fallback_groups=0, fps_device_cache_conversion=False,
        fps_reduction_guard='4 * dim * FP32 epsilon relative top-two gap')
    if keys.device.type != 'cpu':
        raise ValueError('Optimized FPS requires an owned original CPU Key slab')
    if (device.type != 'cuda' or keys.dtype != torch.float8_e4m3fn or keys.shape[-1] > 4096):
        representatives, indices = select_key_representatives(keys, valid_counts)
        stats.update(fps_cpu_fallback_groups=groups,
            fps_fallback_reason='GPU construction requires FP8 e4m3fn CPU Keys and dim <= 4096')
        return representatives, indices, stats
    # The additional converted output can coexist with the selected FP8 values.
    # Include it before any device allocation, even though the FPS reduction
    # temporaries normally die before conversion starts.
    converted_bytes = (keys.shape[0]*keys.shape[1]*REPRESENTATIVES*keys.shape[3]*keys.shape[4]
        * torch.empty((), dtype=cache_dtype).element_size() if cache_dtype is not None else 0)
    if keys.numel()*4*5+converted_bytes > SCRATCH_LIMIT:
        raise MemoryError('Optimized FPS selection/conversion scratch exceeds 512 MiB')
    gpu = keys.to(device, non_blocking=keys.is_pinned())
    representatives, indices, uncertain = _select_key_representatives(gpu, valid_counts,
        track_ambiguity=True)
    if cache_dtype is not None:
        # _select_key_representatives already rejected every nonfinite valid
        # FP8 input. Every finite e4m3fn value embeds exactly in these formats;
        # no host FP32 round-trip validation or arithmetic is needed here.
        representatives = representatives.to(cache_dtype)
        stats['fps_device_cache_conversion'] = True
    representatives, indices = representatives.cpu(), indices.cpu()
    triples = uncertain.nonzero().cpu()
    stats.update(fps_gpu_groups=groups, fps_cpu_fallback_groups=len(triples),
        fps_fallback_reason='near-tie FP32 reduction guard' if len(triples) else None)
    if len(triples):
        layer, block, head = triples.unbind(1)
        # uint8 advanced indexing is supported for archived FP8 on CPU. The
        # selected group shape keeps each original token's dim reduction intact.
        subset = keys.view(torch.uint8)[layer, block, :, head, :].view(keys.dtype)
        subset = subset.unsqueeze(0).unsqueeze(3)
        counts = torch.as_tensor(valid_counts, device='cpu')[block]
        exact, exact_indices = select_key_representatives(subset, counts)
        if cache_dtype is None:
            representatives.view(torch.uint8)[layer, block, :, head, :] = exact[0, :, :, 0].view(torch.uint8)
        else:
            # Only the ambiguous groups use CPU FPS. Conversion of their
            # original finite FP8 values uses the same exact embedding.
            representatives[layer, block, :, head, :] = exact[0, :, :, 0].to(cache_dtype)
        indices[layer, block, :, head] = exact_indices[0, :, :, 0]
    return representatives, indices, stats


class RawKeyFPSCache(WindowMeanCache):
    """Owner-local real-Key index, with shared bounded/LRU archive storage.

    A row depends only on that block's logical location, handle generation and
    accepted count. Neither adjacent blocks nor Query/transport masks enter its
    signature. No archive record or read lease survives snapshot construction.
    """
    def __init__(self, archive, layers, heads, dim, *, dtype, max_bytes):
        if not (type(layers) is int and type(heads) is int and type(dim) is int
                and min(layers, heads, dim) > 0):
            raise ValueError('Invalid representative cache geometry')
        if dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError('Representative cache requires a floating CPU index')
        self.indices = None
        super().__init__(archive, layers, heads, dim, dtype=dtype, max_bytes=max_bytes)
        # Token offsets fit in signed int8 (-1 means absent); count these bytes
        # in addition to the parent's bounded Python metadata allowance.
        self.row_bytes += layers * REPRESENTATIVES * heads

    def clear(self):
        with self._lock:
            self.indices = None
            super().clear()

    def _reserve(self, records):
        super()._reserve(records)
        if self.capacity and self.indices is None:
            layers, heads, _ = self.geometry
            try:
                self.indices = torch.full((layers, self.capacity, REPRESENTATIVES, heads),
                    -1, dtype=torch.int8)
            except BaseException:
                self.clear()
                raise

    def _publish_batch(self, batch, representatives, indices, signatures, *, optimized=False):
        """Publish a whole batch only after both value and offset writes succeed.

        Any partial write leaves every affected signature invalid. Subsequent
        snapshots reconstruct all affected records before exposing those slots.
        """
        for i in batch:
            self.signatures[i] = None
        contiguous = optimized and batch[-1]-batch[0]+1 == len(batch)
        if contiguous:
            begin, end = batch[0], batch[-1]+1
            self.tensor[:, begin:end].copy_(representatives)
            self.indices[:, begin:end].copy_(indices)
        else:
            slots = torch.tensor(batch, dtype=torch.long)
            self.tensor.index_copy_(1, slots, representatives)
            self.indices.index_copy_(1, slots, indices.to(torch.int8))
        for i in batch:
            self.signatures[i] = signatures[i]
        return contiguous

    def snapshot(self, authority, records, *, optimized=False, device='cpu'):
        starts, counts = [r.history_begin for r in records], [r.count for r in records]
        if (any(type(s) is not int or s < 0 or s % BLOCK_TOKENS for s in starts)
                or any(type(n) is not int or not 0 < n <= BLOCK_TOKENS for n in counts)
                or any(a >= b for a, b in zip(starts, starts[1:]))
                or any(s+n > authority.history_end for s, n in zip(starts, counts))):
            raise ValueError('Invalid original 32-token representative history index')
        self._reserve(len(records))
        signatures = [(r.history_begin, r.handle.slot, r.handle.generation, r.count)
            for r in records]
        dirty = [i for i, sig in enumerate(signatures) if self.signatures[i] != sig]
        layers, heads, dim = self.geometry
        work = dict(key_read_bytes=0, value_read_bytes=0, draft_key_read_bytes=0,
            key_read_batches=0, key_resident_bytes=0, key_region_seconds=0.,
            fps_gpu_groups=0, fps_cpu_fallback_groups=0, fps_read_seconds=0.,
            fps_select_seconds=0., fps_publish_seconds=0.)
        work.update(fps_key_slab_bytes_max=0, fps_selection_scratch_bound_bytes=0)
        work.update(fps_device_cache_conversion_batches=0, fps_publication_slice_batches=0,
            fps_publication_index_copy_batches=0, fps_publication_strict_check_batches=0)
        fallback_reasons = set()
        for begin in range(0, len(dirty), 32):
            batch = dirty[begin:begin+32]
            started = time.perf_counter()
            payloads = None
            exact_device_conversion = False
            if optimized:
                scratch = (layers*len(batch)*BLOCK_TOKENS*heads*dim*4*5
                    + layers*len(batch)*REPRESENTATIVES*heads*dim
                    * torch.empty((), dtype=self.dtype).element_size())
                if scratch > SCRATCH_LIMIT:
                    raise MemoryError('Optimized FPS selection scratch exceeds 512 MiB')
                work['fps_selection_scratch_bound_bytes'] = max(
                    work['fps_selection_scratch_bound_bytes'], scratch)
                packed, read_stats = authority.read_score_keys([records[i].handle for i in batch],
                    pin_memory=torch.device(device).type == 'cuda')
                work['fps_key_slab_bytes_max'] = max(work['fps_key_slab_bytes_max'],
                    packed.numel()*packed.element_size())
                raw = packed.permute(1, 0, 2, 3, 4)
                if raw.shape != (layers, len(batch), BLOCK_TOKENS, heads, dim):
                    raise ValueError('Scoring Key-only archive geometry changed')
                for name, value in read_stats.items():
                    work[name] = work.get(name, 0)+value
            else:
                payloads = authority.read_records([records[i].handle for i in batch])
                if len(payloads) != len(batch):
                    raise ValueError('Incomplete original-Key archive batch')
                dtype = payloads[0][0].dtype
                raw = torch.zeros((layers, len(batch), BLOCK_TOKENS, heads, dim), dtype=dtype)
                for row, (i, payload) in enumerate(zip(batch, payloads)):
                    key = payload[0]
                    if (key.device.type != 'cpu' or key.ndim != 4 or key.shape[0] < layers
                            or key.shape[1:] != (counts[i], heads, dim) or key.dtype != dtype):
                        raise ValueError('Original-Key archive geometry changed')
                    raw[:, row, :counts[i]].copy_(key[:layers])
            selected_at = time.perf_counter()
            work['fps_read_seconds'] += selected_at-started
            if optimized:
                representatives, indices, select_stats = select_key_representatives_optimized(
                    raw, [counts[i] for i in batch], device=device, cache_dtype=self.dtype)
                exact_device_conversion = select_stats['fps_device_cache_conversion']
                work['fps_device_cache_conversion_batches'] += int(exact_device_conversion)
                work['fps_gpu_groups'] += select_stats['fps_gpu_groups']
                work['fps_cpu_fallback_groups'] += select_stats['fps_cpu_fallback_groups']
                if select_stats['fps_fallback_reason']:
                    fallback_reasons.add(select_stats['fps_fallback_reason'])
                del packed
            else:
                representatives, indices = select_key_representatives(raw, [counts[i] for i in batch])
            published_at = time.perf_counter()
            work['fps_select_seconds'] += published_at-selected_at
            if exact_device_conversion:
                if representatives.dtype != self.dtype or raw.dtype != torch.float8_e4m3fn:
                    raise ValueError('Unqualified representative device conversion')
                converted = representatives
            else:
                # CPU/non-FP8/reference paths retain their strict original
                # conversion check; no unqualified dtype gets a fast path.
                converted = representatives.to(self.dtype)
                if not torch.equal(converted.float(), representatives.float()):
                    raise ValueError('Representative cache dtype changes an original Key')
                work['fps_publication_strict_check_batches'] += 1
            contiguous = self._publish_batch(batch, converted, indices, signatures,
                optimized=optimized)
            work['fps_publication_slice_batches' if contiguous else
                'fps_publication_index_copy_batches'] += 1
            work['fps_publish_seconds'] += time.perf_counter()-published_at
            del payloads, raw, representatives, converted, indices
        self.signatures[len(records):] = [None]*(self.capacity-len(records))
        if self.tensor is None:
            representatives = torch.empty((layers, 0, REPRESENTATIVES, heads, dim), dtype=self.dtype)
            indices = torch.empty((layers, 0, REPRESENTATIVES, heads), dtype=torch.int8)
        else:
            representatives, indices = self.tensor[:, :len(records)], self.indices[:, :len(records)]
        distinct = [min(REPRESENTATIVES, n) for n in counts]
        return representatives, indices, dict(**work, optimized=optimized,
            fps_fallback_reasons=sorted(fallback_reasons), cache_bytes=self.bytes,
            cache_capacity_bytes=self.max_bytes, cache_hit_blocks=len(records)-len(dirty),
            cache_rebuilt_blocks=len(dirty), valid_representatives=sum(distinct)*layers*heads,
            distinct_token_offsets_min=min(distinct, default=0),
            distinct_token_offsets_max=max(distinct, default=0))


# Keep the integration name short without creating a second cache type/pool.
KeyFPSCache = RawKeyFPSCache


def key_fps_scores(queries, representatives, indices, score_mask, *, scale=None,
                   query_half=False, query_chunk=8, optimized=False):
    """Real-Key logits -> per-head/Query block max -> block softmax -> mean/sum.

    Only one layer and up to eight Query rows are resident in scratch. Native
    score_mask owns the complete normalization domain; transport is not an input.
    """
    if queries.ndim != 4 or representatives.ndim != 5:
        raise ValueError('Invalid representative scoring tensor ranks')
    layers, rows, heads, dim = queries.shape
    rl, blocks, slots, kvheads, rd = representatives.shape
    if (rl != layers or rd != dim or slots != REPRESENTATIVES or not kvheads
            or heads % kvheads or not layers or not heads or not dim
            or indices.shape != (layers, blocks, REPRESENTATIVES, kvheads)
            or indices.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64)
            or score_mask.shape != (blocks,) or score_mask.dtype != torch.bool
            or type(query_chunk) is not int or not 0 < query_chunk <= 8):
        raise ValueError('Invalid GQA/representative/block score geometry')
    if bool(((indices < -1) | (indices >= BLOCK_TOKENS)).any()):
        raise ValueError('Representative index is outside its original block')
    scale = dim**-.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError('Invalid Query scale')
    device = queries.device
    score_mask = score_mask.to(device)
    scores = torch.zeros(blocks, dtype=torch.float32, device=device)
    if not bool(score_mask.any()) or not rows:
        return scores
    # Includes a possible contiguous Key packing copy, logits, pooled logits,
    # probabilities, masks and Q conversions. No all-layer/all-Query logits.
    scratch = (2*blocks*REPRESENTATIVES*kvheads*dim*4
        + 3*query_chunk*heads*blocks*REPRESENTATIVES*4
        + 3*query_chunk*heads*blocks*4 + 2*query_chunk*heads*dim*4
        + blocks*REPRESENTATIVES*kvheads*2 + blocks*8)
    if optimized:
        scratch += blocks*REPRESENTATIVES*kvheads*dim*representatives.element_size()
    if scratch > SCRATCH_LIMIT:
        raise MemoryError('Raw-Key representative scoring scratch exceeds 512 MiB per layer')
    for layer in range(layers):
        valid = (indices[layer] >= 0).permute(2, 0, 1).to(device)
        if bool((score_mask[None, :] & ~valid.any(-1)).any()):
            raise ValueError('Scorable block has no valid Key representative')
        if optimized:
            # Transfer the compact cached dtype first; convert and pack on the
            # destination device instead of host-side strided FP16 -> FP32 copy.
            source = representatives[layer].to(device=device)
            keys = source.permute(2, 3, 0, 1).reshape(
                kvheads, dim, blocks*REPRESENTATIVES).float().contiguous()
            del source
        else:
            keys = representatives[layer].permute(2, 3, 0, 1).reshape(
                kvheads, dim, blocks*REPRESENTATIVES).to(device=device, dtype=torch.float32).contiguous()
        for begin in range(0, rows, query_chunk):
            query = queries[layer, begin:begin+query_chunk]
            if query_half:
                query = query.half()
            n = query.shape[0]
            query = query.float().reshape(n, kvheads, heads//kvheads, dim)
            logits = torch.einsum('tkgd,kdw->tkgw', query, keys)
            logits.mul_(scale)
            logits = logits.reshape(n, kvheads, heads//kvheads, blocks, REPRESENTATIVES)
            logits.masked_fill_(~valid[None, :, None, :, :], -torch.inf)
            pooled = logits.amax(-1)
            pooled.masked_fill_(~score_mask, -torch.inf)
            probability = pooled.softmax(-1)
            scores.add_(probability.sum((0, 1, 2)), alpha=1/(layers*heads))
            del query, logits, pooled, probability
        del keys, valid
    if not bool(torch.isfinite(scores).all()):
        raise ValueError('Nonfinite real-Key representative block probability')
    return scores


def score_authority_fps(authority, records, queries, block_means, score_mask):
    started = time.perf_counter()
    cache = authority.query_window_cache
    layers, heads, dim = cache.geometry
    optimized = getattr(getattr(authority, 'graph_options', None), 'query_fps_optimized', False)
    if block_means.shape != (layers, len(records), heads, dim):
        raise ValueError('Original block index differs from the representative index')
    with cache.use():
        cache_started = time.perf_counter()
        representatives, indices, stats = cache.snapshot(authority, records,
            optimized=optimized, device=queries.device)
        cache_seconds = time.perf_counter()-cache_started
        scoring_started = time.perf_counter()
        scores = key_fps_scores(queries, representatives, indices,
            torch.tensor(score_mask, dtype=torch.bool),
            query_half=authority.mean_k_backend == 'qw3_cuda', optimized=optimized)
        scoring_seconds = time.perf_counter()-scoring_started
        del representatives, indices
    stats.update(method=MODE, representatives_per_block=REPRESENTATIVES,
        block_tokens=BLOCK_TOKENS, selection='raw_key_farthest_first',
        distance='FP32 direct squared Euclidean on raw Keys',
        first_representative='maximum raw-Key squared norm',
        tie_break='lowest original token offset', fallback_blocks=0,
        normalization='block softmax after per-layer/per-Query/per-head representative max',
        key_normalization='none', cache_key_dtype=str(cache.dtype),
        rope='existing content frame; no additional RoPE',
        cache_build_seconds=cache_seconds,
        cache_timing_scope='snapshot: capacity checks, archive reads, FPS and publication',
        scoring_seconds=scoring_seconds,
        seconds=time.perf_counter()-started)
    return scores, stats
