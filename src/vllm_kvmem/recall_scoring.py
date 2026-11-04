"""Pure FP32 block scorers for the fixed-policy recall experiment.

Inputs are real content-frame Q/K from the same model coordinate system.  This
module neither rotates Q/K nor changes the caller's candidate domain.  Returned
logits must be concatenated across *all* eligible block tiles before softmax.
ExactMass returns a block's token-logsumexp, so that this block softmax equals
global token softmax followed by a block sum, including partial blocks.

Fixed4 follows ``block_kdirection_fixed4_batch_kernel`` in the existing qw3
source: mean-direction first seed, cosine farthest-first remaining seeds,
nearest-seed direction groups, then means of the original (unnormalized) Keys.
Here clustering runs once per common physical block, never per hidden slice.
"""
from dataclasses import dataclass
from typing import Sequence

import torch


METHODS = ('mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass')
FIXED4_SOURCE = {
    'path': '/home/chaidi/qw3/src/kernels_cuda.cu',
    'function': 'block_kdirection_fixed4_batch_kernel',
    'sha256': '017898aaea1261624f017d92dae9392c0150de1476db6610d6c30b4c6116a821',
    'algorithm': 'mean-direction seed; cosine farthest-first; raw-Key cluster mean',
    'empty_cluster': 'duplicate first seed',
    'ties': 'earliest token/earliest seed',
}


@dataclass(frozen=True)
class BlockFeatures:
    method: str
    values: torch.Tensor  # [layer, block, representative, KV head, dim], FP32
    valid: torch.Tensor  # [layer, block, representative, KV head], bool
    counts: torch.Tensor  # [block], int64; original valid token counts

    def to(self, device):
        return BlockFeatures(self.method, self.values.to(device),
                             self.valid.to(device), self.counts.to(device))

    def select_blocks(self, indices):
        index = torch.as_tensor(indices, dtype=torch.long, device=self.values.device)
        if index.ndim != 1:
            raise ValueError('Block indices must be a one-dimensional sequence')
        return BlockFeatures(self.method, self.values.index_select(1, index),
                             self.valid.index_select(1, index),
                             self.counts.index_select(0, index))


def concat_block_features(parts: Sequence[BlockFeatures]):
    """Concatenate independent construction tiles without any score reduction."""
    if not parts:
        raise ValueError('At least one block-feature tile is required')
    first = parts[0]
    geometry = (first.values.shape[0], *first.values.shape[2:])
    for part in parts:
        if (part.method != first.method or part.values.device != first.values.device
                or (part.values.shape[0], *part.values.shape[2:]) != geometry):
            raise ValueError('Cannot concatenate incompatible block-feature tiles')
    return BlockFeatures(first.method, torch.cat([p.values for p in parts], dim=1),
                         torch.cat([p.valid for p in parts], dim=1),
                         torch.cat([p.counts for p in parts]))


def _raw_keys(keys, counts):
    if (not isinstance(keys, torch.Tensor) or keys.ndim != 5
            or not keys.is_floating_point() or any(n <= 0 for n in keys.shape)):
        raise ValueError('Keys must have nonempty [layer,block,token,KV-head,dim] geometry')
    counts = torch.as_tensor(counts, device=keys.device)
    if (counts.ndim != 1 or counts.numel() != keys.shape[1]
            or counts.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or bool(((counts < 0) | (counts > keys.shape[2])).any())):
        raise ValueError('Invalid per-block valid token counts')
    counts = counts.to(torch.long)
    valid = (torch.arange(keys.shape[2], device=keys.device)[None, :] < counts[:, None])
    valid = valid[None, :, :, None].expand(*keys.shape[:-1])
    raw = torch.where(valid[..., None], keys.float(), 0.)
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite valid Key in block features')
    return raw, counts, valid


def _gather_token(raw, index):
    return raw.gather(2, index[:, :, None, :, None].expand(
        raw.shape[0], raw.shape[1], 1, raw.shape[3], raw.shape[4])).squeeze(2)


def _fps4(raw, counts, valid):
    """Raw-Key Euclidean FPS with the existing maximum-norm initial seed."""
    layers, blocks, _, heads, dim = raw.shape
    available = valid.clone()
    criterion = raw.square().sum(-1)
    if not bool(torch.isfinite(criterion).all()):
        raise ValueError('FP32 raw-Key norm overflow')
    nearest = None
    values, slots = [], []
    for slot in range(4):
        chosen = criterion.masked_fill(~available, -torch.inf).argmax(2)
        live = (counts > slot)[None, :, None].expand(layers, blocks, heads)
        selected = _gather_token(raw, chosen)
        values.append(torch.where(live[..., None], selected, 0.))
        slots.append(live)
        available.scatter_(2, chosen[:, :, None, :], False)
        if slot < 3:
            distance = (raw - selected[:, :, None]).square().sum(-1)
            if not bool(torch.isfinite(distance).all()):
                raise ValueError('FP32 raw-Key distance overflow')
            nearest = distance if nearest is None else torch.minimum(nearest, distance)
            criterion = nearest
    return torch.stack(values, dim=2), torch.stack(slots, dim=2)


def _fixed4(raw, counts, valid):
    """Port of the qw3 direction-clustering construction for one whole block.

    A zero-norm direction uses the upstream 1e-20 norm-squared floor. Strict
    improvement preserves the upstream first-token/first-seed tie behavior.
    Empty groups duplicate the first *raw* seed, not the block mean or zero.
    """
    layers, blocks, _, heads, dim = raw.shape
    norm2 = raw.square().sum(-1)
    mean = raw.sum(2) / counts.clamp_min(1)[None, :, None, None]
    mean_norm2 = mean.square().sum(-1)
    if not bool(torch.isfinite(norm2).all() and torch.isfinite(mean_norm2).all()):
        raise ValueError('FP32 Key-direction norm overflow')
    inv_norm = norm2.clamp_min(1e-20).rsqrt()
    mean_inv_norm = mean_norm2.clamp_min(1e-20).rsqrt()
    mean_similarity = ((raw * mean[:, :, None]).sum(-1)
                       * inv_norm * mean_inv_norm[:, :, None])
    first = mean_similarity.masked_fill(~valid, -torch.inf).argmax(2)
    first_key = _gather_token(raw, first)
    first_norm = inv_norm.gather(2, first[:, :, None, :]).squeeze(2)
    best = (raw * first_key[:, :, None]).sum(-1) * inv_norm * first_norm[:, :, None]
    assignment = torch.zeros_like(valid, dtype=torch.long)
    available = valid.clone()
    available.scatter_(2, first[:, :, None, :], False)
    for slot in range(1, 4):
        chosen = best.masked_fill(~available, torch.inf).argmin(2)
        chosen = torch.where(available.any(2), chosen, first)
        selected = _gather_token(raw, chosen)
        selected_norm = inv_norm.gather(2, chosen[:, :, None, :]).squeeze(2)
        similarity = (raw * selected[:, :, None]).sum(-1) * inv_norm * selected_norm[:, :, None]
        better = (similarity > best) & valid
        assignment = torch.where(better, slot, assignment)
        best = torch.where(better, similarity, best)
        available.scatter_(2, chosen[:, :, None, :], False)
    values = []
    for slot in range(4):
        members = valid & (assignment == slot)
        number = members.sum(2)
        prototype = (raw * members[..., None]).sum(2) / number.clamp_min(1)[..., None]
        values.append(torch.where((number > 0)[..., None], prototype, first_key))
    live = (counts > 0)[None, :, None, None].expand(layers, blocks, 4, heads)
    result = torch.stack(values, dim=2)
    return torch.where(live[..., None], result, 0.), live


def build_block_features(keys, counts, method):
    """Construct exactly one physical-block index, independently per layer/head.

    ``fixed4`` and ``fps4`` always have four slots per physical block regardless
    of block size. FPS slots beyond its valid token count are masked; Fixed4
    duplicates the first seed for empty direction groups, matching its source.
    Padded Key entries may be NaN: only accepted valid token rows participate.
    """
    if method not in METHODS:
        raise ValueError(f'Unknown recall method: {method!r}')
    raw, counts, valid = _raw_keys(keys, counts)
    if method == 'mean_k':
        values = raw.sum(2, keepdim=True) / counts.clamp_min(1)[None, :, None, None, None]
        feature_valid = valid.any(2, keepdim=True)
    elif method == 'fixed4':
        values, feature_valid = _fixed4(raw, counts, valid)
    elif method == 'fps4':
        values, feature_valid = _fps4(raw, counts, valid)
    elif method == 'min_max':
        lower = raw.masked_fill(~valid[..., None], torch.inf).amin(2)
        upper = raw.masked_fill(~valid[..., None], -torch.inf).amax(2)
        values = torch.stack((lower, upper), dim=2)
        feature_valid = valid.any(2, keepdim=True).expand(*values.shape[:-1])
        values = torch.where(feature_valid[..., None], values, 0.)
    else:
        values, feature_valid = raw, valid
    if not bool(torch.isfinite(values).all()):
        raise ValueError('FP32 block-feature construction overflow')
    return BlockFeatures(method, values.contiguous(), feature_valid.contiguous(), counts)


@dataclass(frozen=True)
class PreparedQueries:
    """One reusable GQA packing of a Query tile; no original-Q reference.

    ``values`` is FP32 [layer * KV head, Query row * GQA group, dim].
    Optional sign parts avoid repeating Min-Max's splits for every block tile.
    The caller may release the unpacked GPU Query after preparing this object.
    """
    values: torch.Tensor
    shape: tuple[int, int, int, int]
    kv_heads: int
    positive: torch.Tensor | None = None
    negative: torch.Tensor | None = None


def prepare_queries(queries, kv_heads, *, split_sign=False, validate=True):
    """Pack Q once for all block tiles, preserving FP32 scoring precision.

    Internal callers may skip finite-value checks after checking the original
    CPU Query. Geometry checks always run and never read a device scalar.
    """
    if (not isinstance(queries, torch.Tensor) or queries.ndim != 4
            or not queries.is_floating_point() or any(n <= 0 for n in queries.shape)
            or type(kv_heads) is not int or kv_heads <= 0 or queries.shape[2] % kv_heads):
        raise ValueError('Invalid Query GQA geometry')
    q = queries.float()
    if validate and not bool(torch.isfinite(q).all()):
        raise ValueError('Nonfinite Query in recall scoring')
    layers, rows, heads, dim = q.shape
    packed = q.reshape(layers, rows, kv_heads, heads // kv_heads, dim).permute(
        0, 2, 1, 3, 4).reshape(layers * kv_heads, rows * (heads // kv_heads), dim).contiguous()
    return PreparedQueries(packed, tuple(q.shape), kv_heads,
                           packed.clamp_min(0) if split_sign else None,
                           packed.clamp_max(0) if split_sign else None)


def block_logits(queries, features, *, validate=True):
    """Return scaled [layer,Query row,Query head,block] FP32 logits.

    One GQA batched matmul evaluates all representatives in a bounded block
    tile. ExactMass reduces its token dimension with logsumexp; the other
    representative methods use max. Normalization still belongs to the caller
    over the complete eligible block domain. ``validate=False`` is reserved
    for internally checked inputs and performs no GPU-to-host scalar reads.
    """
    values = features.values
    if (features.method not in METHODS or values.ndim != 5 or values.dtype != torch.float32
            or any(n <= 0 for n in values.shape)
            or features.valid.shape != values.shape[:-1] or features.valid.dtype != torch.bool
            or features.valid.device != values.device or features.counts.shape != (values.shape[1],)
            or features.method == 'min_max' and values.shape[2] != 2):
        raise ValueError('Invalid block-feature geometry')
    layers, blocks, representatives, kvheads, dim = values.shape
    if isinstance(queries, PreparedQueries):
        prepared = queries
        shape = prepared.shape
        if (len(shape) != 4 or any(n <= 0 for n in shape) or shape[2] % kvheads
                or prepared.kv_heads != kvheads or shape[0] != layers or shape[3] != dim
                or prepared.values.shape != (layers * kvheads, shape[1] * (shape[2] // kvheads), dim)
                or prepared.values.dtype != torch.float32 or prepared.values.device != values.device
                or any(part is not None and (part.shape != prepared.values.shape
                    or part.dtype != torch.float32 or part.device != values.device)
                    for part in (prepared.positive, prepared.negative))):
            raise ValueError('Prepared Query and block-feature geometry/device disagree')
        if validate and not bool(torch.isfinite(prepared.values).all()):
            raise ValueError('Nonfinite Query in recall scoring')
    else:
        if (not isinstance(queries, torch.Tensor) or queries.ndim != 4
                or queries.shape[0] != layers or queries.shape[-1] != dim
                or queries.device != values.device):
            raise ValueError('Query and block-feature geometry/device disagree')
        prepared = prepare_queries(queries, kvheads, split_sign=features.method == 'min_max',
                                   validate=validate)
    _, rows, heads, _ = prepared.shape
    groups = heads // kvheads
    scale = dim ** -.5

    if features.method == 'min_max':
        positive = prepared.positive if prepared.positive is not None else prepared.values.clamp_min(0)
        negative = prepared.negative if prepared.negative is not None else prepared.values.clamp_max(0)

        def bound(query, slot):
            key = values[:, :, slot].permute(0, 2, 3, 1).reshape(layers * kvheads, dim, blocks)
            return torch.bmm(query, key)

        logits = bound(positive, 1)
        logits.add_(bound(negative, 0)).mul_(scale)
        valid = features.valid[:, :, 0].permute(0, 2, 1).reshape(layers * kvheads, 1, blocks)
        logits.masked_fill_(~valid, -torch.inf)
    else:
        # KV heads remain separate matmul batches: never repeat their Keys for
        # each Query head. Q128/B16/R32 at the qualified geometry uses 96 MiB
        # for scores; in-place scaling/masking avoid another same-sized copy.
        key = values.permute(0, 3, 4, 1, 2).reshape(layers * kvheads, dim, blocks * representatives)
        scores = torch.bmm(prepared.values, key)
        del key  # Drop a possible packing copy before logsumexp's scratch.
        scores.mul_(scale)
        valid = features.valid.permute(0, 3, 1, 2).reshape(layers * kvheads, 1, blocks * representatives)
        scores.masked_fill_(~valid, -torch.inf)
        scores = scores.view(layers * kvheads, rows * groups, blocks, representatives)
        logits = (scores.logsumexp(-1) if features.method == 'exact_mass' else scores.amax(-1))
    logits = logits.view(layers, kvheads, rows, groups, blocks).permute(0, 2, 1, 3, 4).reshape(
        layers, rows, heads, blocks)
    if validate:
        live = features.valid.any(2).permute(0, 2, 1)[:, None, :, None, :]
        finite = torch.isfinite(logits.view(layers, rows, kvheads, groups, blocks))
        if not bool((finite | ~live).all()):
            raise ValueError('FP32 recall logit overflow')
    return logits
