"""Reference attention/value block scores and whole-set deletion diagnostics.

Q/K use the existing ExactMass content frame. Values must be dequantized model
Values, not their uint8 archive encodings. No block-score softmax is applied:
scores are summed over Query rows and averaged over target layers/Query heads.
This deliberately keeps full block moments on the CPU; it is not an index or
an optimized serving kernel.
"""
from dataclasses import dataclass
import math

import torch


VALUE_METHODS = ('block_value_norm', 'block_deletion_error')
REFERENCE_METHODS = ('exact_mass', *VALUE_METHODS)


@dataclass
class AttentionMoments:
    log_mass: torch.Tensor             # [Query/head row, block], float64 CPU
    mean_value: torch.Tensor           # [row, block, Value dimension]
    mean_value_l1: torch.Tensor        # [row, block]


def moments_from_blocks(query, keys, values, counts, eligible, *, host=True, with_value_mean=True):
    """One layer/KV-head tile; token softmax is conditional within each block.

    Returns unnormalized block log-mass and conditional Value moments. Global
    normalization happens only after every candidate block tile is concatenated.
    Padding (including NaNs) and ineligible blocks have no attention mass.
    """
    if (query.ndim != 2 or keys.ndim != 3 or values.ndim != 3 or
            keys.shape[:2] != values.shape[:2] or query.shape[-1] != keys.shape[-1] or
            not query.is_floating_point() or not keys.is_floating_point() or
            not values.is_floating_point() or query.device != keys.device or
            values.device != keys.device):
        raise ValueError('Invalid attention Q/K/V geometry or device')
    counts = torch.as_tensor(counts, device=keys.device, dtype=torch.long)
    eligible = torch.as_tensor(eligible, device=keys.device, dtype=torch.bool)
    if (counts.shape != (keys.shape[0],) or eligible.shape != counts.shape or
            bool(((counts < 0) | (counts > keys.shape[1])).any())):
        raise ValueError('Invalid attention block counts/mask')
    valid = (torch.arange(keys.shape[1], device=keys.device)[None] < counts[:, None]) & eligible[:, None]
    k = torch.where(valid[..., None], keys, 0.).float()
    v = torch.where(valid[..., None], values, 0.).float()
    if not bool(torch.isfinite(query).all() and torch.isfinite(k).all() and torch.isfinite(v).all()):
        raise ValueError('Nonfinite attention Q/K/V')
    logits = (query.float() @ k.flatten(0, 1).T).reshape(query.shape[0], *k.shape[:2])
    logits = logits.mul_(keys.shape[-1] ** -.5).double().masked_fill_(~valid[None], -torch.inf)
    log_mass = logits.logsumexp(-1)
    weights = (logits - log_mass[..., None]).exp().masked_fill_(~valid[None], 0.)
    # A block with zero valid tokens has log_mass=-inf and zero conditional moments.
    conditional = (torch.bmm(weights.permute(1, 0, 2).float(), v).permute(1, 0, 2)
                   if with_value_mean else v.new_empty((len(query),len(keys),0)))
    norms = (weights * v.double().abs().sum(-1)[None]).sum(-1)
    result = AttentionMoments(log_mass, conditional.double(), norms)
    return (AttentionMoments(result.log_mass.cpu(), result.mean_value.cpu(), result.mean_value_l1.cpu())
            if host else result)


def concatenate_moments(parts):
    return AttentionMoments(torch.cat([p.log_mass for p in parts], -1),
        torch.cat([p.mean_value for p in parts], 1), torch.cat([p.mean_value_l1 for p in parts], -1))


def block_scores(moments):
    """Evaluate the three reference scores per Query/layer/head, before aggregation.

    For p<=1/2 the requested formula is well conditioned. For a dominant block
    we form its complementary attention directly in log space and use the
    equivalent p*||mean_B-mean_complement||. This never subtracts 1-p or divides
    by a clamp, including when p has rounded to exactly one. Removing the only
    nonempty block is undefined and fails explicitly rather than inventing a score.
    """
    lm, means = moments.log_mass, moments.mean_value
    if lm.ndim != 2 or means.shape[:2] != lm.shape or moments.mean_value_l1.shape != lm.shape:
        raise ValueError('Invalid block attention moments')
    if bool((torch.isfinite(lm).sum(-1) < 2).any()):
        raise ValueError('Block deletion requires a nonempty complementary attention domain')
    p = lm.softmax(-1)
    output = (p[..., None] * means).sum(1)
    error = (means - output[:, None]).norm(dim=-1) * p / (1-p).clamp_min(.5)
    dominant = (p > .5).nonzero(as_tuple=False)
    for row, block in dominant.tolist():
        complement = lm[row].clone()
        complement[block] = -torch.inf
        without = (complement.softmax(-1)[:, None] * means[row]).sum(0)
        error[row, block] = p[row, block] * (means[row, block] - without).norm()
    result = dict(exact_mass=p, block_value_norm=p*moments.mean_value_l1,
                  block_deletion_error=error)
    if not all(bool(torch.isfinite(value).all() & (value >= 0).all()) for value in result.values()):
        raise ValueError('Nonfinite/negative reference block score')
    return result


def selected_set_error(moments, selected):
    """Recompute the kept set's normalized output; never sum single-block errors."""
    lm = moments.log_mass
    mask = torch.zeros(lm.shape[-1], dtype=torch.bool)
    selected = tuple(selected)
    if not selected or len(set(selected)) != len(selected) or any(i < 0 or i >= len(mask) for i in selected):
        raise ValueError('Selected attention set must contain valid distinct block indices')
    mask[list(selected)] = True
    kept = lm.masked_fill(~mask, -torch.inf)
    if not bool(torch.isfinite(kept).any(-1).all()):
        raise ValueError('Selected attention set has no valid tokens')
    full_output = (lm.softmax(-1)[..., None] * moments.mean_value).sum(1)
    kept_output = (kept.softmax(-1)[..., None] * moments.mean_value).sum(1)
    return (kept_output-full_output).norm(dim=-1)


def iter_attention_moments(queries, read_blocks, counts, eligible, *, kv_heads,
                           device='cpu', query_batch=4, block_batch=64):
    """Yield bounded layer/head/Query tiles; read_blocks returns dequantized K,V.

    read_blocks(begin,end,layer,kv_head) -> [block,token,dim] tensors. Query
    head grouping follows GQA; there is no averaging before nonlinear scores.
    """
    if (queries.ndim != 4 or not queries.is_floating_point() or kv_heads <= 0 or
            queries.shape[2] % kv_heads or not 1 <= query_batch <= 128 or
            not 1 <= block_batch <= 128 or len(counts) != len(eligible) or not counts):
        raise ValueError('Invalid reference attention geometry/tiling')
    if device == 'cuda' and (torch.get_float32_matmul_precision() != 'highest' or
                            torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('Reference attention requires FP32 matmul without TF32')
    layers, rows, heads, dim = queries.shape
    groups = heads//kv_heads
    for layer in range(layers):
        for kvhead in range(kv_heads):
            for start in range(0, rows, query_batch):
                q = queries[layer, start:start+query_batch, kvhead*groups:(kvhead+1)*groups].reshape(-1, dim).to(device)
                parts = []
                for begin in range(0, len(counts), block_batch):
                    end = min(begin+block_batch, len(counts))
                    k, v = read_blocks(begin, end, layer, kvhead)
                    parts.append(moments_from_blocks(q, k.to(device), v.to(device),
                        counts[begin:end], eligible[begin:end]))
                yield concatenate_moments(parts)


def aggregate_scores(iterator, *, layers, heads):
    totals = None
    dominant = 0
    for moments in iterator:
        values = block_scores(moments)
        if totals is None:
            totals = {m:torch.zeros(moments.log_mass.shape[-1], dtype=torch.float64) for m in REFERENCE_METHODS}
        dominant += int((values['exact_mass'] > 1-1e-8).sum())
        for method in totals:
            totals[method].add_(values[method].sum(0))
    if totals is None or min(layers, heads) <= 0:
        raise ValueError('No reference attention rows')
    return {m:(s/(layers*heads)).tolist() for m,s in totals.items()}, dict(
        dominant_mass_rows=dominant, deletion_stability='direct_logspace_complement_for_p_gt_half',
        query_reduction='sum', head_layer_reduction='mean', score_softmax_applied=False)


def aggregate_set_errors(iterator, selections, *, layers, heads):
    totals = {m:dict(sum_l2=0., maximum_l2=0., rows=0) for m in selections}
    for moments in iterator:
        for method, selected in selections.items():
            errors = selected_set_error(moments, selected)
            out = totals[method]
            out['sum_l2'] += float(errors.sum())
            out['maximum_l2'] = max(out['maximum_l2'], float(errors.max()))
            out['rows'] += errors.numel()
    for out in totals.values():
        if not out['rows']:
            raise ValueError('No selected-set diagnostic rows')
        out['mean_query_layer_head_l2'] = out['sum_l2']/out['rows']
        out['sum_query_mean_layer_head_l2'] = out['sum_l2']/(layers*heads)
    return totals
