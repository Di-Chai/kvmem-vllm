"""MeanK correctness oracle matching qw3 softmax-over-pages semantics.

Source algorithm: qw3 src/kernels_cuda.cu, block_attn_score_softmax_pages_kernel.
Queries/means are normalized content-frame tensors (before RoPE). Softmax runs
over eligible pages for each layer/query/head. Heads/layers are averaged, query
tokens summed. The native score frame supplies the exact softmax mask;
additional mandatory blocks remain in its domain.
"""
import math

import torch


def mean_key_scores(queries: torch.Tensor, means: torch.Tensor, eligible: torch.Tensor,
                    *, scale: float | None = None, query_chunk: int = 16) -> torch.Tensor:
    # queries [L,T,H,D], means [L,B,KVH,D], eligible [B].
    # CPU indexes are streamed one layer at a time to the query device.
    if queries.ndim != 4 or means.ndim != 4:
        raise ValueError("Expected LTHD queries and LBHD means")
    layers, tokens, heads, dim = queries.shape
    mlayers, blocks, kvheads, mdim = means.shape
    if (min(layers, tokens, heads, dim, blocks, kvheads) <= 0 or
            layers != mlayers or dim != mdim or heads % kvheads or
            type(query_chunk) is not int or query_chunk <= 0):
        raise ValueError("Incompatible GQA query/index shapes")
    if eligible.shape != (blocks,) or eligible.dtype != torch.bool:
        raise ValueError("Eligible mask must match generation-qualified index rows")
    if (eligible.device != queries.device or
            means.device != queries.device and means.device.type != 'cpu'):
        raise ValueError("Mask must share the Query device; index must be on CPU or that device")
    scale = dim ** -0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid model attention scale")
    score = torch.zeros(blocks, device=queries.device, dtype=torch.float32)
    # Preserve every block in each softmax domain. Only independent layers and
    # query rows are chunked. Packing on the source device avoids an all-layer
    # GPU index upload followed by einsum's hidden contiguous packing copy.
    group = heads // kvheads
    excluded = ~eligible
    for layer in range(layers):
        index = means[layer].permute(1, 2, 0).contiguous().to(
            device=queries.device, dtype=torch.float32)
        for begin in range(0, tokens, query_chunk):
            q = queries[layer, begin:begin+query_chunk].float()
            q = q.reshape(-1, kvheads, group, dim).permute(1, 0, 2, 3)
            q = q.reshape(kvheads, -1, dim).contiguous()
            logits = torch.bmm(q, index)
            logits.mul_(scale).masked_fill_(excluded, -torch.inf)
            # Fully masked rows have zero mass, never NaN propagated into selection.
            mass = torch.softmax(logits, dim=-1)
            mass.nan_to_num_(nan=0.0)
            score += mass.sum(dim=(0, 1)) / (layers * heads)
            del q, logits, mass
        # Do not retain the previous device slab while uploading the next layer.
        del index
    return score
