"""MeanK correctness oracle matching qw3 softmax-over-pages semantics.

Source algorithm: qw3 src/kernels_cuda.cu, block_attn_score_softmax_pages_kernel.
Queries/means are normalized content-frame tensors (before RoPE). Softmax runs
over eligible pages for each layer/query/head. Heads/layers are averaged, query
tokens summed. Always-kept bands are excluded before normalization.
"""
import math

import torch


def mean_key_scores(queries: torch.Tensor, means: torch.Tensor, eligible: torch.Tensor,
                    *, scale: float | None = None, query_chunk: int = 16) -> torch.Tensor:
    # queries [L,T,H,D], means [L,B,KVH,D], eligible [B].
    if queries.ndim != 4 or means.ndim != 4:
        raise ValueError("Expected LTHD queries and LBHD means")
    layers, tokens, heads, dim = queries.shape
    mlayers, blocks, kvheads, mdim = means.shape
    if (min(layers, tokens, heads, dim, blocks, kvheads) <= 0 or
            layers != mlayers or dim != mdim or heads % kvheads or query_chunk <= 0):
        raise ValueError("Incompatible GQA query/index shapes")
    if eligible.shape != (blocks,) or eligible.dtype != torch.bool:
        raise ValueError("Eligible mask must match generation-qualified index rows")
    if queries.device != means.device or eligible.device != means.device:
        raise ValueError("Queries/index/mask must share a device")
    scale = dim ** -0.5 if scale is None else scale
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid model attention scale")
    score = torch.zeros(blocks, device=queries.device, dtype=torch.float32)
    # Bounded temporary logits. Explicit groups avoid materializing repeated KV heads.
    index = means.float().permute(0, 2, 3, 1)
    group = heads // kvheads
    for begin in range(0, tokens, query_chunk):
        q = queries[:, begin:begin+query_chunk].float()
        q = q.reshape(layers, -1, kvheads, group, dim)
        logits = torch.einsum("ltkgd,lkdb->ltkgb", q, index) * scale
        logits = logits.masked_fill(~eligible, -torch.inf)
        # Fully masked rows have zero mass, never NaN propagated into selection.
        mass = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
        score += mass.sum(dim=(0, 1, 2, 3)) / (layers * heads)
    return score
