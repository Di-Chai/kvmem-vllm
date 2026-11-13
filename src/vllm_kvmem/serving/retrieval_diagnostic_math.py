"""Operand-controlled content-frame probes, distinct from native RoPE attention."""
import math

import numpy as np
import torch


def content_distributions(query, keys, counts, eligible, *, means=None, device='cpu', tile_pages=128):
    """Return per-Query-row, Q-head-mean probabilities over original pages.

    All methods share the complete supplied eligible history. ``token`` uses
    per-page logsumexp and a global token softmax; ``mean`` is the MeanK proxy;
    ``token_max`` takes the largest globally normalized token probability
    per page. ``max`` instead normalizes max logits across pages. Neither is
    graph max pooling.
    Keys are flattened [N,KVH,D], optionally FP8 bytes from a read-only mmap.
    """
    if query.ndim != 3 or len(keys.shape)!=3 or query.shape[-1]!=keys.shape[-1]:
        raise ValueError('Invalid diagnostic Q/K geometry')
    rows, heads, dim = query.shape
    kvheads=keys.shape[1]
    if heads%kvheads or len(counts)!=len(eligible) or sum(counts)!=len(keys):
        raise ValueError('Invalid GQA or original page partition')
    if not counts or min(counts)<=0 or max(counts)>32 or not any(eligible):
        raise ValueError('Invalid diagnostic visible score domain')
    q=query.to(device=device,dtype=torch.float16).float().reshape(rows,kvheads,heads//kvheads,dim)
    pages=len(counts)
    logs={name:torch.empty((rows,kvheads,heads//kvheads,pages),device=device) for name in ('mean','max','token')}
    token=0
    for start in range(0,pages,tile_pages):
        ns=counts[start:start+tile_pages];n=sum(ns);b=len(ns)
        source=keys[token:token+n]
        source=(torch.from_numpy(np.array(source,copy=True)) if isinstance(source,np.ndarray) else source)
        if source.dtype==torch.uint8:source=source.view(torch.float8_e4m3fn)
        source=source.to(device=device,dtype=torch.float32)
        if all(c==32 for c in ns):
            padded=source.reshape(b,32,kvheads,dim)
        elif all(c==32 for c in ns[:-1]):
            padded=torch.zeros((b*32,kvheads,dim),device=device)
            padded[:n]=source
            padded=padded.reshape(b,32,kvheads,dim)
        else:
            padded=torch.zeros((b,32,kvheads,dim),device=device)
            off=0
            for i,c in enumerate(ns):padded[i,:c]=source[off:off+c];off+=c
        z=torch.einsum('tkgd,bskd->tkgbs',q,padded)*dim**-.5
        valid=torch.arange(32,device=device)[None,:]<torch.tensor(ns,device=device)[:,None]
        z=z.masked_fill(~valid,-torch.inf)
        logs['max'][...,start:start+b]=z.amax(-1)
        logs['token'][...,start:start+b]=z.logsumexp(-1)
        mu=(padded.sum(1)/torch.tensor(ns,device=device)[:,None,None]
            if means is None else means[start:start+b].to(device))
        mu=mu.to(torch.float16).float()
        logs['mean'][...,start:start+b]=torch.einsum('tkgd,bkd->tkgb',q,mu)*dim**-.5
        token+=n
    mask=torch.tensor(eligible,device=device,dtype=torch.bool)
    masked={name:value.masked_fill(~mask,-torch.inf) for name,value in logs.items()}
    result={name:value.softmax(-1).mean((1,2)).cpu() for name,value in masked.items()}
    denominator=masked['token'].logsumexp(-1,keepdim=True)
    result['token_max']=(masked['max']-denominator).exp().mean((1,2)).cpu()
    return result
