"""Sampled token-attention over the actual paged working K (not MeanK).

The main attention/compiled model is unchanged. Only sampled target Q rows are
replayed against its FP8 K, with the same dense RoPE, Q codec, scale and causal
visibility. Scratch is bounded by sampled rows * heads * active window.
"""
from contextlib import nullcontext
import torch
import triton
import triton.language as tl

from .codec import neox_rope


@triton.jit
def _paged_logits(Q, K, Table, Positions, Scale, Out,
                  QS: tl.constexpr, QH: tl.constexpr,
                  KP: tl.constexpr, KT: tl.constexpr, KH: tl.constexpr,
                  HEADS: tl.constexpr, GROUP: tl.constexpr, DIM: tl.constexpr,
                  PAGE: tl.constexpr, LENGTH, WIDTH: tl.constexpr,
                  ATTENTION_SCALE: tl.constexpr, TILE: tl.constexpr, D: tl.constexpr):
    row, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pos = tile*TILE + tl.arange(0,TILE)
    dim = tl.arange(0,D)
    causal = tl.load(Positions+row)
    valid = (pos < LENGTH) & (pos <= causal)
    # Native FP8 K/V arenas exceed 2 GiB with long concurrent windows.
    # Widen the physical page ID before multiplying by its byte stride.
    pages = tl.load(Table+pos//PAGE,mask=valid,other=0).to(tl.int64)
    keys = tl.load(K+pages[:,None]*KP+(pos%PAGE)[:,None]*KT+
                   (head//GROUP)*KH+dim[None,:],
                   mask=valid[:,None] & (dim[None,:]<DIM),other=0.0).to(tl.float32)
    query = tl.load(Q+row*QS+head*QH+dim,mask=dim<DIM,other=0.0).to(tl.float32)
    logits = tl.sum(keys*query[None,:],axis=1)*tl.load(Scale)*ATTENTION_SCALE
    tl.store(Out+(row*HEADS+head)*WIDTH+pos,
             tl.where(valid,logits,-float('inf')),mask=pos<WIDTH)


@triton.jit
def _segment_mass(P, Starts, Counts, Out, HEADS: tl.constexpr, WIDTH: tl.constexpr,
                  SEGMENTS, H: tl.constexpr, B: tl.constexpr):
    row, segment = tl.program_id(0), tl.program_id(1)
    head, offset = tl.arange(0,H),tl.arange(0,B)
    begin,count=tl.load(Starts+segment),tl.load(Counts+segment)
    p=tl.load(P+(row*HEADS+head[:,None])*WIDTH+begin+offset[None,:],
              mask=(head[:,None]<HEADS)&(offset[None,:]<count),other=0)
    tl.store(Out+row*SEGMENTS+segment,tl.sum(tl.sum(p,axis=1),axis=0)/HEADS)


def page_attention(raw_q, positions, key, table, cos_sin, key_scale,
                   starts, counts, length, *, scale, q_dtype, q_scale, block_tokens, profile=None):
    """Return [sampled rows, historical segments], averaged over Q heads."""
    if (raw_q.ndim!=3 or key.ndim!=4 or raw_q.shape[-1]!=key.shape[-1] or
            raw_q.shape[1]%key.shape[2] or not 0<length<=table.numel()*key.shape[1] or
            starts.numel()!=counts.numel() or positions.shape!=(raw_q.shape[0],) or
            q_dtype not in (torch.bfloat16,torch.float16,torch.float8_e4m3fn)):
        raise ValueError('Invalid sampled action attention geometry/codec')
    if raw_q.shape[0] > 64:
        # Sampling density must not depend on whether vLLM admitted an ordinary
        # chunk or a complete explicit Query. Only split independent Q rows;
        # every row still sees the entire original causal active context.
        return torch.cat([page_attention(raw_q[i:i+64],positions[i:i+64],key,table,
            cos_sin,key_scale,starts,counts,length,scale=scale,q_dtype=q_dtype,
            q_scale=q_scale,block_tokens=block_tokens,
            profile=profile.child(row_tile=i, sample_rows=min(64,raw_q.shape[0]-i)) if profile else None)
            for i in range(0,raw_q.shape[0],64)],dim=0)
    if profile:
        profile = profile.child(query_shape=list(raw_q.shape), key_shape=list(key.shape),
                                position_shape=list(positions.shape))
    with profile.stage('score.trig_gather', gpu=True) if profile else nullcontext():
        trig=cos_sin.index_select(0,positions.long())
    half=trig.shape[-1]//2
    with profile.stage('score.rope', gpu=True) if profile else nullcontext():
        q=neox_rope(raw_q,trig[:,:half],trig[:,half:])
    with profile.stage('score.q_codec', gpu=True, q_dtype=str(q_dtype)) if profile else nullcontext():
        if q_dtype==torch.float8_e4m3fn:
            # Match native scaled_fp8_quant, then carry its scale into the logits.
            q=(q.float()/q_scale).clamp(-448,448).to(q_dtype)
            effective_scale=key_scale*q_scale
        else:
            q=q.to(q_dtype)
            effective_scale=key_scale
    rows,heads,dim=q.shape
    # Length/segment count change every accepted step. Specializing on either
    # would trigger JIT compilation during decode. Only bucket scratch width.
    width=max(32,triton.next_power_of_2(length))
    if profile:
        profile = profile.child(sample_rows=rows, heads=heads, head_dim=dim, width=width,
                                active_tokens=length, segments=starts.numel())
    with profile.stage('score.logits_allocate', allocation_bytes=rows*heads*width*4) if profile else nullcontext():
        logits=torch.empty((rows,heads,width),device=q.device,dtype=torch.float32)
    with profile.stage('score.logits', gpu=True, grid=[rows,heads,triton.cdiv(width,32)]) if profile else nullcontext():
        _paged_logits[(rows,heads,triton.cdiv(width,32))](q,key,table,positions,effective_scale,logits,
            q.stride(0),q.stride(1),*key.stride()[:3],heads,heads//key.shape[2],dim,
            key.shape[1],length,width,scale,32,triton.next_power_of_2(dim),enable_fp_fusion=False)
    with profile.stage('score.softmax', gpu=True, allocation_bytes=rows*heads*width*4) if profile else nullcontext():
        probabilities=logits.softmax(-1)
    with profile.stage('score.mass_allocate', allocation_bytes=rows*starts.numel()*4) if profile else nullcontext():
        mass=torch.empty((rows,starts.numel()),device=q.device,dtype=torch.float32)
    with profile.stage('score.segment_mass', gpu=True) if profile else nullcontext():
        _segment_mass[(rows,starts.numel())](probabilities,starts,counts,mass,heads,width,
            starts.numel(),triton.next_power_of_2(heads),triton.next_power_of_2(block_tokens))
    return mass
