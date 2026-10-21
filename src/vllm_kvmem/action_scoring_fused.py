"""FP32 sampled QK with shifted segment sums, no token-sized matrices."""
import torch
import triton
import triton.language as tl


@triton.jit
def _segment_stats(Q,K,Table,Positions,Scale,Starts,Counts,Maxima,Sums,
                 QS:tl.constexpr,QH:tl.constexpr,KP:tl.constexpr,KT:tl.constexpr,KH:tl.constexpr,
                 HEADS:tl.constexpr,GROUP:tl.constexpr,DIM:tl.constexpr,PAGE:tl.constexpr,
                 LENGTH,SEGMENTS,ATTENTION_SCALE:tl.constexpr,B:tl.constexpr,D:tl.constexpr):
    row,head,segment=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    offset,dim=tl.arange(0,B),tl.arange(0,D)
    begin,count=tl.load(Starts+segment),tl.load(Counts+segment)
    pos=begin+offset
    valid=(offset<count)&(pos<LENGTH)&(pos<=tl.load(Positions+row))
    pages=tl.load(Table+pos//PAGE,mask=valid,other=0).to(tl.int64)
    keys=tl.load(K+pages[:,None]*KP+(pos%PAGE)[:,None]*KT+(head//GROUP)*KH+dim[None,:],
        mask=valid[:,None]&(dim[None,:]<DIM),other=0.).to(tl.float32)
    query=tl.load(Q+row*QS+head*QH+dim,mask=dim<DIM,other=0.).to(tl.float32)
    # Match the reference FP32 point product/reduction and scale order. This
    # is the existing bypass QK, not native FlashInfer/XQA accumulator reuse.
    logits=tl.sum(keys*query[None,:],axis=1)*tl.load(Scale)*ATTENTION_SCALE
    logits=tl.where(valid,logits,-float('inf'))
    center=tl.max(logits,axis=0)
    # Keep the maximum separate: adding log(sum) to a large FP32 maximum
    # discards normalization bits that subtraction cannot recover later.
    shift=tl.where(center==-float('inf'),0.,center)
    total=tl.sum(tl.exp(logits-shift),axis=0)
    index=(row*HEADS+head)*SEGMENTS+segment
    # Empty causal segments must retain -inf, not zero: zero could dominate
    # the row maximum when every visible logit is large and negative.
    tl.store(Maxima+index,center)
    tl.store(Sums+index,total)


@triton.jit
def _normalize_segments(Maxima,Sums,HEADS:tl.constexpr,SEGMENTS,B:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    i=tl.arange(0,B)
    index=(row*HEADS+head)*SEGMENTS+i
    maxima=tl.load(Maxima+index,mask=i<SEGMENTS,other=-float('inf'))
    sums=tl.load(Sums+index,mask=i<SEGMENTS,other=0.)
    center=tl.max(maxima,axis=0)
    shifted=sums*tl.exp(maxima-center)
    total=tl.sum(shifted,axis=0)
    # Each program owns a complete row/head. Reuse its sum buffer for the
    # normalized masses; the following kernel only averages independent heads.
    tl.store(Sums+index,shifted/total,mask=i<SEGMENTS)


@triton.jit
def _mass(Sums,Out,HEADS:tl.constexpr,SEGMENTS,H:tl.constexpr):
    row,segment=tl.program_id(0),tl.program_id(1)
    h=tl.arange(0,H)
    index=(row*HEADS+h)*SEGMENTS+segment
    probability=tl.load(Sums+index,mask=h<HEADS,other=0.)
    tl.store(Out+row*SEGMENTS+segment,tl.sum(probability,axis=0)/HEADS)


def fused_page_attention(q,positions,key,table,effective_scale,starts,counts,length,*,scale,block_tokens):
    """Caller guarantees segments partition the complete visible key domain.

    Each head uses one common maximum and denominator across all segments.
    Reduction rounding can still differ from the reference token softmax;
    near-tied graph edges are not guaranteed to have identical ordering.
    """
    rows,heads,dim=q.shape;segments=starts.numel()
    maxima=torch.empty((rows,heads,segments),device=q.device,dtype=torch.float32)
    sums=torch.empty_like(maxima)
    mass=torch.empty((rows,segments),device=q.device,dtype=torch.float32)
    _segment_stats[(rows,heads,segments)](q,key,table,positions,effective_scale,starts,counts,maxima,sums,
        q.stride(0),q.stride(1),*key.stride()[:3],heads,heads//key.shape[2],dim,key.shape[1],
        length,segments,scale,triton.next_power_of_2(block_tokens),triton.next_power_of_2(dim),
        enable_fp_fusion=False)
    _normalize_segments[(rows,heads)](maxima,sums,heads,segments,triton.next_power_of_2(segments),enable_fp_fusion=False)
    _mass[(rows,segments)](sums,mass,heads,segments,triton.next_power_of_2(heads),enable_fp_fusion=False)
    mass._action_scoring_backend='fused'
    return mass
