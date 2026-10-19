"""FP32 sampled QK with compact segment logsumexp, no token-sized matrices."""
import torch
import triton
import triton.language as tl


@triton.jit
def _segment_lse(Q,K,Table,Positions,Scale,Starts,Counts,LSE,
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
    center=tl.where(center==-float('inf'),0.,center)
    total=tl.sum(tl.exp(logits-center),axis=0)
    tl.store(LSE+(row*HEADS+head)*SEGMENTS+segment,center+tl.log(total))


@triton.jit
def _row_lse(LSE,Norm,HEADS:tl.constexpr,SEGMENTS,B:tl.constexpr):
    row,head=tl.program_id(0),tl.program_id(1)
    i=tl.arange(0,B)
    values=tl.load(LSE+(row*HEADS+head)*SEGMENTS+i,mask=i<SEGMENTS,other=-float('inf'))
    center=tl.max(values,axis=0)
    value=center+tl.log(tl.sum(tl.exp(values-center),axis=0))
    tl.store(Norm+row*HEADS+head,value)


@triton.jit
def _mass(LSE,Norm,Out,HEADS:tl.constexpr,SEGMENTS,H:tl.constexpr):
    row,segment=tl.program_id(0),tl.program_id(1)
    h=tl.arange(0,H)
    lse=tl.load(LSE+(row*HEADS+h)*SEGMENTS+segment,mask=h<HEADS,other=-float('inf'))
    norm=tl.load(Norm+row*HEADS+h,mask=h<HEADS,other=0.)
    tl.store(Out+row*SEGMENTS+segment,tl.sum(tl.exp(lse-norm),axis=0)/HEADS)


def fused_page_attention(q,positions,key,table,effective_scale,starts,counts,length,*,scale,block_tokens):
    """Caller guarantees segments partition the complete visible key domain."""
    rows,heads,dim=q.shape;segments=starts.numel()
    lse=torch.empty((rows,heads,segments),device=q.device,dtype=torch.float32)
    norm=torch.empty((rows,heads),device=q.device,dtype=torch.float32)
    mass=torch.empty((rows,segments),device=q.device,dtype=torch.float32)
    _segment_lse[(rows,heads,segments)](q,key,table,positions,effective_scale,starts,counts,lse,
        q.stride(0),q.stride(1),*key.stride()[:3],heads,heads//key.shape[2],dim,key.shape[1],
        length,segments,scale,triton.next_power_of_2(block_tokens),triton.next_power_of_2(dim),
        enable_fp_fusion=False)
    _row_lse[(rows,heads)](lse,norm,heads,segments,triton.next_power_of_2(segments),enable_fp_fusion=False)
    _mass[(rows,segments)](lse,norm,mass,heads,segments,triton.next_power_of_2(heads),enable_fp_fusion=False)
    mass._action_scoring_backend='fused'
    return mass
