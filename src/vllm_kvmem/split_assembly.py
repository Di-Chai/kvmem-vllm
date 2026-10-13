"""Immutable raw K and independently sourced V into native FP8 pages.

Resident V is staged before any destination writes, so cross-block permutations
cannot overwrite a later source. Raw-K hits keep their original BF16 bytes; K
always uses the original RoPE and FP8 rounding sequence at its new dense position.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _gather_v(source, bases, pages, output,
              S0:tl.constexpr, S1:tl.constexpr, S2:tl.constexpr,
              PAGE:tl.constexpr, TOKENS:tl.constexpr, LAYERS:tl.constexpr,
              LAYER:tl.constexpr, HEADS:tl.constexpr, DIM:tl.constexpr, BLOCK:tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    record, offset = token//TOKENS, token%TOKENS
    position = tl.load(bases+record).to(tl.int64) + offset
    page = tl.load(pages+position//PAGE).to(tl.int64)
    d = tl.arange(0,BLOCK)
    value = tl.load(source+page*S0+(position%PAGE)*S1+head*S2+d, mask=d<DIM, other=0)
    dst = ((record*LAYERS+LAYER)*TOKENS+offset)*HEADS*DIM+head*DIM+d
    tl.store(output+dst,value,mask=d<DIM)


def gather_resident_values(layers, from_bases, block_tokens,destination=None):
    first=layers[0]
    heads,dim=first.value.shape[-2:]
    shape=(from_bases.numel(),len(layers),block_tokens,heads,dim)
    output=(torch.empty(shape,dtype=torch.uint8,device=first.value.device) if destination is None
            else destination.view(shape))
    for i,layer in enumerate(layers):
        _gather_v[(from_bases.numel()*block_tokens,heads)](
            layer.value.view(torch.uint8),from_bases,layer.kernel_page_table,output,
            *layer.value.stride()[:3],layer.value.shape[1],block_tokens,len(layers),i,
            heads,dim,triton.next_power_of_2(dim),num_warps=4)
    return output


@triton.jit
def _store_raw(payload, slots, cache, E:tl.constexpr, SRC_BYTES:tl.constexpr, BLOCK:tl.constexpr):
    record=tl.program_id(0)
    slot=tl.load(slots+record).to(tl.int64)
    if slot>=0:
        x=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        source=payload.to(tl.pointer_type(tl.bfloat16))
        values=tl.load(source+record*(SRC_BYTES//2)+x,mask=x<E,other=0)
        tl.store(cache+slot*E+x,values,mask=x<E)


def store_raw_records(payload, slots, cache, source_record_bytes):
    elements=cache[0].numel()
    _store_raw[(slots.numel(),triton.cdiv(elements,4096))](
        payload,slots,cache,elements,source_record_bytes,4096,num_warps=4)


@triton.jit
def _assemble_split(raw_payload, raw_indices, values, value_indices, bases,
                    pages,trig,key,value,scale,
                    K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                    V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
                    RAW_BYTES:tl.constexpr,VALUE_BYTES:tl.constexpr,VALUE_OFFSET:tl.constexpr,
                    TRIG:tl.constexpr,TOKENS:tl.constexpr,LAYERS:tl.constexpr,LAYER:tl.constexpr,
                    HEADS:tl.constexpr,DIM:tl.constexpr,HALF:tl.constexpr,PAGE:tl.constexpr,
                    BLOCK:tl.constexpr):
    token,head=tl.program_id(0),tl.program_id(1)
    record,offset=token//TOKENS,token%TOKENS
    position=tl.load(bases+record).to(tl.int64)+offset
    page=tl.load(pages+position//PAGE).to(tl.int64)
    raw_record=tl.load(raw_indices+record).to(tl.int64)
    value_record=tl.load(value_indices+record).to(tl.int64)
    layer_elements:tl.constexpr=TOKENS*HEADS*DIM
    inner=LAYER*layer_elements+offset*HEADS*DIM+head*DIM
    d=tl.arange(0,BLOCK)
    raw=raw_payload.to(tl.pointer_type(tl.bfloat16))
    base=raw_record*(RAW_BYTES//2)+inner
    x=tl.load(raw+base+d,mask=d<DIM,other=0).to(tl.float32)
    mate_d=tl.where(d<HALF,d+HALF,d-HALF)
    mate=tl.load(raw+base+mate_d,mask=d<2*HALF,other=0).to(tl.float32)
    freq=d%HALF
    cos=tl.load(trig+position*TRIG+freq).to(tl.float32)
    sin=tl.load(trig+position*TRIG+HALF+freq).to(tl.float32)
    rotated=tl.where(d<HALF,x*cos-mate*sin,x*cos+mate*sin)
    rotated=tl.where(d<2*HALF,rotated,x).to(tl.bfloat16).to(tl.float32)
    quantized=tl.minimum(tl.maximum(rotated/tl.load(scale),-448.0),448.0)
    tl.store(key+page*K0+(position%PAGE)*K1+head*K2+d,quantized,mask=d<DIM)
    v=tl.load(values+value_record*VALUE_BYTES+VALUE_OFFSET+inner+d,mask=d<DIM,other=0)
    tl.store(value.to(tl.pointer_type(tl.uint8))+page*V0+(position%PAGE)*V1+head*V2+d,v,mask=d<DIM)


def assemble_split_layer(raw, raw_indices, values, value_indices, bases, layer, *,
                         layer_index,layers,block_tokens,raw_record_bytes,
                         value_record_bytes,value_offset=0):
    key,value=layer.key,layer.value
    heads,dim=key.shape[-2:]
    _assemble_split[(bases.numel()*block_tokens,heads)](
        raw,raw_indices,values,value_indices,bases,layer.kernel_page_table,
        layer.cos_sin,key,value,layer.key_scale,*key.stride()[:3],*value.stride()[:3],
        raw_record_bytes,value_record_bytes,value_offset,layer.cos_sin.stride(0),block_tokens,
        layers,layer_index,heads,dim,layer.cos_sin.shape[1]//2,key.shape[1],
        triton.next_power_of_2(dim),num_warps=4)
