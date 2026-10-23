"""Immutable raw K and independently sourced V into native FP8 pages.

Resident V is staged before any destination writes, so cross-block permutations
cannot overwrite a later source. Raw-K hits keep their original source bytes; K
always uses the original RoPE and FP8 rounding sequence at its new dense position.
"""
import torch
from dataclasses import dataclass
from vllm.triton_utils import tl, triton
from .fp8_compat import fp8_storage, software_fp8, fp8_to_float, float_to_fp8


@dataclass(frozen=True)
class FusedLayerGeometry:
    heads: int
    dim: int
    half: int
    page: int
    table_dtype: object
    trig_dtype: object


@dataclass(frozen=True)
class FusedLayerSpec:
    rows: tuple
    signature: tuple
    geometry: FusedLayerGeometry
    owners: tuple


def _positive_nonoverlap(tensor, *, inner_width=None):
    if any(size <= 0 for size in tensor.shape) or any(step <= 0 for step in tensor.stride()):
        return False
    if inner_width is None:
        occupied = 1
        dimensions = zip(tensor.shape, tensor.stride())
    else:
        occupied = inner_width
        dimensions = zip(tensor.shape[:-1], tensor.stride()[:-1])
    for size, step in sorted(dimensions, key=lambda pair: pair[1]):
        if size > 1 and step < occupied:
            return False
        occupied += (size-1)*step
    return True


def _byte_interval(tensor):
    begin = tensor.data_ptr()
    extent = (1+sum((size-1)*step for size,step in zip(tensor.shape,tensor.stride()))) * tensor.element_size()
    return begin, begin+extent


def _intersects(left, right):
    return left[0] < right[1] and right[0] < left[1]


def prepare_fused_layer_rows(layers):
    """Prove geometry/alias eligibility using host metadata only, before writes."""
    if not layers:
        return None, 'empty'
    rows, signatures, owners, writes, reads = [], [], [], [], []
    geometry = None
    device = layers[0].key.device
    for layer in layers:
        key,value,pages,trig,scale = (layer.key,layer.value,layer.kernel_page_table,
                                      layer.cos_sin,layer.key_scale)
        tensors = pages,trig,key,value,scale
        if device.type != 'cuda' or any(t.device != device for t in tensors):
            return None, 'device'
        if (key.dtype != torch.float8_e4m3fn or value.dtype != key.dtype or
                scale.dtype != torch.float32 or pages.dtype not in (torch.int32,torch.int64) or
                trig.dtype not in (torch.float16,torch.bfloat16,torch.float32)):
            return None, 'dtype'
        if (key.ndim != 4 or value.ndim != 4 or key.shape != value.shape or
                pages.ndim != 1 or trig.ndim != 2 or scale.numel() != 1 or
                not 0 < trig.shape[1] <= key.shape[-1] or trig.shape[1]%2):
            return None, 'geometry'
        if (key.stride(-1) != 1 or value.stride(-1) != 1 or pages.stride(0) != 1 or
                trig.stride(-1) != 1 or any(not _positive_nonoverlap(t) for t in tensors)):
            return None, 'strides'
        current = FusedLayerGeometry(key.shape[-2],key.shape[-1],trig.shape[1]//2,
                                     key.shape[1],pages.dtype,trig.dtype)
        if geometry is not None and geometry != current:
            return None, 'geometry'
        geometry = current
        pair = _byte_interval(key),_byte_interval(value)
        if _intersects(*pair):
            # Canonical interleaved K/V views share a containing byte interval,
            # but each 2D cell contains disjoint D-byte key and value halves.
            if (key.stride() != value.stride() or abs(key.data_ptr()-value.data_ptr()) != current.dim or
                    not _positive_nonoverlap(key,inner_width=2*current.dim)):
                return None, 'alias'
        if any(_intersects(a,b) for earlier in writes for a in earlier for b in pair):
            return None, 'alias'
        writes.append(pair)
        reads.extend(_byte_interval(t) for t in (pages,trig,scale))
        rows.append(tuple(t.data_ptr() for t in tensors)+key.stride()[:3]+value.stride()[:3]+(trig.stride(0),))
        signatures.append(tuple((t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors))
        owners.extend(tensors)
    if any(_intersects(read,write) for read in reads for pair in writes for write in pair):
        return None, 'alias'
    return FusedLayerSpec(tuple(rows),tuple(signatures),geometry,tuple(owners)), None


@triton.jit
def _gather_v(source, bases, pages, output,
              S0:tl.constexpr, S1:tl.constexpr, S2:tl.constexpr,
              PAGE:tl.constexpr, TOKENS:tl.constexpr, LAYERS:tl.constexpr,
              LAYER:tl.constexpr, HEADS:tl.constexpr, DIM:tl.constexpr, BLOCK:tl.constexpr):
    # The interleaved V snapshot exceeds 2 GiB at K=128K with 17 layers.
    # Promote before multiplying; casting the final offset would be too late.
    token, head = tl.program_id(0).to(tl.int64), tl.program_id(1)
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
def _store_raw(payload, slots, cache, E:tl.constexpr, SRC_BYTES:tl.constexpr,
               RAW_BYTES:tl.constexpr, RAW_TYPE:tl.constexpr, BLOCK:tl.constexpr):
    record=tl.program_id(0)
    slot=tl.load(slots+record).to(tl.int64)
    if slot>=0:
        x=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
        source=payload.to(tl.pointer_type(RAW_TYPE))
        values=tl.load(source+record*(SRC_BYTES//RAW_BYTES)+x,mask=x<E,other=0.0)
        tl.store(cache+slot*E+x,values,mask=x<E)


def store_raw_records(payload, slots, cache, source_record_bytes):
    elements=cache[0].numel()
    cache_input,software=fp8_storage(cache)
    _store_raw[(slots.numel(),triton.cdiv(elements,4096))](
        payload,slots,cache_input,elements,source_record_bytes,cache.element_size(),
        ((tl.uint8 if software else tl.float8e4nv) if cache.dtype==torch.float8_e4m3fn else
         tl.bfloat16 if cache.dtype==torch.bfloat16 else tl.float16),4096,num_warps=4)


@triton.jit
def _assemble_split(raw_payload, raw_indices, values, value_indices, bases,
                    pages,trig,key,value,scale,
                    K0:tl.constexpr,K1:tl.constexpr,K2:tl.constexpr,
                    V0:tl.constexpr,V1:tl.constexpr,V2:tl.constexpr,
                    RAW_BYTES:tl.constexpr,VALUE_BYTES:tl.constexpr,VALUE_OFFSET:tl.constexpr,
                    TRIG:tl.constexpr,TOKENS:tl.constexpr,LAYERS:tl.constexpr,LAYER:tl.constexpr,
                    HEADS:tl.constexpr,DIM:tl.constexpr,HALF:tl.constexpr,PAGE:tl.constexpr,
                    BLOCK:tl.constexpr,RAW_TYPE:tl.constexpr,
                    RAW_ELEMENT_BYTES:tl.constexpr,ROPE_TYPE:tl.constexpr,
                    SOFTWARE_FP8:tl.constexpr):
    token,head=tl.program_id(0),tl.program_id(1)
    record,offset=token//TOKENS,token%TOKENS
    position=tl.load(bases+record).to(tl.int64)+offset
    page=tl.load(pages+position//PAGE).to(tl.int64)
    raw_record=tl.load(raw_indices+record).to(tl.int64)
    value_record=tl.load(value_indices+record).to(tl.int64)
    layer_elements:tl.constexpr=TOKENS*HEADS*DIM
    inner=LAYER*layer_elements+offset*HEADS*DIM+head*DIM
    d=tl.arange(0,BLOCK)
    raw=raw_payload.to(tl.pointer_type(RAW_TYPE))
    base=raw_record*(RAW_BYTES//RAW_ELEMENT_BYTES)+inner
    x=fp8_to_float(tl.load(raw+base+d,mask=d<DIM,other=0.0),SOFTWARE_FP8 and RAW_ELEMENT_BYTES == 1)
    mate_d=tl.where(d<HALF,d+HALF,d-HALF)
    mate=fp8_to_float(tl.load(raw+base+mate_d,mask=d<2*HALF,other=0.0),SOFTWARE_FP8 and RAW_ELEMENT_BYTES == 1)
    freq=d%HALF
    cos=tl.load(trig+position*TRIG+freq).to(tl.float32)
    sin=tl.load(trig+position*TRIG+HALF+freq).to(tl.float32)
    rotated=tl.where(d<HALF,x*cos-mate*sin,x*cos+mate*sin)
    rotated=tl.where(d<2*HALF,rotated,x).to(ROPE_TYPE).to(tl.float32)
    if RAW_ELEMENT_BYTES == 1:
        scaled=tl.div_rn(rotated,tl.load(scale))
    else:
        scaled=rotated/tl.load(scale)
    quantized=tl.minimum(tl.maximum(scaled,-448.0),448.0)
    if SOFTWARE_FP8:
        quantized=float_to_fp8(quantized)
    tl.store(key+page*K0+(position%PAGE)*K1+head*K2+d,quantized,mask=d<DIM)
    v=tl.load(values+value_record*VALUE_BYTES+VALUE_OFFSET+inner+d,mask=d<DIM,other=0)
    tl.store(value.to(tl.pointer_type(tl.uint8))+page*V0+(position%PAGE)*V1+head*V2+d,v,mask=d<DIM)


def assemble_split_layer(raw, raw_indices, values, value_indices, bases, layer, *,
                         layer_index,layers,block_tokens,raw_record_bytes,
                         value_record_bytes,value_offset=0,raw_dtype=torch.bfloat16):
    key,value=layer.key,layer.value
    heads,dim=key.shape[-2:]
    key_input,software=fp8_storage(key)
    value_input,_=fp8_storage(value)
    raw_input,_=fp8_storage(raw)
    values_input,_=fp8_storage(values)
    _assemble_split[(bases.numel()*block_tokens,heads)](
        raw_input,raw_indices,values_input,value_indices,bases,layer.kernel_page_table,
        layer.cos_sin,key_input,value_input,layer.key_scale,*key.stride()[:3],*value.stride()[:3],
        raw_record_bytes,value_record_bytes,value_offset,layer.cos_sin.stride(0),block_tokens,
        layers,layer_index,heads,dim,layer.cos_sin.shape[1]//2,key.shape[1],
        triton.next_power_of_2(dim),
        ((tl.uint8 if software else tl.float8e4nv) if raw_dtype==torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype==torch.bfloat16 else tl.float16),
        raw_dtype.itemsize,
        (tl.float32 if raw_dtype==torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype==torch.bfloat16 else tl.float16),software,num_warps=4,
        enable_fp_fusion=raw_dtype != torch.float8_e4m3fn)


@triton.jit
def _assemble_split_layers(raw_payload, raw_indices, values, value_indices, bases, table,
                    RAW_BYTES:tl.constexpr,VALUE_BYTES:tl.constexpr,VALUE_OFFSET:tl.constexpr,
                    TOKENS:tl.constexpr,LAYERS:tl.constexpr,
                    HEADS:tl.constexpr,DIM:tl.constexpr,HALF:tl.constexpr,PAGE:tl.constexpr,
                    BLOCK:tl.constexpr,RAW_TYPE:tl.constexpr,
                    RAW_ELEMENT_BYTES:tl.constexpr,ROPE_TYPE:tl.constexpr,
                    TABLE_TYPE:tl.constexpr,TRIG_TYPE:tl.constexpr,
                    SOFTWARE_FP8:tl.constexpr):
    token,head,layer=tl.program_id(0).to(tl.int64),tl.program_id(1),tl.program_id(2).to(tl.int64)
    row=table+layer*12
    pages=tl.load(row+0).to(tl.pointer_type(TABLE_TYPE))
    trig=tl.load(row+1).to(tl.pointer_type(TRIG_TYPE))
    if SOFTWARE_FP8:
        key=tl.load(row+2).to(tl.pointer_type(tl.uint8))
    else:
        key=tl.load(row+2).to(tl.pointer_type(tl.float8e4nv))
    value=tl.load(row+3).to(tl.pointer_type(tl.uint8))
    scale=tl.load(row+4).to(tl.pointer_type(tl.float32))
    K0,K1,K2=tl.load(row+5),tl.load(row+6),tl.load(row+7)
    V0,V1,V2=tl.load(row+8),tl.load(row+9),tl.load(row+10)
    TRIG=tl.load(row+11)
    record,offset=token//TOKENS,token%TOKENS
    position=tl.load(bases+record).to(tl.int64)+offset
    page=tl.load(pages+position//PAGE).to(tl.int64)
    raw_record=tl.load(raw_indices+record).to(tl.int64)
    value_record=tl.load(value_indices+record).to(tl.int64)
    layer_elements:tl.constexpr=TOKENS*HEADS*DIM
    inner=layer*layer_elements+offset*HEADS*DIM+head*DIM
    d=tl.arange(0,BLOCK)
    raw=raw_payload.to(tl.pointer_type(RAW_TYPE))
    base=raw_record*(RAW_BYTES//RAW_ELEMENT_BYTES)+inner
    x=fp8_to_float(tl.load(raw+base+d,mask=d<DIM,other=0.0),SOFTWARE_FP8 and RAW_ELEMENT_BYTES == 1)
    mate_d=tl.where(d<HALF,d+HALF,d-HALF)
    mate=fp8_to_float(tl.load(raw+base+mate_d,mask=d<2*HALF,other=0.0),SOFTWARE_FP8 and RAW_ELEMENT_BYTES == 1)
    freq=d%HALF
    cos=tl.load(trig+position*TRIG+freq).to(tl.float32)
    sin=tl.load(trig+position*TRIG+HALF+freq).to(tl.float32)
    rotated=tl.where(d<HALF,x*cos-mate*sin,x*cos+mate*sin)
    rotated=tl.where(d<2*HALF,rotated,x).to(ROPE_TYPE).to(tl.float32)
    if RAW_ELEMENT_BYTES == 1:
        scaled=tl.div_rn(rotated,tl.load(scale))
    else:
        scaled=rotated/tl.load(scale)
    quantized=tl.minimum(tl.maximum(scaled,-448.0),448.0)
    if SOFTWARE_FP8:
        quantized=float_to_fp8(quantized)
    tl.store(key+page*K0+(position%PAGE)*K1+head*K2+d,quantized,mask=d<DIM)
    v=tl.load(values+value_record*VALUE_BYTES+VALUE_OFFSET+inner+d,mask=d<DIM,other=0)
    tl.store(value+page*V0+(position%PAGE)*V1+head*V2+d,v,mask=d<DIM)


def assemble_split_layers(raw,raw_indices,values,value_indices,bases,table,geometry,*,
                          layers,block_tokens,raw_record_bytes,value_record_bytes,
                          value_offset=0,raw_dtype=torch.bfloat16):
    """Launch a prevalidated whole layer bundle; launch errors never fall back."""
    software=software_fp8(raw.device)
    raw_input,_=fp8_storage(raw)
    values_input,_=fp8_storage(values)
    _assemble_split_layers[(bases.numel()*block_tokens,geometry.heads,layers)](
        raw_input,raw_indices,values_input,value_indices,bases,table,
        raw_record_bytes,value_record_bytes,value_offset,block_tokens,layers,
        geometry.heads,geometry.dim,geometry.half,geometry.page,triton.next_power_of_2(geometry.dim),
        ((tl.uint8 if software else tl.float8e4nv) if raw_dtype==torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype==torch.bfloat16 else tl.float16),raw_dtype.itemsize,
        (tl.float32 if raw_dtype==torch.float8_e4m3fn else
         tl.bfloat16 if raw_dtype==torch.bfloat16 else tl.float16),
        tl.int32 if geometry.table_dtype==torch.int32 else tl.int64,
        (tl.float32 if geometry.trig_dtype==torch.float32 else
         tl.bfloat16 if geometry.trig_dtype==torch.bfloat16 else tl.float16),
        software,
        num_warps=4,enable_fp_fusion=raw_dtype != torch.float8_e4m3fn)
