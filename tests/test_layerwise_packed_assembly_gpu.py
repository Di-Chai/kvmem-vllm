"""Byte equality of reused packed addressing on real strided, permuted KV pages."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Exclusive GPU qualification only')


@pytest.mark.parametrize('scale',[1.,.04052734375])
@pytest.mark.parametrize('rotary',[64,256])
@pytest.mark.parametrize('read_blocks',[8,256])
def test_packed_layer_chunk_is_byte_exact_with_token_addressed_assembly(scale,rotary,read_blocks):
    from vllm_kvmem.adapters.assembly import assemble_fp8
    from vllm_kvmem.serving.layerwise_pressure_efficiency import packed_layer_chunk
    from vllm_kvmem.serving.window_assembly import LayerWindow
    torch.manual_seed(1209)
    total,page,heads,dim=36864,64,4,256
    pages=(total+page-1)//page+3
    # Interleaved KV has noncontiguous outer strides, as in native vLLM.
    before=torch.zeros((pages,page,heads,2,dim),dtype=torch.float8_e4m3fn,device='cuda')
    after=torch.zeros_like(before)
    table=torch.randperm(pages,device='cuda',dtype=torch.int64).to(torch.int32)
    angles=torch.randn((total,rotary//2),device='cuda')
    trig=torch.cat((angles.cos(),angles.sin()),dim=1)
    scale_tensor=torch.tensor(scale,device='cuda')
    layer=LayerWindow(after[:,:,:,0],after[:,:,:,1],trig,scale_tensor,table)
    state=dict(selected_tokens=total)
    # Gaps represent original skipped native remaps; both paths must leave
    # them untouched. Different group sizes exercise the final short group.
    groups=[]
    for begin,length in [(0,96),(160,8192),(9216,8192),(32768,3552)]:
        for cursor in range(begin,begin+length,read_blocks*32):
            groups.append((cursor,min(read_blocks,(begin+length-cursor)//32)))
    for cursor,blocks in groups:
        n=blocks*32;half=n*heads*dim
        raw=torch.randn((n,heads,dim),device='cuda').to(torch.float8_e4m3fn)
        values=torch.randint(0,256,(n,heads,dim),dtype=torch.uint8,device='cuda')
        payload=torch.cat((raw.view(torch.uint8).flatten(),values.flatten()))
        positions=torch.arange(cursor,cursor+n,dtype=torch.int64,device='cuda')
        slots=table[positions//page].long()*page+positions%page
        assemble_fp8(raw,values.view(torch.float8_e4m3fn),positions,slots,trig,
                     before[:,:,:,0],before[:,:,:,1],scale_tensor)
        assert packed_layer_chunk(state,layer,payload,half,cursor,[NS(count=32)]*blocks,32*heads*dim,read_blocks)
    torch.cuda.synchronize()
    assert torch.equal(before.view(torch.uint8),after.view(torch.uint8))
