"""Original RoPE must not alter physical destinations or the causal mask."""
import os
import pytest
import torch

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Explicit GPU check')


def test_packed_original_key_matches_direct_rope_at_sparse_positions():
    from vllm_kvmem.adapters.assembly import assemble_fp8
    from vllm_kvmem.serving.packed_assembly import assemble_packed_layer
    from vllm_kvmem.serving.window_assembly import LayerWindow
    torch.manual_seed(8)
    raw=torch.randn((64,1,8),device='cuda').to(torch.float8_e4m3fn)
    v=torch.randn_like(raw.float()).to(torch.float8_e4m3fn)
    positions=torch.cat([torch.arange(32),torch.arange(180000,180032)]).cuda()
    phase=torch.arange(180032,device='cuda')[:,None]*torch.tensor([.1,.01,.001,.0001],device='cuda')
    trig=torch.cat([phase.cos(),phase.sin()],dim=1)
    scale=torch.tensor(.04052734375,device='cuda')
    key=torch.zeros((4,32,1,8),device='cuda',dtype=torch.float8_e4m3fn);value=torch.zeros_like(key)
    table=torch.tensor([2,0],device='cuda',dtype=torch.int32)
    slots=torch.cat([torch.arange(64,96),torch.arange(32)]).cuda()
    assemble_fp8(raw,v,positions,slots,trig,key,value,scale)
    packed_key=torch.zeros_like(key);packed_value=torch.zeros_like(value)
    payload=torch.cat([raw.view(torch.uint8).reshape(2,-1),v.view(torch.uint8).reshape(2,-1)],dim=1).flatten()
    layer=LayerWindow(packed_key,packed_value,trig,scale,table)
    assemble_packed_layer(payload,torch.tensor([0,32],device='cuda'),layer,
        layer_index=0,layers=1,block_tokens=32,raw_dtype=raw.dtype,
        rope_bases=torch.tensor([0,180000],device='cuda'))
    assert torch.equal(key.view(torch.uint8),packed_key.view(torch.uint8))
    assert torch.equal(value.view(torch.uint8),packed_value.view(torch.uint8))


@pytest.mark.parametrize('fused',[False,True])
def test_attention_uses_original_rotation_but_dense_causal_boundary(fused):
    from vllm_kvmem.serving.action_scoring import page_attention
    from vllm_kvmem.core.codec import neox_rope
    torch.manual_seed(3)
    q=torch.randn((2,2,8),device='cuda',dtype=torch.bfloat16)
    key=torch.randn((2,32,1,8),device='cuda').to(torch.float8_e4m3fn)
    phase=torch.arange(180002,device='cuda')[:,None]*torch.tensor([.1,.01,.001,.0001],device='cuda')
    trig=torch.cat([phase.cos(),phase.sin()],dim=1)
    dense=torch.tensor([40,41],device='cuda');logical=torch.tensor([180000,180001],device='cuda')
    table=torch.tensor([0,1],device='cuda',dtype=torch.int32)
    starts=torch.tensor([0,32],device='cuda',dtype=torch.int32)
    counts=torch.tensor([32,32],device='cuda',dtype=torch.int32)
    scale=torch.tensor(1.,device='cuda')
    actual=page_attention(q,dense,key,table,trig,scale,starts,counts,64,scale=8**-.5,
        q_dtype=q.dtype,q_scale=scale,block_tokens=32,fused=fused,complete_partition=True,
        rope_positions=logical)
    rot=neox_rope(q,trig[logical,:4],trig[logical,4:]).to(q.dtype).float()
    scores=torch.einsum('rhd,td->rht',rot,key.reshape(64,8).float())*(8**-.5)
    scores.masked_fill_(torch.arange(64,device='cuda')[None,None,:]>dense[:,None,None],-torch.inf)
    probs=scores.softmax(-1).mean(1)
    expected=torch.stack([probs[:,:32].sum(1),probs[:,32:].sum(1)],dim=1)
    torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
