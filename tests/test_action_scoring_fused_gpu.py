"""Fixed operands isolate bypass fusion from native QK reuse."""
import pytest
import torch
from vllm_kvmem.action_scoring import page_attention

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA qualification')


def operands(length,rows,dim,heads,kv,block):
    g=torch.Generator(device='cuda').manual_seed(20261020)
    pages=(length+15)//16+7
    cache=(torch.randn(pages,kv,16,dim*2,generator=g,device='cuda')*.5).to(torch.float8_e4m3fn)
    key=cache.transpose(1,2)[...,:dim]
    table=torch.randperm(pages,generator=g,device='cuda',dtype=torch.int32)
    q=torch.randn(rows,heads,dim,generator=g,device='cuda',dtype=torch.bfloat16)
    positions=torch.linspace(0,length-1,rows,device='cuda').long() if rows>1 else torch.tensor([length-1],device='cuda')
    angle=torch.randn(length,dim//4,generator=g,device='cuda')
    trig=torch.cat((angle.cos(),angle.sin()),dim=-1)
    starts=[0,3,5,17]+list(range(32,length,block))
    counts=[b-a for a,b in zip(starts,starts[1:]+[length])]
    return (q,positions,key,table,trig,torch.tensor(.7,device='cuda'),
        torch.tensor(starts,device='cuda',dtype=torch.int32),
        torch.tensor(counts,device='cuda',dtype=torch.int32),length),dict(
        scale=dim**-.5,q_scale=torch.tensor(.3,device='cuda'),block_tokens=block)


@pytest.mark.parametrize('q_dtype',[torch.bfloat16,torch.float16,torch.float8_e4m3fn])
@pytest.mark.parametrize('geometry',[(73,3,128,8,2,16),(53152,1,256,24,4,32),
    (1024,32,256,24,4,32),(3104,97,256,8,2,16)])
def test_fused_matches_reference_full_domain_gqa_scales_and_partial_pages(q_dtype,geometry):
    args,kwargs=operands(*geometry);kwargs['q_dtype']=q_dtype
    expected=page_attention(*args,**kwargs)
    actual=page_attention(*args,**kwargs,fused=True,complete_partition=True)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual.sum(-1),torch.ones(geometry[1],device='cuda'),rtol=2e-5,atol=2e-6)
    # Prefix acceptance consumes precisely the same rows from either output.
    accepted=torch.arange(geometry[1],device='cuda')<max(1,geometry[1]//2)
    torch.testing.assert_close(actual[accepted].sum(0),expected[accepted].sum(0),rtol=2e-5,atol=2e-6)


def test_missing_complete_partition_contract_falls_back_to_full_softmax():
    args,kwargs=operands(73,3,128,8,2,16);kwargs['q_dtype']=torch.bfloat16
    args=list(args);args[6]=args[6][:2];args[7]=args[7][:2]
    expected=page_attention(*args,**kwargs)
    actual=page_attention(*args,**kwargs,fused=True)
    assert actual._action_scoring_backend=='reference'
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_graph_replay_changes_operands_without_reading_old_lse():
    args,kwargs=operands(259,3,256,8,2,16);kwargs['q_dtype']=torch.float8_e4m3fn
    page_attention(*args,**kwargs,fused=True,complete_partition=True);torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual=page_attention(*args,**kwargs,fused=True,complete_partition=True)
    args[0].mul_(.5);args[1].copy_(torch.tensor([2,15,258],device='cuda'))
    args[3].copy_(args[3].flip(0));graph.replay()
    expected=page_attention(*args,**kwargs)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)


def test_fused_native_page_stride_past_two_gib_uses_wide_addressing():
    physical,page,kv,heads,dim=650,1616,4,8,256
    cache=torch.empty((physical,kv,page,dim*2),device='cuda',dtype=torch.float8_e4m3fn)
    cache[649].fill_(.5);cache[648].fill_(-.5)
    key=cache.transpose(1,2)[...,:dim]
    assert 649*key.stride(0)>2**31
    length=page+1
    q=torch.full((1,heads,dim),.125,device='cuda',dtype=torch.bfloat16)
    position=torch.tensor([length-1],device='cuda');table=torch.tensor([649,648],device='cuda',dtype=torch.int32)
    trig=torch.cat((torch.ones(length,dim//4,device='cuda'),torch.zeros(length,dim//4,device='cuda')),dim=-1)
    unit=torch.ones((),device='cuda');starts=torch.arange(0,length,32,device='cuda',dtype=torch.int32)
    kwargs=dict(scale=dim**-.5,q_dtype=torch.bfloat16,q_scale=unit,block_tokens=32)
    args=(q,position,key,table,trig,unit,starts,(length-starts).clamp(max=32),length)
    expected=page_attention(*args,**kwargs)
    actual=page_attention(*args,**kwargs,fused=True,complete_partition=True)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
