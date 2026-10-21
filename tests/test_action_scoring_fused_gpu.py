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


def test_uniform_logits_preserve_causal_mass_under_large_per_head_shifts():
    # Softmax is invariant to a common logit offset within each head. Absolute
    # FP32 logsumexp loses the small normalization term at these offsets.
    length,dim,heads=259,256,8
    positions=torch.tensor([0,2,17,35,258],device='cuda')
    q=torch.zeros(len(positions),heads,dim,device='cuda',dtype=torch.bfloat16)
    shifts=torch.tensor([-1048576.,-65536.,-4096.,-16.,16.,4096.,65536.,1048576.],device='cuda')
    q[...,0]=shifts*16
    key=torch.zeros(18,16,2,dim,device='cuda')
    key[...,0]=1.
    key=key.to(torch.float8_e4m3fn)
    table=torch.arange(17,-1,-1,device='cuda',dtype=torch.int32)
    trig=torch.cat((torch.ones(length,dim//4,device='cuda'),
                    torch.zeros(length,dim//4,device='cuda')),dim=-1)
    starts=torch.arange(0,length,32,device='cuda',dtype=torch.int32)
    counts=(length-starts).clamp(max=32)
    unit=torch.ones((),device='cuda')
    args=(q,positions,key,table,trig,unit,starts,counts,length)
    kwargs=dict(scale=1/16,q_dtype=torch.bfloat16,q_scale=unit,block_tokens=32)
    visible=torch.minimum((positions[:,None]-starts[None,:]+1).clamp(min=0),counts)
    expected=visible.float()/(positions[:,None]+1)
    reference=page_attention(*args,**kwargs)
    actual=page_attention(*args,**kwargs,fused=True,complete_partition=True)
    torch.testing.assert_close(reference,expected,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
    assert torch.equal(actual[visible==0],torch.zeros_like(actual[visible==0]))

    # Replay must overwrite both statistics, including completely masked pages.
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed=page_attention(*args,**kwargs,fused=True,complete_partition=True)
    q.mul_(-8)
    positions.copy_(torch.tensor([258,35,17,2,0],device='cuda'))
    graph.replay()
    visible=torch.minimum((positions[:,None]-starts[None,:]+1).clamp(min=0),counts)
    expected=visible.float()/(positions[:,None]+1)
    torch.testing.assert_close(replayed,expected,rtol=2e-6,atol=2e-7)
    assert torch.equal(replayed[visible==0],torch.zeros_like(replayed[visible==0]))


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
