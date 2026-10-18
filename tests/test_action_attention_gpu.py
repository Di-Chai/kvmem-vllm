import pytest
import torch

from vllm_kvmem.action_scoring import page_attention
from vllm_kvmem.codec import neox_rope


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA attention qualification')
def test_action_attention_reads_native_pages_past_two_gib():
    # Actual 4-way K=G=128K arena geometry: page 649 crosses 2**31
    # bytes. Short windows suffice to expose physical-address overflow.
    physical, page, kvheads, heads, dim = 752, 1616, 4, 8, 256
    cache = torch.empty((physical, kvheads, page, dim*2), device='cuda',
                        dtype=torch.float8_e4m3fn).fill_(0)
    cache[648].fill_(-.5)
    cache[649].fill_(.5)
    key = cache.transpose(1, 2)[..., :dim]
    assert 649 * key.stride(0) > 2**31
    table = torch.tensor([649, 648], device='cuda', dtype=torch.int32)
    length = page + 1
    q = torch.full((1, heads, dim), .125, device='cuda', dtype=torch.bfloat16)
    pos = torch.tensor([length - 1], device='cuda')
    trig = torch.cat((torch.ones(length, dim//4, device='cuda'),
                      torch.zeros(length, dim//4, device='cuda')), dim=-1)
    scale = torch.ones((), device='cuda')
    starts = torch.arange(0, length, 32, device='cuda', dtype=torch.int32)
    counts = (length-starts).clamp(max=32)
    actual = page_attention(q, pos, key, table, trig, scale, starts, counts,
        length, scale=dim**-.5, q_dtype=torch.bfloat16, q_scale=scale, block_tokens=32)
    # Constant +/- K yields known logits for every head, no large dense oracle.
    logits = torch.full((length,), 1., device='cuda')
    logits[-1] = -1.
    probability = logits.softmax(-1)
    expected = torch.stack([probability[i:i+32].sum() for i in range(0, length, 32)])
    torch.testing.assert_close(actual[0], expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual.sum(), torch.ones((), device='cuda'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA attention qualification')
@pytest.mark.parametrize('q_dtype',[torch.bfloat16,torch.float8_e4m3fn])
@pytest.mark.parametrize('length,dim,rows',[(73,128,3),(259,256,3),(1024,256,32),(3104,256,97)])
def test_paged_attention_matches_dense_oracle_with_causal_speculation(q_dtype,length,dim,rows):
    g=torch.Generator(device='cuda').manual_seed(841)
    page,heads,kvheads=16,8,2
    physical=(length+page-1)//page+2
    # Strided native BHN(2D) bytes, deliberately permuted physical pages.
    cache=(torch.randn(physical,kvheads,page,dim*2,generator=g,device='cuda')*.5).to(torch.float8_e4m3fn)
    key=cache.transpose(1,2)[...,:dim]
    table=torch.randperm(physical,device='cuda',generator=g,dtype=torch.int32)
    q=torch.randn(rows,heads,dim,generator=g,device='cuda',dtype=torch.bfloat16)
    pos=torch.tensor([length-5,length-3,length-1] if rows==3 else list(range(31,length,32)),device='cuda')
    angle=torch.randn(length,dim//4,generator=g,device='cuda')
    trig=torch.cat([angle.cos(),angle.sin()],-1)
    ks,qs=torch.tensor(.8,device='cuda'),torch.tensor(.3,device='cuda')
    # Include a split partial historical page to exercise nonaligned segments.
    starts=[0,16,21]+list(range(32,length,16))
    counts=[b-a for a,b in zip(starts,starts[1:]+[length])]
    actual=page_attention(q,pos,key,table,trig,ks,
        torch.tensor(starts,device='cuda',dtype=torch.int32),
        torch.tensor(counts,device='cuda',dtype=torch.int32),length,
        scale=dim**-.5,q_dtype=q_dtype,q_scale=qs,block_tokens=16)
    rotated=neox_rope(q,trig[pos,:dim//4],trig[pos,dim//4:])
    if q_dtype==torch.float8_e4m3fn:
        rotated=(rotated.float()/qs).clamp(-448,448).to(q_dtype).float()*qs
    else:rotated=rotated.float()
    p=torch.arange(length,device='cuda')
    dense=key[table[p//page].long(),p%page].float()*ks
    dense=dense.repeat_interleave(heads//kvheads,dim=1)
    logits=torch.einsum('rhd,thd->rht',rotated,dense)*dim**-.5
    logits=logits.masked_fill(p[None,None,:]>pos[:,None,None],-torch.inf)
    mean=logits.softmax(-1).mean(1)
    expected=torch.stack([mean[:,a:a+n].sum(-1) for a,n in zip(starts,counts)],-1)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(actual.sum(-1),torch.ones(rows,device='cuda'))
