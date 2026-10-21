"""Opt-in bounded physical / 2K logical FA2 qualification; no throughput claim.

The score tolerance is the existing native_attention_stats_gpu qualification
(2% relative, 2e-4 absolute), not a newly relaxed threshold. Both physical
shapes must independently pass a full-context FP32 oracle. Cross-shape model
outputs are measured, not assumed bitwise equivalent; same-shape instrumented
and original FA2 outputs must remain bitwise identical.
"""
import json
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.logical_action import LogicalScorePlan, project_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.native_attention import NativeAttentionStats
from vllm_kvmem.request_plan import RequestPlan


pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit physical-prefill CUDA qualification')
NAME = 'model.layers.3.attn'
RTOL, ATOL = .02, 2e-4


def segments(end, split):
    starts = ([0,32,37] if split == 'partial' else [0,5,32]) + list(range(64,end,32))
    counts = [b-a for a,b in zip(starts,starts[1:]+[end])]
    history = [((start//32)*37%128)*32 if start < 4096 else 8192+(start-4096)//32*32
               for start in starts]
    return starts, counts, history


def geometry(tmp_path, *, prefix=4096, chunks=2):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(9361)
    heads,kvheads,dim,page,count = 24,4,256,16,chunks*2048
    physical = (prefix+count)//page+7
    cache = (torch.randn(physical,kvheads,page,2*dim,device='cuda')*.4).to(torch.float8_e4m3fn)
    key,value = cache.transpose(1,2).split(dim,-1)
    table = torch.randperm(physical,device='cuda',dtype=torch.int32)[:(prefix+count)//page]
    query = torch.randn(count,heads,dim,device='cuda',dtype=torch.bfloat16)
    workspace = torch.zeros(32<<20,device='cuda',dtype=torch.uint8)
    wrappers = []
    for qlen,length in [*( (2048,prefix+(i+1)*2048) for i in range(chunks)),(count,prefix+count)]:
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD',backend='fa2')
        wrapper.plan(torch.tensor([0,qlen],device='cuda',dtype=torch.int32),
            torch.tensor([0,(length+page-1)//page],device='cuda',dtype=torch.int32),
            table[:(length+page-1)//page],
            torch.tensor([(length-1)%page+1],device='cuda',dtype=torch.int32),
            heads,kvheads,dim,page,causal=True,sm_scale=dim**-.5,
            q_data_type=query.dtype,kv_data_type=key.dtype)
        wrappers.append(wrapper)
    class Impl:
        def forward(self,layer,q,k,v,cache,meta,out):
            return meta.prefill.wrapper.run(q,(key,value),q_scale=1.,k_scale=.37,v_scale=.3,out=out)
    impl = Impl()
    original = impl.forward
    layer = NS(num_heads=heads,head_dim=dim,q_norm=NS(weight=query),
               attn=NS(layer_name=NAME,impl=impl,_q_scale_float=1.))
    runner = NS(model=NS(named_modules=lambda:[(NAME,layer)]),_kvmem_layers=[NAME],
                max_num_tokens=count,max_num_reqs=2)
    options = NS(action_native_scores='reuse',action_layers='all',action_native_max_rows=chunks*64,
        action_native_max_bytes=256<<20,memory=NS(select_budget=8192,gen_budget=4096,block_tokens=32))
    bank = NativeAttentionStats(runner,options,tmp_path)
    assert bank.disabled is None,bank.disabled
    queries = [*(query[i*2048:(i+1)*2048] for i in range(chunks)),query]
    outputs = [torch.empty_like(q) for q in queries]
    metas = [NS(prefill=NS(wrapper=w),q_data_type_prefill=query.dtype,num_decode_tokens=0) for w in wrappers]
    def execute(index, baseline=False):
        if baseline:
            return original(layer.attn,queries[index],None,None,None,metas[index],torch.empty_like(queries[index]))
        return impl.forward(layer.attn,queries[index],None,None,None,metas[index],outputs[index])
    return NS(bank=bank,key=key,value=value,table=table,query=query,outputs=outputs,
              execute=execute,heads=heads,kvheads=kvheads,dim=dim,page=page,prefix=prefix,
              count=count,chunks=chunks,physical_index=chunks)


def plan(local_begin,count,rows,split='partial',index=0):
    starts,counts,history = segments(local_begin+count,split)
    return dict(index=index,begin=0,rows=list(rows),starts=starts,counts=counts,history=history,
                local_begin=local_begin,count=count)


def oracle(env,rows,starts,counts):
    # FP8 cache entries are decoded to exact FP32 values; BF16 Q is similarly
    # widened. TF32 is disabled explicitly so this is the old FP32-dot oracle.
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        length = starts[-1]+counts[-1]
        positions = torch.arange(length,device='cuda')
        keys = env.key[env.table[positions//env.page].long(),positions%env.page].float()
        keys = keys.repeat_interleave(env.heads//env.kvheads,1)
        sampled = env.query[rows].float()
        logits = torch.einsum('rhd,thd->rht',sampled,keys)*(.37*env.dim**-.5)
        logits.masked_fill_(positions[None,None,:] >
            (env.prefix+torch.tensor(rows,device='cuda'))[:,None,None],-torch.inf)
        probability = logits.softmax(-1).mean(1)
        return torch.stack([probability[:,start:start+size].sum(-1)
                            for start,size in zip(starts,counts)],-1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32


def errors(actual,expected):
    left,right = actual.detach().float(),expected.detach().float()
    delta = (left-right).abs()
    return dict(max_abs=delta.max().item(),rms=(delta.square().mean().sqrt()).item(),
                mean_abs=delta.mean().item(),different_values=int((left!=right).sum().item()),
                different_bits=int((left.contiguous().view(torch.int32)!=
                                    right.contiguous().view(torch.int32)).sum().item()),
                elements=left.numel(),max_row_l1=delta.sum(-1).max().item())


@pytest.mark.parametrize('chunks',[2,3,4,7,8])
def test_logical_2k_scores_vs_one_physical_full_causal_domain(tmp_path,chunks):
    env = geometry(tmp_path,chunks=chunks)
    rows = list(range(31,2048,32))
    physical_rows = [i*2048+row for i in range(chunks) for row in rows]
    original_plans = tuple(plan(env.prefix+i*2048,2048,rows) for i in range(chunks))
    old_scores,old_outputs = [],[]
    for index,pending in enumerate(original_plans):
        env.bank.arm({'r':pending});env.execute(index)
        mass = env.bank.score(NAME,'r',True)
        assert mass is not None,env.bank.reasons
        old_scores.append(mass.clone())
        old_outputs.append(env.outputs[index].clone())
        torch.testing.assert_close(env.outputs[index],env.execute(index,True),rtol=0,atol=0)
    physical = plan(env.prefix,env.count,physical_rows)
    env.bank.arm({'r':physical});env.execute(chunks)
    masses = env.bank.score(NAME,'r',True)
    assert masses is not None,env.bank.reasons
    masses = masses.clone()
    torch.testing.assert_close(env.outputs[chunks],env.execute(chunks,True),rtol=0,atol=0)
    spans = tuple(LogicalPrefillSpan(i,8192+i*2048,4096+i*2048,2048,i*2048) for i in range(chunks))
    group = LogicalPrefillGroup('r',(1,0,0),7,4096,8192+env.count+2048,env.prefix+env.count,spans,
                               RequestPlan(None,block_tokens=32,gen_budget=4096))
    score_plans = tuple(LogicalScorePlan(span,tuple(rows),tuple(p['starts']),tuple(p['counts']),
                       tuple(p['history'])) for span,p in zip(spans,original_plans))
    projected = project_logical_scores(group,score_plans,masses.cpu(),physical_rows=tuple(physical_rows))
    report = dict(geometry=dict(prefix=env.prefix,physical_tokens=env.count,logical_tokens=2048,
        q_heads=24,kv_heads=4,dim=256,page=16,k_scale=.37,noncontiguous_physical_pages=True),
        tolerance=dict(rtol=RTOL,atol=ATOL,source='existing native_attention_stats_gpu qualification'),
        logical=[],model_attention_cross_shape=errors(env.outputs[chunks],torch.cat(old_outputs)))
    comparisons = []
    for i,(pending,part) in enumerate(zip(original_plans,projected)):
        selected_rows = [i*2048+row for row in rows]
        expected = oracle(env,selected_rows,pending['starts'],pending['counts'])
        candidate = part.masses.to('cuda')
        report['logical'].append(dict(index=i,old_vs_oracle=errors(old_scores[i],expected),
            physical_vs_oracle=errors(candidate,expected),physical_vs_old=errors(candidate,old_scores[i])))
        comparisons.append((old_scores[i],candidate,expected))
    (tmp_path/'physical-prefill-stats.json').write_text(json.dumps(report,indent=2)+'\n')
    for old,candidate,expected in comparisons:
        torch.testing.assert_close(old,expected,rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(candidate,expected,rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(candidate,old,rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(candidate.sum(-1),torch.ones(64,device='cuda'),rtol=0,atol=2e-5)
    # Future columns must contribute exactly zero, never enter the old domain.
    for i,pending in enumerate(original_plans[:-1]):
        assert torch.count_nonzero(masses[i*64:(i+1)*64,len(pending['starts']):]).item() == 0
    assert env.bank.validation_failures == 0
    assert not any(key.startswith('fallback:') for key in env.bank.counts)


@pytest.mark.parametrize('chunks',[2,3,4,7,8])
def test_physical_graph_replay_changes_owner_rows_and_partial_segments(tmp_path,chunks):
    env = geometry(tmp_path,chunks=chunks)
    initial = plan(env.prefix,env.count,list(range(31,env.count,32)))
    env.bank.arm({'initial':initial});env.execute(chunks)
    assert env.bank.score(NAME,'initial',True) is not None
    stream = torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream): env.execute(chunks)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): env.execute(chunks)
    addresses = tuple(t.data_ptr() for t in (env.bank.rows,env.bank.ranges,env.bank.lses[NAME]))
    reports = []
    for owner,index,rows,split in (('replacement',1,[17,1023,2048,env.count-1],'changed'),
                                  ('again',0,[0,31,2047,env.count-1025,env.count-1],'partial')):
        env.query.mul_(.75)
        current = plan(env.prefix,env.count,rows,split,index)
        env.bank.arm({owner:current});env.execute(chunks)
        expected_same_shape = env.bank.score(NAME,owner,True).clone()
        expected_output = env.outputs[chunks].clone()
        # Deliberately contaminate prior rows/columns before arming the replay.
        env.bank.lses[NAME].fill_(1000.)
        env.bank.arm({owner:current});graph.replay()
        actual = env.bank.score(NAME,owner,True)
        assert actual is not None,env.bank.reasons
        expected_math = oracle(env,rows,current['starts'],current['counts'])
        reports.append(dict(owner=owner,index=index,rows=rows,split=split,
                            scores_vs_oracle=errors(actual,expected_math)))
        torch.testing.assert_close(actual,expected_same_shape,rtol=0,atol=0)
        torch.testing.assert_close(env.outputs[chunks],expected_output,rtol=0,atol=0)
        torch.testing.assert_close(actual,expected_math,rtol=RTOL,atol=ATOL)
        assert addresses == tuple(t.data_ptr() for t in (env.bank.rows,env.bank.ranges,env.bank.lses[NAME]))
    (tmp_path/'physical-prefill-replay.json').write_text(json.dumps(reports,indent=2)+'\n')
    assert env.bank.validation_failures == 0


@pytest.mark.parametrize('fused',[False,True])
@pytest.mark.parametrize('chunks',[2,3,4,7,8])
def test_reference_fallback_preserves_original_logical_widths_and_exact_scores(tmp_path,monkeypatch,fused,chunks):
    """Exercise the production fallback helper across the 8K width boundary.

    The first old transaction uses length/softmax width 8192; the second uses
    length 10240/width 16384. Scoring the first with the final physical length
    would violate the original reference computation even with a causal mask.
    """
    import vllm_kvmem.action_scoring as scoring
    from vllm_kvmem.logical_action import reference_logical_scores
    env = geometry(tmp_path,prefix=6144,chunks=chunks)
    prefix,history_begin = env.prefix,12288
    rows = tuple(range(31,2048,32))
    physical_rows = tuple(i*2048+row for i in range(chunks) for row in rows)
    spans = tuple(LogicalPrefillSpan(i,history_begin+i*2048,prefix+i*2048,2048,i*2048)
                  for i in range(chunks))
    group = LogicalPrefillGroup('r',(1,0,0),7,history_begin-prefix,
                               history_begin+env.count+2048,prefix+env.count,spans,
                               RequestPlan(None,block_tokens=32,gen_budget=4096))
    plans = []
    for span in spans:
        starts = [0,32,37]+list(range(64,span.local_end,32))
        counts = [b-a for a,b in zip(starts,starts[1:]+[span.local_end])]
        history = [((start//32)*37%(prefix//32))*32 if start < prefix else
                   history_begin+(start-prefix)//32*32 for start in starts]
        plans.append(LogicalScorePlan(span,rows,tuple(starts),tuple(counts),tuple(history)))
    plans = tuple(plans)
    row_ids = torch.tensor(physical_rows,device='cuda',dtype=torch.int64)
    query = env.query.index_select(0,row_ids)
    positions = row_ids+prefix
    starts = torch.tensor(plans[-1].starts,device='cuda',dtype=torch.int32)
    counts = torch.tensor(plans[-1].counts,device='cuda',dtype=torch.int32)
    # Identity RoPE isolates the original BF16 Q and FP8 K arithmetic while
    # exercising the actual production codec/normalization/segment kernels.
    trig = torch.zeros((prefix+env.count,env.dim),device='cuda',dtype=torch.float32)
    trig[:,:env.dim//2] = 1
    key_scale = torch.tensor(.37,device='cuda',dtype=torch.float32)
    q_scale = torch.ones((),device='cuda',dtype=torch.float32)
    keywords = dict(scale=env.dim**-.5,q_dtype=torch.bfloat16,q_scale=q_scale,
                    block_tokens=32,fused=fused)
    original = scoring.page_attention
    expected = []
    for i,plan in enumerate(plans):
        begin,end = i*len(rows),(i+1)*len(rows)
        expected.append(original(query[begin:end],positions[begin:end],env.key,env.table,trig,
            key_scale,starts[:len(plan.starts)],counts[:len(plan.counts)],plan.span.local_end,
            complete_partition=True,**keywords))
    calls = []
    def observed(*args,**kwargs):
        calls.append((args[0].shape[0],args[6].numel(),args[8]))
        return original(*args,**kwargs)
    monkeypatch.setattr(scoring,'page_attention',observed)
    candidate = reference_logical_scores(group,plans,query,positions,env.key,env.table,trig,
        key_scale,starts,counts,**keywords)
    assert calls == [(64,len(plan.starts),plan.span.local_end) for plan in plans]
    parts = project_logical_scores(group,plans,candidate.cpu(),physical_rows=physical_rows)
    reports = []
    for i,(part,old) in enumerate(zip(parts,expected)):
        actual = part.masses.to('cuda')
        expected_math = oracle(env,list(physical_rows[i*64:(i+1)*64]),
                               plans[i].starts,plans[i].counts)
        reports.append(dict(logical_index=i,original_length=plans[i].span.local_end,
            physical_vs_old=errors(actual,old),old_vs_fp32=errors(old,expected_math)))
        # Same original operators, shapes, and token/head reduction order.
        assert torch.equal(actual.contiguous().view(torch.int32),old.contiguous().view(torch.int32))
        torch.testing.assert_close(old,expected_math,rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(actual.sum(-1),torch.ones(64,device='cuda'),rtol=0,atol=2e-5)
    for i,child in enumerate(plans[:-1]):
        assert torch.count_nonzero(candidate[i*64:(i+1)*64,len(child.starts):]).item() == 0
    (tmp_path/f'physical-prefill-reference-fused-{fused}.json').write_text(
        json.dumps(dict(fused=fused,chunks=chunks,calls=calls,logical=reports),indent=2)+'\n')
