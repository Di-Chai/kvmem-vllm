"""Real batched FA2 score isolation and shared DMA byte qualification."""
import os
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.native_attention import NativeAttentionStats
from vllm_kvmem.logical_action import LogicalScorePlan, project_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.capture_runtime import CapturedPacket
from vllm_kvmem.multiowner_prefill import attach_owner_views, bind_owner_payload
from test_physical_prefill_stats_gpu import plan, oracle, RTOL, ATOL
from vllm_kvmem.authority import RequestAuthority, append_authority_batch
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory_storage import MemoryArchive

pytestmark=pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS')!='1',reason='Exclusive multiowner CUDA qualification')


@pytest.mark.parametrize('chunks',[(3,3),(2,4),(1,3),(2,2,2,2,2,2),
                                 (3,3,2,2,2,2),(2,2,2,2,2,2,1,1)])
@pytest.mark.parametrize('static',[False,True])
@pytest.mark.parametrize('aligned',[False,True])
def test_batched_native_scores_keep_each_owner_and_logical_domain(tmp_path,chunks,static,aligned):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    torch.manual_seed(8433)
    logical=1024;heads=24;kvheads=4;dim=256;page=16
    prefixes=tuple(4096 if i%2==0 else 2048 for i in range(len(chunks)))
    counts=tuple(n*logical for n in chunks)
    lengths=tuple(p+n for p,n in zip(prefixes,counts))
    npages=sum(lengths)//page+11
    cache=(torch.randn(npages,kvheads,page,2*dim,device='cuda')*.4).to(torch.float8_e4m3fn)
    key,value=cache.transpose(1,2).split(dim,-1)
    permutation=torch.randperm(npages,device='cuda',dtype=torch.int32)
    tables=[];page_offsets=[0];query_offsets=[0]
    for length,count in zip(lengths,counts):
        page_offsets.append(page_offsets[-1]+length//page)
        query_offsets.append(query_offsets[-1]+count)
        tables.append(permutation[page_offsets[-2]:page_offsets[-1]])
    query=torch.randn(sum(counts),heads,dim,device='cuda',dtype=torch.bfloat16)
    workspace=torch.zeros(32<<20,device='cuda',dtype=torch.uint8)
    wrapper=BatchPrefillWithPagedKVCacheWrapper(workspace,'NHD',backend='fa2')
    ints=lambda v:torch.tensor(v,device='cuda',dtype=torch.int32)
    wrapper.plan(ints(query_offsets),ints(page_offsets),
        torch.cat(tables),ints([page]*len(chunks)),heads,kvheads,dim,page,causal=True,
        sm_scale=dim**-.5,q_data_type=query.dtype,kv_data_type=key.dtype)
    class Impl:
        def forward(self,layer,q,k,v,cache,meta,out):
            return wrapper.run(q,(key,value),q_scale=1.,k_scale=.37,v_scale=.3,out=out)
    impl=Impl();original=impl.forward;name='model.layers.3.attn'
    layer=NS(num_heads=heads,head_dim=dim,q_norm=NS(weight=query),attn=NS(layer_name=name,impl=impl,_q_scale_float=1.))
    runner=NS(model=NS(named_modules=lambda:[(name,layer)]),_kvmem_layers=[name],max_num_tokens=sum(counts),max_num_reqs=len(chunks))
    options=NS(action_native_scores='reuse',action_layers='all',action_native_max_rows=448,
        action_native_max_bytes=256<<20,action_native_prefill_static_aligned=static,
        memory=NS(select_budget=8192,gen_budget=4096,block_tokens=32))
    bank=NativeAttentionStats(runner,options,tmp_path);assert bank.disabled is None
    def score_plan(begin,count,rows,index=0):
        p=plan(begin,count,rows,index=index)
        if aligned:
            history=dict(zip(p['starts'],p['history']))
            p['starts']=list(range(0,begin+count,32))
            p['counts']=[min(32,begin+count-start) for start in p['starts']]
            p['history']=[history[start] for start in p['starts']]
        return p
    plans={}
    for i,n in enumerate(counts):
        plans[str(i)]=score_plan(prefixes[i],n,list(range(31,n,32)),index=i)
        plans[str(i)]['begin']=query_offsets[i]
    bank.arm(plans);meta=NS(prefill=NS(wrapper=wrapper),q_data_type_prefill=query.dtype,num_decode_tokens=0,
                           num_prefills=len(chunks),num_prefill_tokens=sum(counts))
    actual=impl.forward(layer.attn,query,None,None,None,meta,torch.empty_like(query))
    plain=original(layer.attn,query,None,None,None,meta,torch.empty_like(query))
    torch.testing.assert_close(actual,plain,rtol=0,atol=0)
    for i,(prefix,n) in enumerate(zip(prefixes,counts)):
        masses=bank.score(name,str(i),True);assert masses is not None,bank.reasons
        offset=query_offsets[i]
        env=NS(key=key,table=tables[i],query=query[offset:offset+n],page=page,
            heads=heads,kvheads=kvheads,dim=dim,prefix=prefix)
        if chunks[i]==1:
            expected=oracle(env,plans[str(i)]['rows'],plans[str(i)]['starts'],plans[str(i)]['counts'])
            torch.testing.assert_close(masses,expected,rtol=RTOL,atol=ATOL)
            continue
        spans=tuple(LogicalPrefillSpan(j,8192+j*logical,prefix+j*logical,logical,j*logical) for j in range(chunks[i]))
        group=LogicalPrefillGroup(str(i),(1,0,0),7,8192-prefix,8192+n+2048,prefix+n,spans,
            RequestPlan(None,block_tokens=32,gen_budget=4096))
        old=[score_plan(prefix+j*logical,logical,list(range(31,logical,32))) for j in range(chunks[i])]
        parts=tuple(LogicalScorePlan(s,tuple(p['rows']),tuple(p['starts']),tuple(p['counts']),tuple(p['history']))
            for s,p in zip(spans,old))
        projected=project_logical_scores(group,parts,masses.cpu(),physical_rows=tuple(plans[str(i)]['rows']))
        for j,(p,part) in enumerate(zip(old,projected)):
            expected=oracle(env,[j*logical+r for r in p['rows']],p['starts'],p['counts'])
            torch.testing.assert_close(part.masses.cuda(),expected,rtol=RTOL,atol=ATOL)
        torch.testing.assert_close(masses.sum(-1),torch.ones(len(plans[str(i)]['rows']),device='cuda'),rtol=0,atol=2e-5)
    assert bank.validation_failures==0 and not any(k.startswith('fallback:') for k in bank.counts)
    if static and aligned:
        assert bank.static_prefill_counts['static_dispatch_calls']>0,bank.static_prefill_counts
    elif static:
        assert bank.static_prefill_counts['static_dispatch_calls']==0
        assert bank.static_prefill_counts['generic_dispatch:unaligned_logical_segments']>0


@pytest.mark.parametrize('dtype',[torch.float8_e4m3fn,torch.bfloat16])
def test_shared_pinned_dma_strided_owners_full_kv_geometry(dtype):
    # Exercise production byte offsets with 17 layers, including different
    # owner lengths; every owner view has a gap between its layer planes.
    sizes=(128,256);total=sum(sizes);shape=(17,total,4,256)
    gen=torch.Generator(device='cuda').manual_seed(852)
    raw=torch.randn(shape,device='cuda',generator=gen).to(dtype)
    values=torch.randint(0,256,shape,device='cuda',dtype=torch.uint8,generator=gen)
    header=torch.tensor([0,sizes[0],total,0,0,0,0,0,0],dtype=torch.int64,device='cuda')
    packed=torch.cat((header.view(torch.uint8),raw.view(torch.uint8).flatten(),values.flatten()))
    host=torch.empty(packed.shape,dtype=torch.uint8,pin_memory=True)
    host.copy_(packed,non_blocking=True);event=torch.cuda.Event();event.record();event.synchronize()
    download=NS(payload=host)
    packet=CapturedPacket(('a','b'),total,download,request_tokens=sizes,source_generation=8,
        action_inputs={'a':None,'b':None})
    attach_owner_views(packet,{rid:(NS(count=n,history_begin=0,local_begin=0),()) for rid,n in zip(packet.request_ids,sizes)})
    runtime=NS(row_elements=17*4*256,raw_dtype=dtype,layers=list(range(17)),heads=4,dim=256)
    begin=0
    for i,n in enumerate(sizes):
        child=bind_owner_payload(runtime,packet,i);r,v=child.logical_payload
        assert r.untyped_storage().data_ptr()==host.untyped_storage().data_ptr()
        assert not r.is_contiguous()
        config=KVMemConfig(block_tokens=32,select_budget=1024,gen_budget=256,
            repository_blocks=64,sink_blocks=1,recent_blocks=0)
        archive=MemoryArchive('',17*32*4*256*(dtype.itemsize+1),64,cpu_direct_record_seal=True)
        archive.async_seal=True
        authority=RequestAuthority(str(i),config,archive,17,4,256,raw_dtype=dtype,
            scoring_layers=16,mean_k_backend='qw3_cuda',resident_records=8)
        try:
            assert append_authority_batch([(authority,r,v,dict(history_begin=0,local_rope_begin=0))])==1
            authority.publish_sealed(lambda:archive.wait_for_capacity(1))
            pairs=[authority.read_record(record.handle) for record in authority.records.values()]
            restored_k=torch.cat([k for k,v in pairs],dim=1).cuda()
            restored_v=torch.cat([v for k,v in pairs],dim=1).cuda()
            assert torch.equal(restored_k.view(torch.uint8),raw[:,begin:begin+n].contiguous().view(torch.uint8))
            assert torch.equal(restored_v,values[:,begin:begin+n])
        finally:
            assert authority.close() and archive.close()
        begin+=n
