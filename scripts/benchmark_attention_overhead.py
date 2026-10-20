"""Independent fixed-input costs for each attention-overhead switch."""
import argparse
import json
from pathlib import Path
import statistics
import hashlib
import time
from types import SimpleNamespace as NS
import numpy as np
import torch


def summary(samples):
    return {mode:dict(mean_ms=statistics.mean(v['ms'] for v in samples if v['mode']==mode),
        peak_gpu_scratch_bytes=max((v.get('gpu_scratch_bytes',0) for v in samples if v['mode']==mode),default=0))
        for mode in ('reference','optimized')}


def decode(repeats):
    from vllm_kvmem import _native
    from vllm_kvmem.action_attention import ActionHistory,ActionRuntime,ActionPacket,_AcceptanceParser
    from vllm_kvmem.block_graph import BlockGraph
    targets=[0]+list(range(0,53152,32))
    values=np.random.default_rng(43).random((1,len(targets))).astype('float32')
    values/=values.sum(axis=1,keepdims=True)
    banks={}
    for mode in ('reference','optimized'):
        cfg=NS(graph_mode='collect',action_scope='non-thinking',action_interval=16,action_decay=.5,
            prefill_attention=False,action_native_decode_commit=mode=='optimized',action_commit_max_bytes=16777216)
        runtime=ActionRuntime.__new__(ActionRuntime);runtime.options=cfg;runtime.runtime=NS(output=Path('/tmp'))
        owners=[]
        for i in range(8):
            state=ActionHistory(cfg,[10,11,12,13,14]);state.begin('r',[14])
            state.mass={t:.01 for t in targets}
            owners.append((state,BlockGraph((mode,i),32,initial_position=53152)))
        banks[mode]=runtime,owners
    samples=[]
    try:
        for repetition in range(repeats):
            for mode in (('reference','optimized') if repetition%2==0 else ('optimized','reference')):
                runtime,owners=banks[mode];packets=[]
                for state,g in owners:
                    state.parser=_AcceptanceParser(_native.ActionParser('non-thinking',16,[10,11,12,13],False))
                    state.parser.prepare([20]*15);state.parser.commit(15)
                    mask=state.parser.prepare([20]*5)
                    selected=[i for i in range(5) if mask>>i&1]
                    assert len(selected)==1 and selected[0]<3
                    begin=g.accepted_through
                    packet=NS(action_inputs={'r':ActionPacket(state,selected,targets,torch.from_numpy(values),None,
                        source_positions=tuple(begin+i for i in selected),history_begin=begin,candidate_count=5)})
                    packets.append((g,packet,begin))
                began=time.perf_counter()
                for g,packet,begin in packets:
                    update=runtime.prepare_accept(packet,'r',3,graph=g,history_begin=begin)
                    update.commit()
                samples.append(dict(mode=mode,repetition=repetition,ms=(time.perf_counter()-began)*1000))
        for (a,ga),(b,gb) in zip(banks['reference'][1],banks['optimized'][1]):
            assert a.mass==b.mass and list(a.mass)==list(b.mass) and a.samples==b.samples
            assert ga.rows==gb.rows and ga.partial==gb.partial
        return dict(component='decode_native_commit',shape=[8,1,len(targets)],accepted_prefix=3,candidates=5,
            bit_exact_scores_and_graph=True,measurements=summary(samples),samples=samples,
            includes='Runtime prepare_accept + capacity/validation + legacy staging + graph preparation + commit.',
            excludes='Scope preparation, GPU scoring/download and raw archive publication.')
    finally:
        for _,owners in banks.values():
            for _,g in owners:g.close();assert not g.budget.used


def propagation(repeats):
    from vllm_kvmem.block_graph import BlockGraph,GraphSample
    from vllm_kvmem.graph_scoring import score_graph
    g=BlockGraph('propagation',32)
    rng=np.random.default_rng(103);rows=[]
    for i in range(8192):
        targets=tuple(int(v)*32 for v in rng.choice(i+1,min(4,i+1),replace=False))
        values=rng.random(len(targets));values/=values.sum()
        rows.append(GraphSample(i*32+31,targets,tuple(values)))
    with g.prepare(0,262144,tuple(rows)) as update:update.commit()
    del rows
    raw=rng.random(8192).tolist();starts=list(range(0,262144,32));samples=[]
    try:
        with g.snapshot() as snap:
            began=time.perf_counter()
            with score_graph(raw,starts,[True]*8192,snap,scoring_version='v2',native=True) as first:
                first_ms=(time.perf_counter()-began)*1000
                index_bytes=first.diagnostics['native_index_bytes']
            for repetition in range(repeats):
                mask=[(i+repetition)%5!=0 for i in range(8192)]
                outputs={}
                for mode in (('reference','optimized') if repetition%2==0 else ('optimized','reference')):
                    began=time.perf_counter()
                    with score_graph(raw,starts,mask,snap,scoring_version='v2',native=mode=='optimized') as result:
                        elapsed=(time.perf_counter()-began)*1000;outputs[mode]=result.scores
                    samples.append(dict(mode=mode,repetition=repetition,ms=elapsed))
                assert outputs['reference']==outputs['optimized']
            return dict(component='native_graph_propagation',nodes=8192,edges=sum(len(r.targets) for r in snap.rows),
                cold_index_and_propagation_ms=first_ms,index_reserved_bytes=index_bytes,
                exact_scores=True,measurements=summary(samples),samples=samples,
                includes='Budget preflight, per-call seed/mask filtering and normalization, V2 propagation, output/diagnostics.',
                cache='One immutable owner/version CSR; changing masks on each repetition. First build reported separately.')
    finally:g.close();assert not g.budget.used


def pool(repeats):
    from vllm_kvmem.action_download_pool import ActionDownloadPool
    results=[]
    for rows in (32,1):
        source=torch.arange(8*rows*1661,device='cuda',dtype=torch.float32).reshape(8,rows,1661)
        metadata=torch.arange(3*8*(1024 if rows==32 else 5),device='cuda',dtype=torch.int64).reshape(3,-1)
        bank=ActionDownloadPool();samples=[];expected=None
        def run(mode):
            packets=[]
            for value in [metadata,*source]:
                if mode=='optimized':host,event=bank.copy(value)
                else:
                    host=torch.empty(value.shape,dtype=value.dtype,pin_memory=True)
                    host.copy_(value,non_blocking=True);event=torch.cuda.Event();event.record()
                packets.append((host,event))
            total=0.
            for host,event in packets:
                event.synchronize();total+=float(host.numpy().sum(dtype='float64'))
            return total
        for mode in ('reference','optimized'):expected=run(mode)
        for repetition in range(repeats):
            for mode in (('reference','optimized') if repetition%2==0 else ('optimized','reference')):
                began=time.perf_counter();actual=run(mode)
                samples.append(dict(mode=mode,repetition=repetition,ms=(time.perf_counter()-began)*1000))
                assert actual==expected
        results.append(dict(phase='prefill' if rows==32 else 'decode',shape=list(source.shape),
            metadata_shape=list(metadata.shape),
            exact_downloads=True,measurements=summary(samples),samples=samples,pool=bank.snapshot()))
    return dict(component='pinned_download_pool',results=results,
        includes='One int64 metadata and eight mass D2H submissions, allocations/leases/events, completion fences and CPU NumPy consumption.',
        excludes='QK computation and archive publication.')


def fused(repeats):
    from vllm_kvmem.action_scoring import page_attention
    # Verified against the original full-C8 action_config/cache-geometry:
    # both prefill and XQA decode use BF16 Q, FP8 KV, and 1616-token KV pages.
    length,heads,kv,dim,page,batch=53152,24,4,256,1616,8
    physical=batch*((length+page-1)//page)+7
    gen=torch.Generator(device='cuda').manual_seed(71)
    cache=(torch.randn(physical,kv,page,dim*2,device='cuda',generator=gen)*.5).to(torch.float8_e4m3fn)
    key=cache.transpose(1,2)[...,:dim]
    tables=torch.randperm(physical,device='cuda',generator=gen,dtype=torch.int32)[:batch*((length+page-1)//page)].reshape(batch,-1)
    trig=torch.cat((torch.ones(length,dim//4,device='cuda'),torch.zeros(length,dim//4,device='cuda')),dim=-1)
    starts=torch.arange(0,length,32,device='cuda',dtype=torch.int32);counts=(length-starts).clamp(max=32)
    ks,qs=torch.ones((),device='cuda'),torch.ones((),device='cuda');results=[]
    for rows in (32,1):
        q=torch.randn(batch,rows,heads,dim,device='cuda',dtype=torch.bfloat16,generator=gen)
        positions=torch.arange(length-rows*32+31,length,32,device='cuda')
        def run(mode):
            packets=[]
            for owner in range(batch):
                total=None
                for layer in range(3):
                    value=page_attention(q[owner],positions,key,tables[owner],trig,ks,starts,counts,length,
                        scale=dim**-.5,q_dtype=torch.bfloat16,q_scale=qs,block_tokens=32,
                        fused=mode=='optimized',complete_partition=True)
                    total=value if total is None else total+value
                total=total/3
                host=torch.empty(total.shape,dtype=total.dtype,pin_memory=True)
                host.copy_(total,non_blocking=True);event=torch.cuda.Event();event.record();packets.append((host,event))
            for host,event in packets:event.synchronize()
            return [host.numpy().copy() for host,event in packets]
        reference=run('reference');optimized=run('optimized')
        error=max(float(np.max(np.abs(a-b))) for a,b in zip(reference,optimized))
        assert all(np.allclose(a,b,rtol=2e-5,atol=2e-6) for a,b in zip(reference,optimized))
        samples=[]
        for repetition in range(repeats):
            for mode in (('reference','optimized') if repetition%2==0 else ('optimized','reference')):
                torch.cuda.synchronize();allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                began=time.perf_counter();run(mode)
                samples.append(dict(mode=mode,repetition=repetition,ms=(time.perf_counter()-began)*1000,
                    gpu_scratch_bytes=torch.cuda.max_memory_allocated()-allocated))
        results.append(dict(phase='prefill' if rows==32 else 'decode',batch=batch,sampled_rows=rows,
            layers=3,q_dtype='bfloat16',max_absolute_error=error,measurements=summary(samples),samples=samples))
    return dict(component='fused_bypass_scoring',active_tokens=length,q_heads=heads,kv_heads=kv,head_dim=dim,
        native_page_tokens=page,key_scale=1.,q_scale=1.,
        results=results,includes='Existing RoPE/Q codec, three bypass QK calls, full causal normalization, page/head/layer reductions, allocations and pinned D2H.',
        excludes='Unchanged native model attention, CPU window/metadata preparation and archive publication.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component',choices=('decode','pool','fused','propagation'),required=True)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve previous attempts; use a new output')
    report=globals()[args.component](args.repeats)
    report.update(repeats=args.repeats,scope='Fixed inputs and interleaved independent component comparison, not model throughput.')
    report['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','results')},indent=2))


if __name__=='__main__':main()
