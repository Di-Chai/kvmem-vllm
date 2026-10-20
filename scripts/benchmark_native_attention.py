"""Alternating fixed-operand FA2/XQA cost of three sampled attention layers.

This is a component gate, not a real-model throughput estimate. Both arms execute
native attention, layer averaging and pinned D2H. Reuse includes its metadata
upload, scratch clear, LSE merge and validation readback. Baseline includes its
original per-request metadata uploads. The two arms retain the actual ordering:
all native layers first, then scoring by request and layer.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime
from vllm_kvmem.native_attention import NativeAttentionStats


def setup(args, phase):
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    torch.manual_seed(7231)
    batch, length, page, heads, kh, dim = args.batch, args.length, 16, 24, 4, 256
    qlen = args.chunk if phase == 'prefill' else 5
    sampled = list(range(31, qlen, 32)) if phase == 'prefill' else [0]
    pages = math.ceil(length/page)
    # Same strided native BHN(2D) FP8 arena as the actual serving path.
    cache = (torch.randn(batch*pages, kh, page, 2*dim, device='cuda')*.3).to(torch.float8_e4m3fn)
    k, v = cache.transpose(1, 2).split(dim, -1)
    table = torch.randperm(batch*pages, device='cuda', dtype=torch.int32).reshape(batch,pages)
    q = torch.randn(batch*qlen, heads, dim, dtype=torch.bfloat16, device='cuda')
    lengths = torch.full((batch,), length, dtype=torch.int32, device='cuda')
    mask = torch.tensor([[(1 << (row+1))-1, 0] for _ in range(batch) for row in range(qlen)],
                        device='cuda', dtype=torch.uint16) if phase == 'decode' else None
    workspace = torch.zeros(128*1024*1024, device='cuda', dtype=torch.uint8)
    wrapper = None
    if phase == 'prefill':
        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace, 'NHD', backend='fa2')
        wrapper.plan(torch.arange(batch+1,device='cuda',dtype=torch.int32)*qlen,
            torch.arange(batch+1,device='cuda',dtype=torch.int32)*pages, table.flatten(),
            torch.full((batch,), (length-1)%page+1,device='cuda',dtype=torch.int32),
            heads,kh,dim,page,causal=True,q_data_type=q.dtype,kv_data_type=k.dtype)

    class Impl:
        scale = dim**-.5
        def forward(self, layer, query, key, value, kv_cache, metadata, output):
            if wrapper is not None:
                return wrapper.run(query, (k,v),k_scale=.8,v_scale=.6,out=output)
            return xqa_batch_decode_with_kv_cache(query=query,kv_cache=(k,v),workspace_buffer=workspace,
                block_tables=table,seq_lens=lengths,max_seq_len=length,bmm1_scale=dim**-.5*.8,
                bmm2_scale=.6,out=output,kv_layout='NHD',q_len_per_req=qlen,mask=mask)

    layers = [SimpleNamespace(num_heads=heads, head_dim=dim,q_norm=SimpleNamespace(weight=q),
        attn=SimpleNamespace(layer_name=f'model.layers.{n}.attn',impl=Impl())) for n in (3,31,63)]
    names = [layer.attn.layer_name for layer in layers]
    originals = [layer.attn.impl.forward for layer in layers]
    runner = SimpleNamespace(model=SimpleNamespace(named_modules=lambda: zip(names,layers)),
        _kvmem_layers=names,max_num_tokens=batch*qlen,max_num_reqs=batch)
    options = SimpleNamespace(action_native_scores='reuse',action_layers='all',
        action_native_max_rows=max(32,batch*(len(sampled) if phase=='prefill' else qlen)),action_native_max_bytes=2*1024**3,
        action_scope='non-thinking',action_interval=16,action_decay=.5,
        prefill_attention=True,prefill_attention_interval=1024,prefill_attention_stride=32,
        prefill_attention_half_life=8192,
        memory=SimpleNamespace(select_budget=36864,gen_budget=16384,block_tokens=32))
    bank = NativeAttentionStats(runner,options,args.output.parent/'build')
    if bank.disabled:
        raise RuntimeError(bank.disabled)
    metadata = SimpleNamespace(prefill=SimpleNamespace(wrapper=wrapper) if wrapper else None,
        q_data_type_prefill=q.dtype,num_decode_tokens=0 if wrapper else batch*qlen)
    output = torch.empty_like(q)
    starts = list(range(0,length,32)); counts = [min(32,length-p) for p in starts]
    scale = torch.tensor(.8,device='cuda'); qscale = torch.ones((),device='cuda')
    trig = torch.cat((torch.ones(length,dim//4,device='cuda'),torch.zeros(length,dim//4,device='cuda')),-1)
    for layer in layers:
        layer.num_kv_heads = kh
        layer.rotary_emb = SimpleNamespace(cos_sin_cache=trig)
        layer.attn.kv_cache, layer.attn._k_scale, layer.attn._q_scale = cache, scale, qscale
    request_ids = [str(i) for i in range(batch)]
    runner.req_states = SimpleNamespace(req_id_to_index={request:i for i,request in enumerate(request_ids)})
    runner.block_tables = SimpleNamespace(block_tables=[SimpleNamespace(gpu=table)])
    positions = torch.arange(length-qlen,length,device='cuda').repeat(batch)
    input_batch = SimpleNamespace(num_reqs=batch,req_ids=request_ids,
        is_prefilling_np=[phase=='prefill']*batch,query_start_loc_np=[i*qlen for i in range(batch+1)],
        num_computed_tokens_np=[length-qlen]*batch,seq_lens_cpu_upper_bound=[length]*batch,
        input_ids=torch.full((batch*qlen,),20,device='cuda',dtype=torch.int32),positions=positions)
    captured = [SimpleNamespace(query=q,positions=positions) for _ in layers]
    records=[(i,1,start,n,start) for i,(start,n) in enumerate(zip(starts,counts))]
    authorities = {request:SimpleNamespace(action_history=None,block_graph=None,
        memory=SimpleNamespace(_store=SimpleNamespace(working_records=lambda:records))) for request in request_ids}
    action = ActionRuntime.__new__(ActionRuntime)
    action.runner,action.options = runner,options
    action.runtime = SimpleNamespace(config=options.memory,output=args.output.parent,
        ensure_authority=lambda request:authorities[request],drain_requests=lambda *a,**k:True)
    action.native,action.native_plans,action.metadata = None,None,None
    action.layers,action.groups = list(enumerate(layers)),dict.fromkeys(names,0)
    action.q_dtypes = action.prefill_q_dtypes = dict.fromkeys(names,torch.bfloat16)
    action.ids = [10,11,12,13,14]
    action.profile = action.metadata_upload = None
    action.capture_seconds = action.candidate_samples = action.packets = 0
    action.metadata_early_batches = action.metadata_fallback_batches = action.metadata_wait_seconds = 0

    def run(mode):
        # Reset the policy state for fixed, operand-identical sampling on every
        # repetition. Archive/accepted-history publication is a separate gate.
        for request,authority in authorities.items():
            authority.action_history = ActionHistory(options,action.ids)
            authority.action_history.begin(request,[14])
        action.native = bank if mode=='reuse' else None
        action.prepare(input_batch,positions)
        for layer, original in zip(layers, originals):
            forward = layer.attn.impl.forward if mode == 'reuse' else original
            forward(layer.attn,q,None,None,None,metadata,output)
        packets = action.capture(captured,input_batch)
        for packet in packets.values():
            if packet.event is not None:packet.event.synchronize()
        if mode=='reuse' and any(k.startswith('fallback:') for k in bank.counts):
            raise RuntimeError('Component reuse unexpectedly fell back: '+str(bank.counts))
        return [(packets[request].history,packets[request].masses.clone()) for request in request_ids]

    # Compile and warm both operand-identical arms before timing.
    old, new = run('bypass'), run('reuse')
    def logical(values):
        history,mass=values
        keys=sorted(set(history));indices={key:i for i,key in enumerate(keys)}
        result=torch.zeros((mass.shape[0],len(keys)),dtype=mass.dtype)
        for col,key in enumerate(history):result[:,indices[key]]+=mass[:,col]
        return keys,result
    delta=0.
    for a,b in zip(old,new):
        ka,a=logical(a);kb,b=logical(b)
        assert ka==kb
        delta=max(delta,float((a-b).abs().max()))
    for _ in range(2):
        run('reuse'); run('bypass')
    return run, bank, dict(phase=phase,batch=batch,active_tokens=length,query_tokens_per_request=qlen,
        sampled_rows_per_request=len(sampled),layers=3,q_heads=heads,kv_heads=kh,head_dim=dim,
        page_tokens=page,max_abs_difference=delta,additional_buffer_bytes=bank.allocated_bytes,
        baseline_metadata_preuploaded=False,source_fingerprint=bank.source_fingerprint,
        component_driver='ActionRuntime.prepare/capture; reference metadata is uploaded only when needed',
        window_segment_iteration_included=True,store_directory_snapshot_serialization_included=False,
        acceptance_and_archive_publication_included=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--length',type=int,default=53152)
    parser.add_argument('--chunk',type=int,default=1024)
    parser.add_argument('--repeats',type=int,default=6)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError('Use a new output file; evidence is not overwritten')
    results = []
    for phase in ('prefill','decode'):
        run, bank, result = setup(args,phase)
        measurements = []
        for repetition in range(args.repeats):
            for mode in (('bypass','reuse') if repetition%2 == 0 else ('reuse','bypass')):
                torch.cuda.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                began = time.perf_counter(); start.record()
                run(mode); end.record(); end.synchronize()
                measurements.append(dict(repetition=repetition,mode=mode,
                    wall_ms=(time.perf_counter()-began)*1000,cuda_interval_ms=start.elapsed_time(end)))
        result['samples'] = measurements
        result['mean_wall_ms'] = {mode:statistics.mean(r['wall_ms'] for r in measurements if r['mode']==mode)
                                  for mode in ('bypass','reuse')}
        result['counts'] = dict(bank.counts)
        results.append(result)
        del run,bank
        torch.cuda.empty_cache()
    evidence = dict(scope='Fixed operands; three native layers plus ALL sampler work and D2H. No model throughput claim.',
        gpu=torch.cuda.get_device_name(),results=results,
        component_performance_passed=all(r['mean_wall_ms']['reuse'] < r['mean_wall_ms']['bypass'] for r in results))
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({r['phase']:r['mean_wall_ms'] for r in results},indent=2))


if __name__ == '__main__':
    main()
