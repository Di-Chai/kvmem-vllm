"""Paired archived-operand preparation plus scoring; never a model throughput claim."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace as NS

import torch

from vllm_kvmem.recall_value_reference import score_value_records
from vllm_kvmem.recall_value_scoring import VALUE_METHODS


def run(args):
    used,util=map(int,subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip().split(','))
    if used>1000 or util>5:raise ValueError('GPU occupied; do not benchmark over another job')
    if 'RTX PRO 6000' not in torch.cuda.get_device_name():raise ValueError('Expected local Pro6000')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    os.environ.update(KVMEM_RECALL_DEVICE='cuda',KVMEM_RECALL_QUERY_BATCH='128')
    gen=torch.Generator().manual_seed(2601101)
    shape=(args.blocks,args.layers,32,4,256)
    keys=torch.randn(shape,generator=gen).to(torch.float8_e4m3fn)
    values=torch.randn(shape,generator=gen).to(torch.float8_e4m3fn).view(torch.uint8)
    counts=[32]*args.blocks;counts[-1]=13
    records=[NS(handle=NS(slot=i,generation=1),history_begin=i*32,count=n) for i,n in enumerate(counts)]
    authority=NS(scoring_layers=args.layers,shape=(args.layers,32,4,256),
        records={i:r for i,r in enumerate(records)},mean_k_backend='qw3_cuda',
        recall_value_scales=torch.linspace(.371,1.293,args.layers))
    authority.read_records=lambda hs:[(keys[h.slot,:,:counts[h.slot]],
        values[h.slot,:,:counts[h.slot]].contiguous()) for h in hs]
    queries=torch.randn(args.layers,args.queries,16,256,generator=gen)
    variants={'legacy':('streaming',128),'packed':('streaming_packed',128),
              'packed_wide':('streaming_packed',1024)}
    measurements={m:{v:[] for v in variants} for m in VALUE_METHODS};scores={m:{} for m in VALUE_METHODS}

    def evaluate(method,name):
        backend,batch=variants[name]
        os.environ.update(KVMEM_VALUE_RECALL_BACKEND=backend,KVMEM_RECALL_BLOCK_BATCH=str(batch))
        authority._value_recall_records={}
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode():out,stats=score_value_records(authority,records,queries,[True]*args.blocks,method)
        torch.cuda.synchronize()
        return out,dict(seconds=time.perf_counter()-start,stats=stats,
                        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())

    for method in VALUE_METHODS:
        for name in variants:evaluate(method,name)
        for repeat in range(args.repeats):
            for name in (list(variants) if repeat%2==0 else list(reversed(variants))):
                scores[method][name],report=evaluate(method,name)
                measurements[method][name].append(report)
    differences={};selections={}
    def pick(v):return sorted([0,*sorted(range(1,len(v)),key=lambda i:(-v[i],i))[:args.blocks//4-1]])
    for method,results in scores.items():
        differences[method]={};selections[method]={}
        for name in ('packed','packed_wide'):
            a=torch.tensor(results['legacy'],dtype=torch.float64);b=torch.tensor(results[name],dtype=torch.float64)
            torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
            differences[method][name]=float((a-b).abs().max())
            selections[method][name]=pick(results['legacy'])==pick(results[name])
            if not selections[method][name]:raise ValueError('Synthetic selected set changed')
    means={m:{v:statistics.mean(r['seconds'] for r in reports) for v,reports in variants_.items()}
           for m,variants_ in measurements.items()}
    report=dict(status='passed',gpu=torch.cuda.get_device_name(),measurements=measurements,
        geometry=dict(blocks=args.blocks,layers=args.layers,queries=args.queries,kv_heads=4,query_heads=16,dim=256,block_tokens=32),
        mean_seconds=means,speedup={m:means[m]['legacy']/means[m]['packed_wide'] for m in means},
        max_abs_difference=differences,selected_sets_equal=selections,
        scope='Synthetic archived FP8 CPU operands, cold adapter caches, GPU scoring. Native archive/model validation is separate.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('Do not replace existing benchmark evidence')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','mean_seconds','speedup','max_abs_difference','selected_sets_equal']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layers',type=int,default=2);p.add_argument('--blocks',type=int,default=1024)
    p.add_argument('--queries',type=int,default=46);p.add_argument('--repeats',type=int,default=2)
    run(p.parse_args())
