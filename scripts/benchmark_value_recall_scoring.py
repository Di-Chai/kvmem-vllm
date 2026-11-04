"""Controlled scoring-only efficiency gate, with identical frozen synthetic Q/K/V.

This is neither model prefill throughput nor a RULER task-accuracy result. The
real-model qualification subsequently compares native selections and whole-set
output errors on original task states before admitting a full scored campaign.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from vllm_kvmem.recall_value_scoring import REFERENCE_METHODS, aggregate_scores, iter_attention_moments
from vllm_kvmem.recall_value_streaming import streaming_scores, streaming_value_norm


def benchmark(args):
    if args.device=='cuda':
        row=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        used,util=map(int,row.split(','))
        if used>1000 or util>5:
            raise RuntimeError('GPU is occupied; no benchmark was started')
        if 'RTX PRO 6000' not in torch.cuda.get_device_name():
            raise ValueError('This campaign targets the local RTX PRO 6000')
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    g=torch.Generator().manual_seed(1101260)
    shape=(args.layers,args.blocks,32,args.kv_heads,args.dim)
    keys=torch.randn(shape,generator=g)
    values=torch.randn(shape,generator=g)
    queries=torch.randn(args.layers,args.queries,args.kv_heads*6,args.dim,generator=g)
    counts=[32]*args.blocks;counts[-1]=19;mask=[True]*args.blocks
    def reader(b,e,l,h):return keys[l,b:e,:,h],values[l,b:e,:,h]
    def evaluate(backend):
        if args.device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        with torch.inference_mode():
            if backend=='reference':
                iterator=iter_attention_moments(queries,reader,counts,mask,kv_heads=args.kv_heads,
                    device=args.device,query_batch=4,block_batch=16)
                result,stats=aggregate_scores(iterator,layers=args.layers,heads=args.kv_heads*6)
            elif backend=='streaming':
                result,stats=streaming_scores(queries,reader,counts,mask,kv_heads=args.kv_heads,
                    device=args.device,query_batch=128,block_batch=128)
            else:
                value,stats=streaming_value_norm(queries,reader,counts,mask,kv_heads=args.kv_heads,
                    device=args.device,query_batch=128,block_batch=128)
                result={'block_value_norm':value}
        if args.device=='cuda':torch.cuda.synchronize()
        return result,dict(seconds=time.perf_counter()-start,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
            stats=stats)
    measurements={b:[] for b in ('reference','streaming','value_norm')};results={}
    # One untimed warmup per backend, then alternating paired repetitions.
    for backend in measurements:evaluate(backend)
    for repeat in range(args.repeats):
        for backend in (('reference','streaming','value_norm') if repeat%2==0 else ('value_norm','streaming','reference')):
            results[backend],row=evaluate(backend);measurements[backend].append(row)
    differences={};sets={}
    def pick(scores):
        optional=sorted(range(1,args.blocks),key=lambda i:(-scores[i],i))
        return sorted([0,*optional[:max(1,args.blocks//4)-1]])
    for method in REFERENCE_METHODS:
        a=torch.tensor(results['reference'][method],dtype=torch.float64)
        b=torch.tensor(results['streaming'][method],dtype=torch.float64)
        torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-5)
        differences[method]=float((a-b).abs().max())
        sets[method]=pick(a.tolist())==pick(b.tolist())
        if not sets[method]:raise ValueError('Streaming changes selected blocks for '+method)
    torch.testing.assert_close(torch.tensor(results['reference']['block_value_norm']),
        torch.tensor(results['value_norm']['block_value_norm']),atol=3e-5,rtol=3e-5)
    if pick(results['reference']['block_value_norm'])!=pick(results['value_norm']['block_value_norm']):
        raise ValueError('One-pass ValueNorm changes selected blocks')
    averages={b:statistics.mean(v['seconds'] for v in rows) for b,rows in measurements.items()}
    report=dict(status='passed',device=args.device,
        gpu=torch.cuda.get_device_name() if args.device=='cuda' else None,
        geometry=dict(layers=args.layers,blocks=args.blocks,block_tokens=32,
            kv_heads=args.kv_heads,query_heads=args.kv_heads*6,queries=args.queries,dim=args.dim),
        measurements=measurements,mean_seconds=averages,speedup=averages['reference']/averages['streaming'],
        value_norm_speedup=averages['reference']/averages['value_norm'],
        max_abs_score_difference=differences,selected_sets_equal=sets,
        scope='Synthetic scoring only; no task accuracy or model-throughput claim')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise ValueError('Keep existing benchmark evidence immutable')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    for name,default in (('layers',2),('blocks',256),('queries',128),('kv-heads',2),('dim',128),('repeats',2)):
        p.add_argument('--'+name,type=int,default=default)
    benchmark(p.parse_args())
