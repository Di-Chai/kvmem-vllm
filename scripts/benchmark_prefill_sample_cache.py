"""Component-only static FA2 sample-ID shared cache, unchanged floating body."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

from prefill_sample_cache import build_modules


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    if args.output.exists():
        raise ValueError('Use a fresh result path')
    modules=build_modules(args.output)
    result=dict(scope=__doc__,compiled=modules[3],production_changed=False,results=[])
    if args.prepare_only:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        return
    if args.prepared is None:
        raise ValueError('A separately prepared compiled identity is required')
    old=json.loads(args.prepared.read_text())
    if old['compiled']!=result['compiled']:
        raise ValueError('Compiled source/library changed since preparation')
    import torch
    import benchmark_native_prefill_stats_floor as floor
    import benchmark_native_prefill_readonly as harness
    for qlen,length in [(2048,16384),(2048,32768),(2048,49152),
                        (8192,32768),(8192,49152),(14336,32768),(14336,49152)]:
        execute,original,arms,audit,item=harness.setup(length,args.output,
            qlen=qlen,max_rows=512,modules=modules)
        floor.qualify(original,audit)
        item['bitwise_output_row_lse_full_piece_lse_scores']=True
        item['samples']=[]
        for _ in range(2):
            for mode in ('static','readonly'):
                value=execute(mode);del value
        for repetition in range(args.rounds):
            for mode in (('static','readonly') if repetition%2==0 else ('readonly','static')):
                torch.cuda.synchronize()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                began=time.perf_counter();start.record()
                value=execute(mode);end.record();end.synchronize()
                item['samples'].append(dict(repetition=repetition,
                    mode='reference' if mode=='static' else 'candidate',
                    wall_ms=1000*(time.perf_counter()-began),cuda_ms=start.elapsed_time(end)))
                del value
        item['summary']={mode:{key:statistics.mean(row[key] for row in item['samples'] if row['mode']==mode)
                             for key in ('wall_ms','cuda_ms')} for mode in ('reference','candidate')}
        item['dispatch_audit']=audit()
        item['candidate_checked_dispatches']=modules[2].checked_dispatches
        if not item['candidate_checked_dispatches']:
            raise ValueError('Candidate did not execute')
        for _,bank,_,_ in arms.values():
            if not bank.static_prefill_counts.get('static_dispatch_calls') or any(
                    key.startswith('generic_dispatch:') for key in bank.static_prefill_counts):
                raise ValueError('Unexpected generic statistics fallback')
        # Independent dispatch proof; profiling never enters paired timings.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as trace:
            profiled=execute('readonly')
            torch.cuda.synchronize()
            del profiled
        kernels=[event.name for event in trace.events() if str(event.device_type).endswith('CUDA')
                 and 'BatchPrefillWithPagedKVCacheKernel' in event.name]
        if len(kernels)!=3 or any('KVMemStaticCachedAttention' not in name for name in kernels):
            raise ValueError('Actual candidate FA2 kernel differs from the prepared module')
        item['actual_candidate_kernels']=kernels
        result['results'].append(item)
        print(json.dumps(dict(q=qlen,kv=length,summary=item['summary'])),flush=True)
        del execute,original,arms,audit,item
        gc.collect();torch.cuda.synchronize()
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--rounds',type=int,default=12)
    args=parser.parse_args()
    if not 2 <= args.rounds <= 30:
        parser.error('Use bounded paired rounds')
    run(args)
