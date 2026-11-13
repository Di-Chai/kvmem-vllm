"""Finite synthetic-geometry MeanK timing; not full-model accuracy evidence.

Run only with exclusive GPU ownership. The 8183-page long Query and 12823-page
short Queries match the saved Query diagnostic geometry. Every backend sees
the identical FP16 operands, complete eligible domain and original normalization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


def comparison(before, after, quota):
    difference=(after-before).abs()
    expected=before.topk(quota).indices.sort().values
    actual=after.topk(quota).indices.sort().values
    ordered=before.sort(descending=True).values
    return dict(max_abs_error=float(difference.max()),
        max_relative_error=float((difference/before.abs().clamp_min(1e-12)).max()),
        rms_error=float(difference.square().mean().sqrt()),
        bit_equal=torch.equal(before,after),
        close_reference_tolerance=bool(torch.allclose(before,after,rtol=5e-5,atol=2e-5)),
        selected_set_equal=torch.equal(expected,actual),
        selected_intersection=len(set(expected.tolist())&set(actual.tolist())),quota=quota,
        baseline_cutoff_margin=float(ordered[quota-1]-ordered[quota]) if quota<len(before) else None)


def measure_case(tokens, pages, warmup, rounds, output_dir):
    from vllm_kvmem.core.cuda_scoring import prepare_mean_key_index,prepared_layer_scores
    generator=torch.Generator(device='cuda').manual_seed(7901+tokens)
    means=torch.randn(1,pages+41,4,256,device='cuda',dtype=torch.bfloat16,generator=generator)
    mask=torch.ones(pages+41,device='cuda',dtype=torch.bool);mask[:41]=False
    queries=torch.randn(1,tokens,24,256,device='cuda',dtype=torch.bfloat16,generator=generator)
    index=prepare_mean_key_index(means,mask)
    assert index.indices.numel()==pages
    def score(backend):
        return prepared_layer_scores(queries,index,0,backend=backend)
    for _ in range(warmup):
        score('legacy');score('tiled')
    torch.cuda.synchronize()
    timings={name:[] for name in ('legacy','tiled')}
    snapshots={name:[] for name in timings}
    for _ in range(rounds):
        for backend in ('legacy','tiled','tiled','legacy'):
            started,finished=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();started.record()
            result=score(backend)
            finished.record();finished.synchronize()
            timings[backend].append(dict(cuda_ms=started.elapsed_time(finished),
                                         wall_ms=1000*(time.perf_counter()-wall)))
            snapshots[backend].append(result[41:].detach().cpu())
    baseline,candidate=snapshots['legacy'][0],snapshots['tiled'][0]
    quota=min(1111,pages)
    score_file=output_dir/f'scores-t{tokens}-b{pages}.pt'
    torch.save(dict(legacy=snapshots['legacy'],tiled=snapshots['tiled']),score_file)
    report=dict(query_shape=[1,tokens,24,256],means_shape=[1,pages,4,256],
        input_dtype='bfloat16',scoring_dtype='float16',eligible_pages=pages,mandatory_pages=41,
        warmup=warmup,rounds=rounds,order_per_round=['legacy','tiled','tiled','legacy'],
        timings=timings,median_cuda_ms={k:statistics.median(t['cuda_ms'] for t in v) for k,v in timings.items()},
        candidate_comparison=comparison(baseline,candidate,quota),
        repeated_legacy_comparison=comparison(baseline,snapshots['legacy'][-1],quota),
        repeated_tiled_comparison=comparison(candidate,snapshots['tiled'][-1],quota),
        finite=all(torch.isfinite(s).all().item() for values in snapshots.values() for s in values),
        mass={k:[float(s.sum()) for s in v] for k,v in snapshots.items()},
        saved_scores=str(score_file.resolve()))
    report['cuda_reduction_percent']=100*(1-report['median_cuda_ms']['tiled']/report['median_cuda_ms']['legacy'])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args()
    if args.warmup<1 or args.rounds<1: parser.error('Require positive warmup and rounds')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    from vllm_kvmem.core.cuda_scoring import kernel
    kernel()  # Exclude extension compilation and loading from timing.
    report=dict(scope='Synthetic geometry kernel diagnostic only; does not prove real Query ranks, '
                      'full output equivalence, TTFT or prefill throughput.',
        seed_rule='7901 + Query tokens',device=torch.cuda.get_device_name(),cases=[])
    package=Path(__import__('vllm_kvmem.core.cuda_scoring',fromlist=['x']).__file__).parent
    report['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (package/'cuda_scoring.py',package/'cuda_sources/qw3_mean_k.cu',
         package/'cuda_sources/qw3_mean_k_kernel.cuh',package/'cuda_sources/qw3_mean_k_tiled.cuh')}
    for tokens,pages in ((175,12823),(205,12823),(3718,8183)):
        report['cases'].append(measure_case(tokens,pages,args.warmup,args.rounds,args.output_dir))
    with (args.output_dir/'comparison.json').open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps([dict(tokens=c['query_shape'][1],pages=c['eligible_pages'],
        median_cuda_ms=c['median_cuda_ms'],comparison=c['candidate_comparison']) for c in report['cases']],indent=2))


if __name__=='__main__':main()
