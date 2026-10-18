"""Frozen, single-factor offline prefill-coalescing comparison."""
import asyncio
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('coalesced_shared_benchmark',ROOT/'scripts/benchmark_timed_agents.py')
benchmark=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=benchmark
spec.loader.exec_module(benchmark)


def coalesced_profiles(candidate, maximum):
    from vllm_kvmem.serve import build_launch
    result={}
    for arm in ('A0','P1'):
        p=copy.deepcopy(candidate)
        p['kvmem'].update(offline_agent_policy=True,offline_phase_policy=False,
            offline_prefill_coalescing=arm=='P1',offline_prefill_max_tokens=maximum,
            step_latency_interval=1)
        build_launch(p,ROOT,ROOT/'unused-coalesced-validation',check_paths=False)
        result[arm]=p
    return result


def validate_prepared(root):
    if ROOT.resolve()!=(root/'frozen').resolve():
        raise ValueError('Execute this experiment frozen script')
    manifest=benchmark.read(root/'manifest.json')
    if manifest.get('comparison_kind')!='offline_prefill_coalescing':
        raise ValueError('Not a prepared coalescing experiment')
    a,b=[copy.deepcopy(benchmark.read(root/(arm+'.json'))) for arm in ('A0','P1')]
    if (a['kvmem'].pop('offline_prefill_coalescing',None) is not False or
            b['kvmem'].pop('offline_prefill_coalescing',None) is not True or a!=b or
            not a['kvmem']['offline_agent_policy'] or a['kvmem'].get('offline_phase_policy') or
            a['kvmem']['step_latency_interval']!=1):
        raise ValueError('Only coalescing may differ; both arms must retain continuous offline scheduling and full timing')


def main(argv=None):
    parser=benchmark.parser();parser.description=__doc__
    parser.add_argument('--coalesced-max-tokens',type=int,default=4096)
    args=parser.parse_args(argv)
    if args.run_prepared:
        if args.config or args.fixture or args.prepare_only:
            parser.error('Frozen execution uses prepared input only')
        validate_prepared(args.output.resolve())
        return asyncio.run(benchmark.run(args))
    if not args.prepare_only or not args.config or not args.fixture:
        parser.error('Prepare explicitly, then execute the frozen script')
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds<=0 or args.pairs<1 or
            not math.isfinite(args.idle_seconds) or args.idle_seconds<0 or args.warmup_turns<1 or
            args.epoch_turns<=args.warmup_turns or min(args.output_tokens,args.context_tokens)<1 or
            len(set(args.modes))!=len(args.modes)):
        parser.error('Invalid fixed window or workload')
    benchmark.prepare(args)
    root=args.output.resolve()
    try:
        profiles=coalesced_profiles(benchmark.read(root/'P1.json'),args.coalesced_max_tokens)
        for arm,p in profiles.items():benchmark.q.save(root/(arm+'.json'),p)
        manifest=benchmark.read(root/'manifest.json')
        manifest.update(comparison_kind='offline_prefill_coalescing',
            profile_sha256={a:hashlib.sha256((root/(a+'.json')).read_bytes()).hexdigest() for a in profiles},
            coalescing_scope='Continuous decode remains eligible; bounded calibration and coalescing of ordinary long prefill only. '
                'Identical original RPCs, context scoring, CPU residency, native slots and timing in both arms.')
        benchmark.q.save(root/'manifest.json',manifest)
        benchmark.q.save(root/'status.json',dict(status='prepared',at=time.time(),gpu_started=False))
    except BaseException as error:
        benchmark.q.save(root/'status.json',dict(status='failed',gpu_started=False,error=repr(error),at=time.time()))
        raise


if __name__=='__main__':main()
