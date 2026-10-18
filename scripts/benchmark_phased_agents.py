"""Fixed-window comparison of continuous and phased public scheduling.

Both arms retain the same offline policy, CPU gather, attention scoring, model
and RPC boundaries. Only P1 enables the experimental public phase gate.
"""
import asyncio
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('phase_shared_benchmark', ROOT/'scripts/benchmark_timed_agents.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def phase_profiles(candidate, wave_tokens, max_wait_ms):
    import copy
    from vllm_kvmem.serve import build_launch
    profiles = {}
    for arm in ('A0', 'P1'):
        profile = copy.deepcopy(candidate)
        profile['kvmem'].update(offline_phase_policy=arm == 'P1',
            phase_prefill_wave_tokens=wave_tokens, phase_max_wait_ms=max_wait_ms)
        build_launch(profile, ROOT, ROOT/'unused-phase-validation', check_paths=False)
        profiles[arm] = profile
    return profiles


def prepare(args):
    benchmark.prepare(args)
    root = args.output.resolve()
    benchmark.q.save(root/'status.json', dict(status='preparing_phase_profiles', at=time.time()))
    try:
        profiles = phase_profiles(benchmark.read(root/'P1.json'), args.wave_tokens, args.phase_wait_ms)
        for arm, profile in profiles.items():
            benchmark.q.save(root/(arm+'.json'), profile)
        manifest = benchmark.read(root/'manifest.json')
        manifest.update(comparison_kind='public_phases',
            phase_parameters=dict(prefill_wave_tokens=args.wave_tokens, max_wait_ms=args.phase_wait_ms),
            profile_sha256={arm:hashlib.sha256((root/(arm+'.json')).read_bytes()).hexdigest() for arm in profiles},
            phase_scope='Same original RPCs and native authority limit; no extra CPU decode-ready pool. '
                        'Separates public batches only; Query replay and restoration retain their original semantics.')
        benchmark.q.save(root/'manifest.json', manifest)
        benchmark.q.save(root/'status.json', dict(status='prepared', at=time.time()))
    except BaseException as error:
        benchmark.q.save(root/'status.json', dict(status='failed', gpu_started=False, error=repr(error), at=time.time()))
        raise


def validate_prepared(root):
    if ROOT.resolve() != (root/'frozen').resolve():
        raise ValueError('Execute the frozen phased benchmark, not the live checkout')
    manifest = benchmark.read(root/'manifest.json')
    if manifest.get('comparison_kind') != 'public_phases':
        raise ValueError('Prepared experiment is not a public phase comparison')
    profiles = {arm:benchmark.read(root/(arm+'.json')) for arm in ('A0', 'P1')}
    import copy
    baseline = copy.deepcopy(profiles['A0'])
    candidate = copy.deepcopy(profiles['P1'])
    if (baseline['kvmem'].pop('offline_phase_policy', None) is not False or
            candidate['kvmem'].pop('offline_phase_policy', None) is not True or
            baseline != candidate or not baseline['kvmem'].get('offline_agent_policy')):
        raise ValueError('Phase arms may differ only in the explicit phase opt-in')


def main(argv=None):
    p = benchmark.parser()
    p.description = __doc__
    p.add_argument('--wave-tokens', type=int, default=8192)
    p.add_argument('--phase-wait-ms', type=float, default=1000.)
    args = p.parse_args(argv)
    if args.run_prepared:
        if args.prepare_only or args.config or args.fixture:
            p.error('Frozen execution uses the prepared profiles and workload')
        validate_prepared(args.output.resolve())
        return asyncio.run(benchmark.run(args))
    if not args.prepare_only or not args.config or not args.fixture:
        p.error('Prepare explicitly, then execute this experiment frozen script with --run-prepared')
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or args.pairs < 1 or
            not math.isfinite(args.idle_seconds) or args.idle_seconds < 0 or
            args.warmup_turns < 1 or args.epoch_turns <= args.warmup_turns or
            args.output_tokens < 1 or args.context_tokens < 1024 or
            len(set(args.modes)) != len(args.modes)):
        p.error('Invalid fixed window or workload')
    prepare(args)


if __name__ == '__main__':
    main()
