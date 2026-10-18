"""Freeze and compare cap=0/2048 using identical finite decode/prefill work.

Preparation is CPU only. Execute only the resulting frozen copy. The shared
benchmark retains GPU ownership, foreign-process checks, ordered pairs, native
counter reconciliation and complete owned-server cleanup.
"""
import asyncio
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('wave_shared_benchmark', ROOT / 'scripts/benchmark_timed_agents.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)
wave = benchmark.module('fixed_prefill_wave', ROOT / 'scripts/prefill_wave_workload.py')
KIND = 'fixed_work_shared_prefill_cap'
ARM_SCOPE = ('A0/P1 are inherited harness labels only: both use the same qualified P1 runtime and profile, '
             'with cap=0/cap>0. A0 here is not the optimization goal baseline with three feature switches disabled.')


def validate_arguments(args):
    integers = (args.resident_requests, args.seed_requests, args.context_tokens, args.output_tokens,
                args.seed_context_tokens, args.seed_output_tokens, args.prefill_cap_tokens, args.pairs)
    if any(type(n) is not int or n <= 0 for n in integers):
        raise ValueError('Cohort sizes, inputs, quotas, cap and pair count must be positive integers')
    if (args.concurrency != 8 or args.resident_requests+args.seed_requests != args.concurrency
            or args.modes != ['saturated'] or args.warmup_turns != 1):
        raise ValueError('This probe requires eight slots, one warmup and a single saturated mode')
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or
            not math.isfinite(args.seed_offset_seconds) or not 0 < args.seed_offset_seconds < args.duration_seconds
            or not math.isfinite(args.idle_seconds) or args.idle_seconds < 0):
        raise ValueError('Require a finite window, an internal fixed seed offset and nonnegative idle guard')
    if min(args.context_tokens, args.seed_context_tokens) < 2:
        raise ValueError('Need at least two input tokens')


def cap_profiles(candidate, cap):
    from vllm_kvmem.serve import build_launch
    profiles = {}
    for arm, amount in (('A0', 0), ('P1', cap)):
        profile = copy.deepcopy(candidate)
        profile['kvmem'].update(offline_agent_policy=True, offline_phase_policy=False,
            offline_prefill_coalescing=False, adaptive_prefill=False,
            prefill_batch_tokens=amount, step_latency_interval=1)
        build_launch(profile, ROOT, ROOT/'unused-wave-validation', check_paths=False)
        profiles[arm] = profile
    return profiles


def validate_prepared(root):
    manifest = benchmark.read(root/'manifest.json')
    if manifest.get('comparison_kind') != KIND or 'runtime_sha256' in manifest:
        raise ValueError('Require one shared frozen runtime for the cap-only probe')
    profiles = []
    for arm, amount in (('A0', 0), ('P1', manifest['workload']['prefill_cap_tokens'])):
        path = root/(arm+'.json')
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['profile_sha256'][arm]:
            raise ValueError('Prepared profile changed')
        profile = benchmark.read(path)
        settings = profile['kvmem']
        if (settings.pop('prefill_batch_tokens', None) != amount or
                not settings.get('offline_agent_policy') or settings.get('offline_phase_policy') or
                settings.get('offline_prefill_coalescing') or settings.get('adaptive_prefill') or
                settings.get('step_latency_interval') != 1):
            raise ValueError('Unqualified scheduling combination in cap probe')
        profiles.append(profile)
    if profiles[0] != profiles[1]:
        raise ValueError('Only prefill_batch_tokens may differ between arms')
    return manifest


def compare(measurements):
    """A slower complete fixed workload cannot be a win from shifting prefill."""
    grouped = {(r['pair'], r['arm']):r['results']['saturated'] for r in measurements}
    results = []
    for pair in sorted(i for i,arm in grouped if arm == 'A0' and (i,'P1') in grouped):
        a, b = (grouped[pair,arm] for arm in ('A0','P1'))
        for row in (a, b):
            if (row.get('status') != 'completed' or not row.get('drained') or
                    not row.get('cleanup_complete') or row.get('cancelled_requests') or
                    not row.get('native_client_total_output_reconciliation', {}).get('passed')):
                raise ValueError('Incomplete, cancelled or unreconciled work cannot be compared')
        if any(a[key] != b[key] for key in ('work_identity', 'fixed_work_input_tokens', 'fixed_work_output_tokens')):
            raise ValueError('Compared arms did not execute identical fixed input/quota work')
        coverage = a['coverage_valid'] and b['coverage_valid']
        completion_ratio = a['full_work_makespan_seconds']/b['full_work_makespan_seconds']
        retired_ratio = a['full_work_retired_makespan_seconds']/b['full_work_retired_makespan_seconds']
        output_ratio = b['output_tokens_per_second']/a['output_tokens_per_second'] if a['output_tokens_per_second'] else None
        prefill_equal = (a.get('full_public_prefill', {}).get('passed') is True and
                         b.get('full_public_prefill', {}).get('passed') is True and
                         a['full_public_prefill']['completed_tokens'] == b['full_public_prefill']['completed_tokens'])
        a_steps, b_steps = a['public_mixed_steps'], b['public_mixed_steps']
        mechanism = (a_steps['mixed_with_multiple_prefills'] > 0 and
                     (a_steps['maximum_ordinary_tokens'] or 0) > b['prefill_cap_tokens'] and
                     b_steps['ordinary_only_mixed_steps'] > 0 and
                     b_steps['ordinary_steps_exceeding_cap'] == 0)
        results.append(dict(pair=pair, coverage_valid=coverage, full_prefill_demand_equal=prefill_equal,
            mechanism_observed=mechanism,
            fixed_work_speedup=completion_ratio, window_output_ratio=output_ratio,
            fixed_work_and_retirement_speedup=retired_ratio,
            window_output_only_not_an_offline_win=bool(output_ratio and output_ratio > 1 and
                                                      (completion_ratio <= 1 or retired_ratio <= 1)),
            faster_complete_fixed_work=coverage and prefill_equal and mechanism and
                                       completion_ratio > 1 and retired_ratio > 1,
            seed_ttft_seconds={arm:[s['full_ttft_seconds'] for s in row['seeds']] for arm,row in (('A0',a),('P1',b))},
            seed_progress_at_t1={arm:[row['seed_completed_tokens_lower_bound_at_t1'],
                                      row['seed_completed_tokens_upper_bound_at_t1']] for arm,row in (('A0',a),('P1',b))},
            decode_overlap={arm:row['decode_seed_overlap'] for arm,row in (('A0',a),('P1',b))},
            t0_resident_offsets={arm:row['t0_resident_offsets'] for arm,row in (('A0',a),('P1',b))}))
    return dict(kind=KIND, completed_pairs=len(results), pairs=results,
                arm_scope=ARM_SCOPE,
                scope='Mechanism probe only; report overlap, prefill progress, TTFT and complete work together. '
                      'No continuous-agent performance target is awarded from this finite workload.')


def prepare(args):
    validate_arguments(args)
    original = benchmark.prepare_fixture
    benchmark.prepare_fixture = wave.prepare_fixture
    try:
        benchmark.prepare(args)
    finally:
        benchmark.prepare_fixture = original
    root = args.output.resolve()
    try:
        profiles = cap_profiles(benchmark.read(root/'P1.json'), args.prefill_cap_tokens)
        for arm, profile in profiles.items():
            benchmark.q.save(root/(arm+'.json'), profile)
        manifest = benchmark.read(root/'manifest.json')
        manifest.update(comparison_kind=KIND, fixture_format='fixed_prefill_wave_v1',
            arm_scope=ARM_SCOPE,
            protocol='Fixed-prefix warmup -> resident first-token T0 -> fixed-offset seed wave -> T1 observation '
                     '-> full fixed-quota drain -> exact retirement and namespace cleanup',
            scope='Finite interference probe; fixed inputs and quotas, not DeepSWE task quality',
            profile_sha256={arm:hashlib.sha256((root/(arm+'.json')).read_bytes()).hexdigest() for arm in profiles},
            input_identity='Formal resident prefix is the exact warmup input plus one fixed token; warmup output is discarded',
            performance_gate='No window-only win: preserve full seed demand and report dispatch-to-completion makespan',
            single_difference={'A0':{'prefill_batch_tokens':0}, 'P1':{'prefill_batch_tokens':args.prefill_cap_tokens}})
        benchmark.q.save(root/'manifest.json', manifest)
        validate_prepared(root)
        benchmark.q.save(root/'status.json', dict(status='prepared', at=time.time(), gpu_started=False))
    except BaseException as error:
        benchmark.q.save(root/'status.json', dict(status='failed', at=time.time(), gpu_started=False, error=repr(error)))
        raise


async def run(args):
    root = args.output.resolve()
    if ROOT.resolve() != (root/'frozen').resolve():
        raise ValueError('Execute the prepared frozen script only')
    manifest = validate_prepared(root)
    for key, value in manifest['workload'].items():
        if key not in ('prepare_only', 'run_prepared'):
            setattr(args, key, value)
    validate_arguments(args)
    original_module, original_compare = benchmark.module, benchmark.compare
    def load(name, path):
        return wave if name == 'timed_agent_workload' else original_module(name, path)
    benchmark.module, benchmark.compare = load, compare
    try:
        await benchmark.run(args)
    finally:
        benchmark.module, benchmark.compare = original_module, original_compare


def main(argv=None):
    parser = benchmark.parser()
    parser.description = __doc__
    parser.set_defaults(concurrency=8, warmup_turns=1, context_tokens=131072, output_tokens=32768,
                        duration_seconds=300., modes=['saturated'], port=18492)
    parser.add_argument('--resident-requests', type=int, default=5)
    parser.add_argument('--seed-requests', type=int, default=3)
    parser.add_argument('--seed-context-tokens', type=int, default=131072)
    parser.add_argument('--seed-output-tokens', type=int, default=256)
    parser.add_argument('--seed-offset-seconds', type=float, default=10.)
    parser.add_argument('--prefill-cap-tokens', type=int, default=2048)
    args = parser.parse_args(argv)
    if args.run_prepared:
        if args.prepare_only or args.config or args.fixture:
            parser.error('Frozen execution takes only the prepared output directory')
        return asyncio.run(run(args))
    if not args.prepare_only or not args.config or not args.fixture:
        parser.error('Prepare explicitly with a config and an existing token fixture before running the frozen script')
    prepare(args)


if __name__ == '__main__':
    main()
