"""CLI: python -m vllm_kvmem.planner {probe,calibrate,simulate,compile}."""
import argparse
import json
import math
import sys
from pathlib import Path

from . import emit
from .costs import Costs, calibrate, SWAP_STACKS
from .geometry import from_cache_geometry, from_model_config
from .hardware import probe, bench
from .memory import Machine, Placement, plan_memory
from .search import candidates, evaluate, choose, refine, BUILT_STACKS
from .simulate import simulate
from .workload import preset


def _geometry(args):
    if args.geometry:
        return from_cache_geometry(args.geometry)
    if not args.model_config or not args.kv_pool_bytes:
        sys.exit('Give --geometry cache-geometry.json, or --model-config with --kv-pool-bytes')
    return from_model_config(args.model_config, kv_pool_bytes=args.kv_pool_bytes,
                             spec_tokens=args.spec_tokens, select_budget=args.window,
                             gen_budget=args.gen_budget)


def _workload(args):
    over = {}
    if args.window:
        over['window_tokens'] = args.window
    if args.history:
        over['history_tokens'] = args.history
    if args.max_history:
        over['max_history_tokens'] = args.max_history
    if args.p90_limit:
        over['turn_latency_p90_limit'] = args.p90_limit
    return preset(args.workload, args.agents, **over)


def _costs(args):
    costs = Costs.from_dict(json.loads(Path(args.costs).read_text())) if args.costs else Costs()
    if getattr(args, 'ssd_bench', False):
        if not args.ssd_dir:
            raise SystemExit('--ssd-bench needs --ssd-dir')
        from dataclasses import replace
        from .hardware import bench_spill_device
        bench = bench_spill_device(args.ssd_dir)
        args.spill_bench = bench                 # reported with the machine
        print(f"spill device {args.ssd_dir}: write {bench['spill_write_gbps']:.2f} GB/s, random read "
              f"{bench['spill_read_gbps']:.2f} GB/s ({bench['spill_threads']} threads)", file=sys.stderr)
        costs = replace(costs, swap=replace(costs.swap, ssd_read_gbps=bench['spill_read_gbps'],
                                            ssd_write_gbps=bench['spill_write_gbps']))
    return costs


def _machine(args):
    machine, info = probe(ssd_dir=args.ssd_dir)
    info.update(getattr(args, 'spill_bench', None) or {})
    if args.cpu_budget_gib:
        machine = Machine(**{**machine.__dict__, 'cpu_archive_budget_bytes': int(args.cpu_budget_gib * 2**30)})
    return machine, info


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m vllm_kvmem.planner')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('probe', help='machine limits (+ --bench copy bandwidths)')
    p.add_argument('--bench', action='store_true'); p.add_argument('--ssd-dir'); p.add_argument('--out')

    v = sub.add_parser('validate', help='re-predict the measured 11-12 runs')
    v.add_argument('--geometry', required=True); v.add_argument('--costs')

    c = sub.add_parser('calibrate', help='fit costs from a server capture directory')
    c.add_argument('capture'); c.add_argument('--out')

    for name in ('simulate', 'compile'):
        s = sub.add_parser(name)
        s.add_argument('--workload', default='toolwait-sim'); s.add_argument('--agents', type=int, required=True)
        s.add_argument('--geometry'); s.add_argument('--model-config'); s.add_argument('--kv-pool-bytes', type=int)
        s.add_argument('--spec-tokens', type=int, default=4); s.add_argument('--gen-budget', type=int, default=16384)
        s.add_argument('--window', type=int); s.add_argument('--history', type=int); s.add_argument('--max-history', type=int)
        s.add_argument('--p90-limit', type=float); s.add_argument('--costs'); s.add_argument('--ssd-dir')
        s.add_argument('--cpu-budget-gib', type=float); s.add_argument('--duration', type=float, default=300.0)
        s.add_argument('--ssd-bench', action='store_true', help='measure --ssd-dir and use its bandwidth')
        if name == 'simulate':
            s.add_argument('--slots', type=int, default=8); s.add_argument('--residency', type=float, default=0.5)
            s.add_argument('--gpu-ttl', type=float, default=math.inf)
            s.add_argument('--victim', default='fifo'); s.add_argument('--stack', default='direct_prepared_exempt')
            s.add_argument('--tiered', action='store_true'); s.add_argument('--cpu-ttl', type=float, default=math.inf)
            s.add_argument('--seed', type=int, default=1)
        else:
            s.add_argument('--slots', default='8'); s.add_argument('--stacks', default='direct_prepared_exempt')
            s.add_argument('--tiered', action='store_true'); s.add_argument('--include-unbuilt', action='store_true')
            s.add_argument('--base-profile'); s.add_argument('--out', default='kvmem-plan')
            s.add_argument('--refine-seeds', type=int, default=8)
    args = ap.parse_args(argv)

    if args.cmd == 'probe':
        machine, info = probe(ssd_dir=args.ssd_dir)
        result = dict(machine=info, bench=bench() if args.bench else {})
        if args.bench and args.ssd_dir:
            from .hardware import bench_spill_device
            result['bench'].update(bench_spill_device(args.ssd_dir))
        text = json.dumps(result, indent=2, default=str)
        Path(args.out).write_text(text) if args.out else print(text)
        return 0
    if args.cmd == 'validate':
        from .validate import validate
        costs = Costs.from_dict(json.loads(Path(args.costs).read_text())) if args.costs else Costs()
        result = validate(from_cache_geometry(args.geometry), costs)
        for r in result['rows']:
            print(f"{r['arm']:7s} N={r['agents']:2d} R={r['residency']:<4} {r['stack']:24s} measured "
                  f"{r['measured']:6.1f} predicted {r['predicted']:6.1f} ({100*r['error']:+5.1f}%)"
                  f"{'' if r['current_stack'] else '  [older defaults]'}")
        print('current stack:', result['current'], '| older-default runs:', result['confounded'])
        return 0
    if args.cmd == 'calibrate':
        costs, rep = calibrate(args.capture)
        text = json.dumps(dict(costs=costs.to_dict(), report=rep), indent=2)
        if args.out:
            Path(args.out).write_text(json.dumps(costs.to_dict(), indent=2))
        print(text)
        return 0

    geometry, workload, costs = _geometry(args), _workload(args), _costs(args)
    machine, info = _machine(args)
    if args.cmd == 'simulate':
        placement = Placement(slots=args.slots, gpu_policy='ttl' if math.isfinite(args.gpu_ttl) else 'cap',
                              residency_fraction=args.residency, gpu_ttl=args.gpu_ttl, victim=args.victim,
                              archive='tiered' if args.tiered else 'cpu', cpu_ttl=args.cpu_ttl,
                              swap_stack=args.stack)
        mem = plan_memory(geometry, workload, placement, machine)
        res = simulate(geometry, workload, placement, costs.for_stack(args.stack), duration=args.duration,
                       seed=args.seed, cpu_budget_bytes=mem.cpu_budget_bytes if args.tiered else None)
        print(json.dumps(dict(placement=placement.label(), memory=mem.to_dict(), result=res.to_dict()),
                         indent=2, default=str))
        return 0

    stacks = list(BUILT_STACKS if args.stacks == 'all' else args.stacks.split(','))
    if args.include_unbuilt and 'pinned' not in stacks:
        stacks.append('pinned')
    slots = tuple(int(x) for x in str(args.slots).split(','))
    cands = list(candidates(geometry, workload, slots=slots, stacks=tuple(stacks), tiered=args.tiered))
    evaluated = evaluate(geometry, workload, machine, costs, cands, duration=args.duration,
                         include_unbuilt=args.include_unbuilt)
    if args.refine_seeds > 1:
        evaluated = refine(geometry, workload, machine, costs, evaluated, duration=args.duration,
                           seeds=tuple(range(1, args.refine_seeds + 1)))
    ranked = sorted(evaluated, key=lambda e: e.score, reverse=True)
    chosen = choose(evaluated, workload)
    best_any = choose(evaluated, workload, deployable_only=False)
    caveats = []
    if any(s != 8 for s in slots):
        caveats.append('Costs were fitted at 8 running slots; other slot counts are extrapolated.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.md').write_text(emit.report(workload, geometry, info, chosen, ranked, extrapolated=caveats,
                                                     best_any=best_any))
    plan = dict(workload=workload.describe(), geometry=geometry.to_dict(), machine=info,
                chosen=None if chosen is None else dict(placement=chosen.placement.label(),
                    result=chosen.result.to_dict(), memory=chosen.memory.to_dict(),
                    overlay=emit.overlay(chosen.placement), env=emit.environment(chosen.placement)),
                candidates=[dict(placement=e.placement.label(), legal=e.memory.legal, deployable=e.deployable,
                                 needs=list(e.needs), violations=e.memory.violations,
                                 result=None if e.result is None else e.result.to_dict()) for e in ranked])
    (out / 'plan.json').write_text(json.dumps(plan, indent=2, default=str))
    if chosen is not None:
        patch = emit.overlay(chosen.placement)
        if 'cpu_session_ttl_ms' in patch['kvmem']:
            patch['kvmem'].update(cpu_spill_dir=args.ssd_dir or '<ssd directory>',
                                  cpu_spill_max_bytes=int(chosen.memory.ssd_bytes_needed * 1.5) or 1 << 30)
        (out / 'profile-overlay.json').write_text(json.dumps(patch, indent=2))
        if args.base_profile:
            base = json.loads(Path(args.base_profile).read_text())
            (out / 'profile.json').write_text(json.dumps(emit.merge(base, patch), indent=2))
        (out / 'env.json').write_text(json.dumps(emit.environment(chosen.placement), indent=2))
    print((out / 'report.md').read_text())
    return 0


if __name__ == '__main__':
    sys.exit(main())
