"""Enumerate placements, drop illegal ones, simulate the rest, pick the best.

Each candidate carries whether serving supports it today: the swap stacks map
to existing opt-in flags, the GPU residency cap and TTL map to existing profile
keys, while expected-wait victims (C2) and SSD demotion by CPU TTL (C3) are
marked as requiring the upcoming runtime policy. The planner never emits an
unbuilt option as deployable.
"""
from dataclasses import dataclass, replace
import itertools
import math

from .costs import STACK_ENV
from .memory import Placement, plan_memory
from .simulate import simulate

BUILT_STACKS = [s for s, env in STACK_ENV.items() if env is not None]


@dataclass
class Evaluated:
    placement: Placement
    memory: object
    result: object = None
    deployable: bool = True
    needs: tuple = ()
    runs: tuple = ()                 # per-seed output tok/s

    @property
    def score(self):
        return self.result.output_tok_s if self.result is not None else -math.inf

    @property
    def stderr(self):
        n = len(self.runs)
        if n < 2:
            return math.inf
        mean = sum(self.runs) / n
        return math.sqrt(sum((x - mean) ** 2 for x in self.runs) / (n - 1) / n)

    @property
    def complexity(self):
        """Fewer moving parts first: plain cap, then TTL, then learned victims, then SSD."""
        p = self.placement
        return ((p.gpu_policy != 'cap') + (p.victim != 'fifo') + (p.archive != 'cpu') +
                abs(p.residency_fraction - 0.5))


def requirements(placement):
    needs = []
    if placement.victim == 'expected_wait_oracle':
        needs.append('tool-class hints (not built)')
    if placement.gpu_policy == 'ttl' and isinstance(placement.gpu_ttl, dict):
        needs.append('C2: per-tool-class GPU TTL')
    if placement.pinned_archive or STACK_ENV.get(placement.swap_stack) is None:
        needs.append('pinned archive (not built)')
    return tuple(needs)


def candidates(geometry, workload, *, slots=(8,), stacks=('direct_prepared_exempt',),
               residencies=(0.01, 0.25, 0.5, 0.75, 0.9), ttls=(1.0, 2.0, 5.0, 15.0, 60.0),
               victims=('fifo', 'expected_wait'), tiered=False, cpu_ttls=(60.0, 300.0, 1800.0)):
    for s, stack, victim in itertools.product(slots, stacks, victims):
        for r in residencies:
            yield Placement(slots=s, gpu_policy='cap', residency_fraction=r, victim=victim, swap_stack=stack)
        for ttl in ttls:
            yield Placement(slots=s, gpu_policy='ttl', residency_fraction=max(residencies), gpu_ttl=ttl,
                            victim=victim, swap_stack=stack)
        if tiered:
            for r, cpu_ttl in itertools.product(residencies, cpu_ttls):
                yield Placement(slots=s, gpu_policy='cap', residency_fraction=r, victim=victim,
                                archive='tiered', cpu_ttl=cpu_ttl, swap_stack=stack)


def evaluate(geometry, workload, machine, costs, placements, *, seeds=(1, 2), duration=300.0,
             warmup=60.0, include_unbuilt=False):
    out = []
    for p in placements:
        needs = requirements(p)
        if needs and not include_unbuilt and any('not built' in n for n in needs):
            continue
        mem = plan_memory(geometry, workload, p, machine)
        e = Evaluated(p, mem, deployable=not needs, needs=needs)
        if mem.legal:
            stack_costs = costs.for_stack(p.swap_stack)
            budget = mem.cpu_budget_bytes if p.archive == 'tiered' else None
            runs = [simulate(geometry, workload, p, stack_costs, duration=duration, warmup=warmup, seed=s,
                             cpu_budget_bytes=budget) for s in seeds]
            e.runs = tuple(r.output_tok_s for r in runs)
            e.result = replace(runs[0], output_tok_s=sum(r.output_tok_s for r in runs) / len(runs),
                               turn_latency_p50=sum(r.turn_latency_p50 for r in runs) / len(runs),
                               turn_latency_p90=sum(r.turn_latency_p90 for r in runs) / len(runs))
        out.append(e)
    return out


MODEL_ERROR = 0.03     # validated median ~2%, max ~8% on current-stack runs (validate.py)


def choose(evaluated, workload, *, deployable_only=True, noise_sigmas=2.0, model_error=MODEL_ERROR):
    """Best mean throughput; but if a simpler candidate is within the
    simulation noise (noise_sigmas combined standard errors) or the model's own
    validated error (model_error, relative), prefer the simpler one."""
    legal = [e for e in evaluated if e.memory.legal and e.result is not None and
             e.result.turn_latency_p90 <= workload.turn_latency_p90_limit and
             (e.deployable or not deployable_only)]
    if not legal:
        return None
    best = max(legal, key=lambda e: e.score)
    close = [e for e in legal if best.score - e.score <=
             max(noise_sigmas * math.sqrt(best.stderr ** 2 + e.stderr ** 2), model_error * best.score)]
    return min(close, key=lambda e: (e.complexity, -e.score))


def refine(geometry, workload, machine, costs, evaluated, *, top=6, seeds=(1, 2, 3, 4, 5, 6, 7, 8),
           duration=300.0, warmup=60.0):
    """Re-simulate the best legal candidates with more seeds (noise-aware choice)."""
    ranked = sorted((e for e in evaluated if e.result is not None), key=lambda e: e.score, reverse=True)
    keep = ranked[:top]
    baseline = [e for e in ranked if e.complexity == 0 and e not in keep]
    again = evaluate(geometry, workload, machine, costs, [e.placement for e in keep + baseline[:1]],
                     seeds=seeds, duration=duration, warmup=warmup, include_unbuilt=True)
    by_label = {e.placement.label(): e for e in again}
    return [by_label.get(e.placement.label(), e) for e in evaluated]
