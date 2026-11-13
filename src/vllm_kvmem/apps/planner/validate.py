"""Re-predict measured runs and report the error (keeps the cost model honest).

MEASURED holds the 11-11/11-12 tool-wait sweep (toolwait-sim workload, 24K
history, 8 slots): arm, agents, residency, swap stack, duration (s), output
tok/s. Older baseline-stack arms ran with earlier defaults, so their errors are
reported separately ('confounded').
"""
from .costs import Costs
from .memory import Placement
from .simulate import simulate
from .workload import preset

MEASURED = [
    # arm, agents, residency, stack, duration, tok/s, current-stack?
    ('dX32b', 32, 0.5, 'direct_prepared_exempt', 300, 488.5, True),
    ('dX32', 32, 0.5, 'direct_prepared_exempt', 300, 486.4, True),
    ('dP32', 32, 0.5, 'direct_prepared', 300, 478.9, True),
    ('dV32', 32, 0.5, 'direct_prepared', 300, 484.3, True),
    ('dL32b', 32, 0.5, 'direct', 300, 458.7, True),
    ('iP32b', 32, 0.5, 'isolate', 300, 462.9, True),
    ('fS32', 32, 0.01, 'direct_prepared', 300, 423.5, True),
    ('tB32', 32, 0.5, 'baseline', 300, 375.5, False),
    ('gR32', 32, 0.75, 'baseline', 300, 390.4, False),
    ('gF32', 32, 1.0, 'baseline', 300, 394.1, False),
    ('gE24b', 24, 0.5, 'baseline', 600, 389.3, False),
    ('gR24', 24, 0.75, 'baseline', 600, 446.9, False),
    ('gE16', 16, 0.5, 'baseline', 600, 450.1, False),
    ('gR16', 16, 0.75, 'baseline', 600, 449.1, False),
    ('gD8', 8, 0.5, 'baseline', 600, 338.7, False),
]


def validate(geometry, costs=None, *, seeds=(1, 2)):
    costs = costs or Costs()
    rows = []
    for arm, agents, residency, stack, duration, measured, current in MEASURED:
        w = preset('toolwait-sim', agents)
        p = Placement(slots=8, residency_fraction=residency, swap_stack=stack)
        runs = [simulate(geometry, w, p, costs.for_stack(stack), duration=duration, seed=s) for s in seeds]
        pred = sum(r.output_tok_s for r in runs) / len(runs)
        rows.append(dict(arm=arm, agents=agents, residency=residency, stack=stack, measured=measured,
                         predicted=round(pred, 1), error=round((pred - measured) / measured, 4),
                         current_stack=current))
    def summary(sel):
        errs = sorted(abs(r['error']) for r in sel)
        return dict(arms=len(sel), median_abs_error=errs[len(errs) // 2] if errs else None,
                    max_abs_error=errs[-1] if errs else None)
    return dict(rows=rows, current=summary([r for r in rows if r['current_stack']]),
                confounded=summary([r for r in rows if not r['current_stack']]))
