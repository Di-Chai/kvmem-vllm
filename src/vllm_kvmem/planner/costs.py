"""Step and swap costs, with defaults fitted from captured runs.

Engine step time (ms), fitted per run from model_step_latency.jsonl:

    step = a + per_decoder*n + per_1k_prefill*p/1000 + prefill_step*[p>0]
           + per_resident_inflight*R + per_assembly_inflight*A

R and A are swap-ins in flight on the resident and assembly lanes: their host
and copy work slows concurrent steps, which is how a swap-in costs throughput
beyond its own slot. Swap-in latency (started -> ready) and lane widths come
from async_restore.jsonl / restore_admission. Defaults below: 11-12 tool-wait
sweep, 32 agents, Qwen3.8-27B NVFP4 + MTP4 on one RTX PRO 6000.
"""
from dataclasses import dataclass, field, asdict, replace
import json
from pathlib import Path


@dataclass(frozen=True)
class StepCost:
    a: float = 15.2
    per_decoder: float = 3.63
    per_1k_prefill: float = 96.2
    prefill_step: float = 116.3
    per_resident_inflight: float = 10.2
    per_assembly_inflight: float = 26.9
    host_gap: float = 2.0                  # median gap between busy steps

    def ms(self, decoders, prefill_tokens, resident_inflight=0, assembly_inflight=0):
        return (self.a + self.per_decoder * decoders + self.per_1k_prefill * prefill_tokens / 1000 +
                (self.prefill_step if prefill_tokens else 0.0) +
                self.per_resident_inflight * resident_inflight +
                self.per_assembly_inflight * assembly_inflight + self.host_gap)


@dataclass(frozen=True)
class SwapCost:
    """Swap-in latency (s) by source tier at full load (all slots busy) and idle."""
    resident_s: float = 0.25               # window still on GPU: GDN upload + bookkeeping
    assembly_idle_s: float = 0.20          # CPU -> GPU, uncontended
    assembly_loaded_s: float = 0.53        # CPU -> GPU, all slots decoding
    resident_lanes: int = 2
    assembly_lanes: int = 1
    ssd_read_gbps: float = 3.0             # SSD -> CPU session promotion (whole history; 8 threads, measured 11-12)
    ssd_write_gbps: float = 1.5            # CPU -> SSD session spill (shares the device with promotions)
    ssd_fixed_s: float = 0.02

    def assembly_s(self, load):
        load = min(1.0, max(0.0, load))
        return self.assembly_idle_s + (self.assembly_loaded_s - self.assembly_idle_s) * load


# Measured swap-in implementations (32 agents, residency 0.5; see PROGRESS.md 11-12).
# 'pinned' is a modelled estimate of a page-locked archive (not built).
SWAP_STACKS = {
    'baseline': (SwapCost(resident_s=0.20, assembly_idle_s=3.70, assembly_loaded_s=3.26),
                 dict(per_assembly_inflight=628.6, per_resident_inflight=0.0)),
    'isolate': (SwapCost(resident_s=0.20, assembly_idle_s=0.53, assembly_loaded_s=1.02),
                dict(per_assembly_inflight=113.5, per_resident_inflight=0.0)),
    'direct': (SwapCost(resident_s=0.20, assembly_idle_s=0.53, assembly_loaded_s=1.71),
               dict(per_assembly_inflight=40.0)),
    'direct_prepared': (SwapCost(resident_s=0.20, assembly_idle_s=0.20, assembly_loaded_s=0.74),
                        dict(per_assembly_inflight=29.1)),
    'direct_prepared_exempt': (SwapCost(resident_s=0.20, assembly_idle_s=0.20, assembly_loaded_s=0.53),
                               dict(per_assembly_inflight=26.9)),
    'pinned': (SwapCost(resident_s=0.20, assembly_idle_s=0.07, assembly_loaded_s=0.25),
               dict(per_assembly_inflight=12.0)),
}

# Swap flags that select a stack (the planner emits these as environment).
_FAST = ('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', 'KVMEM_RESTORE_DIRECT_LAYERS', 'KVMEM_RESTORE_PREPARED_READS',
         'KVMEM_RESTORE_GATE_EXEMPT')


def _stack(*on):
    # The fast stack is default-on (defaults.py, 2026-11-12): slower stacks must
    # switch the remaining flags off explicitly.
    return {flag: '1' if flag in on else '0' for flag in _FAST}


STACK_ENV = {
    'baseline': _stack(),
    'isolate': _stack('KVMEM_ASSEMBLY_ISOLATE_PARTIAL'),
    'direct': _stack('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', 'KVMEM_RESTORE_DIRECT_LAYERS'),
    'direct_prepared': _stack('KVMEM_ASSEMBLY_ISOLATE_PARTIAL', 'KVMEM_RESTORE_DIRECT_LAYERS',
                              'KVMEM_RESTORE_PREPARED_READS'),
    'direct_prepared_exempt': {},       # the default-on stack
    'pinned': None,     # not implemented
}


@dataclass(frozen=True)
class Costs:
    step: StepCost = StepCost()
    swap: SwapCost = SwapCost()
    tokens_per_decode_step: float = 4.9    # MTP4: output tokens per request per decode step
    max_batched_tokens: int = 16424
    prefill_steps_per_turn: float = 1.5    # prefill-bearing engine steps per turn (logical pieces)

    def for_stack(self, stack):
        swap, step_overrides = SWAP_STACKS[stack]
        return replace(self, swap=replace(swap, resident_s=self.swap.resident_s,
                                          resident_lanes=self.swap.resident_lanes,
                                          assembly_lanes=self.swap.assembly_lanes,
                                          ssd_read_gbps=self.swap.ssd_read_gbps,
                                          ssd_write_gbps=self.swap.ssd_write_gbps),
                       step=replace(self.step, **step_overrides))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(step=StepCost(**d['step']), swap=SwapCost(**d['swap']),
                   **{k: v for k, v in d.items() if k not in ('step', 'swap')})


def _lstsq(rows, cols):
    import numpy as np
    X = np.array([[r[c] for c in cols] for r in rows], float)
    y = np.array([r['y'] for r in rows], float)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    err = float(np.median(np.abs(X @ coef - y)))
    return dict(zip(cols, map(float, coef))), err


def calibrate(capture_dir, *, skip_fraction=0.2):
    """Fit StepCost and swap latencies from one server capture directory.

    Steps of the first ``skip_fraction`` (the history grow phase) are skipped.
    Returns (Costs, report) where report holds the fit error and sample counts.
    """
    capture = Path(capture_dir)
    steps = [json.loads(l) for l in (capture / 'model_step_latency.jsonl').open()]
    steps = steps[int(len(steps) * skip_fraction):]
    restores = []
    path = capture / 'async_restore.jsonl'
    if path.exists():
        restores = [json.loads(l) for l in path.open()]
    spans = {'resident': [], 'assembly': []}
    for r in restores:
        spans.setdefault(r['lane'], []).append((r['started'], r['finished']))
    for lane in spans:
        spans[lane].sort()

    def inflight(lane, t):
        return sum(1 for s, f in spans.get(lane, ()) if s <= t < f)

    rows = []
    for s in steps:
        p = s.get('prefill_tokens', 0)
        rows.append(dict(y=s['step_ms'], one=1.0, n=float(s.get('decode_requests', 0)), p=p / 1000,
                         pf=1.0 if p else 0.0, rr=float(inflight('resident', s['started_at'])),
                         ra=float(inflight('assembly', s['started_at']))))
    coef, err = _lstsq(rows, ['one', 'n', 'p', 'pf', 'rr', 'ra'])
    gaps = sorted(max(0.0, s.get('previous_end_to_model_ms') or 0.0) for s in steps)
    step = StepCost(a=coef['one'], per_decoder=coef['n'], per_1k_prefill=coef['p'], prefill_step=coef['pf'],
                    per_resident_inflight=coef['rr'], per_assembly_inflight=coef['ra'],
                    host_gap=gaps[len(gaps) // 2] if gaps else 2.0)

    t0, t1 = steps[0]['started_at'], steps[-1]['started_at']
    turns = sum(1 for r in restores if t0 <= r['started'] <= t1)
    prefill_steps = sum(1 for s in steps if s.get('prefill_tokens', 0))
    pieces = prefill_steps / turns if turns else Costs.prefill_steps_per_turn
    decode_request_steps = sum(s.get('decode_requests', 0) for s in steps)

    def p50(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else float('nan')
    res = [r['finished'] - r['started'] for r in restores if r['lane'] == 'resident']
    asm = [r['finished'] - r['started'] for r in restores if r['lane'] == 'assembly']
    swap = SwapCost(resident_s=p50(res) if res else SwapCost.resident_s,
                    assembly_idle_s=SwapCost.assembly_idle_s,
                    assembly_loaded_s=p50(asm) if asm else SwapCost.assembly_loaded_s)
    report = dict(steps=len(steps), step_fit_median_abs_error_ms=err, turns=turns,
                  prefill_steps=prefill_steps, decode_request_steps=decode_request_steps,
                  resident_restores=len(res), assembly_restores=len(asm))
    return Costs(step=step, swap=swap, prefill_steps_per_turn=pieces), report

