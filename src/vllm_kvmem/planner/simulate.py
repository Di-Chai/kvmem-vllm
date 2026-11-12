"""Engine-step simulation of agents alternating generation and tool waits.

Models what decides throughput in the tool-wait experiments:

* S running slots; each running request holds its window pages plus the GDN
  state pages (geometry.running_pages);
* parked windows stay in the GPU pool up to the residency cap (pages), are
  evicted FIFO or by expected remaining wait, and optionally expire by TTL;
* swap-ins use bounded lanes (resident / assembly) with measured latencies; an
  in-flight swap-in slows concurrent engine steps (costs.StepCost);
* CPU sessions idle past the CPU TTL (or beyond the CPU budget) move to SSD and
  pay a promotion read on return (tiered archive).

Time advances one engine step at a time; idle periods jump to the next event.
Deterministic for a given seed.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import bisect
import math
import random

from .workload import MixtureWait


@dataclass
class _Agent:
    id: int
    history: int
    state: str = 'idle'          # idle, waiting, swap, prefill, decode
    location: str = 'cpu'        # where the window lives while idle: gpu, cpu, ssd
    next_arrival: float = 0.0
    arrived: float = 0.0
    idle_since: float = 0.0
    wait_class: str = 'default'
    lane: str = ''
    ready_at: float = 0.0
    prefill_left: int = 0
    pieces_left: int = 1
    decode_left: float = 0.0
    append: int = 0
    pages: int = 0
    on_ssd: bool = False         # CPU archive spilled (independent of a GPU-parked window)
    turn_end: float = 0.0


@dataclass
class SimResult:
    output_tok_s: float
    turns: int
    turn_latency_p50: float
    turn_latency_p90: float
    swaps: dict
    evictions: dict
    gpu_busy_fraction: float
    mean_parked: float
    mean_running: float
    ssd_sessions_peak: int = 0

    def to_dict(self):
        return dict(self.__dict__)


class _Remaining:
    """Expected remaining tool wait by class, tabulated on a log grid."""

    def __init__(self, dist):
        self.tables = {}
        classes = dist.classes if isinstance(dist, MixtureWait) else {'default': dist}
        self.grid = [0.0] + [0.01 * 1.25 ** k for k in range(48)]
        for name, d in classes.items():
            self.tables[name] = [d.expected_remaining(t, horizon=max(60.0, 8 * t + 60.0), steps=200)
                                 for t in self.grid]

    def __call__(self, cls, elapsed):
        table = self.tables.get(cls) or next(iter(self.tables.values()))
        i = min(len(self.grid) - 1, bisect.bisect_right(self.grid, elapsed))
        return table[i]


def _ttl_for(placement, cls):
    ttl = placement.gpu_ttl
    if isinstance(ttl, dict):
        return ttl.get(cls, ttl.get('default', float('inf')))
    return ttl


def simulate(geometry, workload, placement, costs, *, duration=300.0, warmup=60.0, seed=1,
             cpu_budget_bytes=None, query_tokens=32):
    rng = random.Random(seed)
    S = placement.slots
    pool = geometry.pool_pages
    gdn = geometry.recurrent_pages_per_request
    window = min(workload.window_tokens, geometry.select_budget or workload.window_tokens)
    cap = 0 if placement.gpu_policy == 'none' else int(pool * placement.residency_fraction)
    m = costs.tokens_per_decode_step
    mixture = isinstance(workload.tool_wait, MixtureWait)
    # expected_wait mirrors the serving estimator (residency_policy.ReturnGapModel):
    # learned from observed gaps, not from the true tool class. 'expected_wait_oracle'
    # uses the true class distribution (an upper bound for tool-class hints).
    remaining = _Remaining(workload.tool_wait) if placement.victim == 'expected_wait_oracle' else None
    learned = None
    if placement.victim == 'expected_wait':
        from ..residency_policy import ReturnGapModel
        learned = ReturnGapModel(clock=lambda: now)
    tiered = placement.archive == 'tiered'

    agents = [_Agent(i, workload.history_tokens) for i in range(workload.agents)]
    for a in agents:
        a.next_arrival = rng.uniform(0.0, 5.0)
        a.idle_since = 0.0
    parked = OrderedDict()                   # agent id -> (pages, parked_at)
    parked_pages = 0
    running = OrderedDict()                  # agent id -> agent
    running_pages = 0
    waiting = []
    lanes = {'resident': 0, 'assembly': 0}
    lane_cap = {'resident': costs.swap.resident_lanes, 'assembly': costs.swap.assembly_lanes}
    swaps = {'resident': 0, 'cpu': 0, 'ssd': 0}
    evictions = {'pressure': 0, 'cap': 0, 'ttl': 0}
    latencies, out_tokens, busy, turns = [], 0.0, 0.0, 0
    parked_area = running_area = 0.0
    ssd_peak = 0
    device_free_at = 0.0                     # the spill device serves one job at a time
    spill_done = {}                          # agent id -> time its spill finishes
    end = warmup + duration
    now = 0.0

    def window_pages(agent):
        return geometry.window_pages(geometry.window_tokens(agent.history, window))

    def evict_one(reason, exclude=None):
        nonlocal parked_pages
        candidates = [k for k in parked if k != exclude]
        if not candidates:
            return False
        if learned is not None:
            victim = max(candidates, key=lambda k: learned.expected_remaining(k, now - parked[k][1]))
        elif remaining is None:
            victim = candidates[0]                       # FIFO: oldest parked first
        else:
            victim = max(candidates, key=lambda k: remaining(agents[k].wait_class, now - parked[k][1]))
        pages, _ = parked.pop(victim)
        parked_pages -= pages
        agents[victim].location = 'cpu'
        agents[victim].idle_since = now
        evictions[reason] += 1
        return True

    def spill(a):
        # Spills only start on an idle device; promotions are never queued behind them.
        nonlocal device_free_at
        if device_free_at > now:
            return False
        device_free_at = now + a.history * geometry.cpu_bytes_per_token / (costs.swap.ssd_write_gbps * 1e9)
        spill_done[a.id] = device_free_at
        a.on_ssd = True
        if a.location == 'cpu':
            a.location = 'ssd'
        return True

    def demote_cpu():
        nonlocal ssd_peak
        if not tiered:
            return
        # Serving (session_tiering) spills any session idle past the CPU TTL, even
        # if its window is still parked on GPU; every swap-in then waits for it.
        idle_cpu = [a for a in agents if a.state in ('idle', 'waiting') and a.location == 'cpu']
        for a in agents:
            if (a.state == 'idle' and not a.on_ssd and now - a.turn_end > placement.cpu_ttl):
                spill(a)
        if cpu_budget_bytes is not None:
            cpu_bytes = sum(a.history for a in agents if a.location != 'ssd') * geometry.cpu_bytes_per_token
            for a in sorted(idle_cpu, key=lambda x: x.idle_since):
                if cpu_bytes <= cpu_budget_bytes:
                    break
                if a.location == 'cpu' and a.state == 'idle' and not a.on_ssd and spill(a):
                    cpu_bytes -= a.history * geometry.cpu_bytes_per_token
        ssd_peak = max(ssd_peak, sum(a.on_ssd for a in agents))

    def finish(a):
        nonlocal running_pages, parked_pages, turns
        running.pop(a.id)
        running_pages -= a.pages + gdn
        a.history += a.append + query_tokens + workload.output_tokens
        a.history = min(a.history, workload.max_history_tokens)
        if warmup <= now < end:
            latencies.append(now - a.arrived)
            turns += 1
        pages = window_pages(a)
        a.state, a.idle_since, a.turn_end = 'idle', now, now
        if cap and pages <= cap:
            while parked and parked_pages + pages > cap:
                evict_one('cap')
            parked[a.id] = (pages, now)
            parked_pages += pages
            a.location = 'gpu'
        else:
            a.location = 'cpu'
        if mixture and workload.sticky_tool_class and a.wait_class != 'default':
            wait = workload.tool_wait.classes[a.wait_class].sample(rng)
        elif mixture:
            a.wait_class, wait = workload.tool_wait.sample_class(rng)
        else:
            wait = workload.tool_wait.sample(rng)
        a.next_arrival = now + wait
        if learned is not None:
            learned.finished(a.id)

    while now < end:
        for a in agents:
            if a.state == 'idle' and a.next_arrival <= now:
                a.state, a.arrived = 'waiting', a.next_arrival
                waiting.append(a)
                if learned is not None:
                    learned.arrival(a.id)
        if placement.gpu_policy == 'ttl':
            for k in [k for k, (_, at) in parked.items() if now - at > _ttl_for(placement, agents[k].wait_class)]:
                pages, _ = parked.pop(k)
                parked_pages -= pages
                agents[k].location, agents[k].idle_since = 'cpu', now
                evictions['ttl'] += 1
        demote_cpu()

        # Admission: FIFO, skipping requests whose lane is busy.
        decoding = sum(1 for r in running.values() if r.state == 'decode')
        for a in list(waiting):
            if len(running) >= S:
                break
            lane = 'resident' if a.id in parked else 'assembly'
            if lanes[lane] >= lane_cap[lane]:
                continue
            pages = parked[a.id][0] if lane == 'resident' else window_pages(a)
            need = gdn + (0 if lane == 'resident' else pages)
            while pool - running_pages - parked_pages < need and evict_one('pressure', exclude=a.id):
                pass
            if pool - running_pages - parked_pages < need:
                break
            if lane == 'resident':
                parked.pop(a.id)
                parked_pages -= pages
                latency = costs.swap.resident_s
                swaps['resident'] += 1
                if a.on_ssd:            # the session's CPU archive must come back first
                    promote_start = max(now, device_free_at, spill_done.pop(a.id, 0.0))
                    promoted_at = promote_start + costs.swap.ssd_fixed_s + a.history * geometry.cpu_bytes_per_token / (
                        costs.swap.ssd_read_gbps * 1e9)
                    device_free_at = promoted_at
                    latency += promoted_at - now
                    swaps['ssd'] += 1
                    a.on_ssd = False
            else:
                latency = costs.swap.assembly_s(decoding / max(1, S))
                if a.location == 'ssd':
                    # Session tiering promotes the whole archived history (not just the
                    # window) after any running device job, including its own spill.
                    promote_start = max(now, device_free_at, spill_done.pop(a.id, 0.0))
                    promoted_at = promote_start + costs.swap.ssd_fixed_s + a.history * geometry.cpu_bytes_per_token / (
                        costs.swap.ssd_read_gbps * 1e9)
                    device_free_at = promoted_at
                    latency += promoted_at - now
                    swaps['ssd'] += 1
                    a.on_ssd = False
                else:
                    swaps['cpu'] += 1
            waiting.remove(a)
            running[a.id] = a
            running_pages += pages + gdn
            a.pages, a.lane, a.state, a.ready_at = pages, lane, 'swap', now + latency
            lanes[lane] += 1
            a.append = workload.sample_append(rng)
            a.prefill_left = a.append + query_tokens
            whole = int(costs.prefill_steps_per_turn)
            a.pieces_left = max(1, whole + (rng.random() < costs.prefill_steps_per_turn - whole))
            a.decode_left = workload.output_tokens

        dec = [r for r in running.values() if r.state == 'decode']
        pre = [r for r in running.values() if r.state == 'prefill']
        if not dec and not pre:
            events = [r.ready_at for r in running.values() if r.state == 'swap']
            events += [a.next_arrival for a in agents if a.state == 'idle']
            nxt = min(events) if events else end
            if nxt <= now:
                nxt = now + 1e-3
            dt = min(nxt, end) - now
            parked_area += dt * len(parked)
            running_area += dt * len(running)
            now += dt
        else:
            # One request's logical prefill piece per step (KVMem schedules a
            # turn's prefill as ~prefill_steps_per_turn separate pieces).
            prefill_tokens = 0
            if pre:
                r = pre[0]
                take = math.ceil(r.prefill_left / r.pieces_left)
                take = min(take, costs.max_batched_tokens - len(dec) * 5)
                r.prefill_left -= take
                r.pieces_left -= 1
                prefill_tokens = take
            dt = costs.step.ms(len(dec), prefill_tokens, lanes['resident'], lanes['assembly']) / 1000.0
            parked_area += dt * len(parked)
            running_area += dt * len(running)
            now += dt
            if warmup <= now <= end:
                busy += dt
            for r in dec:
                produced = min(m, r.decode_left)
                r.decode_left -= produced
                if warmup <= now <= end:
                    out_tokens += produced
            for r in list(running.values()):
                if r.state == 'decode' and r.decode_left <= 0:
                    finish(r)
                elif r.state == 'prefill' and (r.prefill_left <= 0 or r.pieces_left <= 0):
                    r.state = 'decode'
        for r in list(running.values()):
            if r.state == 'swap' and r.ready_at <= now:
                r.state = 'prefill'
                lanes[r.lane] -= 1

    latencies.sort()
    q = lambda p: latencies[int(p * (len(latencies) - 1))] if latencies else float('nan')
    return SimResult(output_tok_s=out_tokens / duration, turns=turns, turn_latency_p50=q(0.5),
                     turn_latency_p90=q(0.9), swaps=swaps, evictions=evictions,
                     gpu_busy_fraction=busy / duration, mean_parked=parked_area / max(now, 1e-9),
                     mean_running=running_area / max(now, 1e-9), ssd_sessions_peak=ssd_peak)
