"""Workload description: agents, window, history, turn shape and tool waits.

Tool waits drive every tiering decision, so they are first-class: a capped
lognormal, an empirical sample list, or a weighted mixture of named tool
classes (e.g. quick file reads vs test runs). Each distribution exposes
sampling, the survival function and the hazard rate h(t) = f(t) / S(t), which
the TTL rule uses: keep a parked session in a tier while the chance that it
returns soon still pays for the memory it holds.
"""
from dataclasses import dataclass, field, asdict
import bisect
import math
import random


class ToolWait:
    def sample(self, rng):
        raise NotImplementedError

    def survival(self, t):
        raise NotImplementedError

    def density(self, t):
        raise NotImplementedError

    def hazard(self, t):
        s = self.survival(t)
        return self.density(t) / s if s > 1e-12 else 0.0

    def mean(self):
        raise NotImplementedError

    def expected_remaining(self, elapsed, horizon=600.0, steps=600):
        """E[T - elapsed | T > elapsed], integrated up to ``horizon`` seconds."""
        s0 = self.survival(elapsed)
        if s0 <= 1e-12:
            return 0.0
        dt = max(1e-3, (horizon - elapsed) / steps)
        total, t = 0.0, elapsed
        while t < horizon:
            total += self.survival(t + dt / 2) * dt
            t += dt
        return total / s0


@dataclass
class LogNormalWait(ToolWait):
    median: float
    sigma: float
    cap: float = float('inf')

    def sample(self, rng):
        return min(self.cap, self.median * math.exp(self.sigma * rng.gauss(0.0, 1.0)))

    def _z(self, t):
        return (math.log(t) - math.log(self.median)) / self.sigma

    def survival(self, t):
        if t <= 0:
            return 1.0
        if t >= self.cap:
            return 0.0
        return 0.5 * math.erfc(self._z(t) / math.sqrt(2))

    def density(self, t):
        if t <= 0 or t >= self.cap:
            return 0.0
        z = self._z(t)
        return math.exp(-0.5 * z * z) / (t * self.sigma * math.sqrt(2 * math.pi))

    def mean(self):
        # Mean of the capped distribution, numerically (cap rarely binds for planning).
        return self.expected_remaining(0.0, horizon=min(self.cap, self.median * math.exp(6 * self.sigma)))


@dataclass
class EmpiricalWait(ToolWait):
    samples: list

    def __post_init__(self):
        if not self.samples:
            raise ValueError('Empirical tool waits need at least one sample')
        self.samples = sorted(float(s) for s in self.samples)
        self._bandwidth = max(1e-3, 0.1 * (self.samples[len(self.samples) // 2] or 1e-2))

    def sample(self, rng):
        return self.samples[rng.randrange(len(self.samples))]

    def survival(self, t):
        return 1.0 - bisect.bisect_right(self.samples, t) / len(self.samples)

    def density(self, t):
        h = self._bandwidth
        return (bisect.bisect_right(self.samples, t + h) - bisect.bisect_left(self.samples, t - h)) / (
            2 * h * len(self.samples))

    def mean(self):
        return sum(self.samples) / len(self.samples)


@dataclass
class MixtureWait(ToolWait):
    """Named tool classes with weights; a session's next wait picks one class."""
    classes: dict              # name -> ToolWait
    weights: dict              # name -> weight

    def _norm(self):
        total = sum(self.weights.values())
        return {k: w / total for k, w in self.weights.items()}

    def sample(self, rng):
        return self.sample_class(rng)[1]

    def sample_class(self, rng):
        r, acc = rng.random(), 0.0
        for name, w in self._norm().items():
            acc += w
            if r <= acc:
                return name, self.classes[name].sample(rng)
        name = next(reversed(self.weights))
        return name, self.classes[name].sample(rng)

    def survival(self, t):
        return sum(w * self.classes[k].survival(t) for k, w in self._norm().items())

    def density(self, t):
        return sum(w * self.classes[k].density(t) for k, w in self._norm().items())

    def mean(self):
        return sum(w * self.classes[k].mean() for k, w in self._norm().items())


@dataclass
class Workload:
    name: str
    agents: int
    window_tokens: int              # active window restored on a swap-in (select budget)
    history_tokens: int             # history per session at start
    max_history_tokens: int         # history kept on CPU per session (GC limit)
    append_median: float            # new input tokens per turn (lognormal)
    append_sigma: float
    output_tokens: int              # output tokens per turn
    tool_wait: ToolWait
    turn_latency_p90_limit: float = float('inf')
    append_cap: int = 6000
    sticky_tool_class: bool = False     # each session keeps one tool class (mixtures)

    def describe(self):
        d = {k: v for k, v in asdict(self).items() if k != 'tool_wait'}
        d['tool_wait'] = type(self.tool_wait).__name__
        d['tool_wait_mean_s'] = round(self.tool_wait.mean(), 3)
        return d

    def sample_append(self, rng):
        return int(min(self.append_cap, self.append_median * math.exp(self.append_sigma * rng.gauss(0, 1))))


def preset(name, agents, **overrides):
    """Named workloads. ``toolwait-sim`` mirrors agent_sim.py's defaults (the
    11-11/11-12 sweeps); the others are starting points to replace with traces."""
    base = dict(window_tokens=36864, history_tokens=24000, max_history_tokens=1 << 20,
                append_median=300, append_sigma=1.0, output_tokens=320)
    waits = {
        'toolwait-sim': LogNormalWait(0.9, 1.81, cap=120.0),
        # DeepSWE-like: mostly quick file/grep tools, a tail of test runs.
        'deepswe': MixtureWait({'quick': LogNormalWait(0.15, 0.6, cap=10.0),
                                'tests': LogNormalWait(4.0, 1.0, cap=300.0)},
                               {'quick': 0.9, 'tests': 0.1}),
        # Terminal-Bench-like: builds, installs and long commands dominate.
        'terminal-bench': MixtureWait({'quick': LogNormalWait(0.3, 0.8, cap=10.0),
                                       'long': LogNormalWait(20.0, 1.2, cap=1800.0)},
                                      {'quick': 0.4, 'long': 0.6}),
        # Heterogeneous agents: most edit/read quickly, some keep running builds/tests.
        'mixed-agents': MixtureWait({'quick': LogNormalWait(0.2, 0.8, cap=10.0),
                                     'long': LogNormalWait(8.0, 1.0, cap=300.0)},
                                    {'quick': 0.7, 'long': 0.3}),
    }
    if name == 'mixed-agents':
        base.setdefault('sticky_tool_class', True)
    if name not in waits:
        raise ValueError(f'Unknown workload preset {name!r}; choose one of {sorted(waits)}')
    base.update(overrides)
    return Workload(name=name, agents=agents, tool_wait=waits[name], **base)
