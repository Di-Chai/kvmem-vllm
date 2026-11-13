"""Expected remaining tool wait of an idle session, learned from return gaps.

The server sees each session's completion -> next arrival gap (tool execution
plus client work). Gaps feed a bounded global reservoir and a per-session mean.
For an idle session that has waited ``elapsed`` seconds,

    E[remaining] = mean(g - elapsed for g in reservoir if g > elapsed) * session factor

where the factor (session mean / global mean, clipped to [1/4, 4]) separates
sessions that run long tools from quick ones. Agent tool waits are heavy-tailed
(decreasing hazard): beyond the observed samples the estimate falls back to
``elapsed``. Used only to order revocation of idle GPU views
(``active_resident_victim: expected_wait``); it never owns pages.
"""
from collections import OrderedDict, deque
import bisect
import time


class ReturnGapModel:
    def __init__(self, *, capacity=4096, sessions=4096, clock=time.monotonic):
        self.clock = clock
        self.samples = deque(maxlen=capacity)
        self._sorted = None
        self.session_mean = OrderedDict()
        self.finished_at = OrderedDict()
        self.max_sessions = sessions
        self.observations = 0

    def finished(self, namespace):
        if namespace is None:
            return
        self.finished_at[namespace] = self.clock()
        self.finished_at.move_to_end(namespace)
        while len(self.finished_at) > self.max_sessions:
            self.finished_at.popitem(last=False)

    def arrival(self, namespace):
        if namespace is None:
            return None
        finished = self.finished_at.pop(namespace, None)
        if finished is None:
            return None
        gap = max(0.0, self.clock() - finished)
        self.samples.append(gap)
        self._sorted = None
        mean = self.session_mean.get(namespace)
        self.session_mean[namespace] = gap if mean is None else 0.7 * mean + 0.3 * gap
        self.session_mean.move_to_end(namespace)
        while len(self.session_mean) > self.max_sessions:
            self.session_mean.popitem(last=False)
        self.observations += 1
        return gap

    def _sorted_samples(self):
        if self._sorted is None:
            self._sorted = sorted(self.samples)
            prefix, total = [0.0], 0.0
            for g in self._sorted:
                total += g
                prefix.append(total)
            self._prefix = prefix
        return self._sorted

    def global_mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else None

    def expected_remaining(self, namespace, elapsed):
        data = self._sorted_samples()
        if not data:
            base = elapsed
        else:
            i = bisect.bisect_right(data, elapsed)
            above = len(data) - i
            base = ((self._prefix[-1] - self._prefix[i]) / above - elapsed) if above else elapsed
        mean, overall = self.session_mean.get(namespace), self.global_mean()
        if mean is not None and overall:
            base *= min(4.0, max(0.25, mean / overall))
        return base

    def snapshot(self):
        data = self._sorted_samples()
        q = (lambda p: data[int(p * (len(data) - 1))]) if data else (lambda p: None)
        return dict(observations=self.observations, samples=len(data), p50_s=q(0.5), p90_s=q(0.9),
                    tracked_sessions=len(self.session_mean))
