"""Explicit synthetic GC throughput fixture; no model or attention observation.

Loaded only by an owned loopback qualification worker extension. RPC arms a
frozen plan before decode. The existing authority poll boundary publishes the
same evidence in off/delete; no RPC, GPU operation or file write runs per token.
"""
from dataclasses import asdict, dataclass
import copy
import functools
import os
import sys
import time
import weakref

SOURCE = 'synthetic_gc_throughput_fixture'
ENABLED = 'HISTORICAL_GC_THROUGHPUT_FIXTURE'
COUNTERS = ('retired', 'cpu_released')
_controller = None


@dataclass(frozen=True)
class Plan:
    prompt_tokens: int
    decode_tokens: int
    block_tokens: int
    source_begin: int
    source_end: int
    result_begin: int
    result_end: int
    trigger_decode_tokens: int

    def __post_init__(self):
        if any(type(v) is not int for v in asdict(self).values()):
            raise ValueError('Fixture coordinates must be integers')
        b = self.block_tokens
        if (b <= 0 or self.prompt_tokens < 0 or self.prompt_tokens % b or
                not self.prompt_tokens < self.source_begin < self.source_end <= self.result_begin < self.result_end or
                any(v % b for v in (self.source_begin, self.source_end, self.result_begin, self.result_end)) or
                not self.result_end + b <= self.prompt_tokens + self.trigger_decode_tokens or
                not 0 < self.trigger_decode_tokens < self.decode_tokens):
            raise ValueError('Fixture needs sealed generated source/result blocks before its trigger')

    @classmethod
    def standard(cls, prompt_tokens, decode_tokens):
        if prompt_tokens != 256 or decode_tokens < 512:
            raise ValueError('Synthetic C1 qualification requires 256 prompt tokens and >=512 decode tokens')
        return cls(prompt_tokens, decode_tokens, 32, prompt_tokens+32,
                   prompt_tokens+160, prompt_tokens+160, prompt_tokens+192, 256)


class Controller:
    def __init__(self, clock=time.perf_counter, sample_stride=16):
        if type(sample_stride) is not int or sample_stride <= 0:
            raise ValueError('Sampling stride must be a positive accepted-token count')
        self.clock, self.sample_stride = clock, sample_stride
        self.state = None
        self.runner = None

    def arm(self, runner, namespace, plan):
        if (not isinstance(namespace, (list, tuple)) or len(namespace) != 2 or
                any(not isinstance(v, str) or not v for v in namespace)):
            raise ValueError('A nonempty salt/session namespace is required')
        runtime = getattr(runner, '_kvmem_capture_runtime', None)
        if runtime is None or runtime.archive._history_policy != 'statistical':
            raise ValueError('Warm an owned statistical GC server before arming')
        if getattr(runner, '_kvmem_native_attention', None) is not None:
            raise ValueError('This fixture must not enable SM12 native attention')
        self.runner = weakref.ref(runner)
        self.plan = plan
        self.state = dict(source=SOURCE, semantic_measurement=False,
            namespace=list(namespace), plan=asdict(plan), phase='armed',
            armed_at=self.clock(), sample_stride=self.sample_stride, events=[], poll_calls=0, attempts=0,
            observations_published=0, registered=False, complete=False,
            flushed=False, error=None, request_id=None, baseline=None, last_sample=None,
            candidate_handles=[], result_handles=[])
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy(self.state)

    def _sample(self, authority):
        start = self.clock()
        stats = dict(authority.gc.scope.stats)
        sample = dict(started_at=start, finished_at=self.clock(),
                      counters={k: stats[k] for k in COUNTERS},
                      policy_failures=stats['policy_failures'],
                      accepted_history=authority.history_end)
        state = self.state
        previous = state['last_sample']
        if previous is not None:
            for name in COUNTERS:
                old, new = previous['counters'][name], sample['counters'][name]
                if new < old:
                    raise RuntimeError('GC counters went backwards within one owned scope')
                if new > old:
                    state['events'].append(dict(kind=name, before=old, after=new,
                        interval_begin=previous['started_at'], interval_end=sample['finished_at'],
                        accepted_history=authority.history_end))
        # A later accepted-token append proves decode continued after the
        # reclamation observation; last SSE arrival alone could lag finalization.
        for event in state['events']:
            if (event['kind'] in COUNTERS and 'following_decode_at' not in event and
                    sample['started_at'] > event['interval_end'] and
                    sample['accepted_history'] > event['accepted_history']):
                event['following_decode_at'] = sample['finished_at']
                event['following_accepted_history'] = sample['accepted_history']
        if state['baseline'] is None:
            state['baseline'] = sample
        state['last_sample'] = sample
        if sample['policy_failures']:
            raise RuntimeError('Native statistical policy failed')

    def poll(self, authority):
        state = self.state
        if state is None or state['error'] is not None:
            return
        runner = self.runner()
        sessions = getattr(runner, '_kvmem_sessions', None) if runner is not None else None
        binding = getattr(sessions, 'bindings', {}).get(authority.memory.request_id)
        if binding is None or list(binding.namespace) != state['namespace']:
            return
        state['poll_calls'] += 1
        try:
            plan = self.plan
            if len(state['events']) > 64:
                raise RuntimeError('Fixture event budget exceeded')
            if binding.history or len(binding.tokens) != plan.prompt_tokens:
                raise ValueError('Fixture requires a cold request with the frozen prompt')
            state['request_id'] = authority.memory.request_id
            if authority.memory.config.block_tokens != plan.block_tokens:
                raise ValueError('Fixture block geometry differs')
            if authority.history_end < plan.prompt_tokens+plan.trigger_decode_tokens:
                return
            # Both modes sample through the same accepted-history range, even
            # after delete has finished. Never stop instrumentation only in A/B.
            previous = state['last_sample']
            if previous is None or authority.history_end-previous['accepted_history'] >= self.sample_stride:
                self._sample(authority)
            if state['flushed']:
                return
            state['attempts'] += 1
            if state['attempts'] > 2048:
                raise RuntimeError('Fixture publication did not finish within its bounded attempts')
            if not state['registered']:
                sources = [r for r in authority.records.values()
                    if plan.source_begin <= r.history_begin < plan.source_end]
                results = [r for r in authority.records.values()
                    if plan.result_begin <= r.history_begin < plan.result_end]
                expected = (plan.source_end-plan.source_begin)//plan.block_tokens
                witnesses = (plan.result_end-plan.result_begin)//plan.block_tokens
                if (len(sources) != expected or len(results) != witnesses or
                        any(not r.archived or r.count != plan.block_tokens for r in sources+results)):
                    return  # Ordinary publication will make these visible.
                ledger = authority.gc.evidence
                if ledger is None:
                    ledger = authority.gc.open_evidence(authority.records,
                        epoch=tuple(state['namespace']), max_spans=1)
                if not ledger.register_span('synthetic-step', plan.source_begin, plan.source_end,
                        kind='completed_step', epoch=tuple(state['namespace'])):
                    return
                state['candidate_handles'] = [[r.handle.slot, r.handle.generation] for r in sources]
                state['result_handles'] = [[r.handle.slot, r.handle.generation] for r in results]
                state['registered'] = True
                state['phase'] = 'publishing'
                state['events'].append(dict(kind='registered', at=self.clock(),
                    accepted_history=authority.history_end, candidate_blocks=expected))
            ledger = authority.gc.evidence
            while state['observations_published'] < 3:
                observation = state['observations_published']
                # Deliberate synthetic input: no claim of attention/model data.
                if not ledger.observe('synthetic-step', epoch=tuple(state['namespace']),
                        observation_id=observation, importance=.01, cold=True):
                    return
                state['observations_published'] += 1
            if not state['complete']:
                results = [r for r in authority.records.values()
                    if plan.result_begin <= r.history_begin < plan.result_end]
                if not ledger.complete_span('synthetic-step', results,
                        epoch=tuple(state['namespace'])):
                    return
                state['complete'] = True
            if ledger.flush(max_blocks=32):
                state['flushed'], state['phase'] = True, 'observing'
                state['events'].append(dict(kind='published', at=self.clock(),
                    accepted_history=authority.history_end, observations=3,
                    importance=.01, cold=True, source=SOURCE))
        except Exception as error:
            # Do not alter model execution; the qualification will fail closed.
            state['error'] = type(error).__name__+': '+str(error)
            state['phase'] = 'failed'


def install():
    global _controller
    if _controller is not None:
        return
    from vllm_kvmem.authority import RequestAuthority
    controller = Controller()
    original = RequestAuthority.poll

    @functools.wraps(original)
    def poll(authority, *args, **kwargs):
        result = original(authority, *args, **kwargs)
        controller.poll(authority)
        return result

    RequestAuthority.poll = poll
    _controller = controller


class Extension:
    def historical_gc_fixture_arm(self, namespace, prompt_tokens, decode_tokens):
        if os.environ.get(ENABLED) != '1':
            raise RuntimeError('Explicit fixture startup is required')
        # A fork worker may inherit this module from the launcher before its
        # opt-in environment was applied. Install idempotently in that worker.
        install()
        return _controller.arm(self.model_runner, namespace,
                               Plan.standard(prompt_tokens, decode_tokens))

    def historical_gc_fixture_snapshot(self):
        if _controller is None:
            raise RuntimeError('Explicit fixture startup is required')
        return _controller.snapshot()


def serve():
    """Owned harness bootstrap; ordinary product serving is unchanged."""
    from vllm_kvmem import serve as serving
    original = serving.build_launch

    def build(*args, **kwargs):
        plan = original(*args, **kwargs)
        if plan['profile']['host'] != '127.0.0.1':
            raise ValueError('GC fixture development RPC requires an owned loopback server')
        plan['vllm_argv'] += ['--worker-extension-cls', __name__+'.Extension']
        plan['environment'].update({ENABLED: '1', 'VLLM_SERVER_DEV_MODE': '1'})
        return plan

    serving.build_launch = build
    serving.main(sys.argv[1:])


if os.environ.get(ENABLED) == '1':
    install()
