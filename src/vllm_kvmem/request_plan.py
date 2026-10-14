"""Explicit, token-coordinate Query plans for bounded synchronous requests.

Query replay starts at the preceding sealed block. The complete tail + Query
is admitted atomically; a caller receives a capacity error instead of a
silently shortened Query. Prefix pressure never supplies a semantic Query.
"""
from dataclasses import dataclass


FIELDS = {'kvmem_query_span', 'kvmem_query_message_range', 'kvmem_message_spans',
          'kvmem_reselect', 'kvmem_prefill_window'}


def _range(value, limit, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(x) is not int for x in value) or not 0 <= value[0] < value[1] <= limit):
        raise ValueError(f'Invalid {label}: expected a nonempty half-open range')
    return tuple(value)


@dataclass(frozen=True)
class RequestPlan:
    query: tuple[int, int] | None
    reselect: str = 'auto'
    prefill_window: str = 'pressure'
    block_tokens: int = 128
    gen_budget: int = 16384

    @property
    def replay_begin(self):
        return None if self.query is None else self.query[0] // self.block_tokens * self.block_tokens

    def prefill_boundary(self, history, capacity):
        # Reserve a replay block AND native MTP allocation lookahead. Pressure
        # must be visited before allocation; a filled old view cannot be probed.
        boundary = capacity - (1 if self.prefill_window == 'keep_selected' else 2) * self.block_tokens
        if self.query is not None and history < self.replay_begin:
            query_boundary = (capacity - (self.query[1] - self.replay_begin) - self.block_tokens) // self.block_tokens * self.block_tokens
            boundary = min(boundary, query_boundary)
        return boundary

    @classmethod
    def parse(cls, payload, prompt_tokens, block_tokens, gen_budget):
        if not isinstance(payload, dict) or set(payload) - FIELDS:
            raise ValueError('Unknown KVMem request-plan fields')
        mode = payload.get('kvmem_reselect', 'auto')
        window = payload.get('kvmem_prefill_window', 'pressure')
        if mode not in ('auto', 'force', 'off') or window not in ('pressure', 'keep_selected', 'semantic_chunk'):
            raise ValueError('Invalid KVMem reselection/prefill policy')
        query = payload.get('kvmem_query_span')
        spans = payload.get('kvmem_message_spans', [])
        if not isinstance(spans, list):
            raise ValueError('Message spans must be an ordered list')
        previous = 0
        for span in spans:
            if (not isinstance(span, dict) or set(span) != {'role', 'begin', 'end'} or
                    span['role'] not in ('system', 'developer', 'user', 'assistant', 'tool')):
                raise ValueError('Invalid message span')
            begin, end = _range([span['begin'], span['end']], prompt_tokens, 'message span')
            if begin < previous:
                raise ValueError('Message spans overlap or are out of order')
            previous = end
        if 'kvmem_query_message_range' in payload:
            if query is not None:
                raise ValueError('Specify Query token span or message range, not both')
            lo, hi = _range(payload['kvmem_query_message_range'], len(spans), 'Query message range')
            if any(s['role'] != 'user' for s in spans[lo:hi]):
                raise ValueError('Query message range must contain user messages')
            query = [spans[lo]['begin'], spans[hi-1]['end']]
        if query is not None:
            query = _range(query, prompt_tokens, 'Query span')
        if mode == 'force' and query is None:
            raise ValueError('force requires an explicit Query')
        if mode == 'off' and query is not None:
            raise ValueError('off forbids a Query')
        if window == 'semantic_chunk' and query is not None:
            raise ValueError('semantic_chunk and explicit Query are separate policies')
        plan = cls(query, mode, window, block_tokens, gen_budget)
        if query is not None and query[1] - plan.replay_begin > gen_budget:
            raise ValueError('Complete partial tail + Query exceeds G replay capacity')
        if query is not None and query[1] - plan.replay_begin > gen_budget - block_tokens:
            raise ValueError('Query replay must leave one block of pressure headroom')
        return plan

    def admit(self, history, desired, prompt_tokens, offset, capacity):
        local = history - offset
        if min(history, desired, local) < 0 or local > capacity:
            raise ValueError('Invalid explicit working-set extent')
        if history >= prompt_tokens:
            return min(desired, capacity - local)
        if self.query is not None:
            begin, end = self.replay_begin, self.query[1]
            if history < begin:
                desired = min(desired, begin - history)
            elif history == begin:
                # Wait for enough native batch credit; never split the Query.
                return end - begin if desired >= end - begin else 0
            elif history < end:
                raise RuntimeError('Explicit Query was split or its boundary was skipped')
        boundary = self.prefill_boundary(history, capacity)
        if local < boundary:
            desired = min(desired, boundary - local)
        else:
            desired = min(desired, self.block_tokens)
        tokens = min(desired, capacity - local, prompt_tokens - history)
        if history + tokens < prompt_tokens:
            tokens = max(0, (history + tokens) // self.block_tokens * self.block_tokens - history)
        return tokens

    def action(self, history, prompt_tokens, offset, capacity):
        if self.query is not None and history == self.replay_begin:
            return 'query'
        if history % self.block_tokens:
            return None
        if self.prefill_window == 'semantic_chunk' and history < prompt_tokens:
            return 'chunk'
        boundary = (self.prefill_boundary(history, capacity) if history < prompt_tokens
                    else capacity - self.block_tokens)
        if history - offset >= boundary:
            return 'pressure' if self.prefill_window != 'keep_selected' else 'exhausted'
        return None


def request_plan(request, block_tokens, gen_budget):
    params = getattr(request, 'kv_transfer_params', None) or {}
    payload = params.get('kvmem_request_plan')
    return None if payload is None else RequestPlan.parse(payload, request.num_prompt_tokens,
                                                         block_tokens, gen_budget)
