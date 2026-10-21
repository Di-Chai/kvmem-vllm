"""Explicit, token-coordinate Query plans for bounded synchronous requests.

Query replay starts at the preceding sealed block. The complete tail + Query
is admitted atomically; a caller receives a capacity error instead of a
silently shortened Query. Prefix pressure never supplies a semantic Query.
"""
from dataclasses import dataclass, replace


FIELDS = {'kvmem_query_span', 'kvmem_query_message_range', 'kvmem_message_spans',
          'kvmem_reselect', 'kvmem_prefill_window', 'kvmem_replay_span',
          'kvmem_mandatory_spans', 'kvmem_mandatory_parts',
          'kvmem_locality_spans', 'kvmem_locality_provenance',
          'kvmem_query_mode', 'kvmem_query_id', 'kvmem_evidence_changed'}
PARTS = {'system', 'root', 'query', 'policy', 'tool-call', 'tool-result'}
KINDS = PARTS | {'file', 'explicit'}


class RequestCapacityError(ValueError):
    """A supported plan whose combined reservations cannot fit this engine."""


def selection_config(config, select_budget, *, selection, pressure_policy='recency'):
    """Fit reservations using the bands of the actual native picker.

    Attention/graph pressure keeps sink and mandatory pages but has no fixed
    recent band. Keep the owner configuration unchanged for subsequent Queries.
    """
    recent = 0 if selection == 'pressure' and pressure_policy in ('attention', 'graph') else config.recent_blocks
    return replace(config, select_budget=select_budget, recent_blocks=recent)


def _parts(value):
    selected = {'system', 'root', 'query', 'policy'}
    if value is None or value == 'default':
        return selected
    if isinstance(value, str):
        return set() if value == 'none' else _parts(value.split(','))
    if isinstance(value, dict):
        if set(value) - PARTS or any(type(v) is not bool for v in value.values()):
            raise ValueError('Invalid mandatory parts')
        return (selected - set(value)) | {k for k,v in value.items() if v}
    if not isinstance(value, list) or any(not isinstance(v,str) or v not in PARTS for v in value) or len(set(value)) != len(value):
        raise ValueError('Invalid mandatory parts')
    return set(value)


def _range(value, limit, label):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(x) is not int for x in value) or not 0 <= value[0] < value[1] <= limit):
        raise ValueError(f'Invalid {label}: expected a nonempty half-open range')
    return tuple(value)


def _adds_historical_mandatory(mandatory, previous, history):
    """Only new requirements inside the verified prefix invalidate its view.

    An envelope can extend into a new Query without adding a historical pin.
    Compare interval coverage by kind, without expanding long token ranges or
    conflating soft root/policy requirements with hard query/file requirements.
    """
    covered = {}
    for begin, end, kind in previous:
        if begin < history:
            covered.setdefault(kind, []).append((begin, min(end, history)))
    for kind, spans in covered.items():
        merged = []
        for begin, end in sorted(spans):
            if merged and begin <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((begin, end))
        covered[kind] = merged
    return any(begin < history and not any(left <= begin and min(end, history) <= right
                   for left, right in covered.get(kind, ()))
               for begin, end, kind in mandatory)


@dataclass(frozen=True)
class RequestPlan:
    query: tuple[int, int] | None
    reselect: str = 'auto'
    prefill_window: str = 'pressure'
    block_tokens: int = 128
    gen_budget: int = 16384
    replay: tuple[int, int] | None = None
    mandatory: tuple[tuple[int, int, str], ...] = ()
    explicit_mandatory: bool = False
    probe_tokens: int = 0
    output_tokens: int = 0
    rolling_active_window: bool = False
    # Original message coordinates, never dense-window positions. Append to
    # preserve positional callers of the pre-existing request-plan contract.
    message_spans: tuple[tuple[int, int, str], ...] = ()
    locality_spans: tuple[tuple[int, int, str], ...] = ()
    locality_provenance: tuple = ()
    query_mode: str = 'raw'
    query_id: str | None = None
    evidence_changed: bool = False

    @property
    def output_reserve(self):
        # The public generation limit stays intact. Rolling mode instead visits
        # an accepted-prefix pressure boundary when the active view fills.
        return 0 if self.rolling_active_window else self.output_tokens

    def generation_checkpoint(self, prompt_tokens):
        # Visit a sealed point while public input still remains. If the final
        # block belongs to an atomic Query, reserve its entire replay instead.
        point = (prompt_tokens - 1) // self.block_tokens * self.block_tokens
        if self.query is not None and point < self.replay_end:
            point = self.replay_begin
        return point

    def generation_pressure(self, history, prompt_tokens, offset, capacity):
        return (self.output_reserve > 0 and history < prompt_tokens and
                history == self.generation_checkpoint(prompt_tokens) and
                prompt_tokens - offset + self.output_reserve > capacity - self.block_tokens)

    def selection_budget(self, history, prompt_tokens, capacity):
        # K counts the final prompt view, including the replay suffix. Leave
        # another block for native speculation; G is never shortened here.
        budget = capacity - self.gen_budget
        if history == 0 and prompt_tokens + self.output_reserve <= capacity - self.block_tokens:
            return budget  # Dense cold Query needs no selected historical prefix.
        if self.output_reserve and history == self.generation_checkpoint(prompt_tokens):
            budget = min(budget, (capacity - self.output_reserve - (prompt_tokens - history)
                                 - self.block_tokens) // self.block_tokens * self.block_tokens)
        if budget <= 0:
            raise RequestCapacityError('Complete Query and output reserve exceed active capacity')
        return budget

    def validate_capacity(self, prompt_tokens, select_budget, *, output_tokens=None,
                          sink_blocks=0, recent_blocks=0, rolling=None,
                          resume_history=0, resume_window=0, pressure_policy='recency'):
        """Validate a finalized plan before allocation or checkpoint mutation.

        The API can call this for cold/forced requests. Session owners must first
        suppress an unchanged auto Query using their exact-prefix proof: an old
        Query which is not replayed consumes no new replay reservation. Preserve
        the same generation checkpoint and fixed keep bands as the real picker.
        """
        plan = replace(self,
            output_tokens=self.output_tokens if output_tokens is None else output_tokens,
            rolling_active_window=self.rolling_active_window if rolling is None else rolling)
        capacity = select_budget + self.gen_budget
        checkpoint = plan.generation_checkpoint(prompt_tokens)
        # A restored terminal tail may lie beyond the final sealed input point.
        # SessionLedger then pressure-compacts at that exact import position.
        if (resume_history > checkpoint and plan.query is None and
                resume_window + prompt_tokens - resume_history + plan.output_reserve >
                capacity - self.block_tokens):
            budget = min(select_budget,
                (capacity - plan.output_reserve - (prompt_tokens - resume_history) -
                 self.block_tokens) // self.block_tokens * self.block_tokens)
            needs_selection = True
            selection = 'pressure'
        else:
            budget = plan.selection_budget(checkpoint, prompt_tokens, capacity)
            needs_selection = checkpoint >= budget
            selection = plan.action(checkpoint, prompt_tokens, 0, capacity)
        if selection == 'pressure' and pressure_policy in ('attention', 'graph'):
            recent_blocks = 0
        minimum = max(1, sink_blocks + recent_blocks) * self.block_tokens
        if needs_selection and budget < minimum:
            raise RequestCapacityError(
                'Complete Query, output reserve and fixed sink/recent bands exceed '
                f'active capacity (selected budget {budget}, required {minimum})')
        return budget

    @property
    def replay_begin(self):
        if self.query is None:
            return None
        from . import _native
        start = _native.replay_capture_begin(self.query[0], *(self.replay or (0,0)))
        return start // self.block_tokens * self.block_tokens

    @property
    def replay_end(self):
        return None if self.query is None else (self.replay[1] if self.replay else self.query[1])

    def payload(self):
        out = {'kvmem_reselect':self.reselect,'kvmem_prefill_window':self.prefill_window,
               'kvmem_mandatory_spans':[{'begin':b,'end':e,'kind':k} for b,e,k in self.mandatory]}
        if self.query is not None:
            out['kvmem_query_span'] = list(self.query)
        if self.replay is not None:
            out['kvmem_replay_span'] = list(self.replay)
        if self.message_spans:
            out['kvmem_message_spans'] = [{'begin': b, 'end': e, 'role': role}
                                        for b, e, role in self.message_spans]
            # Mandatory policy is already resolved in self.mandatory. Message
            # boundaries are independent evidence, so reparsing this transport
            # must not silently re-enable default root/system/query pins.
            out['kvmem_mandatory_parts'] = 'none'
        if self.locality_spans:
            from .prompt_spans import span_payload
            out['kvmem_locality_spans'] = span_payload(self.locality_spans)
        if self.locality_provenance:
            out['kvmem_locality_provenance'] = dict(self.locality_provenance)
        if self.query_mode != 'raw' or self.query_id is not None or self.evidence_changed:
            out.update(kvmem_query_mode=self.query_mode, kvmem_query_id=self.query_id,
                       kvmem_evidence_changed=self.evidence_changed)
        return out

    def continuation(self, prompt_tokens, output_tokens, history, window, capacity,
                     previous_query=None, previous_mandatory=()):
        """Finalize against the exact checkpoint, preserving a stable epoch."""
        query = self.query
        if self.reselect == 'auto' and query == previous_query and query is not None and query[1] <= history:
            query = None  # Exact prompt-prefix validation proves the old Q is unchanged.
        mandatory = self.mandatory
        if not self.explicit_mandatory:
            inherited = tuple(s for s in previous_mandatory if query is None or s[2] != 'query')
            mandatory = tuple(dict.fromkeys(inherited + mandatory))
        changed = _adds_historical_mandatory(mandatory, previous_mandatory, history)
        reserve = 0 if self.rolling_active_window else output_tokens
        fits = window + prompt_tokens - history + reserve <= capacity - self.block_tokens
        policy = self.prefill_window
        if query is None and policy != 'semantic_chunk':
            policy = 'keep_selected' if fits and not changed else 'pressure'
        return replace(self,query=query,replay=self.replay if query else None,
                       mandatory=mandatory,prefill_window=policy), changed

    def mandatory_fit(self, history, config, *, preserve_tail=False):
        from . import _native
        bt = self.block_tokens
        count = (history + bt - 1) // bt
        spans = [(b//bt,min(count,(e+bt-1)//bt),1 if k in ('root','policy') else 0)
                 for b,e,k in self.mandatory if b < history]
        spans = [s for s in spans if s[0] < s[1]]
        tail = count - 1 if preserve_tail and history % bt else None
        fixed = set()
        if tail is not None:
            fixed.update(range(min(count, config.sink_blocks)))
            fixed.add(tail)
            # Explicit hard evidence cannot be softened to make room for a
            # resumed partial record. Root/policy spans retain native fitting.
            for begin, end, kind in self.mandatory:
                if kind == 'explicit':
                    fixed.update(range(begin // bt, min(count, (end + bt - 1) // bt)))
            if len(fixed) > config.select_budget // bt:
                raise RequestCapacityError('Sink, explicit mandatory and accepted tail exceed active capacity')
        result = _native.fit_mandatory_blocks(count,config.select_budget//bt,
            config.sink_blocks,config.recent_blocks,spans,
            live_begin=tail or 0,live_end=count if tail is not None else 0)
        if not fixed.issubset(result['hard_blocks']):
            raise RequestCapacityError('Explicit mandatory evidence cannot fit beside accepted tail')
        return result

    def prefill_checkpoint(self,prompt_tokens):
        if self.query is not None:
            return self.replay_begin
        if self.prefill_window=='semantic_chunk' or any(k=='query' for b,e,k in self.mandatory):
            return None  # Retain the old Query origin on stable tool callbacks.
        from . import _native
        point=_native.branch_safe_checkpoint(prompt_tokens,0,self.block_tokens)
        return point if point and point%self.block_tokens==0 else None

    def prefill_boundary(self, history, capacity):
        # Reserve a replay block AND native MTP allocation lookahead. Pressure
        # must be visited before allocation; a filled old view cannot be probed.
        boundary = capacity - (1 if self.prefill_window == 'keep_selected' else 2) * self.block_tokens
        if self.query is not None and history < self.replay_begin:
            query_boundary = (capacity - (self.replay_end - self.replay_begin) - self.block_tokens - self.probe_tokens) // self.block_tokens * self.block_tokens
            boundary = min(boundary, query_boundary)
        return boundary

    @classmethod
    def parse(cls, payload, prompt_tokens, block_tokens, gen_budget, probe_tokens=0):
        if not isinstance(payload, dict) or set(payload) - FIELDS:
            raise ValueError('Unknown KVMem request-plan fields')
        mode = payload.get('kvmem_reselect', 'auto')
        window = payload.get('kvmem_prefill_window', 'pressure')
        if mode not in ('auto', 'force', 'off') or window not in ('pressure', 'keep_selected', 'semantic_chunk'):
            raise ValueError('Invalid KVMem reselection/prefill policy')
        query_mode = payload.get('kvmem_query_mode', 'raw')
        query_id = payload.get('kvmem_query_id')
        changed = payload.get('kvmem_evidence_changed', False)
        if query_mode not in ('raw', 'agent') or type(changed) is not bool:
            raise ValueError('Invalid Query lifecycle mode or evidence-change flag')
        if query_id is not None and (not isinstance(query_id, str) or not 0 < len(query_id) <= 1024):
            raise ValueError('Query identity must be a nonempty bounded string')
        query = payload.get('kvmem_query_span')
        spans = payload.get('kvmem_message_spans', [])
        if not isinstance(spans, list) or any(not isinstance(s, dict) for s in spans):
            raise ValueError('Message spans must be an ordered list')
        from .prompt_spans import normalize_spans, normalize_provenance, combine_spans
        normalized_messages = normalize_spans(spans, prompt_tokens, label='Message')
        locality = payload.get('kvmem_locality_spans', [])
        if not isinstance(locality, list) or any(not isinstance(s, dict) for s in locality):
            raise ValueError('Locality spans must be an ordered list')
        normalized_locality = normalize_spans(locality, prompt_tokens)
        combine_spans(normalized_messages, normalized_locality)
        provenance = normalize_provenance(payload.get('kvmem_locality_provenance'))
        if 'kvmem_query_message_range' in payload:
            if query is not None:
                raise ValueError('Specify Query token span or message range, not both')
            lo, hi = _range(payload['kvmem_query_message_range'], len(spans), 'Query message range')
            # Explicit whole-round Queries preserve all supplied role/token
            # boundaries, including intervening assistant and tool messages.
            # The caller owns tokenization; this does not infer a chat Query.
            query = [spans[lo]['begin'], spans[hi-1]['end']]
        if query is not None:
            query = _range(query, prompt_tokens, 'Query span')
        if mode == 'force' and query is None:
            raise ValueError('force requires an explicit Query')
        if mode == 'off' and query is not None:
            raise ValueError('off forbids a Query')
        if window == 'semantic_chunk' and query is not None:
            raise ValueError('semantic_chunk and explicit Query are separate policies')
        replay = payload.get('kvmem_replay_span')
        if replay is not None:
            replay = _range(replay,prompt_tokens,'live replay span')
            if query is None or not replay[0] <= query[0] < query[1] <= replay[1]:
                raise ValueError('Live replay must contain the complete scoring Query')
        mandatory = []
        parts = _parts(payload.get('kvmem_mandatory_parts'))
        root = next((s for s in spans if s['role']=='user'),None)
        for span in spans:
            kind = ('system' if span['role']=='system' else 'policy' if span['role']=='developer'
                    else 'root' if span is root else 'tool-result' if span['role']=='tool' else None)
            if kind in parts:
                mandatory.append((span['begin'],span['end'],kind))
        if query is not None and 'query' in parts:
            mandatory.append((*query,'query'))
        explicit = payload.get('kvmem_mandatory_spans',[])
        if not isinstance(explicit,list):
            raise ValueError('Mandatory spans must be a list')
        for span in explicit:
            if not isinstance(span,dict) or set(span) != {'begin','end','kind'} or span['kind'] not in KINDS:
                raise ValueError('Invalid mandatory span')
            begin,end = _range([span['begin'],span['end']],prompt_tokens,'mandatory span')
            mandatory.append((begin,end,span['kind']))
        plan = cls(query, mode, window, block_tokens, gen_budget, replay,
                   tuple(dict.fromkeys(mandatory)), bool(set(payload)&{'kvmem_mandatory_spans','kvmem_message_spans','kvmem_mandatory_parts'}),probe_tokens,
                   message_spans=normalized_messages, locality_spans=normalized_locality,
                   locality_provenance=provenance, query_mode=query_mode, query_id=query_id,
                   evidence_changed=changed)
        if query is not None and plan.replay_end - plan.replay_begin > gen_budget:
            raise ValueError('Complete partial tail + Query exceeds G replay capacity')
        if query is not None and plan.replay_end - plan.replay_begin > gen_budget - block_tokens - probe_tokens:
            raise ValueError('Query replay must leave one block of pressure headroom')
        return plan

    def admit(self, history, desired, prompt_tokens, offset, capacity):
        local = history - offset
        if min(history, desired, local) < 0 or local > capacity:
            raise ValueError('Invalid explicit working-set extent')
        if history >= prompt_tokens:
            return min(desired, capacity - local)
        checkpoint=self.prefill_checkpoint(prompt_tokens)
        if checkpoint is not None and history<checkpoint:
            desired=min(desired,checkpoint-history)
        generation = self.generation_checkpoint(prompt_tokens)
        if (self.output_reserve and history < generation and
                prompt_tokens - offset + self.output_reserve > capacity - self.block_tokens):
            desired = min(desired, generation - history)
        if self.query is not None:
            begin, end = self.replay_begin, self.replay_end
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
        if self.generation_pressure(history, prompt_tokens, offset, capacity):
            return 'pressure'
        if self.prefill_window == 'semantic_chunk' and history < prompt_tokens:
            return 'chunk'
        boundary = (self.prefill_boundary(history, capacity) if history < prompt_tokens
                    else capacity - self.block_tokens)
        if history - offset >= boundary:
            rolling_decode = self.rolling_active_window and history >= prompt_tokens
            return 'pressure' if rolling_decode or self.prefill_window != 'keep_selected' else 'exhausted'
        return None


def request_plan(request, block_tokens, gen_budget):
    import os, json
    config = json.loads(os.getenv('KVMEM_REQUEST_PLAN_CONFIG','{}'))
    reserve = config.get('guided_reserve',0)
    rolling = config.get('rolling_active_window',False)
    if type(rolling) is not bool:
        raise ValueError('rolling_active_window must be a server boolean')
    params = getattr(request, 'kv_transfer_params', None) or {}
    payload = params.get('kvmem_request_plan')
    if payload is None:
        return None
    plan = RequestPlan.parse(payload, request.num_prompt_tokens, block_tokens, gen_budget,reserve)
    from .session_control import SessionLedger
    return replace(plan, output_tokens=SessionLedger.output_tokens(request),
                   rolling_active_window=rolling)
