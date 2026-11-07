"""Pure planning for bounded original logical transactions in one forward.

This module neither authorizes a scheduler allocation nor mutates publication,
sampling or graph state. The caller supplies an immutable snapshot of *existing*
native/shared-budget gates. RequestPlan is then evaluated at every original
logical boundaries. A coalescing authorization id is only an existing offline
controller's cost-feedback receipt; it never bypasses these semantic checks.
"""
from dataclasses import dataclass, fields

from .request_plan import RequestPlan


LOGICAL_PREFILL_TOKENS = 2048
SUPPORTED_LOGICAL_PREFILL_TOKENS = (1024, 2048)
# Opt-in clipped final span (window limit): whole authority records only.
PARTIAL_SPAN_TOKENS = 32
# Retained as the compatible minimum physical group, not the selected size.
PHYSICAL_PREFILL_TOKENS = 2 * LOGICAL_PREFILL_TOKENS
# A 16K physical ceiling can retain sixteen original 1K transactions. Native
# token capacity, sample rows and pressure boundaries still gate every span.
MAX_LOGICAL_PREFILL_CHUNKS = 16


def _integer(value, label, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}')


def _immutable_identity(value):
    return (value is None or type(value) in (str,int,bool) or
            type(value) is tuple and all(_immutable_identity(item) for item in value))


@dataclass(frozen=True)
class LogicalPrefillSpan:
    index: int
    history_begin: int
    local_begin: int
    count: int
    physical_offset: int

    def __post_init__(self):
        for name in ('index','history_begin','local_begin','count','physical_offset'):
            _integer(getattr(self,name),name)
        full = self.count in SUPPORTED_LOGICAL_PREFILL_TOKENS
        # A partial span is only legal as the clipped final span of a group;
        # LogicalPrefillGroup checks that position and its chunk size.
        partial = 0 < self.count < max(SUPPORTED_LOGICAL_PREFILL_TOKENS) and self.count % PARTIAL_SPAN_TOKENS == 0
        if (not 0 <= self.index < MAX_LOGICAL_PREFILL_CHUNKS or not (full or partial) or
                (full and self.physical_offset != self.index * self.count) or
                (not full and self.physical_offset not in tuple(self.index * size for size in
                                                                SUPPORTED_LOGICAL_PREFILL_TOKENS))):
            raise ValueError('A logical span must preserve a complete 1K or 2K boundary')

    @property
    def history_end(self):
        return self.history_begin + self.count

    @property
    def local_end(self):
        return self.local_begin + self.count


@dataclass(frozen=True)
class LogicalPrefillGroup:
    request_id: str
    owner_epoch: tuple
    scheduler_step: int
    offset: int
    prompt_tokens: int
    capacity: int
    spans: tuple[LogicalPrefillSpan,...]
    request_plan: RequestPlan
    coalescing_authorization_id: int | None = None

    def __post_init__(self):
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError('Logical prefill requires a nonempty request identity')
        if type(self.owner_epoch) is not tuple or not self.owner_epoch or not _immutable_identity(self.owner_epoch):
            raise ValueError('Owner epoch must be a nonempty immutable identity tuple')
        for name in ('scheduler_step','offset','prompt_tokens','capacity'):
            _integer(getattr(self,name),name)
        if self.coalescing_authorization_id is not None:
            _integer(self.coalescing_authorization_id,'coalescing authorization',minimum=1)
        if (not isinstance(self.request_plan,RequestPlan) or
                any(not _immutable_identity(getattr(self.request_plan,item.name))
                    for item in fields(self.request_plan))):
            raise ValueError('Logical prefill requires an immutable RequestPlan')
        if (type(self.spans) is not tuple or not 2 <= len(self.spans) <= MAX_LOGICAL_PREFILL_CHUNKS or
                any(not isinstance(span,LogicalPrefillSpan) for span in self.spans)):
            raise ValueError('A physical group must contain two to sixteen immutable spans')
        chunk = self.spans[0].count
        if (chunk not in SUPPORTED_LOGICAL_PREFILL_TOKENS or
                any(span.count != chunk for span in self.spans[:-1]) or
                not 0 < self.spans[-1].count <= chunk or
                any(span.physical_offset != span.index * chunk for span in self.spans) or
                any(span.index != index for index,span in enumerate(self.spans)) or
                any(right.history_begin != left.history_end or right.local_begin != left.local_end
                    for left,right in zip(self.spans,self.spans[1:])) or
                any(span.history_begin-span.local_begin != self.offset for span in self.spans) or
                self.history_end >= self.prompt_tokens or self.spans[-1].local_end > self.capacity):
            raise ValueError('Logical spans must be contiguous ordinary nonterminal prefill')

    @property
    def history_begin(self):
        return self.spans[0].history_begin

    @property
    def local_begin(self):
        return self.spans[0].local_begin

    @property
    def history_end(self):
        return self.spans[-1].history_end

    @property
    def count(self):
        return sum(span.count for span in self.spans)

    def accepted_counts(self, retained):
        """Only clip counts; zero spans must bypass append in the consumer."""
        _integer(retained,'retained prefix')
        if retained > self.count:
            raise ValueError('Retained prefix exceeds the physical group')
        return tuple(min(span.count,max(0,retained-span.physical_offset)) for span in self.spans)

    def matches(self, *, request_id, owner_epoch, scheduler_step, history_begin,
                offset, prompt_tokens, capacity, request_plan):
        """Reject stale owner/view/restore/policy snapshots before allocation.

        This does not replace the later physical packet/source-generation lease.
        The caller's epoch must include authority, window and restore identities.
        """
        return (self.request_id,self.owner_epoch,self.scheduler_step,self.history_begin,
                self.offset,self.prompt_tokens,self.capacity,self.request_plan) == (
                request_id,owner_epoch,scheduler_step,history_begin,offset,prompt_tokens,capacity,request_plan)


@dataclass(frozen=True)
class LogicalPrefillEligibility:
    """Snapshot existing admission gates without charging or refunding them.

    native_limits are the native admission ceilings at two to sixteen *original*
    logical boundaries, before RequestPlan.admit. Their length is the attempt limit,
    not permission to cross a control boundary or combine admissions.
    blocked_reasons records prepared refresh/restore, replay, terminal, unknown
    composition or other scheduler gates. boundary_positions records dynamic
    guided/checkpoint/mandatory lifecycle boundaries not owned by RequestPlan.
    """
    scheduled_request_ids: tuple[str,...]
    native_limits: tuple[int,...]
    physical_token_budget: int
    phase_budget: int | None = None
    batch_budget: int | None = None
    native_prefill_lookahead: int = 1
    phase: str = 'prefill'
    blocked_reasons: tuple[str,...] = ()
    boundary_positions: tuple[int,...] = ()
    terminal_history: int | None = None

    def __post_init__(self):
        if (type(self.scheduled_request_ids) is not tuple or
                any(type(rid) is not str or not rid for rid in self.scheduled_request_ids)):
            raise ValueError('Scheduled request ids must be an immutable tuple')
        if (type(self.native_limits) is not tuple or
                not 2 <= len(self.native_limits) <= MAX_LOGICAL_PREFILL_CHUNKS):
            raise ValueError('Two to sixteen native logical admission limits are required')
        for limit in self.native_limits:
            _integer(limit,'native logical limit')
        _integer(self.physical_token_budget,'physical token budget')
        _integer(self.native_prefill_lookahead,'native prefill lookahead',minimum=1)
        for name in ('phase_budget','batch_budget','terminal_history'):
            if getattr(self,name) is not None:
                _integer(getattr(self,name),name)
        if self.phase not in ('prefill','decode','maintenance'):
            raise ValueError('Unknown logical prefill phase')
        if (type(self.blocked_reasons) is not tuple or
                any(type(reason) is not str or not reason for reason in self.blocked_reasons)):
            raise ValueError('Blocked reasons must be an immutable tuple')
        if type(self.boundary_positions) is not tuple:
            raise ValueError('Dynamic boundaries must be an immutable tuple')
        for position in self.boundary_positions:
            _integer(position,'dynamic boundary')


def plan_logical_prefill(request_plan, *, request_id, owner_epoch, scheduler_step,
                         history_begin, offset, prompt_tokens, capacity, eligibility,
                         coalescing_authorization_id=None, logical_chunk_tokens=LOGICAL_PREFILL_TOKENS,
                         partial_tail=False):
    """Return (immutable group, None), or (None, explicit ineligibility reason).

    Select the longest qualifying prefix of two to sixteen original admissions.
    The original minimum-pair checks retain their order and rejection reasons.
    Additional spans stop at the first failed gate/admission, preserving the
    already qualified prefix. Starting at or crossing a control boundary is
    disallowed; ending exactly at one preserves its next scheduler visit. No
    alignment to global history modulo the logical chunk size is introduced.
    """
    if not isinstance(request_plan,RequestPlan) or not isinstance(eligibility,LogicalPrefillEligibility):
        raise ValueError('Logical prefill requires explicit policy and eligibility snapshots')
    if type(logical_chunk_tokens) is not int or logical_chunk_tokens not in SUPPORTED_LOGICAL_PREFILL_TOKENS:
        raise ValueError('Logical prefill must retain a supported original chunk size')
    _integer(request_plan.block_tokens,'logical page tokens',minimum=1)
    for name,value in (('history',history_begin),('offset',offset),('prompt',prompt_tokens),('capacity',capacity)):
        _integer(value,name)
    if history_begin < offset or history_begin-offset > capacity:
        raise ValueError('Invalid logical prefill working extent')
    if eligibility.scheduled_request_ids != (request_id,):
        return None,'single_owner'
    if eligibility.phase != 'prefill':
        return None,'phase'
    if eligibility.blocked_reasons:
        return None,'blocked:'+eligibility.blocked_reasons[0]
    def prefix_gate(count):
        end = history_begin + count
        if history_begin >= prompt_tokens or end >= prompt_tokens:
            return 'terminal_prefill'
        if eligibility.terminal_history is not None and eligibility.terminal_history <= end:
            return 'terminal_prefix'
        if logical_chunk_tokens % request_plan.block_tokens or history_begin % request_plan.block_tokens:
            return 'partial_logical_block'
        if request_plan.query is not None and history_begin < request_plan.replay_end and end > request_plan.replay_begin:
            return 'query_replay'
        checkpoints = (request_plan.prefill_checkpoint(prompt_tokens),request_plan.generation_checkpoint(prompt_tokens))
        if any(point is not None and history_begin <= point < end for point in checkpoints):
            return 'checkpoint'
        if any(history_begin <= point < end for point in eligibility.boundary_positions):
            return 'dynamic_boundary'
        for name,limit in (('physical',eligibility.physical_token_budget),
                           ('phase',eligibility.phase_budget),('batch',eligibility.batch_budget)):
            if limit is not None and limit < count:
                return name+'_budget'
        if prompt_tokens-end < eligibility.native_prefill_lookahead:
            return 'native_lookahead'
        return None

    reason = prefix_gate(2 * logical_chunk_tokens)
    if reason is not None:
        return None,reason
    spans = []
    for index,native_limit in enumerate(eligibility.native_limits):
        if index >= 2 and prefix_gate((index+1)*logical_chunk_tokens) is not None:
            break
        history = history_begin + index*logical_chunk_tokens
        action = request_plan.action(history,prompt_tokens,offset,capacity)
        if action is not None:
            if len(spans) >= 2:
                break
            return None,'action:'+action
        desired = min(logical_chunk_tokens,native_limit)
        count = request_plan.admit(history,desired,prompt_tokens,offset,capacity)
        if count != logical_chunk_tokens:
            # Opt-in: keep a window-clipped final chunk in this physical step.
            # Its boundaries are exactly those of the standalone step it
            # replaces; sampling/statistics are planned per logical span.
            if (partial_tail and len(spans) >= 2 and 0 < count < logical_chunk_tokens and
                    count % request_plan.block_tokens == 0 and count % PARTIAL_SPAN_TOKENS == 0 and
                    prefix_gate(index*logical_chunk_tokens + count) is None):
                spans.append(LogicalPrefillSpan(index,history,history-offset,count,index*logical_chunk_tokens))
            if len(spans) >= 2:
                break
            return None,'logical_admission'
        spans.append(LogicalPrefillSpan(index,history,history-offset,count,index*logical_chunk_tokens))
    return LogicalPrefillGroup(request_id,owner_epoch,scheduler_step,offset,prompt_tokens,
                              capacity,tuple(spans),request_plan,coalescing_authorization_id),None
