"""qw3 automatic retrieval policy, independent of CUDA and HTTP boundaries.

The epoch anchor advances only after a committed semantic selection. Public
output and tool inputs share that anchor; a checkpoint is not a selection.
Raw explicit Query requests retain qw3's separate boundary-query behavior.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RefreshInput:
    known: bool = False
    same_query: bool = False
    selected: bool = False
    incoming: bool = True
    force: bool = False
    reset: bool = False
    prompt: int = 0
    budget: int = 0
    trigger: int = 0
    delta: int = 0
    evidence_changed: bool = False


@dataclass(frozen=True)
class RefreshDecision:
    reselect: bool = False
    private: bool = False
    grace: bool = False
    initial_grace: bool = False
    until_refresh: int = 0
    reason: str = 'none'


def refresh_decision(value, enabled=True):
    """Port of qw3::detail::kvmem_harness_refresh_decision."""
    v = value
    if v.force:
        return RefreshDecision(reselect=True, reason='explicit-force')
    if not v.budget or v.prompt <= v.budget:
        return RefreshDecision()
    if not v.known or v.reset:
        return RefreshDecision(reselect=v.incoming,
                               reason='cold-above-budget' if v.incoming else 'none')
    if not v.same_query:
        return RefreshDecision(reselect=True, reason='new-user-query')
    if v.evidence_changed:
        return RefreshDecision(reselect=True, private=enabled, reason='evidence-changed')
    if not enabled:
        return RefreshDecision()
    if not v.selected:
        target = v.budget + v.trigger
        if not v.trigger or v.prompt >= target:
            return RefreshDecision(reselect=True, private=bool(v.trigger),
                                   reason='initial-headroom')
        return RefreshDecision(grace=True, initial_grace=True,
                               until_refresh=target-v.prompt)
    if v.trigger and v.delta >= v.trigger:
        return RefreshDecision(reselect=True, private=True, reason='cross-turn-growth')
    if v.trigger and v.delta < v.trigger:
        return RefreshDecision(grace=True, until_refresh=v.trigger-v.delta)
    return RefreshDecision()


def requires_pressure(private, selected, delta, prompt, budget, generation):
    return bool(private and generation and
                (delta if selected else max(0, prompt-budget)) >= generation)


def text_safe(text):
    """Same complete-fragment boundaries as qw3's native decoder."""
    if not text:
        return False
    for marker in ('<tool_call>', '</tool_call>', '<function=', '</function>', '```'):
        if any(text.endswith(marker[:n]) for n in range(1, len(marker))):
            return False
    for opening, closing in (('<tool_call>', '</tool_call>'), ('<function=', '</function>')):
        if text.rfind(opening) > text.rfind(closing):
            return False
    if text.count('```') % 2:
        return False
    if text.endswith(('</tool_call>', '```')):
        return True
    return text.rstrip(' \t\r').endswith(('\n', '.', '!', '?', ';', ':', '。', '！', '？'))


@dataclass
class RequestEpoch:
    anchor: int
    selected: bool
    trigger: int
    max_refreshes: int
    boundary: int | None = None
    boundary_reason: str = 'none'
    boundary_private: bool = False
    raw_private: bool = False
    decode_enabled: bool = False
    refreshes: int = 0
    decision_history: int = -1
    decision: tuple = ()

    @property
    def deadline(self):
        return self.anchor + self.trigger

    def committed(self, history, *, automatic=True):
        if history < self.anchor:
            raise ValueError('Guided epoch moved behind its committed anchor')
        self.anchor, self.selected = history, True
        self.refreshes += int(automatic)
        self.boundary = None
        self.decision_history, self.decision = -1, ()


def prepare_request(request, plan, entry, options, *, imported=False):
    """Finalize once after the ledger has proved the exact imported lineage."""
    if plan is None or not options.guided_query_tokens:
        return plan
    mode = options.guided_reselect
    if mode == 'off':
        return plan
    budget, generation = options.memory.select_budget, options.memory.gen_budget
    trigger = options.guided_trigger
    anchor = getattr(entry, 'guided_anchor', 0) if imported else 0
    if anchor > entry.history:
        anchor = getattr(entry, 'prompt_guided_anchor', 0)
    selected = bool(anchor)
    epoch = RequestEpoch(anchor if selected else budget, selected, trigger,
                         options.guided_max_refreshes,
                         raw_private=plan.query_mode == 'raw' and mode in ('boundary', 'both'),
                         decode_enabled=mode in ('middecode', 'both'))
    if plan.query_mode == 'agent':
        same = imported and plan.query is None
        decision = refresh_decision(RefreshInput(
            known=imported, same_query=same, selected=selected,
            incoming=plan.reselect != 'off', force=plan.reselect == 'force',
            prompt=request.num_prompt_tokens, budget=budget,
            trigger=trigger if epoch.decode_enabled else 0,
            delta=max(0, request.num_prompt_tokens-anchor) if selected else 0,
            evidence_changed=plan.evidence_changed), enabled=True)
        if decision.private:
            # The final sealed prefill point leaves a public tail for vLLM's
            # replay transaction. The full tool result is ingested beforehand.
            point = plan.generation_checkpoint(request.num_prompt_tokens)
            end = max(entry.history if imported else 0, point)
            epoch.boundary = (end+plan.block_tokens-1)//plan.block_tokens*plan.block_tokens
            epoch.boundary_private = True
            epoch.boundary_reason = decision.reason
            overflow = requires_pressure(True, selected,
                max(0, request.num_prompt_tokens-anchor), request.num_prompt_tokens,
                budget, generation)
            plan = replace(plan, query=None, replay=None,
                           prefill_window='pressure' if overflow else 'keep_selected')
        elif same and decision.grace:
            plan = replace(plan, prefill_window='keep_selected')
        elif same:
            # Explicit incoming auto never replays an old task Query.
            plan = replace(plan, query=None, replay=None)
        epoch.boundary_reason = decision.reason
    request._kvmem_guided_epoch = epoch
    return plan


def scheduled_refresh(request, plan, history, offset, capacity, tokenizer=None):
    """Read-only refresh decision shared by admission and command creation."""
    epoch = getattr(request, '_kvmem_guided_epoch', None)
    if epoch is None:
        return None
    if epoch.decision_history == history:
        return epoch.decision or None
    result = None
    if epoch.boundary is not None and history >= epoch.boundary and history % plan.block_tokens == 0:
        result = ('guided', epoch.boundary_reason, True)
    elif (epoch.decode_enabled and epoch.refreshes < epoch.max_refreshes and
          history >= request.num_prompt_tokens and history >= epoch.deadline):
        # Native history transactions require a sealed logical page. Never
        # sample proposed/rejected drafts to find the text boundary.
        if history % plan.block_tokens == 0:
            local = history-offset
            emergency = local >= capacity-max(512, plan.block_tokens*2)
            if emergency:
                result = ('guided', 'emergency-open-fragment', False)
            elif tokenizer is not None:
                accepted = request.all_token_ids[request.num_prompt_tokens:history]
                if text_safe(tokenizer.decode(accepted, skip_special_tokens=False)):
                    result = ('guided', 'middecode-growth', True)
    epoch.decision_history, epoch.decision = history, result or ()
    return result


def admit_epoch(request, plan, history, tokens, offset, capacity):
    """Do not skip the pending final-input or accepted decode boundary."""
    epoch = getattr(request, '_kvmem_guided_epoch', None)
    if epoch is None:
        return tokens
    if history < request.num_prompt_tokens:
        if epoch.boundary is not None and history < epoch.boundary:
            return min(tokens, epoch.boundary-history)
        if epoch.boundary == history:
            remaining = request.num_prompt_tokens-history
            return remaining if tokens >= remaining else 0
        return tokens
    if epoch.boundary is not None and history < epoch.boundary:
        return min(tokens, epoch.boundary-history)
    if not epoch.decode_enabled or epoch.refreshes >= epoch.max_refreshes:
        return tokens
    if history < epoch.deadline:
        return min(tokens, epoch.deadline-history)
    # Accepted speculative rows can otherwise jump over the next sealed page.
    return min(tokens, plan.block_tokens-history % plan.block_tokens or plan.block_tokens, 1)
