"""Lifecycle regressions derived from qw3's refresh-policy tests."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.guided_refresh import (
    RefreshInput, RequestEpoch, refresh_decision, requires_pressure,
    text_safe, prepare_request, scheduled_refresh, admit_epoch,
)
from vllm_kvmem.request_plan import RequestPlan


def test_qw3_initial_grace_then_one_private_refresh():
    value = RefreshInput(known=True, same_query=True, prompt=65537,
                         budget=65536, trigger=28672)
    before = refresh_decision(value)
    assert before.grace and before.initial_grace and before.until_refresh == 28671
    assert not before.reselect and not before.private
    assert refresh_decision(replace(value, prompt=94207)).until_refresh == 1
    at = refresh_decision(replace(value, prompt=94208))
    assert at.reselect and at.private and at.reason == 'initial-headroom'


def test_qw3_new_user_is_public_and_stable_tools_share_epoch():
    base = RefreshInput(known=True, same_query=True, selected=True,
                        prompt=264000, budget=65536, trigger=28672, delta=28000)
    grace = refresh_decision(base)
    assert grace.grace and not grace.reselect and grace.until_refresh == 672
    due = refresh_decision(replace(base, delta=28672))
    assert due.reselect and due.private and due.reason == 'cross-turn-growth'
    user = refresh_decision(replace(base, same_query=False))
    assert user.reselect and not user.private and user.reason == 'new-user-query'
    assert not refresh_decision(replace(base, delta=40000), enabled=False).reselect
    assert not refresh_decision(replace(base, force=True)).private
    assert not refresh_decision(replace(base, known=False)).private
    evidence = refresh_decision(replace(base, evidence_changed=True))
    assert evidence.reselect and evidence.private


def test_qw3_large_tool_callback_requires_pressure_ingest_only_at_capacity():
    assert not requires_pressure(True, True, 28672, 400000, 65536, 32768)
    assert requires_pressure(True, True, 32768, 400000, 65536, 32768)
    assert not requires_pressure(True, False, 0, 94208, 65536, 32768)
    assert requires_pressure(True, False, 0, 98304, 65536, 32768)
    assert not requires_pressure(False, True, 999999, 999999, 65536, 32768)


@pytest.mark.parametrize('text,safe', [
    ('analysis complete', False), ('analysis complete.', True),
    ('analysis <tool_', False), ('<tool_call>{"args":"unfinished', False),
    ('<tool_call>x</tool_call>', True), ('```json\n{', False),
    # qw3 treats the closing backticks themselves as a partial-marker suffix;
    # the following newline is its first safe point.
    ('```\nx\n```', False), ('```\nx\n```\n', True), ('done。\t ', True), ('\n', True),
    ('<function=foo>argument.\n', False), ('<function=foo>x</function>\n', True),
])
def test_qw3_safe_fragment_boundaries(text, safe):
    assert text_safe(text) is safe


def settings():
    return NS(guided_query_tokens=512, guided_reselect='both', guided_trigger=12288,
              guided_max_refreshes=2, guided_reserve=768,
              memory=NS(select_budget=36864, gen_budget=16384))


def test_explicit_raw_query_and_real_agent_user_have_separate_guidance():
    plan = RequestPlan((60000, 60030), block_tokens=32)
    entry = NS(history=0, guided_anchor=0)
    raw, agent = NS(num_prompt_tokens=60032), NS(num_prompt_tokens=60032)
    prepare_request(raw, plan, entry, settings())
    prepare_request(agent, replace(plan, query_mode='agent'), entry, settings())
    assert raw._kvmem_guided_epoch.raw_private
    assert not agent._kvmem_guided_epoch.raw_private
    assert agent._kvmem_guided_epoch.boundary is None


def test_tool_boundary_preserves_epoch_and_only_committed_selection_resets_anchor():
    entry = NS(history=80000, guided_anchor=70000, prompt_guided_anchor=0)
    request = NS(num_prompt_tokens=82320)
    plan = RequestPlan(None, block_tokens=32, query_mode='agent', rolling_active_window=True)
    result = prepare_request(request, plan, entry, settings(), imported=True)
    epoch = request._kvmem_guided_epoch
    assert result.prefill_window == 'keep_selected'
    assert epoch.boundary == 82304 and epoch.boundary_reason == 'cross-turn-growth'
    assert epoch.anchor == 70000 and epoch.refreshes == 0
    assert scheduled_refresh(request, result, 82304, 33136, 53248) == (
        'guided', 'cross-turn-growth', True)
    assert epoch.anchor == 70000  # Preparing or retrying is not committing.
    assert admit_epoch(request, result, 82304, 8, 33136, 53248) == 0
    assert admit_epoch(request, result, 82304, 32, 33136, 53248) == 16
    epoch.committed(82304)
    assert epoch.anchor == 82304 and epoch.boundary is None
    assert epoch.deadline == 94592 and epoch.refreshes == 1


def test_restoring_before_refresh_cannot_reuse_future_anchor():
    entry = NS(history=60000, guided_anchor=80000, prompt_guided_anchor=50000)
    request = NS(num_prompt_tokens=61000)
    plan = RequestPlan(None, query_mode='agent', block_tokens=32)
    result = prepare_request(request, plan, entry, settings(), imported=True)
    assert result.prefill_window == 'keep_selected'
    assert request._kvmem_guided_epoch.anchor == 50000
    assert request._kvmem_guided_epoch.boundary is None


def test_decode_never_reads_draft_suffix_and_emergency_skips_private_generation():
    epoch = RequestEpoch(36864, True, 12288, 2, decode_enabled=True)
    request = NS(num_prompt_tokens=49000, all_token_ids=list(range(50000)),
                 _kvmem_guided_epoch=epoch)
    plan = RequestPlan(None, block_tokens=32)
    seen = []
    tokenizer = NS(decode=lambda tokens, **kw: (seen.append(tokens) or 'complete.'))
    assert scheduled_refresh(request, plan, 49152, 0, 53248, tokenizer)[2]
    assert seen == [list(range(49000, 49152))]
    assert admit_epoch(request, plan, 49151, 8, 0, 53248) == 1
    assert admit_epoch(request, plan, 49152, 8, 0, 53248) == 1
    request._kvmem_guided_epoch.decision_history = -1
    tokenizer.decode = lambda *a, **kw: '<tool_call>unfinished'
    assert scheduled_refresh(request, plan, 49184, 0, 53248, tokenizer) is None
    assert scheduled_refresh(request, plan, 52736, 0, 53248, tokenizer) == (
        'guided', 'emergency-open-fragment', False)


def test_seed_is_bounded_immutable_and_never_valid_before_its_query():
    import torch
    from vllm_kvmem.guided_seed import publish_seed, original_queries
    class Owner:
        pass
    runtime, first, other = NS(), Owner(), Owner()
    q = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    publish_seed(runtime, first, q, 72)
    q.zero_()
    assert original_queries(first, 71, 'cpu') is None
    assert original_queries(first, 72, 'cpu').sum() == 120
    assert original_queries(other, 999, 'cpu') is None
    assert first.guided_seed.bytes == 64


def test_seed_memory_exhaustion_cannot_leave_previous_query_as_current(monkeypatch):
    import torch
    import vllm_kvmem.guided_seed as seed
    class Owner:
        pass
    runtime, first, second = NS(), Owner(), Owner()
    monkeypatch.setattr(seed, 'MAX_TOTAL_BYTES', 64)
    q = torch.ones((1, 2, 2, 4))
    seed.publish_seed(runtime, first, q, 20)
    seed.publish_seed(runtime, second, q, 30)
    assert second.guided_seed is None
    assert second.guided_seed_fallback == 'original-query-memory-limit'
    seed.publish_seed(runtime, first, torch.ones((1, 4, 2, 4)), 40)
    assert first.guided_seed is None
