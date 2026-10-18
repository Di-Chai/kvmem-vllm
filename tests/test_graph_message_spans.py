"""Keep original message boundary evidence through public planning/transport."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.session_control import SessionBinding, SessionLedger


SPANS = [{'begin': 0, 'end': 4, 'role': 'system'},
         {'begin': 6, 'end': 17, 'role': 'user'},
         {'begin': 17, 'end': 24, 'role': 'tool'}]
EXPECTED = ((0, 4, 'system'), (6, 17, 'user'), (17, 24, 'tool'))


def test_message_span_metadata_roundtrip_is_immutable_and_retains_unknown_gaps():
    payload = {'kvmem_message_spans': [dict(s) for s in SPANS], 'kvmem_mandatory_parts': 'none'}
    plan = RequestPlan.parse(payload, 24, 4, 12)
    payload['kvmem_message_spans'][0]['end'] = 3
    assert plan.message_spans == EXPECTED
    assert not plan.mandatory
    assert plan.payload()['kvmem_message_spans'] == SPANS
    replay = RequestPlan.parse(plan.payload(), 24, 4, 12)
    assert replay.message_spans == EXPECTED and replay.mandatory == plan.mandatory
    assert replace(plan, prefill_window='keep_selected').message_spans == EXPECTED


def test_old_positional_request_and_binding_construction_stays_compatible():
    old = RequestPlan(None, 'auto', 'pressure', 4, 12, None, (), False, 0, 2, False)
    assert old.output_tokens == 2 and old.message_spans == ()
    assert 'kvmem_message_spans' not in old.payload()
    binding = SessionBinding(('tenant', 'session'), 1, 'req', (1, 2), 0, 0, False, None, ())
    assert binding.message_spans == ()


def test_native_http_and_session_binding_preserve_message_spans(monkeypatch):
    from vllm_kvmem.http_sessions import normalize_query_plan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(block_tokens=4,
        gen_budget=12, select_budget=16)))
    payload = dict(prompt=list(range(24)), max_tokens=2,
        kvmem_message_spans=SPANS, kvmem_mandatory_parts='none')
    assert normalize_query_plan(payload)
    assert payload['kv_transfer_params']['kvmem_request_plan']['kvmem_message_spans'] == SPANS
    request = SimpleNamespace(request_id='r', session_id='owned', cache_salt='tenant',
        prompt_token_ids=payload['prompt'], num_prompt_tokens=24, num_computed_tokens=0,
        max_tokens=2, kv_transfer_params=payload['kv_transfer_params'])
    ledger = SessionLedger(2, 4, lambda *args: None, lambda *args: None, lambda *args: None,
        select_budget=16, gen_budget=12)
    assert ledger.lookup(request, 0) == 0
    ledger.allocated(request, 0)
    binding, = ledger.take_commands().bindings
    assert binding.message_spans == EXPECTED
    assert binding.namespace == ('tenant', 'owned')


def test_continuation_plan_does_not_discard_supplied_message_boundaries():
    plan = RequestPlan.parse({'kvmem_message_spans': SPANS}, 24, 4, 12)
    effective, _ = plan.continuation(24, 2, 17, 12, 28)
    assert effective.message_spans == EXPECTED
    assert effective.payload()['kvmem_message_spans'] == SPANS
    assert RequestPlan.parse(effective.payload(), 24, 4, 12).mandatory == effective.mandatory


def test_transport_does_not_reenable_filtered_query_mandatory_policy():
    payload = {'kvmem_message_spans': SPANS, 'kvmem_query_span': [10, 13],
               'kvmem_mandatory_parts': ['tool-result']}
    plan = RequestPlan.parse(payload, 24, 4, 12)
    assert plan.mandatory == ((17, 24, 'tool-result'),)
    for _ in range(3):
        plan = RequestPlan.parse(plan.payload(), 24, 4, 12)
        assert plan.mandatory == ((17, 24, 'tool-result'),)


@pytest.mark.parametrize('spans', [
    [{'begin': 0, 'end': 12, 'role': 'user'}, {'begin': 8, 'end': 16, 'role': 'tool'}],
    [{'begin': 0, 'end': 25, 'role': 'user'}],
    [{'begin': 0, 'end': 4, 'role': 'unknown'}],
])
def test_invalid_message_boundaries_still_fail_before_transport(spans):
    with pytest.raises(ValueError):
        RequestPlan.parse({'kvmem_message_spans': spans}, 24, 4, 12)


def test_scheduler_refresh_carries_original_message_boundaries():
    pytest.importorskip('vllm')
    from test_scheduler_transactions import setup_scheduler
    scheduler, request, _ = setup_scheduler()
    scheduler.begin_refresh(request, 32, message_spans=EXPECTED)
    command = scheduler.transactions[request.request_id][0]
    assert command.message_spans == EXPECTED
    assert scheduler.connector.commands[-1].message_spans == EXPECTED


def test_explicit_close_transports_a_narrow_worker_allowlist_without_replacement_binding():
    ledger = SessionLedger(2, 4, lambda *args: None, lambda *args: None, lambda *args: None)
    close = SimpleNamespace(request_id='close', session_id='owned', cache_salt='tenant',
        prompt_token_ids=[1], num_prompt_tokens=1, num_computed_tokens=0,
        max_tokens=1, kv_transfer_params={'kvmem_release_session': True})
    assert ledger.lookup(close, 0) == 0
    ledger.allocated(close, 0)
    commands = ledger.take_commands()
    assert commands.release_requests == {'close'}
    assert not commands.bindings and not commands.captures and not commands.terminals
    assert not ledger.take_commands().release_requests
    ledger.finished(close)
    assert not ledger.uncached_requests
    anonymous = SimpleNamespace(**{**vars(close), 'request_id': 'unowned', 'session_id': None,
        'kv_transfer_params': {}})
    assert ledger.lookup(anonymous, 0) == 0
    ledger.allocated(anonymous, 0)
    assert not ledger.take_commands().release_requests


def test_ordinary_uncached_overflow_session_does_not_receive_release_exemption():
    ledger = SessionLedger(1, 4, lambda *args: None, lambda *args: None, lambda *args: None)
    def request(name):
        return SimpleNamespace(request_id=name, session_id=name, cache_salt='tenant',
            prompt_token_ids=[1], num_prompt_tokens=1, num_computed_tokens=0,
            max_tokens=1, kv_transfer_params={})
    one, overflow = request('one'), request('overflow')
    assert ledger.lookup(one, 0) == 0
    ledger.allocated(one, 0)
    assert ledger.lookup(overflow, 0) == 0
    ledger.allocated(overflow, 0)
    commands = ledger.take_commands()
    assert overflow.request_id in ledger.uncached_requests
    assert not commands.release_requests


def test_graph_ledger_waits_for_owned_capacity_and_rejects_anonymous_requests():
    ledger = SessionLedger(1, 4, lambda *args: None, lambda *args: None, lambda *args: None,
        require_owned=True)
    def request(name, session):
        return SimpleNamespace(request_id=name, session_id=session, cache_salt='tenant',
            prompt_token_ids=[1], num_prompt_tokens=1, num_computed_tokens=0,
            max_tokens=1, kv_transfer_params={})
    first, queued = request('first', 'first'), request('queued', 'queued')
    assert ledger.lookup(first, 0) == 0
    ledger.allocated(first, 0)
    assert ledger.lookup(queued, 0) is None
    assert not ledger.uncached_requests and queued.request_id not in ledger.requests
    with pytest.raises(ValueError, match='owned session'):
        ledger.lookup(request('anonymous', None), 0)
    # A close request is explicitly uncached and may retire an absent namespace
    # even while every graph slot is in use.
    close = request('close', 'absent')
    close.kv_transfer_params = {'kvmem_release_session': True}
    assert ledger.lookup(close, 0) == 0
    ledger.allocated(close, 0)
    assert ledger.take_commands().release_requests == {'close'}


def test_graph_http_ownership_gate_rejects_before_native_and_accepts_body_header_extras(monkeypatch):
    import asyncio
    import httpx
    from fastapi import FastAPI
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from vllm_kvmem.http_sessions import validate_prompt_session
    monkeypatch.setenv('KVMEM_REQUIRE_OWNED_SESSION', '1')
    monkeypatch.delenv('KVMEM_REQUEST_PLAN_CONFIG', raising=False)
    received = []
    async def endpoint(request):
        payload = await request.json()
        received.append(payload)
        return JSONResponse(payload)
    app = FastAPI(routes=[Route('/v1/completions', endpoint, methods=['POST'])])
    app.middleware('http')(validate_prompt_session)
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            rejected = await client.post('/v1/completions', json={'prompt': [1], 'max_tokens': 1})
            assert rejected.status_code == 400 and not received
            for body, headers in (({'session_id': 'body'}, {}), ({}, {'X-Session-ID': 'header'}),
                                  ({'vllm_xargs': {'session_id': 'extra'}}, {})):
                valid = await client.post('/v1/completions', json=dict(prompt=[1], max_tokens=1, **body), headers=headers)
                assert valid.status_code == 200
            close = await client.post('/v1/completions', json=dict(prompt=[1], max_tokens=1,
                session_id='close', kv_transfer_params={'kvmem_release_session': True}))
            assert close.status_code == 200
            monkeypatch.delenv('KVMEM_REQUIRE_OWNED_SESSION')
            off = await client.post('/v1/completions', json={'prompt': [1], 'max_tokens': 1})
            assert off.status_code == 200
    asyncio.run(run())
