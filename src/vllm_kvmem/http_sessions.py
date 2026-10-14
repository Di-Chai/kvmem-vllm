"""Reject unsupported namespace sizes before they reach the shared scheduler.

Opted in through vLLM's public --middleware argument. Request-body replay and
streaming/disconnect handling belong to Starlette's native HTTP middleware.
"""
import json
import os

from .session_control import session_namespace


def validate_session_payload(payload, header):
    if not isinstance(payload, dict):
        return  # Native request-schema validation owns malformed payloads.
    # Match GenerateBaseServing._get_session_id in the pinned vLLM version:
    # body, then header, then a nonempty string in vllm_xargs.
    session = payload.get('session_id') or header
    if not session:
        extras = payload.get('vllm_xargs')
        if isinstance(extras, dict) and isinstance(extras.get('session_id'), str):
            session = extras['session_id']
    session_namespace(session, payload.get('cache_salt'))


def normalize_query_plan(payload):
    if not isinstance(payload, dict):
        return False  # Preserve native request-schema errors.
    from .request_plan import FIELDS, RequestPlan
    fields = {k: payload[k] for k in FIELDS if k in payload}
    params = dict(payload.get('kv_transfer_params') or {})
    if fields and 'kvmem_request_plan' in params:
        raise ValueError('Duplicate request plan')
    fields = fields or params.get('kvmem_request_plan')
    config = os.getenv('KVMEM_REQUEST_PLAN_CONFIG')
    if fields is not None and not config:
        raise ValueError('Explicit Query planning is not enabled for this backend')
    if not config:
        return False
    cfg = json.loads(config)
    prompt = payload.get('prompt')
    if not isinstance(prompt, list) or any(type(x) is not int or x < 0 for x in prompt):
        if fields:
            raise ValueError('Explicit Query coordinates require a token-ID completion prompt')
        return False  # Native chat remains an independent legacy entry.
    plan = RequestPlan.parse(fields or {}, len(prompt), cfg['block_tokens'], cfg['gen_budget'])
    max_tokens = payload.get('max_tokens')
    if max_tokens is None:
        max_tokens = 16
        payload['max_tokens'] = max_tokens
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if max_tokens > cfg['gen_budget']:
        raise ValueError('Single-turn output must remain within G')
    if plan.prefill_window == 'keep_selected' and len(prompt) + max_tokens > cfg['select_budget'] + cfg['gen_budget'] - cfg['block_tokens']:
        raise ValueError('keep_selected requires verified active headroom; this cold prompt exceeds K+G')
    params['kvmem_request_plan'] = fields or {}
    for key in FIELDS:
        payload.pop(key, None)
    payload['kv_transfer_params'] = params
    return True


async def validate_prompt_session(request, call_next):
    if request.method == 'POST' and request.url.path.rstrip('/') in (
            '/v1/chat/completions', '/v1/completions', '/v1/responses'):
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass  # Preserve native malformed-JSON responses.
        else:
            try:
                validate_session_payload(payload, request.headers.get('X-Session-ID'))
                if normalize_query_plan(payload):
                    request._body = json.dumps(payload).encode()
                    request._json = payload
            except ValueError as error:
                from starlette.responses import JSONResponse
                return JSONResponse({'error': {'message': str(error),
                    'type': 'BadRequestError', 'param': 'session_id/cache_salt',
                    'code': 400}}, status_code=400)
    return await call_next(request)
