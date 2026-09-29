"""Reject unsupported namespace sizes before they reach the shared scheduler.

Opted in through vLLM's public --middleware argument. Request-body replay and
streaming/disconnect handling belong to Starlette's native HTTP middleware.
"""
import json

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
            except ValueError as error:
                from starlette.responses import JSONResponse
                return JSONResponse({'error': {'message': str(error),
                    'type': 'BadRequestError', 'param': 'session_id/cache_salt',
                    'code': 400}}, status_code=400)
    return await call_next(request)
