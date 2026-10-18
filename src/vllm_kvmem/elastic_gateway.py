"""Authenticated, versioned token-history turns above a native offload engine.

The caller appends exact template/tool token IDs. A tool wait spans completed
generation calls, so it owns no native request pages. CPU KV is a bounded cache;
a miss recomputes the exact prompt, rather than silently trimming history.
"""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import time

from .elastic import ActiveBudget, CacheGeometry, HistoryStore, select_context
from .multi_user import _load_users, _loads
from .elastic_stream import generate


class _TurnTimeout(TimeoutError):
    pass


async def _wait_for_request(request, future, turn_deadline, queue_deadline=None):
    """Bound wall time and retain the caller's lease until cancellation drains.

    asyncio.wait does not cancel shared admission futures. The turn's finally
    block removes a queued ticket, or releases an active lease after its native
    HTTP task has closed, including when a client disconnects while queued.
    """
    while True:
        if await request.is_disconnected():
            raise asyncio.CancelledError('Client disconnected')
        now = time.monotonic()
        if now >= turn_deadline:
            raise _TurnTimeout('Turn deadline exceeded; history unchanged')
        if queue_deadline is not None and now >= queue_deadline:
            raise TimeoutError('Active admission deadline exceeded')
        if future.done():
            return future.result()
        deadline = min(turn_deadline, queue_deadline or turn_deadline)
        await asyncio.wait({future}, timeout=min(.1, deadline - now))


def create_app(config):
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from starlette.responses import JSONResponse

    geometry_data = json.loads(Path(config['geometry_file']).read_text())
    geometry = CacheGeometry(tuple(geometry_data['groups']), geometry_data['capacity_bytes'])
    if config['active_bytes'] > geometry.capacity_bytes:
        raise ValueError('Active budget exceeds native cache capacity')
    histories = HistoryStore(config['token_history_bytes'], config['max_sessions'], config['session_ttl_seconds'],
        per_user_bytes=config.get('token_history_bytes_per_user'),
        per_user_sessions=config.get('max_sessions_per_user'))
    budget = ActiveBudget(config['active_bytes'], config['max_inflight'], config['max_queue'])
    restores = ActiveBudget(config.get('restore_bytes', config['active_bytes']),
        config.get('max_restore_inflight', config['max_inflight']), config['max_queue'])
    users = _load_users(config['users_file'])
    counts = {'completed': 0, 'aborted': 0, 'rejected': 0}

    @asynccontextmanager
    async def lifespan(app):
        async with httpx.AsyncClient(base_url=config['backend_url'], timeout=config['turn_timeout_seconds'], trust_env=False) as client:
            app.state.client = client
            yield

    app = FastAPI(lifespan=lifespan)
    # Local annotations avoid importing HTTP dependencies in the host core.
    def authenticate(request):
        headers = request.headers.getlist('authorization')
        parts = headers[0].split() if len(headers) == 1 else []
        user = users.get(hashlib.sha256(parts[1].encode()).digest()) if len(parts) == 2 and parts[0].lower() == 'bearer' else None
        if user is None:
            raise HTTPException(401, 'A valid Bearer API key is required')
        return user

    @app.get('/health')
    async def health():
        response = await app.state.client.get('/health')
        return JSONResponse({'ready': response.status_code == 200}, status_code=response.status_code)

    @app.get('/v1/kvmem/status')
    async def status(request: Request):
        user = authenticate(request)
        histories.expire()
        return {'sessions': [{'session_id': key[1], 'version': s.version,
            'history_tokens': len(s.tokens), 'running': s.busy, 'waiting_for_tool': s.waiting_for_tool}
            for key, s in histories.entries.items() if key[0] == user],
            'active_reserved_bytes': budget.bytes, 'active_requests': len(budget.active),
            'queued_requests': len(budget.waiting), 'token_history_bytes': histories.bytes,
            'restore_reserved_bytes': restores.bytes, 'restoring_requests': len(restores.active),
            'peak_restore_bytes': restores.peak_bytes, 'peak_restoring_requests': restores.peak_requests,
            'cpu_kv_capacity_bytes': geometry_data['cpu_offload_bytes'],
            'peak_active_bytes': budget.peak_bytes, 'peak_active_requests': budget.peak_requests,
            'counters': counts}

    @app.delete('/v1/kvmem/sessions/{session_id}')
    async def delete(session_id: str, request: Request):
        user = authenticate(request)
        try:
            return {'deleted': histories.delete(user, session_id)}
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from None

    @app.post('/v1/kvmem/sessions/{session_id}/turn')
    async def turn(session_id: str, request: Request):
        user = authenticate(request)
        raw = bytearray()
        try:
            async with asyncio.timeout(config.get('body_timeout_seconds', 30)):
                async for chunk in request.stream():
                    if len(raw) + len(chunk) > config['max_body_bytes']:
                        raise HTTPException(413, 'Turn body too large')
                    raw.extend(chunk)
        except TimeoutError:
            raise HTTPException(408, 'Turn body timed out') from None
        try:
            body = _loads(raw)
            from .request_plan import FIELDS
            allowed = {'version', 'append_token_ids', 'max_tokens', 'waiting_for_tool', 'queue_timeout_seconds', 'ignore_eos', 'active_ranges'} | FIELDS
            if not isinstance(body, dict) or set(body) - allowed:
                raise ValueError('Unknown turn fields')
            append = body['append_token_ids']; max_output = body.get('max_tokens', 256)
            if type(max_output) is not int or not 1 <= max_output <= config['max_output_tokens']:
                raise ValueError('Invalid max_tokens')
            for flag in ('waiting_for_tool', 'ignore_eos'):
                if flag in body and type(body[flag]) is not bool:
                    raise ValueError(f'{flag} must be boolean')
            timeout = body.get('queue_timeout_seconds', config['queue_timeout_seconds'])
            if type(timeout) not in (int, float) or not 0 < timeout <= config['queue_timeout_seconds']:
                raise ValueError('Invalid queue deadline')
            key, session, prompt = histories.begin(user, session_id, body.get('version', 0), append, max_output)
        except (KeyError, ValueError) as error:
            raise HTTPException(400, str(error)) from None
        except (RuntimeError, LookupError) as error:
            raise HTTPException(409, str(error)) from None
        except BufferError as error:
            counts['rejected'] += 1
            raise HTTPException(429, str(error)) from None

        ticket = None
        restore_ticket = None
        started = time.monotonic()
        turn_deadline = started + config['turn_timeout_seconds']
        queue_deadline = min(started + timeout, turn_deadline)
        task = None
        try:
            full_prompt_tokens = len(prompt)
            limit = config.get('active_context_tokens', 0)
            mean_k = config.get('context_mode') == 'meank'
            fields = {k: body[k] for k in FIELDS if k in body}
            if fields and not config.get('request_planning'):
                raise ValueError('Explicit Query planning is not enabled')
            if config.get('request_planning'):
                from .request_plan import RequestPlan
                guided = config.get('guided_query_tokens',0)
                block = config.get('block_tokens',128)
                reserve = ((256+guided+block-1)//block)*block if guided else 0
                plan = RequestPlan.parse(fields, len(prompt), block, config['gen_budget_tokens'],reserve)
                if not session.version or plan.reselect == 'force':
                    # Real launch profiles carry resolved bands; hand-built
                    # legacy profiles defer unknown bands to the native owner.
                    plan.validate_capacity(len(prompt), limit, output_tokens=max_output,
                        sink_blocks=config.get('resolved_sink_blocks', config.get('sink_blocks')) or 0,
                        recent_blocks=config.get('resolved_recent_blocks', config.get('recent_blocks')) or 0,
                        rolling=config.get('rolling_active_window', False))
                if plan.query is not None and reserve and plan.replay_end+reserve>config['max_model_len']:
                    raise ValueError('Guided Query scratch exceeds the model position limit')
                # Exact-prefix checkpoint and active-headroom finalization belong
                # to the native session owner, not the logical prompt length.
            ranges = body.get('active_ranges')
            if ranges is not None:
                if mean_k:
                    raise ValueError('MeanK selects historical KV with model queries; manual token ranges are unsupported')
                if not limit:
                    raise ValueError('Context selection is disabled in full-context mode')
                prompt = select_context(prompt, ranges, limit)
            elif limit and len(prompt) > limit and not mean_k:
                raise BufferError('Full history exceeds active context limit; explicit active_ranges required')
            if len(prompt) + max_output > config['max_model_len']:
                raise ValueError('Full context exceeds model limit; no automatic history truncation')
            extent = (limit + config['gen_budget_tokens'] + 8 if mean_k else len(prompt) + max_output + 4)
            ticket = budget.enter(user, geometry.active_bytes(extent), queue_deadline - time.monotonic())
            await _wait_for_request(request, ticket.ready, turn_deadline, queue_deadline)
            if session.version and config.get('max_restore_inflight'):
                remaining = queue_deadline - time.monotonic()
                restore_extent = limit + config['gen_budget_tokens'] + 8 if mean_k else len(prompt) + 4
                restore_ticket = restores.enter(user, geometry.active_bytes(restore_extent), remaining)
                await _wait_for_request(request, restore_ticket.ready, turn_deadline, queue_deadline)
            admitted = time.monotonic()
            payload = {'model': config['served_model_name'], 'prompt': prompt,
                'max_tokens': max_output, 'temperature': 0, 'seed': 104729,
                'return_token_ids': True, 'cache_salt': histories.cache_salt(user, session_id),
                'ignore_eos': body.get('ignore_eos', False)}
            if mean_k:
                payload['session_id'] = session_id
                if config.get('request_planning'):
                    payload['kv_transfer_params'] = {'kvmem_request_plan': fields}
            def restored():
                if restore_ticket is not None:
                    restores.release(restore_ticket)
            if config.get('max_restore_inflight'):
                task = asyncio.create_task(generate(app.state.client, payload, restored))
            else:
                task = asyncio.create_task(app.state.client.post('/v1/completions', json=payload))
            response = await _wait_for_request(request, task, turn_deadline)
            if isinstance(response, dict):
                result = response
            else:
                response.raise_for_status()
                result = response.json()
            choices = result.get('choices', [])
            if len(choices) != 1:
                raise ValueError('Native response must contain one choice')
            output = choices[0].get('token_ids')
            if (not isinstance(output, list) or len(output) != result['usage']['completion_tokens']
                    or result['usage']['prompt_tokens'] != len(prompt)):
                raise ValueError('Native accepted token IDs do not match usage')
            context = {'mode': 'selected_turn_context' if ranges is not None else 'full_context',
                'full_prompt_tokens': full_prompt_tokens, 'active_prompt_tokens': len(prompt), 'ranges': ranges}
            if mean_k:
                context.update(mode='meank_historical_kv', selection='original_mean_k',
                    active_prompt_tokens=None, select_budget=limit, gen_budget=config['gen_budget_tokens'])
            histories.finish(key, append, output, waiting_for_tool=body.get('waiting_for_tool', False))
            counts['completed'] += 1
            return {'session_id': session_id, 'version': session.version,
                'history_tokens': len(session.tokens), 'waiting_for_tool': session.waiting_for_tool,
                'input_context': context,
                'queue_seconds': admitted - started, 'turn_seconds': time.monotonic() - started,
                'native_response': result}
        except BaseException as error:
            if task is not None:
                if not task.done():
                    task.cancel()
                try:
                    await task
                except BaseException:
                    pass
            histories.abort(key)
            counts['aborted'] += 1
            if isinstance(error, _TurnTimeout):
                counts['rejected'] += 1
                raise HTTPException(504, str(error)) from None
            if isinstance(error, (TimeoutError, BufferError)):
                counts['rejected'] += 1
                raise HTTPException(429, str(error)) from None
            if isinstance(error, ValueError):
                raise HTTPException(400, str(error)) from None
            if isinstance(error, httpx.HTTPError):
                raise HTTPException(502, 'Native engine request failed; history unchanged') from None
            raise
        finally:
            if restore_ticket is not None:
                restores.release(restore_ticket)
            if ticket is not None:
                budget.release(ticket)

    return app
