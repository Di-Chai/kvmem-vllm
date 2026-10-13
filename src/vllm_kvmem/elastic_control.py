"""Loopback backend cache control for verified CPU-restore qualification.

The public turn gateway does not proxy this endpoint. Native reset refuses
live page owners; it never preempts running requests to make a reset succeed.
"""
import hmac
import json
import os
from pathlib import Path


class ElasticControlMiddleware:
    def __init__(self, app):
        self.app = app
        self.key = Path(os.environ['KVMEM_ELASTIC_ADMIN_FILE']).read_text().strip()

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/internal/kvmem/cache/reset':
            return await self.app(scope, receive, send)
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        request = Request(scope, receive)
        auth = request.headers.get('authorization', '')
        if not hmac.compare_digest(auth, 'Bearer ' + self.key):
            return await JSONResponse({'error': 'Unauthorized'}, status_code=401)(scope, receive, send)
        if scope['method'] != 'POST':
            return await JSONResponse({'error': 'POST required'}, status_code=405)(scope, receive, send)
        try:
            body = json.loads(await request.body())
            if (not isinstance(body, dict) or set(body) != {'reset_cpu'}
                    or type(body['reset_cpu']) is not bool):
                raise ValueError('reset_cpu boolean required')
        except (ValueError, UnicodeError):
            return await JSONResponse({'error': 'Invalid reset request'}, status_code=400)(scope, receive, send)
        success = await request.app.state.engine_client.reset_prefix_cache(
            reset_running_requests=False, reset_connector=body['reset_cpu'])
        response = JSONResponse({'reset': bool(success), 'cpu_reset': body['reset_cpu']},
            status_code=200 if success else 409)
        await response(scope, receive, send)
