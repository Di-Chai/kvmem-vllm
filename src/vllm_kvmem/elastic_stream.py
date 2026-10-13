"""Consume exact native token-ID SSE; release restore credit at first output."""
import json
import time


async def generate(client, payload, first_token):
    start = time.monotonic()
    ids, pieces, usage = [], [], None
    finish, response_id, created, first_at = None, None, None, None
    data_lines, event_size, done = [], 0, False
    # return_token_ids also puts the complete prompt IDs in the first event.
    # Reserve a bounded envelope for uint32 IDs (including JSON spacing), not
    # just the small generated delta. Gateway input/model limits bound this.
    prompt = payload.get('prompt', [])
    max_event_chars = 1024**2 + 16 * len(prompt)
    async with client.stream('POST', '/v1/completions', json={**payload,
            'stream': True, 'stream_options': {'include_usage': True}}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            event_size += len(line)
            if event_size > max_event_chars:
                raise ValueError('Native SSE event exceeds bounded parser capacity')
            if line:
                if line.startswith('data:'):
                    data_lines.append(line[5:].lstrip())
                continue
            if not data_lines:
                event_size = 0
                continue
            raw = '\n'.join(data_lines)
            data_lines, event_size = [], 0
            if raw == '[DONE]':
                done = True
                break
            event = json.loads(raw)
            if event.get('error'):
                raise ValueError('Native SSE reported a generation error')
            if response_id not in (None, event.get('id')):
                raise ValueError('Native SSE changed response identity')
            response_id, created = event.get('id'), event.get('created')
            if event.get('usage') is not None:
                usage = event['usage']
            choices = event.get('choices', [])
            if not isinstance(choices, list) or len(choices) > 1:
                raise ValueError('Native SSE returned unexpected choices')
            for choice in choices:
                if choice.get('index') != 0:
                    raise ValueError('Native SSE changed choice identity')
                delta = choice.get('token_ids')
                if delta is not None:
                    if not isinstance(delta, list) or any(type(t) is not int or t < 0 for t in delta):
                        raise ValueError('Native SSE returned invalid accepted token IDs')
                    if delta and first_at is None:
                        first_at = time.monotonic()
                        first_token()
                    ids.extend(delta)
                    if len(ids) > payload['max_tokens']:
                        raise ValueError('Native SSE exceeded output reservation')
                pieces.append(choice.get('text') or '')
                if choice.get('finish_reason') is not None:
                    finish = choice['finish_reason']
    if (not done or usage is None or not ids or first_at is None
            or usage.get('completion_tokens') != len(ids) or finish not in ('stop', 'length')):
        raise ValueError('Native SSE completion/usage evidence is incomplete')
    return {'id': response_id, 'object': 'text_completion', 'created': created,
        'model': payload['model'], 'choices': [{'index': 0, 'text': ''.join(pieces),
            'token_ids': ids, 'finish_reason': finish}], 'usage': usage,
        'metrics': {'time_to_first_token_ms': (first_at - start) * 1000,
            'measurement_source': 'gateway_native_SSE_first_accepted_token'}}
