"""Finite active-KV qualification with raw SSE timing and explicit session cleanup.

Requires an owned KVMem server. This client does not start/stop servers and does
not retry inference. Streaming arrival times include client/network buffering.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import statistics
import time
from uuid import uuid4

import httpx
from qualify_systemone import origin, validate_answer, capture_snapshot, release_evidence


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return dict(count=len(values), mean=statistics.mean(values),
                median=statistics.median(values), p95=values[min(len(values)-1, int(.95*len(values)))],
                maximum=max(values))


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', arms=[], negative_checks=[], started_at=time.time(),
                  order=args.order, expected_rollback=args.expect_rollback)
    dump = lambda: save(args.output/'results.json', report)
    dump()
    previous = capture_snapshot(args.capture)
    save(args.output/'resources-before.json', previous)
    try:
        async with httpx.AsyncClient(base_url=origin(args.base_url), timeout=240, trust_env=False) as client:
            model_response = await client.get('/v1/models')
            model_response.raise_for_status()
            model = model_response.json()['data'][0]['id']
            text = ('Record: the parcel was damaged during shipping.\n' * args.context_repetitions +
                    'The parcel arrived broken. I want a refund. '
                    'Write a detailed numbered list of practical steps for handling this return. '
                    'Continue adding useful steps.')
            tokenize = {'model': model, 'messages': [{'role': 'user', 'content': text}],
                        'add_generation_prompt': True, 'chat_template_kwargs': {'enable_thinking': False}}
            save(args.output/'tokenize-request.json', tokenize)
            response = await client.post('/tokenize', json=tokenize)
            response.raise_for_status()
            save(args.output/'tokenize-response.json', response.json())
            tokens = response.json()['tokens']
            report['prompt_tokens'] = len(tokens)
            for arm_index, mode in enumerate(args.order.split(',')):
                active = mode == 'active'
                session = 'active-4090-' + uuid4().hex
                arm = dict(active=active, session=session, token_ids=[], events=[], decisions=[])
                report['arms'].append(arm)
                arm_start = time.perf_counter()
                def elapsed():
                    return time.perf_counter()-arm_start
                question = {'type': 'noul', 'instructions': 'Was the parcel damaged?'}
                payload = {'model': model, 'state': '', 'cache_salt': session,
                           'kvmem_session': {'session_id': session}, 'questions': {
                               'first': question, 'middle': {'type': 'choice',
                                   'instructions': 'What does the customer want? Return only the option letter.',
                                   'criteria': {'refund': 'A refund', 'replacement': 'A replacement'}},
                               'repeat': question}}
                request = {'model': model, 'prompt': tokens, 'session_id': session, 'cache_salt': session,
                           'temperature': 0, 'max_tokens': 512, 'ignore_eos': True, 'seed': 42,
                           'return_token_ids': True, 'stream': True}
                save(args.output/f'arm-{arm_index}-parent-request.json', request)
                async def consume():
                    with (args.output/f'arm-{arm_index}-sse.jsonl').open('x') as stream:
                        async with client.stream('POST', '/v1/completions', json=request) as response:
                            response.raise_for_status()
                            async for line in response.aiter_lines():
                                if not line.startswith('data: '):
                                    continue
                                at = elapsed()
                                stream.write(json.dumps({'seconds': at, 'line': line})+'\n')
                                stream.flush()
                                if line[6:] == '[DONE]':
                                    arm['done'] = True
                                    break
                                chunk = json.loads(line[6:])
                                assert 'error' not in chunk, chunk
                                arm['request_id'] = chunk.get('id', arm.get('request_id'))
                                ids = [token for choice in chunk.get('choices', [])
                                       for token in (choice.get('token_ids') or [])]
                                if ids:
                                    arm['token_ids'].extend(ids)
                                    arm['events'].append({'seconds': at, 'count': len(ids),
                                                          'total': len(arm['token_ids'])})
                        arm['seconds'] = elapsed()
                        assert len(arm['token_ids']) == 512, len(arm['token_ids'])
                task = asyncio.create_task(consume())
                async def wait_tokens(count):
                    async with asyncio.timeout(180):
                        while len(arm['token_ids']) < count:
                            if task.done():
                                await task
                                raise RuntimeError('Parent ended before decision boundary')
                            await asyncio.sleep(.01)
                async def decision(name, data):
                    save(args.output/f'arm-{arm_index}-{name}-request.json', data)
                    before = elapsed()
                    result = await client.post('/v1/systemone', json=data)
                    item = dict(name=name, status=result.status_code, start_seconds=before,
                                end_seconds=elapsed(), body=result.json())
                    item['seconds'] = item['end_seconds']-before
                    save(args.output/f'arm-{arm_index}-{name}-response.json', item)
                    return item
                try:
                    if active:
                        await wait_tokens(32)
                        if args.negatives:
                            for name, mutate, expected in [
                                ('wrong-salt', lambda p: p.update(cache_salt='unrelated-'+session), 409),
                                ('stale-generation', lambda p: p['kvmem_session'].update(generation=2147483647), 409),
                                ('nonempty-state', lambda p: p.update(state='not allowed'), 400),
                            ]:
                                bad = copy.deepcopy(payload)
                                mutate(bad)
                                item = await decision(name, bad)
                                report['negative_checks'].append(item)
                                dump()
                                assert item['status'] == expected, item
                        for index, count in enumerate((32, 160)):
                            await wait_tokens(count)
                            assert not task.done(), 'Parent already finished'
                            item = await decision(f'decision-{index}', payload)
                            arm['decisions'].append(item)
                            dump()
                            assert item['status'] == 200, item
                            body = item['body']
                            validate_answer(body, payload)
                            kv = body['kvmem']
                            assert kv['mode'] == 'active_branch' and kv['session_id'] == session
                            assert kv['history_end'] >= len(tokens)
                            assert kv['rollback_verified'] == args.expect_rollback
                            assert body['usage']['input_tokens'] == kv['question_tokens']
                            assert body['answers']['first'] == body['answers']['repeat']
                            assert body['diagnostics']['first'] == body['diagnostics']['repeat']
                            if args.context_repetitions:
                                assert kv['active_tokens'] < kv['history_end'], kv
                    await task
                    if active and args.negatives:
                        item = await decision('completed-session', payload)
                        report['negative_checks'].append(item)
                        assert item['status'] == 409, item
                finally:
                    # Always finish the bounded parent and release only our own namespace.
                    try:
                        await task
                    finally:
                        released_at = time.monotonic()
                        release = await client.post('/v1/completions', json={
                            'model': model, 'prompt': [tokens[0]], 'session_id': session,
                            'cache_salt': session, 'max_tokens': 1, 'temperature': 0,
                            'kv_transfer_params': {'kvmem_release_session': True}})
                        arm['release_status'] = release.status_code
                        dump()
                        release.raise_for_status()
                times = [e['seconds'] for e in arm['events']]
                arm['ttft_seconds'] = times[0]
                arm['chunk_gap_seconds'] = distribution([b-a for a, b in zip(times, times[1:])])
                for item in arm['decisions']:
                    overlaps = [dict(start=a, end=b, seconds=b-a) for a, b in zip(times, times[1:])
                                if a <= item['end_seconds'] and b >= item['start_seconds']]
                    item['overlapping_chunk_gaps'] = overlaps
                dump()
                # Capture one matching retirement for the parent; private question
                # tokens must not be published as additional history.
                deadline = time.monotonic()+20
                while True:
                    snap = capture_snapshot(args.capture)
                    proof = release_evidence(snap, previous, 0, released_at)
                    if proof['passed'] or time.monotonic() >= deadline:
                        break
                    await asyncio.sleep(.2)
                save(args.output/f'arm-{arm_index}-resources-after.json', snap)
                arm['release_proof'] = proof
                assert proof['passed'], proof
                completions = snap['archive']['completed']
                parent = [row for row in completions if row.get('request_id', '').startswith(arm['request_id'])
                          and row.get('terminal_prefix', {}).get('output_tokens') == 512]
                assert len(parent) == 1, (arm.get('request_id'), parent)
                arm['parent_retirement'] = parent[0]
                assert parent[0]['history_end'] == len(tokens)+511, parent[0]
                dump()
            controls = [a for a in report['arms'] if not a['active']]
            assert controls and any(a['active'] for a in report['arms'])
            report['parent_token_equal'] = all(a['token_ids'] == controls[0]['token_ids'] for a in report['arms'])
            assert report['parent_token_equal'], 'Deterministic foreground tokens changed'
            report['status'] = 'passed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        report['finished_at'] = time.time()
        dump()
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:18994')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--capture', required=True, type=Path)
    parser.add_argument('--context-repetitions', type=int, default=0)
    parser.add_argument('--order', default='control,active')
    parser.add_argument('--expect-rollback', action='store_true')
    parser.add_argument('--negatives', action='store_true')
    args = parser.parse_args()
    assert set(args.order.split(',')) == {'control', 'active'}
    result = asyncio.run(run(args))
    print(json.dumps({k: v for k, v in result.items() if k not in ('arms', 'negative_checks')}))
