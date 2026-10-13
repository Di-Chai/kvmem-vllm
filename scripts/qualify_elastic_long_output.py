"""One isolated, authenticated C8 batch through the real elastic gateway.

Tap the gateway's native SSE without changing its parsing or generation policy.
The public gateway still returns JSON at completion; TTFT here is internal.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import uuid

import httpx
import uvicorn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import qualify_decode as q
from qualify_multi_user import distribution, load_users
from vllm_kvmem.elastic import HistoryStore
import vllm_kvmem.elastic_gateway as gateway


def metrics(raw):
    out = {}
    for line in raw.splitlines():
        match = re.match(r'^(vllm:[\w:]+)(?:\{.*\})?\s+([-+\w.eE]+)(?:\s+\d+)?$', line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                out[match[1]] = out.get(match[1], 0) + value
    return out


def corroborate(samples, window, concurrency):
    if window.get('status') != 'passed':
        return {'status': 'not_applicable'}
    selected = [s for s in samples if window['start'] + .002 < s['read_started_at'] <= s['at'] < window['end'] - .002]
    seconds, intervals, violations = 0, 0, []
    for left, right in zip(selected, selected[1:]):
        a, b = left.get('metrics', {}), right.get('metrics', {})
        good = (a.get('vllm:num_requests_running') == concurrency == b.get('vllm:num_requests_running')
            and a.get('vllm:num_requests_waiting') == b.get('vllm:num_requests_waiting') == 0
            and 'vllm:prompt_tokens_total' in a
            and a['vllm:prompt_tokens_total'] == b.get('vllm:prompt_tokens_total')
            and a.get('vllm:num_preemptions_total') == b.get('vllm:num_preemptions_total'))
        if good:
            seconds += right['at'] - left['at']
            intervals += 1
        else:
            violations.append({'at': right['at'], 'running': b.get('vllm:num_requests_running'),
                'waiting': b.get('vllm:num_requests_waiting')})
    coverage = seconds / window['seconds']
    return {'status': 'passed' if intervals >= 3 and coverage >= .9 and not violations else 'insufficient_evidence',
        'coverage_fraction': coverage, 'intervals': intervals, 'violations': violations,
        'scope': 'Engine-wide gauges with roughly 1s publication; isolated batch, no ownership-ID snapshots.'}


def band_window(traces, low, high, input_tokens):
    starts, ends = [], []
    for trace in traces:
        points = trace['timeline']
        starts.append(next(p['at'] for p in points if p['tokens'] >= max(1, low)))
        ends.append(max(p['at'] for p in points if p['tokens'] <= high))
    start, end = max(starts), min(ends)
    if end <= start:
        return {'status': 'not_applicable', 'generated_token_band': [low, high]}
    per_request = []
    for trace in traces:
        before, after = q.count_at(trace, start), q.count_at(trace, end)
        per_request.append({'client_index': trace['client_index'], 'tokens': after - before,
            'context_start_tokens': input_tokens + before, 'context_end_tokens': input_tokens + after})
    tokens = sum(p['tokens'] for p in per_request)
    return {'status': 'passed', 'generated_token_band': [low, high], 'start': start, 'end': end,
        'seconds': end - start, 'active_streams': len(traces), 'output_tokens': tokens,
        'output_tokens_per_second': tokens / (end - start), 'per_request': per_request}


class TappedResponse:
    def __init__(self, response, trace, origin):
        self.response, self.trace, self.origin = response, trace, origin

    def __getattr__(self, name):
        return getattr(self.response, name)

    async def aiter_lines(self):
        data = []
        async for line in self.response.aiter_lines():
            if line.startswith('data:'):
                data.append(line[5:].removeprefix(' '))
            elif not line and data:
                raw, data = '\n'.join(data), []
                if raw != '[DONE]':
                    q.consume_event(self.trace, json.loads(raw), time.monotonic() - self.origin)
            yield line


class TappedClient:
    def __init__(self, client, trace, origin):
        self.client, self.trace, self.origin = client, trace, origin

    @asynccontextmanager
    async def stream(self, *args, **kwargs):
        async with self.client.stream(*args, **kwargs) as response:
            yield TappedResponse(response, self.trace, self.origin)


async def run(args):
    launch = json.loads(args.launch.read_text())
    cfg = {**launch['config'], 'port': args.port, 'max_output_tokens': args.output_tokens,
        'max_inflight': args.concurrency}
    prompt = json.loads(args.input.read_text())['input_ids']
    if not prompt or any(type(t) is not int or t < 0 for t in prompt):
        raise ValueError('Frozen input must contain unsigned token IDs')
    users = load_users(Path(cfg['users_file']))[:args.concurrency]
    assert len(users) == args.concurrency
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'gateway-config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    origin = time.monotonic()
    prefix = uuid.uuid4().hex
    traces, by_salt, samples = [], {}, []
    for i, user in enumerate(users):
        session = f'long-output-{prefix}-{i}'
        trace = {'client_index': i, 'session_id': session, 'outcome': 'failure', 'timeline': [],
            'observed_tokens': 0, 'saw_token_ids': False, '_token_digest': hashlib.sha256()}
        traces.append(trace)
        by_salt[HistoryStore.cache_salt(user['id'], session)] = trace
    original_generate = gateway.generate

    async def tapped_generate(client, payload, first_token):
        trace = by_salt[payload['cache_salt']]
        trace['native_started_at'] = time.monotonic() - origin
        result = await original_generate(TappedClient(client, trace, origin), payload, first_token)
        q.finish_trace(trace, args.output_tokens)
        ids = result['choices'][0]['token_ids']
        trace['distinct_output_token_ids'] = len(set(ids))
        trace['token_ids_sha256'] = trace.pop('_token_digest').hexdigest()
        return result

    gateway.generate = tapped_generate
    server = uvicorn.Server(uvicorn.Config(gateway.create_app(cfg), host='127.0.0.1', port=args.port,
        log_level='warning', access_log=False))
    server_task = asyncio.create_task(server.serve())
    done = asyncio.Event()
    poller = None
    base = f'http://127.0.0.1:{args.port}'
    async with httpx.AsyncClient(timeout=cfg['turn_timeout_seconds'] + 30, trust_env=False,
            limits=httpx.Limits(max_connections=args.concurrency + 8)) as client:
        try:
            while not server.started:
                if server_task.done():
                    await server_task
                    raise RuntimeError('Temporary gateway failed to start')
                await asyncio.sleep(.05)
            response = await client.get(cfg['backend_url'] + '/metrics')
            response.raise_for_status()
            before = metrics(response.text)
            if before.get('vllm:num_requests_running') or before.get('vllm:num_requests_waiting'):
                raise RuntimeError('Backend has other active requests')
            start_together = asyncio.Event()

            async def request(i):
                trace = traces[i]
                await start_together.wait()
                trace['started_at'] = time.monotonic() - origin
                try:
                    response = await client.post(base + '/v1/kvmem/sessions/' + trace['session_id'] + '/turn',
                        headers={'Authorization': 'Bearer ' + users[i]['api_key']}, json={
                            'version': 0, 'append_token_ids': prompt, 'max_tokens': args.output_tokens,
                            'waiting_for_tool': True, 'ignore_eos': True})
                    trace['http_status'] = response.status_code
                    response.raise_for_status()
                    result = response.json()
                    native = result['native_response']
                    assert native['usage']['prompt_tokens'] == len(prompt)
                    assert native['usage']['completion_tokens'] == args.output_tokens
                    assert len(native['choices'][0]['token_ids']) == args.output_tokens
                    assert result['history_tokens'] == len(prompt) + args.output_tokens
                    expected_mode = 'meank_historical_kv' if cfg.get('context_mode') == 'meank' else 'full_context'
                    assert result['input_context']['mode'] == expected_mode
                    trace.update(outcome='success', history_tokens=result['history_tokens'],
                        queue_seconds=result['queue_seconds'], turn_seconds=result['turn_seconds'],
                        gateway_internal_ttft_ms=native['metrics']['time_to_first_token_ms'])
                except Exception as error:
                    trace['error'] = type(error).__name__
                finally:
                    trace['ended_at'] = time.monotonic() - origin
                    trace['elapsed_seconds'] = trace['ended_at'] - trace['started_at']

            async def poll():
                last_print = -20
                while not done.is_set():
                    read_started = time.monotonic() - origin
                    sample = {'read_started_at': read_started}
                    try:
                        response = await client.get(cfg['backend_url'] + '/metrics')
                        response.raise_for_status()
                        sample['metrics'] = metrics(response.text)
                        response = await client.get(base + '/v1/kvmem/status',
                            headers={'Authorization': 'Bearer ' + users[0]['api_key']})
                        response.raise_for_status()
                        sample['gateway'] = {k: v for k, v in response.json().items() if k != 'sessions'}
                    except Exception as error:
                        sample['error'] = type(error).__name__
                    sample['at'] = time.monotonic() - origin
                    samples.append(sample)
                    if sample['at'] - last_print >= 20:
                        progress = {'at': round(sample['at'], 1), 'output_tokens': [t['observed_tokens'] for t in traces],
                            'running': sample.get('metrics', {}).get('vllm:num_requests_running'),
                            'preemptions': sample.get('metrics', {}).get('vllm:num_preemptions_total', 0) - before.get('vllm:num_preemptions_total', 0)}
                        (args.output / 'progress.json').write_text(json.dumps(progress) + '\n')
                        print(json.dumps(progress), flush=True)
                        last_print = sample['at']
                    try:
                        await asyncio.wait_for(done.wait(), .25)
                    except TimeoutError:
                        pass

            started = time.monotonic()
            poller = asyncio.create_task(poll())
            tasks = [asyncio.create_task(request(i)) for i in range(args.concurrency)]
            start_together.set()
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                done.set()
                await poller
            wall = time.monotonic() - started
            await asyncio.sleep(1.3)
            response = await client.get(cfg['backend_url'] + '/metrics')
            response.raise_for_status()
            final = metrics(response.text)
            response = await client.get(base + '/v1/kvmem/status',
                headers={'Authorization': 'Bearer ' + users[0]['api_key']})
            final_gateway = response.json()
            windows = q.batch_windows(SimpleNamespace(trim_tokens=128, min_window_seconds=2,
                min_events_per_request=4, allow_queued=False, capture=None), traces, [], args.concurrency, wall)
            steady = windows['steady_decode_window']
            confirmation = corroborate(samples, steady, args.concurrency)
            bands = []
            if all(t['outcome'] == 'success' for t in traces):
                for low in range(0, args.output_tokens, 4096):
                    window = band_window(traces, low, min(low + 4096, args.output_tokens), len(prompt))
                    window['native_confirmation'] = corroborate(samples, window, args.concurrency)
                    bands.append(window)
            delta = {k: final[k] - v for k, v in before.items() if k.endswith('_total') and k in final}
            passed = (windows['status'] == 'passed' and confirmation['status'] == 'passed'
                and delta.get('vllm:num_preemptions_total') == 0
                and final.get('vllm:num_requests_running') == final.get('vllm:num_requests_waiting') == 0
                and final_gateway['active_requests'] == final_gateway['active_reserved_bytes'] == final_gateway['queued_requests'] == 0)
            report = {'status': 'passed' if passed else 'failed', 'configuration': {
                'concurrency': args.concurrency, 'distinct_authenticated_users': len(users),
                'input_tokens_per_user': len(prompt), 'output_tokens_per_user': args.output_tokens,
                'context_end_tokens_per_user': len(prompt) + args.output_tokens,
                'mtp_k': launch['config']['mtp_k'], 'kv_cache_dtype': 'fp8_e4m3',
                'cpu_kv_bytes': cfg['cpu_kv_bytes'], 'active_bytes': cfg['active_bytes'],
                'temporary_gateway_port': args.port, 'context_mode': cfg.get('context_mode', 'full'),
                'draft_method': cfg.get('draft_method', 'mtp'),
                'draft_tokens': cfg.get('draft_tokens') if cfg.get('context_mode') == 'meank' else None,
                'draft_quantization': cfg.get('draft_quantization'),
                'unique_session_salts': True, 'ignore_eos': True,
                'input_token_ids_sha256': hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest(),
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                **windows, 'status': 'passed' if passed else 'failed', 'native_confirmation': confirmation,
                'context_growth_windows': bands, 'requests': traces, 'samples': samples,
                'native_counter_deltas': delta, 'final_gateway': final_gateway,
                'peak_gateway_active_requests': max(s.get('gateway', {}).get('active_requests', 0) for s in samples),
                'peak_gateway_active_bytes': max(s.get('gateway', {}).get('active_reserved_bytes', 0) for s in samples),
                'peak_native_kv_usage': max(s.get('metrics', {}).get('vllm:kv_cache_usage_perc', 0) for s in samples),
                'metric_notes': ['Gateway receives final JSON; TTFT is its internal native-SSE first accepted token.',
                    'Fixed-output synthetic capacity test; no task-quality or mixed-workload SLO claim.',
                    ('Original MeanK selects historical KV within K and retains generation within G; all token IDs are preserved.'
                     if cfg.get('context_mode') == 'meank' else 'No context truncation; complete KV remains active.'),
                    'Unique salts exclude cross-user prefix sharing.',
                    'All stalls inside common decode windows remain in the denominator.']}
            public = q.public_evidence(report)
            (args.output / 'results.json').write_text(json.dumps(public, indent=2) + '\n')
            summary = {k: public[k] for k in ('status', 'configuration', 'end_to_end', 'steady_decode_window',
                'native_confirmation', 'context_growth_windows', 'peak_gateway_active_requests',
                'peak_gateway_active_bytes', 'peak_native_kv_usage')}
            (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps(summary), flush=True)
            return 0 if passed else 1
        finally:
            done.set()
            if poller is not None:
                await poller
            server.should_exit = True
            await server_task
            gateway.generate = original_generate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=16384)
    parser.add_argument('--port', type=int, default=18384)
    raise SystemExit(asyncio.run(run(parser.parse_args())))
