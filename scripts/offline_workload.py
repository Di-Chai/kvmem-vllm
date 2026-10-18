"""Finite offline inference batches with bounded, continuously replenished ingress.

Independent samples have no retained session. A unique cache salt prevents
cross-sample prefix reuse from manufacturing throughput. KVMem still retains
each live request's complete CPU history until its normal retirement fence.
"""
import asyncio
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def prepare(args, fixture, profile, output):
    if profile['kvmem'].get('adaptive_prefill'):
        raise ValueError('Offline throughput mode requires adaptive_prefill=false; use B0/B3')
    if args.offline_concurrency < profile['max_num_seqs']:
        raise ValueError('Offline ingress must cover the native active request limit')
    if (args.offline_requests < 1 or args.offline_concurrency < 1 or
            not math.isfinite(args.offline_warmup_seconds) or args.offline_warmup_seconds < 0 or
            not math.isfinite(args.offline_sample_seconds) or args.offline_sample_seconds < .5):
        raise ValueError('Invalid offline workload or telemetry limits')
    if args.offline_dataset:
        raw = [json.loads(line) for line in Path(args.offline_dataset).read_text().splitlines() if line.strip()]
    else:
        if not args.offline_input_lengths or any(n < 2 or n > len(fixture['prompt'])
                                                for n in args.offline_input_lengths):
            raise ValueError('Offline synthetic input lengths exceed the prepared fixture')
        raw = [dict(id=str(i), prompt_token_ids=fixture['prompt'][:
                    args.offline_input_lengths[i % len(args.offline_input_lengths)]],
                    max_tokens=args.offline_output_tokens, ignore_eos=True)
               for i in range(args.offline_requests)]
    tokenizer = None
    rows, seen = [], set()
    for index, sample in enumerate(raw):
        if not isinstance(sample,dict):
            raise ValueError('Each offline JSONL row must be an object')
        identity = str(sample.get('id', index))
        if identity in seen:
            raise ValueError('Offline dataset IDs must be unique')
        seen.add(identity)
        tokens = sample.get('prompt_token_ids')
        if tokens is None and isinstance(sample.get('prompt'), str):
            if tokenizer is None:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
            tokens = tokenizer.encode(sample['prompt'], add_special_tokens=False)
        count = sample.get('max_tokens', args.offline_output_tokens)
        if (not isinstance(tokens, list) or not tokens or
                any(type(x) is not int or x < 0 for x in tokens) or
                type(count) is not int or not 0 < count <= profile['kvmem']['gen_budget'] or
                len(tokens) + count > profile['max_model_len']):
            raise ValueError(f'Invalid input/output extent for offline sample {identity}')
        query = sample.get('query_span', [max(0, len(tokens)-128), len(tokens)-1] if len(tokens)>1 else None)
        if query is not None and (not isinstance(query, list) or len(query)!=2 or
                any(type(v) is not int for v in query) or not 0 <= query[0] < query[1] <= len(tokens)):
            raise ValueError(f'Invalid half-open Query span for offline sample {identity}')
        ignore_eos = sample.get('ignore_eos', False)
        if type(ignore_eos) is not bool:
            raise ValueError('ignore_eos must be boolean')
        rows.append(dict(index=index, id=identity, prompt_token_ids=tokens, max_tokens=count,
                         ignore_eos=ignore_eos, query_span=query))
    if not rows:
        raise ValueError('Offline dataset is empty')
    path = output/'offline-dataset.jsonl'
    path.write_text(''.join(json.dumps(row, separators=(',', ':'))+'\n' for row in rows))
    return dict(samples=len(rows), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input_tokens=sum(len(r['prompt_token_ids']) for r in rows),
                maximum_output_tokens=sum(r['max_tokens'] for r in rows),
                ingress_concurrency=args.offline_concurrency,
                scope='Independent offline samples; no session retention or cross-sample prefix reuse')


async def infer(client, profile, job, *, clock=time.perf_counter):
    row = dict(index=job['index'], id=job['id'], begin=clock(), success=False,
               input_tokens=len(job['prompt_token_ids']), input_sha256=digest(job['prompt_token_ids']),
               output_ids=[], output_text='', timeline=[], response_id=None)
    payload = dict(model=profile['served_model_name'], prompt=job['prompt_token_ids'],
        cache_salt='offline-'+digest(job['id']), max_tokens=job['max_tokens'],
        temperature=0, seed=1000, ignore_eos=job['ignore_eos'], return_token_ids=True,
        stream=True, stream_options={'include_usage': True}, kvmem_reselect='auto')
    if job['query_span'] is not None:
        payload['kvmem_query_span'] = job['query_span']
    usage, completed = None, False
    try:
        async with client.stream('POST', f"http://127.0.0.1:{profile['port']}/v1/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith('data: '):
                    continue
                if line[6:] == '[DONE]':
                    completed = True
                    break
                event = json.loads(line[6:])
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                row['response_id'] = event.get('id', row['response_id'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    row['output_text'] += choice.get('text') or ''
                    tokens = choice.get('token_ids') or []
                    if tokens:
                        row['output_ids'].extend(tokens)
                        row['timeline'].append([clock(), len(tokens)])
        count = len(row['output_ids'])
        if (not completed or not usage or usage['completion_tokens'] != count or
                not 0 < count <= job['max_tokens'] or (job['ignore_eos'] and count != job['max_tokens'])):
            raise RuntimeError('Incomplete or inconsistent offline output stream')
        row.update(success=True, usage=usage)
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    row.update(end=clock(), output_sha256=digest(row['output_ids']))
    return row


async def dispatch(jobs, concurrency, invoke, *, clock=time.perf_counter, on_result=None, state=None):
    """Work-conserving client pool: no batch barrier, think time or retry."""
    state = state if state is not None else {}
    state.update(started=clock(), submitted=0, completed=0, in_flight=0,
                 pending=len(jobs), peak_in_flight=0, drain_started=None)
    cursor, rows = 0, []

    async def worker():
        nonlocal cursor
        while cursor < len(jobs):
            job = jobs[cursor]
            cursor += 1  # No await between claiming a sample and updating state.
            state['submitted'] += 1; state['pending'] -= 1; state['in_flight'] += 1
            state['peak_in_flight'] = max(state['peak_in_flight'], state['in_flight'])
            if cursor == len(jobs):
                state['drain_started'] = clock()
            row = await invoke(job)
            state['in_flight'] -= 1; state['completed'] += 1
            rows.append(row)
            if on_result is not None:
                on_result(row)
    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(jobs))):
            group.create_task(worker())
    state['finished'] = clock()
    return sorted(rows, key=lambda row:row['index'])


def summarize(rows, state, warmup_seconds, percentile):
    good = [row for row in rows if row['success']]
    wall = state['finished']-state['started']
    gaps = [b[0]-a[0] for row in good for a,b in zip(row['timeline'],row['timeline'][1:])]
    start, end = state['started']+warmup_seconds, state['drain_started']
    steady = None
    if end is not None and end > start:
        count = sum(n for row in good for t,n in row['timeline'] if start <= t < end)
        steady = dict(begin=start, end=end, seconds=end-start, output_tokens=count,
                      output_tokens_per_second=count/(end-start),
                      scope='Fixed warmup excluded; stops when the final sample is dispatched')
    return dict(requests=len(rows), successful_requests=len(good), failed_requests=len(rows)-len(good),
        wall_seconds=wall, output_tokens=sum(len(r['output_ids']) for r in good),
        output_tokens_per_second=sum(len(r['output_ids']) for r in good)/wall,
        completed_requests_per_second=len(good)/wall,
        logical_input_tokens_per_second=sum(r['input_tokens'] for r in good)/wall,
        output_event_gap_p95_seconds=percentile(gaps,.95), common_decode=None,
        request_seconds_p95=percentile([r['end']-r['begin'] for r in good],.95),
        ttft_p95_seconds=percentile([r['timeline'][0][0]-r['begin'] for r in good],.95),
        steady_backlog_window=steady,
        peak_in_flight=state['peak_in_flight'],
        drain_seconds=state['finished']-end if end is not None else None,
        metric_scope='Whole finite batch includes ramp and drain; failed partial outputs excluded, failures retained; logical input rate is not uncached GPU prefill speed')


def metric_values(text):
    names = ('num_requests_running', 'num_requests_waiting', 'kv_cache_usage_perc',
             'num_preemptions_total', 'spec_decode_num_drafts_total', 'spec_decode_num_accepted_tokens_total')
    result = {}
    for line in text.splitlines():
        for name in names:
            if line.startswith('vllm:'+name+'{') or line.startswith('vllm:'+name+' '):
                result[name] = result.get(name,0.) + float(line.split()[-1])
    return result


def model_occupancy(path, window, active_limit):
    """Step samples describe computed batches, unlike the running-request gauge.

    Sample means are not time-weighted. Event durations include host launch
    gaps, and a missing sample is not evidence of GPU idle time.
    """
    rows = []
    if path.exists() and window:
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A final timing record can still be publishing.
            if window['begin'] <= row['started_at'] < window['end']:
                rows.append(row)
    occupied = [r['decode_requests'] + r['prefill_requests'] for r in rows]
    return dict(samples=len(rows),
        requests_per_step_mean=statistics.mean(occupied) if rows else None,
        full_batch_sample_fraction=sum(n >= active_limit for n in occupied)/len(rows) if rows else None,
        decode_requests_per_step_mean=statistics.mean(r['decode_requests'] for r in rows) if rows else None,
        prefill_requests_per_step_mean=statistics.mean(r['prefill_requests'] for r in rows) if rows else None,
        maintenance_ms_mean=statistics.mean(r['maintenance_ms'] for r in rows) if rows else None,
        model_ms_mean=statistics.mean(r['model_ms'] for r in rows) if rows else None,
        scope='Completed model step samples in the fixed backlog window; sample means, not time-weighted or pure kernel timings')


async def workload(client, profile, directory, args, helpers):
    directory.mkdir()
    dataset = directory.parent.parent/'offline-dataset.jsonl'
    jobs = [json.loads(line) for line in dataset.read_text().splitlines()]
    state, telemetry = {}, []
    base = f"http://127.0.0.1:{profile['port']}"
    before = (await client.get(base+'/metrics')).text

    async def observe():
        while True:
            sample = dict(at=time.perf_counter(), client=dict(state))
            try:
                response = await client.get(base+'/metrics', timeout=5)
                response.raise_for_status()
                sample['engine'] = metric_values(response.text)
                path = directory.parent/'server/capture/scheduler_resources.json'
                if path.exists():
                    snapshot = json.loads(path.read_text())
                    sample['scheduler'] = {k:v for k,v in snapshot.items()
                        if k in ('at','requests','refresh_pending','connector_pending','authority_limit',
                                 'authority_rejections','archive_capacity_deferrals','cpu_history_admission')}
            except Exception as error:
                sample['telemetry_error'] = f'{type(error).__name__}: {error}'
            telemetry.append(sample)
            with (directory/'load-samples.jsonl').open('a') as stream:
                stream.write(json.dumps(sample)+'\n')
            helpers.save(directory/'progress.json', state)
            await asyncio.sleep(args.offline_sample_seconds)

    with (directory/'responses.jsonl').open('w') as stream:
        def completed(row):
            stream.write(json.dumps(row)+'\n'); stream.flush()
        observer = asyncio.create_task(observe())
        try:
            rows = await dispatch(jobs, args.offline_concurrency,
                lambda job:infer(client,profile,job), on_result=completed, state=state)
        finally:
            observer.cancel()
            await asyncio.gather(observer, return_exceptions=True)
            helpers.save(directory/'progress.json', state)
    after = (await client.get(base+'/metrics')).text
    (directory/'metrics-before.txt').write_text(before)
    (directory/'metrics-after.txt').write_text(after)
    helpers.save(directory/'traces.json', rows)
    result = summarize(rows,state,args.offline_warmup_seconds,helpers.percentile)
    window = result['steady_backlog_window']
    chosen = [s for s in telemetry if window and window['begin'] <= s['at'] < window['end']]
    running = [s['engine']['num_requests_running'] for s in chosen
               if 'num_requests_running' in s.get('engine',{})]
    waiting = [s['engine']['num_requests_waiting'] for s in chosen
               if 'num_requests_waiting' in s.get('engine',{})]
    gpu_path = directory.parent/'gpu_samples.jsonl'
    gpu = []
    if gpu_path.exists():
        for line in gpu_path.read_text().splitlines():
            try:
                gpu.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The guard may be publishing its final sample.
    gpu = [s for s in gpu if window and window['begin'] <= s['at'] < window['end']]
    result['load_observations'] = dict(samples=len(chosen), engine_running_mean=statistics.mean(running) if running else None,
        engine_waiting_mean=statistics.mean(waiting) if waiting else None,
        gpu_samples=len(gpu), gpu_utilization_mean=statistics.mean(s['utilization'] for s in gpu) if gpu else None,
        gpu_low_utilization_sample_fraction=sum(s['utilization']<10 for s in gpu)/len(gpu) if gpu else None,
        scope='Sampled during a client-backlogged window; GPU utilization and queued requests do not prove work is GPU-ready')
    result['model_occupancy'] = model_occupancy(
        directory.parent/'server/capture/model_step_latency.jsonl', window, profile['max_num_seqs'])
    helpers.save(directory/'results.json',result)
    helpers.save(directory/'diagnostics.json',helpers.collect_diagnostics(directory.parent/'server',
                 started=state['started'],finished=state['finished']))
    return result
