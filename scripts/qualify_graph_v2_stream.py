"""Continuous original-256K requests with coordinator-owned warmup/window/drain."""
import argparse
import ast
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid

import graph_v2_protocol as protocol
from graph_v2_protocol import (STREAM_SCHEMA as SCHEMA,
    STREAM_WINDOW_SECONDS as WINDOW_SECONDS,
    STREAM_WARMUP_COMPLETIONS as WARMUP_COMPLETIONS, STREAM_SLOTS as SLOTS)
import qualify_block_graph as q
import qualify_graph_niah as original
import qualify_graph_v2 as v2
import qualify_graph_v2_performance as counters

save, require = original.save, protocol.require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Recorder:
    def __init__(self, directory, clock=time.monotonic):
        self.directory, self.clock = directory, clock
        self.lock = threading.Lock()
        self.sequence = 0

    def event(self, kind, **values):
        # Worker callbacks record evidence only. They never change T0/T1.
        with self.lock:
            self.sequence += 1
            row = dict(sequence=self.sequence, at=self.clock(), kind=kind, **values)
            with (self.directory/'events.jsonl').open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            return row


class WindowQueue:
    """The sole coordinator owns credits, admission and immutable window bounds."""
    def __init__(self, rows, credits, capacity, *, slots=SLOTS,
                 warmup=WARMUP_COMPLETIONS, seconds=WINDOW_SECONDS, clock=time.monotonic):
        require(rows and len({r['id'] for r in rows}) == len(rows), 'Unique finite request queue required')
        require(all(type(v) is int and v > 0 for v in (capacity, slots, warmup)) and
                math.isfinite(seconds) and seconds > 0, 'Invalid window queue limits')
        require(len(rows) > warmup+slots and all(0 < credits[r['id']] <= capacity for r in rows),
                'Insufficient candidate queue or invalid full-history credit')
        self.rows, self.credits, self.capacity = rows, credits, capacity
        self.slots, self.warmup, self.seconds, self.clock = slots, warmup, seconds, clock
        self.begin = self.end = None
        self.inflight = self.reserved = self.peak = self.peak_bytes = 0
        self.completed = 0
        self.closed = self.started = False
        self.states = {r['id']:dict(id=r['id'], state='waiting', history_bytes=credits[r['id']]) for r in rows}

    def snapshot(self):
        return dict(measurement_started_at=self.begin, measurement_finished_at=self.end,
            admission_closed=self.closed, inflight=self.inflight, reserved_bytes=self.reserved,
            peak_inflight=self.peak, peak_reserved_bytes=self.peak_bytes, capacity_bytes=self.capacity,
            completed=self.completed, states=list(self.states.values()))

    def run(self, work, release, recorder, *, check=lambda: None, boundary=lambda: None):
        require(not self.started, 'No queue reuse or retry')
        self.started = True
        waiting, running, retiring, results = deque(self.rows), {}, {}, []
        model = ThreadPoolExecutor(max_workers=self.slots, thread_name_prefix='stream-request')
        fences = ThreadPoolExecutor(max_workers=self.slots, thread_name_prefix='stream-release')
        current = None

        def publish():
            save(recorder.directory/'queue.json', self.snapshot())

        def cutoff():
            if self.end is not None and self.clock() >= self.end and not self.closed:
                self.closed = True
                recorder.event('admission_closed', deadline=self.end)
                for row in waiting:
                    self.states[row['id']]['state'] = 'not_admitted_deadline'
                waiting.clear()
                boundary()
                publish()

        try:
            publish()
            while waiting or running or retiring:
                check()
                cutoff()
                for future in tuple(retiring):
                    if not future.done():
                        continue
                    current = retiring.pop(future)
                    result = future.result()  # Must have a per-owner scheduler release ACK.
                    self.reserved -= self.credits[current['id']]
                    self.inflight -= 1
                    self.completed += 1
                    self.states[current['id']].update(state='completed', released_at=self.clock())
                    results.append(result)
                    recorder.event('released', source_request=current['id'],
                        inflight=self.inflight, reserved_bytes=self.reserved)
                    if self.begin is None and self.completed >= self.warmup:
                        self.begin = self.clock()
                        self.end = self.begin+self.seconds
                        recorder.event('measurement_started', measurement_started_at=self.begin,
                            measurement_finished_at=self.end, warmup_completed=self.completed)
                        save(recorder.directory/'window.json', dict(start=self.begin, end=self.end,
                            seconds=self.seconds, warmup_completed=self.completed))
                        boundary()
                    publish()
                    current = None
                for future in tuple(running):
                    if not future.done():
                        continue
                    current = running.pop(future)
                    pending = future.result()
                    self.states[current['id']].update(state='retiring', inference_done_at=self.clock())
                    retiring[fences.submit(release, current, pending)] = current
                    publish()
                    current = None
                cutoff()
                while waiting and self.inflight < self.slots:
                    # Every admission checks the same coordinator's fixed deadline.
                    cutoff()
                    if self.closed:
                        break
                    row = waiting[0]
                    credit = self.credits[row['id']]
                    if self.reserved+credit > self.capacity:
                        break
                    admitted_at = self.clock()
                    if self.end is not None and admitted_at >= self.end:
                        cutoff()
                        break
                    current = waiting.popleft()
                    self.reserved += credit
                    self.inflight += 1
                    self.peak = max(self.peak, self.inflight)
                    self.peak_bytes = max(self.peak_bytes, self.reserved)
                    self.states[current['id']].update(state='running', admitted_at=admitted_at)
                    recorder.event('admitted', source_request=current['id'], original_case=current['original_case'],
                        admitted_at=admitted_at, history_bytes=credit, reserved_bytes=self.reserved, inflight=self.inflight)
                    running[model.submit(work, current)] = current
                    publish()
                    current = None
                if not waiting and not self.closed:
                    raise RuntimeError('Finite candidate queue exhausted before deadline; sustained load unproven')
                pending = set(running) | set(retiring)
                if pending:
                    wait(pending, timeout=.05, return_when=FIRST_COMPLETED)
                elif waiting:
                    raise RuntimeError('Full-history FIFO cannot progress without an owner')
            require(self.begin is not None and self.closed and self.inflight == self.reserved == 0,
                    'Incomplete fixed window or unreleased credits')
            recorder.event('drain_finished', measurement_finished_at=self.end)
            publish()
            return results
        except BaseException as error:
            if current is not None:
                self.states[current['id']].update(state='failed', failed_at=self.clock())
            for future in (*running, *retiring):
                future.cancel()
            save(recorder.directory/'client-failure.json', dict(error=repr(error), at=self.clock()))
            publish()  # Outstanding owner credits stay charged on failure.
            raise
        finally:
            # The existing owning controller observes client-failure before
            # waiting and aborts network I/O by terminating owned process groups.
            model.shutdown(wait=False, cancel_futures=True)
            fences.shutdown(wait=False, cancel_futures=True)


def stream(profile, source_request, session, tokens, query, sampling, timeout, recorder):
    body = dict(sampling, model=profile['served_model_name'], request_id=session,
        session_id=session, cache_salt=session, prompt=tokens, return_token_ids=True,
        stream=True, stream_options={'include_usage':True}, kvmem_query_span=query, kvmem_reselect='force')
    request = urllib.request.Request(f"http://127.0.0.1:{profile['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    begin = time.monotonic()
    recorder.event('request_started', source_request=source_request, session=session)
    result = dict(output_ids=[], text='', response_id=None, usage=None, chunks=[], finish_reason=None,
                  started_at=begin)
    done = False
    try:
        with opener.open(request, timeout=timeout) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                line = line[5:].strip()
                if line == b'[DONE]':
                    done = True
                    break
                event = json.loads(line)
                require(not event.get('error'), 'Native stream error: '+str(event.get('error')))
                if event.get('id'):
                    require(result['response_id'] in (None, event['id']), 'Response identity changed')
                    result['response_id'] = event['id']
                result['usage'] = event.get('usage') or result['usage']
                for choice in event.get('choices', []):
                    require(choice.get('index', 0) == 0, 'Multiple choices unsupported')
                    ids = choice.get('token_ids') or []
                    require(isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids),
                            'Invalid accepted stream token IDs')
                    if ids:
                        captured = recorder.event('accepted_tokens', source_request=source_request,
                            response_id=result['response_id'], token_ids=ids)
                        result['output_ids'].extend(ids)
                        result['chunks'].append([captured['at']-begin, len(ids)])
                    result['text'] += choice.get('text') or ''
                    result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    result.update(finished_at=time.monotonic(), seconds=time.monotonic()-begin,
        ttft_seconds=result['chunks'][0][0] if result['chunks'] else None)
    require(done and result['response_id'] and result['usage'], 'Incomplete accepted output stream')
    require(result['usage']['prompt_tokens'] == len(tokens) and
        result['usage']['completion_tokens'] == len(result['output_ids']) <= sampling['max_tokens'],
        'Accepted SSE token IDs differ from native usage')
    require(result['finish_reason'] in ('stop', 'length'), 'Unexpected finish reason')
    recorder.event('request_finished', source_request=source_request, response_id=result['response_id'],
        accepted_output_tokens=len(result['output_ids']))
    return result


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def read_observation(path):
    """Stat the opened inode: a concurrent atomic replacement cannot misdate it."""
    try:
        with path.open() as source:
            value = json.load(source)
            metadata = os.fstat(source.fileno())
    except FileNotFoundError:
        return None, dict(observed=False)
    wall = time.time()
    age = wall-metadata.st_mtime
    return value, dict(observed=True,mtime_ns=metadata.st_mtime_ns,
        source_published_wall_time=metadata.st_mtime,observed_wall_time=wall,
        age_seconds=age if age >= 0 else None,
        stale=age > 5 if age >= 0 else None,stale_threshold_seconds=5,
        source_monotonic_at=value.get('at'),
        scope='Opened file inode mtime is the file publication time, not GPU completion or client read time. '
              'Five-second staleness is diagnostic; stale progress is not evidence of idle GPU work.')


class Sampler:
    def __init__(self, capture, output):
        self.capture, self.output = capture, output
        self.stop, self.wake = threading.Event(), threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.run, name='stream-observer', daemon=True)

    def check(self):
        if self.error is not None:
            raise RuntimeError('Progress/archive/public-work/NVML observer failed') from self.error

    def run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            while not self.stop.is_set():
                start = time.monotonic()
                files, sources = {}, {}
                for name in ('progress', 'archive', 'public_work', 'scheduler_resources'):
                    files[name], sources[name] = read_observation(self.capture/(name+'.json'))
                memory = pynvml.nvmlDeviceGetMemoryInfo(device)
                require(memory.free >= 4096*1024**2, 'Measured GPU headroom below 4 GiB')
                row = dict(read_started_at=start, at=time.monotonic(), **files, source_observations=sources,
                    nvml=dict(gpu_used_mib=memory.used/1024**2, gpu_free_mib=memory.free/1024**2,
                        gpu_utilization_percent=pynvml.nvmlDeviceGetUtilizationRates(device).gpu,
                        gpu_power_w=pynvml.nvmlDeviceGetPowerUsage(device)/1000))
                with (self.output/'observations.jsonl').open('a') as output:
                    output.write(json.dumps(row, allow_nan=False)+'\n')
                self.wake.wait(max(0., 1.-(time.monotonic()-start)))
                self.wake.clear()
        except BaseException as error:
            self.error = error
            save(self.output/'client-failure.json', dict(error=repr(error), observer=True))
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=10)
        require(not self.thread.is_alive(), 'Observer did not stop')
        self.check()


def public_prefill_evidence(rows, begin, end):
    require(rows and len({r['counter_id'] for r in rows}) == 1, 'Missing or reset public work counter')
    rows = sorted(rows, key=lambda r:r['sequence'])
    require(len({r['sequence'] for r in rows}) == len(rows), 'Duplicate public work publication')
    for index, row in enumerate(rows):
        require(row.get('schema_version') == 1 and math.isfinite(row['at']), 'Unknown public work schema')
        for side in ('submitted', 'completed'):
            value = row[side]['public_prefill_tokens']
            require(type(value) is int and value >= 0, 'Invalid public prefill work')
            if index:
                require(value >= rows[index-1][side]['public_prefill_tokens'] and
                    row['at'] >= rows[index-1]['at'], 'Public work watermark regressed')
        require(row['submitted']['public_prefill_tokens'] >= row['completed']['public_prefill_tokens'],
                'Completed work exceeds submitted work')
    before = [r for r in rows if r['at'] <= begin]
    inside_end = [r for r in rows if r['at'] < end]
    after_begin = [r for r in rows if r['at'] >= begin]
    after_end = [r for r in rows if r['at'] >= end]
    a, b = (before[-1] if before else None), (inside_end[-1] if inside_end else None)
    c, d = (after_begin[0] if after_begin else None), (after_end[0] if after_end else None)
    observed = b['completed']['public_prefill_tokens']-a['completed']['public_prefill_tokens'] if a and b else None
    lower = max(0, b['completed']['public_prefill_tokens']-c['submitted']['public_prefill_tokens']) if b and c else None
    upper = max(0, d['submitted']['public_prefill_tokens']-a['completed']['public_prefill_tokens']) if d and a else None
    return dict(counter_id=rows[0]['counter_id'], completed_observed_delta=observed,
        completed_observed_tokens_per_second=observed/(end-begin) if observed is not None else None,
        completed_within_window_lower_bound=lower, completed_within_window_upper_bound=upper,
        boundary_brackets_complete=all(x is not None for x in (a,b,c,d)),
        boundary_snapshots=dict(before_start=a, after_start=c, before_end=b, after_end=d),
        scope='Executed public prefill positions, including re-prefill/replay; not unique input. '
              'Completed counters are asynchronously observed fence totals, not exact GPU wall timestamps. '
              'Delta uses last publications at/before T0 and before T1. Bounds exclude uncertain boundary work '
              'using submitted/completed brackets; no private Query or speculative accepted-output inference.')


def frontier_evidence(observations, cases, states, begin, end):
    """Strict unique lower bound only for owners newly admitted inside the window."""
    frontiers = {key:[] for key in cases}
    mapping, last = {}, {}
    for observation in observations:
        progress = observation.get('progress') or {}
        for native, value in progress.get('accepted_history', {}).items():
            matches = [key for key, case in cases.items() if q.native_matches(case, {'request_id':native})]
            if len(matches) != 1 or type(value) is not int or value < last.get(native, 0):
                return dict(unique_input_tokens=None, unique_input_lower_bound=None,
                    reason='Native/source mapping missing or ambiguous, or publication watermark regressed')
            key = matches[0]
            if key in mapping and mapping[key] != native:
                return dict(unique_input_tokens=None, unique_input_lower_bound=None,
                    reason='One source request matched multiple native owners')
            mapping[key], last[native] = native, value
            if observation['read_started_at'] >= begin and observation['at'] < end:
                frontiers[key].append(dict(value=min(value,cases[key]['input_tokens']),
                    read_started_at=observation['read_started_at'],read_finished_at=observation['at'],
                    source=(observation.get('source_observations') or {}).get('progress')))
    details = {}
    for key, values in frontiers.items():
        fresh = begin <= states[key]['admitted_at'] < end
        counts = [row['value'] for row in values]
        details[key] = dict(native_request=mapping.get(key), original_input_cap=cases[key]['input_tokens'],
            admitted_in_window=fresh, observed_frontiers=values,
            observed_frontier_delta=max(counts)-counts[0] if counts else None,
            stale_observations=sum(bool((row['source'] or {}).get('stale')) for row in values),
            source_time_unavailable=sum(not row['source'] for row in values),
            proved_unique_lower_bound=max(counts) if counts and fresh else 0)
    complete = all(key in mapping for key in cases if states[key]['admitted_at'] < end)
    return dict(unique_input_tokens=None,
        unique_input_lower_bound=sum(v['proved_unique_lower_bound'] for v in details.values()) if complete else None,
        mapping_complete=complete, owners=details,
        scope='No exact unique-input total from sampled progress. Lower bound includes only new owners admitted '
              'at/after T0 and published before T1, capped at original input length. Carry-in owners have '
              'observable frontier deltas only; they contribute zero to this strict unique lower bound. '
              'Read timestamps are not source update timestamps; stale/unknown publication times remain visible.')


def measurements(capture, output, cases, queue):
    begin, end = queue.begin, queue.end
    events = q.records(output/'events.jsonl')
    accepted = [r for r in events if r['kind'] == 'accepted_tokens' and begin <= r['at'] < end]
    for key, case in cases.items():
        all_ids = [token for row in events if row['kind'] == 'accepted_tokens' and row['source_request'] == key
                   for token in row['token_ids']]
        require(digest(all_ids) == case['output_ids_sha256'] and len(all_ids) == case['output_tokens'],
                'Accepted event journal differs from complete native stream')
    observations = q.records(output/'observations.jsonl')
    observed = [row for row in observations if begin <= row['read_started_at'] and row['at'] < end]
    require(observed and all(row.get('progress') is not None and row.get('public_work') is not None and
        row.get('archive') is not None for row in observed), 'Missing fixed-window counter observations')
    steps = [r for r in q.records(capture/'model_step_latency.jsonl') if begin <= r['started_at'] < end]
    require(steps, 'No completed public model-step evidence')
    prefill = [r for r in steps if r.get('prefill_tokens', 0) > 0 and not r.get('decode_requests', 0)]
    decode = [r for r in steps if r.get('decode_requests', 0) > 0 and not r.get('prefill_tokens', 0)]
    prefill_seconds = sum(r['model_ms'] for r in prefill)/1000
    completed = [r for r in events if r['kind'] == 'released' and begin <= r['at'] < end]
    refresh = [r for r in q.records(capture/'bounded_refresh.jsonl')
               if begin <= r.get('refresh_finished_at', -1) < end]
    hit_tokens = sum(r.get('raw_cache_hit_tokens', 0) for r in refresh)
    seal_rows = [r['archive'] for r in observed]
    direct_delta = None
    if all('seal_direct_bytes' in r for r in seal_rows):
        direct_delta = seal_rows[-1]['seal_direct_bytes']-seal_rows[0]['seal_direct_bytes']
        require(direct_delta >= 0, 'Successful seal watermark regressed')
    return dict(schema=SCHEMA, measurement_started_at=begin, measurement_finished_at=end,
        measurement_seconds=end-begin, accepted_output_tokens=sum(len(r['token_ids']) for r in accepted),
        accepted_output_tokens_per_second=sum(len(r['token_ids']) for r in accepted)/(end-begin),
        completed_requests=len(completed), public_prefill=public_prefill_evidence(q.records(capture/'public_work.jsonl'),begin,end),
        logical_frontiers=frontier_evidence(observations,cases,queue.states,begin,end),
        raw_cache_hit_tokens=hit_tokens, raw_cache_exercised=hit_tokens > 0,
        direct_seal_bytes_observed_delta=direct_delta,
        native=dict(steps=len(steps), pure_prefill_steps=len(prefill),
            pure_prefill_gpu_tokens_per_second=sum(r['prefill_tokens'] for r in prefill)/prefill_seconds if prefill_seconds else None,
            pure_prefill_model_seconds=prefill_seconds, pure_decode_steps=len(decode),
            pure_decode_batch_histogram={str(n):sum(r['decode_requests'] == n for r in decode) for n in range(1,9)},
            actual_batch_peak=max(r.get('prefill_requests',0)+r.get('decode_requests',0) for r in steps),
            scope='Completed timing rows whose host start is in [T0,T1); boundary steps may cross T1. '
                  'Model timing includes native sampling/MTP and capture. No accepted decode rate from verification positions.'),
        nvml=dict(samples=len(observed), peak_used_mib=max(r['nvml']['gpu_used_mib'] for r in observed),
            minimum_free_mib=min(r['nvml']['gpu_free_mib'] for r in observed),
            mean_utilization_percent=statistics.fmean(r['nvml']['gpu_utilization_percent'] for r in observed),
            mean_power_w=statistics.fmean(r['nvml']['gpu_power_w'] for r in observed)),
        zero_output_is_valid=True, gpu_idle_inferred=False, performance_gain_established=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','capture','output','fixture'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=('D',), required=True)
    args = parser.parse_args(argv)
    profile, fixture = read_json(args.config), read_json(args.fixture)
    require(fixture.get('stream_schema') == SCHEMA and fixture['stream_window_seconds'] == WINDOW_SECONDS and
        fixture['stream_warmup_completions'] == WARMUP_COMPLETIONS and fixture['workers'] == SLOTS,
        'Frozen sustained-load protocol changed')
    require(not args.output.exists(), 'Use a fresh client output; no retry')
    args.output.mkdir(parents=True)
    sampler = None
    try:
        from tokenizers import Tokenizer
        import numpy as np
        counters.validate_profile(profile, fixture['candidate'])
        save(args.output/'launch-audit.json', v2.audit_launch(profile,args.capture,'D'))
        launch = read_json(args.capture.parent/'launch.json')['vllm_argv']
        worker = json.loads(launch[launch.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
        require(all(worker.get(key,False) == profile['kvmem'].get(key,False) for key in
            ('gpu_raw_k_cache_bytes','cpu_direct_record_seal','async_action_metadata',
             'native_action_metadata','async_refresh','async_seal','direct_pinned_reads')),
            'Actual worker candidate configuration differs')
        require(protocol.sha(Path(profile['tokenizer'])/'tokenizer.json') == fixture['tokenizer_sha256'], 'Tokenizer changed')
        tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer'])/'tokenizer.json'))
        tree = ast.parse(fixture['scorer_source'])
        require(len(tree.body) == 1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name == 'score', 'Invalid frozen scorer')
        scope = {'re':re}
        exec(compile(tree,'frozen-niah-score','exec'),scope)
        recorder = Recorder(args.output)
        credits = {r['id']:original.history_bytes(r,fixture,profile) for r in fixture['rows']}
        queue = WindowQueue(fixture['rows'], credits, fixture['cpu_payload_credits_bytes'])
        require(sum(sorted(credits.values(),reverse=True)[:SLOTS]) <= queue.capacity, 'Cannot admit eight full histories')
        sampler = Sampler(args.capture,args.output)
        sampler.thread.start()

        def work(row):
            directory = args.output/'cases'/row['id']
            directory.mkdir(parents=True,exist_ok=False)
            source = Path(fixture['input_directory'])/(row['id']+'.i32')
            require(protocol.sha(source) == row['served_sha256'], 'Original input alias changed')
            tokens = np.fromfile(source,dtype='<i4').tolist()
            require(len(tokens) == row['served_input_tokens'] and tokenizer.decode(
                tokens[slice(*row['served_query'])],skip_special_tokens=False) == row['question'], 'Original Query/input differs')
            session = 'graph-stream-'+uuid.uuid4().hex
            sampling = {**fixture['sampling'],'seed':row['seed']}
            save(directory/'request.json',dict(source_request=row['id'],original_case=row['original_case'],
                session_id=session,request_id=session,input_path=str(source),input_sha256=row['served_sha256'],
                input_tokens=len(tokens),sampling=sampling,query=row['served_query'],cpu_reserved_bytes=credits[row['id']]))
            response = stream(profile,row['id'],session,tokens,row['served_query'],sampling,
                fixture['request_timeout_seconds'],recorder)
            save(directory/'response.json',response)
            return dict(id=row['id'],original_case=row['original_case'],session=session,
                response_id=response['response_id'],input_tokens=len(tokens),input_sha256=row['served_sha256'],
                output_tokens=len(response['output_ids']),output_ids_sha256=digest(response['output_ids']),seed=row['seed'])

        def release(row, result):
            terminal = original.wait_retired(args.capture,result['response_id'])
            require(terminal['history_end'] == result['input_tokens']+max(0,result['output_tokens']-1) and
                    not terminal.get('checkpoint_tokens'), 'Accepted full history/independent ownership differs')
            released = original.wait_owner_released(args.capture,result['session'],result,terminal)
            save(args.output/'cases'/row['id']/'release.json',dict(terminal=terminal,release=released))
            result.update(status='completed', release=released)
            save(args.output/'cases'/row['id']/'result.json',result)
            return result

        results = queue.run(work,release,recorder,check=sampler.check,boundary=sampler.wake.set)
        require(queue.peak == SLOTS, 'Eight-owner admission was not observed')
        released = q.wait_released(args.capture,[r['session'] for r in results],timeout=180)
        sampler.close()
        sampler = None
        cases = {r['id']:r for r in results}
        # Expensive per-case graph/answer audits occur after all inference and drain.
        for row in fixture['rows']:
            if row['id'] not in cases:
                continue
            case = cases[row['id']]
            directory = args.output/'cases'/row['id']
            response = read_json(directory/'response.json')
            evidence = protocol.audit_case(args.capture,case['response_id'],profile,row['served_query'],require_pressure=True)
            case.update(selection_sha256=digest([dict(phase='query',ranges=evidence['query']['selected_history_ranges'])]+
                [dict(phase='pressure',ranges=r['selected_history_ranges']) for r in evidence['pressure']]),
                evaluation=scope['score'](response['text'],row['expected']))
            save(directory/'evidence.json',evidence)
            save(directory/'result.json',case)
        measured = measurements(args.capture,args.output,cases,queue)
        save(args.output/'measurements.json',measured)
        archive = released['archive']
        require(archive.get('disk_read_bytes',0) == archive.get('disk_write_bytes',0) ==
            archive.get('nvme_capacity_bytes',0) == 0, 'CPU authority used SSD')
        save(args.output/'results.json',dict(status='passed',schema=SCHEMA,candidate=fixture['candidate'],
            all_owned_sessions_released=True,measurement_started_at=queue.begin,measurement_finished_at=queue.end,
            warmup_completed=WARMUP_COMPLETIONS,admitted=len(results),
            not_admitted_deadline=sum(r['state']=='not_admitted_deadline' for r in queue.states.values()),
            queue=queue.snapshot(),cases=cases,measurements_sha256=protocol.sha(args.output/'measurements.json'),
            final_archive=archive,performance_gain_established=False))
    except BaseException as error:
        save(args.output/'client-failure.json',dict(error=repr(error),at=time.monotonic()))
        raise
    finally:
        if sampler is not None:
            sampler.stop.set()
            sampler.wake.set()


if __name__ == '__main__':
    main()
