"""Owned local Qwen/GDN/MTP history-GC lifecycle and paired decode qualification.

Attach to a server started from a frozen worker package, or run an alternating
off/delete sweep with --config. Output includes requests, SSE traces, source
hashes, real-model byte oracles and native step timings. Never kills other jobs.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
import random
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time
import uuid

import httpx

from qualify_engine import freeze_worker_package


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def rows(path):
    result = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A worker may be writing its final row.
    return result


def wait_for(predicate, why, timeout=90):
    deadline = time.monotonic()+timeout
    while True:
        result = predicate()
        if result:
            return result
        if time.monotonic() > deadline:
            raise TimeoutError(why)
        time.sleep(.01)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered)-1)*fraction
    low = int(position)
    return ordered[low]+(ordered[min(low+1, len(ordered)-1)]-ordered[low])*(position-low)


class QualificationClient:
    def __init__(self, server_run, output):
        self.run, self.output = server_run, output
        self.capture = server_run/'serve/capture'
        state = read(server_run/'server-state.json')
        if state is None or state['state'] != 'ready':
            raise ValueError('Qualification requires an owned ready server state')
        self.state = state
        self.config = read(server_run/'profile.json')
        self.url = f"http://127.0.0.1:{state['port']}/v1/completions"
        self.salt = 'history-qualification-'+uuid.uuid4().hex
        self.calls = []
        self.owned_sessions = set()
        output.mkdir(parents=True, exist_ok=False)

    def payload(self, tokens, count, session, *, query=True):
        plan = {'kvmem_reselect':'force' if query else 'auto', 'kvmem_mandatory_parts':'none'}
        if query:
            plan['kvmem_query_span'] = [max(0, len(tokens)-20), len(tokens)]
        return dict(model=self.config['served_model_name'], prompt=tokens,
            max_tokens=count, temperature=0, seed=19, ignore_eos=True,
            session_id=session, cache_salt=self.salt, return_token_ids=True,
            stream=True, stream_options={'include_usage':True},
            kv_transfer_params={'kvmem_request_plan':plan})

    def call(self, name, tokens, count, session, *, cancel_after=None, query=True, timeout=900):
        self.owned_sessions.add(session)
        payload = self.payload(tokens, count, session, query=query)
        save(self.output/(name+'-request.json'), payload)
        result = dict(name=name, started_at=time.perf_counter(), token_ids=[], timeline=[])
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            with client.stream('POST', self.url, json=payload) as response:
                if response.is_error:
                    raise RuntimeError(response.read().decode())
                with (self.output/(name+'-sse.jsonl')).open('w') as stream:
                    for line in response.iter_lines():
                        if not line.startswith('data: ') or line == 'data: [DONE]':
                            continue
                        event = json.loads(line[6:])
                        stream.write(json.dumps(event)+'\n')
                        if event.get('error'):
                            raise RuntimeError(event['error'])
                        result['id'] = event.get('id', result.get('id'))
                        if event.get('usage'):
                            result['usage'] = event['usage']
                        for choice in event.get('choices', []):
                            ids = choice.get('token_ids') or []
                            if ids:
                                result['token_ids'].extend(ids)
                                result['timeline'].append([time.perf_counter(), len(ids)])
                        if cancel_after and len(result['token_ids']) >= cancel_after:
                            result['cancelled_at'] = time.perf_counter()
                            break
        result['finished_at'] = time.perf_counter()
        result['output_sha256'] = hashlib.sha256(json.dumps(result['token_ids']).encode()).hexdigest()
        if not cancel_after and len(result['token_ids']) != count:
            raise AssertionError('Public output count differs from its declared bound')
        save(self.output/(name+'-result.json'), result)
        self.calls.append(result)
        return result

    def idle(self):
        def done():
            state = read(self.capture/'scheduler_resources.json', {})
            return state if state and not any(state.get(k) for k in
                ('requests', 'owned_blocks', 'refresh_pending', 'connector_pending', 'reservations')) else None
        return wait_for(done, 'Native requests/pages did not drain')

    def release(self, session):
        payload = dict(model=self.config['served_model_name'], prompt=[1], max_tokens=1,
            session_id=session, cache_salt=self.salt, temperature=0,
            kv_transfer_params={'kvmem_release_session':True})
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.post(self.url, json=payload)
            if response.is_error:
                raise RuntimeError(response.text)
        state = self.idle()
        if any(row['namespace'] == [self.salt, session] for row in state.get('session_owners', [])):
            raise AssertionError('Explicit session release retained an owned scope')
        self.owned_sessions.discard(session)
        return state

    def close(self):
        for session in tuple(self.owned_sessions):
            self.release(session)

    async def cancel_refresh(self, stage, prompt):
        session = 'cancel-'+uuid.uuid4().hex
        self.owned_sessions.add(session)
        payload = self.payload(prompt,128,session)
        started = time.perf_counter()
        save(self.output/('cancel-'+stage+'-request.json'),payload)
        async with httpx.AsyncClient(timeout=300,trust_env=False) as client:
            async def consume():
                async with client.stream('POST',self.url,json=payload) as response:
                    if response.is_error:
                        raise RuntimeError((await response.aread()).decode())
                    async for _ in response.aiter_lines():
                        pass
            task = asyncio.create_task(consume())
            event = None
            try:
                deadline = time.monotonic()+180
                while time.monotonic() < deadline:
                    event = next((r for r in rows(self.capture/'async_refresh.jsonl')
                                  if r['at'] >= started and r['stage'] == stage),None)
                    if event is not None:
                        break
                    if task.done():
                        task.result()
                        raise AssertionError('Request finished before its refresh cancellation stage')
                    await asyncio.sleep(.005)
                if event is None:
                    raise TimeoutError(stage)
                cancelled_at = time.perf_counter()
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        state = self.idle()
        events = rows(self.capture/'async_refresh.jsonl')
        ack = next((r for r in events if event is not None and
                    r['request_id'] == event['request_id'] and r['stage'] == 'cancelled'),None)
        if ack is None:
            raise AssertionError(('Missing fenced refresh cancellation',stage,events[-8:]))
        result = dict(stage=stage,trigger=event,cancelled_at=cancelled_at,ack=ack,
                      owned_blocks=state['owned_blocks'])
        save(self.output/('cancel-'+stage+'-result.json'),result)
        self.release(session)
        return result

    def restored(self, result, expected_history):
        prefix = result['id']+'-0-'
        evidence = [r for r in rows(self.capture/'verified_restores.jsonl')
                    if r['request_id'].startswith(prefix)]
        if (len(evidence) != 1 or not evidence[0]['byte_equal'] or
                evidence[0]['history_tokens'] != expected_history or
                not evidence[0]['recurrent_tensors'] or not evidence[0]['mtp_attention_layers']):
            raise AssertionError(('Missing exact target/GDN/MTP restoration', evidence, expected_history))
        return evidence[0]

    def fixture_rpc(self, method, **kwargs):
        with httpx.Client(timeout=15, trust_env=False) as client:
            response = client.post(self.url.rsplit('/v1/', 1)[0]+'/collective_rpc',
                json=dict(method=method, args=[], kwargs=kwargs, timeout=10))
            response.raise_for_status()
        values = response.json().get('results')
        if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], dict):
            raise AssertionError('Fixture requires one native worker response')
        return values[0]

    def metrics(self):
        archive = read(self.capture/'archive.json')
        history = archive.get('historical_gc') or {}
        if (archive['gc_wait_count'] or archive['gc_wait_seconds'] or history.get('policy_failures') or
                history.get('admission_retries')):
            raise AssertionError(('Unexpected foreground GC wait/retry or policy error', archive))
        return dict(archive=archive, scheduler=self.idle())


def lifecycle(client, prompt, append):
    session = 'pm-'+uuid.uuid4().hex
    first = client.call('first', prompt, 129, session)
    client.idle()
    continued = prompt+first['token_ids']+append
    second = client.call('m-resume', continued, 33, session, query=False)
    client.idle()
    m = client.restored(second, len(prompt)+128)
    branch = continued+second['token_ids']+append[:8]
    cancelled = client.call('cancel-partial-tail', branch, 128, session, cancel_after=7, query=False)
    client.idle()
    retry = client.call('m-retry', branch, 33, session, query=False)
    client.idle()
    rebased = client.restored(retry, len(continued)+32)
    alternative = client.call('p-alternative', prompt, 17, session)
    client.idle()
    p = [r for r in rows(client.capture/'verified_restores.jsonl')
         if r['request_id'].startswith(alternative['id']+'-0-')]
    if len(p) != 1 or not p[0]['byte_equal'] or not 0 < p[0]['history_tokens'] <= len(prompt):
        raise AssertionError(('Missing P fallback byte oracle', p))
    snapshot = client.metrics()
    if not snapshot['archive']['historical_gc']['retired']:
        raise AssertionError('Lifecycle did not exercise actual LRU deletion')
    client.release(session)
    refresh_cancellations = [asyncio.run(client.cancel_refresh(stage,prompt))
                             for stage in ('query','assembly_started')]
    report = dict(status='passed', scope='real_model_pm_cancel_gdn_mtp',
        m=m, partial_tail_cancel=cancelled['cancelled_at'], rebased_m=rebased, p=p[0],
        source_digest=client.state['source_digest'], metrics=snapshot)
    report['refresh_cancellations'] = refresh_cancellations
    save(client.output/'results.json', report)
    return report


def audit_statistical_fixture(fixture, result, mode, refreshes):
    """A changed counter brackets real work between two host samples.

    All worker/client times use Linux perf_counter in the same host clock
    domain. The entire observation interval must lie strictly inside first/last
    received token, and later accepted decode progress must follow the
    reclamation observation before the final output bound. A request-level
    before/after delta or last SSE arrival alone is insufficient.
    """
    if (not fixture or fixture.get('source') != 'synthetic_gc_throughput_fixture' or
            fixture.get('semantic_measurement') is not False or fixture.get('error') or
            not fixture.get('flushed') or fixture.get('observations_published') != 3):
        raise AssertionError(('Incomplete synthetic statistical publication', fixture))
    first, last = result['timeline'][0][0], result['timeline'][-1][0]
    published = [r for r in fixture['events'] if r['kind'] == 'published']
    if len(published) != 1 or not first < published[0]['at'] < last:
        raise AssertionError('Synthetic publication did not occur during measured decode')
    overlapping = []
    for row in refreshes:
        begin = row.get('refresh_started_at', row.get('at'))
        end = row.get('refresh_finished_at', begin)
        if begin is not None and end is not None and begin < last and end > first:
            overlapping.append(row)
    if overlapping:
        raise AssertionError(('Stable-window decode performed a refresh', overlapping))
    witnesses = {}
    for name in ('retired', 'cpu_released'):
        events = [r for r in fixture['events'] if r['kind'] == name]
        prompt = fixture['plan']['prompt_tokens']
        terminal = prompt+fixture['plan']['decode_tokens']
        inside = [r for r in events if first < r['interval_begin'] <= r['interval_end'] < last
                  and r['after'] > r['before']
                  and prompt < r['accepted_history'] < terminal
                  and r['interval_end'] < r.get('following_decode_at', 0) < last
                  and r['accepted_history'] < r.get('following_accepted_history', 0) < terminal]
        delta = (fixture['last_sample']['counters'][name] -
                 fixture['baseline']['counters'][name])
        reclaimed_inside = sum(r['after']-r['before'] for r in inside)
        if mode == 'delete' and (not inside or delta != len(fixture['candidate_handles']) or
                reclaimed_inside != delta):
            raise AssertionError(('Missing exact candidate reclamation inside decode', name, events, delta))
        if mode == 'off' and delta:
            raise AssertionError(('Off retired fixture blocks', name, delta))
        witnesses[name] = inside
    return dict(source=fixture['source'], semantic_measurement=False,
        clock='time.perf_counter; same Linux host for client and worker',
        first_token_at=first, last_token_at=last, publication=published[0],
        overlapping_refreshes=0, release_witnesses=witnesses, raw=fixture)


def relative_regression(changed, baseline):
    if changed is None and baseline is None:
        return None
    if changed is None or baseline is None or baseline <= 0:
        raise AssertionError('Paired instrumentation differs or baseline is invalid')
    return changed/baseline-1


def throughput_uncertainty(comparisons, *, draws=10000, seed=19):
    """Paired percentile-bootstrap CI; descriptive, not a zero-cost proof."""
    losses = [r['throughput_loss'] for r in comparisons]
    if not losses:
        raise ValueError('At least one pair is required')
    interval = None
    if len(losses) >= 2:
        rng = random.Random(seed)
        medians = [percentile([rng.choice(losses) for _ in losses], .5) for _ in range(draws)]
        interval = [percentile(medians, .025), percentile(medians, .975)]
    unresolved = len(losses) < 5 or interval is None or interval[1] > .01
    return dict(statistic='paired median throughput loss', losses=losses,
        minimum=min(losses), maximum=max(losses), median=percentile(losses, .5),
        confidence=.95, bootstrap_draws=draws, bootstrap_seed=seed,
        percentile_interval=interval, noninferiority_margin=.01,
        assessment='unresolved' if unresolved else 'upper_bound_within_1pct',
        more_pairs_recommended=unresolved,
        limitation='Descriptive paired bootstrap under this workload; no proof of zero overhead')


def performance(client, prompt, count):
    warm = 'warm-'+uuid.uuid4().hex
    client.call('warmup', prompt, count, warm)
    client.idle()
    client.release(warm)
    before = client.metrics()
    session = 'measure-'+uuid.uuid4().hex
    synthetic = client.state.get('statistical_fixture', False)
    if synthetic:
        client.fixture_rpc('historical_gc_fixture_arm',
            namespace=[client.salt, session], prompt_tokens=len(prompt), decode_tokens=count)
    result = client.call('measure', prompt, count, session)
    client.idle()
    after = client.metrics()
    timings = [r for r in rows(client.capture/'model_step_latency.jsonl')
        if result['started_at'] <= r['started_at'] < result['finished_at'] and
           r['decode_requests'] == 1 and r['prefill_requests'] == 0]
    gaps = [(b['started_at']-a['started_at'])*1000 for a,b in zip(timings, timings[1:])]
    measured_tails = bool(client.config['kvmem'].get('step_latency_interval', 0))
    if measured_tails and len(gaps) < 100:
        raise AssertionError('Not enough native decode intervals for tail-latency qualification')
    if not measured_tails and not synthetic:
        raise AssertionError('Legacy qualification requires native step timing')
    h0, h1 = (m['archive'].get('historical_gc') or {} for m in (before,after))
    mode = (client.config['kvmem'].get('historical_gc') or {'mode':'legacy'})['mode']
    if mode == 'delete' and (h1['retired'] <= h0['retired'] or h1['cpu_released'] <= h0['cpu_released']):
        raise AssertionError('Measured decode did not overlap actual GC work')
    first, last = result['timeline'][0], result['timeline'][-1]
    report = dict(mode=mode, source_digest=client.state['source_digest'],
        throughput=(count-first[1])/(last[0]-first[0]),
        p95_ms=percentile(gaps,.95) if measured_tails else None,
        p99_ms=percentile(gaps,.99) if measured_tails else None,
        tail_latency_measured=measured_tails, intervals=len(gaps), output_sha256=result['output_sha256'],
        retired=h1.get('retired',0)-h0.get('retired',0),
        cpu_released=h1.get('cpu_released',0)-h0.get('cpu_released',0),
        metrics=after)
    if synthetic:
        fixture = client.fixture_rpc('historical_gc_fixture_snapshot')
        save(client.output/'statistical-fixture.json', fixture)
        refreshes = rows(client.capture/'bounded_refresh.jsonl')+rows(client.capture/'async_refresh.jsonl')
        report['statistical_fixture'] = audit_statistical_fixture(fixture, result, mode, refreshes)
    client.release(session)
    save(client.output/'results.json', report)
    return report


def capacity_pressure(client, prompt, append):
    """An admitted peer must finish while another turn waits for idle GC."""
    sessions = ['quota-'+uuid.uuid4().hex for _ in range(3)]
    first = client.call('quota-first',prompt,128,sessions[0]);client.idle()
    client.call('quota-second',prompt,128,sessions[1]);client.idle()
    before = client.metrics()
    admission = before['scheduler']['cpu_history_admission']
    block = client.config['kvmem']['block_tokens']
    if admission['available_records'] >= (len(prompt)+128+block-1)//block+4:
        raise AssertionError('Pressure profile has too much free CPU capacity; use the documented small quota')
    samples = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        continued = prompt+first['token_ids']+append
        peer = executor.submit(client.call,'quota-peer',continued,128,sessions[0],query=False)
        wait_for(lambda: read(client.capture/'scheduler_resources.json',{}).get('requests'),
                 'Peer did not reach native execution')
        cold = executor.submit(client.call,'quota-waiting',prompt,128,sessions[2],timeout=90)
        deadline = time.monotonic()+180
        signature = None
        while not peer.done() or not cold.done():
            state = read(client.capture/'scheduler_resources.json',{})
            current = state.get('cpu_history_admission',{})
            if current:
                if current['reserved_bytes'] > current['capacity_bytes']:
                    raise AssertionError('Native scheduler exceeded its whole-turn CPU quota')
                new = (current['active'],current['waiting'],current['reserved_records'],current['deferrals'])
                if new != signature:
                    samples.append(dict(at=time.perf_counter(),admission=current,requests=state.get('requests')))
                    signature = new
            if time.monotonic() > deadline:
                raise TimeoutError('Idle capacity pressure did not unblock the queued turn')
            time.sleep(.01)
        peer_result,cold_result=peer.result(),cold.result()
    client.idle()
    after = client.metrics()
    if (not any(s['admission']['waiting'] and s['admission']['active'] for s in samples) or
            after['scheduler']['cpu_history_admission']['deferrals'] <= admission['deferrals'] or
            after['archive']['historical_gc']['retired'] <= before['archive']['historical_gc']['retired']):
        raise AssertionError('Missing actual queued admission and background idle LRU progress')
    restored=client.restored(peer_result,len(prompt)+127)
    result=dict(status='passed',scope='public_whole_turn_reservation_and_idle_gc',
        source_digest=client.state['source_digest'],before=before,after=after,samples=samples,
        restored_peer=restored,peer_finished_at=peer_result['finished_at'],cold_finished_at=cold_result['finished_at'])
    for session in sessions:
        client.release(session)
    save(client.output/'results.json',result)
    return result


def workload(args, client, fixture):
    if args.kind == 'lifecycle':
        return lifecycle(client,fixture['input_ids'],fixture['append_ids'])
    if args.kind == 'pressure':
        return capacity_pressure(client,fixture['input_ids'],fixture['append_ids'])
    return performance(client,fixture['input_ids'],args.decode_tokens)


@contextmanager
def owned_server(root, config, output, source, digest, fixture_source=None):
    output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(json.dumps(config))
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        profile['port'] = sock.getsockname()[1]
    profile['host'] = '127.0.0.1'
    save(output/'profile.json', profile)
    env = dict(os.environ, PYTHONPATH=str(source))
    env.pop('VLLM_API_KEY',None)
    command = [sys.executable,'-m','vllm_kvmem','serve','--config',str(output/'profile.json'),
               '--run-dir',str(output/'serve')]
    if fixture_source is not None:
        env['PYTHONPATH'] = os.pathsep.join((str(source), str(fixture_source)))
        command = [sys.executable, '-c',
            'import historical_gc_throughput_fixture as fixture; fixture.serve()',
            '--config', str(output/'profile.json'), '--run-dir', str(output/'serve')]
    with (output/'server.log').open('w') as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=root)
        state = dict(pid=process.pid, port=profile['port'], source_digest=digest, state='starting',
                     statistical_fixture=fixture_source is not None)
        save(output/'server-state.json',state)
        try:
            def healthy():
                if process.poll() is not None:
                    raise RuntimeError(f'Server exited; inspect {output}/server.log')
                try:
                    with httpx.Client(timeout=1,trust_env=False) as client:
                        return client.get(f"http://127.0.0.1:{profile['port']}/health").status_code == 200
                except httpx.HTTPError:
                    return False
            wait_for(healthy,'Model startup exceeded ten minutes',600)
            (output/'serve/capture/armed').touch()
            state['state']='ready';save(output/'server-state.json',state)
            yield output
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
            state.update(state='stopped',exit_code=process.returncode)
            save(output/'server-state.json',state)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--attach-server-run',type=Path)
    parser.add_argument('--input',type=Path,required=True,help='Frozen input_ids and append_ids JSON')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--kind',choices=('lifecycle','pressure','performance'),default='lifecycle')
    parser.add_argument('--pairs',type=int,default=5)
    parser.add_argument('--decode-tokens',type=int,default=512)
    parser.add_argument('--statistical-fixture', action='store_true',
        help='Synthetic generated-block evidence, C1 only; never a semantic/attention measurement')
    parser.add_argument('--baseline',choices=('off','legacy'),default='off',
                        help='legacy omits historical_gc and measures the complete feature overhead')
    args=parser.parse_args()
    fixture=read(args.input)
    if (not fixture or not isinstance(fixture.get('input_ids'),list) or
            any(type(i) is not int or i < 0 for i in fixture['input_ids'])):
        raise ValueError('Frozen fixture requires nonnegative integer token IDs')
    args.output=args.output.resolve()
    if args.statistical_fixture and args.attach_server_run:
        raise ValueError('Synthetic evidence requires a fresh owned worker extension')
    if args.statistical_fixture and (args.kind != 'performance' or args.baseline != 'off' or
            len(fixture['input_ids']) != 256 or args.decode_tokens < 512):
        raise ValueError('Synthetic C1 pairs require performance/off, a 256-token prompt and >=512 output')
    if args.attach_server_run:
        client=QualificationClient(args.attach_server_run.resolve(),args.output)
        try:
            result=workload(args,client,fixture)
        finally:
            client.close()
        print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
        return
    if args.config is None or not 1 <= args.pairs <= 20:
        raise ValueError('A config and 1..20 pairs are required for an owned sweep')
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    driver=Path(__file__).read_bytes()
    (args.output/'qualification_driver.py').write_bytes(driver)
    save(args.output/'frozen-input.json',fixture)
    input_digest = hashlib.sha256((args.output/'frozen-input.json').read_bytes()).hexdigest()
    manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (root/'src/vllm_kvmem').rglob('*.py')}
    source,digest=freeze_worker_package(root,args.output,manifest)
    save(args.output/'source_manifest.json',manifest)
    config=read(args.config)
    fixture_source = None
    fixture_digest = None
    if args.statistical_fixture:
        kv = config['kvmem']
        if (config.get('max_num_seqs') != 1 or config.get('execution_mode') != 'eager' or
                config.get('draft_method') != 'mtp' or config.get('draft_tokens') != 1 or
                kv.get('historical_gc', {}).get('policy') != 'statistical' or
                kv.get('block_tokens') != 32 or kv.get('graph_mode', 'off') != 'off' or
                kv.get('action_native_scores', 'off') != 'off' or
                kv.get('select_budget', 0)+kv.get('gen_budget', 0) < len(fixture['input_ids'])+args.decode_tokens+64):
            raise ValueError('Fixture requires eager C1/MTP1, statistical GC, graph/native-attention off and a stable window')
        fixture_source = args.output/'frozen_diagnostics'
        fixture_source.mkdir()
        diagnostic = root/'diagnostics/historical_gc_throughput_fixture.py'
        shutil.copy2(diagnostic, fixture_source/diagnostic.name)
        fixture_digest = hashlib.sha256((fixture_source/diagnostic.name).read_bytes()).hexdigest()
        if hashlib.sha256(diagnostic.read_bytes()).hexdigest() != fixture_digest:
            raise RuntimeError('Diagnostic source changed while freezing')
        save(args.output/'fixture_manifest.json', dict(source='synthetic_gc_throughput_fixture',
            semantic_measurement=False, sha256=fixture_digest,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
    reports=[]
    with open('/tmp/kvmem-vllm-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if active:
            raise RuntimeError('GPU has existing compute processes; refusing an overlapping sweep')
        modes=(args.baseline,'delete') if args.kind=='performance' else ('delete',)
        for pair in range(args.pairs if args.kind=='performance' else 1):
            for mode in (modes if pair%2==0 else tuple(reversed(modes))):
                profile=json.loads(json.dumps(config))
                if mode == 'legacy':
                    profile['kvmem'].pop('historical_gc',None)
                else:
                    profile['kvmem']['historical_gc']['mode']=mode
                run=args.output/f'{pair+1:02d}-{mode}'
                print(f'Loading pair {pair+1}, mode {mode}',flush=True)
                with owned_server(root,profile,run,source,digest,fixture_source):
                    client=QualificationClient(run,run/'client')
                    try:
                        value=workload(args,client,fixture)
                    finally:
                        client.close()
                    value['pair']=pair+1;reports.append(value)
                    save(args.output/'runs.json',reports)
                    print(json.dumps({k:v for k,v in value.items() if k!='metrics'}),flush=True)
    if args.kind=='performance':
        comparisons=[]
        for pair in range(1,args.pairs+1):
            off,delete=({r['mode']:r for r in reports if r['pair']==pair}[m] for m in (args.baseline,'delete'))
            comparisons.append(dict(pair=pair,
                throughput_loss=1-delete['throughput']/off['throughput'],
                p95_regression=relative_regression(delete['p95_ms'],off['p95_ms']),
                p99_regression=relative_regression(delete['p99_ms'],off['p99_ms']),
                identical_output=off['output_sha256']==delete['output_sha256']))
        medians={k:(percentile([r[k] for r in comparisons],.5)
                    if all(r[k] is not None for r in comparisons) else None)
                 for k in ('throughput_loss','p95_regression','p99_regression')}
        tails = medians['p95_regression'] is not None
        passed=(args.pairs>=5 and medians['throughput_loss']<=.01 and
                (not tails or medians['p95_regression']<=.02 and medians['p99_regression']<=.05) and
                all(r['identical_output'] for r in comparisons))
        result=dict(status='passed' if passed else 'failed',source_digest=digest,
                    comparisons=comparisons,paired_medians=medians,
                    throughput_uncertainty=throughput_uncertainty(comparisons),
                    baseline='historical_gc off' if args.baseline=='off' else 'historical_gc omitted',
                    driver_sha256=hashlib.sha256(driver).hexdigest(),
                    fixture_sha256=fixture_digest, input_sha256=input_digest, tail_latency_measured=tails,
                    scope=('Native statistical policy and reclamation overhead with shared synthetic producer/instrumentation in C1 stable-window decode; '
                           'no semantic quality or attention-estimator claim' if args.statistical_fixture else
                           'Historical GC paired decode qualification'))
        save(args.output/'results.json',result)
        print(json.dumps(result),flush=True)
        if not passed:
            raise SystemExit(1)


if __name__=='__main__':
    main()
