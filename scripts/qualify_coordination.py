"""Paired native HTTP qualification, queued until the GPU is continuously idle.

Synthetic workloads or an offline JSONL dataset. Never starts DeepSWE or terminates foreign processes.
All arms preserve model/selector/precision/graph and memory budget settings.
"""
import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import random
import statistics
import subprocess
import sys
import time

import httpx
import psutil

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('B0', 'B1', 'B2', 'B3', 'B4')


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def source_hashes():
    paths = [Path(__file__), *sorted((ROOT/'src/vllm_kvmem').glob('*.py'))]
    offline = ROOT/'scripts/offline_workload.py'
    if offline.exists():
        paths.append(offline)
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def offline_support():
    spec = importlib.util.spec_from_file_location('kvmem_offline_workload', ROOT/'scripts/offline_workload.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lo = int(position); hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def gpu_state():
    def query(args):
        return subprocess.check_output(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                                       text=True, timeout=10).strip().splitlines()
    rows = query(['--query-gpu=memory.used,utilization.gpu'])
    if len(rows) != 1:
        raise RuntimeError('Qualification requires exactly one visible GPU')
    memory, utilization = map(int, rows[0].split(','))
    processes = query(['--query-compute-apps=pid'])
    return dict(memory_mib=memory, utilization=utilization, pids=[int(p.strip()) for p in processes])


def owned_pids(server):
    try:
        return {server.pid, *(p.pid for p in psutil.Process(server.pid).children(recursive=True))}
    except psutil.NoSuchProcess:
        return {server.pid}


def stop_owned(server):
    # A dedicated process group contains only this arm's backend and workers.
    try:
        os.killpg(server.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        server.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # Parent exit is not proof that its workers exited. Reap the owned group.
    try:
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait(timeout=10)


async def wait_idle(output, stable_seconds):
    idle_since = None
    while True:
        state = await asyncio.to_thread(gpu_state)
        idle = not state['pids'] and state['memory_mib'] < 512 and state['utilization'] < 5
        idle_since = (idle_since or time.monotonic()) if idle else None
        save(output/'status.json', dict(status='waiting_for_gpu', at=time.time(), gpu=state,
             stable_idle_seconds=time.monotonic()-idle_since if idle_since else 0,
             required_idle_seconds=stable_seconds))
        if idle_since is not None and time.monotonic() - idle_since >= stable_seconds:
            return
        await asyncio.sleep(10)


def make_profiles(config, output, *, arms=('B0', 'B1', 'B2'), step_latency_interval=0):
    base = json.loads(config.read_text())
    if base['kvmem'].get('archive_backend') != 'cpu' or not base['kvmem'].get('async_refresh'):
        raise ValueError('Use a CPU-history asynchronous-refresh native serving profile')
    if base['kvmem']['session_capacity'] < 8 or base['max_num_seqs'] < 8:
        raise ValueError('Qualification requires eight checkpoint slots and native requests')
    profiles = {}
    from vllm_kvmem.serve import build_launch, DEFAULTS, PATHS
    if not arms or any(arm not in ARMS for arm in arms):
        raise ValueError('Unknown or empty qualification arms')
    if type(step_latency_interval) is not int or not 0 <= step_latency_interval <= 1024:
        raise ValueError('Step latency interval must be an integer in [0, 1024]')
    # Instrument the baseline equally: a v2-only measurement overhead would
    # otherwise be indistinguishable from an actual scheduling regression.
    interval = step_latency_interval or (4 if {'B3', 'B4'} & set(arms) else 0)
    for arm in ARMS:
        if arm not in ('B0', 'B1', 'B2') and arm not in arms:
            continue
        profile = json.loads(json.dumps(base))
        profile['kvmem'].update(cpu_history_coordination=arm != 'B0', cpu_progress_reserve=True,
            maintenance_max_steps=8 if arm in ('B2', 'B3', 'B4') else 0, maintenance_busy_ms=2.,
            maintenance_idle_ms=8., maintenance_aging_ms=250., maintenance_inflight_bytes=0,
            async_restore=arm in ('B3', 'B4'), adaptive_prefill=arm == 'B4',
            step_latency_interval=interval)
        validated = build_launch(profile, config.parent, output/'validation'/arm)['profile']
        # The emitted native plan disables long-prefill clipping for atomic Q;
        # it is not itself a reusable input profile (threshold 0 is internal).
        profiles[arm] = {**DEFAULTS, **profile}
        for name in PATHS:
            if name in validated:
                profiles[arm][name] = validated[name]
        save(output/f'{arm}.json', profiles[arm])
    return profiles


def collect_diagnostics(server_directory, *, started=None, finished=None):
    """Summarize completed telemetry separately from client throughput.

    Events may be published after a case ends; raw server logs remain the
    authoritative source. These are the observations visible at collection,
    not a claim that every GPU step or restore has been sampled.
    """
    result = {'scope': 'completed telemetry visible at collection; model step latency is not ITL'}
    for filename, timestamp in (('model_step_latency.jsonl', 'started_at'),
                                ('async_restore.jsonl', 'started')):
        rows = []; skipped = 0; paths = sorted(server_directory.rglob(filename))
        if not paths:
            continue
        for path in paths:
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1  # A concurrent final write may be incomplete.
                    continue
                if started is not None and row[timestamp] < started:
                    continue
                if finished is not None and row[timestamp] >= finished:
                    continue
                rows.append(row)
        summary = dict(samples=len(rows), incomplete_lines=skipped,
                       sources=[str(p.relative_to(server_directory)) for p in paths])
        if filename == 'model_step_latency.jsonl':
            for label, selected in (
                    ('all', rows),
                    ('decode_only', [r for r in rows if r['decode_requests'] and not r['prefill_tokens']]),
                    ('mixed', [r for r in rows if r['decode_requests'] and r['prefill_tokens']]),
                    ('prefill_only', [r for r in rows if not r['decode_requests'] and r['prefill_tokens']])):
                summary[label] = dict(samples=len(selected),
                    step_ms_p50=percentile([r['step_ms'] for r in selected], .5),
                    step_ms_p95=percentile([r['step_ms'] for r in selected], .95),
                    host_submit_ms_p95=percentile([r['host_submit_ms'] for r in selected], .95),
                    model_ms_p50=percentile([r['model_ms'] for r in selected if 'model_ms' in r], .5),
                    maintenance_ms_p95=percentile([r['maintenance_ms'] for r in selected if 'maintenance_ms' in r], .95))
            result['model_step_latency'] = summary
        else:
            summary.update(duration_seconds_p50=percentile([r['finished']-r['started'] for r in rows], .5),
                duration_seconds_p95=percentile([r['finished']-r['started'] for r in rows], .95),
                host_prepare_seconds_p95=percentile([r['host_prepare_seconds'] for r in rows], .95),
                uploaded_bytes=sum(r['uploaded_bytes'] for r in rows),
                model_bytes=sum(r['model_bytes'] for r in rows),
                split_tokens=sum(r.get('split_tokens', 0) for r in rows),
                partial_tokens=sum(r.get('partial_tokens', 0) for r in rows),
                raw_cache_hit_tokens=sum(r.get('raw_cache_hit_tokens', 0) for r in rows),
                resident_v_reused_tokens=sum(r.get('resident_v_reused_tokens', 0) for r in rows),
                resident_active_tokens=sum(r.get('resident_active_tokens', 0) for r in rows),
                resident_restores=sum(r.get('resident_active_tokens', 0) > 0 for r in rows),
                with_public_peers=sum(r['public_peer_count'] > 0 for r in rows),
                cancelled=sum(bool(r['cancelled']) for r in rows))
            result['async_restore'] = summary
    return result


async def call(client, profile, tokens, session, count, *, turn=0, ready=None, query=True, request_id=None):
    begin = time.perf_counter()
    payload = dict(model=profile['served_model_name'], prompt=tokens, session_id=session, cache_salt=session,
        max_tokens=count, temperature=0, seed=1000, ignore_eos=True, return_token_ids=True,
        stream=True, stream_options={'include_usage':True}, kvmem_reselect='auto' if query else 'off')
    if request_id is not None:
        payload['request_id'] = request_id
    if query:
        # A stable old Query across tool rounds preserves each arm's real output
        # prefix. Initial inputs/tool return IDs are identical across arms.
        initial_end = min(len(tokens), 32768)
        payload['kvmem_query_span'] = [initial_end-128, initial_end-1]
    else:
        # A reset replaces the entire prefix, including its historical pins.
        payload['kvmem_mandatory_spans'] = []
    timeline = []; ids = []; usage = None; response_id = None
    async with client.stream('POST', f"http://127.0.0.1:{profile['port']}/v1/completions", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith('data: '):
                continue
            data = line[6:]
            if data == '[DONE]':
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            response_id = event['id']
            usage = event.get('usage') or usage
            for choice in event.get('choices', []):
                chunk = choice.get('token_ids') or []
                if chunk:
                    ids.extend(chunk)
                    timeline.append([time.perf_counter(), len(chunk)])
                    if ready is not None:
                        ready.set()
    if len(ids) != count or not usage or usage['completion_tokens'] != count:
        raise RuntimeError(f'Incomplete fixed-length stream: {len(ids)}/{count}')
    return dict(begin=begin, end=time.perf_counter(), session=session, turn=turn,
        input_tokens=len(tokens), input_sha256=digest(tokens), output_ids=ids,
        output_sha256=digest(ids), response_id=response_id, timeline=timeline, usage=usage)


def summarize(rows, wall_seconds):
    gaps = [b[0]-a[0] for r in rows for a,b in zip(r['timeline'], r['timeline'][1:])]
    output = sum(len(r['output_ids']) for r in rows)
    # Common decode window is meaningful only for one simultaneous turn.
    common = None
    if len(rows) == 8:
        begin = max(r['timeline'][0][0] for r in rows)
        end = min(r['timeline'][-1][0] for r in rows)
        if end > begin:
            count = sum(n for r in rows for t,n in r['timeline'] if begin < t <= end)
            common = dict(seconds=end-begin, tokens=count, tokens_per_second=count/(end-begin))
    return dict(requests=len(rows), output_tokens=output, wall_seconds=wall_seconds,
        output_tokens_per_second=output/wall_seconds, common_decode=common,
        output_event_gap_p95_seconds=percentile(gaps,.95),
        ttft_p95_seconds=percentile([r['timeline'][0][0]-r['begin'] for r in rows],.95),
        request_seconds_p95=percentile([r['end']-r['begin'] for r in rows],.95))


async def workload(client, profile, fixture, case, directory, args):
    if case == 'offline':
        from types import SimpleNamespace
        return await offline_support().workload(client, profile, directory, args,
            SimpleNamespace(save=save, percentile=percentile, collect_diagnostics=collect_diagnostics))
    sessions = [f'coordination-{i}' for i in range(8)]
    # Explicit prefix replacement releases the prior case's history through the
    # same fenced path in every arm. It is outside the measured workload.
    await asyncio.gather(*(call(client, profile, [fixture['eos']], s, 1, query=False) for s in sessions))
    before = (await client.get(f"http://127.0.0.1:{profile['port']}/metrics")).text
    diagnostics_started = time.monotonic()
    started = time.perf_counter()
    rows = []
    if case in ('decode', 'prefill'):
        n = args.decode_tokens if case == 'decode' else 1
        rows = await asyncio.gather(*(call(client, profile, fixture['prompt'][:32768], s, n) for s in sessions))
    elif case == 'mixed':
        events = [asyncio.Event() for _ in range(7)]
        async with asyncio.TaskGroup() as group:
            peers = [group.create_task(call(client, profile, fixture['prompt'][:8192], s,
                args.mixed_decode_tokens, ready=events[i])) for i,s in enumerate(sessions[:7])]
            await asyncio.wait_for(asyncio.gather(*(e.wait() for e in events)), timeout=300)
            target = group.create_task(call(client, profile, fixture['prompt'][:args.long_tokens], sessions[7], 512))
        rows = [t.result() for t in peers] + [target.result()]
    else:
        barrier = asyncio.Barrier(8)
        async def agent(i, session):
            tokens = fixture['prompt'][:32768]
            results=[]
            for turn in range(args.turns):
                if turn:
                    if case == 'burst':
                        await barrier.wait()
                    else:
                        await asyncio.sleep((i % 4) * .75)
                    tokens = tokens + fixture['tool']
                row = await call(client, profile, tokens, session, args.agent_tokens, turn=turn)
                results.append(row)
                tokens = tokens + row['output_ids']
            # A completed trajectory releases its historical residency even if
            # another agent is waiting for CPU credits. Otherwise the harness
            # itself could deadlock a correctly bounded admission controller.
            await call(client, profile, [fixture['eos']], session, 1, query=False)
            return results
        groups = await asyncio.gather(*(agent(i,s) for i,s in enumerate(sessions)))
        rows = [row for group in groups for row in group]
    wall = max(row['end'] for row in rows)-started
    diagnostics_finished = time.monotonic()
    after = (await client.get(f"http://127.0.0.1:{profile['port']}/metrics")).text
    directory.mkdir()
    (directory/'metrics-before.txt').write_text(before)
    (directory/'metrics-after.txt').write_text(after)
    save(directory/'traces.json', rows)
    result = summarize(rows, wall)
    if case == 'prefill':
        result['prefill_service_tokens_per_second'] = sum(r['input_tokens'] for r in rows)/wall
    save(directory/'results.json', result)
    save(directory/'diagnostics.json', collect_diagnostics(directory.parent/'server',
         started=diagnostics_started, finished=diagnostics_finished))
    return result


async def run_arm(args, profile, fixture, output, arm, pair):
    directory = output/f'pair-{pair:02d}-{arm}'
    directory.mkdir()
    # Two minutes without compute processes avoids mistaking a loading/tool gap
    # in the other test for a free GPU. Recheck immediately before launch.
    await wait_idle(output, args.idle_seconds)
    if source_hashes() != json.loads((output/'manifest.json').read_text())['source_sha256']:
        raise RuntimeError('Source changed while queued; prepare a fresh qualification')
    state = gpu_state()
    if state['pids'] or state['memory_mib'] >= 512:
        raise RuntimeError('GPU ownership changed before launch')
    environment = {**os.environ, 'PYTHONPATH':str(ROOT/'src'), 'TOKENIZERS_PARALLELISM':'false'}
    with (directory/'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable,'-m','vllm_kvmem','serve','--config',str(output/f'{arm}.json'),
            '--run-dir',str(directory/'server')], cwd=ROOT, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        save(output/'status.json', dict(status='running', pair=pair, arm=arm, server_pid=server.pid, at=time.time()))
        async def guard():
            while True:
                state = await asyncio.to_thread(gpu_state)
                if 'offline' in args.cases:
                    with (directory/'gpu_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(at=time.perf_counter(), **state))+'\n')
                foreign = set(state['pids']) - owned_pids(server)
                if foreign:
                    raise RuntimeError(f'Foreign GPU work appeared; stopping only this arm: {sorted(foreign)}')
                if server.poll() is not None:
                    raise RuntimeError('Backend exited; inspect owned server.log')
                await asyncio.sleep(2)
        async def measure():
            connections = max(32, getattr(args,'offline_concurrency',8)+8)
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800,connect=5), trust_env=False,
                    limits=httpx.Limits(max_connections=connections,max_keepalive_connections=connections)) as client:
                for _ in range(600):
                    try:
                        response = await client.get(f"http://127.0.0.1:{profile['port']}/health", timeout=2)
                        if response.status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError('Backend startup timed out')
                # Equal warmup, fixed sessions reused so preserve-history mode
                # does not require LRU eviction merely to run the test harness.
                await asyncio.gather(*(call(client, profile, fixture['prompt'][:1024], f'coordination-{i}',128)
                                       for i in range(8)))
                results = {}
                for case in args.cases:
                    save(output/'status.json', dict(status='running', pair=pair, arm=arm, case=case,
                         server_pid=server.pid, at=time.time()))
                    results[case] = await workload(client, profile, fixture, case, directory/case, args)
                return results
        try:
            async with asyncio.TaskGroup() as group:
                monitor = group.create_task(guard())
                task = group.create_task(measure())
                results = await task
                monitor.cancel()
            save(directory/'results.json', results)
            save(directory/'diagnostics.json', collect_diagnostics(directory/'server'))
            return results
        finally:
            stop_owned(server)


async def main(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('pair-*')):
        raise ValueError('Use a fresh output directory; existing measurements are immutable')
    # Only execution needs the cooperative GPU ownership lock. CPU-only profile
    # preparation must remain possible while an immutable older run is queued.
    # Foreign test ownership is checked independently through NVML above.
    import fcntl
    execution_lock = (nullcontext() if args.prepare_only else
                      open('/tmp/kvmem-coordination-qualification.lock', 'w'))
    with execution_lock as lock:
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        profiles = make_profiles(args.config.resolve(), output, arms=args.arms,
                                 step_latency_interval=args.step_latency_interval)
        if not (output/'fixture.json').exists():
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(profiles['B0']['tokenizer'], local_files_only=True)
            text = ''.join(f'Record {i}: worker {i%8} observed value {(i*104729)%1000003}. Keep the evidence for analysis.\n'
                           for i in range(20000))
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) < max(32768,args.long_tokens):
                raise ValueError('Synthetic input is too short')
            fixture = dict(prompt=ids[:max(32768,args.long_tokens)], tool=ids[1000:9192], eos=tokenizer.eos_token_id)
            save(output/'fixture.json', fixture)
        fixture = json.loads((output/'fixture.json').read_text())
        offline_dataset = None
        if 'offline' in args.cases:
            if 'B4' in args.arms:
                raise ValueError('Offline throughput comparison uses B0/B3; B4 enables a latency controller')
            offline_dataset = offline_support().prepare(args,fixture,profiles['B0'],output)
        save(output/'manifest.json', dict(config=str(args.config.resolve()), pairs=args.pairs,
             arms=args.arms, cases=args.cases, fixture_sha256=digest(fixture),
             effective_step_latency_interval=profiles['B0']['kvmem']['step_latency_interval'],
             workload={k:v for k,v in vars(args).items() if not isinstance(v,Path)},
             source_sha256=source_hashes(),
             offline_dataset=offline_dataset,
             scope='Offline dataset or synthetic scheduling qualification; no DeepSWE task resumed'))
        if args.prepare_only:
            save(output/'status.json', dict(status='prepared', at=time.time()))
            return
        results=[]
        try:
            for pair in range(args.pairs):
                order = args.arms if pair%2 == 0 else list(reversed(args.arms))
                for arm in order:
                    measured = await run_arm(args, profiles[arm], fixture, output, arm, pair)
                    results.append(dict(pair=pair, arm=arm, results=measured))
                    save(output/'measurements.json', results)
            save(output/'comparison.json', compare(results))
            save(output/'status.json', dict(status='completed', at=time.time(), measurements=len(results)))
        except BaseException as error:
            save(output/'status.json', dict(status='interrupted' if isinstance(error,asyncio.CancelledError) else 'failed',
                 at=time.time(), error=str(error)))
            raise


def compare(measurements):
    """Paired ratios, with resampling over pairs rather than individual tokens."""
    by_arm = {(r['pair'],r['arm']):r['results'] for r in measurements}
    comparison = {}
    rng = random.Random(104729)
    for arm in ARMS[1:]:
        pairs = sorted(i for i,a in by_arm if a == arm and (i,'B0') in by_arm)
        if not pairs:
            continue
        for case in by_arm[pairs[0],arm]:
            if case == 'offline' and any(by_arm[i,'B0'][case]['output_tokens_per_second']<=0 for i in pairs):
                comparison[f'{arm}/{case}'] = dict(pairs=len(pairs),
                    passes_offline_throughput_gate=False,
                    error='A baseline batch produced no successful output; ratio is undefined',
                    failed_requests={a:sum(by_arm[i,a][case]['failed_requests'] for i in pairs)
                                     for a in ('B0',arm)})
                continue
            ratios = [by_arm[i,arm][case]['output_tokens_per_second'] /
                      by_arm[i,'B0'][case]['output_tokens_per_second'] for i in pairs]
            samples = [statistics.median(rng.choices(ratios,k=len(ratios))) for _ in range(5000)]
            row = dict(pairs=len(pairs), throughput_ratio_median=statistics.median(ratios),
                paired_ratio_ci95=([percentile(samples,.025),percentile(samples,.975)]
                                   if len(pairs)>1 else None))
            if case == 'offline':
                row['failed_requests'] = {a:sum(by_arm[i,a][case]['failed_requests'] for i in pairs)
                                          for a in ('B0',arm)}
                steady = [by_arm[i,arm][case]['steady_backlog_window']['output_tokens_per_second'] /
                          by_arm[i,'B0'][case]['steady_backlog_window']['output_tokens_per_second']
                          for i in pairs if by_arm[i,arm][case]['steady_backlog_window'] and
                          by_arm[i,'B0'][case]['steady_backlog_window'] and
                          by_arm[i,'B0'][case]['steady_backlog_window']['output_tokens_per_second']>0]
                row['steady_backlog_ratio_median'] = statistics.median(steady) if steady else None
                row['passes_offline_throughput_gate'] = (len(pairs)>=5 and
                    not any(row['failed_requests'].values()) and row['paired_ratio_ci95'][0]>1.)
                comparison[f'{arm}/{case}'] = row
                continue
            if case != 'prefill':
                gap_ratios = [by_arm[i,arm][case]['output_event_gap_p95_seconds'] /
                              by_arm[i,'B0'][case]['output_event_gap_p95_seconds'] for i in pairs]
                row['event_gap_p95_ratio_median'] = statistics.median(gap_ratios)
            if case == 'decode':
                common = [by_arm[i,arm][case]['common_decode']['tokens_per_second'] /
                          by_arm[i,'B0'][case]['common_decode']['tokens_per_second'] for i in pairs
                          if by_arm[i,arm][case]['common_decode'] and by_arm[i,'B0'][case]['common_decode']]
                row['common_decode_ratio_median'] = statistics.median(common) if common else None
                row['passes_regression_gate'] = bool(len(common) >= 5 and statistics.median(common) >= .98)
            elif case != 'prefill':
                row['passes_proposed_gate'] = (len(pairs) >= 5 and statistics.median(ratios) >= 1.10 and
                    row['paired_ratio_ci95'][0] > 1. and row['event_gap_p95_ratio_median'] <= 1.05)
            comparison[f'{arm}/{case}'] = row
    return comparison


async def entry(args):
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    await main(args)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--pairs', type=int, default=5)
    p.add_argument('--arms', nargs='+', choices=ARMS, default=['B0','B2'])
    p.add_argument('--step-latency-interval', type=int, default=0,
                   help='Equal timing sampling in every arm; 0 means off for v1, every 4 steps for v2')
    p.add_argument('--cases', nargs='+', choices=('decode','prefill','mixed','agents','burst','offline'),
                   default=['decode','prefill','mixed','agents','burst'])
    p.add_argument('--idle-seconds', type=float, default=120.)
    p.add_argument('--long-tokens', type=int, choices=(65536,131072,262144), default=65536)
    p.add_argument('--decode-tokens', type=int, default=4096)
    p.add_argument('--mixed-decode-tokens', type=int, default=8192)
    p.add_argument('--agent-tokens', type=int, default=2048)
    p.add_argument('--turns', type=int, default=4)
    p.add_argument('--offline-dataset', help='JSONL with id, prompt_token_ids or prompt text, max_tokens, optional query_span/ignore_eos')
    p.add_argument('--offline-requests', type=int, default=64)
    p.add_argument('--offline-concurrency', type=int, default=16, help='Outstanding HTTP requests; native GPU concurrency remains in the profile')
    p.add_argument('--offline-input-lengths', nargs='+',type=int,default=[8192,32768,65536])
    p.add_argument('--offline-output-tokens',type=int,default=1024)
    p.add_argument('--offline-warmup-seconds',type=float,default=30.)
    p.add_argument('--offline-sample-seconds',type=float,default=1.)
    args=p.parse_args()
    if args.pairs < 1 or args.idle_seconds < 120 or args.turns < 2:
        p.error('Need positive pairs, >= 120 seconds idle and >= 2 agent turns')
    if not 0 <= args.step_latency_interval <= 1024:
        p.error('Step latency interval must be in [0, 1024]')
    asyncio.run(entry(args))
