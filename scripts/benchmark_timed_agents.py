"""Frozen, fixed-duration A0/P1 measurements with continuous agent ingress."""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


q = module('timed_coordination', ROOT / 'scripts/qualify_coordination.py')


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.suffix != '.pyc'}


def resolve_profile(profile, base, run_dir):
    from vllm_kvmem.serve import PATHS, build_launch
    plan = build_launch(profile, base, run_dir)
    resolved = json.loads(json.dumps(profile))
    # build_launch also converts public settings to native internal controls
    # (e.g. threshold=0). Those are not a round-trippable public profile.
    for name in PATHS:
        if name in resolved and resolved[name] is not None:
            resolved[name] = plan['profile'][name]
    return resolved


def prepare_fixture(fixture, args, profile, tokenizer=None):
    """Preserve trace inputs verbatim; only supply the cleanup EOS token."""
    fixture = json.loads(json.dumps(fixture))
    trace = 'agent_plans' in fixture
    if trace:
        workload = module('timed_prepare_workload', ROOT / 'scripts/timed_agent_workload.py')
        workload.validate_agent_plans(fixture, concurrency=args.concurrency,
            warmup_turns=args.warmup_turns, epoch_turns=args.epoch_turns,
            gen_budget=profile['kvmem']['gen_budget'])
    elif (len(fixture['prompt']) < args.context_tokens or
            len(fixture['tool']) < max(fixture['tool_sizes'])):
        raise ValueError('Frozen fixture is too short')
    if 'eos' not in fixture or (not trace and 'user_prefixes' not in fixture):
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(profile['tokenizer'], local_files_only=True)
        if 'eos' not in fixture:
            fixture['eos'] = tokenizer.eos_token_id
        if not trace and 'user_prefixes' not in fixture:
            fixture['user_prefixes'] = [tokenizer.encode(
                f'Unique tenant {i}: evidence identifier USER_{i}.\n', add_special_tokens=False)
                for i in range(args.concurrency)]
    if type(fixture['eos']) is not int or fixture['eos'] < 0:
        raise ValueError('Fixture requires a single nonnegative cleanup EOS token')
    return fixture


def prepare(args):
    from vllm_kvmem.serve import build_launch
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = read(args.config.resolve())
    if (base['max_num_seqs'] != args.concurrency or args.concurrency not in (4, 8) or
            base['model_format'] != 'safetensors' or base['draft_method'] != 'mtp' or base['draft_tokens'] != 4 or
            base['kvmem']['select_budget'] != 131072 or base['kvmem']['gen_budget'] != 131072 or
            base['kvmem']['archive_backend'] != 'cpu' or
            base['kvmem']['cpu_read_cache_bytes'] != 100_000_000_000 or
            not base['kvmem']['cpu_history_coordination'] or not base['kvmem']['async_restore']):
        raise ValueError('Require a four- or eight-user K128K/G128K CPU-history profile with matching native slots')
    for arm in ('A0', 'P1'):
        profile = json.loads(json.dumps(base))
        profile['port'] = args.port
        profile['verify_session_restore'] = False
        profile['kvmem'].update(offline_agent_policy=arm == 'P1', active_resident_ttl_ms=0.,
            active_resident_pressure_only=arm == 'P1', active_resident_max_fraction=.5,
            adaptive_prefill=False, prefill_batch_tokens=0, maintenance_transfer_bytes_per_second=0,
            step_latency_interval=4)
        resolved = resolve_profile(profile, args.config.resolve().parent, root / 'validation' / arm)
        build_launch(resolved, root, root / 'validation-roundtrip' / arm)
        q.save(root / (arm + '.json'), resolved)
    fixture = prepare_fixture(read(args.fixture.resolve()), args, read(root / 'P1.json'))
    q.save(root / 'fixture.json', fixture)
    frozen = root / 'frozen'
    for directory in ('src', 'scripts', 'csrc'):
        if (ROOT / directory).exists():
            shutil.copytree(ROOT / directory, frozen / directory,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    q.save(root / 'manifest.json', dict(created_at=time.time(), source_sha256=hashes(frozen),
        fixture_sha256=q.digest(fixture), base_config=str(args.config.resolve()),
        profile_sha256={arm:hashlib.sha256((root / (arm + '.json')).read_bytes()).hexdigest()
                        for arm in ('A0', 'P1')},
        workload={k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        protocol='fixed work warmup -> 300-second continuous window -> excluded drain/cleanup',
        scope=(fixture.get('scope', 'External-message performance trace; not task quality') if 'agent_plans' in fixture else
               'Controlled multi-turn performance workload; not original DeepSWE task quality'),
        fixture_format='agent_plans' if 'agent_plans' in fixture else 'synthetic',
        fixture_source=str(args.fixture.resolve())))
    q.save(root / 'status.json', dict(status='prepared', at=time.time(), gpu_started=False))


def distribution(values):
    return dict(count=len(values), mean=statistics.mean(values) if values else None,
        p50=q.percentile(values, .5), p95=q.percentile(values, .95),
        maximum=max(values, default=None))


def parse_native_metrics(body):
    wanted = {'vllm:generation_tokens_total', 'vllm:num_preemptions_total',
              'vllm:num_requests_running', 'vllm:num_requests_waiting'}
    values = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in wanted:
            if name in values:
                raise ValueError('Require exactly one native engine metric series')
            value = float(line.rsplit(' ', 1)[1])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid native metric value')
            values[name] = value
    if not {'vllm:generation_tokens_total', 'vllm:num_preemptions_total'} <= values.keys():
        raise ValueError('Missing native accepted-output/preemption counters')
    return values


def native_boundaries(rows, begin, end):
    valid = [r for r in rows if 'values' in r]
    if not valid:
        return dict(available=False)
    before = max((r for r in valid if r['finished_at'] <= begin),
                 key=lambda r:r['finished_at'], default=None)
    after = min((r for r in valid if r['started_at'] >= end),
                key=lambda r:r['started_at'], default=None)
    if before is None or after is None:
        return dict(available=False, reason='Missing bracketing native snapshots')
    names = ('vllm:generation_tokens_total', 'vllm:num_preemptions_total')
    if any(after['values'][n] < before['values'][n] for n in names):
        raise ValueError('Native counter reset during fixed window')
    return dict(available=True, before=before, after=after,
        deltas={n:after['values'][n]-before['values'][n] for n in names},
        scope='Outer bracketing native interval includes boundary margins and cleanup outputs; '
              'primary effective output uses exact client event window, not this counter difference.')


def counter_bracket(rows, begin, end):
    """Use cumulative completion observations without inventing GPU timestamps."""
    def at(row):
        return row.get('observed_at', row.get('at', row.get('completed_observed_at')))
    valid = [r for r in rows if isinstance(at(r), (int, float))]
    return dict(before_begin=max((r for r in valid if at(r) <= begin), key=at, default=None),
        after_begin=min((r for r in valid if at(r) >= begin), key=at, default=None),
        before_end=max((r for r in valid if at(r) <= end), key=at, default=None),
        after_end=min((r for r in valid if at(r) >= end), key=at, default=None))


def public_work_summary(rows, begin, end):
    if not rows:
        raise ValueError('Missing unsampled public work counters')
    if len({r['counter_id'] for r in rows}) != 1:
        raise ValueError('Public work counter identity changed')
    for a, b in zip(rows, rows[1:]):
        if b['at'] < a['at'] or any(b[mode]['public_prefill_tokens'] < a[mode]['public_prefill_tokens']
                                  for mode in ('submitted', 'completed')):
            raise ValueError('Public work clock/counter reset')
    bounds = counter_bracket(rows, begin, end)
    before, after, last, trailing = [bounds[k] for k in
        ('before_begin', 'after_begin', 'before_end', 'after_end')]
    result = dict(boundary_snapshots=bounds, complete_brackets=all(r is not None for r in bounds.values()),
        scope='Actual public prompt positions forwarded; restored KV and private Query excluded. '
              'Submission times and asynchronous completion observations have distinct boundaries.')
    if result['complete_brackets']:
        # For t inside two snapshots, completed(left) <= C(t) <= submitted(right).
        low = max(0, last['completed']['public_prefill_tokens'] - after['submitted']['public_prefill_tokens'])
        high = max(0, trailing['submitted']['public_prefill_tokens'] - before['completed']['public_prefill_tokens'])
        result['completed_prefill_tokens_in_window_bounds'] = [low, high]
        result['completed_prefill_tokens_per_second_bounds'] = [low/(end-begin), high/(end-begin)]
    if after is not None and last is not None and last['at'] > after['at']:
        seconds = last['at']-after['at']
        delta = {k:last['submitted'][k]-after['submitted'][k] for k in
            ('public_batches','public_prefill_tokens','public_non_prefill_tokens',
             'prefill_request_batches','decode_request_batches')}
        result['submitted_inner_window'] = dict(begin=after['at'], end=last['at'], seconds=seconds,
            counters=delta, prefill_tokens_per_second=delta['public_prefill_tokens']/seconds,
            decode_requests_per_batch=(delta['decode_request_batches']/delta['public_batches']
                                      if delta['public_batches'] else None),
            prefill_requests_per_batch=(delta['prefill_request_batches']/delta['public_batches']
                                       if delta['public_batches'] else None))
    return result


def gpu_summary(samples, begin, end):
    intervals = []
    for left, right in zip(samples, samples[1:]):
        start, stop = max(begin, left['at']), min(end, right['at'])
        # Do not silently extend a stale observation over a sampling outage.
        if stop > start and right['at'] - left['at'] <= 3.:
            intervals.append((stop - start, left['utilization']))
    covered = sum(n for n, _ in intervals)
    return dict(time_weighted_utilization_percent=sum(n * u for n, u in intervals) / covered if covered else None,
        observed_seconds=covered, coverage_fraction=covered / (end - begin),
        utilization_below_ten_percent_fraction=sum(n for n, u in intervals if u < 10) / covered if covered else None,
        samples=sum(begin <= r['at'] < end for r in samples),
        maximum_sample_gap_seconds=max((b['at']-a['at'] for a,b in zip(samples,samples[1:])
            if b['at'] >= begin and a['at'] <= end),default=None),
        scope='One-second NVML utilization observations; not SM occupancy')


def runtime_directory(root, arm):
    """A runtime comparison keeps its client uniform and freezes both servers."""
    manifest = read(root / 'manifest.json')
    if 'runtime_sha256' not in manifest:
        return root / 'frozen'
    expected = manifest['runtime_sha256']
    if set(expected) != {'A0', 'P1'} or arm not in expected:
        raise ValueError('Runtime comparison requires both frozen server sources')
    source = root / 'runtime' / arm
    if hashes(source) != expected[arm] or not (source / 'src/vllm_kvmem').is_dir():
        raise RuntimeError('Frozen server runtime changed or is incomplete')
    return source


async def run_arm(root, profile, fixture, arm, pair, args):
    directory = root / f'pair-{pair:02d}-{arm}'
    directory.mkdir()
    await q.wait_idle(root, args.idle_seconds)
    if hashes(root / 'frozen') != read(root / 'manifest.json')['source_sha256']:
        raise RuntimeError('Frozen source changed; never mutate a running experiment')
    runtime = runtime_directory(root, arm)
    if q.gpu_state()['pids']:
        raise RuntimeError('GPU ownership changed before launch')
    environment = {**os.environ, 'PYTHONPATH': str(runtime / 'src'), 'TOKENIZERS_PARALLELISM': 'false'}
    gpu_rows = []
    native_rows = []
    workload_module = module('timed_agent_workload', ROOT / 'scripts/timed_agent_workload.py')
    with (directory / 'server.log').open('w') as log:
        server = subprocess.Popen([sys.executable, '-m', 'vllm_kvmem', 'serve', '--config', str(root / (arm + '.json')),
            '--run-dir', str(directory / 'server')], cwd=runtime, env=environment, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True)
        q.save(root / 'status.json', dict(status='loading', pair=pair, arm=arm, server_pid=server.pid, at=time.time()))

        async def guard():
            while True:
                state = await asyncio.to_thread(q.gpu_state)
                row = dict(at=time.perf_counter(), wall_at=time.time(), **state)
                gpu_rows.append(row)
                with (directory / 'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                foreign = set(state['pids']) - q.owned_pids(server)
                if foreign:
                    raise RuntimeError('Foreign GPU work appeared; stopping this owned arm only')
                if server.poll() is not None:
                    raise RuntimeError('Backend exited; inspect preserved server.log')
                await asyncio.sleep(1.)

        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=5), trust_env=False,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32)) as client:
                for _ in range(600):
                    try:
                        if (await client.get(f"http://127.0.0.1:{profile['port']}/health", timeout=2)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError('Backend startup timed out')
                async def telemetry():
                    while True:
                        row = dict(started_at=time.perf_counter())
                        try:
                            response = await client.get(f"http://127.0.0.1:{profile['port']}/metrics", timeout=5)
                            response.raise_for_status()
                            row['values'] = parse_native_metrics(response.text)
                        except (httpx.HTTPError, ValueError) as error:
                            row['error'] = str(error)
                        row['finished_at'] = time.perf_counter()
                        native_rows.append(row)
                        with (directory / 'native_samples.jsonl').open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
                        await asyncio.sleep(1.)

                telemetry_task = asyncio.create_task(telemetry())
                results = {}
                try:
                    for mode in args.modes:
                        q.save(root / 'status.json', dict(status='running', pair=pair, arm=arm, mode=mode,
                            server_pid=server.pid, at=time.time()))
                        case_args = argparse.Namespace(**vars(args))
                        case_args.mode = mode
                        case_args.run_tag = f'timed-{pair}-{arm}-{mode}'
                        case_args.capture_directory = directory / 'server/capture'
                        result = await workload_module.workload(client, profile, fixture, directory / mode, case_args)
                        window = read(directory / mode / 'window.json')
                        begin, end = window['begin'], window['end']
                        # Bracket T1 without extending the primary client interval.
                        while not any(r.get('started_at', -1) >= end for r in native_rows):
                            await asyncio.sleep(.05)
                        # Every stream has now drained and its usage was verified.
                        # Native cumulative output includes warmup and cleanup too.
                        expected = sum(v['total_stream_output_tokens'] for v in results.values()) + result['total_stream_output_tokens']
                        deadline = time.monotonic()+20.
                        while time.monotonic() < deadline:
                            valid = [r for r in native_rows if 'values' in r]
                            if valid and valid[-1]['values']['vllm:generation_tokens_total'] >= expected:
                                break
                            await asyncio.sleep(.05)
                        actual = valid[-1]['values']['vllm:generation_tokens_total'] if valid else None
                        if actual != expected:
                            raise RuntimeError(f'Native/client accepted-output counts differ: {actual}/{expected}')
                        result['native_client_total_output_reconciliation'] = dict(passed=True,
                            client_tokens=expected, native_tokens=actual,
                            scope='All streams since fresh engine start, including warmup, cleanup and drain')
                        state = await asyncio.to_thread(q.gpu_state)
                        gpu_rows.append(dict(at=time.perf_counter(), wall_at=time.time(), **state))
                        result['gpu'] = gpu_summary(gpu_rows, begin, end)
                        result['native_counter_boundaries'] = native_boundaries(native_rows, begin, end)
                        capture = directory / 'server/capture'
                        if (capture / 'public_work.jsonl').exists():
                            work = [json.loads(line) for line in (capture / 'public_work.jsonl').read_text().splitlines()]
                            result['public_work'] = public_work_summary(work, begin, end)
                        else:
                            raise RuntimeError('Missing unsampled public work evidence')
                        from analyze_agent_scheduling import analyze
                        q.save(directory / mode / 'scheduling-analysis.json',
                            analyze(directory / 'server', begin=begin, end=end))
                        archive = read(capture / 'archive.json')
                        if archive.get('disk_read_bytes', 0) or archive.get('disk_write_bytes', 0):
                            raise RuntimeError('CPU-only workload unexpectedly used KV disk I/O')
                        result['disk_read_bytes'] = archive.get('disk_read_bytes', 0)
                        result['disk_write_bytes'] = archive.get('disk_write_bytes', 0)
                        q.save(directory / mode / 'results.json', result)
                        results[mode] = result
                    return results
                finally:
                    telemetry_task.cancel()
                    await asyncio.gather(telemetry_task, return_exceptions=True)
        try:
            async with asyncio.TaskGroup() as group:
                monitor = group.create_task(guard())
                task = group.create_task(measure())
                result = await task
                monitor.cancel()
            q.save(directory / 'results.json', result)
            return result
        finally:
            await asyncio.to_thread(q.stop_owned, server)


def compare(measurements):
    by_key = {(r['pair'], r['arm']): r['results'] for r in measurements}
    results = {}
    pairs = sorted(i for i, arm in by_key if arm == 'A0' and (i, 'P1') in by_key)
    if not pairs:
        return results
    for mode in by_key[pairs[0], 'A0']:
        ratios = []
        for pair in pairs:
            baseline, candidate = [by_key[pair, arm][mode] for arm in ('A0', 'P1')]
            for row in (baseline, candidate):
                if (row.get('status') != 'completed' or not row.get('drained') or
                        not row.get('native_client_total_output_reconciliation', {}).get('passed')):
                    raise ValueError('Failed or unreconciled workload cannot become a throughput win')
            if baseline['output_tokens_per_second'] <= 0:
                raise ValueError('No baseline output; throughput ratio is undefined')
            ratios.append(candidate['output_tokens_per_second'] / baseline['output_tokens_per_second'])
        results[mode] = dict(pairs=len(pairs), throughput_ratios=ratios,
            throughput_ratio_median=statistics.median(ratios),
            preliminary=len(pairs) < 3,
            passes_throughput_target=len(pairs) >= 3 and all(r > 1 for r in ratios) and statistics.median(ratios) >= 1.10,
            scope='Fixed-duration accepted output throughput; correctness and starvation are separate required gates')
    return results


async def run(args):
    root = args.output.resolve()
    if ROOT.resolve() != (root / 'frozen').resolve():
        raise ValueError('Execute output/frozen/scripts/benchmark_timed_agents.py, not the live checkout')
    manifest = read(root / 'manifest.json')
    if read(root / 'status.json').get('status') != 'prepared':
        raise ValueError('Use a fresh prepared experiment; no automatic restarts')
    active = asyncio.current_task()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, active.cancel)
    saved = manifest['workload']
    # Execute exactly the prepared workload even when CLI defaults later change.
    for key, value in saved.items():
        if key not in ('prepare_only', 'run_prepared'):
            setattr(args, key, value)
    profiles = {arm: read(root / (arm + '.json')) for arm in ('A0', 'P1')}
    fixture = read(root / 'fixture.json')
    if (q.digest(fixture) != manifest['fixture_sha256'] or any(
            hashlib.sha256((root / (arm + '.json')).read_bytes()).hexdigest() != manifest['profile_sha256'][arm]
            for arm in profiles)):
        raise RuntimeError('Prepared profile or fixture changed')
    measurements = []
    with open('/tmp/kvmem-coordination-qualification.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for pair in range(args.pairs):
                for arm in (('A0', 'P1') if pair % 2 == 0 else ('P1', 'A0')):
                    results = await run_arm(root, profiles[arm], fixture, arm, pair, args)
                    measurements.append(dict(pair=pair, arm=arm, results=results))
                    q.save(root / 'measurements.json', measurements)
                    q.save(root / 'comparison.json', compare(measurements))
            q.save(root / 'status.json', dict(status='completed_pending_review', at=time.time()))
        except BaseException as error:
            q.save(root / 'status.json', dict(status='interrupted' if isinstance(error, asyncio.CancelledError)
                else 'failed', error=repr(error), at=time.time()))
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--run-prepared', action='store_true')
    p.add_argument('--duration-seconds', type=float, default=300.)
    p.add_argument('--pairs', type=int, default=1)
    p.add_argument('--modes', nargs='+', choices=('saturated', 'tools'), default=['saturated'])
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--warmup-turns', type=int, default=2)
    p.add_argument('--output-tokens', type=int, default=1024)
    p.add_argument('--context-tokens', type=int, default=141870)
    p.add_argument('--epoch-turns', type=int, default=8)
    p.add_argument('--idle-seconds', type=float, default=15.)
    p.add_argument('--port', type=int, default=18483)
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or args.pairs < 1 or
            not math.isfinite(args.idle_seconds) or args.idle_seconds < 0 or
            args.warmup_turns < 1 or args.epoch_turns <= args.warmup_turns or
            args.output_tokens < 1 or args.context_tokens < 1024 or len(set(args.modes)) != len(args.modes)):
        raise ValueError('Invalid fixed window or pair count')
    if args.run_prepared and args.prepare_only:
        raise ValueError('Choose preparation or frozen execution')
    if not args.run_prepared:
        if not args.config or not args.fixture:
            raise ValueError('Preparation requires a base profile and frozen fixture')
        prepare(args)
    if not args.prepare_only:
        if not args.run_prepared:
            raise ValueError('Run the frozen prepared script with --run-prepared')
        asyncio.run(run(args))
