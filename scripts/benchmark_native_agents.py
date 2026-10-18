"""Pure vLLM agent replay, with an immutable recent KVMem reference.

The server is the installed native CLI with plugins disabled. Only the client
reuses the fixed-window replay/accepted-token accounting. Prefix caching is
enabled within an epoch; a new epoch gets a fresh cache namespace.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


bench = module('native_agent_shared_bench', ROOT/'scripts/benchmark_timed_agents.py')
launch = module('native_agent_launch', ROOT/'scripts/native_agent_launch.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(body):
    """Keep single-engine counters; distinguish the labeled prompt sources."""
    wanted = {'vllm:generation_tokens_total', 'vllm:num_preemptions_total',
              'vllm:num_requests_running', 'vllm:num_requests_waiting',
              'vllm:prompt_tokens_total', 'vllm:prompt_tokens_cached_total',
              'vllm:prompt_tokens_by_source_total',
              'vllm:prefix_cache_hits_total', 'vllm:prefix_cache_queries_total'}
    result = {}
    for line in body.splitlines():
        if not line or line.startswith('#'):
            continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name not in wanted:
            continue
        key = name
        if name == 'vllm:prompt_tokens_by_source_total':
            source = re.search(r'(?:\{|,)source="([^"]+)"', line)
            if source is None:
                raise ValueError('Prompt source label missing')
            key += ':'+source.group(1)
        if key in result:
            raise ValueError('Require exactly one native engine metric series')
        value = float(line.rsplit(' ', 1)[1])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid native metric value')
        result[key] = value
    if not {'vllm:generation_tokens_total', 'vllm:num_preemptions_total',
            'vllm:num_requests_running', 'vllm:num_requests_waiting'} <= result.keys():
        raise ValueError('Missing required native counters')
    return result


def prepare(args):
    reference = args.reference.resolve()
    profile = bench.read(reference/'A0.json')
    source_manifest = bench.read(reference/'manifest.json')
    fixture = bench.read(reference/'fixture.json')
    if (profile.get('max_num_seqs') != 8 or
            source_manifest.get('workload', {}).get('concurrency') != 8 or
            len(fixture.get('agent_plans', [])) != 8):
        raise ValueError('Require the eight-user KVMem reference fixture/profile')
    if profile['kvmem'].get('offline_prefill_coalescing'):
        raise ValueError('Reference must use the current default without coalescing')
    workload = module('native_prepare_workload', ROOT/'scripts/timed_agent_workload.py')
    workload.validate_agent_plans(fixture, concurrency=8, warmup_turns=2,
                                 epoch_turns=16, gen_budget=131072)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    bench.q.save(root/'profile.json', profile)
    bench.q.save(root/'fixture.json', fixture)
    for directory in ('src', 'scripts', 'csrc'):
        shutil.copytree(ROOT/directory, root/'frozen'/directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    reference_files = ['A0.json', 'fixture.json', 'manifest.json', 'review.json',
                       'pair-00-A0/saturated/results.json',
                       'pair-00-A0/saturated/window.json',
                       'pair-00-A0/saturated/events.jsonl']
    reference_hashes = {str(reference/p): sha(reference/p) for p in reference_files}
    bench.q.save(root/'reference-results.json', bench.read(reference/'pair-00-A0/saturated/results.json'))
    bench.q.save(root/'manifest.json', dict(
        comparison_kind='pure_native_vllm_agents', created_at=time.time(),
        reference=str(reference), reference_sha256=reference_hashes,
        source_sha256=bench.hashes(root/'frozen'),
        profile_sha256=sha(root/'profile.json'), fixture_sha256=bench.q.digest(fixture),
        workload=dict(concurrency=8, warmup_turns=2, epoch_turns=16,
                      duration_seconds=args.duration_seconds, mode='saturated'),
        execution=dict(arms=args.arms, port=args.port, idle_seconds=args.idle_seconds),
        scope='Native sync and async deployment comparisons against a recent KVMem reference; '
              'full prompt plus native prefix cache, not an isolated capture-overhead measurement.',
        cache_scope='Within-epoch prefix reuse; each agent epoch has a fresh cache_salt, '
                    'matching the KVMem replay logical EOS reset. No cross-epoch seed reuse.'))
    bench.q.save(root/'status.json', dict(status='prepared', at=time.time()))


def verify(root):
    m = bench.read(root/'manifest.json')
    if ROOT.resolve() != (root/'frozen').resolve():
        raise ValueError('Execute this experiment frozen script')
    if m.get('comparison_kind') != 'pure_native_vllm_agents':
        raise ValueError('Not a prepared native comparison')
    if (bench.hashes(root/'frozen') != m['source_sha256'] or
            sha(root/'profile.json') != m['profile_sha256'] or
            bench.q.digest(bench.read(root/'fixture.json')) != m['fixture_sha256'] or
            any(sha(Path(p)) != h for p, h in m['reference_sha256'].items())):
        raise ValueError('Frozen experiment or reference changed')
    return m


async def run_arm(root, arm, manifest):
    await bench.q.wait_idle(root, manifest['execution']['idle_seconds'])
    verify(root)
    if bench.q.gpu_state()['pids']:
        raise RuntimeError('GPU ownership changed before native launch')
    directory = root/arm
    directory.mkdir()
    profile = bench.read(root/'profile.json')
    plan = launch.build_launch(profile, port=manifest['execution']['port'],
                               async_scheduling=arm == 'native-async')
    bench.q.save(directory/'native-launch.json', {k: v for k, v in plan.items() if k != 'environment'})
    gpu_rows, native_rows = [], []
    workload = module('native_timed_workload', ROOT/'scripts/timed_agent_workload.py')
    with (directory/'server.log').open('w') as log:
        server = subprocess.Popen(plan['argv'], cwd=directory, env=plan['environment'],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        bench.q.save(root/'status.json', dict(status='loading', arm=arm, server_pid=server.pid, at=time.time()))

        async def guard():
            while True:
                state = await asyncio.to_thread(bench.q.gpu_state)
                row = dict(at=time.monotonic(), wall_at=time.time(), **state)
                gpu_rows.append(row)
                with (directory/'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                if set(state['pids']) - bench.q.owned_pids(server):
                    raise RuntimeError('Foreign GPU work appeared; stopping owned native arm only')
                if server.poll() is not None:
                    raise RuntimeError('Native server exited; inspect preserved server.log')
                await asyncio.sleep(1)

        async def measure():
            port = manifest['execution']['port']
            async with httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=5), trust_env=False,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32)) as client:
                for _ in range(600):
                    try:
                        if (await client.get(f'http://127.0.0.1:{port}/health', timeout=2)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1)
                else:
                    raise TimeoutError('Native startup timed out')

                async def snapshot():
                    row = dict(started_at=time.monotonic())
                    response = await client.get(f'http://127.0.0.1:{port}/metrics', timeout=5)
                    response.raise_for_status()
                    row.update(values=metrics(response.text), finished_at=time.monotonic())
                    native_rows.append(row)
                    with (directory/'native_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    return row

                async def quiescent(*, stage, not_before):
                    deadline = time.monotonic()+30
                    while time.monotonic() < deadline:
                        row = await snapshot()
                        values = row['values']
                        if (row['started_at'] >= not_before and
                                values['vllm:num_requests_running'] == 0 and
                                values['vllm:num_requests_waiting'] == 0):
                            return dict(stage=stage, native_metrics=row,
                                        scope='Fresh native running/waiting gauges; no KVMem session-resource claim')
                        await asyncio.sleep(.25)
                    raise TimeoutError('Native engine did not become quiescent')

                await snapshot()
                async def telemetry():
                    while True:
                        try:
                            await snapshot()
                        except (httpx.HTTPError, ValueError) as error:
                            with (directory/'telemetry-errors.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=time.time(), error=str(error)))+'\n')
                        await asyncio.sleep(1)

                monitor = asyncio.create_task(telemetry())
                try:
                    bench.q.save(root/'status.json', dict(status='running', arm=arm, server_pid=server.pid, at=time.time()))
                    settings = argparse.Namespace(**manifest['workload'], run_tag=arm,
                                                  native_quiescent=quiescent)
                    client_profile = dict(profile, port=port, native_vllm=True)
                    result = await workload.workload(client, client_profile, bench.read(root/'fixture.json'),
                                                     directory/'saturated', settings)
                    expected = result['total_stream_output_tokens']
                    deadline = time.monotonic()+20
                    while time.monotonic() < deadline:
                        last = await snapshot()
                        if last['values']['vllm:generation_tokens_total'] >= expected:
                            break
                        await asyncio.sleep(.1)
                    actual = last['values']['vllm:generation_tokens_total']
                    if actual != expected:
                        raise RuntimeError(f'Native accepted output mismatch: {actual}/{expected}')
                    result['native_client_total_output_reconciliation'] = dict(
                        passed=True, client_tokens=expected, native_tokens=actual,
                        scope='All streams including warmup, cleanup and drain')
                    window = bench.read(directory/'saturated/window.json')
                    begin, end = window['begin'], window['end']
                    result['gpu'] = bench.gpu_summary(gpu_rows, begin, end)
                    result['native_counter_boundaries'] = bench.native_boundaries(native_rows, begin, end)
                    result['native_prefill_counter_scope'] = (
                        'Prometheus prompt_tokens_by_source_total{source=local_compute} is reported at request prefill boundaries; '
                        'not the KVMem per-public-chunk completion-fence metric. Do not equate the two.')
                    result['max_native_running_requests'] = max(r['values']['vllm:num_requests_running'] for r in native_rows)
                    result['max_native_preemptions'] = max(r['values']['vllm:num_preemptions_total'] for r in native_rows)
                    bench.q.save(directory/'saturated/results.json', result)
                    bench.q.save(directory/'results.json', result)
                    return result
                finally:
                    monitor.cancel()
                    await asyncio.gather(monitor, return_exceptions=True)

        try:
            async with asyncio.TaskGroup() as tasks:
                monitor = tasks.create_task(guard())
                task = tasks.create_task(measure())
                result = await task
                monitor.cancel()
            return result
        finally:
            await asyncio.to_thread(bench.q.stop_owned, server)


async def run(args):
    root = args.output.resolve()
    manifest = verify(root)
    if bench.read(root/'status.json').get('status') != 'prepared':
        raise ValueError('Use a fresh prepared run; no automatic restart')
    loop = asyncio.get_running_loop()
    active = asyncio.current_task()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, active.cancel)
    reference = bench.read(root/'reference-results.json')
    comparison = dict(reference_kvmem_output_tps=reference['output_tokens_per_second'],
                      reference=str(manifest['reference']), arms={}, preliminary=True,
                      scope=manifest['scope'])
    with open('/tmp/kvmem-coordination-qualification.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for arm in manifest['execution']['arms']:
                result = await run_arm(root, arm, manifest)
                rate = result['output_tokens_per_second']
                comparison['arms'][arm] = dict(output_tps=rate,
                    kvmem_relative_output_difference=reference['output_tokens_per_second']/rate-1,
                    native_over_kvmem_ratio=rate/reference['output_tokens_per_second'])
                bench.q.save(root/'comparison.json', comparison)
            bench.q.save(root/'status.json', dict(status='completed_pending_review', at=time.time()))
        except BaseException as error:
            bench.q.save(root/'status.json', dict(status='interrupted' if isinstance(error, asyncio.CancelledError)
                                                 else 'failed', at=time.time(), error=repr(error)))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--duration-seconds', type=float, default=300.)
    parser.add_argument('--arms', nargs='+', choices=('native-sync', 'native-async'),
                        default=['native-sync', 'native-async'])
    parser.add_argument('--port', type=int, default=18487)
    parser.add_argument('--idle-seconds', type=float, default=15.)
    args = parser.parse_args()
    if args.run_prepared:
        if args.reference or args.prepare_only:
            parser.error('Frozen execution uses prepared inputs only')
        return asyncio.run(run(args))
    if not args.prepare_only or not args.reference:
        parser.error('Prepare explicitly using the recent KVMem reference')
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or
            not math.isfinite(args.idle_seconds) or args.idle_seconds < 0 or
            not 0 < args.port < 65536 or len(set(args.arms)) != len(args.arms)):
        parser.error('Invalid fixed window or execution settings')
    prepare(args)


if __name__ == '__main__':
    main()
