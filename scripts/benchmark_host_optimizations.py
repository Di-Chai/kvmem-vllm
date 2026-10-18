"""Ablate KVMem host dependencies with one frozen runtime and fixed clients.

A0 disables exactly three host optimizations; P1 enables them. Model, scoring,
budgets, scheduling policy and workload remain identical. The shared harness
owns the GPU lock, foreign-process guard, fixed windows and process cleanup.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('split_capture_dma', 'native_prefill_commit', 'async_checkpoint_capture')
spec = importlib.util.spec_from_file_location('host_ablation_benchmark',
    ROOT / 'scripts/benchmark_timed_agents.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def prepare(args):
    benchmark.prepare(args)
    root = args.output.resolve()
    common = benchmark.read(root / 'P1.json')
    common['kvmem']['step_latency_interval'] = 1
    for arm in ('A0', 'P1'):
        profile = json.loads(json.dumps(common))
        profile['kvmem'].update({name: arm == 'P1' for name in FEATURES})
        from vllm_kvmem.serve import build_launch
        build_launch(profile, root, root / 'validation-host' / arm)
        benchmark.q.save(root / (arm + '.json'), profile)
    manifest = benchmark.read(root / 'manifest.json')
    manifest.update(comparison_kind='host_dependency_ablation',
        feature_switches=list(FEATURES),
        profile_sha256={arm: hashlib.sha256((root / (arm + '.json')).read_bytes()).hexdigest()
                        for arm in ('A0', 'P1')},
        comparison_scope='One frozen Python/native runtime and identical profiles except three '
            'explicit host optimization switches. CPU timing is not GPU idle time. '
            'Closed-loop continuations and MTP acceptance can still differ.')
    benchmark.q.save(root / 'manifest.json', manifest)
    validate_profiles(root)


def validate_profiles(root):
    manifest = benchmark.read(root / 'manifest.json')
    if (manifest.get('comparison_kind') != 'host_dependency_ablation' or
            manifest.get('feature_switches') != list(FEATURES) or 'runtime_sha256' in manifest):
        raise ValueError('Require one runtime and the three explicit host ablations')
    profiles = []
    for arm in ('A0', 'P1'):
        profile = benchmark.read(root / (arm + '.json'))
        if hashlib.sha256((root / (arm + '.json')).read_bytes()).hexdigest() != manifest['profile_sha256'][arm]:
            raise ValueError('Host ablation profile changed after preparation')
        for name in FEATURES:
            if profile['kvmem'].pop(name, None) is not (arm == 'P1'):
                raise ValueError('Host ablation switch does not match its arm')
        profiles.append(profile)
    if profiles[0] != profiles[1]:
        raise ValueError('Non-ablation profile settings must be identical')
    return manifest


def _read(path, *, last=False):
    try:
        if not last:
            return json.loads(path.read_text())
        with path.open('rb') as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - 65536))
            lines = stream.read().splitlines()
        for line in reversed(lines):
            try:
                return json.loads(line)
            except ValueError:
                continue
    except (OSError, ValueError):
        pass
    return None


def observe(capture):
    """Read only bounded snapshots; never query the engine or synchronize CUDA."""
    result = dict(started_at=time.perf_counter(), wall=time.time())
    sources = {
        'progress': ('progress.json', False),
        'resources': ('scheduler_resources.json', False),
        'action': ('action_attention.jsonl', True),
        'checkpoint': ('checkpoint_timing.jsonl', True),
    }
    for name, (filename, last) in sources.items():
        path = capture / filename
        value = _read(path, last=last)
        if value is None:
            continue
        if name == 'progress':
            value = {key: item for key, item in value.items() if key not in
                     ('accepted_history', 'query_states', 'completed', 'request_reports')}
        result[name] = value
        try:
            result[name + '_modified_wall'] = path.stat().st_mtime
        except OSError:
            pass
    result['finished_at'] = time.perf_counter()
    return result


async def run(args):
    root = args.output.resolve()
    if ROOT.resolve() != (root / 'frozen').resolve():
        raise ValueError('Execute the prepared frozen script')
    validate_profiles(root)
    original = benchmark.run_arm

    async def observed_arm(root, profile, fixture, arm, pair, arguments):
        directory = root / f'pair-{pair:02d}-{arm}'
        done = asyncio.Event()

        async def sample():
            while not done.is_set():
                if directory.is_dir():
                    row = await asyncio.to_thread(observe, directory / 'server/capture')
                    with (directory / 'host_counter_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                try:
                    await asyncio.wait_for(done.wait(), timeout=1.)
                except asyncio.TimeoutError:
                    pass

        observer = asyncio.create_task(sample())
        try:
            return await original(root, profile, fixture, arm, pair, arguments)
        finally:
            done.set()
            await observer
            if directory.is_dir():
                with (directory / 'host_counter_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(observe(directory / 'server/capture')) + '\n')

    benchmark.run_arm = observed_arm
    try:
        await benchmark.run(args)
        if benchmark.read(root / 'status.json').get('status') != 'completed_pending_review':
            raise RuntimeError('Host ablation did not complete; inspect preserved status and logs')
    finally:
        benchmark.run_arm = original


def main():
    parser = benchmark.parser()
    parser.description = __doc__
    args = parser.parse_args()
    if args.run_prepared:
        if args.prepare_only or args.config is not None or args.fixture is not None:
            parser.error('Frozen execution uses its saved profile and fixture')
        return asyncio.run(run(args))
    if not args.prepare_only or args.config is None or args.fixture is None:
        parser.error('Prepare with --prepare-only, --config and --fixture first')
    if args.duration_seconds != 300 or args.concurrency != 8 or args.warmup_turns != 2 or args.epoch_turns != 16:
        parser.error('Require the matched C8, 300-second, 2-warmup/16-turn protocol')
    if args.pairs < 1 or args.idle_seconds < 0 or args.modes != ['saturated']:
        parser.error('Require positive pairs, nonnegative idle guard, and saturated mode')
    prepare(args)


if __name__ == '__main__':
    main()
