"""Compare frozen server runtimes using one fixed policy and timed clients.

Prepare with explicit --baseline-runtime and --candidate-runtime source roots.
Then execute this experiment's frozen script with --run-prepared. Both servers
use the shared benchmark's GPU ownership guard, lock, and owned-process cleanup.
"""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('src', 'scripts', 'csrc')
ARMS = ('A0', 'P1')


def load_benchmark():
    spec = importlib.util.spec_from_file_location(
        'timed_runtime_shared_benchmark', ROOT/'scripts/benchmark_timed_agents.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark()


def runtime_source_hashes(root):
    """Hash only runtime trees, never artifacts or a source checkout's .git."""
    return {str(Path(directory)/name): digest
            for directory in DIRECTORIES if (root/directory).is_dir()
            for name, digest in benchmark.hashes(root/directory).items()}


def source_metadata(source, snapshot_hashes):
    commit = None
    try:
        top = subprocess.run(['git', '-C', str(source), 'rev-parse', '--show-toplevel'],
                             text=True, capture_output=True, timeout=5, check=True)
        # A historical frozen tree inside a checkout does not represent the
        # ancestor checkout's HEAD; retain its own source hash instead.
        if Path(top.stdout.strip()).resolve() == source:
            revision = subprocess.run(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                      text=True, capture_output=True, timeout=5, check=True)
            commit = revision.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    source_hash = hashlib.sha256(json.dumps(snapshot_hashes, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return dict(path=str(source), commit=commit, source_hash=source_hash,
                identity='Exact copied file hashes are authoritative, including uncommitted changes.')


def validate_sources(args):
    root = args.output.resolve()
    sources = {}
    for arm, option in (('A0', 'baseline_runtime'), ('P1', 'candidate_runtime')):
        value = getattr(args, option, None)
        if value is None:
            raise ValueError('Preparation requires explicit baseline and candidate runtime sources')
        source = value.expanduser().resolve()
        if not (source/'src/vllm_kvmem/serve.py').is_file() or not (source/'scripts').is_dir():
            raise ValueError(f'Missing server runtime source trees: {source}')
        for directory in DIRECTORIES:
            if root.is_relative_to(source/directory):
                raise ValueError('Experiment output cannot be inside a copied runtime source tree')
        sources[arm] = source
    return sources


def prepare(args):
    policy = getattr(args, 'policy', 'P1')
    if policy not in ARMS:
        raise ValueError('Runtime comparison policy must be A0 or P1')
    sources = validate_sources(args)
    # All client settings, fixture validation, and client-source freezing come
    # from one implementation. The variant changes only each server's runtime.
    benchmark.prepare(args)
    root = args.output.resolve()
    benchmark.q.save(root/'status.json', dict(status='preparing_runtime_snapshots',
                                             gpu_started=False, at=time.time()))
    try:
        candidate = benchmark.read(root/(policy+'.json'))
        enabled = policy == 'P1'
        if (candidate.get('kvmem', {}).get('offline_agent_policy') is not enabled or
                candidate['kvmem'].get('active_resident_pressure_only') is not enabled):
            raise ValueError('Runtime comparison requires the selected qualified residence policy')
        for arm in ARMS:
            benchmark.q.save(root/(arm+'.json'), candidate)
        manifest = benchmark.read(root/'manifest.json')
        runtime_hashes, provenance = {}, {}
        for arm in ARMS:
            source = sources[arm]
            expected = runtime_source_hashes(source)
            destination = root/'runtime'/arm
            for directory in DIRECTORIES:
                if (source/directory).is_dir():
                    shutil.copytree(source/directory, destination/directory,
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            actual = benchmark.hashes(destination)
            if actual != expected or runtime_source_hashes(source) != expected:
                raise RuntimeError(f'{arm} runtime source changed while taking its snapshot')
            runtime_hashes[arm] = actual
            provenance[arm] = source_metadata(source, actual)
        manifest.update(comparison_kind='runtime', policy=policy, runtime_sha256=runtime_hashes,
            runtime_source=provenance,
            profile_sha256={arm:hashlib.sha256((root/(arm+'.json')).read_bytes()).hexdigest()
                            for arm in ARMS},
            runtime_comparison_scope=f'Identical {policy} serving profiles, client/harness and fixture; '
                'A0 and P1 identify baseline and candidate server runtimes, not policy switches. '
                'Controlled performance diagnosis, not a task-quality evaluation or a change to production defaults.')
        benchmark.q.save(root/'manifest.json', manifest)
        benchmark.q.save(root/'status.json', dict(status='prepared', gpu_started=False, at=time.time()))
    except BaseException as error:
        benchmark.q.save(root/'status.json', dict(status='failed', gpu_started=False,
            error=repr(error), at=time.time()))
        raise


def validate_prepared(root):
    if ROOT.resolve() != (root/'frozen').resolve():
        raise ValueError('Execute output/frozen/scripts/benchmark_timed_runtime_variants.py, not the live checkout')
    manifest = benchmark.read(root/'manifest.json')
    if manifest.get('comparison_kind') != 'runtime' or set(manifest.get('runtime_sha256', {})) != set(ARMS):
        raise ValueError('Prepared experiment is not a two-runtime comparison')
    for arm in ARMS:
        expected = manifest['runtime_sha256'][arm]
        if not expected or benchmark.hashes(root/'runtime'/arm) != expected:
            raise RuntimeError(f'Frozen {arm} runtime changed or is missing')
    policy = manifest.get('policy')
    if policy not in ARMS:
        raise ValueError('Prepared runtime comparison is missing its fixed policy')
    profiles = [benchmark.read(root/(arm+'.json')) for arm in ARMS]
    enabled = policy == 'P1'
    if (profiles[0] != profiles[1] or
            profiles[0].get('kvmem', {}).get('offline_agent_policy') is not enabled or
            profiles[0]['kvmem'].get('active_resident_pressure_only') is not enabled):
        raise ValueError('Runtime arms require identical serving profiles for their fixed policy')
    return manifest


async def run(args):
    validate_prepared(args.output.resolve())
    # Shared execution rechecks prepared status, fixture/profile hashes, and
    # client-source hashes; owns the GPU lock and cleanup; never auto-restarts.
    return await benchmark.run(args)


def parser():
    result = benchmark.parser()
    result.description = __doc__
    result.add_argument('--policy', choices=ARMS, default='P1',
                        help='Same policy for both server runtimes; A0 isolates cold restoration')
    result.add_argument('--baseline-runtime', type=Path,
                        help='Explicit original server source root; required during preparation')
    result.add_argument('--candidate-runtime', type=Path,
                        help='Explicit candidate server source root; required during preparation')
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.run_prepared:
        if args.prepare_only:
            command.error('Choose preparation or frozen execution')
        if any(getattr(args, key) is not None for key in
               ('config', 'fixture', 'baseline_runtime', 'candidate_runtime')):
            command.error('Frozen execution uses prepared sources, profiles, and fixture only')
        return asyncio.run(run(args))
    if not args.prepare_only:
        command.error('Prepare with --prepare-only, then execute this experiment\'s frozen script with --run-prepared')
    if any(getattr(args, key) is None for key in
           ('config', 'fixture', 'baseline_runtime', 'candidate_runtime')):
        command.error('Preparation requires config, fixture, baseline-runtime and candidate-runtime')
    if (not math.isfinite(args.duration_seconds) or args.duration_seconds <= 0 or args.pairs < 1 or
            not math.isfinite(args.idle_seconds) or args.idle_seconds < 0 or
            args.warmup_turns < 1 or args.epoch_turns <= args.warmup_turns or
            args.output_tokens < 1 or args.context_tokens < 1024 or
            len(set(args.modes)) != len(args.modes)):
        command.error('Invalid fixed window, workload, or pair count')
    prepare(args)


if __name__ == '__main__':
    main()
