"""One frozen P1 restore trace; diagnostic only, never a throughput comparison.

Prepare from an existing host-ablation experiment, then execute this script
from the new output's frozen tree. Loading, warmup and drain remain outside the
300-second formal window; there is no process deadline or automatic retry.
"""
import argparse
import asyncio
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
FEATURES = ('split_capture_dma', 'native_prefill_commit', 'async_checkpoint_capture')
LOCK_PATH = '/tmp/kvmem-coordination-qualification.lock'
WORKLOAD_KEYS = ('duration_seconds', 'modes', 'concurrency', 'warmup_turns',
                 'output_tokens', 'context_tokens', 'epoch_turns', 'idle_seconds')
spec = importlib.util.spec_from_file_location('restore_diagnostic_benchmark',
    ROOT / 'scripts/benchmark_timed_agents.py')
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_reference(profile, fixture, manifest):
    workload = manifest['workload']
    if (manifest.get('comparison_kind') != 'host_dependency_ablation' or
            manifest.get('feature_switches') != list(FEATURES) or
            any(profile['kvmem'].get(name) is not True for name in FEATURES) or
            profile.get('verify_session_restore') is not False or
            profile.get('max_num_seqs') != 8 or profile['kvmem'].get('step_latency_interval') != 1):
        raise ValueError('Require the original P1 profile with all three optimizations and no byte oracle')
    if (workload['concurrency'] != 8 or workload['duration_seconds'] != 300 or
            workload['warmup_turns'] != 2 or workload['epoch_turns'] != 16 or
            workload['modes'] != ['saturated'] or workload['output_tokens'] < 1 or
            workload['context_tokens'] < 1024 or not math.isfinite(workload['idle_seconds']) or
            workload['idle_seconds'] < 0 or 'agent_plans' not in fixture):
        raise ValueError('Require the matched C8/300s/2-warmup/16-turn frozen trace')
    if profile['kvmem'].get('restore_stage_timing', False) is not False:
        raise ValueError('Reference P1 must not already enable restore stage timing')


def diagnostic_profile(reference, port):
    if type(port) is not int or not 0 < port < 65536:
        raise ValueError('Invalid diagnostic port')
    profile = json.loads(json.dumps(reference))
    profile['port'] = port
    profile['kvmem']['restore_stage_timing'] = True
    return profile


def profile_changes(reference, profile):
    changes = [dict(path='kvmem.restore_stage_timing',
                    before_present='restore_stage_timing' in reference['kvmem'],
                    before=reference['kvmem'].get('restore_stage_timing'), after=True)]
    if reference['port'] != profile['port']:
        changes.insert(0, dict(path='port', before_present=True,
                               before=reference['port'], after=profile['port']))
    return changes


def prepared_args(root, workload):
    args = benchmark.parser().parse_args(['--output', str(root)])
    for key in WORKLOAD_KEYS:
        setattr(args, key, workload[key])
    args.port = workload['port']
    args.pairs = 1
    args.prepare_only, args.run_prepared = True, False
    return args


def prepare(args):
    source, root = args.source_experiment.resolve(), args.output.resolve()
    # The original experiment is read-only, including its completed run files.
    if root == source or source in root.parents:
        raise ValueError('Diagnostic output must be outside the source experiment')
    reference_manifest = benchmark.read(source / 'manifest.json')
    profile, fixture = benchmark.read(source / 'P1.json'), benchmark.read(source / 'fixture.json')
    profile_hash, fixture_hash = sha256(source / 'P1.json'), sha256(source / 'fixture.json')
    if (profile_hash != reference_manifest['profile_sha256']['P1'] or
            benchmark.q.digest(fixture) != reference_manifest['fixture_sha256']):
        raise ValueError('Reference profile or fixture differs from its frozen manifest')
    check_reference(profile, fixture, reference_manifest)
    final_profile = diagnostic_profile(profile, args.port if args.port is not None else profile['port'])
    workload = {key: reference_manifest['workload'][key] for key in WORKLOAD_KEYS}
    workload.update(port=final_profile['port'], pairs=1)
    preparation = prepared_args(root, workload)
    preparation.config, preparation.fixture = source / 'P1.json', source / 'fixture.json'
    benchmark.prepare(preparation)
    benchmark.q.save(root / 'status.json', dict(status='preparing_diagnostic', at=time.time(), gpu_started=False))
    # Shared preparation supplies freezing/validation, but changes public P1
    # defaults. Restore the exact reference profile, including timing cadence.
    from vllm_kvmem.serve import build_launch
    build_launch(final_profile, source, root / 'validation-diagnostic')
    benchmark.q.save(root / 'P1.json', final_profile)
    (root / 'A0.json').unlink()
    (root / 'fixture.json').write_bytes((source / 'fixture.json').read_bytes())
    reference = root / 'reference'
    reference.mkdir()
    for name in ('P1.json', 'manifest.json'):
        (reference / name).write_bytes((source / name).read_bytes())
    manifest = benchmark.read(root / 'manifest.json')
    frozen = root / 'frozen'
    # Detect concurrent source edits while the shared preparer copied files.
    for directory in ('src', 'scripts', 'csrc'):
        if (ROOT / directory).exists() and benchmark.hashes(ROOT / directory) != benchmark.hashes(frozen / directory):
            raise RuntimeError('Live source changed while freezing diagnostic')
    manifest.update(diagnostic_only=True, performance_claim_allowed=False,
        comparison_kind='restore_stage_diagnostic', arms=['P1'],
        reference_experiment=str(source),
        reference_sha256={name: sha256(reference / name) for name in ('P1.json', 'manifest.json')},
        reference_fixture_sha256=fixture_hash,
        profile_sha256={'P1': sha256(root / 'P1.json')},
        fixture_file_sha256=sha256(root / 'fixture.json'), workload=workload,
        profile_changes=profile_changes(profile, final_profile),
        protocol='One P1 arm; same fixed warmup -> 300-second window -> excluded drain/cleanup',
        scope='Diagnostic only. Restore stage observation can perturb timing and cannot establish a '
              'performance win. Nested host intervals overlap; wall minus thread CPU is not GPU idle.',
        execution_changes=dict(reference_pairs=reference_manifest['workload']['pairs'],
                               diagnostic_pairs=1, reference_arms=['A0', 'P1'], diagnostic_arms=['P1']))
    benchmark.q.save(root / 'manifest.json', manifest)
    validate(root)
    benchmark.q.save(root / 'status.json', dict(status='prepared', diagnostic_only=True,
                                               at=time.time(), gpu_started=False))


def validate(root):
    manifest = benchmark.read(root / 'manifest.json')
    if (manifest.get('diagnostic_only') is not True or manifest.get('performance_claim_allowed') is not False or
            manifest.get('comparison_kind') != 'restore_stage_diagnostic' or manifest.get('arms') != ['P1'] or
            'runtime_sha256' in manifest or (root / 'A0.json').exists()):
        raise ValueError('Require one frozen diagnostic P1 arm')
    for name, digest in manifest['reference_sha256'].items():
        if sha256(root / 'reference' / name) != digest:
            raise ValueError('Frozen reference changed')
    reference = benchmark.read(root / 'reference/P1.json')
    reference_manifest = benchmark.read(root / 'reference/manifest.json')
    profile, fixture = benchmark.read(root / 'P1.json'), benchmark.read(root / 'fixture.json')
    if (sha256(root / 'reference/P1.json') != reference_manifest['profile_sha256']['P1'] or
            sha256(root / 'P1.json') != manifest['profile_sha256']['P1'] or
            sha256(root / 'fixture.json') != manifest['fixture_file_sha256'] or
            manifest['fixture_file_sha256'] != manifest['reference_fixture_sha256'] or
            benchmark.q.digest(fixture) != manifest['fixture_sha256'] or
            manifest['fixture_sha256'] != reference_manifest['fixture_sha256']):
        raise ValueError('Frozen diagnostic profile or fixture changed')
    check_reference(reference, fixture, reference_manifest)
    if (profile != diagnostic_profile(reference, manifest['workload']['port']) or
            manifest['profile_changes'] != profile_changes(reference, profile) or
            manifest['workload']['pairs'] != 1 or any(manifest['workload'][key] !=
                reference_manifest['workload'][key] for key in WORKLOAD_KEYS)):
        raise ValueError('Diagnostic changed settings beyond port and restore observation')
    if benchmark.hashes(root / 'frozen') != manifest['source_sha256']:
        raise ValueError('Frozen diagnostic source changed')
    return manifest, profile, fixture


async def run(args):
    root = args.output.resolve()
    if ROOT.resolve() != (root / 'frozen').resolve() or benchmark.ROOT.resolve() != ROOT.resolve():
        raise ValueError('Execute the prepared frozen scripts/diagnose_restore_stages.py')
    manifest, profile, fixture = validate(root)
    manifest_hash = sha256(root / 'manifest.json')
    saved_args = prepared_args(root, manifest['workload'])
    saved_args.prepare_only, saved_args.run_prepared = False, True
    # Reuse exactly the same cooperative lock and per-arm resource guards.
    with open(LOCK_PATH, 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if benchmark.read(root / 'status.json').get('status') != 'prepared':
            raise ValueError('Require a fresh prepared diagnostic; no automatic restarts')
        loop, active = asyncio.get_running_loop(), asyncio.current_task()
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(signum, active.cancel)
        original_wait = benchmark.q.wait_idle

        async def checked_wait(output, stable_seconds):
            await original_wait(output, stable_seconds)
            if sha256(root / 'manifest.json') != manifest_hash:
                raise ValueError('Frozen manifest changed while waiting for GPU')
            validate(root)  # Waiting must not allow disk/server and client inputs to diverge.

        benchmark.q.wait_idle = checked_wait
        try:
            benchmark.q.save(root / 'status.json', dict(status='diagnostic_starting',
                                                       diagnostic_only=True, at=time.time()))
            results = await benchmark.run_arm(root, profile, fixture, 'P1', 0, saved_args)
            benchmark.q.save(root / 'diagnostic.json', dict(diagnostic_only=True,
                performance_claim_allowed=False, pair=0, arm='P1', results=results,
                scope=manifest['scope']))
            benchmark.q.save(root / 'status.json', dict(status='diagnostic_completed',
                                                       diagnostic_only=True, at=time.time()))
        except BaseException as error:
            benchmark.q.save(root / 'status.json', dict(status='interrupted' if isinstance(error,
                asyncio.CancelledError) else 'failed', diagnostic_only=True, error=repr(error), at=time.time()))
            raise
        finally:
            benchmark.q.wait_idle = original_wait
            for signum in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-experiment', type=Path)
    parser.add_argument('--port', type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    args = parser.parse_args()
    if args.run_prepared:
        if args.source_experiment is not None or args.port is not None:
            parser.error('Frozen execution uses only its saved profile and fixture')
        return asyncio.run(run(args))
    if args.source_experiment is None:
        parser.error('Preparation requires --source-experiment')
    prepare(args)


if __name__ == '__main__':
    main()
