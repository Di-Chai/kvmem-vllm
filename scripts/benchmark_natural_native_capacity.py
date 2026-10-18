"""Frozen native-only .915 capacity control for the natural eight-agent run.

Reuse the reference workload/launcher unchanged. Preparation and verification
are CPU-only; only --run-prepared may start the owned model server.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
import importlib.metadata
import json
import subprocess
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_natural_native_capacity.py'
EXPERIMENT = 'natural_native_capacity_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ready = module('natural_capacity_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def deployment_profile(source, port):
    cap.require(type(port) is int and 1024 <= port <= 65535, 'Invalid dedicated port')
    cap.require(source.get('max_num_seqs') == 8 and source.get('max_model_len') == 262144
        and source.get('gpu_memory_utilization') == .90,
        'Require original completed eight-slot .90 natural control')
    result = copy.deepcopy(source)
    result.update(port=port, gpu_memory_utilization=.915)
    return result


def launch_identity(profile):
    launch = ready.native.launch.build_launch(profile, port=profile['port'],
        async_scheduling=True, environ={})
    # Safe projection only: never persist inherited environment values.
    return launch['native_profile']


def reference_proof(reference, expected_manifest=None):
    if expected_manifest is not None:
        cap.require(cap.sha(reference/'manifest.json') == expected_manifest,
                    'Reference manifest changed')
    wrapper = module('capacity_completed_reference', reference/'frozen/scripts/benchmark_natural_native_deployment.py')
    manifest, proofs = wrapper.verify(reference)
    cap.require(manifest.get('experiment_kind') == 'natural_native_deployment_v1'
        and manifest['workload'] == dict(concurrency=8, warmup_turns=2,
        epoch_turns=16, duration_seconds=300, mode='tools', generation_mode='natural',
        generation_limit=32000), 'Reference natural workload changed')
    cap.require(cap.read(reference/'status.json')['status'] == 'completed_pending_review',
                'Reference .90 control must actually complete')
    result = cap.read(reference/'native/results.json')
    cap.require(result.get('status') == 'completed' and result.get('drained') is True
        and result.get('native_client_tokens_equal') is True,
        'Reference lacks completed reconciled normal drain')
    source = cap.read(reference/'native.json')
    actual = cap.read(reference/'native/launch.json')
    expected = ready.native.launch.build_launch(source, port=source['port'],
        async_scheduling=True, environ={})
    cap.require(actual.get('native') is True and actual.get('argv') == expected['argv']
        and expected['native_profile'] == manifest['effective_native_profile'],
        'Reference actual native launch changed')
    return manifest, proofs


def installed_native_proof(path):
    identity = cap.read(path)
    source = Path(identity['source_directory'])
    actual_source = Path(importlib.metadata.distribution('vllm').locate_file('vllm')).resolve()
    cap.require(source.resolve() == actual_source and identity.get('files'),
                'Missing installed native source contract')
    actual = {str(p):cap.sha(p) for p in sorted(source.rglob('*'))
              if p.is_file() and p.suffix in ('.py', '.so')}
    cap.require(actual == identity['files'], 'Installed native Python/shared-library identity changed')
    return identity


def device_memory_identity():
    raw = subprocess.check_output(['nvidia-smi',
        '--query-gpu=uuid,memory.total,memory.reserved', '--format=csv,noheader,nounits'], text=True)
    rows = [line.split(',') for line in raw.splitlines() if line.strip()]
    cap.require(len(rows) == 1 and len(rows[0]) == 3, 'Require one measured device')
    uuid, total, reserved = [x.strip() for x in rows[0]]
    return dict(uuid=uuid, total_mib=int(total), reserved_mib=int(reserved))


def completed_control_identity(reference):
    return {name:cap.sha(reference/name) for name in
            ('native/launch.json', 'native/results.json', 'native/tools/window.json')}


def deployment_acceptance(root, manifest):
    rows = ready.records(root/'native/gpu_samples.jsonl')
    cap.require(rows, 'Missing all-phase GPU observations')
    result = cap.read(root/'native/results.json')
    cap.require(result.get('status') == 'completed' and result.get('drained') is True
        and result.get('native_client_tokens_equal') is True, 'Incomplete normal drain/output reconciliation')
    identity = manifest['device_memory_identity']
    peak = max(rows, key=lambda x:x['memory_mib'])
    free = identity['total_mib']-identity['reserved_mib']-peak['memory_mib']
    native_samples = ready.records(root/'native/native_samples.jsonl')
    cap.require(native_samples, 'Missing native lifetime counters')
    preemptions = max(x['values']['vllm:num_preemptions_total'] for x in native_samples)
    accepted = free >= 4096 and preemptions == 0
    return dict(accepted=accepted, status='accepted' if accepted else 'contract_failed',
        reason=None if accepted else 'contract_failed_headroom' if free < 4096 else 'preemptions',
        required_free_mib=4096, minimum_sampled_free_mib=free,
        maximum_used_mib=peak['memory_mib'], peak_wall=peak['wall_at'], samples=len(rows),
        zero_preemptions=preemptions == 0, full_process_preemptions=preemptions, normal_drain=True,
        source='native/gpu_samples.jsonl', source_sha256=cap.sha(root/'native/gpu_samples.jsonl'),
        scope='All recorded loading, initial cleanup, warmup, formal and normal-drain samples. '
              'One-second NVML may miss subsecond peaks; not an allocator high-water proof. '
              'A negative deployment verdict preserves completed_pending_review and never retries.')


def prepare(args):
    root, reference = args.output.resolve(), args.reference.resolve()
    cap.require(not root.exists(), 'Use a fresh deployment directory')
    original, proofs = reference_proof(reference)
    installed_native_proof(args.installed_native)
    device = device_memory_identity()
    control_identity = completed_control_identity(reference)
    source = cap.read(reference/'native.json')
    profile = deployment_profile(source, args.port)
    before, after = launch_identity(source), launch_identity(profile)
    expected = dict(before, port=args.port, gpu_memory_utilization=.915)
    cap.require(after == expected, 'Effective native controls changed beyond port and GPU budget')
    root.mkdir(parents=True)
    shutil.copytree(reference/'frozen', root/'frozen',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    shutil.copyfile(Path(__file__), root/'frozen'/ENTRY)
    shutil.copyfile(reference/'fixture.json', root/'fixture.json')
    shutil.copyfile(args.installed_native, root/'installed-native-identity.json')
    cap.save(root/'native.json', profile)
    cap.save(root/'manifest.json', dict(kind='ready_agents_v1', experiment_kind=EXPERIMENT,
        at=time.time(), reference=str(reference), reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        installed_native_sha256=cap.sha(root/'installed-native-identity.json'),
        installed_native_scope='Current full installed .py/.so identity, rechecked prelaunch and after idle; not retrospective proof of every prior installation file.',
        device_memory_identity=device, minimum_device_free_mib=4096,
        reference_control_identity=control_identity,
        source_sha256=cap.hashes(root/'frozen'), wrapper_sha256=cap.sha(root/'frozen'/ENTRY),
        runtime_sha256={}, profiles_sha256={'native':cap.sha(root/'native.json')},
        fixture_sha256=cap.sha(root/'fixture.json'), native_source=original['native_source'],
        workload=original['workload'], generation_sampling=original['generation_sampling'],
        budget=original['budget'], arms=['native'], qualify=False, port=args.port,
        idle_seconds=original['idle_seconds'], effective_native_profile=after,
        declared_profile_delta=dict(port=[source['port'],args.port],gpu_memory_utilization=[.90,.915]),
        scope='Same original natural eight-agent reference, external messages/waits, own accepted IDs, '
              'two warmup turns,300-second window,32000 natural limit and normal drain. '
              'Only native GPU budget .90 to .915 and dedicated port change. '
              'No KVMem runtime is loaded by the native server. Not task-quality or pure-kernel throughput.'))
    cap.save(root/'reference-qualification-proof.json', dict(checked_at=time.time(), proofs=proofs))
    cap.save(root/'status.json', dict(status='prepared', at=time.time(), gpu_started=False))


def verify(root):
    manifest = ready.verify(root)
    cap.require(manifest.get('experiment_kind') == EXPERIMENT and manifest.get('arms') == ['native']
        and manifest.get('qualify') is False and manifest.get('runtime_sha256') == {},
        'Not the isolated native deployment experiment')
    reference = Path(manifest['reference'])
    original, proofs = reference_proof(reference, manifest['reference_manifest_sha256'])
    expected_sources = dict(original['source_sha256'])
    expected_sources[ENTRY] = manifest['wrapper_sha256']
    cap.require(manifest['source_sha256'] == expected_sources,
                'Frozen source differs from reference beyond deployment entry')
    for key in ('workload', 'generation_sampling', 'budget', 'native_source', 'idle_seconds'):
        cap.require(manifest[key] == original[key], f'Reference {key} changed')
    cap.require(cap.sha(root/'fixture.json') == original['fixture_sha256'], 'Original fixture changed')
    source = cap.read(reference/'native.json')
    profile = cap.read(root/'native.json')
    cap.require(profile == deployment_profile(source, manifest['port']),
                'Native profile changed beyond declared deployment delta')
    delta = dict(port=[source['port'], manifest['port']], gpu_memory_utilization=[.90,.915])
    cap.require(manifest['declared_profile_delta'] == delta, 'Deployment delta declaration changed')
    expected = dict(launch_identity(source), port=manifest['port'], gpu_memory_utilization=.915)
    cap.require(launch_identity(profile) == manifest['effective_native_profile'] == expected,
                'Effective native launcher settings changed')
    cap.require(manifest.get('minimum_device_free_mib') == 4096, 'Device headroom contract changed')
    cap.require(manifest['device_memory_identity'] == device_memory_identity(), 'Device memory identity changed')
    cap.require(completed_control_identity(reference) == manifest['reference_control_identity'], 'Completed reference results changed')
    cap.require(cap.sha(root/'installed-native-identity.json') == manifest['installed_native_sha256'], 'Installed native lock changed')
    installed_native_proof(root/'installed-native-identity.json')
    return manifest, proofs


async def run(args):
    root = args.output.resolve()
    manifest, proofs = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    cap.save(root/'reference-qualification-proof.json', dict(checked_at=time.time(), proofs=proofs))
    # Reuse the unchanged frozen run_arm. Recheck all identities after its
    # potentially long idle wait and before it constructs/spawns the server.
    original_wait = ready.bench.q.wait_idle
    async def checked_wait(*values, **kwargs):
        await original_wait(*values, **kwargs)
        current, _ = verify(root)
        cap.require(current == manifest, 'Deployment identity changed while waiting')
    ready.bench.q.wait_idle = checked_wait
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, active.cancel)
    started, monotonic = time.time(), time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            await ready.run_arm(root, 'native', manifest)
        analysis = module('natural_native_analysis', ROOT/'scripts/analyze_rolling_agents.py')
        cap.save(root/'analysis.json', analysis.review(root))
        cap.save(root/'deployment-acceptance.json', deployment_acceptance(root, manifest))
        cap.save(root/'status.json', dict(status='completed_pending_review', at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        ready.bench.q.wait_idle = original_wait
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json', dict(started_wall=started, finished_wall=time.time(),
            elapsed_seconds=time.monotonic()-monotonic, scope='Native launch through warmup, formal window and normal drain'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--installed-native', type=Path)
    parser.add_argument('--port', type=int, default=18566)
    args = parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(args.reference is not None and args.installed_native is not None, 'Explicit completed natural reference and native install lock required')
    prepare(args)


if __name__ == '__main__':
    main()
