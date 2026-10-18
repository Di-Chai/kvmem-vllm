"""Frozen reverse-order natural agent replay with the completed native .90 control.

Only this wrapper is added to the original frozen benchmark. The workload,
runtime algorithms, warmup, formal window, drain and owned-process cleanup are
reused unchanged. Preparation does not start or enqueue GPU work.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.metadata
import importlib.util
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_rolling_reverse.py'
EXPERIMENT = 'rolling_agents_reverse_v1'
ARMS = ('rolling-candidate-v2', 'pipeline-base', 'native')
WORKLOAD = dict(concurrency=8, warmup_turns=2, epoch_turns=16, duration_seconds=300,
                mode='tools', generation_mode='natural', generation_limit=32000)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ready = module('reverse_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def profile(source, arm, port):
    cap.require(arm in ARMS and type(port) is int and 1024 <= port <= 65535, 'Invalid arm or port')
    cap.require(source.get('max_num_seqs') == 8 and source.get('max_model_len') == 262144 and
                source.get('gpu_memory_utilization') == .81, 'Require original natural eight-slot .81 controls')
    result = copy.deepcopy(source)
    result['port'] = port
    if arm == 'native':
        result['gpu_memory_utilization'] = .90
    return result


def without_port(argv):
    result = list(argv)
    cap.require(result.count('--port') == 1, 'Native launch requires exactly one explicit port')
    index = result.index('--port') + 1
    cap.require(index < len(result), 'Native launch port is missing')
    result[index] = '<PORT>'
    return result


def reference_proof(reference, expected_manifest=None):
    if expected_manifest is not None:
        cap.require(cap.sha(reference/'manifest.json') == expected_manifest, 'Reference manifest changed')
    natural = module('reverse_original', reference/'frozen/scripts/benchmark_rolling_agents.py')
    manifest = natural.verify(reference)
    cap.require(manifest['workload'] == WORKLOAD, 'Original natural workload changed')
    cap.require(cap.read(reference/'status.json')['status'] == 'completed_pending_review',
                'Original three-arm run must complete before reversal')
    return manifest, natural.qualification_proofs(manifest)


def native_control_proof(control, reference, expected_manifest=None):
    if expected_manifest is not None:
        cap.require(cap.sha(control/'manifest.json') == expected_manifest, 'Native control manifest changed')
    wrapper = module('reverse_native_control', control/'frozen/scripts/benchmark_natural_native_deployment.py')
    manifest, _ = wrapper.verify(control)
    cap.require(Path(manifest['reference']).resolve() == reference.resolve(), 'Native control uses another workload reference')
    cap.require(cap.read(control/'status.json')['status'] == 'completed_pending_review',
                'Strong native control must actually complete')
    result = cap.read(control/'native/results.json')
    cap.require(result.get('status') == 'completed' and result.get('drained') is True and
                result.get('native_client_tokens_equal') is True, 'Strong native control lacks completed reconciled drain')
    source = cap.read(control/'native.json')
    launch = ready.native.launch.build_launch(source, port=source['port'], async_scheduling=True, environ={})
    recorded = cap.read(control/'native/launch.json')
    cap.require(recorded.get('native') is True and recorded.get('argv') == launch['argv'],
                'Native actual launch differs from its declared effective controls')
    cap.require(launch['native_profile'] == manifest['effective_native_profile'], 'Effective native control changed')
    return dict(manifest_sha256=cap.sha(control/'manifest.json'), launch_sha256=cap.sha(control/'native/launch.json'),
        result_sha256=cap.sha(control/'native/results.json'), argv_without_port=without_port(recorded['argv']),
        effective_profile=launch['native_profile'])


def installed_native_proof(path):
    identity = cap.read(path)
    source = Path(identity['source_directory'])
    actual_source = Path(importlib.metadata.distribution('vllm').locate_file('vllm')).resolve()
    cap.require(source.resolve() == actual_source and identity.get('files'), 'Missing installed native source contract')
    actual = {str(p): cap.sha(p) for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.py', '.so')}
    cap.require(actual == identity['files'], 'Installed native Python/shared-library identity changed')
    return identity


def native_launch_identity(p, expected):
    launch = ready.native.launch.build_launch(p, port=p['port'], async_scheduling=True, environ={})
    cap.require(without_port(launch['argv']) == expected['argv_without_port'],
                'Reverse native launch differs from completed .90 control beyond port')
    cap.require(launch['native_profile'] == dict(expected['effective_profile'], port=p['port']),
                'Reverse effective native profile differs from completed .90 control')
    return launch['native_profile']


def prepare(args):
    root, reference, control = args.output.resolve(), args.reference.resolve(), args.native_control.resolve()
    cap.require(not root.exists(), 'Use a fresh reverse-run directory')
    original, proofs = reference_proof(reference)
    native_proof = native_control_proof(control, reference)
    installed_native_proof(args.installed_native)
    profiles = {arm: profile(cap.read(reference/(arm+'.json')), arm, args.port) for arm in ARMS}
    native_launch_identity(profiles['native'], native_proof)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(Path(__file__), root/'frozen'/ENTRY)
    shutil.copytree(reference/'runtime', root/'runtime', ignore=ignore)
    shutil.copyfile(reference/'fixture.json', root/'fixture.json')
    shutil.copyfile(args.installed_native, root/'installed-native-identity.json')
    for arm, p in profiles.items():
        cap.save(root/(arm+'.json'), p)
    manifest = dict(kind='ready_agents_v1', experiment_kind=EXPERIMENT, created_at=time.time(),
        reference=str(reference), reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        native_control=str(control), native_control_proof=native_proof,
        source_sha256=cap.hashes(root/'frozen'), wrapper_sha256=cap.sha(root/'frozen'/ENTRY),
        runtime_sha256={arm:cap.hashes(root/'runtime'/arm) for arm in ARMS if arm != 'native'},
        profiles_sha256={arm:cap.sha(root/(arm+'.json')) for arm in ARMS},
        fixture_sha256=cap.sha(root/'fixture.json'), installed_native_sha256=cap.sha(root/'installed-native-identity.json'),
        native_source=original['native_source'], qualifications=original['qualifications'],
        workload=original['workload'], generation_sampling=original['generation_sampling'], budget=original['budget'],
        arms=list(ARMS), port=args.port, idle_seconds=original['idle_seconds'], qualify=False,
        declared_profile_delta={arm:dict(port=[cap.read(reference/(arm+'.json'))['port'],args.port],
            **({'gpu_memory_utilization':[.81,.90]} if arm=='native' else {})) for arm in ARMS},
        scope='Reverse order: rolling-candidate-v2, pipeline-base, native .90. Exact original two K runtimes '
              'and frozen workload: eight users, two complete warmup turns,300-second natural32000 window, '
              'original sampling/history budgets/tool waits/finite epochs, normal drain. Native .90 matches '
              'the completed standalone control except port. No MeanK startup, early release or Action diagnostic overlay.',
        timing_scope='Formal steady-state order repeat, not a controlled cold-start comparison. Original loader '
                     'and cold cache behavior remain; copied source paths may trigger first compilation during startup/warmup. '
                     'T0 still follows all users completing two warmup turns. Any new formal-window JIT must be separately '
                     'reported and must not be attributed to rolling. No implicit preloading or cache changes.',
        installed_native_scope='Current full installed .py/.so lock, rechecked before launch and after idle; '
                               'not retrospective proof of the earlier qualifications entire installation.')
    cap.require(manifest['runtime_sha256'] == original['runtime_sha256'], 'Runtime changed while copying')
    cap.save(root/'manifest.json', manifest)
    cap.save(root/'reference-qualification-proof.json', dict(checked_at=time.time(), proofs=proofs))
    cap.save(root/'status.json', dict(status='prepared', at=time.time(), gpu_started=False))


def verify(root):
    manifest = ready.verify(root)
    cap.require(manifest.get('experiment_kind') == EXPERIMENT and manifest.get('arms') == list(ARMS) and
                manifest.get('qualify') is False, 'Reverse order or experiment identity changed')
    reference = Path(manifest['reference'])
    original, proofs = reference_proof(reference, manifest['reference_manifest_sha256'])
    expected_sources = dict(original['source_sha256'], **{ENTRY:manifest['wrapper_sha256']})
    cap.require(manifest['source_sha256'] == expected_sources, 'Frozen source changed beyond reverse wrapper')
    for key in ('runtime_sha256','qualifications','workload','generation_sampling','budget','native_source','idle_seconds'):
        cap.require(manifest[key] == original[key], 'Original '+key+' changed')
    cap.require(cap.sha(root/'fixture.json') == original['fixture_sha256'], 'Original fixture changed')
    expected_control = manifest['native_control_proof']
    actual_control = native_control_proof(Path(manifest['native_control']), reference, expected_control['manifest_sha256'])
    cap.require(actual_control == expected_control, 'Completed native launch/result provenance changed')
    for arm in ARMS:
        source = cap.read(reference/(arm+'.json'))
        p = cap.read(root/(arm+'.json'))
        cap.require(p == profile(source,arm,manifest['port']), 'Profile changed beyond declared delta: '+arm)
        delta = dict(port=[source['port'],manifest['port']],
                     **({'gpu_memory_utilization':[.81,.90]} if arm=='native' else {}))
        cap.require(manifest['declared_profile_delta'][arm] == delta, 'Profile delta declaration changed')
    native_launch_identity(cap.read(root/'native.json'), actual_control)
    cap.require(cap.sha(root/'installed-native-identity.json') == manifest['installed_native_sha256'],
                'Installed native lock changed')
    installed_native_proof(root/'installed-native-identity.json')
    return manifest, proofs


async def run(args):
    root = args.output.resolve()
    manifest, proofs = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic restart')
    cap.save(root/'reference-qualification-proof.json', dict(checked_at=time.time(), proofs=proofs))
    original_wait = ready.bench.q.wait_idle
    async def checked_wait(*values, **kwargs):
        await original_wait(*values, **kwargs)
        current, _ = verify(root)
        cap.require(current == manifest, 'Reverse identity changed while waiting')
    ready.bench.q.wait_idle = checked_wait
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, active.cancel)
    started, monotonic = time.time(), time.monotonic()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            results = {}
            for arm in ARMS:
                current, _ = verify(root)
                cap.require(current == manifest, 'Reverse identity changed before next arm')
                results[arm] = await ready.run_arm(root, arm, manifest)
                cap.save(root/'comparison.json', {key:dict(output_tps=value['output_tokens_per_second'],
                    completed_turns_per_second=value['requests_completed_in_window']/300,
                    gpu=value['gpu'], preemptions=value['preemptions']) for key,value in results.items()})
            analysis = module('reverse_analysis', ROOT/'scripts/analyze_rolling_agents.py')
            cap.save(root/'analysis.json', analysis.review(root))
            cap.save(root/'status.json', dict(status='completed_pending_review',at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed',error=repr(error),at=time.time()))
        raise
    finally:
        ready.bench.q.wait_idle = original_wait
        for sig in (signal.SIGINT,signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json', dict(started_wall=started,finished_wall=time.time(),
            elapsed_seconds=time.monotonic()-monotonic,scope='All reversed arms including startup, warmup and normal drain'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--native-control',type=Path)
    parser.add_argument('--installed-native',type=Path)
    parser.add_argument('--port',type=int,default=18554)
    args=parser.parse_args()
    if args.run_prepared:
        return asyncio.run(run(args))
    cap.require(all((args.reference,args.native_control,args.installed_native)), 'Explicit reference/native control/install lock required')
    prepare(args)


if __name__=='__main__':
    main()
