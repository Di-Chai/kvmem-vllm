"""One qualified natural-agent replay with sampled Action diagnostics.

This is an attribution run, never a throughput comparison. Preparation copies
the reference's frozen workload and overlays only the reviewed diagnostic patch.
"""
import argparse
import asyncio
import copy
import fcntl
import importlib.util
from pathlib import Path
import shutil
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARM = 'rolling-candidate-v2'
FILES = {'src/vllm_kvmem/'+name+'.py' for name in (
    'action_attention', 'action_scoring', 'action_profile', 'capture_runtime', 'options')}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


ready = module('action_diagnostic_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def diagnostic_profile(reference, port):
    cap.require(type(port) is int and 0 < port < 65536, 'Invalid diagnostic port')
    cap.require(reference['kvmem'].get('action_attention') is True and
                reference['kvmem'].get('rolling_active_window') is True and
                reference.get('verify_session_restore') is False,
                'Require qualified rolling Action runtime without byte oracle')
    result = copy.deepcopy(reference)
    result['port'] = port
    result['kvmem'].update(action_profiling=True, action_profile_interval=8,
                           action_profile_max_events=4096)
    return result


def overlay_contract(base, overlay, manifest):
    rows = manifest['files']
    cap.require(len(rows) == len(FILES) and {r['path'] for r in rows} == FILES,
                'Only the five reviewed diagnostic files may change')
    expected = cap.hashes(base)
    for row in rows:
        name = row['path']
        cap.require(expected.get(name) == row['base_sha256'], 'Diagnostic base runtime differs')
        cap.require(cap.sha(overlay/name) == row['overlay_sha256'], 'Diagnostic overlay changed')
        expected[name] = row['overlay_sha256']
    return expected


def reference_gate(reference):
    runner = module('action_diagnostic_reference', reference/'frozen/scripts/benchmark_rolling_agents.py')
    manifest = runner.verify(reference)
    runner.qualification_proofs(manifest)
    return manifest


def prepare(args):
    root, reference, overlay_root = args.output.resolve(), args.reference.resolve(), args.overlay.resolve()
    cap.require(not root.exists() and reference not in root.parents, 'Use a separate fresh output directory')
    ref = reference_gate(reference)
    overlay = cap.read(overlay_root/'manifest.json')
    cap.require(cap.sha(overlay_root/'diagnostic-only.patch') == overlay['patch_sha256'],
                'Diagnostic patch changed')
    expected = overlay_contract(reference/'runtime'/ARM,
                                overlay_root/'candidate-v2-overlay', overlay)
    profile = diagnostic_profile(cap.read(reference/(ARM+'.json')), args.port)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    for script in ('diagnose_action_capture.py', 'analyze_action_capture.py'):
        shutil.copyfile(ROOT/'scripts'/script, root/'frozen/scripts'/script)
    runtime = root/'runtime'/ARM
    shutil.copytree(reference/'runtime'/ARM, runtime, ignore=ignore)
    shutil.copytree(overlay_root/'candidate-v2-overlay', root/'diagnostic-overlay', ignore=ignore)
    for row in overlay['files']:
        shutil.copyfile(root/'diagnostic-overlay'/row['path'], runtime/row['path'])
    cap.require(cap.hashes(runtime) == expected, 'Overlay affected unrelated runtime files')
    shutil.copyfile(overlay_root/'manifest.json', root/'overlay-manifest.json')
    shutil.copyfile(overlay_root/'diagnostic-only.patch', root/'diagnostic-only.patch')
    shutil.copyfile(reference/'fixture.json', root/'fixture.json')
    cap.save(root/(ARM+'.json'), profile)
    m = copy.deepcopy(ref)
    m.update(diagnostic_only=True, performance_claim_allowed=False,
        experiment_kind='action_capture_diagnostic_v1', created_at=time.time(), arms=[ARM],
        reference=str(reference), reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        overlay_manifest_sha256=cap.sha(root/'overlay-manifest.json'),
        source_sha256=cap.hashes(root/'frozen'), runtime_sha256={ARM:expected},
        profiles_sha256={ARM:cap.sha(root/(ARM+'.json'))}, port=args.port,
        scope='Sampled Action attribution only. Same natural-agent workload as reference; '
              'profile=8-step sampling, 4096 pending event pairs. No added CUDA synchronization. '
              'Host spans and nested GPU intervals overlap and are not idle. Diagnostic logging '
              'and events perturb execution; throughput is not performance acceptance.')
    cap.save(root/'manifest.json', m)
    cap.save(root/'status.json', dict(status='prepared', diagnostic_only=True, at=time.time()))


def validate(root):
    m = ready.verify(root)
    cap.require(m.get('diagnostic_only') is True and m.get('performance_claim_allowed') is False
        and m.get('experiment_kind') == 'action_capture_diagnostic_v1' and m['arms'] == [ARM],
        'Require one diagnostic arm')
    reference = Path(m['reference'])
    cap.require(cap.sha(reference/'manifest.json') == m['reference_manifest_sha256'], 'Reference changed')
    ref = reference_gate(reference)
    cap.require(m['workload'] == ref['workload'] and m['budget'] == ref['budget'] and
        m['fixture_sha256'] == ref['fixture_sha256'] and m['generation_sampling'] == ref['generation_sampling'],
        'Diagnostic changed workload, budget, fixture or sampling')
    cap.require(cap.read(root/(ARM+'.json')) == diagnostic_profile(cap.read(reference/(ARM+'.json')), m['port']),
                'Diagnostic changed non-observation profile controls')
    cap.require(cap.sha(root/'overlay-manifest.json') == m['overlay_manifest_sha256'], 'Overlay manifest changed')
    overlay = cap.read(root/'overlay-manifest.json')
    cap.require(cap.sha(root/'diagnostic-only.patch') == overlay['patch_sha256'], 'Diagnostic patch changed')
    expected = overlay_contract(reference/'runtime'/ARM, root/'diagnostic-overlay', overlay)
    cap.require(m['runtime_sha256'] == {ARM:expected}, 'Diagnostic runtime is not the reviewed overlay')
    return m


async def run(args):
    root = args.output.resolve()
    cap.require(ROOT.resolve() == root/'frozen', 'Run the frozen diagnostic entry point')
    m = validate(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic diagnostic restart')
    with open('/tmp/kvmem-coordination-qualification.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        loop, task = asyncio.get_running_loop(), asyncio.current_task()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, task.cancel)
        original_wait = ready.bench.q.wait_idle

        async def checked_wait(path, seconds):
            await original_wait(path, seconds)
            validate(root)

        ready.bench.q.wait_idle = checked_wait
        try:
            result = await ready.run_arm(root, ARM, m)
            cap.save(root/'diagnostic.json', dict(diagnostic_only=True, performance_claim_allowed=False,
                                                 result=result, scope=m['scope']))
            cap.save(root/'status.json', dict(status='diagnostic_completed', at=time.time()))
        except BaseException as error:
            cap.save(root/'status.json', dict(status='interrupted' if isinstance(error, asyncio.CancelledError)
                                             else 'failed', error=repr(error), at=time.time()))
            raise
        finally:
            ready.bench.q.wait_idle = original_wait
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.remove_signal_handler(sig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--overlay', type=Path)
    p.add_argument('--port', type=int)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    a = p.parse_args()
    if a.run_prepared:
        if any(x is not None for x in (a.reference, a.overlay, a.port)):
            p.error('Frozen run only accepts its output directory')
        asyncio.run(run(a))
    else:
        if a.reference is None or a.overlay is None or a.port is None:
            p.error('Preparation requires reference, reviewed overlay and port')
        prepare(a)


if __name__ == '__main__':
    main()
