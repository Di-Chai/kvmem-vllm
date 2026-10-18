"""Bounded mixed-step profiling with the original eight G16 agent trajectories.

This is a separate diagnostic process. Start32 CPU/CUDA profiler steps only
after both original warmup rounds retire, then run60 seconds with original tool
waits and normal drain. Instrumented rates never qualify as performance wins.
"""
import argparse
import asyncio
import copy
import fcntl
import gzip
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/profile_capacity_chain.py'
SERVER_ENTRY = 'scripts/qualify_rolling_window.py'
DIAGNOSTIC_SOURCES = (ENTRY, SERVER_ENTRY)
SCOPES = 'src/vllm_kvmem/profile_scopes.py'
STEPS = 32
DURATION = 60


def module(name, path):
    spec = importlib.util.spec_from_file_location(name,path)
    value = importlib.util.module_from_spec(spec);sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


chain = module('profile_capacity_chain_base',ROOT/'scripts/benchmark_capacity_chain.py')
shared = module('profile_capacity_chain_server',ROOT/'scripts/qualify_rolling_window.py')
cap = chain.cap


def verified_reference(reference):
    # The baseline verifier deliberately checks its own frozen source root.
    # Loading it from this diagnostic's tree would verify the wrong identity.
    frozen = module('profile_capacity_frozen_reference',
                    reference/'frozen/scripts/benchmark_capacity_chain.py')
    return frozen.verify(reference)


def configuration(base, root, port):
    from vllm_kvmem.profiling import native_profiler_config
    cap.require(base.get('profiler_config') is None and base.get('max_num_seqs') == 8,
                'Require the unprofiled eight-user reference')
    cap.require(type(port) is int and 1024 <= port <= 65535, 'Dedicated loopback port required')
    result = copy.deepcopy(base)
    result.update(port=port,profiler_config=native_profiler_config(root/'traces',STEPS))
    return result


def freeze_diagnostic_sources(reference, root):
    """Carry the diagnostic server's budget checks into the frozen runner too."""
    ignore = shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    for entry in DIAGNOSTIC_SOURCES:
        shutil.copyfile(ROOT/entry, root/'frozen'/entry)
    return {entry: cap.sha(root/'frozen'/entry) for entry in DIAGNOSTIC_SOURCES}


def prepare(args):
    root,reference = args.output.resolve(),args.reference.resolve()
    cap.require(not root.exists(), 'Use a fresh diagnostic; no overwrite or retry')
    before = verified_reference(reference)
    cap.require(before.get('release_qualification') and before['arms'] == [chain.ARMS[1]],
                'Require the fresh release performance reference')
    profile = configuration(cap.read(reference/(chain.ARMS[1]+'.json')),root,args.port)
    root.mkdir()
    ignore = shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    diagnostic_source_delta = freeze_diagnostic_sources(reference, root)
    shutil.copytree(reference/'runtime'/chain.ARMS[1],root/'runtime',ignore=ignore)
    shutil.copyfile(ROOT/SCOPES,root/'runtime'/SCOPES)
    shutil.copyfile(reference/'fixture.json',root/'fixture.json')
    cap.save(root/'profile.json',profile)
    cap.save(root/'manifest.json',dict(kind='capacity_mixed_steps_profile_v1',at=time.time(),
        reference=str(reference),reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        source_sha256=cap.hashes(root/'frozen'),runtime_sha256=cap.hashes(root/'runtime'),
        diagnostic_source_delta=diagnostic_source_delta,
        profile_sha256=cap.sha(root/'profile.json'),fixture_sha256=cap.sha(root/'fixture.json'),
        runtime_delta={SCOPES:cap.sha(root/'runtime'/SCOPES)},budget=before['budget'],
        workload=dict(before['workload'],duration_seconds=DURATION),steps=STEPS,port=args.port,
        idle_seconds=10,performance_eligible=False,
        scope='Original inputs, own accepted outputs, tool waits and two warmup rounds. '
              'Separate60s diagnostic with normal drain and32 profiled steps; not a300s performance comparison.'))
    cap.save(root/'status.json',dict(status='prepared',gpu_started=False,at=time.time()))


def verify(root, *, require_performance=False):
    m = cap.read(root/'manifest.json');reference = Path(m['reference'])
    cap.require(m['kind'] == 'capacity_mixed_steps_profile_v1' and m['performance_eligible'] is False,
                'Diagnostic must never be eligible as a performance result')
    cap.require(cap.sha(reference/'manifest.json') == m['reference_manifest_sha256'], 'Reference changed')
    before = verified_reference(reference)
    delta = {entry: cap.sha(ROOT/entry) for entry in DIAGNOSTIC_SOURCES}
    cap.require(m.get('diagnostic_source_delta') == delta and
        m['source_sha256'] == {**before['source_sha256'],**delta} and
        cap.hashes(root/'frozen') == m['source_sha256'], 'Diagnostic source delta changed')
    cap.require(set(m['runtime_delta']) == {SCOPES} and
        m['runtime_sha256'] == {**before['runtime_sha256'][chain.ARMS[1]],**m['runtime_delta']} and
        cap.hashes(root/'runtime') == m['runtime_sha256'], 'Only diagnostic CPU scopes may change runtime')
    expected = configuration(cap.read(reference/(chain.ARMS[1]+'.json')),root,m['port'])
    cap.require(cap.read(root/'profile.json') == expected and
        cap.sha(root/'profile.json') == m['profile_sha256'] and
        cap.sha(root/'fixture.json') == m['fixture_sha256'] == before['fixture_sha256'],
        'Profile or full original fixture changed')
    cap.require(m['workload'] == dict(before['workload'],duration_seconds=DURATION) and
        m['steps'] == STEPS and m['budget'] == before['budget'] and m['idle_seconds'] == 10,
        'Diagnostic workload or complete CPU admission changed')
    if require_performance:
        cap.require(cap.read(reference/'status.json')['status'] == 'completed_pending_review' and
            cap.read(reference/chain.ARMS[1]/'deployment-acceptance.json')['accepted'] is True,
            'Finish and accept the unprofiled release comparison before collecting a diagnostic')
    return m,before


async def measured_workload(client,profile,fixture,directory,capture,*,manifest):
    work = chain.work
    old_retirement = work.wait_retirement
    started = False
    stopped = False
    profile_events = []
    url = f"http://127.0.0.1:{profile['port']}"
    async def retirement(*args,**kwargs):
        nonlocal started
        state = await old_retirement(*args,**kwargs)
        expected = kwargs.get('expected_views') or {}
        if not started and len(expected) == 8 and all(n > 1 for n in expected.values()):
            before = time.monotonic()
            response = await client.post(url+'/start_profile',timeout=180)
            response.raise_for_status()
            started = True
            profile_events.append(dict(stage='started_after_full_warmup',begin=before,
                end=time.monotonic(),expected_full_views=expected))
        return state
    work.wait_retirement = retirement
    try:
        settings = SimpleNamespace(**manifest['workload'],run_tag='mixed-profile',capture_directory=capture)
        reference = cap.read(Path(manifest['reference'])/'manifest.json')
        result = await chain.released_workload(work.workload,client,profile,fixture,directory,
            settings,reference['release_qualification'])
        cap.require(started, 'No full eight-user warmup boundary; profiler was never started')
        response = await client.post(url+'/stop_profile',timeout=180)
        response.raise_for_status()
        stopped = True
        profile_events.append(dict(stage='stopped_after_drain',at=time.monotonic()))
        cap.save(directory/'profile-control.json',profile_events)
        rows = cap.read(directory/'traces.json')
        cap.save(directory/'own-history-proof.json',chain.own_history_proof(fixture,rows,
            chain.ready.records(directory/'events.jsonl')))
        result.update(performance_eligible=False,diagnostic_only=True)
        return result
    finally:
        work.wait_retirement = old_retirement
        if started and not stopped:
            try:
                response = await client.post(url+'/stop_profile',timeout=30)
                response.raise_for_status()
                profile_events.append(dict(stage='stopped_on_failure',at=time.monotonic()))
            except Exception as error:
                profile_events.append(dict(stage='stop_failed',error=repr(error),at=time.monotonic()))
            if directory.exists():cap.save(directory/'profile-control.json',profile_events)


def trace_coverage(paths):
    native = [];scopes = {};kernels = 0
    pattern = re.compile(r'^execute_(?:\d+_)?context_(\d+)\([^)]*\)_generation_(\d+)\([^)]*\)$')
    cap.require(len(paths) == 1, 'Exactly one complete worker trace required')
    for path in paths:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path,'rt') as stream:events = json.load(stream)['traceEvents']
        for e in events:
            name = e.get('name','')
            native_event = e.get('cat') == 'user_annotation' and name.startswith('execute_')
            cpu_scope = e.get('cat') == 'user_annotation' and name.startswith('kvmem::')
            if native_event:
                cap.require(e.get('ph') == 'X', 'Incomplete native iteration annotation')
            if e.get('ph') != 'X':continue
            if native_event or e.get('cat') == 'kernel' or cpu_scope:
                cap.require(all(type(e.get(k)) in (int,float) and math.isfinite(e[k])
                    for k in ('ts','dur')) and e['dur']>0, 'Incomplete or invalid timed event')
            if e.get('cat') == 'kernel':kernels += 1
            if native_event:
                match = pattern.fullmatch(name)
                cap.require(match is not None and e.get('dur',0)>0, 'Incomplete native iteration annotation')
                native.append(dict(prefill_requests=int(match[1]),decode_requests=int(match[2])))
            if cpu_scope:
                entry = scopes.setdefault(name,dict(calls=0,summed_cpu_us=0))
                entry['calls'] += 1;entry['summed_cpu_us'] += e['dur']
    mixed = sum(r['prefill_requests']>0 and r['decode_requests']>0 for r in native)
    cap.require(len(native) == STEPS and kernels > 0, 'Trace did not cover32 native steps and CUDA kernels')
    cap.require(mixed > 0, 'Trace contains no mixed prefill/decode; retain it as negative coverage, no automatic retry')
    cap.require(all(name in scopes for name in ('kvmem::capture.submit','kvmem::capture.pack_fp8',
        'kvmem::capture.drain_requests','kvmem::action.capture')), 'Fine capture/attention scopes missing')
    return dict(native_steps=len(native),mixed_steps=mixed,kernels=kernels,
        composition=native,cpu_scopes=scopes,performance_eligible=False,
        scope='CPU ranges are nested and may overlap across threads; sums are not exclusive idle time. '
              'CUDA event correlation is still required to attribute host gaps and actual device cost.')


async def run_diagnostic_server(root, manifest):
    try:
        return await shared.run_server(root, manifest)
    finally:
        # run_server stops its owned model even when startup validation fails.
        # Preserve an independent exit observation for the queued next job.
        (root/'diagnostic').mkdir(exist_ok=True)
        chain.post_stop_proof(root, 'diagnostic')


async def run(args):
    import psutil
    root = args.output.resolve()
    cap.require(ROOT == root/'frozen', 'Run the frozen diagnostic entry')
    m,_ = verify(root,require_performance=True)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic retry')
    owner = psutil.Process()
    cap.save(root/'owner.json',dict(pid=owner.pid,birth=owner.create_time(),at=time.time(),argv=sys.argv))
    original = shared.workload
    async def workload(*values):return await measured_workload(*values,manifest=m)
    shared.workload = workload
    loop,current = asyncio.get_running_loop(),asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,current.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            await run_diagnostic_server(root,m)
        coverage = trace_coverage(list((root/'traces').glob('*.pt.trace.json*')))
        cap.save(root/'trace-coverage.json',coverage)
        cap.save(root/'status.json',dict(status='diagnostic_completed',performance_eligible=False,at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='failed',error=repr(error),at=time.time(),performance_eligible=False))
        raise
    finally:
        shared.workload = original
        for sig in (signal.SIGINT,signal.SIGTERM):loop.remove_signal_handler(sig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only',action='store_true');modes.add_argument('--run-prepared',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--port',type=int,default=18578)
    args = parser.parse_args()
    if args.run_prepared:return asyncio.run(run(args))
    cap.require(args.reference is not None, 'Explicit unprofiled release reference required')
    prepare(args)


if __name__ == '__main__':main()
