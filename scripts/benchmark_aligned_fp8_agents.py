"""Freeze a new-runtime FP8 baseline using the original natural G16 agent replay.

This is a new protocol, not a relaxation of the old weight-only qualification.
The original eight trajectories, sampling, tools, K64/G32 budgets and complete
CPU history remain fixed. A newly frozen FP8 cold/rolling byte qualification
must pass before each runtime can serve a measured window. Diagnostics are
separate instances and cannot qualify as throughput improvements.
"""
import argparse
import asyncio
import copy
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/benchmark_aligned_fp8_agents.py'
KIND = 'aligned_fp8_agents_v1'
ARM = 'rolling-candidate-v2'
TREES = ('src', 'csrc', 'scripts')
EXTRA = (ENTRY, 'scripts/analyze_agent_scheduling.py', 'scripts/profile_capacity_chain.py',
         'scripts/host_admission_calibration.py')
CUDA_CACHE_GC = dict(garbage_collection_threshold=.9, per_process_memory_fraction=.99)


def context_profile(profile):
    """Normalize only the explicitly declared CUDA cache-policy ablation."""
    result = copy.deepcopy(profile)
    policy = result.pop('cuda_allocator_conf', None)
    cap.require(policy is None or policy == CUDA_CACHE_GC,
        'Only the declared native90% cache GC /99% process policy may differ')
    return result


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


chain = module('aligned_fp8_chain', ROOT/'scripts/benchmark_capacity_chain.py')
weight = module('aligned_fp8_weight', ROOT/'scripts/benchmark_weight_precision.py')
cap = chain.cap
host = module('aligned_fp8_host_calibration', ROOT/'scripts/host_admission_calibration.py')


def production(mapping):
    return {k:v for k,v in mapping.items() if k.startswith(('src/', 'csrc/'))}


def runtime_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in TREES for p in sorted((root/folder).rglob('*'))
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def declared_delta(before, after, allowed=None):
    before, after = production(before), production(after)
    delta = {k:[before.get(k), after.get(k)] for k in sorted(before.keys() | after.keys())
        if before.get(k) != after.get(k)}
    if allowed is not None:
        cap.require(set(delta) == set(allowed), 'Candidate production changes differ from the explicit file declaration')
        cap.require(bool(delta), 'Candidate must contain an explicitly declared production change')
    return delta


def checked_workload(duration, diagnostic):
    cap.require(type(diagnostic) is int and diagnostic in (0, 32), 'Diagnostic length must be 0 or 32 steps')
    cap.require(type(duration) in (int, float) and 180 <= duration <= 300,
        'Performance windows must be fixed in [180,300] seconds')
    return dict(chain.WORKLOAD, duration_seconds=60 if diagnostic else float(duration))


def selected_profile(reference, port, root, diagnostic, *, cuda_allocator_policy=None):
    cap.require(type(port) is int and 1024 <= port <= 65534, 'Dedicated local server and qualification ports required')
    profile = copy.deepcopy(reference)
    cap.require(profile.get('profiler_config') is None and profile['max_num_seqs'] == 8 and
        profile['max_model_len'] == 262144 and profile['draft_method'] == 'mtp' and
        profile['draft_tokens'] == 4 and profile['execution_mode'] == 'compiled_graph',
        'Require the original C8/256Ki/MTP4 compiled FP8 profile')
    kv = profile['kvmem']
    cap.require(kv['select_budget'] == 65536 and kv['gen_budget'] == 32768 and
        kv['archive_backend'] == 'cpu' and kv['cpu_read_cache_bytes'] == 100_000_000_000 and
        kv.get('graph_mode', 'off') == 'off' and not kv.get('repository_gc', False),
        'Keep K64/G32/CPU100GB and graph off; no SSD or history collection')
    profile['port'] = port
    cap.require(cuda_allocator_policy in (None, 'native_gc_90'), 'Unknown CUDA allocator policy')
    if cuda_allocator_policy:
        cap.require(not diagnostic, 'Allocator ablation must have its own unprofiled qualification')
        profile['cuda_allocator_conf'] = dict(CUDA_CACHE_GC)
    if diagnostic:
        from vllm_kvmem.profiling import native_profiler_config
        profile['profiler_config'] = native_profiler_config(root/'traces', diagnostic)
    return profile


def reference_proof(root):
    gate = module('aligned_fp8_reference', root/'frozen/scripts/benchmark_weight_precision.py')
    m = gate.verify(root)
    cap.require(cap.read(root/'status.json').get('status') == 'completed_pending_review' and
        cap.read(root/ARM/'deployment-acceptance.json').get('accepted') is True,
        'Original FP8 reference must have completed and passed deployment checks')
    cap.require(m['workload'] == chain.WORKLOAD and m['arms'] == [ARM],
        'Reference must be the original natural eight-agent G16 protocol')
    return m


def alignment_proof(root, runtime, *, completed):
    """Bind the independent qw3-policy check; it does not qualify FP8 weights."""
    m = cap.read(root/'manifest.json')
    cap.require(m.get('qualification_script') == 'qualify_context_alignment.py',
        'Require an independent context-alignment qualification')
    cap.require({'scripts/qualify_context_alignment.py', 'scripts/qualify_block_graph.py'} <=
        m['runtime_sha256'].keys(), 'Alignment client and evidence checker must be frozen')
    expected = {k:v for k,v in runtime.items() if k.startswith('src/')}
    actual = {k:v for k,v in m['runtime_sha256'].items() if k.startswith('src/')}
    cap.require(actual == expected and cap.hashes(root/'runtime') == m['runtime_sha256'],
        'Alignment proof used another production source')
    cap.require(cap.sha(root/'profile.json') == m['profile_sha256'] and
        cap.sha(Path(m['fixture'])) == m['fixture_sha256'], 'Alignment fixture/profile changed')
    if completed:
        result = cap.read(root/'qualification/results.json')
        cleanup = cap.read(root/'cleanup.json')
        cap.require(cap.read(root/'status.json').get('status') == 'passed' and
            result.get('status') == 'passed' and result.get('all_owned_sessions_released') is True and
            cleanup.get('remaining_owned_process_groups') == {} and cleanup.get('remaining_gpu_pids') == [],
            'Alignment qualification has not passed and completely cleaned up')
    return dict(path=str(root), manifest_sha256=cap.sha(root/'manifest.json'),
        scope='Context-policy source identity only; a separate exact-runtime FP8 byte qualification is mandatory')


def fp8_qualification(root, profile, runtime, *, completed):
    gate = module('aligned_fp8_cold_gate', root/'frozen/scripts/qualify_rolling_window.py')
    m = gate.verify(root)
    cap.require(m['runtime_sha256'] == runtime, 'FP8 cold qualification runtime differs from measurement')
    qp = cap.read(root/'profile.json')
    unprofiled = dict(profile, profiler_config=None)
    cap.require(qp == gate.qualification_profile(unprofiled, qp['port']),
        'FP8 qualification and performance model or context policy differ')
    if completed:
        result = cap.read(root/'results.json')
        cap.require(cap.read(root/'status.json').get('status') == 'completed_pending_review' and
            result.get('status') == 'passed' and result.get('cleanup_completed') is True and
            len(result.get('history_proofs', [])) == 6, 'Exact-runtime FP8 qualification has not passed')
        cap.require(gate.oracle_evidence(root/'server/capture', cap.read(root/'workload/traces.json')) ==
            result.get('oracle'), 'FP8 target/MTP/recurrent/rolling byte evidence changed')
        counts = result.get('native_reconciliation', {})
        cap.require(counts.get('passed') is True and counts.get('preemptions') == 0 and
            type(counts.get('native_tokens')) in (int, float) and counts['native_tokens'] > 0 and
            counts.get('native_tokens') == counts.get('client_tokens'), 'FP8 qualification output accounting failed')
        cap.require((root/'process/post-stop.json').is_file() and
            (root/'process/qualification-cleanup.json').is_file(),
            'FP8 qualification lacks protected owned-process cleanup and device-exit proof')
        cleanup = cap.read(root/'process/post-stop.json')
        binding = cap.read(root/'process/qualification-cleanup.json')
        cap.require(cleanup.get('gpu_empty') is True and cleanup.get('observed_gpu_pids') == [] and
            binding == dict(protocol='aligned_fp8_qualification_cleanup_v1',
                qualification_manifest_sha256=cap.sha(root/'manifest.json'),
                performance_manifest_sha256=cap.sha(root.parent/'manifest.json'),
                post_stop_sha256=cap.sha(root/'process/post-stop.json')),
            'FP8 qualification lacks protected owned-process cleanup and device-exit proof')
    return dict(path=str(root), manifest_sha256=cap.sha(root/'manifest.json'))


def baseline_proof(root):
    gate = module('aligned_fp8_baseline_gate', root/'frozen'/ENTRY)
    m = gate.verify(root, completed=False)
    # CPU-only revalidation still requires completed model evidence; only the
    # live physical-device probe is deferred to the actual GPU run.
    if 'alignment_qualification' in m and 'fp8_runtime_qualification' in m:
        runtime = m['runtime_sha256'][ARM]
        alignment_proof(Path(m['alignment_qualification']['path']), runtime, completed=True)
        fp8_qualification(Path(m['fp8_runtime_qualification']['path']),
            cap.read(root/(ARM+'.json')), runtime, completed=True)
    result = cap.read(root/ARM/'results.json')
    history = cap.read(root/ARM/'own-history-proof.json')
    cap.require(m.get('experiment_kind') == KIND and m.get('performance_eligible') is True and
        cap.read(root/'status.json').get('status') == 'completed_pending_review' and
        cap.read(root/ARM/'deployment-acceptance.json').get('accepted') is True and
        result.get('drained') is True and result.get('native_client_tokens_equal') is True and
        result.get('explicit_session_release_verified') is True and result.get('preemptions') == 0 and
        history.get('passed') is True and history.get('complete_own_outputs') is True,
        'Candidate requires a completed, accepted, fully reconciled new-runtime baseline')
    files = ('manifest.json', 'status.json', ARM+'/results.json', ARM+'/deployment-acceptance.json',
        ARM+'/own-history-proof.json', ARM+'/tools/explicit-release-proof.json',
        ARM+'/tools/window.json', ARM+'/tools/traces.json', ARM+'/tools/events.jsonl',
        ARM+'/history_publications.jsonl', ARM+'/post-stop.json')
    return m, dict(path=str(root), evidence_sha256={name:cap.sha(root/name) for name in files})


def calibrate_host(reference, output):
    reference, output = reference.resolve(), output.resolve()
    cap.require(not output.exists() and not output.is_relative_to(reference),
        'Calibration must be a new file outside the original run')
    manifest, binding = baseline_proof(reference)
    report = host.source_report(reference, manifest, binding, cap.sha)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    return report


def host_contract(original, profile, workload, binding):
    if binding is None:
        return copy.deepcopy(original['budget'])
    path = Path(binding['path'])
    cap.require(cap.sha(path) == binding['sha256'], 'Host calibration artifact changed')
    report = cap.read(path)
    reference = Path(report['source_run'])
    source, proof = baseline_proof(reference)
    cap.require(host.source_report(reference, source, proof, cap.sha) == report,
        'Host calibration is not reproducible from the accepted all-phase source')
    cap.require(source['budget'] == original['budget'] and source['workload'] == workload and
        source['fixture_sha256'] == original['fixture_sha256'] and
        source['device_memory_identity'] == original['device_memory_identity'] and
        context_profile(cap.read(reference/(ARM+'.json'))) ==
        dict(context_profile(profile), port=source['port']),
        'Host calibration requires the identical model/device/full-history replay')
    cap.require(cap.sha(path) == binding['sha256'], 'Host calibration changed during verification')
    return host.calibrated_budget(original['budget'], report)


def diagnostic_contract(baseline, baseline_profile, profile, runtime, fixture_sha):
    cap.require(runtime == baseline['runtime_sha256'][ARM] and
        baseline_profile == dict(profile, port=baseline['port'], profiler_config=None) and
        baseline['fixture_sha256'] == fixture_sha,
        'Diagnostic must use the exact accepted baseline runtime, profile and fixture')


def prepare(args):
    root, reference, runtime = args.output.resolve(), args.reference.resolve(), args.runtime.resolve()
    cap.require(not root.exists(), 'Use a fresh experiment directory; never overwrite an attempt')
    cap.require(not root.is_relative_to(runtime/'src') and not root.is_relative_to(runtime/'scripts'),
        'Output must be outside runtime source trees')
    original = reference_proof(reference)
    before = runtime_hashes(runtime)
    cap.require('src/vllm_kvmem/serve.py' in before, 'Missing production runtime')
    alignment = alignment_proof(args.alignment_qualification.resolve(), before, completed=False)
    workload = checked_workload(args.duration_seconds, args.diagnostic_steps)
    allocator_policy = getattr(args, 'cuda_allocator_policy', None)
    profile = selected_profile(cap.read(reference/(ARM+'.json')), args.port, root,
        args.diagnostic_steps, cuda_allocator_policy=allocator_policy)
    cap.require(not allocator_policy or (args.baseline and
        'src/vllm_kvmem/serve.py' in args.allow_production_change),
        'Allocator ablation requires an accepted baseline and declared launcher change')
    calibration_path = getattr(args, 'host_calibration', None)
    cap.require(not (args.diagnostic_steps and calibration_path),
        'Host calibration currently applies to unprofiled performance only')
    calibration = (dict(path=str(calibration_path.resolve()), sha256=cap.sha(calibration_path))
        if calibration_path else None)
    budget = host_contract(original, profile, workload, calibration)
    baseline = None
    diagnostic_reference = None
    if args.diagnostic_steps:
        cap.require(args.diagnostic_reference is not None and not args.baseline,
            'A diagnostic requires a completed unprofiled --diagnostic-reference')
        dm, diagnostic_reference = baseline_proof(args.diagnostic_reference.resolve())
        diagnostic_contract(dm, cap.read(args.diagnostic_reference/(ARM+'.json')),
            profile, before, original['fixture_sha256'])
    else:
        cap.require(args.diagnostic_reference is None, 'Diagnostic reference requires --diagnostic-steps32')
    if args.baseline:
        base = args.baseline.resolve()
        bm, baseline = baseline_proof(base)
        cap.require(bm['workload'] == workload and not args.diagnostic_steps and
            context_profile(cap.read(base/(ARM+'.json'))) == dict(context_profile(profile), port=bm['port']) and
            bm['fixture_sha256'] == original['fixture_sha256'] and bm['budget'] == budget and
            bm.get('host_calibration') == calibration,
            'Candidate changed a non-runtime comparison parameter')
        delta = declared_delta(bm['runtime_sha256'][ARM], before, args.allow_production_change)
        baseline['production_delta'] = delta
    else:
        cap.require(not args.allow_production_change, 'Production change allowlist requires --baseline')
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    for name in EXTRA:
        shutil.copyfile(ROOT/name, root/'frozen'/name)
    for folder in TREES:
        shutil.copytree(runtime/folder, root/'runtime'/ARM/folder, ignore=ignore)
    cap.require(runtime_hashes(runtime) == before == cap.hashes(root/'runtime'/ARM),
        'Source changed while freezing the experiment')
    shutil.copytree(reference/'diagnostics', root/'diagnostics', ignore=ignore)
    for name in ('fixture.json', 'native.json', 'installed-native-identity.json', 'observer.json'):
        shutil.copyfile(reference/name, root/name)
    cap.save(root/(ARM+'.json'), profile)
    cap.save(root/'qualification-base-profile.json', dict(profile, profiler_config=None))
    qualification_reference = getattr(args, 'fp8_qualification_reference', None)
    cap.require(not (diagnostic_reference and qualification_reference),
        'Diagnostic already binds its completed FP8 qualification')
    if qualification_reference:
        # Reusing exactly the already-qualified runtime avoids repeating a
        # model test when only the controller's launch allowance changed.
        proof = fp8_qualification(qualification_reference.resolve(), profile, before, completed=True)
    elif diagnostic_reference:
        proof = dm['fp8_runtime_qualification']
    else:
        qualifier = module('aligned_fp8_prepare_cold', ROOT/'scripts/qualify_rolling_window.py')
        qualifier.prepare(SimpleNamespace(output=root/'qualification', config=root/'qualification-base-profile.json',
            fixture=root/'fixture.json', candidate=root/'runtime'/ARM, port=args.port+1))
        proof = fp8_qualification(root/'qualification', profile, before, completed=False)
    m = copy.deepcopy(original)
    m.update(experiment_kind=KIND, created_at=time.time(), port=args.port,
        source_sha256=cap.hashes(root/'frozen'), runtime_sha256={ARM:before},
        profiles_sha256={a:cap.sha(root/(a+'.json')) for a in chain.ARMS},
        workload=workload, new_runtime_reference=str(reference),
        new_runtime_reference_sha256=cap.sha(reference/'manifest.json'),
        production_delta_from_old_reference=declared_delta(original['runtime_sha256'][ARM], before),
        alignment_qualification=alignment, fp8_runtime_qualification=proof,
        host_calibration=calibration, budget=budget,
        cuda_allocator_policy=allocator_policy,
        weight_model_identity=weight.model_identity(profile['model']), baseline=baseline,
        diagnostic_reference=diagnostic_reference,
        diagnostic_steps=args.diagnostic_steps, performance_eligible=not args.diagnostic_steps,
        scope='New-runtime FP8 natural G16 replay: eight original tasks from the previously run16, '
            'own complete outputs, original full tool messages/waits and sampling, two warmups, '
            'K64/G32,256Ki CPU history, output ceiling16Ki. Old FP8 result is reference only. '
            'Graph off; no SSD, recomputation, clipping or pressure-driven epoch truncation. '
            +('Independent60s/32-step diagnostic; rates are ineligible as performance evidence.'
              if args.diagnostic_steps else 'Fixed-window output rate; not on-policy agent solve rate or pure decode kernel speed.'))
    cap.save(root/'manifest.json', m)
    cap.save(root/'status.json', dict(status='prepared', gpu_started=False, at=time.time()))
    return m


def verify(root, *, completed=True):
    m = chain.ready.verify(root)
    cap.require(ROOT == root/'frozen' and m.get('experiment_kind') == KIND,
        'Run the new frozen aligned FP8 entry')
    reference = Path(m['new_runtime_reference'])
    cap.require(cap.sha(reference/'manifest.json') == m['new_runtime_reference_sha256'], 'Original reference changed')
    original = reference_proof(reference)
    profile = cap.read(root/(ARM+'.json'))
    expected = selected_profile(cap.read(reference/(ARM+'.json')), m['port'], root,
        m['diagnostic_steps'], cuda_allocator_policy=m.get('cuda_allocator_policy'))
    cap.require(profile == expected and m['workload'] == checked_workload(
        300 if m['diagnostic_steps'] else m['workload']['duration_seconds'], m['diagnostic_steps']) and
        m['performance_eligible'] == (m['diagnostic_steps'] == 0), 'Profile or timing protocol changed')
    immutable = ('fixture_sha256', 'fixture_directory', 'fixture_identity', 'generation_sampling',
        'arms', 'host_admission', 'idle_seconds', 'native_source', 'installed_native_sha256',
        'device_memory_identity', 'minimum_device_free_mib', 'release_qualification')
    cap.require(all(m[k] == original[k] for k in immutable), 'Original fixture, full resource budget or release contract changed')
    cap.require(m['budget'] == host_contract(original, profile, m['workload'], m.get('host_calibration')),
        'Full resource budget or calibrated host envelope changed')
    cap.require(not (m['diagnostic_steps'] and m.get('host_calibration')),
        'Host calibration currently applies to unprofiled performance only')
    cap.require(weight.model_identity(profile['model']) == m['weight_model_identity'], 'FP8 model files changed')
    runtime = m['runtime_sha256'][ARM]
    cap.require(alignment_proof(Path(m['alignment_qualification']['path']), runtime, completed=completed) ==
        m['alignment_qualification'], 'Alignment qualification changed')
    cap.require(fp8_qualification(Path(m['fp8_runtime_qualification']['path']), profile, runtime, completed=completed) ==
        m['fp8_runtime_qualification'], 'Exact-runtime FP8 qualification changed')
    cap.require(declared_delta(original['runtime_sha256'][ARM], runtime) == m['production_delta_from_old_reference'],
        'New baseline production delta changed')
    baseline = m.get('baseline')
    if baseline:
        base = Path(baseline['path']); bm, bp = baseline_proof(base)
        cap.require(bp == {k:v for k,v in baseline.items() if k != 'production_delta'}, 'Comparison baseline changed')
        cap.require(bm['workload'] == m['workload'] and not m['diagnostic_steps'] and
            context_profile(cap.read(base/(ARM+'.json'))) == dict(context_profile(profile), port=bm['port']) and
            bm['fixture_sha256'] == m['fixture_sha256'] and bm['budget'] == m['budget'] and
            bm.get('host_calibration') == m.get('host_calibration'), 'Comparison baseline parameters changed')
        cap.require(declared_delta(bm['runtime_sha256'][ARM], runtime,
            baseline['production_delta']) == baseline['production_delta'], 'Comparison production delta changed')
        cap.require(not m.get('cuda_allocator_policy') or 'src/vllm_kvmem/serve.py' in baseline['production_delta'],
            'Allocator policy requires its declared qualified launcher')
    diagnostic_reference = m.get('diagnostic_reference')
    if m['diagnostic_steps']:
        cap.require(diagnostic_reference is not None, 'Diagnostic has no completed performance reference')
        dm, dp = baseline_proof(Path(diagnostic_reference['path']))
        cap.require(dp == diagnostic_reference, 'Diagnostic performance reference changed')
        diagnostic_contract(dm, cap.read(Path(dp['path'])/(ARM+'.json')), profile, runtime, m['fixture_sha256'])
    chain.installed_native(root/'installed-native-identity.json')
    if completed:
        cap.require(chain.device_identity() == m['device_memory_identity'], 'Physical device identity changed')
    return m


def separated_report(root):
    analysis = module('aligned_fp8_analysis', ROOT/'scripts/analyze_agent_scheduling.py')
    directory = root/ARM
    window = cap.read(directory/'tools/window.json')
    result = cap.read(directory/'results.json')
    model = analysis.analyze(directory/'server', begin=window['begin'], end=window['end'])
    report = dict(performance_eligible=cap.read(root/'manifest.json')['performance_eligible'],
        accepted_output_tokens_per_wall_second=result['output_tokens_per_second'],
        completed_turns=result['requests_completed_in_window'],
        actual_public_prefill=result.get('public_work'),
        scheduling=model, scope='Accepted output uses the whole fixed service window. Public prefill counts '
        'actually forwarded prompt positions; restored KV and private Query excluded. Pure/mixed step '
        'times include MTP/capture and do not attribute accepted output to a GPU step. Host waits '
        'overlap across requests and are not GPU idle seconds.')
    cap.save(root/'separated-metrics.json', report)
    manifest = cap.read(root/'manifest.json')
    if manifest.get('host_calibration'):
        with (directory/'gpu_samples.jsonl').open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        envelope = host.observed_envelope(manifest['budget'],
            cap.read(root/(ARM+'-host-admission.json')), rows)
        envelope['calibration'] = manifest['host_calibration']
        cap.save(root/'host-memory-envelope.json', envelope)
    return report


async def finish_cleanup(awaitable):
    """Repeated cancellation cannot abandon an already-owned cleanup job."""
    task = asyncio.create_task(awaitable)
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    value = task.result()
    if cancelled:
        raise asyncio.CancelledError
    return value


async def guarded_arm(original, root, arm, *args, **kwargs):
    """Cancel the measured task once; retain cleanup and post-stop on every exit."""
    task = asyncio.create_task(original(root, arm, *args, **kwargs))
    cancelled = False
    try:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not cancelled:
                    task.cancel()
                    cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return task.result()
    finally:
        # run_arm's own finally has completed its stop_owned call first. A
        # failure before server launch still records the observed device state.
        (root/arm).mkdir(exist_ok=True)
        await finish_cleanup(asyncio.to_thread(chain.post_stop_proof, root, arm))


async def run_qualification(args):
    """Protect the exact-runtime FP8 oracle's owned cleanup under the GPU lock."""
    root = args.output.resolve()
    measurement = verify(root, completed=False)
    cap.require(not measurement['diagnostic_steps'],
        'Diagnostic reuses its completed exact-runtime FP8 qualification; do not rerun it')
    qualification = root/'qualification'
    cap.require(Path(measurement['fp8_runtime_qualification']['path']) == qualification,
        'This run reuses a completed exact-runtime qualification; run the prepared performance directly')
    gate = module('aligned_fp8_run_cold', qualification/'frozen/scripts/qualify_rolling_window.py')
    manifest = gate.verify(qualification)
    cap.require(cap.read(qualification/'status.json').get('status') == 'prepared',
        'Use a fresh qualification; no retry of an incomplete attempt')
    async def serve(parent, arm):
        return await gate.run_server(qualification, manifest)
    loop, current = asyncio.get_running_loop(), asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, current.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Recheck source and profiles after ownership, before loading.
            cap.require(gate.verify(qualification) == manifest, 'Qualification changed before GPU admission')
            await guarded_arm(serve, qualification, 'process')
            cap.save(qualification/'process/qualification-cleanup.json', dict(
                protocol='aligned_fp8_qualification_cleanup_v1',
                qualification_manifest_sha256=cap.sha(qualification/'manifest.json'),
                performance_manifest_sha256=cap.sha(root/'manifest.json'),
                post_stop_sha256=cap.sha(qualification/'process/post-stop.json')))
            cap.save(qualification/'status.json', dict(status='completed_pending_review', at=time.time()))
    except BaseException as error:
        cap.save(qualification/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


async def run(args):
    root = args.output.resolve(); m = verify(root)
    original_verify, original_workload = chain.verify, chain.work.workload
    original_arm = chain.ready.run_arm
    chain.verify = verify
    async def owned_arm(*a, **kw):
        return await guarded_arm(original_arm, *a, **kw)
    chain.ready.run_arm = owned_arm
    async def diagnostic(client, profile, fixture, directory, values):
        original_retirement = chain.work.wait_retirement
        started = False
        url = f"http://127.0.0.1:{profile['port']}"
        async def retirement(*a, **kw):
            nonlocal started
            state = await original_retirement(*a, **kw)
            views = kw.get('expected_views') or {}
            if not started and len(views) == 8 and all(n > 1 for n in views.values()):
                response = await client.post(url+'/start_profile', timeout=180)
                response.raise_for_status(); started = True
                cap.save(root/'profiler-start.json', dict(at=time.monotonic(), after_complete_warmup=True))
            return state
        chain.work.wait_retirement = retirement
        try:
            result = await original_workload(client, profile, fixture, directory, values)
            cap.require(started, 'No complete warmup boundary; profiler did not start')
            result.update(performance_eligible=False, diagnostic_only=True)
            return result
        finally:
            chain.work.wait_retirement = original_retirement
            if started:
                response = await client.post(url+'/stop_profile', timeout=180)
                response.raise_for_status()
    try:
        if m['diagnostic_steps']:
            chain.work.workload = diagnostic
        await chain.run(args)
        separated_report(root)
        if m['diagnostic_steps']:
            profiler = module('aligned_fp8_trace', ROOT/'scripts/profile_capacity_chain.py')
            cap.save(root/'trace-coverage.json', profiler.trace_coverage(list((root/'traces').glob('*.pt.trace.json*'))))
            cap.save(root/'status.json', dict(status='diagnostic_completed', performance_eligible=False, at=time.time()))
    except BaseException as error:
        cap.save(root/'status.json', dict(status='failed', error=repr(error), at=time.time(),
            performance_eligible=m['performance_eligible']))
        raise
    finally:
        chain.verify, chain.work.workload = original_verify, original_workload
        chain.ready.run_arm = original_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare-only', action='store_true')
    modes.add_argument('--run-prepared', action='store_true')
    modes.add_argument('--run-qualification', action='store_true')
    modes.add_argument('--calibrate-host', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    for name in ('reference', 'runtime', 'alignment-qualification', 'baseline', 'diagnostic-reference',
                 'host-calibration', 'fp8-qualification-reference'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--duration-seconds', type=float, default=300)
    parser.add_argument('--diagnostic-steps', type=int, choices=(0, 32), default=0)
    parser.add_argument('--allow-production-change', action='append', default=[])
    parser.add_argument('--cuda-allocator-policy', choices=('native_gc_90',))
    parser.add_argument('--port', type=int, default=18608)
    args = parser.parse_args()
    if args.calibrate_host:
        cap.require(args.reference is not None, 'Calibration requires an accepted performance reference')
        report = calibrate_host(args.reference, args.output)
        print(json.dumps(dict(output=str(args.output), extra_bytes=report['extra_bytes'],
            required_available_bytes=report['required_available_bytes'])))
        return
    if args.run_prepared:
        return asyncio.run(run(args))
    if args.run_qualification:
        return asyncio.run(run_qualification(args))
    cap.require(all((args.reference, args.runtime, args.alignment_qualification)),
        'Preparation requires original FP8 reference, new runtime and frozen alignment qualification')
    prepare(args)


if __name__ == '__main__':
    main()
