"""Isolated allocator diagnosis: exact cold twelve-agent warmup, then cleanup.

No GPU starts during preparation. This deliberately reproduces the .82 memory
configuration whose four-GiB headroom contract failed; it is not deployment
acceptance and does not produce a throughput window.
"""
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'scripts/diagnose_allocator_prefill.py'
ARM = 'kvmem-ready'
FILES = {'src/vllm_kvmem/'+name+'.py' for name in (
    'runner', 'mtp_runtime', 'capture_runtime', 'action_attention', 'action_scoring', 'allocator_diagnostic')}
KIND = 'allocator_cold_twelve_diagnostic_v1'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


ready = module('allocator_diagnostic_ready', ROOT/'scripts/benchmark_ready_agents.py')
cap = ready.cap


def transform_workload(source):
    """Two exact changes after the original warmup and retirement barrier."""
    replacements = (
        ('        started = time.monotonic()\n        wall = time.time()\n',
         '        allocator_phase("drain")\n        duration = 0.\n'
         '        started = time.monotonic()\n        wall = time.time()\n'),
        ('        results = await asyncio.gather(*(agent(i) for i in range(agents)), return_exceptions=True)\n',
         '        results = []  # Diagnostic ends after the unchanged warmup barrier.\n'))
    for before, after in replacements:
        cap.require(source.count(before) == 1, 'Frozen warmup transformation boundary changed')
        source = source.replace(before, after)
    return source


def diagnostic_summary(rows, window, phase_events):
    cap.require(window['begin'] == window['end'] and window['duration_seconds'] == 0,
                'Allocator diagnostic must have no measured window')
    cap.require(all(r['kind'] in ('warmup', 'cleanup') and r['status'] == 'completed' for r in rows),
                'Diagnostic created a producer or incomplete stream')
    return dict(status='completed', diagnostic_only=True, performance_claim_allowed=False,
        duration_seconds=0, total_stream_output_tokens=sum(len(r['output_ids']) for r in rows),
        stream_output_tokens_by_kind={kind:sum(len(r['output_ids']) for r in rows if r['kind'] == kind)
                                      for kind in ('warmup', 'cleanup')},
        scope='Original cold twelve-agent inputs and two complete warmup rounds, followed by normal cleanup. '
              'No formal interval, output rate, GPU-idle inference, or deployment acceptance.')


def no_window(*args, **kwargs):
    return dict(available=False, diagnostic_only=True, reason='No formal interval was executed')


def add_diagnostic_environment(plan, root):
    result = copy.deepcopy(plan)
    extra = dict(KVMEM_ALLOCATOR_DIAGNOSTIC='1',
                 KVMEM_ALLOCATOR_DIAGNOSTIC_PHASE_FILE=str(root/'allocator_phase.json'))
    cap.require(not set(extra)&set(result['environment']), 'Diagnostic environment unexpectedly predeclared')
    result['environment'].update(extra)
    return result


def server_entry(root):
    """Preserve original serve cleaning while explicitly adding two variables."""
    cap.require(ROOT.resolve() == root/'frozen', 'Use exact frozen server wrapper')
    for name in list(os.environ):
        if name.upper().startswith(('KVMEM_','VLLM_')):
            del os.environ[name]
    # Explicit unchanged run_arm telemetry setting, not an inherited override.
    os.environ['VLLM_LOG_STATS_INTERVAL'] = '1'
    import vllm_kvmem
    from vllm_kvmem import serve
    package = root/'runtime'/ARM/'src/vllm_kvmem'
    cap.require(Path(vllm_kvmem.__file__).resolve() == package/'__init__.py'
        and Path(serve.__file__).resolve() == package/'serve.py', 'Server imported a foreign runtime')
    original = serve.build_launch
    def configured(*values,**kwargs):
        plan = original(*values,**kwargs)
        amended = add_diagnostic_environment(plan,root)
        cap.save(root/'server-plan-identity.json',dict(original=plan,diagnostic=amended,
            package=str(package),scope='Only two allocator variables added after original profile validation.'))
        return amended
    serve.build_launch = configured
    try:
        serve.main(['--config',str(root/(ARM+'.json')),'--run-dir',str(root/ARM/'server')])
    finally:
        serve.build_launch = original


def diagnostic_profile(profile, port):
    cap.require(type(port) is int and 1024 <= port <= 65535, 'Invalid dedicated diagnostic port')
    cap.require(profile['gpu_memory_utilization'] == .82 and profile['max_num_seqs'] == 12
        and profile['kvmem']['select_budget'] == 65536 and profile['kvmem']['gen_budget'] == 32768,
        'Require original twelve-user .82 K64/G32 configuration')
    result = copy.deepcopy(profile)
    result['port'] = port
    return result


def reference_gate(reference):
    frozen = module('allocator_reference_ready', reference/'frozen/scripts/benchmark_ready_agents.py')
    manifest = frozen.verify(reference)
    cap.require(manifest.get('experiment_kind') == 'fixed_r3_kvmem12_gpu_budget_trial_v1'
        and manifest['arms'] == [ARM] and manifest['workload'] == dict(concurrency=12,
            warmup_turns=2,epoch_turns=16,duration_seconds=300.,mode='tools'), 'Unexpected reference protocol')
    cap.require(cap.read(reference/'status.json')['status'] == 'completed_pending_review',
                'Original .82 run must be completed')
    result = cap.read(reference/ARM/'results.json')
    cap.require(result.get('status') == 'completed' and result.get('drained') is True
        and result.get('native_client_tokens_equal') is True, 'Reference completion evidence missing')
    qualification = Path(manifest['qualification'])
    cap.require(cap.sha(qualification/'manifest.json') == manifest['qualification_manifest_sha256']
        and cap.sha(qualification/ARM/'results.json') == manifest['qualification_result_sha256'],
        'Original code qualification identity changed')
    qualifier = module('allocator_base_qualification',qualification/'frozen/scripts/benchmark_ready_agents.py')
    qualified = qualifier.verify(qualification)
    cap.require(cap.read(qualification/'status.json')['status'] == 'completed_pending_review'
        and qualified['runtime_sha256'][ARM] == manifest['runtime_sha256'][ARM],
        'Original base runtime lacks completed exact qualification')
    capture = qualification/ARM/'server/capture'
    target = frozen.records(capture/'verified_restores.jsonl')
    recurrent = frozen.records(capture/'verified_state_restores.jsonl')
    resources = cap.read(capture/'scheduler_resources.json')
    cap.require(target and recurrent and all(r['byte_equal'] for r in target)
        and all(r['recurrent_byte_equal'] for r in recurrent)
        and resources['restore_admission']['peak_occupied']['resident'] > 1
        and not (capture/'restore_mismatch.json').exists(), 'Base byte qualification is absent or failed')
    return manifest


def overlay_contract(base, overlay, manifest):
    cap.require(set(manifest['files']) == FILES and manifest['kind'] == 'isolated_allocator_diagnostic_overlay_v1',
                'Only the six reviewed allocator diagnostic files may change')
    expected = cap.hashes(base)
    for name, row in manifest['files'].items():
        cap.require(expected.get(name) == row['before_sha256'], 'Allocator overlay base changed')
        cap.require(cap.sha(overlay/name) == row['after_sha256'], 'Allocator overlay changed')
        expected[name] = row['after_sha256']
    cap.require(set(cap.hashes(overlay)) == FILES, 'Unexpected extra diagnostic overlay files')
    return expected


def installed_identity(path):
    identity = cap.read(path)
    source = Path(identity['source_directory'])
    installed = Path(importlib.metadata.distribution('vllm').locate_file('vllm')).resolve()
    cap.require(source.resolve() == installed and identity.get('files'), 'Native installation source changed')
    hashes = {str(p):cap.sha(p) for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')}
    cap.require(hashes == identity['files'], 'Native installation bytes changed')


def device_identity():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.reserved',
        '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    cap.require(len(rows) == 1, 'Require one original device')
    uuid, total, reserved = [v.strip() for v in rows[0].split(',')]
    return dict(uuid=uuid,total_mib=int(total),driver_reserved_mib=int(reserved))


def workload_source(reference):
    work = module('allocator_workload_source', reference/'frozen/scripts/timed_agent_workload.py')
    return inspect.getsource(work.workload)


def prepare(args):
    root, reference, overlay_root = args.output.resolve(), args.reference.resolve(), args.overlay.resolve()
    cap.require(not root.exists() and reference not in root.parents and overlay_root not in root.parents,
                'Use a fresh separate diagnostic directory')
    original = reference_gate(reference)
    installed_identity(args.installed_native)
    device = device_identity()
    overlay = cap.read(overlay_root/'manifest.json')
    cap.require(Path(overlay['base']).resolve() == reference/'runtime'/ARM
        and overlay['base_run_manifest']['sha256'] == cap.sha(reference/'manifest.json'), 'Wrong diagnostic base')
    cap.require(cap.sha(overlay_root/'diagnostic.patch') == overlay['patch_sha256'], 'Diagnostic patch changed')
    expected = overlay_contract(reference/'runtime'/ARM, overlay_root/'overlay', overlay)
    transformed = transform_workload(workload_source(reference))
    profile = diagnostic_profile(cap.read(reference/(ARM+'.json')), args.port)
    root.mkdir(parents=True)
    ignore = shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache')
    shutil.copytree(reference/'frozen', root/'frozen', ignore=ignore)
    shutil.copyfile(Path(__file__), root/'frozen'/ENTRY)
    shutil.copytree(reference/'runtime'/ARM, root/'runtime'/ARM, ignore=ignore)
    shutil.copytree(overlay_root/'overlay', root/'diagnostic-overlay', ignore=ignore)
    for name in FILES:
        shutil.copyfile(root/'diagnostic-overlay'/name, root/'runtime'/ARM/name)
    cap.require(cap.hashes(root/'runtime'/ARM) == expected, 'Unrelated runtime changed')
    for source, target in ((reference/'fixture.json',root/'fixture.json'),
            (overlay_root/'manifest.json',root/'overlay-manifest.json'),
            (overlay_root/'diagnostic.patch',root/'diagnostic.patch'),
            (args.installed_native,root/'installed-native-identity.json')):
        shutil.copyfile(source,target)
    (root/'executed-workload.py').write_text(transformed)
    cap.save(root/(ARM+'.json'),profile)
    manifest = copy.deepcopy(original)
    manifest.update(experiment_kind=KIND,diagnostic_only=True,performance_claim_allowed=False,
        deployment_acceptance_allowed=False,reference=str(reference),
        reference_manifest_sha256=cap.sha(reference/'manifest.json'),
        reference_results_sha256=cap.sha(reference/ARM/'results.json'),
        source_sha256=cap.hashes(root/'frozen'),runtime_sha256={ARM:expected},
        profiles_sha256={ARM:cap.sha(root/(ARM+'.json'))},port=args.port,
        overlay_manifest_sha256=cap.sha(root/'overlay-manifest.json'),
        installed_native_sha256=cap.sha(root/'installed-native-identity.json'),
        installed_native_scope='Current complete .py/.so lock; no retrospective claim about earlier runs.',
        device_memory_identity=device,executed_workload_sha256=cap.sha(root/'executed-workload.py'),
        requested_diagnostic_workload=dict(concurrency=12,warmup_turns=2,formal_seconds=0,
            expected_warmup_requests=24,expected_cleanup_requests=24),
        scope='Known .82 headroom failure, repeated solely for allocator correlation. '
              'Unchanged original cold inputs, own outputs, warmup, history retirement and cleanup. '
              'Two source substitutions skip producers after the original barrier. No automatic retry, '
              'budget change, history clipping, formal throughput, or deployment acceptance.',created_at=time.time())
    cap.save(root/'manifest.json',manifest)
    cap.save(root/'status.json',dict(status='prepared',diagnostic_only=True,at=time.time(),gpu_started=False))


def verify(root):
    manifest = ready.verify(root)
    cap.require(manifest.get('experiment_kind') == KIND and manifest.get('diagnostic_only') is True
        and manifest.get('performance_claim_allowed') is False
        and manifest.get('deployment_acceptance_allowed') is False
        and manifest['arms'] == [ARM], 'Not the allocator-only diagnostic')
    reference = Path(manifest['reference'])
    cap.require(cap.sha(reference/'manifest.json') == manifest['reference_manifest_sha256']
        and cap.sha(reference/ARM/'results.json') == manifest['reference_results_sha256'], 'Reference changed')
    original = reference_gate(reference)
    expected_sources = dict(original['source_sha256'])
    expected_sources[ENTRY] = cap.sha(root/'frozen'/ENTRY)
    cap.require(manifest['source_sha256'] == expected_sources, 'Non-wrapper frozen source changed')
    for key in ('workload','budget','fixture_sha256','native_source','idle_seconds','qualify','qualification',
                'qualification_manifest_sha256','qualification_result_sha256'):
        cap.require(manifest[key] == original[key], 'Original '+key+' changed')
    cap.require(cap.read(root/(ARM+'.json')) == diagnostic_profile(cap.read(reference/(ARM+'.json')),manifest['port']),
                'Diagnostic profile changed beyond port')
    cap.require(manifest['requested_diagnostic_workload'] == dict(concurrency=12,warmup_turns=2,
        formal_seconds=0,expected_warmup_requests=24,expected_cleanup_requests=24),
        'Diagnostic workload declaration changed')
    cap.require(cap.sha(root/'overlay-manifest.json') == manifest['overlay_manifest_sha256'], 'Overlay manifest changed')
    overlay = cap.read(root/'overlay-manifest.json')
    cap.require(cap.sha(root/'diagnostic.patch') == overlay['patch_sha256'], 'Overlay patch changed')
    cap.require(manifest['runtime_sha256'] == {ARM:overlay_contract(reference/'runtime'/ARM,
                root/'diagnostic-overlay',overlay)}, 'Wrong diagnostic runtime')
    transformed = transform_workload(workload_source(reference))
    cap.require((root/'executed-workload.py').read_text() == transformed
        and cap.sha(root/'executed-workload.py') == manifest['executed_workload_sha256'], 'Warmup adapter changed')
    cap.require(cap.sha(root/'installed-native-identity.json') == manifest['installed_native_sha256'], 'Native lock changed')
    installed_identity(root/'installed-native-identity.json')
    cap.require(device_identity() == manifest['device_memory_identity'], 'Device memory identity changed')
    return manifest


def phase_writer(root):
    path = root/'allocator_phase.json'
    def write(phase):
        cap.require(phase in ('unknown','warmup','drain'), 'Unexpected allocator phase')
        row = dict(phase=phase,wall=time.time(),monotonic=time.monotonic())
        temporary = path.with_suffix('.tmp')
        cap.save(temporary,row)
        temporary.replace(path)
        with (root/'phase_events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    return write


def replay_proof(root):
    directory = root/ARM/'tools'
    rows = cap.read(directory/'traces.json')
    fixture = cap.read(root/'fixture.json')
    plans = ready.work.validate_agent_plans(fixture,concurrency=12,warmup_turns=2,
                                          epoch_turns=16,gen_budget=32768)
    cap.require(len(rows) == 48 and len({r['request_id'] for r in rows}) == 48,
                'Expected exactly24 warmup and24 cleanup requests')
    views = {}
    for i, plan in enumerate(plans):
        own = sorted((r for r in rows if r['agent'] == i),key=lambda r:r['begin'])
        cap.require([r['kind'] for r in own] == ['cleanup','warmup','warmup','cleanup'], 'Unexpected per-agent sequence')
        tokens = list(plan['initial_tokens'])
        for turn, row in enumerate(own[1:3]):
            if turn:
                tokens.extend(plan['turns'][turn]['tool_tokens'])
            count = plan['turns'][turn]['output_tokens']
            cap.require(row['turn'] == turn and row['epoch'] == 0 and row['status'] == 'completed'
                and row['input_tokens'] == len(tokens) and row['input_sha256'] == ready.work.digest(tokens)
                and len(row['output_ids']) == count == row['usage']['completion_tokens']
                and row['output_sha256'] == ready.work.digest(row['output_ids']),
                'Incomplete or substituted initial/tool/own-output history')
            tokens.extend(row['output_ids'])
        for row in (own[0],own[3]):
            cap.require(row['status'] == 'completed' and row['input_sha256'] == ready.work.digest([fixture['eos']])
                and row['input_tokens'] == 1 and len(row['output_ids']) == row['usage']['completion_tokens'] == 1,
                'Missing original EOS cleanup')
        cap.require(len({r['session'] for r in own}) == 1, 'Agent namespace changed')
        views[own[0]['session']] = len(tokens)-1
    cap.require(len(views) == 12, 'Independent agent namespaces collapsed')
    before = cap.read(directory/'resources-before.json')
    after = cap.read(directory/'resources-after.json')
    cap.require(ready.work.views_match(before,views), 'Warmup H=M view evidence missing')
    cap.require(ready.work.views_match(after,{session:1 for session in views}), 'Final cleanup views missing')
    cap.require(before['at'] >= max(r['begin'] for r in rows if r['kind'] == 'warmup')
        and after['at'] >= max(r['begin'] for r in rows), 'Retirement evidence is stale')
    return dict(warmup_requests=24,cleanup_requests=24,initial_inputs_complete=True,
        own_output_tool_chains_verified=12,warmup_terminal_history=views,
        final_eos_views=12,full_history_and_terminal_m=True)


def observations(root, manifest):
    directory = root/ARM
    allocator = ready.records(directory/'server/capture/allocator_diagnostic.jsonl')
    summaries = [r for r in allocator if r['event'] == 'summary']
    cap.require(summaries and any(r['event'] == 'allocator' and r['kind'] == 'prefill' for r in allocator),
                'Allocator prefill observations absent')
    summary = summaries[-1]
    prefill = [r for r in allocator if r['event'] == 'allocator' and r.get('kind') == 'prefill'
               and r.get('phase') == 'warmup']
    stages = dict(Counter(r['stage'] for r in prefill))
    required = {'target.begin','target.forward.return','action.begin','action.end',
                'capture.raw.begin','capture.raw.end','capture.values.end','capture.cat.end',
                'action.logits.allocated','action.softmax.allocated','action.mass.submitted'}
    gaps = []
    if missing := sorted(required-set(stages)):
        gaps.append(dict(reason='warmup_prefill_stages_missing',stages=missing))
    counts,coverage = summary.get('counts',{}),summary.get('coverage',{})
    lost = {key:counts.get(key,0) for key in ('record_limit_drops','buffer_limit_drops','unfinished_batches')
            if counts.get(key,0)}
    if lost:
        gaps.append(dict(reason='dropped_records_or_unfinished_batches',counts=lost))
    if coverage.get('prefill_batches_selected') != coverage.get('prefill_batch_headers_retained'):
        gaps.append(dict(reason='prefill_batch_headers_incomplete'))
    if coverage.get('prefill_batches_completed',0) != coverage.get('prefill_batches_selected',0):
        gaps.append(dict(reason='prefill_end_boundaries_incomplete'))
    persisted = sum(r['event'] != 'summary' for r in allocator)
    flushed = (allocator[-1]['event'] == 'summary' and summary.get('buffered') == 0
               and summary.get('retained') == summary.get('written') == persisted)
    if not flushed:
        gaps.append(dict(reason='last_reported_buffer_not_fully_persisted'))
    phases = ready.records(root/'phase_events.jsonl')
    barrier = next((r['monotonic'] for r in phases if r['phase'] == 'drain'),None)
    after_barrier = [r for r in allocator if r['event'] == 'allocator' and r.get('phase') == 'drain'
                     and barrier is not None and r.get('host_monotonic',-1) >= barrier]
    warmup_peaks = {name:max(values) for name in (
        'allocated_bytes.all.peak','reserved_bytes.all.peak','active_bytes.all.peak','inactive_split_bytes.all.peak')
        if (values := [r.get('memory',{}).get(name) for r in after_barrier
                       if isinstance(r.get('memory',{}).get(name),(int,float))])}
    # The reviewed logger has no final-close acknowledgement/timestamp. Even a
    # complete last summary cannot prove that no later unsaved sample existed.
    # Preserve that limit rather than pretending process-lifetime peaks closed.
    gaps.append(dict(reason='final_post_cleanup_flush_not_provable',
        detail='Exact reviewed logger records summary counters but no final-close acknowledgement; '
               'later unflushed samples cannot be ruled out. Use retained warmup-stage observations only.'))
    gpu = ready.records(directory/'gpu_samples.jsonl')
    cap.require(gpu,'NVML observations absent')
    peak = max(gpu,key=lambda r:r['memory_mib'])
    identity = manifest['device_memory_identity']
    free = identity['total_mib']-identity['driver_reserved_mib']-peak['memory_mib']
    return dict(diagnostic_only=True,performance_claim_allowed=False,deployment_acceptance_allowed=False,
        allocator_final_summary=summary,allocator_records=len(allocator),
        warmup_prefill_stage_counts=stages,coverage_gaps=gaps,
        warmup_scope=dict(post_retirement_barrier_allocator_samples=len(after_barrier),
            lifetime_peak_bytes_observed_after_barrier=warmup_peaks,
            lifetime_high_water_includes_complete_warmup=bool(warmup_peaks),
            scope='A sampled drain allocator high-water includes all earlier warmup work, '
                  'plus startup and any later allocations; it is not exclusive warmup or stage cost.'),
        last_summary_fully_persisted_as_reported=flushed,complete_process_high_water_proven=False,
        allocator_source_sha256=cap.sha(directory/'server/capture/allocator_diagnostic.jsonl'),
        nvml_peak_used_mib=peak['memory_mib'],nvml_peak_wall=peak['wall_at'],
        sampled_headroom_mib=free,headroom_below_four_gib=free<4096,
        nvml_source_sha256=cap.sha(directory/'gpu_samples.jsonl'),
        scope='Known .82 risk is reproduced without adjustment/retry. NVML1s may miss peaks. '
              'PyTorch allocator lifetime peaks are not exclusive stage contributions; host boundary '
              'correlation is not GPU execution timing. No throughput or deployment verdict.')


async def run(args):
    root = args.output.resolve()
    cap.require(ROOT.resolve() == root/'frozen', 'Use frozen diagnostic entry')
    manifest = verify(root)
    cap.require(cap.read(root/'status.json')['status'] == 'prepared', 'No automatic retry')
    write_phase = phase_writer(root)
    original_wait, original_workload = ready.bench.q.wait_idle, ready.work.workload
    original_popen = ready.subprocess.Popen
    replaced = [(ready,'execution_profile'),(ready.bench,'gpu_summary'),
                (ready.bench,'native_boundaries'),(ready.bench,'public_work_summary')]
    originals = [(obj,name,getattr(obj,name)) for obj,name in replaced]
    launches = []
    async def checked_wait(*values,**kwargs):
        await original_wait(*values,**kwargs)
        cap.require(verify(root) == manifest, 'Identity changed during idle wait')
    expected_command = [sys.executable,'-m','vllm_kvmem','serve','--config',str(root/(ARM+'.json')),
                        '--run-dir',str(root/ARM/'server')]
    def owned_popen(command,*values,**kwargs):
        if isinstance(command,(list,tuple)) and 'vllm_kvmem' in command:
            cap.require(list(command) == expected_command and not launches, 'Unexpected/duplicate model launch')
            cap.require(Path(kwargs['cwd']) == root/'runtime'/ARM, 'Model launch uses wrong runtime')
            original_command = list(command)
            command = [sys.executable,str(root/'frozen'/ENTRY),'--diagnostic-server','--output',str(root)]
            launches.append(dict(original_ready_argv=original_command,actual_popen_argv=command,
                diagnostic_environment=add_diagnostic_environment({'environment':{}},root)['environment'],
                scope='ready launch.json preserves requested original argv; this file records actual wrapper argv.'))
            cap.save(root/'diagnostic-launch.json',launches[-1])
        return original_popen(command,*values,**kwargs)
    namespace = dict(ready.work.__dict__,summarize=diagnostic_summary,allocator_phase=write_phase)
    exec(compile((root/'executed-workload.py').read_text(),str(root/'executed-workload.py'),'exec'),namespace)
    async def warmup_only(*values,**kwargs):
        write_phase('warmup')
        return await namespace['workload'](*values,**kwargs)
    ready.bench.q.wait_idle,ready.work.workload,ready.subprocess.Popen = checked_wait,warmup_only,owned_popen
    for obj,name in replaced:
        setattr(obj,name,no_window)
    loop,task = asyncio.get_running_loop(),asyncio.current_task()
    for sig in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(sig,task.cancel)
    started = time.time()
    try:
        with open('/tmp/kvmem-coordination-qualification.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            write_phase('unknown')
            result = await ready.run_arm(root,ARM,manifest)
        cap.require(len(launches) == 1 and result['native_client_tokens_equal'], 'Missing owned native/client proof')
        proof = replay_proof(root)
        cap.save(root/'replay-proof.json',proof)
        observed = observations(root,manifest)
        cap.save(root/'diagnostic.json',observed)
        cap.save(root/'status.json',dict(status='diagnostic_completed_with_gaps' if observed['coverage_gaps']
            else 'diagnostic_completed',at=time.time(),diagnostic_only=True))
    except BaseException as error:
        cap.save(root/'status.json',dict(status='interrupted' if isinstance(error,asyncio.CancelledError) else 'failed',
            at=time.time(),error=repr(error),diagnostic_only=True))
        raise
    finally:
        ready.bench.q.wait_idle,ready.work.workload,ready.subprocess.Popen = original_wait,original_workload,original_popen
        for obj,name,original in originals:
            setattr(obj,name,original)
        for sig in (signal.SIGINT,signal.SIGTERM):
            loop.remove_signal_handler(sig)
        cap.save(root/'timing.json',dict(started_wall=started,finished_wall=time.time(),diagnostic_only=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--run-prepared',action='store_true')
    mode.add_argument('--diagnostic-server',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--overlay',type=Path)
    parser.add_argument('--installed-native',type=Path)
    parser.add_argument('--port',type=int,default=18568)
    args = parser.parse_args()
    if args.diagnostic_server:
        return server_entry(args.output.resolve())
    if args.run_prepared:
        cap.require(args.reference is None and args.overlay is None and args.installed_native is None,
                    'Frozen execution only accepts output')
        return asyncio.run(run(args))
    cap.require(args.reference is not None and args.overlay is not None and args.installed_native is not None,
                'Explicit completed reference, reviewed overlay and installed native lock required')
    prepare(args)


if __name__ == '__main__':
    main()
