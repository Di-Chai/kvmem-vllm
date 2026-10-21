"""Bounded warm-request Nsight diagnostic; never a throughput benchmark.

Only the model server is injected. The controller, clients, nvidia-smi and
cleanup remain outside Nsight's environment. No production files are changed.
"""
import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shape_helper(root=None):
    # Runtime scripts precede this directory on sys.path during run/serve.
    # Load this diagnostic's own frozen helper without consulting import caches.
    from types import ModuleType
    path = (Path(__file__).parent if root is None else Path(root)) / 'nsight_prefill_shapes.py'
    module = ModuleType('_nsight_observed_shapes')
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


def profiler_config(delay, limit):
    if type(delay) is not int or not 0 <= delay <= 1024:
        raise ValueError('Delay must be an integer in 0..1024')
    if type(limit) is not int or not 1 <= limit <= 64:
        raise ValueError('Capture limit must be an integer in 1..64')
    return dict(profiler='cuda', delay_iterations=delay, max_iterations=limit,
                detailed_trace_annotation=True)


def pressure_config(begin, end, max_seconds, input_tokens):
    if (type(begin) is not int or type(end) is not int or
            not 1 <= begin < end < input_tokens or end-begin > 32768):
        raise ValueError('Pressure range must be inside the long input and at most 32K tokens')
    if not 1 <= max_seconds <= 15:
        raise ValueError('Pressure deadline must be in 1..15 seconds')
    return dict(begin_tokens=begin, end_tokens=end, max_seconds=max_seconds)


def nsys_command(executable, output, server):
    return [executable, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
        '--cpuctxsw=none', '--cuda-graph-trace=node', '--capture-range=cudaProfilerApi',
        '--capture-range-end=stop', '--kill=none', '--wait=primary',
        '--stop-on-exit=true', '--force-overwrite=false', '-o', str(output), *server]


def clean_environment(source):
    # Do not carry a finished Nsight injection into any child, including clients
    # and resource probes. An injected controller is an unsupported invocation.
    if (any(k.startswith(('NSYS_', 'NVTX_INJECTION', 'CUDA_INJECTION')) for k in source) or
            any(part in source.get('LD_PRELOAD', '').lower() for part in ('nsight', 'quadd', 'toolsinjection'))):
        raise ValueError('Run controller outside Nsight injection')
    result = {k:v for k,v in source.items()
              if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    result.update(PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                  MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1')
    result['PATH'] = str(Path(sys.executable).parent) + os.pathsep + result.get('PATH', os.defpath)
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit) / 'bin/nvcc').exists():
            result['CUDA_HOME'] = toolkit
            # vLLM checks shutil.which('nvcc') while importing its FlashInfer
            # wrappers, before any model execution. CUDA_HOME alone leaves
            # custom ops unregistered when the optional cubin package is absent.
            # Match the normal serving entry point before spawning the server.
            result['PATH'] = str(Path(toolkit) / 'bin') + os.pathsep + result['PATH']
            break
    return result


def prepare(args):
    source, case_path, output = args.source.resolve(), args.case.resolve(), args.root.resolve()
    runtime = source / 'runtime'
    proof = read(source / 'runtime-proof.json')
    for name, expected in proof['files'].items():
        if sha(runtime / name) != expected:
            raise ValueError('Frozen runtime changed: ' + name)
    case, profile = read(case_path), read(source / 'profile.json')
    kind = getattr(args, 'trace_kind', 'ordinary')
    pressure = kind == 'pressure'
    input_tokens = case['row']['served_input_tokens']
    if not pressure and input_tokens != 32768:
        raise ValueError('First diagnostic requires the full frozen 32768-token input')
    if pressure and (args.mode != 'kvmem' or not 256000 <= input_tokens <= 262144):
        raise ValueError('Pressure diagnostic requires KVMem and the complete frozen 256K input')
    if pressure and (profile['kvmem'].get('select_budget') != 32768 or
                     profile['kvmem'].get('gen_budget') != 16384 or
                     profile['kvmem'].get('guided_query_tokens', 0) or
                     profile['kvmem'].get('prefill_deferred_validation', False) or
                     profile['kvmem'].get('prefill_scatter_after_target', False)):
        raise ValueError('Pressure trace requires frozen K32K/G16K, guided/deferred/scatter disabled')
    if sha(case['input_path']) != case['row']['served_sha256']:
        raise ValueError('Frozen input changed')
    if profile['prefill_chunk_tokens'] != 2048 or profile['max_num_batched_tokens'] != 16424:
        raise ValueError('Expected the frozen 2K chunk / 16424 batch configuration')
    if not 1024 <= args.port <= 65535 or type(args.port) is not int:
        raise ValueError('Invalid local port')
    window = pressure_config(getattr(args, 'begin_tokens', 40960),
        getattr(args, 'end_tokens', 61440), getattr(args, 'pressure_max_seconds', 15), input_tokens) if pressure else None
    config = (dict(profiler='cuda', delay_iterations=0, max_iterations=0,
                   detailed_trace_annotation=True) if pressure else
              profiler_config(args.delay_iterations, args.max_iterations))
    observed = None
    if getattr(args, 'observed_chunks', False):
        if pressure:
            raise ValueError('Observed chunks are separate from the pressure diagnostic')
        observed = shape_helper().observed_shape_plan(source, case_path,
            delay=args.delay_iterations, limit=args.max_iterations)
    if not 1 <= args.startup_timeout <= 1800 or not 1 <= args.request_timeout <= 1800:
        raise ValueError('Diagnostic timeouts must be between 1 and 1800 seconds')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run').mkdir()
    (output / 'probe').mkdir()
    profile = deepcopy(profile)
    profile.update(host='127.0.0.1', port=args.port, profiler_config=None)
    save(output / 'profile.json', profile)
    definitions = []
    family = 'pressure256k' if pressure else 'full32k'
    chunks = (observed['chunks'] if observed is not None else
              [min(2048, input_tokens-i) for i in range(0, input_tokens, 2048)])
    for name in ('warmup', 'traced'):
        target = output / 'requests' / name
        target.mkdir(parents=True)
        copied = deepcopy(case)
        copied.update(id=name, request_id='nsight-' + name, family=family,
                      instrumented=True, warmup=name == 'warmup', chunks=chunks)
        save(target / 'case.json', copied)
        definitions.append(dict(id=name, request_id=copied['request_id'], family=family,
            input_tokens=input_tokens, instrumented=True, warmup=name == 'warmup', chunks=chunks))
    save(output / 'suite.json', dict(cases=definitions))
    helper = output / 'profile_prefill_nsight.py'
    shutil.copy2(__file__, helper)
    shutil.copy2(Path(__file__).with_name('analyze_prefill_nsight.py'), output / 'analyze_prefill_nsight.py')
    if observed is not None:
        save(output / 'observed-shapes.json', observed)
        shutil.copy2(Path(__file__).with_name('nsight_prefill_shapes.py'), output / 'nsight_prefill_shapes.py')
    phases = pressure or bool(getattr(args, 'phase_scopes', False))
    if phases:
        shutil.copy2(Path(__file__).with_name('prefill_phase_probe.py'), output / 'prefill_phase_probe.py')
    if pressure:
        shutil.copy2(Path(__file__).with_name('prefill_pressure_probe.py'), output / 'prefill_pressure_probe.py')
    save(output / 'plan.json', dict(source=str(source), runtime=str(runtime), mode=args.mode,
        port=args.port, profiler=config, input_sha256=case['row']['served_sha256'],
        case_sha256=sha(case_path), profile_sha256=sha(output / 'profile.json'),
        suite_sha256=sha(output / 'suite.json'),
        cases_sha256={name:sha(output / 'requests' / name / 'case.json') for name in ('warmup', 'traced')},
        runtime_proof_sha256=sha(source / 'runtime-proof.json'),
        helper_sha256=sha(helper), analyzer_sha256=sha(output / 'analyze_prefill_nsight.py'),
        observed_shapes_sha256=sha(output / 'observed-shapes.json') if observed is not None else None,
        shape_helper_sha256=sha(output / 'nsight_prefill_shapes.py') if observed is not None else None,
        phase_scopes=phases,
        trace_kind=kind, pressure_range=window,
        pressure_probe_sha256=sha(output / 'prefill_pressure_probe.py') if pressure else None,
        phase_probe_sha256=sha(output / 'prefill_phase_probe.py') if phases else None,
        nsys=str(Path(args.nsys).resolve()), startup_timeout=args.startup_timeout,
        request_timeout=args.request_timeout, diagnostic_only=True,
        note='Identical full input warmup then one profiled request; no accuracy/throughput claim. '
             'Capture steps are verified from NVTX, never assumed from delay counters.'))
    return output


def assert_frozen(root):
    plan = read(root / 'plan.json')
    for name, field in [('profile.json', 'profile_sha256'), ('suite.json', 'suite_sha256'),
                        ('profile_prefill_nsight.py', 'helper_sha256'),
                        ('analyze_prefill_nsight.py', 'analyzer_sha256')]:
        if sha(root / name) != plan[field]:
            raise ValueError('Diagnostic input changed: ' + name)
    if plan.get('phase_scopes') and sha(root / 'prefill_phase_probe.py') != plan['phase_probe_sha256']:
        raise ValueError('Diagnostic input changed: prefill_phase_probe.py')
    if plan.get('trace_kind') == 'pressure' and sha(root / 'prefill_pressure_probe.py') != plan['pressure_probe_sha256']:
        raise ValueError('Diagnostic input changed: prefill_pressure_probe.py')
    source = Path(plan['source'])
    if plan.get('observed_shapes_sha256') is not None:
        for name, field in [('observed-shapes.json', 'observed_shapes_sha256'),
                            ('nsight_prefill_shapes.py', 'shape_helper_sha256')]:
            if sha(root / name) != plan[field]:
                raise ValueError('Diagnostic input changed: ' + name)
        for name, expected in read(root / 'observed-shapes.json')['source_evidence_sha256'].items():
            if sha(source / name) != expected:
                raise ValueError('Observed source evidence changed: ' + name)
    if sha(source / 'runtime-proof.json') != plan['runtime_proof_sha256']:
        raise ValueError('Runtime proof changed')
    for name, expected in read(source / 'runtime-proof.json')['files'].items():
        if sha(Path(plan['runtime']) / name) != expected:
            raise ValueError('Frozen runtime changed: ' + name)
    for name in ('warmup', 'traced'):
        case_path = root / 'requests' / name / 'case.json'
        if sha(case_path) != plan['cases_sha256'][name]:
            raise ValueError('Frozen case changed: ' + name)
        case = read(case_path)
        if sha(case['input_path']) != plan['input_sha256']:
            raise ValueError('Input bytes changed')
    return plan


def phase_worker_argv(argv, mode):
    result = list(argv)
    worker = 'prefill_phase_probe.' + ('NativePhaseWorker' if mode == 'native' else 'KVMemPhaseWorker')
    if result.count('--worker-cls') > 1:
        raise ValueError('Multiple diagnostic worker classes')
    if '--worker-cls' in result:
        index = result.index('--worker-cls')
        if mode != 'native' or result[index+1] != 'native_prefill_shape_probe.NativeProbeWorker':
            raise ValueError('Cannot replace an unknown diagnostic worker')
        result[index+1] = worker
    else:
        result += ['--worker-cls', worker]
    return result


def serve(root):
    plan = assert_frozen(root)
    runtime = Path(plan['runtime'])
    sys.path[:0] = [str(runtime / 'src'), str(runtime / 'scripts')]
    from vllm_kvmem.serve import build_launch
    launch = build_launch(read(root / 'profile.json'), root, root / 'run/server')
    argv = list(launch['vllm_argv'])
    if '--profiler-config' in argv:
        raise ValueError('Frozen serving profile already has a profiler')
    if plan['mode'] == 'native':
        from benchmark_native_prefill import native_argv
        from native_agent_launch import _environment
        argv = native_argv(launch, plan['port'], prefill_chunk=2048)[4:]
        if plan.get('observed_shapes_sha256') is not None:
            # Import the helper frozen beside this controller, not a potentially
            # older copy in the measured production runtime's scripts directory.
            argv = shape_helper(root).observed_native_argv(argv, read(root / 'observed-shapes.json'))
        argv += ['--scheduler-cls', 'native_prefill_shape_probe.NativeShapeScheduler',
                 '--worker-cls', 'native_prefill_shape_probe.NativeProbeWorker']
        # Preserve Nsight injection inside the server tree. _environment only
        # clears KVMem/vLLM options; the external controller was never injected.
        env, _ = _environment(os.environ, allow_long=True)
        os.environ.clear()
        os.environ.update(env)
        os.environ.update(NATIVE_PREFILL_PROBE_PLAN=str(root / 'suite.json'),
                          NATIVE_PREFILL_PROBE_OUTPUT=str(root / 'probe'),
                          PYTHONPATH=str(runtime / 'scripts'))
        save(root / 'native-launch.json', dict(argv=argv, port=plan['port'], diagnostic_only=True))
    else:
        for name in list(os.environ):
            if name.startswith('KVMEM_'):
                del os.environ[name]
        os.environ.update(launch['environment'])
        from vllm_kvmem.compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    if plan.get('phase_scopes'):
        argv = phase_worker_argv(argv, plan['mode'])
        if plan.get('trace_kind') == 'pressure':
            argv[argv.index('--worker-cls') + 1] = 'prefill_pressure_probe.PressurePhaseWorker'
            os.environ['PREFILL_PRESSURE_RANGE'] = json.dumps(plan['pressure_range'])
        sys.path.insert(0, str(root))
        os.environ['PYTHONPATH'] = os.pathsep.join((str(root), os.environ.get('PYTHONPATH', '')))
        os.environ['PREFILL_PHASE_PROBE_OUTPUT'] = str(root / 'probe')
    argv += ['--profiler-config', json.dumps(plan['profiler'])]
    launch.update(vllm_argv=argv, diagnostic_profiler=plan['profiler'])
    (root / 'run/server/capture').mkdir(parents=True, exist_ok=False)
    save(root / 'run/server/launch.json', launch)
    # This file identifies the owned application group even if Nsight exits
    # immediately after capture-range-end=stop. Never kill other GPU processes.
    save(root / 'server-owned.json', dict(pid=os.getpid(), pgid=os.getpgrp()))
    from vllm.entrypoints.cli.main import main
    sys.argv = ['vllm', *argv]
    main()


def verify_shapes(root, report, mode, limit):
    response = read(root / 'requests/traced/response.json')
    path = (root / 'probe/model_step_latency.jsonl' if mode == 'native' else
            root / 'run/server/capture/model_step_latency.jsonl')
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    observed_path = root / 'observed-shapes.json'
    if observed_path.exists():
        observed = read(observed_path)
        shape_helper(root).verify_observed_trace(steps, response['response_id'], observed,
                                                report['profiled_worker_shapes'], limit)
        if mode == 'kvmem' and read(root / 'requests/traced/evidence.json')['refresh']:
            raise ValueError('Fitting observed 32K diagnostic unexpectedly refreshed its window')
        report.update(actual_geometry_verified=True, all_traced_steps_match_observed_prefill=True,
            observed_chunks=observed['chunks'], capture_begin_index=observed['capture_begin_index'])
        return
    recorded = []
    for row in steps:
        for shape in row.get('batch_geometry', []):
            rid, response_id = shape.get('request_id', ''), response['response_id']
            if (rid == response_id or rid.startswith(response_id + '-')) and shape.get('prefill'):
                recorded.append(shape)
    if not recorded:
        raise ValueError('Traced request has no actual recorded batch geometry')
    shapes = report['profiled_worker_shapes']
    if not report['all_traced_steps_single_owner_2k_prefill'] or len(shapes) != limit:
        raise ValueError('Capture did not contain exactly the requested ordinary 2K steps')
    for shape in shapes:
        matching = [r for r in recorded if r['query_tokens'] == shape['query_tokens'] and
                    r['logical_seq_len'] == shape['logical_seq_len']]
        if len(matching) != 1 or matching[0]['active_seq_len'] != shape['logical_seq_len']:
            raise ValueError('NVTX scheduled shape differs from actual active-window geometry')
        shape['active_seq_len'] = matching[0]['active_seq_len']
        shape['actual_geometry_verified'] = True
    if mode == 'kvmem' and read(root / 'requests/traced/evidence.json')['refresh']:
        raise ValueError('Fitting 32K diagnostic unexpectedly refreshed its window')
    report['actual_geometry_verified'] = True


def verify_pressure(root, report, plan):
    """Reject a cold, partial, wrong-owner, or non-pressure trace."""
    from collections import defaultdict
    import sqlite3
    from analyze_prefill_nsight import duration
    gate = read(root / 'probe/pressure-range.json')
    if gate['reason'] != 'logical_end' or not gate['public_steps']:
        raise ValueError('Pressure trace failed to reach its logical end')
    response = read(root / 'requests/traced/response.json')['response_id']
    matches = lambda rid: rid == response or rid.startswith(response + '-')
    if not matches(gate['request_id']):
        raise ValueError('Pressure gate captured the wrong request')
    warm = read(root / 'requests/warmup/evidence.json')
    if not any(row.get('selection') == 'pressure' for row in warm['refresh']):
        raise ValueError('Pressure trace has no complete long-input warmup')
    recorded = []
    for line in (root / 'run/server/capture/model_step_latency.jsonl').read_text().splitlines():
        recorded.extend(r for r in json.loads(line).get('batch_geometry', [])
                        if matches(r.get('request_id', '')) and r.get('prefill'))
    shapes = [s for s in report['profiled_worker_shapes'] if s['scheduled_tokens']]
    if len(shapes) != len(gate['public_steps']):
        raise ValueError('Pressure gate and NVTX public-step counts differ')
    for shape, scheduled in zip(shapes, gate['public_steps']):
        if (shape['context_requests'] != 1 or shape['generation_requests'] or
                shape['query_tokens'] != scheduled['tokens'] or
                shape['logical_seq_len'] != scheduled['computed'] + scheduled['tokens']):
            raise ValueError('Pressure NVTX shape differs from its scheduler gate')
        actual = [r for r in recorded if r['query_tokens'] == shape['query_tokens'] and
                  r['logical_seq_len'] == shape['logical_seq_len']]
        if len(actual) != 1:
            raise ValueError('Pressure trace has no unique actual active-window geometry')
        shape.update(active_seq_len=actual[0]['active_seq_len'], actual_geometry_verified=True)
    events = defaultdict(list)
    for row in gate['pressure_events']:
        if not matches(row['request_id']) or row['selection'] != 'pressure':
            raise ValueError('Pressure trace mixed another owner or Query selection')
        events[row['command_id']].append(row)
    stages = ['suspended', 'assembly_queued', 'assembly_started', 'selection_prepared', 'assembly_submitted', 'ready']
    if len(events) != 1 or [r['stage'] for r in next(iter(events.values()))] != stages:
        raise ValueError('Trace must contain exactly one complete pressure lifecycle')
    db = sqlite3.connect('file:' + str(root / 'trace.sqlite') + '?mode=ro', uri=True)
    try:
        strings = dict(db.execute('SELECT id,value FROM StringIds'))
        marks = [(s,e,t or strings.get(i,''),tid) for s,e,i,t,tid in
                 db.execute('SELECT start,end,textId,text,globalTid FROM NVTX_EVENTS')]
        command = next(iter(events))
        markers = {}
        for stage in stages:
            rows = [(s,e) for s,e,name,tid in marks if name == 'prefill_pressure::'+stage+'::'+str(command)]
            if len(rows) != 1 or rows[0][1] is None or rows[0][1] < rows[0][0]:
                raise ValueError('Missing pressure lifecycle marker in CUDA trace: ' + stage)
            markers[stage] = rows[0]
        begin, end = markers['assembly_started'][0], markers['ready'][1]
        # Count real CUDA activities inside the lifecycle, not CPU NVTX length.
        h2d = db.execute('SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM CUPTI_ACTIVITY_KIND_MEMCPY '
                        'WHERE copyKind=1 AND start>=? AND end<=?', (begin,end)).fetchone()
        rebuild = [(s,e) for s,e,n in db.execute('SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')
                   if begin <= s < e <= end and '_assemble_split' in strings.get(n,'')]
        if not h2d[0] or not rebuild:
            raise ValueError('Pressure trace lacks actual H2D or split-rebuild CUDA kernels')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        apis = []
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
            if table in tables:
                apis.extend(db.execute('SELECT start,end,globalTid,correlationId FROM ' + table))
        def correlated_count(table, predicate, label):
            scoped = {(tid // (1 << 24), corr) for a,b,tid,corr in apis
                      if any(s <= a <= b <= e and tid == scope_tid
                             for s,e,name,scope_tid in marks if name == label)}
            return sum((pid // (1 << 24), corr) in scoped for pid,corr in db.execute(
                'SELECT globalPid,correlationId FROM '+table+' WHERE start>=? AND end<=? AND '+predicate,
                (begin,end)))
        kernel_ids = [str(i) for i,name in strings.items() if '_assemble_split' in name]
        causal_rebuild = correlated_count('CUPTI_ACTIVITY_KIND_KERNEL',
            'demangledName IN ('+','.join(kernel_ids)+')', 'prefill_phase::rebuild.assemble_split_layer')
        causal_h2d = correlated_count('CUPTI_ACTIVITY_KIND_MEMCPY', 'copyKind=1 AND bytes>1048576',
                                     'prefill_phase::transfer.upload_from_reader')
        if not causal_rebuild or not causal_h2d:
            raise ValueError('Missing CUDA launch correlation from pressure rebuild/H2D CPU scopes')
        # These are overlapping occupancy measures, not additive stage costs.
        lifecycle_begin = markers['suspended'][0]
        def clipped(spans):
            return [(max(a,lifecycle_begin),min(b,end)) for a,b in spans
                    if a < end and b > lifecycle_begin]
        kernels = [(s,e,strings.get(n,'')) for s,e,n in db.execute(
            'SELECT start,end,demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL')]
        copies = list(db.execute('SELECT start,end,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY'))
        memset = (list(db.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET'))
                  if 'CUPTI_ACTIVITY_KIND_MEMSET' in tables else [])
        all_gpu = clipped([(s,e) for s,e,_ in kernels] + [(s,e) for s,e,_ in copies] + memset)
        gpu_union = duration(all_gpu)
        pressure_gpu = dict(lifecycle_ms=(end-lifecycle_begin)/1e6,
            activity_union_ms=gpu_union/1e6, idle_ms=(end-lifecycle_begin-gpu_union)/1e6,
            split_rebuild_union_ms=duration(clipped([(s,e) for s,e,n in kernels if '_assemble_split' in n]))/1e6,
            resident_v_gather_union_ms=duration(clipped([(s,e) for s,e,n in kernels if '_gather_v' in n]))/1e6,
            h2d_union_ms=duration(clipped([(s,e) for s,e,kind in copies if kind == 1]))/1e6,
            nonadditive=True)
        scope_labels = {name for _,_,name,_ in marks if name.startswith((
            'prefill_phase::refresh.', 'prefill_phase::assembly.', 'prefill_phase::staging.',
            'prefill_phase::transfer.', 'prefill_phase::archive.', 'prefill_phase::selection.',
            'prefill_phase::publication.', 'prefill_phase::authority.publish_sealed'))}
        host_scopes = {}
        for label in sorted(scope_labels):
            spans = clipped([(s,e) for s,e,name,_ in marks if name == label and e is not None])
            if spans:
                host_scopes[label] = dict(count=len(spans), union_ms=duration(spans)/1e6,
                    summed_ms=sum(e-s for s,e in spans)/1e6,
                    gpu_overlap_ms=duration([(max(a,s),min(b,e)) for a,b in spans
                        for s,e in all_gpu if a < e and b > s])/1e6)
    finally:
        db.close()
    report.update(trace_kind='pressure', actual_geometry_verified=True,
        pressure_verified=dict(command_id=command, stages=stages, marker_times_ns=markers,
            h2d_operations=h2d[0], h2d_bytes=h2d[1], split_rebuild_kernels=len(rebuild),
            correlated_split_rebuild_kernels=causal_rebuild, correlated_bulk_h2d=causal_h2d,
            gpu=pressure_gpu, host_scopes_nonadditive=host_scopes,
            warmup='Same complete 256K input finished before start_profile',
            scope='CPU stages are nested/nonadditive; CUDA correlation is required for attribution'),
        diagnostic_only=True)


def post(port, endpoint, timeout=60):
    request = urllib.request.Request(f'http://127.0.0.1:{port}/{endpoint}', data=b'', method='POST')
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f'{endpoint}: HTTP {response.status}')


def run(root):
    plan = assert_frozen(root)
    runtime = Path(plan['runtime'])
    sys.path.insert(0, str(runtime / 'scripts'))
    import benchmark_block_graph as lifecycle
    from benchmark_native_prefill import http, idle, metrics
    env = clean_environment(os.environ)
    env['PYTHONPATH'] = os.pathsep.join((str(runtime / 'src'), str(runtime / 'scripts')))
    server_command = [sys.executable, '-B', str(root / 'profile_prefill_nsight.py'), 'serve', '--root', str(root)]
    command = nsys_command(plan['nsys'], root / 'trace', server_command)
    save(root / 'nsys-command.json', dict(argv=command, controller_injected=False))
    server = worker = None
    started_profile = False
    with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if lifecycle.gpu_processes():
                raise RuntimeError('Foreign GPU owner; no service is stopped by this diagnostic')
            start = time.monotonic()
            with (root / 'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(root / 'nsys-process.json', dict(pid=server.pid))
            deadline = start + plan['startup_timeout']
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Nsight/server exited during startup; inspect server.log')
                try:
                    with http().open(f"http://127.0.0.1:{plan['port']}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError('Startup deadline exceeded')
            save(root / 'cold-preparation.json', dict(seconds=time.monotonic()-start,
                included_in_trace=False, diagnostic_only=True))
            (root / 'run/server/capture/armed').touch()
            for name in ('warmup', 'traced'):
                if name == 'traced':
                    post(plan['port'], 'start_profile')
                    started_profile = True
                script = 'benchmark_matched_prefill.py' if plan['mode'] == 'kvmem' else 'benchmark_active_prefill.py'
                subcommand = 'client' if plan['mode'] == 'kvmem' else 'native-client'
                with (root / f'{name}-client.log').open('x') as log:
                    worker = subprocess.Popen([sys.executable, '-B', str(runtime / 'scripts' / script),
                        subcommand, '--root', str(root / 'requests' / name)], cwd=runtime, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                code = worker.wait(timeout=plan['request_timeout'])
                if code:
                    raise RuntimeError(f'{name} client failed ({code}); inspect client log')
                deadline = time.monotonic() + 60
                while not idle(metrics(plan['port'])):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Requests did not retire')
                    time.sleep(.1)
                if name == 'traced':
                    post(plan['port'], 'stop_profile')
                    started_profile = False
                else:
                    save(root / 'warm-preparation.json', dict(completed=True, included_in_trace=False,
                        input_sha256=plan['input_sha256'], diagnostic_only=True,
                        kind='Full long input, including pressure/Query' if plan.get('trace_kind') == 'pressure' else 'Full 32K input'))
            save(root / 'status.json', dict(status='requests_completed', diagnostic_only=True))
        finally:
            begin_cleanup()
            if started_profile:
                try:
                    post(plan['port'], 'stop_profile', timeout=15)
                except (OSError, RuntimeError):
                    pass
            # Capture-end=stop leaves the application alive. Gracefully stop
            # its recorded PID before stopping the tracer's process group so
            # Nsight has an opportunity to finish exporting the report.
            owned = root / 'server-owned.json'
            if owned.exists():
                import signal
                try:
                    os.kill(read(owned)['pid'], signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if server is not None:
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    pass
            application = None
            if owned.exists() and server is not None and read(owned)['pgid'] != server.pid:
                from types import SimpleNamespace
                application = SimpleNamespace(pid=read(owned)['pgid'], poll=lambda: None)
            remaining = lifecycle.stop_owned_groups((worker, server, application))
            save(root / 'cleanup.json', dict(remaining_owned_groups=remaining,
                gpu_processes=sorted(lifecycle.gpu_processes()), diagnostic_only=True))
            if remaining:
                raise RuntimeError('Owned processes survived cleanup')
    report = root / 'trace.nsys-rep'
    if not report.is_file():
        raise RuntimeError('Nsight report missing; do not infer zero GPU work')
    subprocess.run([plan['nsys'], 'export', '--type=sqlite', '--force-overwrite=false',
        '--output=' + str(root / 'trace.sqlite'), str(report)], check=True, timeout=180, env=env)
    subprocess.run([sys.executable, '-B', str(root / 'analyze_prefill_nsight.py'),
        str(root / 'trace.sqlite'), '--output', str(root / 'timeline-summary.json')], check=True, env=env)
    report = read(root / 'timeline-summary.json')
    if plan.get('trace_kind') == 'pressure':
        verify_pressure(root, report, plan)
    else:
        verify_shapes(root, report, plan['mode'], plan['profiler']['max_iterations'])
    save(root / 'timeline-summary.json', report)
    save(root / 'status.json', dict(status='completed', diagnostic_only=True,
                                   actual_geometry_verified=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'serve'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--case', type=Path)
    parser.add_argument('--mode', choices=('native', 'kvmem'), default='kvmem')
    parser.add_argument('--port', type=int, default=18866)
    parser.add_argument('--delay-iterations', type=int, default=8)
    parser.add_argument('--max-iterations', type=int, default=4)
    parser.add_argument('--startup-timeout', type=int, default=1200)
    parser.add_argument('--request-timeout', type=int, default=300)
    parser.add_argument('--nsys', default='/usr/local/bin/nsys')
    parser.add_argument('--phase-scopes', action='store_true',
                        help='Add diagnostic-only bounded host NVTX phases; excluded from throughput claims')
    parser.add_argument('--observed-chunks', action='store_true',
                        help='Freeze the complete block table of this completed formal source 32K request')
    parser.add_argument('--trace-kind', choices=('ordinary', 'pressure'), default='ordinary')
    parser.add_argument('--begin-tokens', type=int, default=40960)
    parser.add_argument('--end-tokens', type=int, default=61440)
    parser.add_argument('--pressure-max-seconds', type=float, default=15,
                        help='Worker-boundary pressure deadline; stalled workers use request timeout')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.source is None or args.case is None:
            parser.error('prepare requires --source and --case')
        print(prepare(args))
    else:
        (run if args.command == 'run' else serve)(args.root.resolve())


if __name__ == '__main__':
    main()
