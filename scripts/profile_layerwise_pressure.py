"""Finite, frozen single-owner pressure trace with complete-input equivalence.

Reuse the existing Nsight lifecycle, logical capture gate and phase analyzers.
The profiler is diagnostic only: its wall time is never a performance result.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from profile_prefill_nsight import (read, save, sha, clean_environment,
    nsys_command, pressure_config, profiler_config, post, verify_pressure)


HELPERS = ('profile_layerwise_pressure.py', 'profile_prefill_nsight.py',
    'prefill_phase_probe.py', 'prefill_pressure_probe.py',
    'layerwise_pressure_phase_probe.py', 'analyze_prefill_nsight.py',
    'analyze_prefill_phases.py', 'compare_prefill_nsight.py', 'nsight_prefill_shapes.py',
    'run_layerwise_pressure_efficiency.py')


def controls(source, destination):
    a, b = deepcopy(source), deepcopy(destination)
    for key in ('port', 'served_model_name'):
        a.pop(key, None)
        b.pop(key, None)
    if a != b or b['max_num_seqs'] != 1 or b['execution_mode'] != 'eager':
        raise ValueError('Diagnostic requires the unchanged serial eager control')
    kv = b['kvmem']
    if [kv[k] for k in ('select_budget', 'gen_budget', 'block_tokens',
                       'sink_blocks', 'recent_blocks')] != [36864, 16384, 32, 41, 0]:
        raise ValueError('Frozen KV controls changed')


def exact_comparison(before, after, old_response, response, profile):
    from run_layerwise_pressure_efficiency import signature, full_prefix_candidate_proof
    old, new = signature(before), signature(after)
    return dict(same_pressure=old[0] == new[0], same_query=old[1] == new[1],
        same_output_tokens=old_response['output_ids'] == response['output_ids'],
        same_text=old_response['text'] == response['text'],
        query_scopes=[full_prefix_candidate_proof(e, profile) for e in (before, after)],
        pressure_events=len(new[0]), query_layers=len(after['layers']))


def prepare(args):
    from run_layerwise_query_diagnostic import verify
    source, root = args.source.resolve(), args.root.resolve()
    verify(source)
    fixture, profile = read(source/'fixture.json'), read(source/'profile.json')
    if not fixture['layerwise_prefill_pressure'] or fixture['layerwise_workers'] != 1:
        raise ValueError('Use the qualified serial per-layer pressure source')
    row = fixture['rows'][0]
    case = source/'formal/sequential/cases'/row['id']
    result = read(case/'result.json')
    for name in ('response', 'evidence'):
        if sha(case/(name+'.json')) != result[name+'_sha256']:
            raise ValueError('Source first-answer evidence changed')
    first_pressure = next(e['accepted_history'] for e in read(case/'evidence.json')['events']
                          if e['event'] == 'pressure_selected')
    window = pressure_config(first_pressure-1024, first_pressure+2048, 15,
                             row['served_input_tokens'])
    profiler = profiler_config(0,64)
    # The built-in counter includes idle maintenance iterations. Pressure
    # rebuilds may exceed 64 such polls without making 64 model forwards.
    # The logical gate already enforces <=64 public steps and a 15s deadline.
    profiler['max_iterations'] = 0
    root.mkdir(parents=True, exist_ok=False)
    (root/'probe').mkdir()
    shutil.copytree(source/'runtime-v1', root/'runtime-v1')
    for name in HELPERS:
        shutil.copy2(Path(__file__).parent/name, root/name)
    shutil.copy2(source/'runtime-proof.json', root/'runtime-proof.json')
    shutil.copy2(source/'profile.json', root/'source-profile.json')
    shutil.copy2(source/'fixture.json', root/'fixture.json')
    shutil.copy2(source/'inputs'/(row['id']+'.i32'), root/'input.i32')
    (root/'reference').mkdir()
    for name in ('request', 'response', 'evidence', 'result'):
        shutil.copy2(case/(name+'.json'), root/'reference'/(name+'.json'))
    profile.update(port=args.port, served_model_name='layerwise-pressure-profile')
    controls(read(root/'source-profile.json'), profile)
    save(root/'profile.json', profile)
    from run_concurrent_layerwise_ruler import server_command
    command = server_command(root, root/'run', fixture) + ['--dry-run']
    env = clean_environment(os.environ)
    env.update(PYTHONPATH=str(root/'runtime-v1/src'), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
    launch = subprocess.run(command, cwd=root/'runtime-v1', env=env,
        capture_output=True, text=True, check=True)
    save(root/'frozen-launch.json', json.loads(launch.stdout))
    immutable = (*HELPERS, 'runtime-proof.json', 'source-profile.json', 'fixture.json',
        'profile.json', 'input.i32', 'frozen-launch.json',
        *(f'reference/{name}.json' for name in ('request', 'response', 'evidence', 'result')))
    plan = dict(source=str(source), runtime=str(root/'runtime-v1'),
        sample_id=row['id'], input_sha256=row['served_sha256'],
        pressure_range=window, profiler=profiler, port=args.port,
        nsys=str(Path(args.nsys).resolve()), startup_timeout=900, request_timeout=900,
        expected_active_root=str(args.expected_active.resolve()),
        active_pointer=str(source.parent.parent/'value-recall-active.json'),
        immutable={name:sha(root/name) for name in immutable},
        diagnostic_only=True, independent_full_input_runs=2,
        capture_limits=dict(public_model_steps=64,scheduling_boundary_seconds=15,
                            controller_request_timeout=900,maintenance_polls_not_counted=True),
        scope='Warmup and traced full input must both match source pressure/Query choices and complete output. No throughput claim.')
    save(root/'plan.json', plan)
    save(root/'prepared.json', dict(plan_sha256=sha(root/'plan.json'), at=time.time()))
    save(root/'status.json', dict(status='prepared', diagnostic_only=True, at=time.time()))
    assert_frozen(root)
    # Exercise imports from the frozen package before loading a model. The
    # accepted engine snapshot need not contain newer analysis-only drivers.
    env['PYTHONPATH'] = os.pathsep.join((str(root),str(root/'runtime-v1/src'),
                                      str(root/'runtime-v1/scripts')))
    subprocess.run([sys.executable,'-B',str(root/'profile_layerwise_pressure.py'),
                    'preflight','--root',str(root)],cwd=root,env=env,check=True)


def assert_frozen(root):
    if sha(root/'plan.json') != read(root/'prepared.json')['plan_sha256']:
        raise ValueError('Frozen trace plan changed')
    plan = read(root/'plan.json')
    for name, digest in plan['immutable'].items():
        if sha(root/name) != digest:
            raise ValueError('Frozen diagnostic file changed: '+name)
    for name, digest in read(root/'runtime-proof.json').items():
        if sha(root/'runtime-v1'/name) != digest:
            raise ValueError('Frozen model runtime changed: '+name)
    controls(read(root/'source-profile.json'), read(root/'profile.json'))
    if sha(root/'input.i32') != plan['input_sha256']:
        raise ValueError('Original complete input changed')
    return plan


def handoff_ready(previous, *, is_alive=None):
    if is_alive is None:
        from run_layerwise_pressure_efficiency import alive
        is_alive = alive
    status = read(previous/'status.json')['status']
    if status not in ('completed','failed'):
        raise ValueError('Previous experiment is not terminal')
    process = previous/'process.json'
    if process.exists() and is_alive(read(process)['pid']):
        raise ValueError('Previous controller is still live')
    if status == 'failed':
        # Only a known failed diagnostic with complete cleanup can hand off.
        plan, cleanup = read(previous/'plan.json'), read(previous/'cleanup.json')
        if (not plan.get('diagnostic_only') or cleanup['remaining_owned_groups'] or
                cleanup['gpu_processes']):
            raise ValueError('Failed diagnostic has no clean ownership handoff')


def preflight(root):
    assert_frozen(root)
    from run_concurrent_layerwise_ruler import audit_events
    from analyze_prefill_nsight import analyze
    from analyze_prefill_phases import analyze as phases
    from layerwise_pressure_phase_probe import install_layerwise_scopes
    before, response = read(root/'reference/evidence.json'), read(root/'reference/response.json')
    proof = exact_comparison(before,before,response,response,read(root/'profile.json'))
    if not all(proof[k] for k in ('same_pressure','same_query','same_output_tokens','same_text')):
        raise ValueError('Frozen analysis module failed reference self-check')
    save(root/'preflight.json',dict(passed=True,frozen_analysis_imports_verified=True,
                                   reference_scope_verified=True,at=time.time()))


def serve(root):
    plan = assert_frozen(root)
    launch = read(root/'frozen-launch.json')
    argv = list(launch['vllm_argv'])
    if '--worker-cls' in argv or '--profiler-config' in argv:
        raise ValueError('Unexpected existing diagnostic override')
    argv += ['--worker-cls', 'layerwise_pressure_phase_probe.LayerwisePressurePhaseWorker',
             '--profiler-config', json.dumps(plan['profiler'])]
    for name in list(os.environ):
        if name.startswith('KVMEM_'):
            del os.environ[name]
    os.environ.update(launch['environment'])
    os.environ.update(PREFILL_PRESSURE_RANGE=json.dumps(plan['pressure_range']),
        PREFILL_PHASE_PROBE_OUTPUT=str(root/'probe'))
    if launch['profile']['model_adapter'] == 'qwen38_flash_next_b12x':
        from vllm_kvmem.qsa_config_compat import enable
        enable()
    if launch['profile']['draft_method'] == 'dflash':
        from vllm_kvmem.compat import enable_draft_load_config_fix
        enable_draft_load_config_fix()
    (root/'run/server/capture').mkdir(parents=True, exist_ok=False)
    save(root/'run/server/launch.json', dict(launch, vllm_argv=argv, diagnostic_only=True))
    save(root/'server-owned.json', dict(pid=os.getpid(), pgid=os.getpgrp()))
    from vllm.entrypoints.cli.main import main
    sys.argv = ['vllm', *argv]
    main()


def client(root, name):
    assert_frozen(root)
    import numpy as np
    import qualify_graph_niah as q
    from benchmark_client_io import CaptureSlice
    from recall_comparison_protocol import case_sampling
    from run_concurrent_layerwise_ruler import audit_events
    fixture, profile = read(root/'fixture.json'), read(root/'profile.json')
    row = fixture['rows'][0]
    out = root/'requests'/name
    out.mkdir(parents=True, exist_ok=False)
    capture = root/'run/server/capture'
    tokens = np.fromfile(root/'input.i32', dtype='<i4').tolist()
    sampling = case_sampling(fixture, row)
    reference = read(root/'reference/request.json')
    if (sampling != reference['sampling'] or row['served_query'] != reference['query_span'] or
            row['served_sha256'] != reference['input_sha256'] or len(tokens) != row['served_input_tokens']):
        raise ValueError('Source request controls changed')
    session = 'layerwise-profile-'+uuid.uuid4().hex
    save(out/'request.json', dict(session=session, sampling=sampling,
        query_span=row['served_query'], input_sha256=row['served_sha256']))
    cursor = CaptureSlice(capture/file for file in ('layerwise_query.jsonl','bounded_refresh.jsonl'))
    response = q.stream(profile, session, tokens, row['served_query'], sampling, 900)
    save(out/'response.json', response)  # Preserve the first answer before audits.
    terminal = q.wait_retired(capture,response['response_id'])
    if terminal['history_end'] != len(tokens)+len(response['output_ids'])-1:
        raise ValueError('Input/output accepted token accounting changed')
    released = q.wait_owner_released(capture, session, response, terminal)
    def rows(file):
        return [e for e in cursor.read(capture/file)
                if e.get('request_id','').startswith(response['response_id']+'-')]
    events, bounds = rows('layerwise_query.jsonl'), rows('bounded_refresh.jsonl')
    layers = audit_events(events, bounds, response, row, independent=True)
    evidence = dict(events=events, refresh=bounds, layers=layers, terminal=terminal, release=released)
    save(out/'evidence.json', evidence)
    proof = exact_comparison(read(root/'reference/evidence.json'), evidence,
        read(root/'reference/response.json'), response, profile)
    proof['passed'] = all(proof[k] for k in ('same_pressure','same_query','same_output_tokens','same_text'))
    save(out/'equivalence.json', proof)
    if not proof['passed']:
        raise ValueError('Diagnostic altered complete-input choices or answer; evidence preserved')


def run(root):
    plan = assert_frozen(root)
    import benchmark_block_graph as lifecycle
    from benchmark_native_prefill import http, idle, metrics
    from run_layerwise_memory import normalize_cuda_alias
    env = normalize_cuda_alias(clean_environment(os.environ))
    env.update(PYTHONPATH=os.pathsep.join((str(root), str(root/'runtime-v1/src'),
        str(root/'runtime-v1/scripts'))), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    script = str(root/'profile_layerwise_pressure.py')
    command = nsys_command(plan['nsys'], root/'trace',
        [sys.executable, '-B', script, 'serve', '--root', str(root)])
    server = worker = None
    started_profile = False
    with open('/tmp/kvmem-coordination-qualification.lock','a+') as lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(plan['active_pointer'])['queue_root'] != plan['expected_active_root']:
            raise ValueError('GPU experiment handoff changed; inspect before running')
        handoff_ready(Path(plan['expected_active_root']))
        if lifecycle.gpu_processes():
            raise RuntimeError('GPU has another owner; no process was stopped')
        if (root/'process.json').exists() or (root/'requests').exists():
            raise ValueError('Preserve first attempts; create a new diagnostic version')
        save(root/'process.json', dict(pid=os.getpid(), at=time.time()))
        save(plan['active_pointer'], dict(queue_root=str(root), root=str(root),
            status=str(root/'status.json'), process=str(root/'process.json'),
            experiment_type='bounded-layerwise-pressure-profile', at=time.time()))
        save(root/'nsys-command.json', dict(argv=command, controller_injected=False))
        try:
            save(root/'status.json', dict(status='loading', diagnostic_only=True, at=time.time()))
            with (root/'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=root/'runtime-v1', env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(root/'nsys-process.json', dict(pid=server.pid))
            deadline = time.monotonic()+plan['startup_timeout']
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Nsight/server exited during startup')
                try:
                    with http().open(f"http://127.0.0.1:{plan['port']}/health",timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError('Server startup timed out')
            (root/'run/server/capture/armed').touch()
            for name in ('warmup','traced'):
                save(root/'status.json', dict(status='running', phase=name, diagnostic_only=True, at=time.time()))
                if name == 'traced':
                    post(plan['port'],'start_profile')
                    started_profile = True
                with (root/(name+'-client.log')).open('x') as log:
                    worker = subprocess.Popen([sys.executable,'-B',script,'client','--root',str(root),
                        '--request',name],cwd=root/'runtime-v1',env=env,stdout=log,
                        stderr=subprocess.STDOUT,start_new_session=True)
                if worker.wait(timeout=plan['request_timeout']):
                    raise RuntimeError(name+' failed; first response preserved if received')
                deadline = time.monotonic()+60
                while not idle(metrics(plan['port'])):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Request did not retire')
                    time.sleep(.1)
                if name == 'traced':
                    post(plan['port'],'stop_profile')
                    started_profile = False
            assert_frozen(root)
        except BaseException as error:
            save(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            if started_profile:
                try:
                    post(plan['port'],'stop_profile',timeout=15)
                except (OSError,RuntimeError):
                    pass
            owned = root/'server-owned.json'
            if owned.exists():
                try:
                    os.kill(read(owned)['pid'],signal.SIGTERM)
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
                application = SimpleNamespace(pid=read(owned)['pgid'],poll=lambda:None)
            remaining = lifecycle.stop_owned_groups((worker,server,application))
            save(root/'cleanup.json',dict(remaining_owned_groups=remaining,
                gpu_processes=sorted(lifecycle.gpu_processes()),at=time.time()))
            if remaining:
                raise RuntimeError('Owned processes survived cleanup')
    try:
        analyze(root, env)
    except BaseException as error:
        save(root/'status.json',dict(status='failed',phase='analysis',error=repr(error),at=time.time()))
        raise


def analyze(root, env=None):
    plan = assert_frozen(root)
    if env is None:
        env = clean_environment(os.environ)
    db = root/'trace.sqlite'
    if not db.exists():
        subprocess.run([plan['nsys'],'export','--type=sqlite','--force-overwrite=false',
            '--output='+str(db),str(root/'trace.nsys-rep')],check=True,timeout=180,env=env)
    from analyze_prefill_nsight import analyze as timeline
    report = timeline(db)
    verify_pressure(root,report,plan)
    save(root/'timeline-summary.json',report)
    from analyze_prefill_phases import analyze as phases
    save(root/'phase-summary.json',phases(root))
    save(root/'status.json',dict(status='completed',actual_geometry_verified=True,
        complete_input_equivalence_verified=True,diagnostic_only=True,at=time.time()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','preflight','run','serve','client','analyze'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--expected-active',type=Path)
    parser.add_argument('--port',type=int,default=19104)
    parser.add_argument('--nsys',default='/usr/local/bin/nsys')
    parser.add_argument('--request',choices=['warmup','traced'])
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(args)
        return
    sys.path[:0] = [str(root), str(root/'runtime-v1/src'), str(root/'runtime-v1/scripts')]
    if args.action == 'client':
        client(root,args.request)
    else:
        globals()[args.action](root)


if __name__ == '__main__':
    main()
