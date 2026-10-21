"""Frozen prompts on unmodified vLLM, with explicit owned-service lifecycle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def http():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


KNOWN_DIAGNOSTIC_WORKERS = frozenset((
    'prefill_inductor_worker.SeededAuditWorker',
    'prefill_inductor_worker.SeededNativeProbeWorker',
    'native_prefill_shape_probe.NativeProbeWorker',
))


def _optional_argument(argv, name):
    """Return one CLI argument's span/value, rejecting ambiguous launch files."""
    indices = [i for i, value in enumerate(argv) if value == name or value.startswith(name + '=')]
    if len(indices) > 1:
        raise ValueError('Duplicate native launch argument: ' + name)
    if not indices:
        return None
    index = indices[0]
    if argv[index] == name:
        if index + 1 == len(argv) or argv[index + 1].startswith('--'):
            raise ValueError('Missing native launch argument value: ' + name)
        return index, 2, argv[index + 1]
    value = argv[index].split('=', 1)[1]
    if not value:
        raise ValueError('Missing native launch argument value: ' + name)
    return index, 1, value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate additional-config key: ' + key)
        result[key] = value
    return result


def _remove_diagnostic_launch_options(argv):
    worker = _optional_argument(argv, '--worker-cls')
    if worker is not None:
        index, width, name = worker
        if name not in KNOWN_DIAGNOSTIC_WORKERS:
            raise ValueError('Unknown native reference worker: ' + name)
        del argv[index:index + width]
    additional = _optional_argument(argv, '--additional-config')
    if additional is not None:
        index, width, raw = additional
        config = json.loads(raw, object_pairs_hook=_unique_object)
        if not isinstance(config, dict):
            raise ValueError('Native additional-config must be an object')
        if 'kvmem_capture_opaque_source' in config:
            del config['kvmem_capture_opaque_source']
            argv[index:index + width] = (['--additional-config', json.dumps(config)] if config else [])


def native_argv(saved, port, *, batch_tokens=None, prefill_chunk=None):
    argv = list(saved['vllm_argv'])
    for name in ('--scheduler-cls', '--kv-transfer-config', '--middleware'):
        if argv.count(name) != 1:
            raise ValueError('Missing or duplicate KVMem extension: ' + name)
        i = argv.index(name)
        del argv[i:i + 2]
    _remove_diagnostic_launch_options(argv)
    argv[argv.index('--port') + 1] = str(port)
    if any('vllm_kvmem' in s or 'KVMem' in s for s in argv):
        raise ValueError('KVMem launch extension survived')
    if ('--no-enable-prefix-caching' not in argv or '--no-async-scheduling' not in argv
            or argv[argv.index('--long-prefill-token-threshold') + 1] != '0'):
        raise ValueError('Native baseline cache/scheduling/chunk policy differs')
    if batch_tokens is not None:
        if type(batch_tokens) is not int or batch_tokens != 16384:
            raise ValueError('Only the explicit aligned 16384-token diagnostic is supported')
        index = argv.index('--max-num-batched-tokens') + 1
        if argv[index] != '16424':
            raise ValueError('Expected the frozen 16424-token reference batch budget')
        argv[index] = str(batch_tokens)
    if prefill_chunk is not None:
        if type(prefill_chunk) is not int or prefill_chunk != 2048:
            raise ValueError('Only the explicit 2048-token chunk diagnostic is supported')
        argv[argv.index('--long-prefill-token-threshold') + 1] = str(prefill_chunk)
    return [sys.executable, '-B', '-m', 'vllm.entrypoints.cli.main', *argv]


def install_native_seeds(target, env):
    """Install optional artifact-local seeds; the native arm is its run directory."""
    from prefill_autotune_control import install_seed as install_flashinfer
    from prefill_inductor_control import install_seed as install_inductor

    receipts = dict(flashinfer=install_flashinfer(target, target, env),
                    inductor=install_inductor(target, target, env))
    if receipts['inductor'] is not None:
        env['PREFILL_INDUCTOR_RUN_DIR'] = str(target.resolve())
    return receipts


def audit_native_seeds(target, phase):
    if phase not in ('ready', 'final', 'analysis'):
        raise ValueError('Unknown native seed audit phase')
    from prefill_autotune_control import audit_seed as audit_flashinfer
    from prefill_inductor_control import audit_seed as audit_inductor

    result = dict(flashinfer=audit_flashinfer(target), inductor=audit_inductor(target))
    if any(value is not None for value in result.values()):
        save(target / f'autotune-{phase}-audit.json', result)
    return result


def native_environment_receipt(env):
    """Record controlled settings without serializing inherited credentials."""
    names = ('PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'MAX_JOBS', 'FLASHINFER_NVCC_THREADS',
        'VLLM_LOG_STATS_INTERVAL', 'VLLM_PLUGINS', 'VLLM_USE_V2_MODEL_RUNNER',
        'VLLM_ALLOW_LONG_MAX_MODEL_LEN', 'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR',
        'VLLM_CACHE_ROOT', 'TORCHINDUCTOR_CACHE_DIR', 'PREFILL_INDUCTOR_RUN_DIR',
        'NATIVE_PREFILL_PROBE_PLAN', 'NATIVE_PREFILL_PROBE_OUTPUT')
    return {name: env[name] for name in names if name in env}


def client(root):
    import numpy as np
    import ruler_protocol as protocol
    case, launch = read(root / 'case.json'), read(root / 'native-launch.json')
    source = Path(case['input_path'])
    if sha(source) != case['row']['served_sha256']:
        raise ValueError('Frozen input changed')
    tokens = np.fromfile(source, dtype='<i4').tolist()
    if len(tokens) != case['row']['served_input_tokens']:
        raise ValueError('Truncated input')
    body = dict(case['sampling'], model='kvmem-graph-v2', prompt=tokens,
                return_token_ids=True, stream=True, stream_options={'include_usage': True})
    save(root / 'request.json', dict(sampling=case['sampling'], input_tokens=len(tokens),
        input_sha256=sha(source), sent_field_names=sorted(body)))
    request = urllib.request.Request(f"http://127.0.0.1:{launch['port']}/v1/completions",
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    result = dict(output_ids=[], text='', chunks=[], usage=None, response_id=None, finish_reason=None)
    done = False
    try:
        with http().open(request, timeout=1800) as response:
            for line in response:
                if not line.startswith(b'data:'):
                    continue
                payload = line[5:].strip()
                if payload == b'[DONE]':
                    done = True
                    break
                event = json.loads(payload)
                if event.get('error'):
                    raise RuntimeError(str(event['error']))
                if event.get('id'):
                    if result['response_id'] not in (None, event['id']):
                        raise ValueError('Response identity changed')
                    result['response_id'] = event['id']
                result['usage'] = event.get('usage') or result['usage']
                for choice in event.get('choices', []):
                    if choice.get('index', 0) != 0:
                        raise ValueError('Unexpected output index')
                    ids = choice.get('token_ids') or []
                    if ids:
                        result['output_ids'].extend(ids)
                        result['chunks'].append([time.monotonic() - start, len(ids)])
                    result['text'] += choice.get('text') or ''
                    result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'HTTP {error.code}: {error.read().decode()}') from None
    result['seconds'] = time.monotonic() - start
    if not (done and result['usage'] and result['output_ids'] and result['response_id']):
        raise ValueError('Incomplete native stream')
    if (result['usage']['prompt_tokens'] != len(tokens) or
            result['usage']['completion_tokens'] != len(result['output_ids']) or
            len(result['output_ids']) > case['sampling']['max_tokens'] or
            result['finish_reason'] not in ('stop', 'length')):
        raise ValueError('Native token accounting failed')
    result['ttft_seconds'] = result['chunks'][0][0]
    save(root / 'response.json', result)
    scorer = protocol.load_scorer(Path(case['upstream_path']))
    answer = protocol.final_answer(result['text'], starts_in_thinking=case['starts_in_thinking'])
    config = scorer[0][protocol.TASK_FAMILIES[case['row']['task']]]
    score = config['metric_fn']([scorer[1](answer['text'], config)], [case['row']['outputs']])
    save(root / 'result.json', dict(status='completed', input_tokens=len(tokens),
        input_sha256=sha(source), completion_tokens=len(result['output_ids']),
        ttft_seconds=result['ttft_seconds'], effective_input_tokens_per_second=len(tokens)/result['ttft_seconds'],
        seconds=result['seconds'], finish_reason=result['finish_reason'],
        official_case_score=score, thinking_complete=answer['thinking_complete'],
        answer=answer['text'], token_accounting_passed=True))


def metrics(port):
    with http().open(f'http://127.0.0.1:{port}/metrics', timeout=5) as response:
        return [line for line in response.read().decode().splitlines() if line.startswith('vllm:')]


def idle(lines):
    active = [float(line.rsplit(' ', 1)[1]) for line in lines
              if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
    return len(active) >= 2 and not any(active)


def request_cases(root):
    """A suite reuses one server; each request still submits its entire prompt."""
    if not (root / 'suite.json').exists():
        return [('single', False, root)]
    cases, seen = [], set()
    for row in read(root / 'suite.json')['cases']:
        name = row['id']
        if not isinstance(name, str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) or name in seen:
            raise ValueError('Invalid or duplicate suite case ID')
        if type(row['warmup']) is not bool:
            raise ValueError('Warmup must be explicit')
        target = root / 'requests' / name
        if sha(target / 'case.json') != row['case_sha256']:
            raise ValueError('Frozen suite case changed')
        seen.add(name)
        cases.append((name, row['warmup'], target))
    if not cases or all(warmup for _, warmup, _ in cases):
        raise ValueError('Suite must include a measured request')
    return cases


def finish_service(root, *, restore_service):
    if restore_service:
        import reference_service
        reference_service.restore(root)
    elif (root / 'service-suspended.json').exists():
        plan = read(root / 'restore-plan.json')
        reason = 'User requested no DSH model restart after this benchmark'
        save(Path(plan['status_json']), dict(plan['previous_status'], status='backend_stopped_by_user_request',
            backend_pid=None, last_reference_campaign=str(root), backend_stop_reason=reason))
        save(Path(plan['service_json']), dict(plan['previous_service'], pid=None, current_native=None,
            stopped_native=plan['previous_model'], backend_stopped_at=time.time(), backend_stop_reason=reason))
        save(root / 'service-restore-status.json', dict(status='stopped_by_user_request',
            restored=False, reason=reason, at=time.time()))
    else:
        save(root / 'service-restore-status.json', dict(status='not_started', restored=False,
            reason='No existing service was suspended or started', at=time.time()))


def validate_service_policy(*, restore_service, existing_service):
    if not existing_service and restore_service:
        raise ValueError('Without an existing service, --no-restore-service is required')


def run(root, *, batch_tokens=None, prefill_chunk=None, restore_service=True, existing_service=True):
    import fcntl
    import benchmark_block_graph as lifecycle
    import reference_service
    from native_agent_launch import _environment
    validate_service_policy(restore_service=restore_service, existing_service=existing_service)
    runtime = root / 'runtime'
    if Path(__file__).resolve() != runtime / 'scripts/benchmark_native_prefill.py':
        raise ValueError('Use frozen native controller')
    for name, digest in read(root / 'runtime-proof.json')['files'].items():
        if sha(runtime / name) != digest:
            raise ValueError('Frozen runtime changed: ' + name)
    cases = request_cases(root)
    argv = native_argv(read(root / 'reference-launch.json'), 18854,
                       batch_tokens=batch_tokens, prefill_chunk=prefill_chunk)
    env, overrides = _environment(os.environ, allow_long=True)
    env.update(MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1')
    seed_receipts = install_native_seeds(root, env)
    if seed_receipts['inductor'] is not None:
        env['PYTHONPATH'] = str(runtime / 'scripts')
        argv.extend(('--worker-cls', 'prefill_inductor_worker.SeededAuditWorker'))
    save(root / 'native-launch.json', dict(argv=argv, environment_overrides=overrides, port=18854,
        environment=native_environment_receipt(env), seed_installations=seed_receipts,
        startup_inductor_audit=seed_receipts['inductor'] is not None, formal_inductor_observer=False,
        diagnostic_batch_tokens=batch_tokens,
        diagnostic_prefill_chunk=prefill_chunk, existing_service=existing_service,
        restore_service_after_test=restore_service,
        plugins_disabled=env['VLLM_PLUGINS'] == '', kvmem_environment_absent=not any(
        k.startswith('KVMEM_') for k in env), connector_absent=True, prefix_caching=False,
        scope='Unmodified vLLM; full context, no KVMem hooks, no per-step CUDA timing instrumentation.'))
    server = worker = None
    with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if existing_service:
                save(root / 'status.json', dict(status='waiting_for_idle_service', at=time.time()))
                while not idle(metrics(18846)):
                    time.sleep(5)
                reference_service.suspend(root)
            if lifecycle.gpu_processes():
                raise RuntimeError('Foreign GPU owner')
            start = time.monotonic()
            with (root / 'server.log').open('x') as log:
                server = subprocess.Popen(argv, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(root / 'server-process.json', dict(pid=server.pid, command=argv))
            save(root / 'status.json', dict(status='loading', at=time.time()))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Native server exited during loading')
                try:
                    with http().open('http://127.0.0.1:18854/health', timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Native model loading exceeded 20 minutes')
            audit_native_seeds(root, 'ready')
            cold = time.monotonic() - start
            results = []
            for case_id, warmup, target in cases:
                if target != root:
                    save(target / 'native-launch.json', read(root / 'native-launch.json'))
                initial = metrics(18854)
                if not idle(initial):
                    raise RuntimeError('Native server has an active request before measurement')
                save(target / 'initial-metrics.json', dict(metrics=initial, native_idle=True, at=time.time()))
                save(root / 'status.json', dict(status='testing', case_id=case_id, warmup=warmup,
                    completed_cases=len(results), cold_prepare_seconds=cold, at=time.time()))
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'client', '--root', str(target)]
                with (target / 'client.log').open('x') as log:
                    worker = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                save(target / 'client-process.json', dict(pid=worker.pid, command=command))
                deadline = time.monotonic() + 1800
                with (target / 'resources.jsonl').open('x') as telemetry:
                    while worker.poll() is None:
                        if server.poll() is not None or time.monotonic() > deadline:
                            raise RuntimeError('Native inference failed or timed out')
                        memory = {line.split(':')[0]:int(line.split()[1]) * 1024
                            for line in Path('/proc/meminfo').read_text().splitlines()
                            if line.startswith(('MemAvailable:', 'MemTotal:'))}
                        gpu = subprocess.check_output(['nvidia-smi',
                            '--query-gpu=memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True).strip()
                        telemetry.write(json.dumps(dict(at=time.time(), cpu=memory, gpu=gpu,
                                                       native_metrics=metrics(18854))) + '\n')
                        telemetry.flush()
                        time.sleep(5)
                if worker.returncode != 0:
                    raise RuntimeError('Native client failed; inspect ' + str(target / 'client.log'))
                final = metrics(18854)
                if not idle(final):
                    raise RuntimeError('Native owner has not retired')
                save(target / 'final-metrics.json', dict(metrics=final, native_idle=True, at=time.time()))
                results.append(dict(id=case_id, warmup=warmup, result=read(target / 'result.json')))
            audit_native_seeds(root, 'final')
            save(root / 'status.json', dict(status='completed', cold_prepare_seconds=cold, at=time.time(),
                protocol_passed=True, native_idle=True, results=results,
                **({'result': results[0]['result']} if cases[0][2] == root else {})))
        except BaseException as error:
            save(root / 'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining = lifecycle.stop_owned_groups((worker, server))
            save(root / 'cleanup.json', dict(remaining_owned_groups=remaining,
                                            gpu_processes=sorted(lifecycle.gpu_processes())))
            if remaining:
                raise RuntimeError('Owned native process survived cleanup')
            try:
                finish_service(root, restore_service=restore_service)
            except BaseException as error:
                save(root / 'service-restore-status.json', dict(status='failed', error=repr(error), at=time.time()))
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'client'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--batch-tokens', type=int, choices=(16384,), default=None,
                        help='Explicit native aligned-batch diagnostic; otherwise keep the reference budget')
    parser.add_argument('--no-restore-service', action='store_true',
                        help='Leave the recorded DSH model stopped, only when explicitly requested')
    parser.add_argument('--no-existing-service', action='store_true',
                        help='Start only on an unused GPU; requires --no-restore-service')
    parser.add_argument('--prefill-chunk', type=int, choices=(2048,), default=None,
                        help='Cap each native prefill step without changing the engine batch budget')
    args = parser.parse_args()
    if args.command == 'run':
        run(args.root.resolve(), batch_tokens=args.batch_tokens, prefill_chunk=args.prefill_chunk,
            restore_service=not args.no_restore_service, existing_service=not args.no_existing_service)
    else:
        if (args.batch_tokens is not None or args.no_restore_service or
                args.no_existing_service or args.prefill_chunk is not None):
            parser.error('Server lifecycle/batch flags belong to the controller')
        client(args.root.resolve())
