"""Diagnostic KVMem counterpart to the frozen native full-prompt benchmark.

Runs serial independent owners with an explicit warmup. This is a performance
probe, not a new RULER accuracy cohort. Never suspends or starts a DSH service.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from benchmark_native_prefill import http, idle, metrics, read, request_cases, save, sha


def validate_diagnostic_profile(profile):
    from guided_benchmark_protocol import require, validate_profile
    require(profile['kvmem']['select_budget'] in (32768, 36864), 'Unsupported diagnostic K budget')
    require(profile['kvmem']['action_native_scores'] in ('off', 'reuse'), 'Performance mode must omit dual scoring')
    # Reuse the established policy guard, with this diagnostic's explicit K
    # exception. The actual K is checked against the saved serving launch too.
    canonical = deepcopy(profile)
    canonical['kvmem']['select_budget'] = 36864
    validate_profile(canonical, 'off', 'niah', expected_chunk=2048)


def audit_native_stats(stats):
    from guided_benchmark_protocol import require
    require(stats['requested_mode'] == 'reuse' and stats['disabled_reason'] is None and
            not stats['validation_failures'], 'Native reuse disabled or producer validation failed')
    counts = stats['layer_phase_counts']
    require(counts.get('reuse:prefill', 0) > 0, 'No actual prefill QK reuse')
    unexpected = {k:v for k,v in counts.items() if v and k.startswith('fallback:') and
                  k != 'fallback:prefill:unprepared_forward'}
    require(not unexpected and counts.get('fallback:prefill:unprepared_forward', 0) <= 3,
            'Unexpected native fallback: ' + str(unexpected or counts))


def client(root):
    import numpy as np
    from tokenizers import Tokenizer
    import qualify_block_graph as q
    import qualify_graph_niah as old
    import ruler_protocol
    from benchmark_client_io import CaptureSlice
    from types import SimpleNamespace

    campaign = root.parent.parent
    profile, case = read(campaign / 'profile.json'), read(root / 'case.json')
    capture = campaign / 'run/server/capture'
    source, row = Path(case['input_path']), case['row']
    q.require(sha(source) == row['served_sha256'], 'Frozen input changed')
    tokens = np.fromfile(source, dtype='<i4').tolist()
    tokenizer = Tokenizer.from_file(str(Path(profile['tokenizer']) / 'tokenizer.json'))
    q.require(len(tokens) == row['served_input_tokens'] and
              tokenizer.decode(tokens[slice(*row['served_query'])], skip_special_tokens=False) == row['question'],
              'Full input or Query coordinates changed')
    save(root / 'launch-audit.json', old.audit_launch(profile, capture))
    before_native = read(capture / 'native_attention_stats.json') if (capture / 'native_attention_stats.json').exists() else None
    session = 'matched-prefill-' + uuid.uuid4().hex
    save(root / 'request.json', dict(sampling=case['sampling'], session_id=session,
        prompt_sha256=sha(source), prompt_tokens=len(tokens), kvmem_query_span=row['served_query'],
        kvmem_reselect='force'))
    events = CaptureSlice(capture / name for name in
        ('bounded_refresh.jsonl', 'block_graph.jsonl', 'action_attention.jsonl'))
    response = old.stream(profile, session, tokens, row['served_query'], case['sampling'], 1800)
    save(root / 'response.json', response)
    terminal = old.wait_retired(capture, response['response_id'])
    q.require(terminal['history_end'] == len(tokens) + len(response['output_ids']) - 1,
              'Accepted history accounting failed')
    q.require(not terminal.get('checkpoint_tokens'), 'Independent owner retained a checkpoint')
    save(root / 'release.json', old.wait_owner_released(capture, session, response, terminal))
    refresh = [e for e in events.read(capture / 'bounded_refresh.jsonl') if q.native_matches(response, e)]
    graph = [e for e in events.read(capture / 'block_graph.jsonl') if q.native_matches(response, e)]
    if len(tokens) <= profile['kvmem']['select_budget']:
        q.require(not refresh and not graph, 'Fitting-prompt probe unexpectedly performed a selection')
        selection = dict(refresh=[], graph=[], pressure=[], query=None,
            selection_skipped='All history fits; no pressure or Query selection required')
    else:
        import guided_benchmark_protocol as policy
        selection = dict(refresh=refresh, **policy.audit_case(capture, response['response_id'], profile,
            row['served_query'], require_pressure=True, event_reader=events.read))
    action = [e for e in events.read(capture / 'action_attention.jsonl') if q.native_matches(response, e)]
    expected = q.expected_output_samples(SimpleNamespace(convert_tokens_to_ids=tokenizer.token_to_id),
        profile, tokens, response['output_ids'])
    q.require(len(action) == 1 and action[0]['samples'] == expected, 'Accepted attention sample mismatch')
    q.require(action[0]['prefill_samples'] > 0 and action[0]['block_graph']['edges'] > 0
              and action[0]['block_graph']['accepted_through'] == terminal['history_end'],
              'Input statistics or graph construction missing')
    save(root / 'evidence.json', dict(**selection, terminal=terminal, action=action))
    if profile['kvmem']['action_native_scores'] == 'reuse':
        after_native = read(capture / 'native_attention_stats.json')
        audit_native_stats(after_native)
        before_counts = (before_native or {}).get('layer_phase_counts', {})
        save(root / 'native-stats.json', dict(before=before_native, after=after_native,
            delta={k:v-before_counts.get(k, 0) for k,v in after_native['layer_phase_counts'].items()}))
    scorer = ruler_protocol.load_scorer(Path(case['upstream_path']))
    answer = ruler_protocol.final_answer(response['text'], starts_in_thinking=case['starts_in_thinking'])
    config = scorer[0][ruler_protocol.TASK_FAMILIES[row['task']]]
    score = config['metric_fn']([scorer[1](answer['text'], config)], [row['outputs']])
    save(root / 'result.json', dict(status='completed', input_tokens=len(tokens), input_sha256=sha(source),
        completion_tokens=len(response['output_ids']), ttft_seconds=response['ttft_seconds'],
        effective_prefill_tokens_per_second=len(tokens) / response['ttft_seconds'],
        seconds=response['seconds'], finish_reason=response['finish_reason'], official_case_score=score,
        thinking_complete=answer['thinking_complete'], answer=answer['text'],
        token_accounting_passed=True, pressure_count=len(selection['pressure']),
        query_count=int(selection['query'] is not None)))


def run(root):
    import benchmark_block_graph as lifecycle
    from benchmark_prefill_publication_pipeline import owned_process_memory, PROCESS_MEMORY_SCOPE
    from guided_benchmark_protocol import require

    runtime = root / 'runtime'
    require(Path(__file__).resolve() == runtime / 'scripts/benchmark_matched_prefill.py', 'Use frozen controller')
    for name, digest in read(root / 'runtime-proof.json')['files'].items():
        require(sha(runtime / name) == digest, 'Frozen runtime changed: ' + name)
    profile = read(root / 'profile.json')
    require(sha(root / 'profile.json') == read(root / 'manifest.json')['profile_sha256'], 'Profile changed')
    validate_diagnostic_profile(profile)
    cases = request_cases(root)
    target = root / 'run'
    target.mkdir(exist_ok=False)
    env = {k:v for k,v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_', 'DEEPSEEK_'))}
    env.update(PYTHONPATH=str(runtime / 'src'), PYTHONDONTWRITEBYTECODE='1',
        TOKENIZERS_PARALLELISM='false', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1', VLLM_LOG_STATS_INTERVAL='1')
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', os.defpath)
    for toolkit in ('/usr/local/cuda-13', '/usr/local/cuda'):
        if (Path(toolkit) / 'bin/nvcc').exists():
            env['CUDA_HOME'] = toolkit
            break
    seeded = (read(root / 'manifest.json').get('flashinfer_seed_manifest_sha256') is not None
              or (root / 'flashinfer_seed').exists())
    if seeded:
        from prefill_autotune_control import install_seed
        install_seed(root, target, env)
    inductor_seeded = (read(root / 'manifest.json').get('inductor_seed_manifest_sha256') is not None
                       or (root / 'inductor_seed').exists())
    if inductor_seeded:
        from prefill_inductor_control import install_seed as install_inductor_seed
        install_inductor_seed(root, target, env)
        env['PYTHONPATH'] = os.pathsep.join((str(runtime / 'src'), str(runtime / 'scripts')))
    server = worker = None
    with open('/tmp/kvmem-coordination-qualification.lock', 'a+') as lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            require(not lifecycle.gpu_processes(), 'Foreign GPU owner')
            command = [sys.executable, '-B', '-m', 'vllm_kvmem', 'serve',
                '--config', str(root / 'profile.json'), '--run-dir', str(target / 'server')]
            if inductor_seeded:
                command = [sys.executable, '-B', str(runtime / 'scripts/prefill_inductor_serve.py'),
                    '--config', str(root / 'profile.json'), '--run-dir', str(target / 'server')]
            start = time.monotonic()
            with (target / 'server.log').open('x') as log:
                server = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            save(target / 'server-process.json', dict(pid=server.pid, command=command))
            save(root / 'status.json', dict(status='loading', at=time.time()))
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                require(server.poll() is None, 'Server exited during loading')
                try:
                    with http().open(f"http://127.0.0.1:{profile['port']}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Model loading exceeded 20 minutes')
            cold = time.monotonic() - start
            if seeded:
                from prefill_autotune_control import audit_seed
                save(target / 'flashinfer-seed-ready-audit.json', audit_seed(root))
            if inductor_seeded:
                from prefill_inductor_control import audit_seed as audit_inductor_seed
                save(target / 'inductor-seed-ready-audit.json', audit_inductor_seed(root))
            results = []
            for case_id, warmup, directory in cases:
                before = metrics(profile['port'])
                require(idle(before), 'Native request active before measurement')
                save(directory / 'initial-metrics.json', dict(at=time.time(), metrics=before))
                save(root / 'status.json', dict(status='testing', case_id=case_id, warmup=warmup,
                    completed_cases=len(results), cold_prepare_seconds=cold, at=time.time()))
                command = [sys.executable, '-B', str(Path(__file__).resolve()), 'client', '--root', str(directory)]
                with (directory / 'client.log').open('x') as log:
                    worker = subprocess.Popen(command, cwd=runtime, env=env, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                save(directory / 'client-process.json', dict(pid=worker.pid, command=command))
                deadline = time.monotonic() + 1800
                with (directory / 'resources.jsonl').open('x') as telemetry:
                    while worker.poll() is None:
                        require(server.poll() is None and time.monotonic() < deadline, 'Inference failed or timed out')
                        gpu = subprocess.check_output(['nvidia-smi',
                            '--query-gpu=memory.used,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True).strip()
                        telemetry.write(json.dumps(dict(at=time.time(), gpu=gpu,
                            **owned_process_memory(server.pid),
                            process_tree_memory_scope=PROCESS_MEMORY_SCOPE)) + '\n')
                        telemetry.flush()
                        time.sleep(1)
                require(worker.returncode == 0, 'Client failed; inspect ' + str(directory / 'client.log'))
                after = metrics(profile['port'])
                require(idle(after), 'Native owner did not retire')
                save(directory / 'final-metrics.json', dict(at=time.time(), metrics=after))
                results.append(dict(id=case_id, warmup=warmup, result=read(directory / 'result.json')))
            capture = target / 'server/capture'
            config = read(capture / 'action_config.json')
            mode = profile['kvmem']['action_native_scores']
            require(config['native_scores'] == mode and all(config[k] is True for k in
                ('native_decode_commit', 'download_pool', 'fused_scoring')), 'Optimization mode drift')
            actions = [json.loads(line) for line in (capture / 'action_attention.jsonl').open()]
            require(len(actions) == len(cases) and all(
                (mode == 'reuse' or a['scoring_backend_counts'].get('fused', 0) > 0) and
                not a['scoring_backend_counts'].get('reference', 0) and
                not a.get('decode_commit_python_calls', 0) and
                not a['download_pool_total']['exhaustions'] and
                not a['download_pool_total']['quarantined_slots'] for a in actions),
                'Unexpected scoring/commit fallback or download pool failure')
            if mode == 'reuse':
                audit_native_stats(read(capture / 'native_attention_stats.json'))
            work = read(capture / 'public_work.json')
            require(work['all_submitted_completed'] and work['submitted'] == work['completed'] and
                work['completed']['public_prefill_tokens'] == sum(x['result']['input_tokens'] for x in results),
                'Public input forwarding or final fence accounting failed')
            save(target / 'execution-audit.json', dict(passed=True, action_config=config,
                completed_requests=len(cases),
                pressure_count=sum(x['result']['pressure_count'] for x in results),
                query_count=sum(x['result']['query_count'] for x in results), public_work=work))
            save(root / 'status.json', dict(status='completed', cold_prepare_seconds=cold,
                protocol_passed=True, results=results, at=time.time()))
        except BaseException as error:
            save(root / 'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining = lifecycle.stop_owned_groups((worker, server))
            save(root / 'cleanup.json', dict(remaining_owned_groups=remaining,
                gpu_processes=sorted(lifecycle.gpu_processes())))
            require(not remaining, 'Owned process survived cleanup')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'client'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    (run if args.command == 'run' else client)(args.root.resolve())
