"""Run the frozen RULER cohort on ordinary vLLM with all prompt KV visible.

No KVMem package is imported by the server or client. Admission reserves each
request's complete prompt plus output allowance against the native KV capacity;
the engine's batch, sequence, memory and model settings stay unchanged.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import benchmark_native_prefill as native
import ruler_protocol as protocol
from native_agent_launch import _environment

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ('benchmark_native_ruler.py', 'benchmark_native_prefill.py',
           'native_agent_launch.py', 'ruler_protocol.py', 'benchmark_block_graph.py')


def ordered_rows(fixture):
    tasks = list(protocol.TASK_FAMILIES)
    return sorted(fixture['rows'], key=lambda row: (row['index'], tasks.index(row['task'])))


def capacity_plan(log, fixture, profile):
    capacities = re.findall(r'GPU KV cache size:\s*([\d,]+) tokens', log)
    blocks = re.findall(r'Setting attention block size to (\d+) tokens', log)
    if len(capacities) != 1 or len(blocks) != 1:
        raise ValueError('Require one unambiguous native KV capacity and block size')
    capacity, block = int(capacities[0].replace(',', '')), int(blocks[0])
    if capacity <= 0 or block <= 0:
        raise ValueError('Invalid native cache capacity')
    longest = max(row['served_input_tokens'] for row in fixture['rows'])
    # Include full output allowance, speculative lookahead, block rounding and
    # two extra blocks per owner. Leave a batch plus 5% global capacity unused.
    required = longest + fixture['sampling']['max_tokens'] + profile['draft_tokens'] + 1
    per_request = math.ceil(required / block) * block + 2 * block
    available = math.floor(capacity * .95) - profile['max_num_batched_tokens']
    workers = min(fixture['workers'], profile['max_num_seqs'], max(0, available // per_request))
    if workers < 1:
        raise ValueError('Native KV cannot safely hold even one complete request')
    return dict(workers=workers, reference_workers=fixture['workers'], cache_tokens=capacity,
                native_block_tokens=block, reserved_tokens_per_request=per_request,
                admission_capacity_tokens=available, max_input_tokens=longest,
                output_allowance_tokens=fixture['sampling']['max_tokens'],
                policy='Full prompt + full output + MTP + block slack; global 5% and one batch slack.',
                server_max_num_seqs=profile['max_num_seqs'])


def preemptions(lines):
    values = [float(line.rsplit(' ', 1)[1]) for line in lines
              if line.startswith('vllm:num_preemptions_total{')]
    if not values:
        raise ValueError('Missing native preemption counter')
    return sum(values)


def prepare(reference, output, port):
    reference, output = reference.resolve(), output.resolve()
    fixture = native.read(reference / 'fixture.json')
    profile = native.read(reference / 'profile.json')
    completed = native.read(reference / 'run/answers/completed.json')
    protocol.validate_cohort(fixture['rows'], fixture['samples_per_task'])
    if fixture['cases'] != 260 or fixture['target_length'] != 262144 or completed['cases'] != 260:
        raise ValueError('Require the completed, identical 13 x 20 at 256K reference')
    if native.sha(reference / 'profile.json') != fixture['profile_sha256']:
        raise ValueError('Reference profile changed')
    if native.sha(Path(profile['tokenizer']) / 'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Reference tokenizer changed')
    protocol.load_scorer(Path(fixture['upstream_path']))
    for row in fixture['rows']:
        path = Path(fixture['input_directory']) / (row['id'] + '.i32')
        if native.sha(path) != row['served_sha256'] or path.stat().st_size != 4 * row['served_input_tokens']:
            raise ValueError('Reference input changed: ' + row['id'])
        if row['seed'] != fixture['seed'] + row['index']:
            raise ValueError('Reference per-case seed differs')
    saved = native.read(reference / 'run/server/launch.json')
    argv = native.native_argv(saved, port, prefill_chunk=profile['prefill_chunk_tokens'])
    output.mkdir(parents=True, exist_ok=False)
    scripts = output / 'runtime/scripts'
    scripts.mkdir(parents=True)
    for name in HELPERS:
        shutil.copyfile(ROOT / 'scripts' / name, scripts / name)
    for source, name in ((reference / 'fixture.json', 'fixture.json'),
                         (reference / 'profile.json', 'reference-profile.json'),
                         (reference / 'run/server/launch.json', 'reference-launch.json')):
        shutil.copyfile(source, output / name)
    native.save(output / 'runtime-proof.json', dict(files={
        'scripts/' + name: native.sha(scripts / name) for name in HELPERS}))
    launch = dict(argv=argv, port=port, model=profile['served_model_name'],
        plugins_disabled=True, kvmem_hooks=False, prefix_caching=False,
        full_context=True, max_model_len=profile['max_model_len'],
        native_prefill_chunk=profile['prefill_chunk_tokens'],
        reference=str(reference), reference_fixture_sha256=native.sha(reference / 'fixture.json'),
        reference_profile_sha256=native.sha(reference / 'profile.json'),
        reference_launch_sha256=native.sha(reference / 'run/server/launch.json'),
        reference_completed_sha256=native.sha(reference / 'run/answers/completed.json'),
        runtime_proof_sha256=native.sha(output / 'runtime-proof.json'),
        differences=['KVMem hooks, scheduler, connector, history and graph selection removed.',
                     'Native attention sees every input token; no pressure or Query reselection.',
                     'Native hybrid KV block/state policy; no KVMem state patches.',
                     'Native per-request prefill threshold matches the logical 2048-token chunk.',
                     'Client concurrency is limited by measured full-context KV capacity.',
                     'First scored case runs alone as a protocol qualification, followed by continuous FIFO.'])
    native.save(output / 'native-launch.json', launch)
    native.save(output / 'launch-proof.json', dict(sha256=native.sha(output / 'native-launch.json')))
    native.save(output / 'status.json', dict(status='prepared', cases=260, at=time.time()))


def validate_frozen(root):
    launch = native.read(root / 'native-launch.json')
    if native.sha(root / 'native-launch.json') != native.read(root / 'launch-proof.json')['sha256']:
        raise ValueError('Native launch changed')
    if native.sha(root / 'runtime-proof.json') != launch['runtime_proof_sha256']:
        raise ValueError('Runtime proof changed')
    for name, digest in native.read(root / 'runtime-proof.json')['files'].items():
        if native.sha(root / 'runtime' / name) != digest:
            raise ValueError('Frozen helper changed: ' + name)
    for name, key in (('fixture.json', 'reference_fixture_sha256'),
                      ('reference-profile.json', 'reference_profile_sha256'),
                      ('reference-launch.json', 'reference_launch_sha256')):
        if native.sha(root / name) != launch[key]:
            raise ValueError('Frozen input configuration changed: ' + name)
    expected = native.native_argv(native.read(root / 'reference-launch.json'), launch['port'],
                                prefill_chunk=launch['native_prefill_chunk'])
    if launch['argv'] != expected:
        raise ValueError('Native launch is not the reference with only declared changes')
    return launch


def infer(root, row, fixture, launch):
    target = root / 'run/answers/cases' / row['id']
    target.mkdir(parents=True, exist_ok=False)
    native.save(target / 'case.json', dict(row=row,
        input_path=str(Path(fixture['input_directory']) / (row['id'] + '.i32')),
        sampling=dict(fixture['sampling'], seed=row['seed']),
        upstream_path=fixture['upstream_path'], starts_in_thinking=fixture['starts_in_thinking']))
    native.save(target / 'native-launch.json', dict(port=launch['port']))
    native.client(target)
    result = native.read(target / 'result.json')
    response = native.read(target / 'response.json')
    result.update(id=row['id'], task=row['task'], index=row['index'], seed=row['seed'],
                  response_id=response['response_id'], text=response['text'])
    native.save(target / 'result.json', result)
    return result


def client(root):
    launch = validate_frozen(root)
    fixture = native.read(root / 'fixture.json')
    profile = native.read(root / 'reference-profile.json')
    if native.sha(Path(profile['tokenizer']) / 'tokenizer.json') != fixture['tokenizer_sha256']:
        raise ValueError('Tokenizer changed')
    scorer = protocol.load_scorer(Path(fixture['upstream_path']))
    protocol.validate_cohort(fixture['rows'], fixture['samples_per_task'])
    plan = capacity_plan((root / 'run/server.log').read_text(), fixture, profile)
    native.save(root / 'run/admission.json', plan)
    output = root / 'run/answers'
    output.mkdir(exist_ok=False)
    rows, results = ordered_rows(fixture), []
    started = time.monotonic()

    def record(result):
        results.append(result)
        native.save(output / 'progress.json', dict(completed=len(results), expected=len(rows),
            at=time.time(), workers=plan['workers'], seconds=time.monotonic() - started,
            per_task={task:sum(r['task'] == task for r in results) for task in protocol.TASK_FAMILIES}))
        print(json.dumps(dict(completed=len(results), id=result['id'],
                              score=result['official_case_score'])), flush=True)

    try:
        # Qualify the complete input/stream/token accounting before parallel work.
        record(infer(root, rows[0], fixture, launch))
        native.save(root / 'run/protocol-qualification.json', dict(
            passed=True, case_id=rows[0]['id'], input_sha256=results[0]['input_sha256'],
            input_tokens=results[0]['input_tokens'], token_accounting_passed=True,
            preemptions=preemptions(native.metrics(launch['port'])), at=time.time()))
        remaining = iter(rows[1:])
        with ThreadPoolExecutor(max_workers=plan['workers']) as executor:
            pending = {executor.submit(infer, root, next(remaining), fixture, launch)
                       for _ in range(plan['workers'])}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                # Consume all outcomes before admitting more; fail without
                # sending another request if any completed stream is invalid.
                batch = [future.result() for future in done]
                for result in batch:
                    record(result)
                for _ in batch:
                    row = next(remaining, None)
                    if row is not None:
                        pending.add(executor.submit(infer, root, row, fixture, launch))
        elapsed = time.monotonic() - started
        final_metrics = native.metrics(launch['port'])
        native.save(root / 'run/final-metrics.json', dict(at=time.time(), metrics=final_metrics))
        if not native.idle(final_metrics) or preemptions(final_metrics) != 0:
            raise ValueError('Native requests did not retire cleanly or KV preemption occurred')
        predictions = {(r['task'], r['index']):r['text'] for r in results}
        scores = protocol.score_suite(rows, predictions, scorer,
            samples_per_task=fixture['samples_per_task'], starts_in_thinking=fixture['starts_in_thinking'])
        native.save(output / 'scores.json', scores)
        native.save(output / 'completed.json', dict(status='completed', cases=len(results),
            seconds=elapsed, score=scores['macro_score'], protocol_passed=True,
            native_idle=True, preemptions=0, workers=plan['workers'],
            fixture_sha256=native.sha(root / 'fixture.json'), results=results,
            input_tokens=sum(r['input_tokens'] for r in results),
            completion_tokens=sum(r['completion_tokens'] for r in results),
            output_tokens_per_wall_second=sum(r['completion_tokens'] for r in results) / elapsed,
            throughput_scope='Whole campaign including full-context prefill and qualification; not pure decode.'))
        report(root, fixture, scores, results, elapsed)
    except BaseException as error:
        native.save(output / 'failure.json', dict(error=repr(error), at=time.time()))
        raise


def report(root, fixture, scores, results, elapsed):
    launch = native.read(root / 'native-launch.json')
    reference = Path(launch['reference'])
    baseline = native.read(reference / 'run/answers/scores.json')
    lines = ['# 原生 vLLM 完整上下文 RULER 对照', '',
        '同一套冻结的 256K 输入，每类 20 题，共 260 题。原生路径无 KVMem，全部输入 KV 可见。', '',
        f"总分：原生 **{scores['macro_score']:.2f}**，KVMem **{baseline['macro_score']:.2f}**。", '',
        '| 任务 | 原生 vLLM | KVMem | 原生 − KVMem |', '|---|---:|---:|---:|']
    for task in protocol.TASK_FAMILIES:
        value, old = scores['per_task'][task]['score'], baseline['per_task'][task]['score']
        lines.append(f'| {task} | {value:.2f} | {old:.2f} | {value-old:+.2f} |')
    plan = native.read(root / 'run/admission.json')
    lines += ['', f"原生实际并发 {plan['workers']}，KVMem 参考并发 {fixture['workers']}。",
        f'推理总耗时 {elapsed:.2f} 秒；输出 {sum(r["completion_tokens"] for r in results):,} token。',
        '吞吐口径包含 prefill、生成和首题串行核验，不能当作纯 decode 吞吐；并发和生成轨迹差异需单列。',
        '冷准备、逐题 TTFT、生成时间、MTP 接受统计、原生服务指标及 CPU/GPU 内存见同目录的原始记录。',
        '', '设置差异：', *('- ' + value for value in launch['differences']), '']
    (root / 'run/answers/report.md').write_text('\n'.join(lines))


def run(root):
    import benchmark_block_graph as lifecycle
    if Path(__file__).resolve() != root / 'runtime/scripts/benchmark_native_ruler.py':
        raise ValueError('Use the frozen native controller')
    launch = validate_frozen(root)
    env, overrides = _environment(os.environ, allow_long=True)
    env.update(PYTHONDONTWRITEBYTECODE='1', MAX_JOBS='4', FLASHINFER_NVCC_THREADS='1',
               VLLM_LOG_STATS_INTERVAL='1')
    if env.get('PYTHONPATH') or any(k.upper().startswith('KVMEM_') for k in env) or env['VLLM_PLUGINS']:
        raise ValueError('Native server inherited KVMem import hooks')
    target = root / 'run'
    target.mkdir(exist_ok=False)
    native.save(target / 'environment.json', dict(overrides=overrides,
        controlled=native.native_environment_receipt(env), kvmem_environment_absent=True,
        plugins_disabled=True, pythonpath_absent=True))
    server = worker = None
    with (root / '.campaign.lock').open('a+') as campaign_lock, \
            open('/tmp/kvmem-coordination-qualification.lock', 'a+') as gpu_lock, \
            lifecycle.cleanup_on_termination() as begin_cleanup:
        for lock in (campaign_lock, gpu_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if lifecycle.gpu_processes():
                raise RuntimeError('Foreign GPU owner; no service will be interrupted')
            start = time.monotonic()
            with (target / 'server.log').open('x') as log:
                server = subprocess.Popen(launch['argv'], cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            native.save(target / 'server-process.json', dict(pid=server.pid, command=launch['argv']))
            native.save(root / 'status.json', dict(status='loading', server_pid=server.pid, at=time.time()))
            deadline = start + 1200
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Native model exited while loading')
                try:
                    with native.http().open(f"http://127.0.0.1:{launch['port']}/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError('Native model preparation exceeded 20 minutes')
            prepared = time.monotonic()
            initial = native.metrics(launch['port'])
            if not native.idle(initial) or preemptions(initial):
                raise ValueError('Native server is not initially idle and clean')
            native.save(target / 'initial-metrics.json', dict(at=time.time(), metrics=initial))
            command = [sys.executable, '-B', str(Path(__file__).resolve()), 'client', '--root', str(root)]
            with (target / 'client.log').open('x') as log:
                worker = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            native.save(target / 'client-process.json', dict(pid=worker.pid, command=command))
            native.save(root / 'status.json', dict(status='testing', server_pid=server.pid,
                client_pid=worker.pid, cases=260, cold_prepare_seconds=prepared - start, at=time.time()))
            with (target / 'resources.jsonl').open('x') as telemetry:
                while worker.poll() is None:
                    if server.poll() is not None or time.monotonic() - prepared > 24 * 3600:
                        raise RuntimeError('Native server exited or campaign exceeded 24 hours')
                    if (target / 'answers/failure.json').exists():
                        raise RuntimeError('Native client reported a failure; inspect answers/failure.json')
                    memory = {line.split(':')[0]:int(line.split()[1]) * 1024
                              for line in Path('/proc/meminfo').read_text().splitlines()
                              if line.startswith(('MemAvailable:', 'MemTotal:'))}
                    gpu = subprocess.check_output(['nvidia-smi',
                        '--query-gpu=memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'], text=True).strip()
                    current = native.metrics(launch['port'])
                    if preemptions(current):
                        raise RuntimeError('Unexpected full-context KV preemption; admission must be revised')
                    telemetry.write(json.dumps(dict(at=time.time(), cpu=memory, gpu=gpu,
                                                    native_metrics=current)) + '\n')
                    telemetry.flush()
                    time.sleep(5)
            if worker.returncode != 0:
                raise RuntimeError('Native RULER client failed; inspect client.log')
            completed = native.read(target / 'answers/completed.json')
            if completed['cases'] != 260 or not completed['protocol_passed']:
                raise ValueError('Incomplete native cohort')
            native.save(root / 'status.json', dict(status='completed', cases=260,
                score=completed['score'], cold_prepare_seconds=prepared-start,
                inference_seconds=time.monotonic()-prepared, at=time.time()))
        except BaseException as error:
            native.save(root / 'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise
        finally:
            begin_cleanup()
            remaining = lifecycle.stop_owned_groups((worker, server))
            native.save(target / 'cleanup.json', dict(remaining_owned_groups=remaining,
                                                      gpu_processes=sorted(lifecycle.gpu_processes())))
            if remaining:
                raise RuntimeError('Owned native benchmark processes survived cleanup')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--reference', type=Path, required=True)
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--port', type=int, default=18854)
    for name in ('run', 'client'):
        sub.add_parser(name).add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.reference, args.root, args.port)
    else:
        (run if args.command == 'run' else client)(args.root.resolve())
