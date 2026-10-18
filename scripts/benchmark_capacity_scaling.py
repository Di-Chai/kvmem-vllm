"""Prepare and run a bounded, fixed-work native / KVMem capacity probe.

Preparation never starts a server. Run only the new artifact's frozen script.
The historical r3 KVMem runtime is copied without native-fastpath overlays.
"""
import argparse
import asyncio
import copy
from decimal import Decimal, ROUND_FLOOR
import fcntl
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = 'artifacts/remaining-idle-20261016-r3/performance'
REFERENCE_MANIFEST_SHA = 'a22b00ad516dca4859b6e0513d31eb8d1ea1c00c1766693819ff69b7baaa1db7'
STAGES = ('native-c4', 'kvmem-c4', 'native-offered8', 'kvmem-c8')
INPUT_TOKENS, OUTPUT_TOKENS = 229376, 8192
INDEX_BYTES_PER_REQUEST = 805_306_368  # Explicit host budget, not a measured RSS claim.
ARCHIVE_BYTES_PER_TOKEN = 34816  # Qualified FP8 target + MTP geometry; rechecked after loading.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def native_source_identity():
    distribution = importlib.metadata.distribution('vllm')
    files = ['vllm/v1/core/kv_cache_utils.py', 'vllm/v1/kv_cache_interface.py']
    return dict(version=distribution.version,
        files={str(distribution.locate_file(name).resolve()): sha(distribution.locate_file(name)) for name in files},
        capacity_formula='sum over every device group: ceil(spec.max_memory_usage_bytes(config)/spec.page_size_bytes); pool blocks / summed per-request blocks',
        includes='Resolved full-attention, native MTP attention and recurrent/speculative groups; not FA-only')


def host_memory(pids=()):
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapFree'):
            values[key] = int(value.split()[0]) * 1024
    rss, vanished = {}, []
    for pid in sorted(set(pids)):
        try:
            rows = Path(f'/proc/{pid}/status').read_text().splitlines()
            rss[str(pid)] = next(int(row.split()[1]) * 1024 for row in rows if row.startswith('VmRSS:'))
        except (FileNotFoundError, ProcessLookupError, StopIteration):
            vanished.append(pid)
    return dict(at=time.perf_counter(), wall_at=time.time(), **values,
                process_rss_bytes=rss, owned_rss_sum_bytes=sum(rss.values()), vanished_pids=vanished,
                rss_scope='RSS sum can count shared pages more than once; MemAvailable is the system headroom gate')


def host_budget(extra_bytes, min_free_bytes, *, offered_concurrency, native, fixed_history_limits=None):
    require(type(extra_bytes) is int and extra_bytes >= min_free_bytes > 0,
            'Host extra budget must include a positive system headroom floor')
    require(offered_concurrency in (4, 8) and type(native) is bool, 'Invalid host-budget stage')
    # The 100GB cache setting is a logical upper bound, not an up-front RSS
    # allocation. Budget this stage's complete 256K histories plus two partial
    # records, including the full G32 reserve, not merely its 8192-output quota.
    full_history_credit = (262144 + 2 * 32) * ARCHIVE_BYTES_PER_TOKEN
    archive = 0 if native else offered_concurrency * full_history_credit
    if fixed_history_limits is not None:
        require(len(fixed_history_limits) == offered_concurrency and
                all(type(n) is int and 0 < n <= 262144 for n in fixed_history_limits),
                'Invalid enforced fixed-work historical limits')
        # Only the frozen, fixed-quota diagnostic has a tighter finite bound.
        # The general live-agent budget stays at full256K. Keep the complete
        # MeanK index, 32GiB model/DMA/system allowance and 8GiB free floor.
        archive = 0 if native else sum(((n+31)//32*32+64)*ARCHIVE_BYTES_PER_TOKEN
                                       for n in fixed_history_limits)
    index = 0 if native else offered_concurrency * INDEX_BYTES_PER_REQUEST
    require(archive <= 100_000_000_000, 'Stage history credits exceed the configured CPU cap')
    return dict(configured_archive_cap_bytes=0 if native else 100_000_000_000,
        offered_concurrency=offered_concurrency, native_vllm=native,
        full_256k_history_credit_bytes=full_history_credit, archive_bytes=archive, mean_k_index_bytes=index,
        fixed_history_limits=fixed_history_limits,
        extra_model_dma_checkpoint_system_bytes=extra_bytes, minimum_free_bytes=min_free_bytes,
        required_prelaunch_available_bytes=archive + index + extra_bytes,
        maximum_owned_rss_growth_bytes=archive + index + extra_bytes,
        scope='Stage-specific full-history credits plus MeanK index and model/DMA/checkpoint/system budgets; CPU100GB is a cap, not preallocated RAM')


async def reconcile_outputs(snapshot, before, expected, *, timeout=30., allow_preemptions=False,
                            include_snapshot=False):
    """Wait for exported counters, scoped to an already quiescent boundary."""
    require(type(expected) is int and expected > 0, 'Expected a positive accepted-output quota')
    first = before['values']
    deadline = time.monotonic() + timeout
    while True:
        last = await snapshot()
        actual = last['values']['vllm:generation_tokens_total'] - first['vllm:generation_tokens_total']
        require(actual >= 0, 'Native generation counter reset during the stage')
        if actual >= expected or time.monotonic() >= deadline:
            break
        await asyncio.sleep(.25)
    require(actual == expected, f'Accepted output mismatch: native={actual}, client={expected}')
    preemptions = last['values']['vllm:num_preemptions_total'] - first['vllm:num_preemptions_total']
    require(preemptions >= 0, 'Preemption counter reset')
    require(allow_preemptions or preemptions == 0,
            'Capacity probe preempted: do not use allocator thrashing as a baseline')
    result = dict(passed=True, native_tokens=actual, client_tokens=expected, preemptions=preemptions,
                before_snapshot_started_at=before['started_at'], after_snapshot_started_at=last['started_at'])
    if include_snapshot:
        result['last_snapshot'] = last
    return result


def capacity_profile(reference, port):
    """Same model/device budgets, changed only to the requested capacity regime."""
    result = copy.deepcopy(reference)
    require(result['model_format'] == 'safetensors' and result['draft_method'] == 'mtp'
            and result['draft_tokens'] == 4 and result['execution_mode'] == 'compiled_graph'
            and result['gpu_memory_utilization'] == .81, 'Unexpected reference model/graph/MTP budget')
    result.update(port=port, max_model_len=262144, max_num_seqs=8)
    kv = result['kvmem']
    require(kv['block_tokens'] == 32 and kv['offline_agent_policy'] is True and
            kv['prefill_batch_tokens'] == 0 and kv['archive_backend'] == 'cpu',
            'Expected the qualified continuous policy with cap0 and CPU history')
    require(all(kv.get(name) is True for name in
                ('split_capture_dma', 'native_prefill_commit', 'async_checkpoint_capture')),
            'Use the r3 P1 profile with its three qualified host flags')
    require(not any(kv.get(name, False) for name in ('native_restore_frame', 'native_action_metadata')),
            'Do not overlay unqualified native host fastpaths')
    kv.update(select_budget=65536, gen_budget=32768, repository_blocks=8192,
              cpu_read_cache_bytes=100_000_000_000, cpu_history_cache_bytes_per_request=100_000_000_000)
    require(result['query_replay_budget'] <= kv['gen_budget'], 'Atomic Query budget exceeds G')
    return result


def native_capacity(log, *, max_model_len=262144, max_num_seqs=8):
    """Conservative bound from the native hybrid cache's own startup calculation.

    The printed ratio is rounded to two decimals. Subtract .01 before flooring;
    this avoids admitting N on a rounded-up N.00 report. Do not infer a live
    decode cohort from this bound. Missing/ambiguous geometry fails closed.
    """
    pattern = r'Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x'
    matches = [(int(length.replace(',', '')), ratio, line)
               for line in log.splitlines() for length, ratio in re.findall(pattern, line)]
    require(matches and all(length == max_model_len for length, _, _ in matches),
            'Missing or mismatched native full-history capacity geometry')
    values = {ratio for _, ratio, _ in matches}
    require(len(values) == 1, 'Conflicting native startup capacity reports')
    ratio = Decimal(next(iter(values)))
    require(ratio.is_finite() and ratio > 0, 'Invalid native capacity ratio')
    safe = min(max_num_seqs, int((ratio - Decimal('.01')).to_integral_value(rounding=ROUND_FLOOR)))
    require(safe >= 1, 'Native cache cannot safely admit one full-history request')
    return dict(max_model_len=max_model_len, reported_concurrency=str(ratio),
                conservative_admitted_limit=safe, source_lines=[row[2] for row in matches],
                scope='Startup geometry bound at the full 256K ceiling, not observed concurrent decode')


def kvmem_capacity(geometry, profile):
    kv = profile['kvmem']
    require(geometry['select_budget'] == 65536 and geometry['gen_budget'] == 32768 and
            geometry['historical_tokens'] == 262144 and geometry['cpu_offload_bytes'] == 100_000_000_000
            and geometry['archive_codec'] == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1',
            'Live KVMem geometry differs from the requested 64K+32K/256K FP8 regime')
    # Qualified mamba_cache_mode=none reserves 1 + speculative_blocks per group.
    demand = sum(1 + group['speculative_blocks'] if group['recurrent'] else
                 (65536 + 32768 + group['block_tokens'] - 1) // group['block_tokens']
                 for group in geometry['groups'])
    record_bytes = kv['block_tokens'] * geometry['cpu_archive_bytes_per_token']
    cpu_capacity = 100_000_000_000 // record_bytes * record_bytes
    full_history_credit = (262144 // kv['block_tokens'] + 2) * record_bytes
    request_credit = ((INPUT_TOKENS + OUTPUT_TOKENS + kv['block_tokens'] - 1) // kv['block_tokens'] + 2) * record_bytes
    return dict(gpu_pages_per_lifetime=demand, native_pool_pages=geometry['native_blocks'],
                conservative_gpu_request_limit=(geometry['native_blocks'] - 1) // demand,
                cpu_archive_bytes_per_token=geometry['cpu_archive_bytes_per_token'],
                cpu_capacity_bytes=cpu_capacity, full_256k_history_credit_bytes=full_history_credit,
                cpu_full_history_request_limit=cpu_capacity // full_history_credit,
                this_request_history_credit_bytes=request_credit,
                scope='Separate GPU-page and CPU-history budgets; model state, capture/DMA buffers and process RSS are additional')


def carry_real_native(path, profile, fixture):
    path = path.resolve()
    require(path.name == 'native-c8', 'Carry only a completed real-history native8 stage')
    parent = path.parent
    manifest = read(parent/'manifest.json')
    result = read(path/'results.json')
    require(manifest.get('fixture_kind') == 'real_deepswe_history' and
            read(parent/'profile.json') == profile and read(parent/'fixture.json') == fixture,
            'Carried native input or profile differs')
    require(result.get('status') == 'completed' and result.get('drained') and
            result.get('cleanup_complete') and result.get('fixed_work_output_tokens') == 8*8192 and
            result.get('native_client_total_output_reconciliation', {}).get('passed'),
            'Carried native work or counters incomplete')
    require(hashes(parent/'frozen') == manifest['source_sha256'] and
            hashes(parent/'runtime/KVMem') == manifest['runtime_sha256'],
            'Carried frozen implementation changed')
    require(manifest['native_source'] == native_source_identity(),
            'Carried native implementation differs')
    files = [parent/name for name in ('manifest.json','profile.json','fixture.json')]
    files += [p for p in path.rglob('*') if p.is_file()]
    return dict(path=str(path), sha256={str(p):sha(p) for p in files},
                scope='Completed immutable native stage carried after a KVMem correctness failure; no native rerun or raw-result mutation')


def prepare(args):
    require(not args.output.exists(), 'Use a new output; never reuse an attempt')
    reference = (ROOT / REFERENCE).resolve()
    require(sha(reference / 'manifest.json') == REFERENCE_MANIFEST_SHA, 'Historical r3 manifest changed')
    manifest = read(reference / 'manifest.json')
    source = hashes(reference / 'frozen')
    require(source == manifest['source_sha256'] and
            sha(reference / 'P1.json') == manifest['profile_sha256']['P1'], 'Historical r3 source/profile changed')
    profile = capacity_profile(read(reference / 'P1.json'), args.port)
    model_config = Path(profile['hf_config_path']) / 'config.json'
    config = read(model_config)
    eos = config.get('text_config', config)['eos_token_id']
    require(type(eos) is int and eos >= 0, 'Require one explicit cleanup EOS token')
    history = getattr(args, 'real_deepswe_history', False)
    require(not history or args.native_scheduling == 'async', 'Real history uses the native async deployment baseline')
    work_name = 'deepswe_history_capacity_workload.py' if history else 'capacity_scaling_workload.py'
    work = module('capacity_prepare_workload', work_name)
    fixture = (work.prepare_fixture(args.source_manifest.resolve(), eos_token_id=eos,
               warmup_tokens_path=args.warmup_tokens.resolve()) if history else
               work.prepare_fixture(args.source_manifest.resolve(), context_tokens=INPUT_TOKENS,
                                    output_tokens=OUTPUT_TOKENS, eos_token_id=eos))
    stages = ('native-c8', 'kvmem-c8') if history else STAGES
    carry_path = getattr(args, 'carry_real_native_c8', None)
    carried = carry_real_native(carry_path, profile, fixture) if carry_path is not None else None
    require(not carried or history, 'Native carry requires real DeepSWE inputs')
    if carried:
        stages = ('kvmem-c8',)
    native_identity = native_source_identity()
    memory_budgets = {stage: host_budget(args.host_extra_bytes, args.host_min_free_bytes,
        offered_concurrency=4 if stage.endswith('c4') else 8, native=stage.startswith('native-'),
        fixed_history_limits=[len(s['input_ids'])+OUTPUT_TOKENS for s in fixture['samples']] if history else None)
        for stage in stages}
    output = args.output.resolve()
    require(not output.is_relative_to(reference), 'Do not write into the historical reference')
    output.mkdir(parents=True, exist_ok=False)
    # Client is historical support plus precisely the two new orchestration files.
    shutil.copytree(reference / 'frozen', output / 'frozen', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('benchmark_capacity_scaling.py', 'capacity_scaling_workload.py'):
        shutil.copyfile(ROOT / 'scripts' / name, output / 'frozen/scripts' / name)
    controls = {}
    if history:
        for name in ('deepswe_history_capacity_workload.py', 'benchmark_deepswe_first16.py',
                     'benchmark_deepswe_capacity.py', 'native_agent_launch.py', 'analyze_deepswe_capacity.py'):
            shutil.copyfile(ROOT/'scripts'/name, output/'frozen/scripts'/name)
        primary = module('capacity_primary_build_controls', 'benchmark_deepswe_capacity.py')
        controls = primary.build_controls(SimpleNamespace(sha=sha), 4)
        toolkit = Path('/usr/local/cuda')
        require((toolkit/'bin/nvcc').is_file(), 'Require the already qualified CUDA toolkit')
        controls['environment']['CUDA_HOME'] = str(toolkit)
        controls['cuda_toolkit'] = dict(path=str(toolkit), resolved=str(toolkit.resolve()),
            nvcc_sha256=sha(toolkit/'bin/nvcc'), scope='Same toolkit alias for both engines; startup cache consistency only')
    shutil.copytree(reference / 'frozen', output / 'runtime/KVMem', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    require(hashes(output / 'runtime/KVMem') == source, 'Copied KVMem runtime differs')
    overlay = getattr(args, 'runtime_overlay', None)
    overlay_info = None
    if overlay is not None:
        require(history, 'Runtime repair overlay requires real DeepSWE history')
        names = {'session_control.py','active_residency.py','prompt_checkpoints.py','session_worker.py','async_restore.py',
                 'restore_admission.py','prefill_attention.py'}
        require({p.name for p in overlay.iterdir() if p.is_file()} == names,
                'Repair overlay must contain precisely the seven reviewed restore files')
        overlay_info = dict(path=str(overlay.resolve()), files={})
        for name in sorted(names):
            target = output/'runtime/KVMem/src/vllm_kvmem'/name
            overlay_info['files'][name] = dict(base_sha256=sha(target), repaired_sha256=sha(overlay/name))
            shutil.copyfile(overlay/name, target)
    save(output / 'profile.json', profile)
    save(output / 'fixture.json', fixture)
    save(output / 'manifest.json', dict(kind='capacity_scaling_probe_v1', created_at=time.time(),
        reference=str(reference), reference_manifest_sha256=REFERENCE_MANIFEST_SHA,
        runtime_sha256=hashes(output/'runtime/KVMem'), source_sha256=hashes(output / 'frozen'),
        qualified_runtime_sha256=source, runtime_repair_overlay=overlay_info,
        carried_windows={'native-c8':carried} if carried else {},
        comparison_stages=['native-c8','kvmem-c8'] if history else list(stages),
        profile_sha256=sha(output / 'profile.json'), fixture_sha256=sha(output / 'fixture.json'),
        input_source_manifest=str(args.source_manifest.resolve()), input_source_manifest_sha256=sha(args.source_manifest),
        model_config_sha256={str(model_config): sha(model_config)},
        stages=list(stages), input_tokens=None if history else INPUT_TOKENS, output_tokens=OUTPUT_TOKENS,
        fixture_kind='real_deepswe_history' if history else 'niah_capacity', workload_module=work_name,
        build_controls=controls,
        execution=dict(port=args.port, idle_seconds=args.idle_seconds, native_async=args.native_scheduling == 'async'),
        native_source=native_identity, host_memory_budget=memory_budgets,
        warmup=dict(scope='Separate identical fixed work before each measured cohort; retire, clean up and reconcile counters before formal work',
                    input_tokens=67584, output_tokens=1, query_tokens=128,
                    reason='Exceed K64 to exercise actual MeanK selection and load its CUDA extension in the serving engine'),
        comparison_scope='Native async is a deployment-system baseline when selected; KVMem is synchronous. This is not an isolated plugin-overhead comparison.',
        scope=('Same real DeepSWE history snapshots: prime full histories then resume8 together. Native scheduler owns admission and may preempt; no external native concurrency gate. Fixed8K output, G32K reserve, no task accuracy claim.' if history else
               'One fixed-quota capacity probe, not a repeated final throughput acceptance. 224K input + 8K output; G32K is reserve, not generated length.'),
        gpu_started=False))
    save(output / 'status.json', dict(status='prepared', gpu_started=False, at=time.time()))
    save(output / 'run-command.json', dict(execute_automatically=False,
        argv=[sys.executable, str(output / 'frozen/scripts/benchmark_capacity_scaling.py'),
              '--output', str(output), '--run-prepared']))


def verify(root):
    require(ROOT.resolve() == (root / 'frozen').resolve(), 'Run this artifact frozen entry point')
    manifest = read(root / 'manifest.json')
    require(manifest['kind'] == 'capacity_scaling_probe_v1', 'Wrong prepared experiment kind')
    require(hashes(ROOT) == manifest['source_sha256'] and
            hashes(root / 'runtime/KVMem') == manifest['runtime_sha256'], 'Frozen source/runtime changed')
    require(sha(root / 'profile.json') == manifest['profile_sha256'] and
            sha(root / 'fixture.json') == manifest['fixture_sha256'], 'Frozen profile/fixture changed')
    require(all(sha(Path(path)) == expected for path, expected in manifest['model_config_sha256'].items()),
            'Model configuration changed')
    require(all(sha(Path(path)) == expected for path, expected in manifest['native_source']['files'].items()),
            'Installed native cache capacity implementation changed')
    if manifest.get('build_controls'):
        controls=manifest['build_controls']
        require(all(sha(Path(p))==h for p,h in controls['source_sha256'].items()), 'Build control source changed')
        require(sha(Path(controls['cuda_toolkit']['path'])/'bin/nvcc')==controls['cuda_toolkit']['nvcc_sha256'],
                'CUDA compiler changed')
    for carried in manifest.get('carried_windows', {}).values():
        require(all(sha(Path(p)) == expected for p, expected in carried['sha256'].items()),
                'Carried native measurement changed')
    return manifest


async def run_stage(root, stage, manifest, bench, launch, native_bench, work):
    import httpx
    await bench.q.wait_idle(root, manifest['execution']['idle_seconds'])
    verify(root)
    require(not bench.q.gpu_state()['pids'], 'GPU ownership changed before launch')
    directory = root / stage
    directory.mkdir()
    memory_budget = manifest['host_memory_budget'][stage]
    before_memory = host_memory((os.getpid(),))
    save(directory / 'host-memory-before.json', dict(**before_memory, budget=memory_budget))
    require(before_memory['MemAvailable'] >= memory_budget['required_prelaunch_available_bytes'],
            'Insufficient host MemAvailable for CPU archive + MeanK indices + model/DMA/system headroom')
    profile = read(root / 'profile.json')
    native = stage.startswith('native-')
    history = manifest.get('fixture_kind') == 'real_deepswe_history'
    offered = 4 if stage.endswith('c4') else 8
    build_environment = manifest.get('build_controls', {}).get('environment', {})
    if native:
        plan = launch.build_launch(profile, port=profile['port'], async_scheduling=manifest['execution']['native_async'],
                                   environ={**os.environ, **build_environment})
        plan['environment']['VLLM_LOG_STATS_INTERVAL'] = '1'
        plan['environment_overrides']['VLLM_LOG_STATS_INTERVAL'] = '1'
        save(directory / 'launch.json', {k: v for k, v in plan.items() if k != 'environment'})
        command, environment, cwd = plan['argv'], plan['environment'], directory
    else:
        cwd = root / 'runtime/KVMem'
        environment = {k: v for k, v in os.environ.items() if not k.upper().startswith(('KVMEM_', 'VLLM_'))}
        environment.update(PYTHONPATH=str(cwd / 'src'), PYTHONDONTWRITEBYTECODE='1',
                           TOKENIZERS_PARALLELISM='false', VLLM_LOG_STATS_INTERVAL='1')
        command = [sys.executable, '-m', 'vllm_kvmem', 'serve', '--config', str(root / 'profile.json'),
                   '--run-dir', str(directory / 'server')]
        save(directory / 'launch.json', dict(argv=command, cwd=str(cwd),
             runtime_source_digest=digest(manifest['runtime_sha256'])))
    environment.update(build_environment)
    gpu_rows, native_rows = [], []
    capture = directory / 'server/capture'
    with (directory / 'server.log').open('x') as log:
        server = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        save(root / 'status.json', dict(status='loading', stage=stage, server_pid=server.pid, at=time.time()))
        async def guard():
            while True:
                state = await asyncio.to_thread(bench.q.gpu_state)
                row = dict(at=time.perf_counter(), wall_at=time.time(), **state)
                gpu_rows.append(row)
                with (directory / 'gpu_samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                require(not (set(state['pids']) - bench.q.owned_pids(server)), 'Foreign GPU work appeared')
                require(server.poll() is None, 'Backend exited; preserve server.log')
                memory = host_memory(bench.q.owned_pids(server) | {os.getpid()})
                with (directory / 'host-memory-samples.jsonl').open('a') as stream:
                    stream.write(json.dumps(memory) + '\n')
                require(memory['MemAvailable'] >= memory_budget['minimum_free_bytes'],
                        'Host memory headroom floor reached; stopping owned stage before OOM')
                require(memory['owned_rss_sum_bytes'] <= before_memory['owned_rss_sum_bytes'] +
                        memory_budget['maximum_owned_rss_growth_bytes'], 'Owned process RSS exceeded the planned host budget')
                await asyncio.sleep(1.)

        async def measure():
            async with httpx.AsyncClient(timeout=httpx.Timeout(3600, connect=5), trust_env=False,
                    limits=httpx.Limits(max_connections=32, max_keepalive_connections=32)) as client:
                url = f"http://127.0.0.1:{profile['port']}"
                for _ in range(1800 if history else 900):
                    try:
                        if (await client.get(url + '/health', timeout=2)).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    await asyncio.sleep(1.)
                else:
                    raise TimeoutError('Backend startup timed out')
                if native:
                    geometry = native_capacity((directory / 'server.log').read_text())
                    admitted = offered if history else min(offered, geometry['conservative_admitted_limit'])
                    require(offered != 4 or admitted == 4, 'Native geometry cannot qualify the matched C4 control')
                else:
                    geometry = kvmem_capacity(read(capture / 'cache-geometry.json'), profile)
                    require(geometry['cpu_archive_bytes_per_token'] == ARCHIVE_BYTES_PER_TOKEN and
                            geometry['full_256k_history_credit_bytes'] == memory_budget['full_256k_history_credit_bytes'],
                            'Live CPU archive geometry differs from the stage host-memory reservation')
                    require(geometry['conservative_gpu_request_limit'] >= offered and
                            geometry['cpu_full_history_request_limit'] >= offered,
                            'KVMem geometry cannot safely admit the requested full-history cohort')
                    if history:
                        geometry.pop('this_request_history_credit_bytes')
                    admitted = offered
                save(directory / 'capacity.json', dict(**geometry, offered_concurrency=offered,
                                                       admitted_concurrency=admitted))
                async def snapshot():
                    row = dict(started_at=time.perf_counter())
                    response = await client.get(url + '/metrics', timeout=5)
                    response.raise_for_status()
                    row['values'] = native_bench.metrics(response.text)
                    row['raw_metrics'] = response.text
                    row['finished_at'] = time.perf_counter()
                    native_rows.append(row)
                    with (directory / 'native_samples.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    return row
                async def quiescent(*, stage, not_before):
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        row = await snapshot()
                        if row['started_at'] >= not_before and all(row['values'][key] == 0 for key in
                                ('vllm:num_requests_running', 'vllm:num_requests_waiting')):
                            return dict(stage=stage, native_metrics=row)
                        await asyncio.sleep(.25)
                    raise TimeoutError('Native engine did not drain')
                async def telemetry():
                    while True:
                        try:
                            await snapshot()
                        except (httpx.HTTPError, ValueError) as error:
                            with (directory / 'telemetry-errors.jsonl').open('a') as stream:
                                stream.write(json.dumps(dict(at=time.perf_counter(), error=str(error))) + '\n')
                        await asyncio.sleep(1.)
                monitor = asyncio.create_task(telemetry())
                try:
                    args = SimpleNamespace(offered_concurrency=offered, admitted_concurrency=admitted,
                        output_tokens=OUTPUT_TOKENS, run_tag=stage, capture_directory=capture,
                        native_vllm=native, native_quiescent=quiescent, snapshot=snapshot,
                        reconcile_outputs=reconcile_outputs)
                    fixture = read(root / 'fixture.json')
                    before_warmup = await quiescent(stage='before_warmup', not_before=time.perf_counter())
                    save(root / 'status.json', dict(status='warming', stage=stage, server_pid=server.pid,
                                                   at=time.time()))
                    warmup_begin = time.perf_counter()
                    warmup = await work.warmup(client, dict(profile, native_vllm=native), fixture,
                                               directory / 'warmup', args)
                    warmup_end = time.perf_counter()
                    require(warmup['status'] == 'completed' and warmup['drained'] is True,
                            'Warmup did not finish and drain')
                    warmup_after = await quiescent(stage='after_warmup', not_before=warmup_end)
                    warmup_reconciliation = await reconcile_outputs(snapshot, before_warmup['native_metrics'],
                                                                   warmup['total_stream_output_tokens'])
                    warmup.update(native_client_total_output_reconciliation=warmup_reconciliation,
                                  native_quiescent=warmup_after,
                                  wrapper_seconds=warmup_end-warmup_begin,
                                  gpu=bench.gpu_summary(gpu_rows, warmup_begin, warmup_end))
                    save(directory / 'warmup-results.json', warmup)
                    # Counter publication is caught up and every warmup request has
                    # retired before establishing the measured workload boundary.
                    before = await quiescent(stage='before', not_before=time.perf_counter())
                    save(root / 'status.json', dict(status='running', stage=stage, server_pid=server.pid,
                        offered_concurrency=offered, admitted_concurrency=admitted, at=time.time()))
                    begin = time.perf_counter()
                    result = await work.workload(client, dict(profile, native_vllm=native),
                                               fixture, directory / 'workload', args)
                    end = time.perf_counter()
                    after = await quiescent(stage='after', not_before=end)
                    reconciliation = await reconcile_outputs(snapshot, before['native_metrics'],
                        result['total_stream_output_tokens'], allow_preemptions=native and history)
                    require(result['status'] == 'completed' and result['drained'] is True, 'Incomplete fixed work')
                    if not native:
                        require(result['compression']['passed'] is True,
                                'KVMem quota completed without proved compressed active window and full CPU history')
                        if history:
                            require(result['gpu_decode_concurrency']['passed'] is True,
                                    'No real eight-row GPU decode cohort in the prepared history diagnostic')
                    result.update(capacity=geometry, offered_concurrency=offered, admitted_concurrency=admitted,
                        native_client_total_output_reconciliation=reconciliation,
                        preemptions=reconciliation['preemptions'], native_quiescent=after,
                        warmup_results='warmup-results.json', warmup_excluded_from_fixed_work=True,
                        acceptance_gates=dict(fixed_work_drained=True, zero_preemptions=reconciliation['preemptions']==0,
                            native_preemptions_allowed=native and history,
                            compressed_cpu_authoritative_history=None if native else result['compression']['passed'],
                            target_concurrency_gpu_decode_observed=result['gpu_decode_concurrency']['passed'],
                            interpretation='A finite cold-start workload may finish early streams before all requests reach decode. Missing same-step evidence is not evidence of insufficient capacity.'),
                        gpu=bench.gpu_summary(gpu_rows, begin, end),
                        gpu_fixed_work=bench.gpu_summary(gpu_rows, result['workload_begin'], result['workload_end']),
                        scope='Fixed quota probe. GPU summary spans workload and cleanup; use workload timing for performance rates.')
                    if history:
                        result['gpu_resume'] = bench.gpu_summary(gpu_rows, result['resume_begin'], result['workload_end'])
                        if native:
                            observer=module('history_native_step_observer','analyze_deepswe_capacity.py')
                            separated=[before['native_metrics']]
                            for row in sorted(native_rows,key=lambda r:r['started_at']):
                                if row['started_at']>=separated[-1]['finished_at']:
                                    separated.append(row)
                            result['native_step_request_lower_bound']=observer.native_batch_lower_bound(
                                [dict(at=r['started_at'],raw_metrics=r['raw_metrics']) for r in separated],
                                2*offered,offered,profile['draft_tokens'],True)
                    save(directory / 'results.json', result)
                    return result
                finally:
                    monitor.cancel()
                    await asyncio.gather(monitor, return_exceptions=True)
        try:
            async with asyncio.TaskGroup() as group:
                monitor = group.create_task(guard())
                task = group.create_task(measure())
                result = await task
                monitor.cancel()
            return result
        finally:
            await asyncio.to_thread(bench.q.stop_owned, server)


async def run(args):
    root = args.output.resolve()
    manifest = verify(root)
    require(read(root / 'status.json')['status'] == 'prepared', 'No automatic restart or reuse')
    bench = module('capacity_shared_bench', 'benchmark_timed_agents.py')
    launch = module('capacity_native_launch', 'native_agent_launch.py')
    native_bench = module('capacity_native_metrics', 'benchmark_native_agents.py')
    work = module('capacity_probe_workload', manifest.get('workload_module','capacity_scaling_workload.py'))
    loop, active = asyncio.get_running_loop(), asyncio.current_task()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, active.cancel)
    try:
        with open('/tmp/kvmem-coordination-qualification.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            results = {}
            try:
                for stage in manifest['stages']:
                    results[stage] = await run_stage(root, stage, manifest, bench, launch, native_bench, work)
                    save(root / 'results.json', results)
                save(root / 'status.json', dict(status='completed_pending_review', at=time.time(),
                                              final_performance_acceptance=False))
            except BaseException as error:
                save(root / 'status.json', dict(status='interrupted' if isinstance(error, asyncio.CancelledError)
                     else 'failed', error=repr(error), at=time.time(), no_automatic_restart=True))
                raise
    finally:
        for signum in (signal.SIGTERM, signal.SIGINT):
            loop.remove_signal_handler(signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--native-scheduling', choices=('sync', 'async'))
    parser.add_argument('--real-deepswe-history', action='store_true')
    parser.add_argument('--warmup-tokens', type=Path)
    parser.add_argument('--carry-real-native-c8', type=Path)
    parser.add_argument('--runtime-overlay', type=Path)
    parser.add_argument('--port', type=int, default=18520)
    parser.add_argument('--idle-seconds', type=float, default=15.)
    parser.add_argument('--host-extra-bytes', type=int, default=32 * 1024**3)
    parser.add_argument('--host-min-free-bytes', type=int, default=8 * 1024**3)
    args = parser.parse_args()
    require(1024 <= args.port <= 65535 and math.isfinite(args.idle_seconds) and args.idle_seconds >= 0,
            'Invalid port or idle guard')
    if args.prepare_only:
        require(args.source_manifest is not None and args.native_scheduling is not None,
                'Preparation requires explicit frozen fixture source and native scheduling mode')
        require(not args.real_deepswe_history or args.warmup_tokens is not None,
                'Real history preparation requires the primary initialization token file')
        prepare(args)
    else:
        require(args.source_manifest is None and args.native_scheduling is None
                and not args.real_deepswe_history and args.warmup_tokens is None
                and args.carry_real_native_c8 is None and args.runtime_overlay is None,
                'Execution uses the immutable prepared source and scheduling mode')
        asyncio.run(run(args))


if __name__ == '__main__':
    main()
