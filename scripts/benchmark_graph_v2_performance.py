"""Isolated V2 D single-variable performance pairs on original eight 256K inputs."""
import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

import benchmark_block_graph as bench
import benchmark_graph_v2 as shared
import graph_v2_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'kvmem-graph-v2-performance-3'
CANDIDATES = {
    'raw-k-cache': dict(field='gpu_raw_k_cache_bytes', variants={'raw0': 0, 'raw4g': 4*1024**3}),
    'raw-k-cache-window': dict(field='gpu_raw_k_cache_bytes', variants={'raw0': 0, 'raw10g': 10*1024**3}),
    'cpu-direct-record-seal': dict(field='cpu_direct_record_seal', variants={'seal-copy': False, 'seal-direct': True}),
    'async-action-metadata': dict(field='async_action_metadata', variants={'metadata-sync': False, 'metadata-async': True}),
}
VARIANTS = CANDIDATES['raw-k-cache']['variants']
HELPERS = shared.HELPERS + ('benchmark_graph_v2_performance.py', 'qualify_graph_v2_performance.py')
sha, require, write = protocol.sha, protocol.require, bench.write


def profiles(base, port, candidate='raw-k-cache'):
    require(candidate in CANDIDATES, 'Unknown performance candidate')
    original = protocol.profiles(base, port)['D']
    # Independent screening always starts from the same execution baseline;
    # final quality profiles separately preserve the chosen execution knobs.
    original['kvmem']['gpu_raw_k_cache_bytes'] = 0
    original['kvmem']['cpu_direct_record_seal'] = False
    original['kvmem']['async_action_metadata'] = False
    result = {}
    contract = CANDIDATES[candidate]
    for variant, value in contract['variants'].items():
        result[variant] = deepcopy(original)
        result[variant]['kvmem'][contract['field']] = value
    validate_profiles(result, candidate)
    return result


def validate_profiles(configs, candidate='raw-k-cache'):
    require(candidate in CANDIDATES, 'Unknown performance candidate')
    contract = CANDIDATES[candidate]
    require(set(configs) == set(contract['variants']), 'Both performance variants must be frozen together')
    stripped = []
    for name, profile in configs.items():
        protocol.validate_profile(profile, 'D')
        kv = profile['kvmem']
        expected = contract['variants'][name]
        require(type(kv.get(contract['field'])) is type(expected) and kv[contract['field']] == expected,
                'Performance candidate value differs')
        require(kv.get('async_refresh') and
            kv.get('direct_pinned_reads'), 'Raw-K total budget or async arena contract differs')
        if contract['field'] == 'gpu_raw_k_cache_bytes':
            require(kv.get('cpu_direct_record_seal') is False, 'Raw-K pair must retain the baseline sealing path')
        elif candidate == 'cpu-direct-record-seal':
            require(kv.get('gpu_raw_k_cache_bytes') == 0 and kv.get('async_seal') is True,
                    'Direct sealing pair requires raw-K disabled and async_seal=true')
        else:
            require(kv.get('gpu_raw_k_cache_bytes') == 0 and kv.get('cpu_direct_record_seal') is False,
                    'Action metadata pair must retain raw-K and sealing baselines')
        if candidate != 'async-action-metadata':
            require(kv.get('async_action_metadata') is False, 'Performance pair changed Action metadata policy')
        common = deepcopy(profile)
        common['kvmem'].pop(contract['field'])
        stripped.append(common)
    require(stripped[0] == stripped[1], 'Performance A/B changed more than one profile variable')


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root/'status.json', dict(status='preparing', at=time.time()))
    try:
        base = json.loads(args.profile.read_text())
        identity = shared.model_identity(base)
        candidate = getattr(args, 'candidate', None) or 'raw-k-cache'
        contract = CANDIDATES[candidate]
        variants = contract['variants']
        configurations = profiles(base, args.port, candidate)
        from vllm_kvmem.serve import build_launch
        for variant, profile in configurations.items():
            build_launch(profile, ROOT, root/variant/'server')
            write(root/(variant+'.json'), profile)
        # The shared source freezer hashes this historical filename. It is a
        # copy of the baseline D profile, not an additional A arm or model run.
        baseline = configurations[next(iter(variants))]
        write(root/'niah-A.json', baseline)
        fixture = shared.freeze_niah(SimpleNamespace(reference=args.reference, scope='smoke8'),
                                      root, baseline)
        write(root/'fixture.json', fixture)
        hashes = None
        jobs = []
        for variant in variants:
            target = root/variant
            bench.prepare(SimpleNamespace(output=target, fixture=root/'fixture.json', arm='D',
                profile=root/(variant+'.json'), qualifier=ROOT/'scripts/qualify_graph_v2_performance.py',
                synchronous=False, port=args.port, client_timeout_seconds=7200))
            for name in HELPERS:
                shutil.copyfile(ROOT/'scripts'/name, target/'runtime/scripts'/name)
            current = {str(path.relative_to(target/'runtime')):sha(path)
                       for path in (target/'runtime').rglob('*') if path.is_file()}
            require(any(name.startswith('src/vllm_kvmem/_native') and name.endswith('.so') for name in current),
                    'Frozen runtime lacks the current native extension')
            require(hashes is None or current == hashes, 'Source changed between performance variants')
            hashes = current
            manifest = json.loads((target/'manifest.json').read_text())
            manifest.update(runtime_sha256=current, performance_variant=variant, candidate=candidate, schema=SCHEMA)
            write(target/'manifest.json', manifest)
            jobs.append(dict(id=variant, manifest_sha256=sha(target/'manifest.json')))
        frozen = root/'frozen'
        frozen.mkdir()
        for name in HELPERS:
            shutil.copyfile(ROOT/'scripts'/name, frozen/name)
            require(sha(frozen/name) == hashes['scripts/'+name], 'Controller changed while freezing: '+name)
        patch = subprocess.check_output(['git', 'diff', 'HEAD', '--', 'src', 'csrc', 'third_party', 'scripts'], cwd=ROOT)
        (root/'source.diff').write_bytes(patch)
        worktree = subprocess.check_output(['git', 'status', '--porcelain', '--', 'src', 'csrc', 'third_party', 'scripts'],
                                          cwd=ROOT, text=True)
        maximum = max(row['served_input_tokens'] for row in fixture['rows'])+8192+16384
        credit = ((maximum+31)//32)*32*17*4*256*2
        require(8*credit <= fixture['cpu_payload_credits_bytes'], 'Cannot admit eight intact histories')
        write(root/'manifest.json', dict(schema=SCHEMA, jobs=jobs, created_at=time.time(),
            source_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            source_dirty=bool(worktree.strip()), source_worktree_status=worktree,
            source_diff_sha256=sha(root/'source.diff'), runtime_sha256=hashes,
            frozen_sha256={path.name:sha(path) for path in frozen.iterdir()},
            fixture_sha256=sha(root/'fixture.json'),
            input_sha256={**{str(path.relative_to(root)):sha(path) for path in (root/'dataset').rglob('*') if path.is_file()},
                          'niah-A.json':sha(root/'niah-A.json')},
            model_identity=identity, candidate=candidate, changed_profile_field='kvmem.'+contract['field'],
            variants=variants, per_async_arena_caps={name:configurations[name]['kvmem']['gpu_raw_k_cache_bytes']//2
                                                    for name in variants},
            start_cpu_available_bytes=8*credit+24*1024**3, minimum_cpu_available_bytes=8*1024**3,
            minimum_gpu_free_mib=4096,
            protocol='Cold cohort of eight original 256K inputs, V2 D, up to eight owners; '
                     'independent fresh server per variant; no request warmup, replacement workload, or retries.'))
        write(root/'status.json', dict(status='prepared', at=time.time()))
        print(json.dumps(dict(prepared=str(root), gpu_started=False, candidate=candidate, variants=variants)), flush=True)
    except BaseException as error:
        write(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
        raise


def validate(root):
    manifest = json.loads((root/'manifest.json').read_text())
    candidate = manifest.get('candidate', 'raw-k-cache')
    require(manifest['schema'] == SCHEMA and candidate in CANDIDATES and
            manifest['variants'] == CANDIDATES[candidate]['variants'], 'Unknown performance contract')
    variants = CANDIDATES[candidate]['variants']
    require(sha(root/'source.diff') == manifest['source_diff_sha256'] and
            sha(root/'fixture.json') == manifest['fixture_sha256'], 'Frozen source/fixture changed')
    for name, expected in manifest['frozen_sha256'].items():
        require(sha(root/'frozen'/name) == expected, 'Frozen helper changed: '+name)
    for name, expected in manifest['input_sha256'].items():
        require(sha(root/name) == expected, 'Frozen input changed: '+name)
    configs = {}
    require({job['id'] for job in manifest['jobs']} == set(variants), 'Performance job set changed')
    for job in manifest['jobs']:
        target = root/job['id']
        require(sha(target/'manifest.json') == job['manifest_sha256'], 'Job manifest changed')
        child = json.loads((target/'manifest.json').read_text())
        require(child['runtime_sha256'] == manifest['runtime_sha256'], 'A/B runtimes differ')
        require(sha(target/'profile.json') == child['profile_sha256'] and
            sha(target/'fixture.json') == child['fixture_sha256'] == manifest['fixture_sha256'],
            'Job profile/input changed')
        for name, expected in child['runtime_sha256'].items():
            require(sha(target/'runtime'/name) == expected, 'Frozen runtime changed: '+name)
        configs[job['id']] = json.loads((target/'profile.json').read_text())
    validate_profiles(configs, candidate)
    for group in ('config_sha256', 'tokenizer_sha256'):
        for name, expected in manifest['model_identity'][group].items():
            require(sha(name) == expected, 'Model/tokenizer JSON identity changed')
    for name, expected in manifest['model_identity']['weight_stat_identity'].items():
        stat = Path(name).stat()
        require(dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns) == expected, 'Model weights identity changed')
    return manifest


def sample_resources():
    fields = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.free,memory.used,utilization.gpu,power.draw',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    free, used, utilization, power = [float(value.strip()) for value in fields]
    return dict(at=time.monotonic(), wall_at=time.time(), cpu_available_bytes=shared.cpu_available(),
        gpu_free_mib=free, gpu_used_mib=used, gpu_utilization_percent=utilization, gpu_power_w=power,
        source='nvidia-smi NVML device 0; one-second samples, not a kernel timeline')


def resource_summary(rows):
    require(rows, 'No NVML samples during qualification')
    return dict(samples=len(rows), peak_gpu_used_mib=max(row['gpu_used_mib'] for row in rows),
        minimum_gpu_free_mib=min(row['gpu_free_mib'] for row in rows),
        mean_gpu_utilization_percent=statistics.fmean(row['gpu_utilization_percent'] for row in rows),
        mean_power_w=statistics.fmean(row['gpu_power_w'] for row in rows),
        minimum_cpu_available_bytes=min(row['cpu_available_bytes'] for row in rows),
        scope='Periodic NVML observations within the qualifier interval; no inferred idle time.')


def comparison(root, candidate='raw-k-cache'):
    variants = CANDIDATES[candidate]['variants']
    data = {name:json.loads((root/name/'qualification/performance.json').read_text()) for name in variants}
    for name, metrics in data.items():
        require(metrics.get('schema') == SCHEMA and metrics.get('candidate') == candidate,
                'Performance result schema/candidate differs')
        metrics['nvml'] = json.loads((root/name/'nvml-summary.json').read_text())
    baseline, changed = variants
    a,b = data[baseline], data[changed]
    require(a['cases'].keys() == b['cases'].keys() and len(a['cases']) == 8, 'A/B case identities differ')
    for name in a['cases']:
        require(all(a['cases'][name][field] == b['cases'][name][field] for field in
            ('input_sha256', 'seed', 'input_tokens')), 'A/B input or sampling seed differs')
    identical_selections = all(a['cases'][name]['selection_sha256'] == b['cases'][name]['selection_sha256'] for name in a['cases'])
    identical_outputs = all(a['cases'][name]['output_ids_sha256'] == b['cases'][name]['output_ids_sha256'] for name in a['cases'])
    observed = {'raw-k-cache': b['raw_cache_exercised'],
                'raw-k-cache-window': b['raw_cache_exercised'],
                'cpu-direct-record-seal': b['seal']['direct_path_observed'],
                'async-action-metadata': b['action_metadata']['async_path_observed']}[candidate]
    conditions = dict(identical_selection_sequences=identical_selections,
        identical_accepted_outputs=identical_outputs, candidate_path_observed=observed,
        native_eight_owner_batches_observed=all(
            item['actual_concurrency'].get('executed_request_peak') == 8 for item in (a, b)))
    if candidate == 'cpu-direct-record-seal':
        conditions['complete_seal_measurements'] = all(item['seal']['measurement_complete'] for item in (a, b))
    blockers = [name for name, passed in conditions.items() if not passed]
    return dict(schema=SCHEMA, candidate=candidate, baseline=baseline, changed=changed, variants=data,
        raw_wall_ratio_baseline_over_candidate=a['wall_seconds']/b['wall_seconds'],
        useful_input_rate_ratio_candidate_over_baseline=(
            b['useful_input_tokens_per_wall_second']/a['useful_input_tokens_per_wall_second']),
        accepted_output_rate_ratio_candidate_over_baseline=(
            b['accepted_output_tokens_per_wall_second']/a['accepted_output_tokens_per_wall_second']),
        selection_sequences_identical=identical_selections, accepted_outputs_identical=identical_outputs,
        candidate_path_observed=observed, gpu_idle_inferred=False,
        candidate_screen_status='inconclusive' if blockers else 'comparable_cold_smoke',
        comparison_conditions=conditions, inconclusive_reasons=blockers, performance_gain_established=False,
        interpretation='One cold eight-case run per variant, without sustained replacement, request warmup, '
            'a 300-second measured window, or paired repetitions. Raw ratios are descriptive. '
            'Changed output length/EOS or selections change the work and prevent independent speed attribution; '
            'an unexercised candidate cannot qualify. Even comparable cold smoke is not the performance admission gate.')


def run(args):
    root = args.output.resolve()
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate(root)
        candidate = manifest.get('candidate', 'raw-k-cache')
        require(getattr(args, 'candidate', None) in (None, candidate), 'Requested candidate differs from frozen campaign')
        variants = manifest['variants']
        require(Path(__file__).resolve() == root/'frozen/benchmark_graph_v2_performance.py', 'Use the frozen performance controller')
        require(json.loads((root/'status.json').read_text())['status'] in ('prepared', 'partial'),
                'Failed/interrupted/completed attempts cannot be retried; prepare a new directory')
        pending = []
        selected = args.variants or list(variants)
        require(set(selected) <= set(variants) and len(set(selected)) == len(selected), 'Invalid or duplicate frozen variants')
        for name in selected:
            status = json.loads((root/name/'status.json').read_text())['status']
            require(status in ('prepared', 'passed'), 'Failed/interrupted variant retained: '+name)
            if status == 'prepared': pending.append(name)
        require(pending, 'Selected variants already passed; no silent rerun')
        write(root/'status.json', dict(status='running', variants=pending, at=time.time()))
        try:
            with bench.cleanup_on_termination() as begin_cleanup:
                for name in pending:
                    require(shared.cpu_available() >= manifest['start_cpu_available_bytes'], 'Insufficient host headroom')
                    target = root/name
                    process = None
                    samples = []
                    try:
                        with (target/'controller.log').open('x') as log, (target/'resources.jsonl').open('x') as output:
                            process = subprocess.Popen([sys.executable, '-B', str(target/'runtime/scripts/benchmark_graph_v2_performance.py'),
                                'arm-run', '--output', str(target)], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
                            while process.poll() is None:
                                sample = sample_resources()
                                samples.append(sample)
                                output.write(json.dumps(sample)+'\n'); output.flush()
                                require(sample['cpu_available_bytes'] >= manifest['minimum_cpu_available_bytes'] and
                                    sample['gpu_free_mib'] >= manifest['minimum_gpu_free_mib'], 'Runtime memory guard reached')
                                time.sleep(1)
                        require(process.returncode == 0, 'Variant failed; retained evidence: '+name)
                    finally:
                        if process is not None:
                            require(not bench.stop_owned_groups((process,)), 'Owned controller did not stop')
                    metrics = json.loads((target/'qualification/performance.json').read_text())
                    selected = [row for row in samples if metrics['measurement_started_at'] <= row['at'] <= metrics['measurement_finished_at']]
                    write(target/'nvml-summary.json', resource_summary(selected))
                begin_cleanup()
            complete = all(json.loads((root/name/'status.json').read_text())['status'] == 'passed' for name in variants)
            report = None
            if complete:
                report = comparison(root, candidate)
                write(root/'comparison.json', report)
            write(root/'status.json', dict(status='completed' if complete else 'partial', at=time.time(),
                candidate_screen_status=None if report is None else report['candidate_screen_status']))
        except BaseException as error:
            write(root/'status.json', dict(status='failed', error=repr(error), at=time.time()))
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'validate', 'run', 'arm-run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT/'artifacts/graph-overhead-20261018/gpu-r2/after/profile.json')
    parser.add_argument('--reference', type=Path,
        default=Path('/data/chaidi/qw3-experiments/cacheblend_20260925/official_template_20260927'))
    parser.add_argument('--port', type=int, default=18678)
    parser.add_argument('--candidate', choices=tuple(CANDIDATES),
                        help='Preparation defaults to raw-k-cache; execution reads the frozen candidate')
    parser.add_argument('--variants', nargs='+', choices=tuple(name for item in CANDIDATES.values() for name in item['variants']))
    args = parser.parse_args(argv)
    require(args.variants is None or len(set(args.variants)) == len(args.variants), 'Duplicate variants are not allowed')
    if args.action == 'prepare': prepare(args)
    elif args.action == 'run': run(args)
    elif args.action == 'arm-run':
        require(Path(__file__).resolve() == args.output.resolve()/'runtime/scripts/benchmark_graph_v2_performance.py',
                'Use the frozen arm runner')
        bench.run(args)
    else:
        manifest = validate(args.output.resolve())
        print(json.dumps(dict(status='validated', variants=manifest['variants'])))


if __name__ == '__main__':
    main()
