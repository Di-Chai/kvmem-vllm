"""Frozen ABBA comparison for bounded same-owner prefill publication overlap.

The existing matched-prefill controller retains ownership, input/policy audits,
CUDA fences and cleanup. This driver changes exactly one profile flag and adds
cross-arm semantic checks, explicit process RSS samples, and pipeline evidence.
No inference is started by prepare or analyze.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

FLAG = 'prefill_publication_pipeline'
ORDER = [('off-a', False), ('on-a', True), ('on-b', True), ('off-b', False)]
# Copied historical utilities were never installed in the live repository.
# This is deliberately not a blanket fallback for deleted production code.
FROZEN_UTILITIES = frozenset(('scripts/grade_reference_campaign.py', 'scripts/reference_service.py'))
PROCESS_MEMORY_SCOPE = ('Observed server process-tree RSS; shared pages may be counted more than once. '
                        'This is not physical memory/PSS; inaccessible or exited processes are skipped.')


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def runtime_origins(repo, reference, names):
    expected = read(reference / 'runtime-proof.json')['files']
    origins, inherited = {}, {}
    for name in sorted(names):
        source = repo / name
        if not source.is_file():
            require(name in FROZEN_UTILITIES, 'Runtime source dependency is missing: ' + name)
            source = reference / 'runtime' / name
            require(source.is_file() and name in expected and sha(source) == expected[name],
                    'Historical utility is missing or differs from its recorded hash: ' + name)
            inherited[name] = dict(source=str(source.resolve()), sha256=expected[name])
        origins[name] = source
    return origins, inherited


def prepare(root, repo, repeats, smoke=False):
    reference = repo / 'artifacts/native-qk-budget32k-20261021-r1/reuse'
    profile = read(reference / 'profile.json')
    require(profile['kvmem']['action_native_scores'] == 'reuse', 'Expected native QK reuse')
    require(profile['kvmem']['select_budget'] == 32768 and profile['kvmem']['gen_budget'] == 16384,
            'Expected K32K G16K')
    require(profile['prefill_chunk_tokens'] == 2048, 'Expected 2K chunks')
    require(repeats >= 1, 'At least one measured repetition is required')
    names = set(read(reference / 'runtime-proof.json')['files'])
    names.update(str(p.relative_to(repo)) for p in (repo / 'src/vllm_kvmem').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and
                 p.suffix in ('.py', '.json', '.cuh', '.so'))
    names.update(('scripts/benchmark_prefill_publication_pipeline.py',
                  'scripts/analyze_pressure_profile.py'))
    origins, inherited = runtime_origins(repo, reference, names)
    hashes = {name: sha(path) for name, path in origins.items()}
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    root.mkdir(parents=True, exist_ok=False)
    order = [('smoke-on', True)] if smoke else ORDER
    if smoke:
        require(repeats == 1, 'Smoke runs exactly one full input after warmup')
    for ident, enabled in order:
        target = root / ident
        target.mkdir()
        runtime = target / 'runtime'
        for name, origin in sorted(origins.items()):
            destination = runtime / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, destination)
            require(sha(destination) == hashes[name], 'Source changed while freezing: ' + name)
        save(target / 'runtime-proof.json', dict(files=hashes, source=str(repo), inherited_utilities=inherited, head=head, dirty=True))
        arm = deepcopy(profile)
        arm['kvmem'][FLAG] = enabled
        save(target / 'profile.json', arm)
        cases = []
        for case_id, warmup in [('warmup-32k', True), *[
                ('256k-repeat-' + str(i + 1), False) for i in range(repeats)]]:
            directory = target / 'requests' / case_id
            directory.mkdir(parents=True)
            origin = reference / 'requests' / ('warmup-32k' if warmup else '256k-repeat-1') / 'case.json'
            save(directory / 'case.json', read(origin))
            cases.append(dict(id=case_id, warmup=warmup, case_sha256=sha(directory / 'case.json')))
        save(target / 'suite.json', dict(cases=cases, prefix_caching=False, serial=True))
        save(target / 'manifest.json', dict(profile_sha256=sha(target / 'profile.json'),
            scope='Single active request, full original 262045-token input, K32768 G16384 chunk2048. '
                  'Native QK reuse and four existing optimizations enabled; only bounded prefill '
                  'publication pipeline differs. One 32K warmup per service.'))
    require(all(sha(p) == hashes[name] for name, p in origins.items()), 'Source changed during freeze')
    shutil.copy2(Path(__file__).resolve(), root / 'driver.py')
    save(root / 'plan.json', dict(smoke=smoke, order=[dict(id=name, enabled=enabled) for name, enabled in order],
        only_profile_difference='kvmem.' + FLAG, input_tokens=262045, select_budget=32768,
        gen_budget=16384, prefill_chunk=2048, native_scores='reuse',
        measured_repetitions_per_service=repeats, warmup_tokens=32768,
        runtime_sha256=digest(hashes), driver_sha256=sha(root / 'driver.py'),
        inherited_utilities=inherited,
        production_defaults_changed=False, dsh_started=False))
    print(root)


def owned_process_memory(server_pid):
    """Sum owned process-tree RSS, not physical memory (shared pages may repeat)."""
    pending, visited, entries = [server_pid], set(), []
    while pending:
        pid = pending.pop()
        if pid in visited:
            continue
        visited.add(pid)
        base = Path('/proc') / str(pid)
        try:
            fields = {}
            for line in (base / 'status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value, unit = line.split()
                    require(unit == 'kB', 'Unexpected /proc memory unit')
                    fields[key.rstrip(':') + '_bytes'] = int(value) * 1024
            entries.append(dict(pid=pid, **fields))
            pending.extend(int(x) for x in (base / 'task' / str(pid) / 'children').read_text().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return dict(processes=entries, rss_sum_bytes=sum(p.get('VmRSS_bytes', 0) for p in entries))


def sampled_process_memory(resources):
    """Report only observed RSS samples; old GPU-only logs remain unknown."""
    observed = [row['rss_sum_bytes'] for row in resources
                if type(row.get('rss_sum_bytes')) is int and row['rss_sum_bytes'] >= 0
                and any(type(process.get('VmRSS_bytes')) is int
                        for process in row.get('processes', ()))]
    return dict(max_sampled_process_tree_rss_bytes=max(observed, default=None),
                process_tree_rss_samples=len(observed),
                process_tree_rss_missing_samples=len(resources)-len(observed),
                process_tree_memory_scope=PROCESS_MEMORY_SCOPE)


def run(root):
    plan = read(root / 'plan.json')
    require(sha(Path(__file__)) == plan['driver_sha256'], 'Use the frozen driver')
    for item in plan['order']:
        target = root / item['id']
        require(not (target / 'controller.log').exists(), 'Refusing to rerun/overwrite an arm')
        command = [sys.executable, '-B', str(target / 'runtime/scripts/benchmark_matched_prefill.py'),
                   'run', '--root', str(target)]
        process = None
        try:
            with (target / 'controller.log').open('x') as log, \
                    (target / 'process-resources.jsonl').open('x') as telemetry:
                process = subprocess.Popen(command, cwd=target, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                save(target / 'controller-process.json', dict(pid=process.pid, command=command, at=time.time()))
                save(root / 'status.json', dict(status='running', arm=item['id'], pid=process.pid, at=time.time()))
                while process.poll() is None:
                    server = target / 'run/server-process.json'
                    if server.exists():
                        sample = dict(at=time.time(), **owned_process_memory(read(server)['pid']))
                        telemetry.write(json.dumps(sample) + '\n')
                        telemetry.flush()
                    time.sleep(1)
            require(process.returncode == 0, 'Controller failed: ' + item['id'])
            require(read(target / 'status.json')['status'] == 'completed', 'Incomplete arm')
            require(read(target / 'cleanup.json') == dict(remaining_owned_groups={}, gpu_processes=[]),
                    'Owned service cleanup incomplete')
        except BaseException as error:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=60)
            save(root / 'status.json', dict(status='failed', arm=item['id'], error=repr(error), at=time.time()))
            raise
    save(root / 'status.json', dict(status='completed', at=time.time()))
    analyze(root)


def parse_metrics(path):
    result = {}
    for line in read(path)['metrics']:
        if line.startswith('#'):
            continue
        key, value = line.rsplit(' ', 1)
        result[key] = float(value)
    return result


def mtp_delta(before, after):
    def delta(name):
        prefix = 'vllm:' + name + '{'
        return sum(v - before.get(k, 0) for k, v in after.items() if k.startswith(prefix))
    drafts = delta('spec_decode_num_drafts_total')
    accepted = delta('spec_decode_num_accepted_tokens_total')
    proposed = delta('spec_decode_num_draft_tokens_total')
    return dict(drafts=drafts, accepted_draft_tokens=accepted, proposed_tokens=proposed,
                mean_acceptance_length=(1 + accepted / drafts) if drafts else None)


def prefill_groups(steps):
    groups, current, was_prefill = [], [], False
    for step in steps:
        if step['prefill_tokens']:
            if not was_prefill and current:
                groups.append(current)
                current = []
            current.append(step)
        was_prefill = bool(step['prefill_tokens'])
    if current:
        groups.append(current)
    return groups


def timing(group):
    host = sum(s['host_submit_ms'] for s in group) / 1000
    span = group[-1]['submitted_at'] - group[0]['started_at']
    return dict(steps=len(group), tokens=sum(s['prefill_tokens'] for s in group),
        span_seconds=span, host_submit_seconds=host, host_between_submissions_seconds=span-host,
        model_cuda_seconds=sum(s['model_ms'] for s in group) / 1000,
        maintenance_cuda_seconds=sum(s['maintenance_ms'] for s in group) / 1000)


def semantic_signature(evidence):
    return [{k: event[k] for k in ('selection', 'history_begin', 'selected_tokens',
                                  'selected_history_ranges', 'mandatory_fit')}
            for event in evidence['refresh']]


def audit_pipeline(progress, enabled):
    require(progress.get('prefill_publication_pipeline') is enabled,
            'Actual pipeline enablement differs from profile')
    stats = progress.get('prefill_pipeline_stats', {})
    if enabled:
        # Completion while the next forward runs is a successful overlap:
        # no pending packet need remain by that forward's final submit.
        require(stats.get('eligible_packets', 0) > 0 and stats.get('advance_checks_with_pending', 0) > 0,
                'Pipeline enabled but no forward planning with pending publication was recorded')
        require(stats.get('max_pending_per_owner', 0) <= 2 and stats.get('max_bulk_pending', 0) <= 2,
                'Publication backlog exceeded the two-slot bound')
    require(progress.get('pending_downloads') == 0 and not progress.get('accepted_history'),
            'Owner or download remains after release')
    return {k: v for k, v in progress.items() if 'pipeline' in k}


def analyze(root):
    plan = read(root / 'plan.json')
    first = root / plan['order'][0]['id']
    sys.path.insert(0, str(first / 'runtime/scripts'))
    from analyze_pressure_profile import refresh_intervals
    from qualify_block_graph import native_matches
    canonical = read(first / 'profile.json')
    canonical['kvmem'].pop(FLAG)
    proof = read(first / 'runtime-proof.json')['files']
    require(digest(proof) == plan['runtime_sha256'], 'Runtime proof changed')
    result = dict(scope='Single request full256K K32K G16K chunk2K native QK reuse; ABBA pipeline off/on. '
        'Throughput is original input tokens / TTFT, excludes cold model loading and subsequent decode.',
        timing_caveat='Host and CUDA intervals overlap; submission gaps are not proof of GPU idle.',
        memory_caveat='Process-tree RSS can double-count shared pages; GPU memory is nvidia-smi device usage.',
        arms={}, mode_summary={}, semantic_checks=[])
    samples, expected_case, expected_semantics = [], None, None
    for item in plan['order']:
        arm = root / item['id']
        profile = read(arm / 'profile.json')
        require(profile['kvmem'].pop(FLAG) is item['enabled'] and profile == canonical,
                'Profiles differ beyond pipeline flag')
        require(read(arm / 'runtime-proof.json')['files'] == proof, 'Runtime differs between arms')
        require(all(sha(arm / 'runtime' / name) == value for name, value in proof.items()),
                'Frozen runtime changed after execution')
        status = read(arm / 'status.json')
        require(status['status'] == 'completed' and status['protocol_passed'], 'Arm did not pass controller')
        require(read(arm / 'cleanup.json') == dict(remaining_owned_groups={}, gpu_processes=[]),
                'Incomplete cleanup')
        capture = arm / 'run/server/capture'
        groups = prefill_groups(rows(capture / 'model_step_latency.jsonl'))
        cases = read(arm / 'suite.json')['cases']
        require(len(groups) == len(cases), 'Cannot align prefill groups to requests')
        refresh = refresh_intervals(rows(capture / 'async_refresh.jsonl'), rows(capture / 'bounded_refresh.jsonl'))
        cpu = rows(arm / 'process-resources.jsonl')
        measured = []
        for group, case in zip(groups, cases):
            if case['warmup']:
                continue
            directory = arm / 'requests' / case['id']
            sample = read(directory / 'result.json')
            response = read(directory / 'response.json')
            definition = read(directory / 'case.json')
            evidence = read(directory / 'evidence.json')
            native = read(directory / 'native-stats.json')
            require(sample['input_tokens'] == plan['input_tokens'] and sample['token_accounting_passed']
                    and sample['finish_reason'] == 'stop', 'Incomplete original input/output')
            require(sum(s['prefill_tokens'] for s in group) == plan['input_tokens'], 'Prefill accounting differs')
            require(native['delta'].get('reuse:prefill', 0) > 0 and
                    not any(v for k, v in native['delta'].items() if k.startswith('fallback:')),
                    'Native QK reuse fallback in measured request')
            if expected_case is None:
                expected_case = definition
                expected_semantics = semantic_signature(evidence)
            require(definition == expected_case, 'Inputs or sampling differ')
            signature = semantic_signature(evidence)
            semantic_ok = signature == expected_semantics
            events = [x for x in refresh if native_matches(response, x)]
            before = read(directory / 'initial-metrics.json')
            after = read(directory / 'final-metrics.json')
            resources = rows(directory / 'resources.jsonl')
            memory = [r['rss_sum_bytes'] for r in cpu if before['at'] <= r['at'] <= after['at']]
            output = dict(arm=item['id'], enabled=item['enabled'], id=case['id'], **sample,
                output_sha256=digest(response['output_ids']), selections_identical=semantic_ok,
                selection_sha256=digest(signature), prefill_timing=timing(group),
                pressure_wait_seconds=sum(x['seconds'] for x in events if x['phase'] == 'pressure'),
                query_wait_seconds=sum(x['seconds'] for x in events if x['phase'] == 'query'),
                mtp=mtp_delta(parse_metrics(directory / 'initial-metrics.json'),
                              parse_metrics(directory / 'final-metrics.json')),
                native_reuse_counts=native['delta'],
                peak_gpu_memory_mib=max(float(r['gpu'].split(',')[0]) for r in resources),
                peak_process_tree_rss_bytes=max(memory) if memory else None,
                attention_samples=evidence['action'][0]['samples'],
                prefill_attention_samples=evidence['action'][0]['prefill_samples'])
            samples.append(output)
            measured.append(output)
            result['semantic_checks'].append(dict(arm=item['id'], id=case['id'], selections_identical=semantic_ok))
        progress = read(capture / 'progress.json')
        pipeline = audit_pipeline(progress, item['enabled'])
        result['arms'][item['id']] = dict(enabled=item['enabled'], measured=measured,
            cold_prepare_seconds=status['cold_prepare_seconds'], pipeline=pipeline,
            final_progress=progress, native_stats=read(capture / 'native_attention_stats.json'))
    for enabled, name in ((False, 'off'), (True, 'on')):
        selected = [sample for sample in samples if sample['enabled'] is enabled]
        if not selected:
            continue
        result['mode_summary'][name] = dict(samples=len(selected),
            mean_ttft_seconds=statistics.mean(s['ttft_seconds'] for s in selected),
            pooled_prefill_tokens_per_second=sum(s['input_tokens'] for s in selected) /
                                            sum(s['ttft_seconds'] for s in selected),
            ttft_seconds=[s['ttft_seconds'] for s in selected],
            mean_host_gap_seconds=statistics.mean(s['prefill_timing']['host_between_submissions_seconds'] for s in selected),
            mean_model_cuda_seconds=statistics.mean(s['prefill_timing']['model_cuda_seconds'] for s in selected))
    if not plan.get('smoke'):
        off, on = (result['mode_summary'][mode] for mode in ('off', 'on'))
        result.update(throughput_change_percent=100*(on['pooled_prefill_tokens_per_second']/off['pooled_prefill_tokens_per_second']-1),
            ttft_change_seconds=on['mean_ttft_seconds']-off['mean_ttft_seconds'])
    result.update(
        all_selections_identical=all(s['selections_identical'] for s in samples),
        all_outputs_identical=len({s['output_sha256'] for s in samples}) == 1,
        all_mtp_identical=len({digest(s['mtp']) for s in samples}) == 1,
        all_attention_samples_identical=len({digest([s['attention_samples'], s['prefill_attention_samples']]) for s in samples}) == 1)
    save(root / 'comparison.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('arms', 'semantic_checks')}, indent=2))
    require(result['all_selections_identical'], 'Selection changed; do not report semantic-preserving acceleration')
    require(result['all_attention_samples_identical'], 'Sampling changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'analyze'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--smoke', action='store_true', help='One enabled service, 32K warmup plus one 256K case')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'prepare':
        prepare(root, args.repo.resolve(), args.repeats, args.smoke)
    elif args.command == 'run':
        def terminate(signum, frame):
            raise KeyboardInterrupt('Benchmark driver terminated')
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            run(root)
        finally:
            signal.signal(signal.SIGTERM, previous)
    else:
        analyze(root)


if __name__ == '__main__':
    main()
