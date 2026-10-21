"""Freeze paired current-code prefill variants and compare complete owned runs.

Prepare and compare are CPU-only. Run delegates one frozen arm to the existing
matched-prefill controller, including its GPU ownership lock and cleanup.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

# A frozen driver imports the helpers from its own, hash-checked runtime.
_frozen_scripts = Path(__file__).resolve().parent / 'reference/runtime/scripts'
if _frozen_scripts.is_dir():
    sys.path.insert(0, str(_frozen_scripts))

import benchmark_prefill_publication_pipeline as pipeline
from benchmark_prefill_publication_pipeline import read, save, sha, digest, require, rows


def merge(base, overrides):
    """JSON object merge; leaves the source profile untouched."""
    require(isinstance(base, dict) and isinstance(overrides, dict), 'Overrides must be an object')
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            require(key not in result or isinstance(result[key], dict), 'Override changes object type: ' + key)
            result[key] = merge(result.get(key, {}), value)
        else:
            require(not isinstance(result.get(key), dict), 'Override replaces an object: ' + key)
            result[key] = deepcopy(value)
    return result


def differences(before, after, prefix=''):
    """Lossless leaf differences, distinguishing absent fields from JSON null."""
    result = []
    for key in sorted(before.keys() | after.keys()):
        path = prefix + key
        if key in before and key in after and isinstance(before[key], dict) and isinstance(after[key], dict):
            result.extend(differences(before[key], after[key], path + '.'))
        elif key not in before or key not in after or before[key] != after[key]:
            result.append(dict(path=path, before_present=key in before, after_present=key in after,
                               before=before.get(key), after=after.get(key)))
    return result


def validate_profile(profile):
    from benchmark_matched_prefill import validate_diagnostic_profile
    validate_diagnostic_profile(profile)
    kv = profile['kvmem']
    require(kv['select_budget'] == 32768 and kv['action_native_scores'] == 'reuse'
            and kv['prefill_publication_pipeline'] is True,
            'Require K32, native reuse and publication pipeline')
    require(type(kv.get('prefill_native_plan_arm', False)) is bool, 'Native prefill plan arm must be boolean')
    if 'assembly_chunk_blocks' in kv:
        value = kv['assembly_chunk_blocks']
        require(type(value) is int and 1 <= value <= 256, 'Invalid assembly chunk capacity')


def checked_suite(baseline):
    suite = read(baseline / 'suite.json')
    require(suite.get('serial') is True and suite.get('prefix_caching') is False,
            'Require serial independent owners without prefix caching')
    require(len(suite['cases']) == 5 and sum(not x['warmup'] for x in suite['cases']) == 4,
            'Expected one warmup and four formal cases')
    counts = []
    seen = set()
    for item in suite['cases']:
        ident = item['id']
        require(ident not in seen and Path(ident).name == ident and ident not in ('.', '..'), 'Invalid case ID')
        seen.add(ident)
        path = baseline / 'requests' / ident / 'case.json'
        require(sha(path) == item['case_sha256'], 'Baseline case changed: ' + ident)
        case = read(path)
        source = Path(case['input_path'])
        require(sha(source) == case['row']['served_sha256'] and
                source.stat().st_size == 4 * case['row']['served_input_tokens'], 'Original input changed')
        counts.append((item['warmup'], case['row']['served_input_tokens']))
    require(counts == [(True, 32768), (False, 32768), (False, 32768),
                       (False, 262045), (False, 262045)], 'Expected complete 32K/256K workload')
    return suite


def prepare(root, baseline, repo, overrides, reference_overrides,
            flashinfer_autotune_seed=None, flashinfer_autotune_hashes=(),
            inductor_seed_sources=None, inductor_seed_hashes=None):
    require(bool(flashinfer_autotune_seed) == bool(flashinfer_autotune_hashes),
            'FlashInfer seed and allowed configuration hashes must be supplied together')
    require(bool(inductor_seed_sources) == bool(inductor_seed_hashes),
            'Inductor sources and allowed module hashes must be supplied together')
    suite = checked_suite(baseline)
    old_profile = read(baseline / 'profile.json')
    profiles = {name: merge(old_profile, patch) for name, patch in
                (('reference', reference_overrides), ('candidate', overrides))}
    for profile in profiles.values():
        validate_profile(profile)
    old_hashes = read(baseline / 'runtime-proof.json')['files']
    names = set(old_hashes)
    names.update(str(path.relative_to(repo)) for path in (repo / 'src/vllm_kvmem').rglob('*')
                 if path.is_file() and '__pycache__' not in path.parts and
                 path.suffix in ('.py', '.json', '.cuh', '.so'))
    # A same-library CPU optimization needs the dirty native sources as well
    # as the loaded .so hash; HEAD alone does not describe that implementation.
    names.update(str(path.relative_to(repo)) for source_root in ('csrc', 'third_party/qw3')
                 for path in (repo / source_root).rglob('*')
                 if path.is_file() and path.suffix in ('.cpp', '.hpp', '.h', '.cu', '.cuh'))
    names.update(('setup.py', 'pyproject.toml'))
    names.update(('scripts/benchmark_prefill_variant.py', 'scripts/benchmark_active_prefill.py',
                  'scripts/prefill_autotune_control.py',
                  'scripts/prefill_inductor_control.py', 'scripts/prefill_inductor_worker.py',
                  'scripts/prefill_inductor_serve.py',
                  'scripts/active_prefill_shapes.py',
                  'scripts/benchmark_prefill_publication_pipeline.py', 'scripts/analyze_pressure_profile.py',
                  'scripts/analyze_logical_prefill.py'))
    origins, inherited = pipeline.runtime_origins(repo, baseline, names)
    hashes = {name: sha(origin) for name, origin in origins.items()}
    proof = dict(files=hashes, source=str(repo), inherited_utilities=inherited,
                 head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), dirty=True)
    root.mkdir(parents=True, exist_ok=False)
    for name, profile in profiles.items():
        arm = root / name
        arm.mkdir()
        for relative, origin in origins.items():
            target = arm / 'runtime' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, target)
            require(sha(target) == hashes[relative], 'Source changed during freeze: ' + relative)
        save(arm / 'runtime-proof.json', proof)
        save(arm / 'profile.json', profile)
        shutil.copy2(baseline / 'suite.json', arm / 'suite.json')
        for item in suite['cases']:
            target = arm / 'requests' / item['id']
            target.mkdir(parents=True)
            shutil.copy2(baseline / 'requests' / item['id'] / 'case.json', target / 'case.json')
        save(arm / 'manifest.json', dict(profile_sha256=sha(arm / 'profile.json'),
             scope='Complete matched 32K and 256K, K32/G16, native QK reuse and publication pipeline; '
                   'four formal cases plus warmup. Only declared variant parameters differ.',
             variant=name, source_baseline=str(baseline), suite_sha256=sha(arm / 'suite.json')))
        if flashinfer_autotune_seed is not None:
            from prefill_autotune_control import freeze_seed
            freeze_seed(arm, flashinfer_autotune_seed, flashinfer_autotune_hashes)
        if inductor_seed_sources is not None:
            from prefill_inductor_control import freeze_seed as freeze_inductor_seed
            freeze_inductor_seed(arm, inductor_seed_sources, inductor_seed_hashes)
    require(all(sha(path) == hashes[name] for name, path in origins.items()), 'Source changed during freeze')
    shutil.copy2(repo / 'scripts/benchmark_prefill_variant.py', root / 'driver.py')
    save(root / 'plan.json', dict(schema='kvmem-prefill-variant-1', source_baseline=str(baseline),
         driver_sha256=sha(root / 'driver.py'), runtime_sha256=digest(hashes),
         suite_sha256=sha(baseline / 'suite.json'), overrides=overrides,
         reference_overrides=reference_overrides, runtime_diff_from_baseline=differences(old_hashes, hashes),
         profile_diff_from_baseline={name: differences(old_profile, profile) for name, profile in profiles.items()},
         profile_diff_between_arms=differences(profiles['reference'], profiles['candidate']),
         profile_sha256={name: sha(root / name / 'profile.json') for name in profiles},
         inherited_utilities=inherited, runtime_identical_between_arms=True,
         flashinfer_autotune_seed_sha256=(sha(flashinfer_autotune_seed)
             if flashinfer_autotune_seed is not None else None),
         inductor_seed_manifest_sha256={name: read(root / name / 'manifest.json').get(
             'inductor_seed_manifest_sha256') for name in profiles},
         production_defaults_changed=False, dsh_started=False))
    return root


def assert_runtime(arm):
    from prefill_autotune_control import validate_seed
    from prefill_inductor_control import validate_seed as validate_inductor_seed
    proof = read(arm / 'runtime-proof.json')['files']
    require(all(sha(arm / 'runtime' / name) == value for name, value in proof.items()), 'Frozen runtime changed')
    require(sha(arm / 'profile.json') == read(arm / 'manifest.json')['profile_sha256'], 'Profile changed')
    checked_suite(arm)
    validate_seed(arm)
    validate_inductor_seed(arm)
    return proof


def assert_pair(root):
    from prefill_autotune_control import validate_seed
    plan = read(root / 'plan.json')
    require(sha(root / 'driver.py') == plan['driver_sha256'], 'Frozen driver changed')
    profiles = {}
    for name in ('reference', 'candidate'):
        arm = root / name
        require(digest(assert_runtime(arm)) == plan['runtime_sha256'], 'Runtime differs from frozen plan')
        require(sha(arm / 'suite.json') == plan['suite_sha256'], 'Suite changed')
        require(sha(arm / 'profile.json') == plan['profile_sha256'][name], 'Variant profile changed')
        profiles[name] = read(arm / 'profile.json')
        seed = validate_seed(arm)
        require((seed['seed_sha256'] if seed is not None else None) ==
                plan.get('flashinfer_autotune_seed_sha256'), 'Autotune seed differs from frozen plan')
        require(read(arm / 'manifest.json').get('inductor_seed_manifest_sha256') ==
                plan.get('inductor_seed_manifest_sha256', {}).get(name),
                'Inductor seed differs from frozen plan')
    require(differences(profiles['reference'], profiles['candidate']) == plan['profile_diff_between_arms'],
            'Undeclared profile difference')
    return plan


SELECTION_KEYS = ('selection', 'history_begin', 'selected_tokens', 'replay_span', 'query_span',
    'selected_history_ranges', 'mandatory_fit', 'query_score_tokens', 'pressure_policy', 'alpha',
    'scoring_version', 'score_domain_version', 'seed_source')
ACTION_KEYS = ('samples', 'rounds', 'positive_pages', 'score_sum', 'scope', 'prefill_samples',
    'prefill_chunks', 'prefill_end', 'prefill_pages', 'block_graph')
GRAPH_KEYS = ('phase', 'seed_source', 'pressure_policy', 'mode', 'scoring_version', 'alpha',
    'propagation_applied', 'score_domain_version', 'candidate_count', 'score_mask_count',
    'transport_mask_count', 'revision', 'runtime_epoch', 'accepted_through', 'selected_tokens',
    'budget', 'graph_bytes', 'graph_live_bytes', 'locality_provenance')


def signatures(events, keys):
    return [{key: event.get(key) for key in keys} for event in events]


def audit_opaque_compile_source(arm, profile, progress):
    """Match the executed ABI to this arm's source and native compile key."""
    if not profile['kvmem'].get('capture_opaque_projection', False):
        return None
    import runpy
    source = runpy.run_path(str(arm / 'runtime/src/vllm_kvmem/capture_compile.py'))
    expected = source['capture_compile_fingerprint']()
    report = progress.get('capture_opaque_stats', {})
    require(report.get('source_fingerprint') == expected,
            'Opaque capture projection differs from frozen source')
    argv = read(arm / 'run/server/launch.json')['vllm_argv']
    require(argv.count('--additional-config') == 1,
            'Opaque capture projection lacks a unique compilation key')
    additional = json.loads(argv[argv.index('--additional-config')+1])
    require(additional.get(source['CAPTURE_COMPILE_KEY']) == expected,
            'Opaque capture projection compilation key differs from frozen source')
    return dict(source_fingerprint=expected, native_compilation_key_verified=True)


def audit_optional_branches(profile, progress, archive, native=None):
    """Require execution evidence when a newer variant is explicitly requested.

    Historical frozen baselines without these options/counters remain usable.
    A newer same-code reference must demonstrate that its disabled branches did
    not execute. Hash counters describe completed hashing, before publication.
    """
    kv, result = profile['kvmem'], {}
    if 'cpu_seal_pack_workers' in kv or 'cpu_seal_pack_workers' in archive:
        workers = kv.get('cpu_seal_pack_workers', 1)
        require(type(workers) is int and 1 <= workers <= 8 and
                archive.get('cpu_seal_pack_workers') == workers,
                'Actual CPU seal pack worker count differs')
        completed = archive.get('seal_pack_parallel_batches_completed', 0)
        shards = archive.get('seal_pack_parallel_shards_completed', 0)
        require(type(completed) is int and completed >= 0 and type(shards) is int and shards >= 0,
                'Invalid CPU seal pack execution counts')
        if workers > 1:
            require(completed > 0 and 2*completed <= shards <= workers*completed,
                    'Parallel CPU seal pack has no bounded completed execution evidence')
        else:
            require(completed == shards == 0, 'Single-worker CPU seal pack executed parallel work')
        result['cpu_seal_pack'] = dict(workers=workers, completed_batches=completed,
                                       completed_shards=shards)
    if 'capture_opaque_projection' in kv or 'capture_opaque_projection' in progress:
        enabled = kv.get('capture_opaque_projection', False)
        require(type(enabled) is bool and progress.get('capture_opaque_projection') is enabled,
                'Actual opaque capture projection mode differs')
        report = progress.get('capture_opaque_stats', {})
        require(isinstance(report, dict), 'Invalid opaque capture projection report')
        counts = {key: report.get(key, 0) for key in ('host_dispatch_count', 'layers_installed')}
        require(all(type(value) is int and value >= 0 for value in counts.values()),
                'Invalid opaque capture projection counts')
        if enabled:
            fingerprint = report.get('source_fingerprint', '')
            require(report.get('enabled') is True and counts['host_dispatch_count'] > 0
                    and counts['layers_installed'] > 0 and isinstance(fingerprint, str)
                    and len(fingerprint) == 64 and all(c in '0123456789abcdef' for c in fingerprint),
                    'Opaque capture projection lacks qualified source and dispatch evidence')
            require(report.get('graph_replays_excluded') is True,
                    'Opaque capture projection has an unknown counting scope')
        else:
            require(not any(counts.values()), 'Disabled opaque capture projection executed')
        result['capture_opaque_projection'] = {**report, 'enabled': enabled}
    if 'cpu_mean_index_simd' in kv or 'cpu_mean_index_simd' in progress:
        enabled = kv.get('cpu_mean_index_simd', False)
        require(progress.get('cpu_mean_index_simd') is enabled, 'Actual MeanK SIMD mode differs')
        stats = progress.get('cpu_mean_index_stats', {})
        keys = ('completed_batches', 'completed_records', 'vectorized_records', 'fallback_records')
        counts = {key: stats.get(key, 0) for key in keys}
        require(all(type(value) is int and value >= 0 for value in counts.values()),
                'Invalid MeanK SIMD execution counts')
        require(counts['completed_batches'] > 0 and counts['completed_records'] > 0,
                'Missing completed batched MeanK execution')
        require(counts['fallback_records'] == 0, 'MeanK SIMD unexpectedly used scalar fallback')
        require(counts['vectorized_records'] == (counts['completed_records'] if enabled else 0),
                'MeanK SIMD branch differs from requested mode')
        result['cpu_mean_index_simd'] = dict(enabled=enabled, **counts)
    if ('action_native_prefill_aligned_pieces' in kv or
            isinstance(native, dict) and 'prefill_aligned_pieces' in native):
        enabled = kv.get('action_native_prefill_aligned_pieces', False)
        report = native.get('prefill_aligned_pieces', {}) if isinstance(native, dict) else {}
        require(report.get('enabled') is enabled, 'Actual aligned native prefill mode differs')
        stats = report.get('counts', {})
        require(isinstance(stats, dict) and all(type(value) is int and value >= 0
                for value in stats.values()), 'Invalid aligned native prefill counts')
        if enabled:
            require(stats.get('aligned_candidate_owners', 0) > 0 and
                    stats.get('aligned_candidate_rows', 0) > 0 and
                    stats.get('prefill_dispatch_calls', 0) > 0,
                    'Aligned native prefill has no qualified rows or fixed-module dispatch')
            require(report.get('candidate_counts_are_host_qualification') is True and
                    report.get('dispatch_counts_exclude_graph_replays') is True,
                    'Aligned native prefill evidence has an unknown counting scope')
        else:
            require(not any(stats.values()), 'Disabled aligned native prefill executed')
        result['native_prefill_aligned_pieces'] = report
    if ('action_native_prefill_cta_gate' in kv or
            isinstance(native, dict) and 'prefill_cta_gate' in native):
        enabled = kv.get('action_native_prefill_cta_gate', False)
        report = native.get('prefill_cta_gate', {}) if isinstance(native, dict) else {}
        require(type(enabled) is bool and report.get('enabled') is enabled,
                'Actual native prefill CTA gate mode differs')
        counts = report.get('counts', {})
        require(isinstance(counts, dict) and all(type(value) is int and value >= 0
                for value in counts.values()), 'Invalid native prefill CTA gate counts')
        if enabled:
            require(counts.get('prefill_dispatch_calls', 0) > 0 and
                    isinstance(report.get('source_fingerprint'), str) and
                    bool(report['source_fingerprint']),
                    'Native prefill CTA gate has no fixed-module dispatch evidence')
            require(report.get('dispatch_counts_exclude_graph_replays') is True and
                    report.get('device_skipped_ctas_not_counted') is True,
                    'Native prefill CTA gate evidence has an unknown counting scope')
        else:
            require(not any(counts.values()), 'Disabled native prefill CTA gate executed')
        result['native_prefill_cta_gate'] = report
    if ('action_native_prefill_static_aligned' in kv or
            isinstance(native, dict) and 'prefill_static_aligned' in native):
        enabled = kv.get('action_native_prefill_static_aligned', False)
        report = native.get('prefill_static_aligned', {}) if isinstance(native, dict) else {}
        require(type(enabled) is bool and report.get('enabled') is enabled,
                'Actual static aligned native prefill mode differs')
        counts = report.get('counts', {})
        require(isinstance(counts, dict) and all(type(key) is str and
                (key in ('static_dispatch_calls', 'qualified_candidate_frames') or
                 any(key.startswith(prefix) and len(key) > len(prefix)
                     for prefix in ('generic_dispatch:', 'candidate_fallback:'))) and
                type(value) is int and value >= 0 for key, value in counts.items()),
                'Invalid static aligned native prefill counts')
        if enabled:
            require(counts.get('static_dispatch_calls', 0) > 0 and
                    isinstance(report.get('source_fingerprint'), str) and
                    bool(report['source_fingerprint']) and report.get('unavailable') is None,
                    'Static aligned native prefill lacks qualified module dispatch')
            require(report.get('graph_dispatches_use_generic') is True and
                    report.get('dispatch_counts_exclude_graph_replays') is True,
                    'Static aligned native prefill evidence has an unknown counting scope')
        else:
            require(not any(counts.values()), 'Disabled static aligned native prefill executed')
        result['native_prefill_static_aligned'] = report
    # Old callers may omit the new report when plan arming is disabled. A
    # requested candidate cannot pass with only its profile flag as evidence.
    if native is not None or kv.get('prefill_native_plan_arm', False):
        enabled = kv.get('prefill_native_plan_arm', False)
        require(type(enabled) is bool and isinstance(native, dict) and
                isinstance(native.get('layer_phase_counts'), dict), 'Missing native prefill plan execution evidence')
        counts = native['layer_phase_counts']
        planned = counts.get('prepare:prefill_plan', 0)
        eager = counts.get('prepare:eager', 0)
        require(type(planned) is int and type(eager) is int and planned >= 0 and eager >= 0,
                'Invalid native prefill plan counts')
        require(planned > 0 if enabled else planned == 0, 'Native prefill plan execution differs from requested mode')
        require(planned+eager > 0, 'Missing native preparation evidence')
        result['native_prefill_plan'] = dict(enabled=enabled, planned_batches=planned, eager_batches=eager)
    if 'prefill_action_before_mtp' in kv or 'prefill_action_before_mtp' in progress:
        enabled = kv.get('prefill_action_before_mtp', False)
        require(progress.get('prefill_action_before_mtp') is enabled, 'Actual early MTP Action mode differs')
        stats = progress.get('prefill_action_before_mtp_stats', {})
        keys = ('staged_batches', 'resolved_batches', 'fallback_batches', 'cancelled_batches', 'quarantined_frames')
        counts = {key: stats.get(key, 0) for key in keys}
        require(all(type(count) is int and count >= 0 for count in counts.values()), 'Invalid early MTP Action counts')
        require(counts['staged_batches'] == counts['resolved_batches'], 'Unresolved early MTP Action frame')
        require(all(counts[key] == 0 for key in keys[2:]), 'Early MTP Action fallback/cancel/quarantine occurred')
        require(counts['staged_batches'] > 0 if enabled else counts['staged_batches'] == 0,
                'Early MTP Action execution differs from requested mode')
        result['action_before_mtp'] = dict(enabled=enabled, **counts)
    if 'prefill_scatter_after_target' in kv or 'prefill_scatter_after_target' in progress:
        enabled = kv.get('prefill_scatter_after_target', False)
        require(progress.get('prefill_scatter_after_target') is enabled,
                'Actual scatter after target mode differs')
        stats = progress.get('prefill_scatter_after_target_stats', {})
        keys = ('deferred_batches', 'after_target_started_batches',
                'resumed_before_target_batches', 'no_work_batches',
                'quarantined_batches', 'already_running_batches', 'eligibility_declines')
        counts = {key: stats.get(key, 0) for key in keys}
        require(all(type(count) is int and count >= 0 for count in counts.values()),
                'Invalid scatter after target counts')
        require(counts['quarantined_batches'] == 0 and
                counts['deferred_batches'] == counts['after_target_started_batches'] +
                counts['resumed_before_target_batches'] + counts['no_work_batches'],
                'Scatter after target lifecycle is incomplete or quarantined')
        require(counts['after_target_started_batches'] > 0 if enabled else
                all(count == 0 for count in counts.values()),
                'Scatter after target execution differs from requested mode')
        result['scatter_after_target'] = dict(enabled=enabled, **counts)
    if 'prefill_deferred_validation' in kv or 'prefill_deferred_validation' in progress:
        enabled = kv.get('prefill_deferred_validation', False)
        require(progress.get('prefill_deferred_validation') is enabled,
                'Actual deferred validation mode differs')
        stats = progress.get('prefill_action_before_mtp_stats', {})
        submitted = stats.get('deferred_batches', 0)
        completed = stats.get('deferred_resolved_batches', 0)
        fallback = stats.get('deferred_fallback_batches', 0)
        require(all(type(x) is int and x >= 0 for x in (submitted, completed, fallback)),
                'Invalid deferred validation counts')
        require(submitted == completed and fallback == 0,
                'Deferred validation is incomplete or used reference fallback')
        require(submitted > 0 if enabled else submitted == 0,
                'Deferred validation branch differs from requested mode')
        pool = progress.get('prefill_deferred_validation_pool')
        if enabled:
            require(isinstance(pool, dict) and pool.get('capacity') == 2 and
                    pool.get('allocated_slots') == 2 and pool.get('active_slots') == 0 and
                    pool.get('disabled_reason') is None and pool.get('quarantined') is False,
                    'Deferred validation pool is unavailable or still leased')
            require(pool.get('resident_bytes', float('inf')) <= pool.get('max_bytes', 0) and
                    pool.get('scratch_bytes', float('inf')) <= pool.get('max_scratch_bytes', 0),
                    'Deferred validation exceeded its resource budgets')
            usage = pool.get('counts', {})
            require(usage.get('frozen_frames') == usage.get('released_frames') == submitted,
                    'Deferred validation snapshots did not complete their lifecycle')
        result['deferred_validation'] = dict(enabled=enabled, submitted=submitted,
            completed=completed, fallback=fallback, pool=pool)
    if 'capture_fused_pack' in kv or 'capture_fused_pack' in progress:
        enabled = kv.get('capture_fused_pack', False)
        require(progress.get('capture_fused_pack') is enabled, 'Actual fused capture pack mode differs')
        stats = progress.get('capture_pack_stats', {})
        fused, reference = stats.get('fused_batches', 0), stats.get('reference_batches', 0)
        payload_bytes = stats.get('fused_payload_bytes', 0)
        metadata_bytes = progress.get('capture_pack_metadata_capacity_bytes')
        fallbacks = {key.removeprefix('fallback:'): count for key, count in stats.items()
                     if key.startswith('fallback:')}
        require(all(type(count) is int and count >= 0 for count in
                    (fused, reference, payload_bytes, *fallbacks.values())), 'Invalid capture pack counts')
        if enabled:
            require(fused > 0 and payload_bytes > 0 and type(metadata_bytes) is int and metadata_bytes > 0,
                    'Fused capture pack requested but no fused payload completed')
            require(sum(fallbacks.values()) == reference, 'Fused pack reference path lacks fallback reasons')
        else:
            require(fused == payload_bytes == 0 and reference > 0 and metadata_bytes == 0 and not fallbacks,
                    'Disabled fused capture pack executed or lacks reference evidence')
        result['capture_pack'] = dict(enabled=enabled, fused_batches=fused, reference_batches=reference,
            fused_payload_bytes=payload_bytes, metadata_capacity_bytes=metadata_bytes, fallback_reasons=fallbacks)
    if 'prefill_capture_ahead' in kv or 'prefill_capture_ahead' in progress:
        enabled = kv.get('prefill_capture_ahead', False)
        require(progress.get('prefill_capture_ahead') is enabled, 'Actual capture ahead mode differs')
        stats = progress.get('prefill_pipeline_stats', {})
        batches = stats.get('capture_ahead_batches', 0)
        require(batches > 0 if enabled else batches == 0, 'Capture ahead execution does not match requested mode')
        require(stats.get('capture_ahead_discarded', 0) == 0, 'Capture ahead discarded a failed Action batch')
        result['capture_ahead'] = dict(enabled=enabled, completed_batches=batches)
    if 'cpu_seal_hash_workers' in kv or 'cpu_seal_hash_workers' in archive:
        workers = kv.get('cpu_seal_hash_workers', 1)
        require(archive.get('cpu_seal_hash_workers') == workers, 'Actual seal hash worker setting differs')
        batches = archive.get('seal_hash_parallel_batches_completed')
        shards = archive.get('seal_hash_parallel_shards_completed')
        require(type(batches) is int and type(shards) is int, 'Missing actual parallel hash counters')
        if workers > 1:
            require(0 < batches <= archive.get('seal_batches', 0) and 2*batches <= shards <= workers*batches,
                    'Parallel hash branch did not complete bounded shard batches')
        else:
            require(batches == shards == 0, 'Disabled parallel hash branch executed')
        result['seal_hash'] = dict(workers=workers, parallel_batches_completed=batches,
                                  parallel_shards_completed=shards)
    if 'cpu_direct_record_seal' in kv:
        enabled, native = kv['cpu_direct_record_seal'], archive.get('cpu_native_payloads', False)
        direct = archive.get('seal_direct_bytes')
        require(type(direct) is int and direct >= 0, 'Missing direct seal byte accounting')
        if enabled:
            require(direct > 0, 'Direct seal requested but no direct bytes published')
        elif not native:
            require(direct == 0, 'Disabled direct seal branch published direct bytes')
        result['direct_seal'] = dict(enabled=enabled, published_bytes=direct,
            native_payloads=native, note='Native payload ownership has a separate always-direct publication path.' if native else None)
    require(archive.get('archive_io_pending_bytes') == 0, 'Archive I/O is still pending')
    return result


def audit_assembly_prefetch(profile, refreshes):
    if ('assembly_read_prefetch' not in profile['kvmem'] and
            not any('assembly_read_prefetch' in row for row in refreshes)):
        return {}
    enabled = profile['kvmem'].get('assembly_read_prefetch', False)
    require(bool(refreshes), 'Missing assembly read prefetch observations')
    keys = ('submitted', 'completed', 'abandoned', 'slot_busy', 'admission_busy', 'eligible', 'declined')
    totals = dict.fromkeys(keys, 0)
    for row in refreshes:
        require(row.get('assembly_read_prefetch') is enabled, 'Actual assembly read prefetch mode differs')
        stats = row.get('assembly_read_stats', {})
        counts = {key: stats.get('assembly_prefetch_'+key, 0) for key in keys}
        require(all(type(value) is int and value >= 0 for value in counts.values()),
                'Invalid assembly prefetch counts')
        require(counts['submitted'] == counts['completed'] and counts['abandoned'] == 0,
                'Assembly prefetch did not complete cleanly')
        if not enabled:
            require(not any(counts.values()), 'Disabled assembly prefetch executed')
        for key in keys:
            totals[key] += counts[key]
    if enabled:
        require(totals['completed'] > 0, 'Assembly prefetch requested but no read was prefetched')
    return dict(enabled=enabled, **totals)


def audit_assembly_fusion(profile, refreshes):
    if ('assembly_fused_layers' not in profile['kvmem'] and
            not any('assembly_fused_layers' in row for row in refreshes)):
        return {}
    enabled = profile['kvmem'].get('assembly_fused_layers', False)
    require(bool(refreshes), 'Missing assembly layer fusion observations')
    totals = dict(chunks=0, layer_launches=0, fallback_chunks=0)
    metadata_bytes = 0
    reasons = {}
    for row in refreshes:
        require(row.get('assembly_fused_layers') is enabled, 'Actual assembly layer fusion mode differs')
        stats = row.get('assembly_read_stats', {})
        counts = {key: stats.get('assembly_fused_'+key, 0) for key in (*totals, 'metadata_bytes')}
        fallback = {key.removeprefix('assembly_fused_fallback:'): value for key, value in stats.items()
                    if key.startswith('assembly_fused_fallback:')}
        require(all(type(value) is int and value >= 0 for value in (*counts.values(), *fallback.values())),
                'Invalid assembly layer fusion counts')
        require(counts['layer_launches'] >= counts['chunks'], 'Fusion layer count is smaller than chunks')
        if not enabled:
            require(not any((*counts.values(), *fallback.values())), 'Disabled assembly layer fusion executed')
        if counts['chunks']:
            require(counts['metadata_bytes'] > 0, 'Fused assembly lacks bounded metadata accounting')
        metadata_bytes = max(metadata_bytes, counts['metadata_bytes'])
        for key in totals:
            totals[key] += counts[key]
        for key, value in fallback.items():
            reasons[key] = reasons.get(key, 0) + value
    if enabled:
        require(totals['chunks'] > 0, 'Assembly layer fusion requested but never executed')
    return dict(enabled=enabled, **totals, max_metadata_bytes=metadata_bytes, fallback_reasons=reasons)


def audit_inductor_pair(seeds):
    """Both arms must actually execute the same frozen common-kernel choices."""
    if not any(seed is not None for seed in seeds):
        return
    require(len(seeds) == 2 and all(seed is not None for seed in seeds),
            'Only one arm used a frozen Inductor seed')
    require(seeds[0]['seed_manifest_sha256'] == seeds[1]['seed_manifest_sha256'],
            'Inductor module seeds differ between arms')
    require(all(seed.get('launch_verified') is True and
                seed.get('common_configurations_unchanged') is True and
                seed.get('formal_dispatch_observer') is False for seed in seeds),
            'Inductor common configurations need actual startup-only launch verification')
    require(set(seeds[0]['modules']) == set(seeds[1]['modules']), 'Inductor module roles differ')
    for role in seeds[0]['modules']:
        modules = [seed['modules'][role] for seed in seeds]
        entries = [{item['key']: item for item in module['common_kernels']} for module in modules]
        common = set(entries[0]) & set(entries[1])
        require(bool(common), 'No actually executed common Inductor kernels: ' + role)
        for key in common:
            fields = ('source_sha256', 'function_source_sha256', 'configs_hash', 'selected_config')
            require(all(entries[0][key][field] == entries[1][key][field] for field in fields),
                    'Actually executed Inductor common kernel differs: ' + role + '/' + key)


def compare(base, candidate, expected_overrides=None):
    """Strict completed-run comparison, also usable for manually frozen arms."""
    import benchmark_active_prefill as geometry_module
    import analyze_pressure_profile as pressure_module
    import qualify_block_graph as selection_module
    import prefill_autotune_control as autotune_module
    import prefill_inductor_control as inductor_module
    audit_seed = autotune_module.audit_seed
    reconstruct_geometry = geometry_module.reconstruct_geometry
    refresh_intervals = pressure_module.refresh_intervals
    native_matches = selection_module.native_matches
    proofs = [assert_runtime(arm) for arm in (base, candidate)]
    require(proofs[0] == proofs[1], 'Variant runtime must be identical between arms')
    profiles = [read(arm / 'profile.json') for arm in (base, candidate)]
    for profile in profiles:
        validate_profile(profile)
    if expected_overrides is not None:
        require(merge(profiles[0], expected_overrides) == profiles[1], 'Undeclared profile difference')
    require(read(base / 'suite.json') == read(candidate / 'suite.json'), 'Case suites differ')
    suite = read(base / 'suite.json')['cases']
    result = dict(runtime_identical=True, runtime_sha256=digest(proofs[0]),
        profile_diff=differences(*profiles), expected_overrides=expected_overrides,
        pairs=[], summary={}, cold_prepare_seconds={}, final_progress={}, final_archive={}, final_native={}, actual_branches={},
        source_paths=dict(base=str(base), candidate=str(candidate)))
    result['analyzer_sources'] = {str(Path(path).resolve()): sha(Path(path)) for path in
        (__file__, pipeline.__file__, geometry_module.__file__, pressure_module.__file__, selection_module.__file__,
         autotune_module.__file__, inductor_module.__file__)}
    arm_samples = {}
    result['flashinfer_autotune_control'] = {}
    result['inductor_autotune_control'] = {}
    for name, arm in (('base', base), ('candidate', candidate)):
        status = read(arm / 'status.json')
        require(status['status'] == 'completed', 'Comparison requires every formal case completed')
        require(read(arm / 'cleanup.json') == dict(remaining_owned_groups={}, gpu_processes=[]), 'Cleanup incomplete')
        result['flashinfer_autotune_control'][name] = audit_seed(arm)
        result['inductor_autotune_control'][name] = inductor_module.audit_seed(arm)
        audit = read(arm / 'run/execution-audit.json')
        require(audit['passed'] and audit['public_work']['all_submitted_completed'], 'Public work did not complete')
        capture = arm / 'run/server/capture'
        progress = read(capture / 'progress.json')
        pipeline.audit_pipeline(progress, True)
        archive = read(capture / 'archive.json')
        native_report = read(capture / 'native_attention_stats.json')
        profile = read(arm / 'profile.json')
        result['actual_branches'][name] = audit_optional_branches(profile, progress, archive, native_report)
        opaque_source = audit_opaque_compile_source(arm, profile, progress)
        if opaque_source is not None:
            result['actual_branches'][name]['capture_opaque_projection'].update(opaque_source)
        result['cold_prepare_seconds'][name] = status['cold_prepare_seconds']
        result['final_progress'][name] = progress
        result['final_archive'][name] = archive
        result['final_native'][name] = native_report
        groups = pipeline.prefill_groups(rows(capture / 'model_step_latency.jsonl'))
        require(len(groups) == len(suite), 'Unexpected or missing prefill timing group')
        refreshes = rows(capture / 'bounded_refresh.jsonl')
        result['actual_branches'][name]['assembly_prefetch'] = audit_assembly_prefetch(
            read(arm / 'profile.json'), refreshes)
        result['actual_branches'][name]['assembly_fusion'] = audit_assembly_fusion(
            read(arm / 'profile.json'), refreshes)
        stages = refresh_intervals(rows(capture / 'async_refresh.jsonl'), refreshes)
        samples = {}
        for item, group in zip(suite, groups):
            if item['warmup']:
                continue
            directory = arm / 'requests' / item['id']
            response, evidence = read(directory / 'response.json'), read(directory / 'evidence.json')
            measured = read(directory / 'result.json')
            native = read(directory / 'native-stats.json')
            require(not any(v for k, v in native['delta'].items() if k.startswith('fallback:')) and
                    native['delta'].get('reuse:prefill', 0) > 0 and
                    not native['after']['validation_failures'] and native['after']['disabled_reason'] is None,
                    'Actual native reuse failed or fell back')
            geometry = reconstruct_geometry(group, evidence['refresh'], measured['input_tokens'])
            matched_stages = [stage for stage in stages if native_matches(response, stage)]
            resource = rows(directory / 'resources.jsonl')
            samples[item['id']] = dict(result=measured, output_sha256=digest(response['output_ids']),
                output_ids=response['output_ids'], evidence=evidence, native_reuse_delta=native['delta'],
                geometry_sha256=digest(geometry), timing=pipeline.timing(group),
                mtp=pipeline.mtp_delta(pipeline.parse_metrics(directory / 'initial-metrics.json'),
                                       pipeline.parse_metrics(directory / 'final-metrics.json')),
                pressure_wait_seconds=sum(x['seconds'] for x in matched_stages if x['phase'] == 'pressure'),
                query_wait_seconds=sum(x['seconds'] for x in matched_stages if x['phase'] == 'query'),
                **pipeline.sampled_process_memory(resource),
                max_sampled_gpu_used_mib=max((int(x['gpu'].split(',')[0]) for x in resource), default=None))
        arm_samples[name] = samples
    seeds = list(result['flashinfer_autotune_control'].values())
    if any(seed is not None for seed in seeds):
        require(all(seed is not None for seed in seeds), 'Only one arm used a frozen FlashInfer seed')
        require(all(seed.get('loaded_files_semantically_identical') is True and
                    seed.get('actual_configs_sha256') is not None for seed in seeds),
                'Loaded FlashInfer files do not share one effective tactic configuration')
        require(seeds[0]['seed_sha256'] == seeds[1]['seed_sha256'] and
                seeds[0]['actual_configs_sha256'] == seeds[1]['actual_configs_sha256'],
                'FlashInfer effective autotune configurations differ between arms')
    audit_inductor_pair(list(result['inductor_autotune_control'].values()))
    for item in suite:
        if item['warmup']:
            continue
        ident = item['id']
        left, right = (arm_samples[name][ident] for name in ('base', 'candidate'))
        checks = dict(output_token_ids_identical=left['output_ids'] == right['output_ids'],
            selections_and_mandatory_identical=signatures(left['evidence']['refresh'], SELECTION_KEYS) ==
                                               signatures(right['evidence']['refresh'], SELECTION_KEYS),
            graph_selection_reports_identical=signatures(left['evidence']['graph'], GRAPH_KEYS) ==
                                               signatures(right['evidence']['graph'], GRAPH_KEYS),
            accepted_score_graph_summary_identical=signatures(left['evidence']['action'], ACTION_KEYS) ==
                                                    signatures(right['evidence']['action'], ACTION_KEYS),
            mtp_identical=left['mtp'] == right['mtp'], geometry_identical=left['geometry_sha256'] == right['geometry_sha256'])
        for sample in (left, right):
            sample.pop('output_ids')
        result['pairs'].append(dict(id=ident, checks=checks, selection_events=len(left['evidence']['refresh']),
            **{name: {key: value for key, value in sample.items() if key != 'evidence'}
               for name, sample in (('base', left), ('candidate', right))}))
    for length in sorted({pair['base']['result']['input_tokens'] for pair in result['pairs']}):
        pairs = [pair for pair in result['pairs'] if pair['base']['result']['input_tokens'] == length]
        summary = {}
        for name in ('base', 'candidate'):
            samples = [pair[name] for pair in pairs]
            summary[name] = dict(repetitions=len(samples),
                mean_ttft_seconds=statistics.mean(x['result']['ttft_seconds'] for x in samples),
                pooled_prefill_tokens_per_second=sum(x['result']['input_tokens'] for x in samples) /
                                                sum(x['result']['ttft_seconds'] for x in samples),
                mean_model_cuda_seconds=statistics.mean(x['timing']['model_cuda_seconds'] for x in samples),
                mean_host_submit_seconds=statistics.mean(x['timing']['host_submit_seconds'] for x in samples),
                mean_host_gap_seconds=statistics.mean(x['timing']['host_between_submissions_seconds'] for x in samples),
                mean_pressure_wait_seconds=statistics.mean(x['pressure_wait_seconds'] for x in samples),
                mean_query_wait_seconds=statistics.mean(x['query_wait_seconds'] for x in samples))
        summary['throughput_change_percent'] = 100 * (summary['candidate']['pooled_prefill_tokens_per_second'] /
                                                     summary['base']['pooled_prefill_tokens_per_second'] - 1)
        result['summary'][str(length)] = summary
    result['all_checks_passed'] = len(result['pairs']) == 4 and all(all(pair['checks'].values()) for pair in result['pairs'])
    result['timing_scope'] = 'CUDA, host and refresh intervals can overlap; they are not additive cost partitions. Cold preparation is separate.'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--root', type=Path, required=True)
    prep.add_argument('--baseline', type=Path, required=True)
    prep.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    prep.add_argument('--overrides', type=json.loads, required=True)
    prep.add_argument('--reference-overrides', type=json.loads, default={})
    prep.add_argument('--flashinfer-autotune-seed', type=Path)
    prep.add_argument('--flashinfer-autotune-hash', action='append', default=[])
    prep.add_argument('--inductor-seed-sources', type=json.loads)
    prep.add_argument('--inductor-seed-hashes', type=json.loads)
    run = commands.add_parser('run')
    run.add_argument('--root', type=Path, required=True)
    run.add_argument('--arm', choices=('reference', 'candidate'), required=True)
    analysis = commands.add_parser('analyze')
    analysis.add_argument('--root', type=Path, required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--base', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--expected-overrides', type=json.loads, required=True)
    comparison.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(prepare(args.root.resolve(), args.baseline.resolve(), args.repo.resolve(),
                      args.overrides, args.reference_overrides,
                      args.flashinfer_autotune_seed, args.flashinfer_autotune_hash,
                      args.inductor_seed_sources, args.inductor_seed_hashes))
    elif args.command == 'run':
        root = args.root.resolve()
        assert_pair(root)
        target = root / args.arm
        require(not (target / 'run').exists(), 'Refusing to rerun an arm')
        command = [sys.executable, '-B', str(target / 'runtime/scripts/benchmark_matched_prefill.py'),
                   'run', '--root', str(target)]
        os.execv(sys.executable, command)
    else:
        if args.command == 'analyze':
            root = args.root.resolve()
            assert_pair(root)
            result = compare(root / 'reference', root / 'candidate')
            out = root / 'comparison.json'
        else:
            result = compare(args.base.resolve(), args.candidate.resolve(), args.expected_overrides)
            out = args.out.resolve()
        save(out, result)
        print(json.dumps({key: result[key] for key in ('profile_diff', 'all_checks_passed', 'summary', 'cold_prepare_seconds')}, indent=2))
        require(result['all_checks_passed'], 'Semantic comparison failed; inspect ' + str(out))


if __name__ == '__main__':
    main()
