"""Freeze and completed-run guardrails; no model loading or GPU work."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_prefill_variant as variant


def test_nested_overrides_do_not_mutate_or_erase_other_options():
    original = {'kvmem': {'assembly_chunk_blocks': 32, 'prefill_native_plan_arm': False}, 'draft_tokens': 4}
    before = deepcopy(original)
    updated = variant.merge(original, {'kvmem': {'assembly_chunk_blocks': 128}})
    assert original == before
    assert updated == {'kvmem': {'assembly_chunk_blocks': 128, 'prefill_native_plan_arm': False}, 'draft_tokens': 4}
    assert variant.differences(original, updated) == [dict(path='kvmem.assembly_chunk_blocks',
        before_present=True, after_present=True, before=32, after=128)]
    with pytest.raises(ValueError, match='replaces an object'):
        variant.merge(original, {'kvmem': None})


def test_diff_distinguishes_missing_default_from_explicit_null():
    assert variant.differences({}, {'option': None}) == [dict(path='option', before_present=False,
        after_present=True, before=None, after=None)]


def test_parallel_pack_requires_completed_bounded_work_and_explicit_worker_count():
    profile = dict(kvmem=dict(cpu_seal_pack_workers=8))
    archive = dict(archive_io_pending_bytes=0, cpu_seal_pack_workers=8,
        seal_pack_parallel_batches_completed=5, seal_pack_parallel_shards_completed=40)
    assert variant.audit_optional_branches(profile, {}, archive)['cpu_seal_pack'] == dict(
        workers=8, completed_batches=5, completed_shards=40)
    for change in ({'cpu_seal_pack_workers': 1},
                   {'seal_pack_parallel_batches_completed': 0},
                   {'seal_pack_parallel_batches_completed': True},
                   {'seal_pack_parallel_shards_completed': 41},
                   {'seal_pack_parallel_shards_completed': 5}):
        with pytest.raises(ValueError, match='CPU seal pack'):
            variant.audit_optional_branches(profile, {}, {**archive, **change})
    disabled = dict(archive_io_pending_bytes=0, cpu_seal_pack_workers=1,
        seal_pack_parallel_batches_completed=0, seal_pack_parallel_shards_completed=0)
    variant.audit_optional_branches(dict(kvmem={}), {}, disabled)
    with pytest.raises(ValueError, match='Single-worker CPU seal pack'):
        variant.audit_optional_branches(dict(kvmem={}), {},
            {**disabled, 'seal_pack_parallel_batches_completed': 1})


def test_mean_index_simd_requires_actual_execution_without_fallback():
    profile = dict(kvmem=dict(cpu_mean_index_simd=True))
    stats = dict(completed_batches=128, completed_records=8192,
                 vectorized_records=8192, fallback_records=0)
    progress = dict(cpu_mean_index_simd=True, cpu_mean_index_stats=stats)
    archive = dict(archive_io_pending_bytes=0)
    report = variant.audit_optional_branches(profile, progress, archive)['cpu_mean_index_simd']
    assert report['vectorized_records'] == 8192
    for updates in ({'vectorized_records': 0}, {'fallback_records': 1},
                    {'completed_records': 0}, {'completed_batches': -1}):
        with pytest.raises(ValueError, match='MeanK SIMD|batched MeanK'):
            variant.audit_optional_branches(profile,
                {**progress, 'cpu_mean_index_stats': {**stats, **updates}}, archive)
    with pytest.raises(ValueError, match='Actual MeanK SIMD'):
        variant.audit_optional_branches(profile, {}, archive)
    reference = dict(cpu_mean_index_simd=False,
        cpu_mean_index_stats={**stats, 'vectorized_records': 0})
    variant.audit_optional_branches(dict(kvmem={}), reference, archive)
    with pytest.raises(ValueError, match='branch differs'):
        variant.audit_optional_branches(dict(kvmem={}),
            {**reference, 'cpu_mean_index_stats': stats}, archive)


def test_cta_gate_requires_actual_dispatch_without_claiming_skipped_cta_count():
    native = dict(layer_phase_counts={'prepare:eager': 3}, prefill_cta_gate=dict(
        enabled=True, counts={'prefill_dispatch_calls': 9}, source_fingerprint='qualified-overlay',
        dispatch_counts_exclude_graph_replays=True, device_skipped_ctas_not_counted=True))
    profile = dict(kvmem=dict(action_native_prefill_cta_gate=True))
    archive = dict(archive_io_pending_bytes=0)
    result = variant.audit_optional_branches(profile, {}, archive, native=native)
    assert result['native_prefill_cta_gate'] == native['prefill_cta_gate']
    for change in ({'enabled': False}, {'counts': {}}, {'counts': {'prefill_dispatch_calls': True}},
                   {'source_fingerprint': None}, {'device_skipped_ctas_not_counted': False},
                   {'dispatch_counts_exclude_graph_replays': False}):
        with pytest.raises(ValueError, match='CTA gate'):
            variant.audit_optional_branches(profile, {}, archive,
                native={**native, 'prefill_cta_gate': {**native['prefill_cta_gate'], **change}})
    disabled = {**native, 'prefill_cta_gate': dict(enabled=False, counts={})}
    variant.audit_optional_branches(dict(kvmem={}), {}, archive, native=disabled)
    disabled['prefill_cta_gate']['counts']['prefill_dispatch_calls'] = 1
    with pytest.raises(ValueError, match='Disabled native prefill CTA gate'):
        variant.audit_optional_branches(dict(kvmem={}), {}, archive, native=disabled)


def test_static_aligned_requires_qualified_dispatch_and_reports_generic_routes():
    report = dict(enabled=True, counts={'static_dispatch_calls': 6,
        'generic_dispatch:graph_wrapper': 3, 'qualified_candidate_frames': 2,
        'candidate_fallback:unaligned_logical_segments': 1}, source_fingerprint='qualified-overlay',
        unavailable=None, graph_dispatches_use_generic=True,
        dispatch_counts_exclude_graph_replays=True)
    native = dict(layer_phase_counts={'prepare:eager': 3}, prefill_static_aligned=report)
    profile = dict(kvmem=dict(action_native_prefill_static_aligned=True))
    archive = dict(archive_io_pending_bytes=0)
    assert variant.audit_optional_branches(profile, {}, archive, native=native)[
        'native_prefill_static_aligned'] == report
    for change in ({'enabled': False}, {'counts': {}},
                   {'counts': {'static_dispatch_calls': True}},
                   {'counts': {'unknown': 1, 'static_dispatch_calls': 1}},
                   {'source_fingerprint': None}, {'unavailable': 'build failed'},
                   {'graph_dispatches_use_generic': False},
                   {'dispatch_counts_exclude_graph_replays': False}):
        with pytest.raises(ValueError, match='[Ss]tatic aligned'):
            variant.audit_optional_branches(profile, {}, archive,
                native={**native, 'prefill_static_aligned': {**report, **change}})
    disabled = {**native, 'prefill_static_aligned': dict(enabled=False, counts={})}
    variant.audit_optional_branches(dict(kvmem={}), {}, archive, native=disabled)
    disabled['prefill_static_aligned']['counts']['static_dispatch_calls'] = 1
    with pytest.raises(ValueError, match='Disabled static aligned'):
        variant.audit_optional_branches(dict(kvmem={}), {}, archive, native=disabled)


def test_opaque_capture_requires_real_dispatch_and_explicit_graph_counting_scope():
    profile = dict(kvmem=dict(capture_opaque_projection=True))
    stats = dict(enabled=True, host_dispatch_count=17, layers_installed=17,
                 source_fingerprint='a'*64, graph_replays_excluded=True)
    progress = dict(capture_opaque_projection=True, capture_opaque_stats=stats)
    archive = dict(archive_io_pending_bytes=0)
    assert variant.audit_optional_branches(profile, progress, archive)['capture_opaque_projection'] == stats
    for change in ({'host_dispatch_count': 0}, {'host_dispatch_count': True},
                   {'layers_installed': 0}, {'source_fingerprint': 'unqualified'},
                   {'graph_replays_excluded': False}, {'enabled': False}):
        with pytest.raises(ValueError, match='[Oo]paque capture projection'):
            variant.audit_optional_branches(profile,
                {**progress, 'capture_opaque_stats': {**stats, **change}}, archive)
    with pytest.raises(ValueError, match='Actual opaque'):
        variant.audit_optional_branches(profile, {}, archive)
    disabled = dict(capture_opaque_projection=False, capture_opaque_stats={})
    assert variant.audit_optional_branches(dict(kvmem={}), disabled, archive) == {
        'capture_opaque_projection': {'enabled': False}}
    with pytest.raises(ValueError, match='Disabled opaque'):
        variant.audit_optional_branches(dict(kvmem={}),
            {**disabled, 'capture_opaque_stats': stats}, archive)


def test_opaque_compile_audit_uses_frozen_helper_and_rejects_stale_launch(tmp_path):
    import json
    arm = tmp_path/'arm'
    helper = arm/'runtime/src/vllm_kvmem/capture_compile.py'
    helper.parent.mkdir(parents=True)
    helper.write_text("CAPTURE_COMPILE_KEY = 'capture_key'\n"
                      "def capture_compile_fingerprint():\n    return 'f'*64\n")
    launch = arm/'run/server/launch.json'
    launch.parent.mkdir(parents=True)
    profile = dict(kvmem=dict(capture_opaque_projection=True))
    progress = dict(capture_opaque_stats=dict(source_fingerprint='f'*64))
    argv = ['serve', 'model', '--additional-config', json.dumps({'capture_key': 'f'*64})]
    variant.save(launch, dict(vllm_argv=argv))
    assert variant.audit_opaque_compile_source(arm, profile, progress) == dict(
        source_fingerprint='f'*64, native_compilation_key_verified=True)
    assert variant.audit_opaque_compile_source(arm, dict(kvmem={}), {}) is None
    with pytest.raises(ValueError, match='frozen source'):
        variant.audit_opaque_compile_source(arm, profile,
            dict(capture_opaque_stats=dict(source_fingerprint='a'*64)))
    for bad in (argv[:2], argv+argv[2:], argv[:3]+[json.dumps({'capture_key': 'a'*64})]):
        variant.save(launch, dict(vllm_argv=bad))
        with pytest.raises(ValueError, match='compilation key'):
            variant.audit_opaque_compile_source(arm, profile, progress)


def test_assembly_prefetch_requires_completed_reads_and_no_abandoned_work():
    profile = dict(kvmem=dict(assembly_read_prefetch=True))
    stats = dict(assembly_prefetch_submitted=7, assembly_prefetch_completed=7,
                 assembly_prefetch_eligible=8)
    row = dict(assembly_read_prefetch=True, assembly_read_stats=stats)
    assert variant.audit_assembly_prefetch(profile, [row])['completed'] == 7
    for updates in ({'assembly_prefetch_completed': 6}, {'assembly_prefetch_abandoned': 1},
                    {'assembly_prefetch_slot_busy': -1}, {'assembly_prefetch_submitted': 0}):
        with pytest.raises(ValueError, match='Assembly prefetch|assembly prefetch'):
            variant.audit_assembly_prefetch(profile,
                [{**row, 'assembly_read_stats': {**stats, **updates}}])
    with pytest.raises(ValueError, match='no read was prefetched'):
        variant.audit_assembly_prefetch(profile, [dict(assembly_read_prefetch=True)])
    with pytest.raises(ValueError, match='Actual assembly read prefetch'):
        variant.audit_assembly_prefetch(profile, [dict(assembly_read_prefetch=False)])
    reference = dict(kvmem={})
    assert variant.audit_assembly_prefetch(reference, []) == {}
    assert variant.audit_assembly_prefetch(reference,
        [dict(assembly_read_prefetch=False)])['completed'] == 0


def test_assembly_fusion_requires_execution_and_reports_capacity_without_summing():
    profile = dict(kvmem=dict(assembly_fused_layers=True))
    stats = dict(assembly_fused_chunks=8, assembly_fused_layer_launches=136,
                 assembly_fused_metadata_bytes=3264, assembly_fused_fallback_chunks=1,
                 **{'assembly_fused_fallback:partial': 1})
    row = dict(assembly_fused_layers=True, assembly_read_stats=stats)
    report = variant.audit_assembly_fusion(profile, [row, row])
    assert report['chunks'] == 16 and report['layer_launches'] == 272
    assert report['max_metadata_bytes'] == 3264
    assert report['fallback_reasons'] == {'partial': 2}
    for updates in ({'assembly_fused_metadata_bytes': 0}, {'assembly_fused_layer_launches': 1},
                    {'assembly_fused_chunks': -1}, {'assembly_fused_fallback_chunks': True}):
        with pytest.raises(ValueError):
            variant.audit_assembly_fusion(profile,
                [{**row, 'assembly_read_stats': {**stats, **updates}}])
    with pytest.raises(ValueError, match='never executed'):
        variant.audit_assembly_fusion(profile, [dict(assembly_fused_layers=True)])
    with pytest.raises(ValueError, match='Disabled'):
        variant.audit_assembly_fusion(dict(kvmem=dict(assembly_fused_layers=False)),
            [{**row, 'assembly_fused_layers': False}])
    assert variant.audit_assembly_fusion(dict(kvmem={}), []) == {}


def baseline_fixture(tmp_path):
    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    suite = dict(cases=[], serial=True, prefix_caching=False)
    for ident, count, warmup in [('warmup-32k', 32768, True), ('32k-repeat-1', 32768, False),
            ('32k-repeat-2', 32768, False), ('256k-repeat-1', 262045, False), ('256k-repeat-2', 262045, False)]:
        source = tmp_path / (str(count) + '.i32')
        if not source.exists():
            source.write_bytes(b'\x01\x00\x00\x00' * count)
        directory = baseline / 'requests' / ident
        directory.mkdir(parents=True)
        variant.save(directory / 'case.json', dict(input_path=str(source),
            row=dict(served_input_tokens=count, served_sha256=variant.sha(source))))
        suite['cases'].append(dict(id=ident, warmup=warmup, case_sha256=variant.sha(directory / 'case.json')))
    variant.save(baseline / 'suite.json', suite)
    return baseline


def test_complete_cases_cannot_be_silently_truncated_or_replaced(tmp_path):
    baseline = baseline_fixture(tmp_path)
    assert len(variant.checked_suite(baseline)['cases']) == 5
    source = tmp_path / '262045.i32'
    source.write_bytes(source.read_bytes()[:-4])
    with pytest.raises(ValueError, match='Original input changed'):
        variant.checked_suite(baseline)


@pytest.mark.parametrize('seeded', [False, True])
def test_prepare_copies_latest_same_runtime_and_keeps_exact_case_hashes(tmp_path, monkeypatch, seeded):
    baseline = baseline_fixture(tmp_path)
    repo = tmp_path / 'repo'
    runtime_file = repo / 'src/vllm_kvmem/changed.py'
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('latest = True\n')
    driver = repo / 'scripts/benchmark_prefill_variant.py'
    driver.parent.mkdir()
    driver.write_text('frozen = True\n')
    variant.save(baseline / 'runtime-proof.json', dict(files={'src/vllm_kvmem/changed.py': 'older-hash'}))
    variant.save(baseline / 'profile.json', dict(kvmem=dict(prefill_native_plan_arm=False)))
    monkeypatch.setattr(variant, 'validate_profile', lambda profile: None)
    monkeypatch.setattr(variant.pipeline, 'runtime_origins', lambda *args:
        ({'src/vllm_kvmem/changed.py': runtime_file, 'scripts/benchmark_prefill_variant.py': driver}, {}))
    monkeypatch.setattr(variant.subprocess, 'check_output', lambda *args, **kwargs: 'abc123\n')
    root = tmp_path / 'paired'
    seed = tmp_path / 'autotune.json' if seeded else None
    if seed is not None:
        variant.save(seed, {'_metadata': {'version': 'fixed'}, 'operation': {'tactic': 4}})
    variant.prepare(root, baseline, repo, {'kvmem': {'assembly_chunk_blocks': 128}},
                    {'kvmem': {'assembly_chunk_blocks': 32}}, seed, ['a'*64] if seeded else ())
    plan = variant.assert_pair(root)
    assert plan['profile_diff_between_arms'] == [dict(path='kvmem.assembly_chunk_blocks',
        before_present=True, after_present=True, before=32, after=128)]
    assert variant.read(root / 'reference/runtime-proof.json') == variant.read(root / 'candidate/runtime-proof.json')
    assert variant.sha(root / 'candidate/suite.json') == variant.sha(baseline / 'suite.json')
    assert any(change['path'] == 'src/vllm_kvmem/changed.py' for change in plan['runtime_diff_from_baseline'])
    if seeded:
        assert plan['flashinfer_autotune_seed_sha256'] == variant.sha(seed)
        assert variant.sha(root / 'reference/flashinfer_seed/seed.json') == variant.sha(seed)
        # A consistent replacement of an arm's local chain must still fail
        # the parent plan, even when the seed helper validates that arm alone.
        from prefill_autotune_control import freeze_seed
        import shutil
        arm = root / 'candidate'
        original_manifest = variant.read(arm / 'manifest.json')
        original_manifest.pop('flashinfer_seed_manifest_sha256')
        variant.save(arm / 'manifest.json', original_manifest)
        shutil.rmtree(arm / 'flashinfer_seed')
        variant.save(seed, {'_metadata': {'version': 'fixed'}, 'operation': {'tactic': 8}})
        freeze_seed(arm, seed, ['a'*64])
        with pytest.raises(ValueError, match='Autotune seed differs from frozen plan'):
            variant.assert_pair(root)
    candidate = root / 'candidate/runtime/src/vllm_kvmem/changed.py'
    candidate.write_text('unrecorded = True\n')
    with pytest.raises(ValueError, match='Frozen runtime changed'):
        variant.assert_pair(root)


def inductor_source_fixture(tmp_path):
    """Exact tiny source/cache pairs with a pure-host key resolver."""
    import prefill_inductor_control as control

    def resolver(source):
        return source.parent / (control._sha(source.name.encode()) + '.best_config')

    sources = {}
    for role, source_hash, block in [('target', 'a' * 64, 512), ('mtp', 'b' * 64, 256)]:
        cache = tmp_path / role / 'torch_aot_compile' / source_hash / 'inductor_cache'
        source = cache / 'aa' / 'kernel.py'
        source.parent.mkdir(parents=True)
        source.write_text("@triton_heuristics.reduction(filename=__file__, "
                          "inductor_meta={'kernel_name':'kernel','autotune_local_cache':True})\n"
                          "@triton.jit\ndef kernel(x):\n    return x + 1\n")
        config = resolver(source)
        variant.save(config, dict(XBLOCK=block, num_warps=8, num_stages=1,
            configs_hash='e' * 64, found_by_coordesc=False))
        log = tmp_path / role / 'source.log'
        log.write_text('saved AOT compiled function to ' + str(cache.parent / 'rank_0_0/model') + '\n')
        sources[role] = dict(cache_dir=str(cache), server_log=str(log),
                             required_common=[str(config.relative_to(cache))])
    return sources, {'target': ['c' * 64], 'mtp': ['d' * 64]}, resolver


def test_prepare_binds_inductor_seed_to_both_arm_manifests_and_parent_plan(tmp_path, monkeypatch):
    import prefill_inductor_control as control

    baseline = baseline_fixture(tmp_path)
    repo = tmp_path / 'repo'
    runtime_file = repo / 'src/vllm_kvmem/changed.py'
    runtime_file.parent.mkdir(parents=True)
    runtime_file.write_text('latest = True\n')
    driver = repo / 'scripts/benchmark_prefill_variant.py'
    driver.parent.mkdir()
    driver.write_text('frozen = True\n')
    variant.save(baseline / 'runtime-proof.json', dict(files={'src/vllm_kvmem/changed.py': 'older-hash'}))
    variant.save(baseline / 'profile.json', dict(kvmem=dict(prefill_native_plan_arm=True)))
    monkeypatch.setattr(variant, 'validate_profile', lambda profile: None)
    requested_names = []

    def origins(_repo, _baseline, names):
        requested_names.extend(names)
        return {'src/vllm_kvmem/changed.py': runtime_file,
                'scripts/benchmark_prefill_variant.py': driver}, {}

    monkeypatch.setattr(variant.pipeline, 'runtime_origins', origins)
    monkeypatch.setattr(variant.subprocess, 'check_output', lambda *args, **kwargs: 'abc123\n')
    sources, allowed, resolver = inductor_source_fixture(tmp_path / 'sources')
    monkeypatch.setattr(control, '_installed_key_for_source', resolver)
    root = tmp_path / 'paired-inductor'
    variant.prepare(root, baseline, repo, {'kvmem': {'capture_opaque_projection': True}},
                    {'kvmem': {'capture_opaque_projection': False}},
                    inductor_seed_sources=sources, inductor_seed_hashes=allowed)
    plan = variant.assert_pair(root)
    assert plan['flashinfer_autotune_seed_sha256'] is None
    assert {'scripts/prefill_inductor_control.py', 'scripts/prefill_inductor_worker.py',
            'scripts/prefill_inductor_serve.py'} <= set(requested_names)
    chains = plan['inductor_seed_manifest_sha256']
    assert chains['reference'] == chains['candidate']
    for name in ('reference', 'candidate'):
        arm = root / name
        manifest = control.validate_seed(arm)
        assert set(manifest['modules']) == {'target', 'mtp'}
        assert variant.read(arm / 'manifest.json')[control.CHAIN_KEY] == chains[name]
        assert variant.assert_runtime(arm) == variant.read(arm / 'runtime-proof.json')['files']
        assert variant.read(arm / 'profile.json')['kvmem']['prefill_native_plan_arm'] is True
    # A self-consistent replacement passes the child chain but must still
    # fail the original paired experiment plan.
    import shutil
    arm = root / 'candidate'
    manifest = variant.read(arm / 'manifest.json')
    manifest.pop(control.CHAIN_KEY)
    variant.save(arm / 'manifest.json', manifest)
    shutil.rmtree(arm / control.DIRECTORY)
    source_cache = Path(sources['target']['cache_dir'])
    config_path = next(source_cache.glob('*/*.best_config'))
    config = variant.read(config_path)
    config['XBLOCK'] = 1024
    variant.save(config_path, config)
    control.freeze_seed(arm, sources, allowed)
    control.validate_seed(arm)
    with pytest.raises(ValueError, match='Inductor seed differs from frozen plan'):
        variant.assert_pair(root)


@pytest.mark.parametrize('sources,hashes', [({'target': {}}, None), (None, {'target': ['a' * 64]})])
def test_inductor_prepare_requires_both_source_and_hash_map_before_writes(tmp_path, sources, hashes):
    root = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='Inductor sources.*together'):
        variant.prepare(root, tmp_path / 'unused-baseline', tmp_path / 'unused-repo', {}, {},
                        inductor_seed_sources=sources, inductor_seed_hashes=hashes)
    assert not root.exists()


def inductor_audit_fixture():
    def entry(key):
        return dict(key=key, source_sha256='a' * 64, function_source_sha256='b' * 64,
                    configs_hash='c' * 64, selected_config={'XBLOCK': 512, 'num_warps': 8},
                    launch_verified=True)
    return dict(seed_manifest_sha256='f' * 64, launch_verified=True,
                common_configurations_unchanged=True, formal_dispatch_observer=False,
                modules={role: dict(common_kernels=[entry(role + '-shared')],
                                    actual_aot_hashes=['d' * 64]) for role in ('target', 'mtp')})


def test_inductor_pair_preserves_legacy_and_allows_distinct_actual_graph_hashes():
    variant.audit_inductor_pair([None, None])
    reference = inductor_audit_fixture()
    candidate = deepcopy(reference)
    candidate['modules']['target']['actual_aot_hashes'] = ['e' * 64]
    variant.audit_inductor_pair([reference, candidate])
    assert reference['modules']['target']['actual_aot_hashes'] == ['d' * 64]


@pytest.mark.parametrize('change', ['one-arm', 'manifest', 'unverified', 'config-unverified',
    'formal-hook', 'module-role', 'no-common', 'source', 'function', 'configs-hash', 'selected'])
def test_inductor_pair_rejects_uncontrolled_or_unverified_launches(change):
    reference = inductor_audit_fixture()
    candidate = deepcopy(reference)
    if change == 'one-arm':
        candidate = None
    elif change == 'manifest':
        candidate['seed_manifest_sha256'] = 'a' * 64
    elif change == 'unverified':
        candidate['launch_verified'] = False
    elif change == 'config-unverified':
        candidate['common_configurations_unchanged'] = False
    elif change == 'formal-hook':
        candidate['formal_dispatch_observer'] = True
    elif change == 'module-role':
        candidate['modules'].pop('mtp')
    elif change == 'no-common':
        candidate['modules']['target']['common_kernels'][0]['key'] = 'other-kernel'
    else:
        key = {'source': 'source_sha256', 'function': 'function_source_sha256',
               'configs-hash': 'configs_hash', 'selected': 'selected_config'}[change]
        candidate['modules']['target']['common_kernels'][0][key] = (
            {'XBLOCK': 1024, 'num_warps': 8} if change == 'selected' else '0' * 64)
    with pytest.raises(ValueError, match='Inductor'):
        variant.audit_inductor_pair([reference, candidate])


def test_prepare_cli_passes_module_seed_maps_without_rewriting(monkeypatch, tmp_path):
    import json
    captured = []
    sources = {'target': {'cache_dir': '/fixture/source', 'server_log': '/fixture/source.log'}}
    hashes = {'target': ['a' * 64]}
    monkeypatch.setattr(sys, 'argv', ['benchmark_prefill_variant.py', 'prepare',
        '--root', str(tmp_path / 'out'), '--baseline', str(tmp_path / 'base'),
        '--repo', str(tmp_path / 'repo'), '--overrides', '{"kvmem":{"capture_opaque_projection":true}}',
        '--inductor-seed-sources', json.dumps(sources), '--inductor-seed-hashes', json.dumps(hashes)])
    monkeypatch.setattr(variant, 'prepare', lambda *args: captured.append(args))
    variant.main()
    assert len(captured) == 1
    assert captured[0][0:3] == (tmp_path / 'out', tmp_path / 'base', tmp_path / 'repo')
    assert captured[0][7:] == (sources, hashes)


def test_profile_guard_allows_boolean_plan_arm_and_preserves_other_constraints(monkeypatch):
    import benchmark_matched_prefill
    monkeypatch.setattr(benchmark_matched_prefill, 'validate_diagnostic_profile', lambda profile: None)
    profile = dict(kvmem=dict(select_budget=32768, action_native_scores='reuse',
                             prefill_publication_pipeline=True, prefill_native_plan_arm=False))
    variant.validate_profile(profile)
    variant.validate_profile(variant.merge(profile, {'kvmem': {'prefill_native_plan_arm': True}}))
    for invalid in (1, None, 'true'):
        with pytest.raises(ValueError, match='must be boolean'):
            variant.validate_profile(variant.merge(profile, {'kvmem': {'prefill_native_plan_arm': invalid}}))
    for key, value in [('select_budget', 36864), ('action_native_scores', 'off'),
                       ('prefill_publication_pipeline', False)]:
        with pytest.raises(ValueError, match='Require K32'):
            variant.validate_profile(variant.merge(profile, {'kvmem': {key: value}}))
    with pytest.raises(ValueError, match='chunk capacity'):
        variant.validate_profile(variant.merge(profile, {'kvmem': {'assembly_chunk_blocks': True}}))


def test_comparison_cannot_report_success_for_zero_or_partial_cases(tmp_path, monkeypatch):
    base, candidate = tmp_path / 'base', tmp_path / 'candidate'
    for arm in (base, candidate):
        arm.mkdir()
        variant.save(arm / 'profile.json', {})
        variant.save(arm / 'suite.json', dict(cases=[]))
        variant.save(arm / 'status.json', dict(status='testing'))
    monkeypatch.setattr(variant, 'assert_runtime', lambda arm: {})
    monkeypatch.setattr(variant, 'validate_profile', lambda profile: None)
    with pytest.raises(ValueError, match='every formal case completed'):
        variant.compare(base, candidate, {})


def test_actual_candidate_branch_counters_are_required_not_just_profile_flags():
    profile = dict(kvmem=dict(prefill_capture_ahead=True, cpu_seal_hash_workers=8, cpu_direct_record_seal=True))
    progress = dict(prefill_capture_ahead=True, prefill_pipeline_stats=dict(capture_ahead_batches=12))
    archive = dict(archive_io_pending_bytes=0, cpu_seal_hash_workers=8, seal_batches=5,
        seal_hash_parallel_batches_completed=4, seal_hash_parallel_shards_completed=32, seal_direct_bytes=4096)
    assert variant.audit_optional_branches(profile, progress, archive)['seal_hash']['parallel_shards_completed'] == 32
    with pytest.raises(ValueError, match='execution'):
        variant.audit_optional_branches(profile, {**progress, 'prefill_pipeline_stats': {}}, archive)
    with pytest.raises(ValueError, match='bounded shard'):
        variant.audit_optional_branches(profile, progress, {**archive, 'seal_hash_parallel_shards_completed': 33})
    with pytest.raises(ValueError, match='no direct bytes'):
        variant.audit_optional_branches(profile, progress, {**archive, 'seal_direct_bytes': 0})


def test_actual_reference_disabled_branches_and_old_artifacts_are_supported():
    profile = dict(kvmem=dict(prefill_capture_ahead=False, cpu_seal_hash_workers=1, cpu_direct_record_seal=False))
    progress = dict(prefill_capture_ahead=False, prefill_pipeline_stats={})
    archive = dict(archive_io_pending_bytes=0, cpu_seal_hash_workers=1, seal_batches=5,
        seal_hash_parallel_batches_completed=0, seal_hash_parallel_shards_completed=0, seal_direct_bytes=0)
    variant.audit_optional_branches(profile, progress, archive)
    with pytest.raises(ValueError, match='Disabled parallel hash'):
        variant.audit_optional_branches(profile, progress, {**archive, 'seal_hash_parallel_shards_completed': 1})
    with pytest.raises(ValueError, match='Actual capture ahead'):
        variant.audit_optional_branches(profile, {}, archive)
    with pytest.raises(ValueError, match='Disabled direct seal'):
        variant.audit_optional_branches(profile, progress, {**archive, 'seal_direct_bytes': 4})
    assert variant.audit_optional_branches(dict(kvmem={}), {}, dict(archive_io_pending_bytes=0)) == {}
    # A separate native payload path is not mistaken for the Python direct-seal flag.
    variant.audit_optional_branches(profile, progress, {**archive, 'cpu_native_payloads': True, 'seal_direct_bytes': 4})


def test_fused_pack_comparison_requires_actual_bytes_and_explained_fallbacks():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(capture_fused_pack=True))
    progress = dict(capture_fused_pack=True, capture_pack_metadata_capacity_bytes=1088,
        capture_pack_stats=dict(fused_batches=140, fused_payload_bytes=4096, reference_batches=1,
                               **{'fallback:split_or_reordered_owners': 1}))
    actual = variant.audit_optional_branches(profile, progress, archive)['capture_pack']
    assert actual['fused_batches'] == 140 and actual['fallback_reasons'] == {'split_or_reordered_owners': 1}
    with pytest.raises(ValueError, match='no fused payload'):
        variant.audit_optional_branches(profile, {**progress, 'capture_pack_stats': {}}, archive)
    with pytest.raises(ValueError, match='lacks fallback reasons'):
        variant.audit_optional_branches(profile, {**progress, 'capture_pack_stats':
            dict(fused_batches=1, fused_payload_bytes=100, reference_batches=1)}, archive)
    reference = dict(capture_fused_pack=False, capture_pack_metadata_capacity_bytes=0,
                     capture_pack_stats=dict(reference_batches=141))
    variant.audit_optional_branches(dict(kvmem=dict(capture_fused_pack=False)), reference, archive)
    with pytest.raises(ValueError, match='Disabled fused'):
        variant.audit_optional_branches(dict(kvmem=dict(capture_fused_pack=False)),
            {**reference, 'capture_pack_metadata_capacity_bytes': 1088}, archive)


def test_early_mtp_action_requires_completed_frames_and_no_silent_fallback():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(prefill_action_before_mtp=True))
    progress = dict(prefill_action_before_mtp=True,
        prefill_action_before_mtp_stats=dict(staged_batches=128, resolved_batches=128))
    assert variant.audit_optional_branches(profile, progress, archive)['action_before_mtp']['resolved_batches'] == 128
    for key in ('fallback_batches', 'cancelled_batches', 'quarantined_frames'):
        with pytest.raises(ValueError, match='fallback/cancel/quarantine'):
            variant.audit_optional_branches(profile, {**progress,
                'prefill_action_before_mtp_stats': {**progress['prefill_action_before_mtp_stats'], key: 1}}, archive)
    with pytest.raises(ValueError, match='Unresolved'):
        variant.audit_optional_branches(profile, {**progress,
            'prefill_action_before_mtp_stats': dict(staged_batches=128, resolved_batches=127)}, archive)
    reference = dict(prefill_action_before_mtp=False, prefill_action_before_mtp_stats={})
    variant.audit_optional_branches(dict(kvmem={}), reference, archive)
    with pytest.raises(ValueError, match='execution differs'):
        variant.audit_optional_branches(profile, {**progress, 'prefill_action_before_mtp_stats': {}}, archive)


def test_native_prefill_plan_requires_real_prepare_counts_and_supports_old_callers():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(prefill_native_plan_arm=True))
    native = dict(layer_phase_counts={'prepare:prefill_plan': 140, 'prepare:eager': 12})
    result = variant.audit_optional_branches(profile, {}, archive, native)
    assert result['native_prefill_plan'] == dict(enabled=True, planned_batches=140, eager_batches=12)
    for report in (None, {}):
        with pytest.raises(ValueError, match='execution evidence'):
            variant.audit_optional_branches(profile, {}, archive, report)
    with pytest.raises(ValueError, match='execution differs'):
        variant.audit_optional_branches(profile, {}, archive, dict(layer_phase_counts={'prepare:eager': 140}))
    reference = dict(kvmem=dict(prefill_native_plan_arm=False))
    assert variant.audit_optional_branches(reference, {}, archive) == {}  # Legacy helper call.
    variant.audit_optional_branches(reference, {}, archive, dict(layer_phase_counts={'prepare:eager': 140}))
    with pytest.raises(ValueError, match='execution differs'):
        variant.audit_optional_branches(reference, {}, archive, native)
    with pytest.raises(ValueError, match='preparation evidence'):
        variant.audit_optional_branches(reference, {}, archive, dict(layer_phase_counts={}))


def test_aligned_native_prefill_counts_preserve_their_host_and_graph_scope():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(action_native_prefill_aligned_pieces=True))
    report = dict(enabled=True, counts=dict(aligned_candidate_owners=12,
        aligned_candidate_rows=768, prefill_dispatch_calls=36),
        candidate_counts_are_host_qualification=True, dispatch_counts_exclude_graph_replays=True)
    native = dict(layer_phase_counts={'prepare:eager': 12}, prefill_aligned_pieces=report)
    result = variant.audit_optional_branches(profile, {}, archive, native)
    assert result['native_prefill_aligned_pieces'] == report
    for bad in ({'counts': {}}, {'dispatch_counts_exclude_graph_replays': False}):
        with pytest.raises(ValueError, match='Aligned native prefill'):
            variant.audit_optional_branches(profile, {}, archive,
                {**native, 'prefill_aligned_pieces': {**report, **bad}})
    reference = {**native, 'prefill_aligned_pieces': {**report, 'enabled': False, 'counts': {}}}
    variant.audit_optional_branches(dict(kvmem={}), {}, archive, reference)
    with pytest.raises(ValueError, match='Disabled aligned'):
        variant.audit_optional_branches(dict(kvmem={}), {}, archive,
            {**native, 'prefill_aligned_pieces': {**report, 'enabled': False}})


def test_scatter_after_target_requires_actual_closed_lifecycle():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(prefill_scatter_after_target=True))
    counts = dict(deferred_batches=12, after_target_started_batches=9,
                  resumed_before_target_batches=2, no_work_batches=1,
                  already_running_batches=4, eligibility_declines=5)
    progress = dict(prefill_scatter_after_target=True,
                    prefill_scatter_after_target_stats=counts)
    result = variant.audit_optional_branches(profile, progress, archive)
    assert result['scatter_after_target']['after_target_started_batches'] == 9
    for bad in ({'quarantined_batches': 1}, {'deferred_batches': 13}):
        with pytest.raises(ValueError, match='lifecycle'):
            variant.audit_optional_branches(profile, {**progress,
                'prefill_scatter_after_target_stats': {**counts, **bad}}, archive)
    with pytest.raises(ValueError, match='execution differs'):
        variant.audit_optional_branches(profile,
            {**progress, 'prefill_scatter_after_target_stats': {}}, archive)
    reference = dict(prefill_scatter_after_target=False, prefill_scatter_after_target_stats={})
    variant.audit_optional_branches(dict(kvmem={}), reference, archive)
    with pytest.raises(ValueError, match='execution differs'):
        variant.audit_optional_branches(dict(kvmem={}), {**reference,
            'prefill_scatter_after_target_stats': {'already_running_batches': 1}}, archive)
    with pytest.raises(ValueError, match='mode differs'):
        variant.audit_optional_branches(profile, reference, archive)


def test_deferred_validation_requires_resolved_bounded_and_released_snapshots():
    archive = dict(archive_io_pending_bytes=0)
    profile = dict(kvmem=dict(prefill_deferred_validation=True))
    pool = dict(capacity=2, allocated_slots=2, active_slots=0, disabled_reason=None,
        quarantined=False, resident_bytes=100, max_bytes=100, scratch_bytes=200,
        max_scratch_bytes=200, counts=dict(frozen_frames=128, released_frames=128))
    progress = dict(prefill_deferred_validation=True,
        prefill_action_before_mtp_stats=dict(deferred_batches=128, deferred_resolved_batches=128),
        prefill_deferred_validation_pool=pool)
    assert variant.audit_optional_branches(profile, progress, archive)['deferred_validation']['completed'] == 128
    for bad in ({'active_slots': 1}, {'quarantined': True}, {'resident_bytes': 101},
                {'scratch_bytes': 201}, {'counts': dict(frozen_frames=128, released_frames=127)}):
        with pytest.raises(ValueError, match='Deferred validation'):
            variant.audit_optional_branches(profile,
                {**progress, 'prefill_deferred_validation_pool': {**pool, **bad}}, archive)
    for key, bad in [('deferred_resolved_batches', 127), ('deferred_fallback_batches', 1)]:
        with pytest.raises(ValueError, match='incomplete or used reference'):
            variant.audit_optional_branches(profile, {**progress, 'prefill_action_before_mtp_stats':
                {**progress['prefill_action_before_mtp_stats'], key: bad}}, archive)
