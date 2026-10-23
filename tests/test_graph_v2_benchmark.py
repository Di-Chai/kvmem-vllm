"""CPU-only contracts for V2 smoke qualification and frozen full campaigns."""
from copy import deepcopy
import importlib
import fcntl
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    protocol = importlib.import_module('graph_v2_protocol')
    controller = importlib.import_module('benchmark_graph_v2')
    client = importlib.import_module('qualify_graph_v2')
finally:
    sys.path.pop(0)


def base():
    return dict(model_format='safetensors', model='model', hf_config_path='model', tokenizer='tokenizer',
        execution_mode='compiled_graph', draft_method='mtp', draft_tokens=4,
        kvmem=dict(mean_k_backend='qw3_cuda', archive_backend='cpu', request_planning=True,
                   async_archive_retirement=True))


@pytest.mark.parametrize('suite', protocol.SUITES)
def test_all_arms_share_frozen_runtime_semantics_except_declared_policies(suite):
    profiles = protocol.profiles(base(), 1234, suite)
    protocol.validate_profiles(profiles, suite)
    for arm, profile in profiles.items():
        kv = profile['kvmem']
        assert (kv['graph_mode'], kv['pressure_policy']) == protocol.ARM_POLICY[arm]
        assert kv['graph_scoring_version'] == 'v2' and kv['graph_alpha'] == .2
        assert kv['action_fraction'] == 0 and not kv['query_history']
        assert profile['max_num_seqs'] == 8 and profile['max_num_batched_tokens'] == 16424
        assert not {'action_pressure', 'graph_locality_weight', 'graph_attention_weight'} & kv.keys()
        if suite == 'memoryagentbench':
            assert profile['max_model_len'] == 1048576 and kv['repository_blocks'] == 32768


@pytest.mark.parametrize('suite', protocol.SUITES)
def test_final_quality_profiles_preserve_the_selected_common_execution_settings(suite):
    selected = base()
    settings = dict(gpu_raw_k_cache_bytes=10*1024**3,
                    cpu_direct_record_seal=True, async_action_metadata=True)
    selected['kvmem'].update(settings)
    profiles = protocol.profiles(selected, 1234, suite)
    protocol.validate_profiles(profiles, suite)
    for profile in profiles.values():
        assert all(profile['kvmem'][key] == value for key, value in settings.items())
    assert selected['kvmem']['gpu_raw_k_cache_bytes'] == 10*1024**3


def reference_inputs(suite='memoryagentbench', length=757259):
    count = 1116 if suite == 'memoryagentbench' else 102
    rows = [dict(id=f'owner{i//100}_q{i:04d}',input_tokens=length,token_sha256=f'hash-{i}') for i in range(count)]
    sampling = {row['id']:dict(token_sha256=row['token_sha256'],temperature=.6,top_p=.95,top_k=20,
        min_p=0,presence_penalty=0,repetition_penalty=1,seed=100,max_tokens=40) for row in rows}
    return rows,sampling


def test_mab_complete_757k_history_fits_and_cpu_credits_remain_explicit():
    rows,sampling = reference_inputs()
    profile = protocol.profiles(base(),1234,'memoryagentbench')['D']
    report = protocol.preflight_reference(rows,sampling,profile,'memoryagentbench')
    assert report['cases'] == 1116 and report['independent_owners'] == 12
    assert report['maximum_owner_credit_bytes'] < report['cpu_payload_credits_bytes']
    assert report['largest_eight_fit_cpu_credits'] is False
    assert report['graph_per_owner_required_bytes'] == 256 * 1024**2
    assert report['graph_total_required_bytes'] == 2 * 1024**3


@pytest.mark.parametrize('limit', ['graph_max_bytes_per_session', 'graph_max_total_bytes'])
def test_mab_preflight_rejects_the_insufficient_cold_niah_graph_limits(limit):
    rows, sampling = reference_inputs()
    profile = protocol.profiles(base(), 1234, 'memoryagentbench')['A']
    profile['kvmem'][limit] = 33554432 if limit.endswith('session') else 268435456
    with pytest.raises(ValueError, match='Frozen-P graph capacity'):
        protocol.preflight_reference(rows, sampling, profile, 'memoryagentbench')


def test_one_million_token_graph_retained_prefix_and_staging_fit_mab_capacity():
    from vllm_kvmem.block_graph import BlockGraph, GraphSample, GraphBudget
    from vllm_kvmem.prompt_spans import install_graph_metadata
    per_owner, total = protocol.frozen_prefix_graph_capacity(32768, 8)
    pool = GraphBudget(total)
    graph = BlockGraph('mab', 32, budget=pool, max_bytes_per_session=per_owner)
    install_graph_metadata(graph, SimpleNamespace())
    end = 1048576 - 32
    samples = tuple(GraphSample(block * 32,
        tuple((block - i) * 32 for i in range(1, 5)), (.25,) * 4, 'prefill')
        for block in range(4, end // 32))
    with graph.prepare(0, end, samples) as update:
        update.commit()
    prefix, scoring = graph.snapshot(), graph.snapshot()
    graph.set_preserved_prefix(prefix, end)
    before = graph.binding, graph.bytes
    install_graph_metadata(graph, SimpleNamespace(), validated_prefix=end)
    assert (graph.binding, graph.bytes) == before
    with graph.prepare(end, end + 32,
            (GraphSample(end, tuple(end - i * 32 for i in range(1, 5)), (.25,) * 4, 'output'),)) as update:
        assert pool.used <= per_owner
        update.commit()
    assert graph.accepted_through == 1048576
    assert graph.bytes <= per_owner and prefix.accepted_through == end
    scoring.close()
    prefix.close()
    graph.close()
    assert pool.used == 0


@pytest.mark.parametrize('mutation', ['context','repository','cpu_credit','source_identity','output'])
def test_reference_preflight_rejects_full_history_or_sampling_drift(mutation):
    rows,sampling = reference_inputs()
    profile = protocol.profiles(base(),1234,'memoryagentbench')['D']
    if mutation == 'context': profile['max_model_len']=524288
    elif mutation == 'repository': profile['kvmem']['repository_blocks']=16384
    elif mutation == 'cpu_credit': profile['kvmem']['cpu_read_cache_bytes']=20*1024**3
    elif mutation == 'source_identity': sampling[rows[0]['id']]['token_sha256']='changed'
    else: sampling[rows[0]['id']]['max_tokens']=20000
    with pytest.raises(ValueError): protocol.preflight_reference(rows,sampling,profile,'memoryagentbench')


@pytest.mark.parametrize('mutation', ['budget', 'pressure', 'domain', 'capture', 'query_history', 'legacy_weights', 'pooling'])
def test_invalid_or_unfair_arms_are_rejected(mutation):
    profiles = protocol.profiles(base(), 1234)
    kv = profiles['D']['kvmem']
    if mutation == 'budget':
        profiles['C']['max_num_batched_tokens'] = 1024
    elif mutation == 'pressure': kv['pressure_policy'] = 'attention'
    elif mutation == 'domain': kv['graph_scoring_version'] = 'v1'
    elif mutation == 'capture': kv['prefill_attention'] = False
    elif mutation == 'query_history': kv['query_history'] = True
    elif mutation == 'pooling': kv['graph_aggregation'] = 'max'
    else: kv['graph_locality_weight'] = .1
    with pytest.raises(ValueError): protocol.validate_profiles(profiles)


@pytest.mark.parametrize('arm', protocol.ARMS)
def test_real_launch_audits_v2_worker_transport_and_actual_budgets(tmp_path, arm):
    from vllm_kvmem.serve import build_launch
    profile = protocol.profiles(base(), 1234)[arm]
    launch = build_launch(profile, tmp_path, tmp_path/'server', check_paths=False)
    (tmp_path/'server').mkdir()
    (tmp_path/'server/launch.json').write_text(json.dumps(launch))
    report = client.audit_launch(profile, tmp_path/'server/capture', arm)
    assert report['worker_policy']['pressure_policy'] == protocol.ARM_POLICY[arm][1]
    assert report['observed']['--max-num-batched-tokens'] == '16424'
    argv = launch['vllm_argv']
    index = argv.index('--kv-transfer-config')+1
    transport = json.loads(argv[index])
    transport['kv_connector_extra_config']['kvmem']['graph_alpha'] = .4
    argv[index] = json.dumps(transport)
    (tmp_path/'server/launch.json').write_text(json.dumps(launch))
    with pytest.raises(ValueError, match='Actual worker'):
        client.audit_launch(profile, tmp_path/'server/capture', arm)


def events(arm):
    profile = protocol.profiles(base(), 1234)[arm]
    common = dict(scoring_version='v2', score_domain_version=protocol.SCORE_DOMAIN,
        pressure_policy=profile['kvmem']['pressure_policy'], alpha=.2, selected_tokens=36864)
    query = dict(common, selection='query', phase='query', seed_source='mean_k', query_span=[100, 120],
        replay_span=[96, 120], mean_k_backend='qw3_cuda', query_history_enabled=False,
        guided_query_tokens=0, action_attention_enabled=True, action_history_pages=0,
        prefill_attention_samples=100, selected_history_ranges=[[0, 128]])
    pressure = dict(common, selection='pressure', phase='pressure', seed_source='prefill',
                    score_mask_count=100, transport_mask_count=100 if arm == 'D' else 0)
    graph_query = dict(common, phase='query', seed_source='mean_k', mode=protocol.ARM_POLICY[arm][0],
        observed_sources=100, edges=300, graph_live_bytes=1000, graph_global_bytes=2000,
        snapshot_bytes=100, accepted_through=96)
    graph = [graph_query]
    if arm == 'D':
        graph.append(dict(graph_query, phase='pressure', seed_source='prefill', candidate_count=100,
                          score_mask_count=100, transport_mask_count=100))
    else:
        graph.append(dict(common, phase='pressure', seed_source='prefill', candidate_count=100,
                          graph_live_bytes=1000, propagation_applied=False,
                          score_mask_count=100, transport_mask_count=0))
    return profile, [query, pressure], graph


@pytest.mark.parametrize('arm', protocol.ARMS)
def test_actual_selection_versions_seeds_phases_and_resource_evidence(arm):
    profile, refresh, graph = events(arm)
    report = protocol.audit_events(refresh, graph, profile, arm, [100, 120], require_pressure=True)
    assert report['phases'] == {'query': 1, 'pressure': 1}
    assert report['seed_sources'] == ['mean_k', 'prefill']


@pytest.mark.parametrize('arm', ['A', 'C'])
@pytest.mark.parametrize('mutation', ['propagated', 'missing_report', 'seed_mismatch', 'count_mismatch'])
def test_attention_pressure_requires_its_unpropagated_actual_report(arm, mutation):
    profile, refresh, graph = events(arm)
    if mutation == 'propagated': graph[1]['propagation_applied'] = True
    elif mutation == 'missing_report': graph.pop()
    elif mutation == 'seed_mismatch': graph[1]['seed_source'] = 'output'
    else: graph[1]['score_mask_count'] = 101
    with pytest.raises(ValueError):
        protocol.audit_events(refresh, graph, profile, arm, [100, 120], require_pressure=True)


@pytest.mark.parametrize('mutation', ['missing_pressure', 'wrong_seed', 'old_domain', 'missing_graph',
    'wrong_policy', 'stale', 'quota', 'fake_capture'])
def test_profile_alone_cannot_substitute_for_actual_v2_evidence(mutation):
    profile, refresh, graph = events('D')
    if mutation == 'missing_pressure': refresh.pop()
    elif mutation == 'wrong_seed': refresh[1]['seed_source'] = 'mean_k'
    elif mutation == 'old_domain': refresh[0]['score_domain_version'] = 'old'
    elif mutation == 'missing_graph': graph.pop()
    elif mutation == 'wrong_policy': refresh[1]['pressure_policy'] = 'attention'
    elif mutation == 'stale': graph[0]['accepted_through'] = 95
    elif mutation == 'quota': graph[0]['graph_global_bytes'] = 10**12
    else: graph[0]['edges'] = 0
    with pytest.raises(ValueError):
        protocol.audit_events(refresh, graph, profile, 'D', [100, 120], require_pressure=True)


def test_eight_admitted_clients_are_not_accepted_as_executed_c8():
    with pytest.raises(ValueError, match='executed eight-owner'):
        protocol.audit_actual_concurrency([dict(model_ms=1., prefill_requests=1, decode_requests=0)], require_eight=True)
    seven = protocol.audit_actual_concurrency([dict(model_ms=1., prefill_requests=7, decode_requests=0)])
    assert seven['native_eight_owner_qualified'] is False and seven['executed_request_peak'] == 7
    proof = protocol.audit_actual_concurrency([dict(model_ms=2., prefill_requests=5, decode_requests=3)])
    assert proof['executed_request_peak'] == 8 and proof['executed_eight_owner_steps'] == 1
    assert proof['pure_decode_rate_inferred'] is False
    with pytest.raises(ValueError, match='No completed'):
        protocol.audit_actual_concurrency([dict(model_ms=0., prefill_requests=8)])


@pytest.mark.parametrize('scope,lengths,depths,seeds,count', [
    ('smoke8', [262144], [10, 90], [1000], 8),
    ('full', [65536, 131072, 262144], [10, 30, 50, 70, 90], [1000, 2000, 3000], 180),
])
def test_prepare_selects_exact_original_cohorts_without_default_full_launch(tmp_path, monkeypatch,
        scope, lengths, depths, seeds, count):
    reference = tmp_path/'reference'
    reference.mkdir()
    (reference/'cohort.json').write_text('[]')
    (reference/'run.py').write_text('def score(text, expected):\n    return {}\n')
    tokenizer = tmp_path/'tokenizer'
    tokenizer.mkdir()
    (tokenizer/'tokenizer.json').write_text('{}')
    profile = protocol.profiles(base(), 1234)['A']
    profile['tokenizer'] = str(tokenizer)
    seen = {}
    def prepare(args, config):
        seen.update(lengths=args.lengths, depths=args.depths, seeds=args.seeds)
        return [dict(id=str(i), nominal_input_tokens=lengths[i % len(lengths)],
                     served_input_tokens=lengths[i % len(lengths)]) for i in range(count)]
    monkeypatch.setattr(controller.niah, 'prepare', prepare)
    fixture = controller.freeze_niah(SimpleNamespace(reference=reference, scope=scope), tmp_path, profile)
    assert seen == dict(lengths=lengths, depths=depths, seeds=seeds)
    assert len(fixture['rows']) == count and fixture['workers'] == 8
    assert fixture['gold_used_for_selection'] is False


def frozen_campaign(tmp_path):
    root = tmp_path/'campaign'
    root.mkdir()
    (root/'source.diff').write_text('frozen source patch')
    (root/'status.json').write_text(json.dumps(dict(status='prepared')))
    frozen = root/'frozen'
    frozen.mkdir()
    (frozen/'benchmark_graph_v2.py').write_text('# frozen controller\n')
    (root/'sample.i32').write_bytes(b'\x01\x00\x00\x00')
    fixture = root/'niah-fixture.json'
    fixture.write_text(json.dumps(dict(schema=protocol.SCHEMA, suite='niah')))
    jobs = []
    native_name = 'src/vllm_kvmem/_native.test.so'
    hashes = None
    for arm, profile in protocol.profiles(base(), 1234).items():
        target = root/('niah-'+arm)
        native = target/'runtime'/native_name
        native.parent.mkdir(parents=True)
        native.write_bytes(b'frozen-current-native')
        (target/'profile.json').write_text(json.dumps(profile))
        (target/'fixture.json').write_text(fixture.read_text())
        (target/'status.json').write_text(json.dumps(dict(status='prepared')))
        hashes = {native_name:protocol.sha(native)}
        manifest = dict(runtime_sha256=hashes, profile_sha256=protocol.sha(target/'profile.json'),
                        fixture_sha256=protocol.sha(target/'fixture.json'))
        (target/'manifest.json').write_text(json.dumps(manifest))
        jobs.append(dict(id=target.name, suite='niah', arm=arm, cases=8,
                         manifest_sha256=protocol.sha(target/'manifest.json')))
    manifest = dict(schema=protocol.SCHEMA, scope='smoke8', jobs=jobs, runtime_sha256=hashes,
        source_diff_sha256=protocol.sha(root/'source.diff'),
        frozen_sha256={'benchmark_graph_v2.py':protocol.sha(frozen/'benchmark_graph_v2.py')},
        input_sha256={'sample.i32':protocol.sha(root/'sample.i32')},
        fixture_sha256={'niah':protocol.sha(fixture)},
        model_identity=dict(config_sha256={}, tokenizer_sha256={}, weight_stat_identity={}))
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root


@pytest.mark.parametrize('relative', ['sample.i32', 'frozen/benchmark_graph_v2.py',
    'niah-A/profile.json', 'niah-C/runtime/src/vllm_kvmem/_native.test.so', 'niah-D/fixture.json'])
def test_frozen_sample_runtime_profile_or_controller_tampering_is_rejected(tmp_path, relative):
    root = frozen_campaign(tmp_path)
    assert controller.validate(root)['scope'] == 'smoke8'
    with (root/relative).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError): controller.validate(root)


@pytest.mark.parametrize('status', ['failed', 'running', 'loading', 'testing', 'completed'])
def test_failed_interrupted_or_completed_campaign_is_never_silently_retried(tmp_path, monkeypatch, status):
    root = frozen_campaign(tmp_path)
    (root/'status.json').write_text(json.dumps(dict(status=status)))
    monkeypatch.setattr(controller, '__file__', str(root/'frozen/benchmark_graph_v2.py'))
    with pytest.raises(ValueError, match='cannot be retried'):
        controller.run(SimpleNamespace(output=root, arms=['D'], suites=['niah']))


def test_failed_job_cannot_be_hidden_by_a_partial_campaign_status(tmp_path, monkeypatch):
    root = frozen_campaign(tmp_path)
    (root/'status.json').write_text(json.dumps(dict(status='partial')))
    (root/'niah-D/status.json').write_text(json.dumps(dict(status='failed')))
    monkeypatch.setattr(controller, '__file__', str(root/'frozen/benchmark_graph_v2.py'))
    with pytest.raises(ValueError, match='retained'):
        controller.run(SimpleNamespace(output=root, arms=['D'], suites=['niah']))


def test_concurrent_campaign_controllers_cannot_consume_same_prepared_attempt(tmp_path):
    root = frozen_campaign(tmp_path)
    with (root/'.campaign.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            controller.run(SimpleNamespace(output=root, arms=['D'], suites=['niah']))
    assert json.loads((root/'status.json').read_text())['status'] == 'prepared'


def test_prepare_cli_defaults_to_smoke8_without_invoking_gpu(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(controller, 'prepare', lambda args:seen.append(args.scope))
    controller.main(['prepare', '--output', str(tmp_path/'fresh')])
    controller.main(['prepare', '--output', str(tmp_path/'full'), '--scope', 'full'])
    assert seen == ['smoke8', 'full']


@pytest.mark.parametrize('suite,module_name', [
    ('memoryagentbench', 'benchmark_memory'), ('longmemeval_s', 'benchmark_longmemeval')])
@pytest.mark.parametrize('arm', protocol.ARMS)
def test_reference_clients_accept_v2_without_old_action_boolean_or_query_history(tmp_path, suite, module_name, arm):
    # Missing source is the intentional stopping point: this exercises each
    # real run entry's version gate without submitting any model request.
    sys.path.insert(0, str(SCRIPTS))
    try:
        module = importlib.import_module(module_name)
        profile = protocol.profiles(base(), 1234, suite)[arm]
        config = tmp_path/'profile.json'
        config.write_text(json.dumps(profile))
        with pytest.raises(FileNotFoundError, match='cohort.json'):
            module.run(SimpleNamespace(config=config, reference=tmp_path/'missing'))
    finally:
        sys.path.pop(0)
