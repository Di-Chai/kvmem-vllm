from copy import deepcopy

import pytest

from vllm_kvmem.serve import build_launch


def profile():
    return dict(model='model', tokenizer='tokenizer', hf_config_path='model',
        model_format='safetensors', draft_method='mtp', draft_tokens=4,
        max_num_seqs=1, prefill_chunk_tokens=512, query_replay_budget=512,
        recall_experiment=dict(method='mean_k', pressure_query_window=128, device='cuda',
                               query_batch=4, block_batch=16),
        kvmem=dict(block_tokens=32, select_budget=36864, gen_budget=16384,
            sink_blocks=41, recent_blocks=0, request_planning=True, async_refresh=True,
            graph_mode='off', graph_scoring_version='v2', graph_alpha=0., guided_query_tokens=0,
            query_history=False, query_score_mode='mean_k',
            query_fps_halo_blocks=0, query_mean_quota_fraction=0., query_fps_prebuild=False))


@pytest.mark.parametrize('method', ['mean_k', 'fixed4', 'fps4', 'min_max', 'exact_mass'])
def test_recall_profile_freezes_worker_environment_through_relaunch(tmp_path, method):
    config = profile()
    config['recall_experiment']['method'] = method
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert plan['environment']['KVMEM_RECALL_METHOD'] == method
    assert plan['environment']['KVMEM_PRESSURE_QUERY_WINDOW'] == '128'
    assert plan['environment']['KVMEM_RECALL_DEVICE'] == 'cuda'
    assert plan['environment']['KVMEM_RECALL_QUERY_BATCH'] == '4'
    assert plan['environment']['KVMEM_RECALL_BLOCK_BATCH'] == '16'
    assert plan['environment']['KVMEM_LOGICAL_KV_CAPACITY'] == '1'
    repeated = build_launch(plan['profile'], tmp_path, tmp_path/'run2', check_paths=False)
    for key, value in plan['environment'].items():
        if key.startswith('KVMEM_RECALL_') or key == 'KVMEM_PRESSURE_QUERY_WINDOW':
            assert repeated['environment'][key] == value
    config['recall_experiment']['method'] = 'changed-after-build'
    assert plan['profile']['recall_experiment']['method'] == method


@pytest.mark.parametrize('change', [dict(method='unknown'), dict(pressure_query_window=32),
    dict(pressure_query_window=True), dict(query_batch=0), dict(query_batch=129),
    dict(block_batch=0), dict(device='automatic'), dict(extra='silent')])
def test_recall_profile_rejects_unfrozen_settings(tmp_path, change):
    config = profile()
    config['recall_experiment'].update(change)
    with pytest.raises(ValueError):
        build_launch(config, tmp_path, tmp_path/'run', check_paths=False)


def test_recall_profile_is_opt_in(tmp_path, monkeypatch):
    config = profile()
    del config['recall_experiment']
    monkeypatch.setenv('KVMEM_RECALL_METHOD', 'fixed4')
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert 'KVMEM_RECALL_METHOD' not in plan['environment']
    assert 'KVMEM_LOGICAL_KV_CAPACITY' not in plan['environment']


def test_recall_profile_does_not_mutate_input(tmp_path):
    config = profile()
    original = deepcopy(config)
    build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert config == original


def test_optimized_profile_freezes_full_query_tile_and_fp32_matmul(tmp_path):
    config = profile()
    config['recall_experiment']['query_batch'] = 128
    plan = build_launch(config, tmp_path, tmp_path/'run', check_paths=False)
    assert plan['environment']['KVMEM_RECALL_QUERY_BATCH'] == '128'
    assert plan['environment']['VLLM_FLOAT32_MATMUL_PRECISION'] == 'highest'


@pytest.mark.parametrize('method',['exact_mass','block_value_norm','block_deletion_error'])
def test_value_reference_freezes_normalization_and_optimized_backend(tmp_path,method):
    config=profile();config['recall_experiment'].update(method=method,
        include_mandatory_in_softmax=True,value_backend='streaming',query_batch=128,block_batch=128)
    plan=build_launch(config,tmp_path,tmp_path/'run',check_paths=False)
    assert plan['environment']['KVMEM_RECALL_METHOD']==method
    assert plan['environment']['KVMEM_VALUE_RECALL_REFERENCE']=='1'
    assert plan['environment']['KVMEM_VALUE_RECALL_BACKEND']=='streaming'


@pytest.mark.parametrize('method',['block_value_norm','block_deletion_error'])
def test_value_method_rejects_missing_mandatory_normalization(tmp_path,method):
    config=profile();config['recall_experiment']['method']=method
    with pytest.raises(ValueError,match='mandatory'):build_launch(config,tmp_path,tmp_path/'run',check_paths=False)


def test_value_diagnostic_cli_survives_clean_worker_environment(tmp_path,monkeypatch,capsys):
    import json
    from vllm_kvmem import serve
    config=profile();config['recall_experiment']['include_mandatory_in_softmax']=True
    path=tmp_path/'profile.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(serve,'build_launch',lambda *args:dict(profile=config,environment={}))
    serve.main(['--config',str(path),'--run-dir',str(tmp_path/'run'),
        '--value-recall-diagnostics','query','--dry-run'])
    assert json.loads(capsys.readouterr().out)['environment']['KVMEM_VALUE_RECALL_DIAGNOSTICS']=='query'


def test_packed_value_backend_allows_wider_compute_tiles_without_changing_physical_blocks(tmp_path):
    config=profile();config['recall_experiment'].update(method='block_value_norm',
        include_mandatory_in_softmax=True,value_backend='streaming_packed',block_batch=1024)
    plan=build_launch(config,tmp_path,tmp_path/'packed',check_paths=False)
    assert plan['environment']['KVMEM_VALUE_RECALL_BACKEND']=='streaming_packed'
    assert plan['environment']['KVMEM_RECALL_BLOCK_BATCH']=='1024'
    assert plan['profile']['kvmem']['block_tokens']==32
    config['recall_experiment']['value_backend']='streaming'
    with pytest.raises(ValueError,match='geometry'):build_launch(config,tmp_path,tmp_path/'legacy',check_paths=False)


def test_region_value_backend_is_explicit_and_keeps_identical_kv_budgets(tmp_path):
    config = profile()
    config['recall_experiment'].update(method='block_deletion_error',
        include_mandatory_in_softmax=True, value_backend='streaming_region', block_batch=1024)
    plan = build_launch(config, tmp_path, tmp_path/'regions', check_paths=False)
    assert plan['environment']['KVMEM_VALUE_RECALL_BACKEND'] == 'streaming_region'
    assert plan['environment']['KVMEM_RECALL_BLOCK_BATCH'] == '1024'
    assert plan['profile']['kvmem']['select_budget'] == 36864
    assert plan['profile']['kvmem']['gen_budget'] == 16384
    assert plan['profile']['kvmem']['block_tokens'] == 32


def test_prefill_graph_sizes_survive_resolved_profile_and_cover_real_chunk(tmp_path):
    import json
    config = profile()
    config.update(execution_mode='compiled_graph', cudagraph_capture_sizes=[512, 1, 5, 64])
    plan = build_launch(config, tmp_path, tmp_path/'graphs', check_paths=False)
    argv = plan['vllm_argv']
    compilation = json.loads(argv[argv.index('--compilation-config')+1])
    assert compilation['cudagraph_capture_sizes'] == [1, 5, 64, 512]
    assert plan['profile']['prefill_chunk_tokens'] == 512
    assert plan['profile']['kvmem']['select_budget'] == 36864
    repeated = build_launch(plan['profile'], tmp_path, tmp_path/'again', check_paths=False)
    assert repeated['vllm_argv'][repeated['vllm_argv'].index('--compilation-config')+1] == argv[argv.index('--compilation-config')+1]


@pytest.mark.parametrize('sizes', [[], [0], [True], [518], [1, 1], '512'])
def test_invalid_graph_shapes_are_rejected_before_launch(tmp_path, sizes):
    config = profile()
    config.update(execution_mode='compiled_graph', cudagraph_capture_sizes=sizes)
    with pytest.raises(ValueError, match='capture sizes'):
        build_launch(config, tmp_path, tmp_path/'invalid', check_paths=False)


def test_eager_cannot_silently_ignore_requested_graph_shapes(tmp_path):
    config = profile()
    config['cudagraph_capture_sizes'] = [512]
    with pytest.raises(ValueError, match='require graph execution'):
        build_launch(config, tmp_path, tmp_path/'eager', check_paths=False)


@pytest.mark.parametrize('owners', [2, 4, 8])
def test_recall_parallelism_is_declared_and_cannot_change_kv_frame(tmp_path, owners):
    config = profile()
    config.update(max_num_seqs=owners, query_replay_budget=512*owners, max_num_batched_tokens=517*owners)
    config['recall_experiment']['concurrent_owners'] = owners
    plan = build_launch(config, tmp_path, tmp_path/'two', check_paths=False)
    assert plan['environment']['KVMEM_RECALL_CONCURRENT_OWNERS'] == str(owners)
    assert plan['profile']['max_num_seqs'] == owners
    assert plan['profile']['kvmem']['select_budget'] == 36864
    assert plan['profile']['kvmem']['gen_budget'] == 16384
    config['recall_experiment']['concurrent_owners'] = 1
    with pytest.raises(ValueError, match='concurrent owners'):
        build_launch(config, tmp_path, tmp_path/'mismatch', check_paths=False)


def test_value_profile_cli_is_explicit_and_survives_clean_worker_environment(tmp_path,monkeypatch,capsys):
    import json
    from vllm_kvmem import serve
    config=profile();config['recall_experiment'].update(value_backend='streaming_packed',
        include_mandatory_in_softmax=True,method='block_value_norm')
    path=tmp_path/'profile.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(serve,'build_launch',lambda *args:dict(profile=config,environment={}))
    serve.main(['--config',str(path),'--run-dir',str(tmp_path/'run'),
        '--value-recall-profile','query','--dry-run'])
    assert json.loads(capsys.readouterr().out)['environment']['KVMEM_VALUE_RECALL_PROFILE']=='query'
