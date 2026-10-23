"""Explicit independently selectable replay optimizations, default off."""
import json

import pytest

from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def test_defaults_preserve_the_original_private_replay_path():
    options = RuntimeOptions()
    assert options.query_replay_backend == 'eager' and options.query_replay_cobatch is False
    assert options.query_replay_max_bytes == 64*1024*1024


@pytest.mark.parametrize('backend', [None, True, 1, 'auto', 'cuda_graph'])
def test_unknown_compilation_modes_cannot_silently_select_a_path(backend):
    with pytest.raises(ValueError, match='Query replay backend'):
        RuntimeOptions(query_replay_backend=backend)


@pytest.mark.parametrize('value', [None, 1, 'false'])
def test_cobatch_requires_an_actual_boolean(value):
    with pytest.raises(ValueError, match='Query replay cobatch'):
        RuntimeOptions(query_replay_cobatch=value)


@pytest.mark.parametrize('value', [0, -1, True, '64M'])
def test_zero_or_unbounded_implicit_memory_is_rejected(value):
    with pytest.raises(ValueError, match='Query replay memory'):
        RuntimeOptions(query_replay_max_bytes=value)


@pytest.mark.parametrize('patch', [dict(query_replay_backend='compiled'),dict(query_replay_cobatch=True)])
def test_optimized_replay_requires_the_owned_asynchronous_lifecycle(patch):
    with pytest.raises(ValueError, match='requires asynchronous refresh'):
        RuntimeOptions(**patch)


@pytest.mark.parametrize('backend,cobatch', [('eager',False),('compiled',False),('eager',True),('compiled',True)])
def test_each_switch_round_trips_through_native_worker_configuration(tmp_path, backend, cobatch):
    from test_serving_launcher import nvfp4_profile
    cfg = nvfp4_profile(tmp_path)
    cfg.update(draft_method='mtp',draft_tokens=4,execution_mode='compiled_graph')
    cfg['kvmem'].update(request_planning=True,async_refresh=True,
        session_checkpoint_policy='pm',session_capture_terminal=True,
        query_replay_backend=backend,query_replay_cobatch=cobatch,query_replay_max_bytes=32768)
    plan = build_launch(cfg,tmp_path,tmp_path/'run')
    argv = plan['vllm_argv']
    observed=json.loads(argv[argv.index('--kv-transfer-config')+1])['kv_connector_extra_config']['kvmem']
    assert observed['query_replay_backend'] == backend and observed['query_replay_cobatch'] is cobatch
    assert observed['query_replay_max_bytes'] == 32768
    build_launch(plan['profile'],tmp_path,tmp_path/'replay')


@pytest.mark.parametrize('mode', ['eager','full_graph'])
def test_compiled_replay_requires_a_compiled_target_model(tmp_path, mode):
    from test_serving_launcher import nvfp4_profile
    cfg = nvfp4_profile(tmp_path)
    cfg.update(draft_method='mtp',draft_tokens=4,execution_mode=mode)
    cfg['kvmem'].update(request_planning=True,async_refresh=True,query_replay_backend='compiled',
                         session_checkpoint_policy='pm',session_capture_terminal=True)
    with pytest.raises(ValueError, match='requires compiled_graph'):
        build_launch(cfg,tmp_path,tmp_path/'run')
