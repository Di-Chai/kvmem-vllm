from copy import deepcopy

import pytest

from test_graph_options import model_profile, runtime, settings, worker_payload
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch


def payload(**changes):
    return settings(prefill_physical_coalescing=True,
        prefill_publication_pipeline=True, prefill_native_plan_arm=True,
        async_cpu_scatter=True, split_capture_dma=True, action_native_scores='reuse',
        batched_cpu_append=True,
        **changes)


def test_default_off_roundtrip_and_logical_chunk_unchanged(tmp_path):
    assert RuntimeOptions().prefill_physical_coalescing is False
    assert RuntimeOptions().prefill_physical_max_chunks == 2
    profile = model_profile(tmp_path)
    profile.update(prefill_chunk_tokens=2048)
    profile['kvmem'] = payload()
    before = deepcopy(profile)
    launch = build_launch(profile, tmp_path, tmp_path/'run')
    assert profile == before
    assert runtime(worker_payload(launch)).prefill_physical_coalescing is True
    assert launch['environment']['KVMEM_PREFILL_CHUNK_TOKENS'] == '2048'


@pytest.mark.parametrize('value', [0, 1, None, 'true', [], {}])
def test_flag_strict_boolean(value):
    with pytest.raises(ValueError, match='prefill_physical_coalescing must be boolean'):
        RuntimeOptions(prefill_physical_coalescing=value)


@pytest.mark.parametrize('key,value', [
    ('prefill_publication_pipeline', False), ('prefill_native_plan_arm', False),
    ('action_native_scores', 'off'), ('request_planning', False),
    ('early_capture_slot_release', True), ('prefill_deferred_validation', True),
    ('prefill_scatter_after_target', True), ('offline_prefill_coalescing', True),
    ('offline_phase_policy', True), ('prefill_batch_tokens', 4096),
    ('adaptive_prefill', True), ('guided_query_tokens', 128), ('action_native_max_rows', 64),
    ('batched_cpu_append', False), ('graph_mode', 'off'),
    ('action_download_pool', False), ('action_download_max_slots', 3),
])
def test_unqualified_path_rejected_before_start(key, value):
    p = payload()
    p[key] = value
    with pytest.raises(ValueError):
        runtime(p)


@pytest.mark.parametrize('chunk', [1024, 4096])
def test_physical_cap_cannot_change_logical_policy(tmp_path, chunk):
    profile = model_profile(tmp_path)
    profile['kvmem'] = payload()
    profile['prefill_chunk_tokens'] = chunk
    with pytest.raises(ValueError):
        build_launch(profile, tmp_path, tmp_path/'bad')


@pytest.mark.parametrize('value', [True,False,0,1,9,None,4.0,'4'])
def test_physical_maximum_has_a_fixed_integer_bound(value):
    with pytest.raises(ValueError,match='prefill_physical_max_chunks'):
        RuntimeOptions(prefill_physical_max_chunks=value)


@pytest.mark.parametrize('maximum,rows', [(3,192),(4,256),(7,448),(8,512)])
def test_larger_groups_reserve_all_original_sample_rows(maximum,rows):
    value = runtime(payload(prefill_physical_max_chunks=maximum,action_native_max_rows=rows))
    assert value.prefill_physical_max_chunks == maximum
    with pytest.raises(ValueError,match='native sample capacity'):
        runtime(payload(prefill_physical_max_chunks=maximum,action_native_max_rows=rows-1))


@pytest.mark.parametrize('maximum', [4,7,8])
def test_launch_budgets_every_inflight_coordinate_and_score_download(tmp_path,maximum):
    profile = model_profile(tmp_path)
    profile['prefill_chunk_tokens'] = 2048
    profile['kvmem'] = payload(prefill_physical_max_chunks=maximum,
        action_native_max_rows=512,gen_budget=16384)
    launch = build_launch(profile,tmp_path,tmp_path/'large')
    assert runtime(worker_payload(launch)).prefill_physical_max_chunks == maximum
    assert launch['environment']['KVMEM_PREFILL_CHUNK_TOKENS'] == '2048'
    profile['kvmem']['action_download_max_bytes'] = 1024*1024
    with pytest.raises(ValueError,match='download byte budget'):
        build_launch(profile,tmp_path,tmp_path/'too-small')


def test_generation_capacity_cannot_be_silently_increased(tmp_path):
    profile = model_profile(tmp_path)
    profile['prefill_chunk_tokens'] = 2048
    profile['kvmem'] = payload(prefill_physical_max_chunks=4,gen_budget=4096)
    with pytest.raises(ValueError,match='maximum physical forward'):
        build_launch(profile,tmp_path,tmp_path/'bad-generation')
