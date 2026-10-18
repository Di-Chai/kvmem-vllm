"""Pure preparation/geometry checks; no engine, model, or GPU is imported."""
import copy
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('capacity_probe', Path(__file__).parents[1] / 'scripts/benchmark_capacity_scaling.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def profile():
    return dict(model_format='safetensors', draft_method='mtp', draft_tokens=4,
        execution_mode='compiled_graph', gpu_memory_utilization=.81, query_replay_budget=16384,
        model='/unchanged/model', tokenizer='/unchanged/tokenizer', max_num_batched_tokens=4224,
        kvmem=dict(block_tokens=32, offline_agent_policy=True, prefill_batch_tokens=0,
            archive_backend='cpu', split_capture_dma=True, native_prefill_commit=True,
            async_checkpoint_capture=True, select_budget=131072, gen_budget=131072))


@pytest.mark.parametrize('changed', ['none','fixture','profile','incomplete','source'])
def test_carry_real_native_requires_identical_complete_frozen_work(tmp_path, monkeypatch, changed):
    stage = tmp_path/'native-c8'; stage.mkdir()
    (tmp_path/'frozen').mkdir()
    (tmp_path/'runtime/KVMem').mkdir(parents=True)
    values = {'manifest.json': dict(fixture_kind='real_deepswe_history',
        source_sha256={},runtime_sha256={},native_source={'identity':'same'}),
        'profile.json': {'port':18522}, 'fixture.json': {'input':'same'}}
    for name,value in values.items(): (tmp_path/name).write_text(json.dumps(value))
    result=dict(status='completed',drained=True,cleanup_complete=True,
        fixed_work_output_tokens=65536,native_client_total_output_reconciliation={'passed':True})
    if changed=='incomplete':result['drained']=False
    (stage/'results.json').write_text(json.dumps(result))
    monkeypatch.setattr(probe,'native_source_identity',lambda:{'identity':'other' if changed=='source' else 'same'})
    profile={'port':18521 if changed=='profile' else 18522}
    fixture={'input':'other' if changed=='fixture' else 'same'}
    if changed=='none':
        carried=probe.carry_real_native(stage,profile,fixture)
        assert str(stage/'results.json') in carried['sha256']
    else:
        with pytest.raises(ValueError):probe.carry_real_native(stage,profile,fixture)


def test_profile_retains_model_and_original_object_and_has_exact_capacity_contract():
    original = profile()
    before = copy.deepcopy(original)
    result = probe.capacity_profile(original, 18520)
    assert original == before
    assert result['model'] == original['model'] and result['tokenizer'] == original['tokenizer']
    assert result['max_model_len'] == 262144 and result['max_num_seqs'] == 8
    assert result['kvmem']['select_budget'] + result['kvmem']['gen_budget'] == 98304
    assert result['kvmem']['repository_blocks'] * result['kvmem']['block_tokens'] == 262144
    assert result['kvmem']['cpu_read_cache_bytes'] == 100_000_000_000
    assert result['max_num_batched_tokens'] == original['max_num_batched_tokens']


@pytest.mark.parametrize('flag', ['native_restore_frame', 'native_action_metadata'])
def test_unqualified_native_overlay_rejected(flag):
    source = profile()
    source['kvmem'][flag] = True
    with pytest.raises(ValueError, match='fastpaths'):
        probe.capacity_profile(source, 18520)


@pytest.mark.parametrize('ratio,expected', [('5.87', 5), ('5.00', 4), ('8.12', 8)])
def test_native_capacity_rounding_cannot_admit_an_extra_full_history(ratio, expected):
    result = probe.native_capacity(f'GPU KV cache size: 1,000 tokens, Maximum concurrency for 262,144 tokens per request: {ratio}x')
    assert result['conservative_admitted_limit'] == expected


@pytest.mark.parametrize('log', ['', 'Maximum concurrency for 524,288 tokens per request: 8.00x',
    'Maximum concurrency for 262,144 tokens per request: 5.01x\nMaximum concurrency for 262,144 tokens per request: 5.02x'])
def test_native_geometry_missing_wrong_or_conflicting_cannot_become_capacity(log):
    with pytest.raises(ValueError):
        probe.native_capacity(log)


def test_kvmem_capacity_accounts_for_recurrent_speculation_and_cpu_history():
    config = probe.capacity_profile(profile(), 18520)
    geometry = dict(select_budget=65536, gen_budget=32768, historical_tokens=262144,
        cpu_offload_bytes=100_000_000_000, archive_codec='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1',
        cpu_archive_bytes_per_token=34816, native_blocks=914,
        groups=[dict(recurrent=True, speculative_blocks=4, block_tokens=262144)] * 3 +
               [dict(recurrent=False, speculative_blocks=0, block_tokens=1616)])
    result = probe.kvmem_capacity(geometry, config)
    assert result['gpu_pages_per_lifetime'] == 76  # 61 attention + 15 recurrent/speculative.
    assert result['conservative_gpu_request_limit'] == 12
    assert result['full_256k_history_credit_bytes'] == 9_129_033_728
    assert result['cpu_full_history_request_limit'] == 10
    assert result['this_request_history_credit_bytes'] < result['full_256k_history_credit_bytes']


def test_host_archive_budget_is_not_mistaken_for_total_ram():
    result = probe.host_budget(32 * 1024**3, 8 * 1024**3, offered_concurrency=8, native=False)
    assert result['configured_archive_cap_bytes'] == 100_000_000_000
    assert result['archive_bytes'] == 8 * 9_129_033_728
    assert result['required_prelaunch_available_bytes'] == 8 * (9_129_033_728 + 805_306_368) + 32 * 1024**3
    assert result['required_prelaunch_available_bytes'] == 113_834_459_136
    assert result['extra_model_dma_checkpoint_system_bytes'] >= result['minimum_free_bytes']
    with pytest.raises(ValueError):
        probe.host_budget(1024, 2048, offered_concurrency=8, native=False)


def test_fixed_history_budget_keeps_full_index_and_system_headroom():
    limits=[229796+8192]*8
    result=probe.host_budget(32*1024**3,8*1024**3,
        offered_concurrency=8,native=False,fixed_history_limits=limits)
    assert result['archive_bytes']==8*((limits[0]+31)//32*32+64)*34816
    assert result['mean_k_index_bytes']==8*805306368
    assert result['extra_model_dma_checkpoint_system_bytes']==32*1024**3
    assert result['minimum_free_bytes']==8*1024**3
    assert result['required_prelaunch_available_bytes']<113834459136
    with pytest.raises(ValueError):
        probe.host_budget(32*1024**3,8*1024**3,
            offered_concurrency=8,native=False,fixed_history_limits=[262145]*8)


def test_host_budget_uses_stage_cohort_and_native_has_no_kvmem_archive_or_index():
    c4 = probe.host_budget(32 * 1024**3, 8 * 1024**3, offered_concurrency=4, native=False)
    c8 = probe.host_budget(32 * 1024**3, 8 * 1024**3, offered_concurrency=8, native=False)
    assert c4['archive_bytes'] == 4 * 9_129_033_728
    assert c4['mean_k_index_bytes'] == 4 * 805_306_368
    assert c8['archive_bytes'] == 2 * c4['archive_bytes']
    assert c8['required_prelaunch_available_bytes'] - c4['required_prelaunch_available_bytes'] == 4 * (9_129_033_728 + 805_306_368)
    for count in (4, 8):
        native = probe.host_budget(32 * 1024**3, 8 * 1024**3, offered_concurrency=count, native=True)
        assert native['archive_bytes'] == native['mean_k_index_bytes'] == native['configured_archive_cap_bytes'] == 0
        assert native['required_prelaunch_available_bytes'] == 32 * 1024**3


def metric_row(at, generated, preemptions=0):
    return dict(started_at=at, values={'vllm:generation_tokens_total': generated,
                                    'vllm:num_preemptions_total': preemptions})


def test_warmup_and_formal_counters_use_separate_completed_boundaries():
    async def run():
        before_warmup, after_warmup = metric_row(1., 100), metric_row(2., 102)
        after_formal = metric_row(4., 120)
        async def warmup_snapshot():
            return after_warmup
        async def formal_snapshot():
            return after_formal
        warmed = await probe.reconcile_outputs(warmup_snapshot, before_warmup, 2)
        measured = await probe.reconcile_outputs(formal_snapshot, after_warmup, 18)
        assert warmed['native_tokens'] == 2
        assert measured['native_tokens'] == 18
        assert measured['before_snapshot_started_at'] == warmed['after_snapshot_started_at']
    asyncio.run(run())


@pytest.mark.parametrize('generated,preemptions,error', [
    (103, 0, 'mismatch'), (101, 0, 'mismatch'), (99, 0, 'reset'), (102, 1, 'preempted')])
def test_output_reconciliation_fails_closed_for_counter_mismatch_or_preemption(generated, preemptions, error):
    async def snapshot():
        return metric_row(2., generated, preemptions)
    with pytest.raises(ValueError, match=error):
        asyncio.run(probe.reconcile_outputs(snapshot, metric_row(1., 100), 2, timeout=0))
