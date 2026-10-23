"""Exercise real vLLM cache planners on CPU, never allocate CUDA or load a model."""
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs, generate_scheduler_kv_cache_config
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm_kvmem.bounded_capacity import (
    BoundedFullAttentionSpec, bounded_specs, qualified_capacity,
    is_qualified_full_spec, validate_manager_coverage,
)


ENV = ('KVMEM_BOUNDED_KV_CAPACITY', 'KVMEM_BOUNDED_TARGET', 'KVMEM_BOUNDED_MTP',
       'KVMEM_WINDOW_METADATA', 'VLLM_USE_V2_MODEL_RUNNER')
PAGE = 55148544  # 17 target+MTP layers, 1584 tokens, 4 heads, dim 256, FP8 K/V.


@pytest.fixture
def enabled(monkeypatch):
    for key in ENV:
        monkeypatch.setenv(key, '1')


def config(logical=32768):
    from vllm.config import CacheConfig
    cache = CacheConfig(block_size=16, cache_dtype='fp8_e4m3', enable_prefix_caching=False,
                        mamba_cache_mode='none')
    cache.kv_cache_layout = 'LBNHC'
    return NS(
        cache_config=cache,
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, prefill_context_parallel_size=1),
        scheduler_config=NS(scheduler_cls='vllm_kvmem.scheduler.KVMemScheduler',
                            max_num_seqs=1, async_scheduling=False,
                            disable_hybrid_kv_cache_manager=False),
        model_config=NS(max_model_len=logical, original_max_model_len=logical),
        attention_config=NS(hisparse_config=None),
        speculative_config=NS(method='mtp', num_speculative_tokens=1, kv_cache_dtype=None,
                              draft_model_config=NS(hf_text_config=NS(model_type='qwen3_5_text', mtp_num_hidden_layers=1)),
                              use_multi_module_mtp=lambda: False, use_eagle_block_drop=lambda: False, use_eagle=lambda: True),
        max_concurrent_batches=1, num_lookahead_tokens=1, max_in_flight_tokens=130,
        kv_transfer_config=NS(kv_connector='KVMemConnector', kv_connector_module_path='vllm_kvmem.connector',
            kv_connector_extra_config={'kvmem': dict(select_budget=4096, gen_budget=4096,
                block_tokens=32, repository_blocks=1028, sink_blocks=1, recent_blocks=1, request_planning=True)}),
    )


def specs(logical=32768):
    attention = FullAttentionSpec(block_size=1584, num_kv_heads=4, head_size=256,
                                  dtype=torch.float8_e4m3fn)
    # Reproduce the recorded Qwen page bytes without constructing model tensors.
    recurrent = MambaSpec(block_size=logical, shapes=((811008,),), dtypes=(torch.float32,),
                          mamba_cache_mode='none', num_speculative_blocks=1)
    return {**{f'mamba.{i}': recurrent for i in range(48)},
            **{f'attention.{i}': attention for i in range(17)}}


def test_opt_out_preserves_original_spec_objects(monkeypatch):
    monkeypatch.delenv('KVMEM_BOUNDED_KV_CAPACITY', raising=False)
    original = specs()
    assert bounded_specs(original, None) is original


def test_merge_retains_declared_cap_and_rejects_mixed_specs(enabled):
    cfg = config()
    converted = bounded_specs(specs(), cfg)
    full = converted['attention.0']
    assert type(full) is BoundedFullAttentionSpec
    merged = full.merge([full, replace(full)])
    assert merged.capacity_tokens == 8192
    assert merged.page_size_bytes == full.page_size_bytes
    assert merged.max_memory_usage_bytes(cfg) == 6 * full.page_size_bytes
    assert KVCacheSpecRegistry.get_manager_class(full) is FullAttentionManager
    with pytest.raises(ValueError, match='inconsistent'):
        full.merge([full, replace(full, capacity_tokens=4096)])
    with pytest.raises(ValueError, match='undeclared'):
        full.merge([full, specs()['attention.0']])
    with pytest.raises(ValueError, match='differs'):
        replace(full, capacity_tokens=4096).max_memory_usage_bytes(cfg)


@pytest.mark.parametrize('logical', [16384, 24576, 32768])
@pytest.mark.parametrize('attention_first', [False, True])
def test_native_per_worker_padding_null_and_byte_checks_are_retained(enabled, logical, attention_first):
    cfg = config(logical)
    original = specs(logical)
    if attention_first:
        original = dict(reversed(list(original.items())))
    converted = bounded_specs(original, cfg)
    configs = get_kv_cache_configs(cfg, [converted], [19 * PAGE])
    assert cfg.model_config.max_model_len == logical
    assert configs[0].num_blocks == 19
    assert {tensor.size for tensor in configs[0].kv_cache_tensors} == {19 * PAGE}
    scheduler = generate_scheduler_kv_cache_config(configs)
    groups = scheduler.kv_cache_groups
    full = [g for g in groups if isinstance(g.kv_cache_spec, FullAttentionSpec)]
    recurrent = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec)]
    from vllm_kvmem.cache_groups import full_attention_groups
    from vllm_kvmem.cache_manifest import archive_bytes_per_token
    assert full_attention_groups(scheduler) == tuple(i for i, g in enumerate(groups) if g in full)
    assert archive_bytes_per_token(scheduler) == 34816
    assert len(full) == 1 and len(full[0].layer_names) == 17
    assert type(full[0].kv_cache_spec) is BoundedFullAttentionSpec
    # Preserve logical-width native block tables; cap only physical demand.
    assert full[0].kv_cache_spec.max_num_blocks_per_req(cfg, logical) == (logical + 1583) // 1584
    assert len(recurrent) == 3 and all(len(g.layer_names) == 16 for g in recurrent)
    assert sum(g.kv_cache_spec.max_memory_usage_bytes(cfg) // g.kv_cache_spec.page_size_bytes for g in groups) == 12
    assert all(converted[f'mamba.{i}'] is original[f'mamba.{i}'] for i in range(48))
    # Exactly twelve usable pages plus the permanent null page is sufficient.
    assert get_kv_cache_configs(cfg, [converted], [13 * PAGE])[0].num_blocks == 13
    with pytest.raises(ValueError, match='larger than the available'):
        get_kv_cache_configs(cfg, [converted], [13 * PAGE - 1])
    assert cfg.model_config.max_model_len == logical  # failed estimate restores config
    with pytest.raises(ValueError):
        get_kv_cache_configs(cfg, [converted], [0])
    assert not torch.cuda.is_initialized()


def test_native_full_logical_capacity_still_fails_without_adapter(enabled):
    cfg = config(32768)
    with pytest.raises(ValueError, match='larger than the available'):
        get_kv_cache_configs(cfg, [specs()], [19 * PAGE])


@pytest.mark.parametrize('field,value', [('tensor_parallel_size', 2), ('pipeline_parallel_size', 2),
                                       ('decode_context_parallel_size', 2), ('prefill_context_parallel_size', 2)])
def test_topology_fail_closed(enabled, field, value):
    cfg = config()
    setattr(cfg.parallel_config, field, value)
    with pytest.raises(ValueError, match='TP=PP=DCP=PCP'):
        bounded_specs(specs(), cfg)


@pytest.mark.parametrize('area,field,value', [
    ('scheduler_config', 'max_num_seqs', 2), ('scheduler_config', 'async_scheduling', True),
    ('scheduler_config', 'disable_hybrid_kv_cache_manager', True),
    ('cache_config', 'enable_prefix_caching', True), ('cache_config', 'mamba_cache_mode', 'all'),
    ('cache_config', 'num_gpu_blocks_override', 100), ('cache_config', 'kv_cache_memory_bytes', 1 << 40),
])
def test_unqualified_execution_and_overrides_fail_closed(enabled, area, field, value):
    cfg = config()
    setattr(getattr(cfg, area), field, value)
    with pytest.raises(ValueError, match='C1 synchronous'):
        bounded_specs(specs(), cfg)


def test_unknown_specs_and_missing_opt_in_rejected(enabled, monkeypatch):
    cfg = config()
    full = bounded_specs(specs(), cfg)['attention.0']
    class UnknownFull(FullAttentionSpec):
        pass
    unknown = UnknownFull(block_size=1584, num_kv_heads=4, head_size=256, dtype=torch.uint8)
    assert not is_qualified_full_spec(unknown)
    with pytest.raises(ValueError, match='Unqualified KV spec'):
        bounded_specs({'unknown': unknown}, cfg)
    monkeypatch.delenv('KVMEM_BOUNDED_KV_CAPACITY')
    with pytest.raises(ValueError, match='without explicit'):
        is_qualified_full_spec(full)


def test_declared_cap_requires_actual_reservation_manager(enabled):
    with pytest.raises(ValueError, match='actual native reservation manager'):
        validate_manager_coverage(NS(), config())



def test_pickled_worker_spec_registers_on_fresh_cpu_engine_process(enabled):
    import os
    import pickle
    import subprocess
    import sys
    full = bounded_specs(specs(), config())['attention.0']
    code = """
import pickle, sys
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
spec = pickle.loads(sys.stdin.buffer.read())
assert type(spec).__name__ == 'BoundedFullAttentionSpec'
assert spec.capacity_tokens == 8192
assert KVCacheSpecRegistry.get_manager_class(spec) is FullAttentionManager
import torch
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run([sys.executable, '-c', code], input=pickle.dumps(full),
                            capture_output=True, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, timeout=60)
    assert result.returncode == 0, result.stderr.decode()


def test_real_hybrid_allocator_mtp_boundary_refresh_and_regrowth(enabled):
    from vllm import SamplingParams
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.request import Request
    from vllm_kvmem.capacity import ReservedKVCacheManager
    from vllm_kvmem.draft_contract import DraftContract
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.window_manager import WorkingSetManager, RefreshRequired

    cfg = config()
    converted = bounded_specs(specs(), cfg)
    worker = get_kv_cache_configs(cfg, [converted], [13 * PAGE])
    scheduler = generate_scheduler_kv_cache_config(worker)
    import math
    granularity = math.lcm(*(g.kv_cache_spec.block_size for g in scheduler.kv_cache_groups))
    native = KVCacheManager(scheduler, 32768, granularity, 16,
                           max_in_flight_tokens=130, enable_caching=False)
    managers = list(native.coordinator.single_type_managers)
    full_index = next(i for i, m in enumerate(managers) if type(m) is FullAttentionManager)
    managers[full_index] = WorkingSetManager(managers[full_index], 4096, 4096)
    native.coordinator.single_type_managers = tuple(managers)
    contract = DraftContract.from_vllm(cfg, RuntimeOptions.from_vllm(cfg).memory)
    gate = ReservedKVCacheManager(native, cfg, draft_contract=contract, max_authorities=1)
    validate_manager_coverage(gate, cfg)
    assert gate._demand(32768) == 12
    assert native.block_pool.get_num_free_blocks() == 12  # null already owns one
    req = Request('long', [1] * 16000, SamplingParams(max_tokens=16000), None)
    window = managers[full_index]
    window.capacity = 4096
    with pytest.raises(ValueError, match='WorkingSetManager'):
        validate_manager_coverage(gate, cfg)
    window.capacity = 8192
    native.coordinator.single_type_managers = tuple(m.native if m is window else m for m in managers)
    with pytest.raises(ValueError, match='coverage changed'):
        validate_manager_coverage(gate, cfg)
    native.coordinator.single_type_managers = tuple(managers)
    for history_end in (8190, 12284):
        while req.num_computed_tokens < history_end:
            count = min(128, history_end - req.num_computed_tokens)
            assert gate.allocate_slots(req, count, num_lookahead_tokens=2) is not None
            req.num_computed_tokens += count
            assert gate._owned(req.request_id) <= 12
        assert native.block_pool.get_num_free_blocks() == 0
        assert len(window.req_to_blocks[req.request_id]) == 6
        assert all(len(m.req_to_blocks[req.request_id]) == 2
                   for i, m in enumerate(managers) if i != full_index)
        before = [tuple(m.req_to_blocks[req.request_id]) for m in managers]
        with pytest.raises(RefreshRequired):
            gate.allocate_slots(req, 1, num_lookahead_tokens=2)
        assert [tuple(m.req_to_blocks[req.request_id]) for m in managers] == before
        ticket = window.prepare_refresh(req.request_id, history_end, 4096)
        assert window.commit_refresh(ticket, NS(query=lambda: True))
        assert native.block_pool.get_num_free_blocks() == 3
    # C1 lifetime admission survives refresh even though free physical pages exist.
    other = Request('other', [1] * 16000, SamplingParams(max_tokens=16000), None)
    assert gate.allocate_slots(other, 128, num_lookahead_tokens=2) is None
    gate.free(req)
    assert native.block_pool.get_num_free_blocks() == 12 and not gate.reservations
    assert not torch.cuda.is_initialized()


def test_source_guard_rejects_unqualified_engine_without_patching_vllm(enabled, monkeypatch):
    from vllm_kvmem import bounded_capacity
    bounded_capacity.verify_sources.cache_clear()
    with monkeypatch.context() as patch:
        patch.setitem(bounded_capacity.SOURCES, 'v1/engine/core.py', '0' * 64)
        with pytest.raises(ValueError, match='Unqualified physical KV capacity source'):
            bounded_capacity.verify_sources()
    bounded_capacity.verify_sources.cache_clear()
    bounded_capacity.verify_sources()


def test_uniform_collection_preserves_native_geometry_and_per_layer_accounting(enabled):
    cfg = config()
    full = bounded_specs(specs(), cfg)['attention.0']
    smaller = replace(full, num_kv_heads=2)
    with pytest.raises(AssertionError, match='same attention spec'):
        full.merge([full, smaller])
    # Native UniformType allows different per-layer byte geometry at the same
    # token width and charges their sum; it must not silently merge their bytes.
    grouped = UniformTypeKVCacheSpecs.from_specs({'a': full, 'b': smaller})
    assert grouped.max_memory_usage_bytes(cfg) == 6 * (full.page_size_bytes + smaller.page_size_bytes)
    assert grouped.max_num_blocks_per_req(cfg, 32768) == 21
    assert UniformTypeKVCacheSpecs.from_specs({'a': full, 'b': replace(full, block_size=792)}) is None
    assert UniformTypeKVCacheSpecs.from_specs({'a': full, 'b': replace(full, capacity_tokens=4096)}) is None
