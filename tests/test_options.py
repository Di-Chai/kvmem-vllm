from types import SimpleNamespace
import pytest
from vllm_kvmem.options import RuntimeOptions


def config(payload):
    return SimpleNamespace(kv_transfer_config=SimpleNamespace(kv_connector_extra_config=payload),
        cache_config=SimpleNamespace(cache_dtype='fp8_e4m3'))


def test_public_connector_config_propagates_granularity_and_resource_options_without_mutation():
    values = {'kvmem': {'block_tokens': 32, 'repository_blocks': 16384,
                       'cpu_read_cache_bytes': 9 * 1024**3, 'packed_assembly': True,
                       'archive_io_max_bytes': 2 * 1024**3, 'cpu_gather_workers': 4}}
    options = RuntimeOptions.from_vllm(config(values))
    assert options.memory.block_tokens == 32
    assert (options.memory.select_budget, options.memory.gen_budget) == (131072, 65536)
    assert options.cpu_read_cache_bytes == 9 * 1024**3 and options.packed_assembly
    assert options.archive_io_max_bytes == 2 * 1024**3
    assert options.cpu_gather_workers == 4
    assert values['kvmem']['cpu_read_cache_bytes'] == 9 * 1024**3
    assert RuntimeOptions.from_vllm(config({})).memory.block_tokens == 128


@pytest.mark.parametrize('payload,match', [
    ({'kvmem': {'block_toknes': 32}}, 'Unknown'),
    ({'kvmem': {'cpu_read_cache_bytes': True}}, 'capacity'),
    ({'kvmem': {'gpu_raw_k_cache_bytes': True}}, 'capacity'),
    ({'kvmem': {'gpu_raw_k_cache_bytes': -1}}, 'capacity'),
    ({'kvmem': {'gpu_raw_k_cache_bytes': 4096}}, 'direct packed'),
    ({'kvmem': {'reuse_resident_v': 1}}, 'boolean'),
    ({'kvmem': {'async_cpu_scatter': 1}}, 'boolean'),
    ({'kvmem': {'batched_cpu_append': 1}}, 'boolean'),
    ({'kvmem': {'cpu_scatter_min_tokens': True}}, 'minimum'),
    ({'kvmem': {'cpu_scatter_min_tokens': 0}}, 'minimum'),
    ({'kvmem': {'cpu_scatter_min_tokens': 65537}}, 'minimum'),
    ({'kvmem': {'reuse_resident_v': True}}, 'direct packed'),
    ({'kvmem': {'cpu_gather_workers': True}}, 'gather'),
    ({'kvmem': {'cpu_gather_workers': 0}}, 'gather'),
    ({'kvmem': {'cpu_gather_workers': 17}}, 'gather'),
    ({'kvmem': {'packed_assembly': 'true'}}, 'boolean'),
    ({'kvmem': {'verify_archive_on_finish': 0}}, 'boolean'),
    ({'kvmem': {'direct_pinned_reads': True}}, 'require packed'),
    ({'kvmem': {'packed_assembly': True, 'direct_pinned_reads': 1}}, 'boolean'),
    ({'kvmem': {'prefill_cadence': True}}, 'cadence'),
    ({'kvmem': {'prefill_cadence': 0}}, 'cadence'),
    ({'kvmem': {'prefill_cadence': 65}}, 'cadence'),
    ({'kvmem': {'async_archive_retirement': 1}}, 'boolean'),
    ({'kvmem': {'repository_gc': 1}}, 'boolean'),
    ({'kvmem': {'nvme_preallocate': 1}}, 'boolean'),
    ({'kvmem': {'nvme_max_bytes': True}}, 'capacity'),
    ({'kvmem': {'nvme_max_bytes': -1}}, 'capacity'),
    ({'kvmem': {'archive_io_max_bytes': -1}}, 'capacity'),
    ({'kvmem': {'archive_io_max_bytes': True}}, 'capacity'),
    ({'kvmem': {'session_capacity': True}}, 'capacity'),
    ({'kvmem': {'session_capture_decode': 1}}, 'boolean'),
    ({'kvmem': {'session_capture_decode': True}}, 'capacity'),
    ({'kvmem': {'session_capture_terminal': 1}}, 'boolean'),
    ({'kvmem': {'session_capture_terminal': True}}, 'capacity'),
    ({'kvmem': {'cpu_read_cache_bytes': 128, 'cpu_history_cache_bytes_per_request': 129}}, 'fit'),
    ({'kvmem': {'session_capacity': 2}}, 'both'),
    ({'kvmem': {'session_checkpoint_bytes': 4096}}, 'both'),
    ({'kvmem': {'session_capacity': 2, 'session_checkpoint_bytes': 4096}}, 'retirement'),
    ({'kvmem': {'kv_dtype': 'bfloat16'}}, 'codec'),
    ({'kvmem': []}, 'object'),
])
def test_runtime_settings_reject_unsupported_or_mistyped_options(payload, match):
    with pytest.raises(ValueError, match=match):
        RuntimeOptions.from_vllm(config(payload))
