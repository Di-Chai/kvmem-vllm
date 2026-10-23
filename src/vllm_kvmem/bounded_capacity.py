"""Opt-in physical KV demand for the qualified single-request KVMem runtime.

Uses vLLM's registered spec interface. Its original per-worker byte, null-block,
group-padding and allocation checks remain authoritative. Logical model length
and block-table width are deliberately unchanged.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import distribution
import os

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, replace_as
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager


SOURCES = {
    'v1/engine/core.py': '268bf40534a853812867ae8e80040f190983cb364b41182b55969aace20c93f5',
    'v1/kv_cache_interface.py': '1cf202f1a44d5bc5c3832b70b41507687ecceea3784e67a4f003bc0210d6eecb',
    'v1/kv_cache_spec_registry.py': '7f8fa47a8e0318a05f50891f77de1dbc8caa963c3a8f4516bfdf20449afcee97',
    'v1/core/kv_cache_utils.py': '2666c9f113584e52e7521058efd2c0d9598544559dc942d3a3da87001a3fedf2',
    'v1/core/single_type_kv_cache_manager.py': '0233531d85aeced9e57a38c76d83db970519c92902b7e55484390cb2b64985bf',
    'v1/worker/gpu/model_runner.py': '174c93db921c23cf0396eee4764be25b2bd2d4b6a06e9fa41ce3598b884ce8ce',
}


def enabled():
    return os.environ.get('KVMEM_BOUNDED_KV_CAPACITY') == '1'


@lru_cache(maxsize=1)
def verify_sources():
    from .integration import check_engine_version
    check_engine_version()
    package = distribution('vllm')
    for name, expected in SOURCES.items():
        if hashlib.sha256(package.locate_file('vllm/' + name).read_bytes()).hexdigest() != expected:
            raise ValueError('Unqualified physical KV capacity source: ' + name)


def qualified_capacity(config):
    from .options import RuntimeOptions
    from .draft_contract import DraftContract
    verify_sources()
    required = ('KVMEM_BOUNDED_KV_CAPACITY', 'KVMEM_BOUNDED_TARGET',
                'KVMEM_BOUNDED_MTP', 'KVMEM_WINDOW_METADATA', 'VLLM_USE_V2_MODEL_RUNNER')
    if any(os.getenv(name) != '1' for name in required):
        raise ValueError('Physical KV capacity requires explicit bounded MTP/window/V2 opt-in')
    parallel, scheduler, cache = config.parallel_config, config.scheduler_config, config.cache_config
    if any(getattr(parallel, name, None) != 1 for name in (
            'tensor_parallel_size', 'pipeline_parallel_size', 'decode_context_parallel_size',
            'prefill_context_parallel_size')):
        raise ValueError('Physical KV capacity currently requires TP=PP=DCP=PCP=1')
    selected = scheduler.scheduler_cls
    if selected != 'vllm_kvmem.scheduler.KVMemScheduler':
        from .scheduler import KVMemScheduler
        if selected is not KVMemScheduler:
            raise ValueError('Physical KV capacity requires the KVMem working-set scheduler')
    transfer = config.kv_transfer_config
    if (transfer is None or transfer.kv_connector != 'KVMemConnector' or
            transfer.kv_connector_module_path != 'vllm_kvmem.connector'):
        raise ValueError('Physical KV capacity requires the KVMem connector')
    if (scheduler.max_num_seqs != 1 or scheduler.async_scheduling or
            config.max_concurrent_batches != 1 or scheduler.disable_hybrid_kv_cache_manager or
            cache.enable_prefix_caching or cache.mamba_cache_mode != 'none' or
            cache.num_gpu_blocks_override is not None or cache.kv_cache_memory_bytes is not None or
            config.attention_config.hisparse_config is not None):
        raise ValueError('Physical KV capacity requires C1 synchronous hybrid APC-off profiled allocation')
    options = RuntimeOptions.from_vllm(config)
    contract = DraftContract.from_vllm(config, options.memory)
    capacity = options.memory.select_budget + options.memory.gen_budget
    if (contract.method != 'mtp' or not options.request_planning or
            not 0 < config.model_config.max_model_len <= 32768 or
            config.model_config.original_max_model_len == -1):
        raise ValueError('Physical KV capacity requires planned MTP and explicit logical context <=32768')
    return capacity


@dataclass(frozen=True, kw_only=True)
class BoundedFullAttentionSpec(FullAttentionSpec):
    capacity_tokens: int

    def __post_init__(self):
        super().__post_init__()
        if type(self.capacity_tokens) is not int or self.capacity_tokens <= 0:
            raise ValueError('A bounded full-attention spec requires an explicit positive token cap')
        if self.sliding_window is not None or self.attention_chunk_size is not None or self.non_causal:
            raise ValueError('Bounded capacity only qualifies causal full attention')

    def max_memory_usage_bytes(self, vllm_config):
        if qualified_capacity(vllm_config) != self.capacity_tokens:
            raise ValueError('Declared physical cap differs from the KVMem working-set capacity')
        tokens = min(vllm_config.model_config.max_model_len, self.capacity_tokens)
        return ((tokens + self.block_size - 1) // self.block_size) * self.page_size_bytes

    @classmethod
    def merge(cls, specs):
        # Native uniformity discovery treats AssertionError as a heterogeneous
        # hybrid group. Preserve that contract regardless of dict layer order.
        if any(not isinstance(spec, FullAttentionSpec) for spec in specs):
            raise AssertionError('Hybrid attention/recurrent specs are not uniform')
        if (not specs or any(type(spec) is not cls for spec in specs) or
                len({spec.capacity_tokens for spec in specs}) != 1):
            raise ValueError('Cannot merge undeclared or inconsistent physical KV caps')
        # Parent merge reconstructs its cls without subclass fields. Merge exact
        # base geometry first, then reattach the explicitly agreed physical cap.
        merged = FullAttentionSpec.merge([
            replace_as(spec, FullAttentionSpec, drop=('capacity_tokens',)) for spec in specs])
        return replace_as(merged, cls, capacity_tokens=specs[0].capacity_tokens)

    def is_uniform_with_collection(self, specs):
        return super().is_uniform_with_collection(specs) and all(
            type(spec) is type(self) and spec.capacity_tokens == self.capacity_tokens
            for spec in specs.values())


# Registration is also performed when EngineCore unpickles worker-produced specs.
# All actual manager instances remain the original FullAttentionManager, wrapped
# later by KVMem's existing WorkingSetManager and ReservedKVCacheManager.
KVCacheSpecRegistry.register(BoundedFullAttentionSpec, FullAttentionManager,
                             uniform_type_base_spec=FullAttentionSpec)


def is_qualified_full_spec(spec):
    if type(spec) is FullAttentionSpec:
        return True
    if type(spec) is BoundedFullAttentionSpec:
        if not enabled():
            raise ValueError('A bounded KV spec appeared without explicit capacity opt-in')
        return True
    return False


def bounded_specs(specs, config):
    if not enabled():
        return specs
    capacity = qualified_capacity(config)
    if capacity > config.model_config.max_model_len:
        raise ValueError('Physical working-set cap exceeds logical context')
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    register_all_kvcache_specs(config)
    result = {}
    full_layers = 0
    speculative_blocks = config.speculative_config.num_speculative_tokens
    for name, spec in specs.items():
        if type(spec) is FullAttentionSpec:
            result[name] = replace_as(spec, BoundedFullAttentionSpec, capacity_tokens=capacity)
            full_layers += 1
        elif type(spec) is MambaSpec:
            if (spec.mamba_cache_mode != 'none' or spec.num_prefill_checkpoint_blocks or
                    spec.num_speculative_blocks != speculative_blocks or
                    spec.block_size < config.model_config.max_model_len):
                raise ValueError('Unqualified recurrent allocation under bounded capacity')
            result[name] = spec
        else:
            raise ValueError('Unqualified KV spec under bounded capacity: ' + type(spec).__name__)
    if not full_layers:
        raise ValueError('Bounded physical capacity requires full-attention layers')
    return result


def validate_manager_coverage(reserved, config):
    """Fail startup if any declared cap lacks the actual allocator wrapper."""
    from .window_manager import WorkingSetManager
    from .capacity import ReservedKVCacheManager
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    if type(reserved) is not ReservedKVCacheManager or type(reserved.native) is not KVCacheManager:
        raise ValueError('Physical KV capacity requires the actual native reservation manager')
    actual = tuple(reserved.native.coordinator.single_type_managers)
    if len(actual) != len(reserved.groups) or any(a is not b for a, b in zip(actual, reserved.groups)):
        raise ValueError('Physical KV capacity manager coverage changed after reservation')
    expected = qualified_capacity(config)
    declared = 0
    for manager in reserved.groups:
        spec = manager.kv_cache_spec
        specs = spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
        for item in specs:
            if isinstance(item, FullAttentionSpec):
                if (type(item) is not BoundedFullAttentionSpec or item.capacity_tokens != expected or
                        type(manager) is not WorkingSetManager or manager.capacity != expected):
                    raise ValueError('Declared physical KV cap lacks an identical WorkingSetManager bound')
                declared += 1
    if not declared or reserved._demand(config.model_config.max_model_len) > reserved.block_pool.get_num_free_blocks():
        raise ValueError('Bounded physical capacity cannot cover its worst-case allocator demand')
