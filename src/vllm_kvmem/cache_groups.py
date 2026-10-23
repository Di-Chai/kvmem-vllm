"""Identify semantic attention groups, not generic slot-mapping capability."""


def _bounded_full_spec(spec):
    if (type(spec).__module__, type(spec).__name__) != ("vllm_kvmem.bounded_capacity", "BoundedFullAttentionSpec"):
        return False
    from .bounded_capacity import is_qualified_full_spec
    return is_qualified_full_spec(spec)


def full_attention_groups(kv_cache_config):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs

    result = []
    for index, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.first_spec
        if type(spec) is FullAttentionSpec or _bounded_full_spec(spec):
            if spec.sliding_window is not None or spec.attention_chunk_size is not None or spec.non_causal:
                raise ValueError("Working windows currently require causal full attention")
            result.append(index)
    if not result:
        raise ValueError("No qualified full-attention cache groups")
    return tuple(result)
