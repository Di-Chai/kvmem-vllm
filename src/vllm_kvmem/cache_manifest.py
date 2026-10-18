"""Native physical geometry shared by the gateway admission calculation."""
import json
from pathlib import Path


def archive_bytes_per_token(kv_cache_config):
    """Unit-scale FP8 raw K + FP8 V, including native MTP attention."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, UniformTypeKVCacheSpecs
    rows = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        specs = (list(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs)
                 else [spec] * len(group.layer_names))
        for item in specs:
            if type(item) is FullAttentionSpec:
                if item.sliding_window is not None or item.head_size_v != item.head_size:
                    raise ValueError('Unqualified CPU archive geometry')
                rows.append((item.num_kv_heads, item.head_size))
    if not rows or len(set(rows)) != 1:
        raise ValueError('CPU history admission requires uniform full-attention layers')
    return sum(heads * dim * 2 for heads, dim in rows)


def write_mean_k_geometry(config, kv_cache_config, directory):
    from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs
    from .options import RuntimeOptions
    options = RuntimeOptions.from_vllm(config)
    # V2 allocate_kv_cache overlays all descriptors on ONE allocation. Each
    # tensor.size describes the same backing, not an additional allocation.
    sizes = {tensor.size for tensor in kv_cache_config.kv_cache_tensors}
    if len(sizes) != 1 or kv_cache_config.num_blocks <= 0:
        raise ValueError('Expected one native V2 KV backing allocation')
    capacity_bytes = sizes.pop()
    pool_page_bytes, remainder = divmod(capacity_bytes, kv_cache_config.num_blocks)
    if remainder:
        raise ValueError('Native KV backing is not an integral shared page pool')
    groups = []
    for group in kv_cache_config.kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            representative, page_bytes = spec.first_spec, spec.page_size_bytes
        else:
            representative, page_bytes = spec, spec.page_size_bytes * len(group.layer_names)
        groups.append({'block_tokens': representative.block_size, 'page_bytes': pool_page_bytes,
            'payload_page_bytes': page_bytes,
            'layers': list(group.layer_names), 'recurrent': isinstance(representative, MambaSpec),
            'speculative_blocks': getattr(representative, 'num_speculative_blocks', 0)})
    manifest = {'schema': 2, 'groups': groups,
        'archive_codec': 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1',
        'raw_k_scale': 1.0,
        'capacity_bytes': capacity_bytes,
        'native_blocks': kv_cache_config.num_blocks,
        'cpu_offload_bytes': options.cpu_read_cache_bytes,
        'cpu_archive_bytes_per_token': archive_bytes_per_token(kv_cache_config),
        'cpu_history_coordination': options.cpu_history_coordination,
        'cpu_progress_reserve': options.cpu_progress_reserve,
        'semantic_mode': 'original_mean_k_raw_key_dense_rope',
        'select_budget': options.memory.select_budget, 'gen_budget': options.memory.gen_budget,
        'historical_tokens': options.memory.repository_blocks * options.memory.block_tokens,
        'graph_mode': str(config.compilation_config.cudagraph_mode)}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'cache-geometry.json').write_text(json.dumps(manifest, indent=2) + '\n')
