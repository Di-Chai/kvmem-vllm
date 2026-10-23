"""Narrow initial contract for a native DFlash2 sliding-window draft.

The target uses dense selected positions; the draft retains its own native
logical positions and sliding pages. Only final target hidden states reach the
native proposer. This contract does not itself enable bounded speculation.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import distribution


DRAFT_SOURCES = {
    'v1/worker/gpu/spec_decode/__init__.py': '1f1f84bfe6f4f5af3a7e21a7a8716ecf7b9013da1d03327668a3ee36afc24c32',
    'v1/worker/gpu/spec_decode/dflash/speculator.py': 'f9c100f42f9326e48eb84e89281897e859ba761943f54d593523555ccd98dc63',
    'v1/worker/gpu/spec_decode/dflash2/speculator.py': 'b70043550938a1df9aecc85d8c8224c38c929ca34f68834572f6e723854d2027',
    'model_executor/models/qwen3_dflash.py': '4150292acb5934276e44bd0ce11927818174848c78b39d5223dd8fb1114e5640',
    'model_executor/models/qwen3_dflash2.py': 'c141daa4b2059c0098224ac36471c2197b7052c100bef0a4dbc2ca79b627053f',
}


@lru_cache(maxsize=1)
def verify_draft_sources():
    from .integration import check_engine_version
    check_engine_version()
    package = distribution('vllm')
    for relative, expected in DRAFT_SOURCES.items():
        if hashlib.sha256(package.locate_file('vllm/' + relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unqualified native draft lifecycle source: {relative}')


@dataclass(frozen=True)
class DraftContract:
    query_tokens: int
    sliding_window: int
    max_position: int
    layers: int
    method: str = 'dflash'

    @classmethod
    def from_vllm(cls, config, memory):
        import torch
        spec = config.speculative_config
        if spec is not None and spec.method == 'mtp':
            verify_mtp_sources()
            text = spec.draft_model_config.hf_text_config
            layers = getattr(text, 'mtp_num_hidden_layers', 1)
            query = spec.num_speculative_tokens + 1
            if (getattr(text, 'model_type', None) != 'qwen3_5_text' or layers != 1 or
                    not 1 < query <= 8 or spec.kv_cache_dtype is not None or
                    config.cache_config.cache_dtype != 'fp8_e4m3' or
                    config.max_concurrent_batches != 1 or config.scheduler_config.async_scheduling or
                    config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none' or
                    config.num_lookahead_tokens != query - 1):
                raise ValueError('Unqualified Qwen MTP bounded transaction or FP8 cache')
            return cls(query, 0, config.model_config.max_model_len, layers, 'mtp')
        verify_draft_sources()
        if spec is None or spec.method != 'dflash':
            raise ValueError('Bounded speculation initially requires DFlash2')
        draft = spec.draft_model_config
        hf = draft.hf_config
        block = hf.dflash_config.get('block_size')
        query = spec.num_speculative_tokens + 1
        window = getattr(hf, 'sliding_window', 0)
        layers = hf.num_hidden_layers
        if (hf.architectures != ['DFlash2DraftModel'] or
                type(block) is not int or not 1 < query <= block or query > memory.block_tokens // 2 or
                type(window) is not int or window <= 0 or
                hf.layer_types != ['sliding_attention'] * layers or layers <= 0 or
                not hf.use_sliding_window or draft.dtype != torch.bfloat16):
            raise ValueError('Unqualified DFlash2 query, layer or KV geometry')
        if memory.recent_blocks * memory.block_tokens < window - 1:
            raise ValueError('Target mandatory recent band must contain the draft context window')
        # Persistent logical histories beyond the draft RoPE table require an
        # explicit frame-renormalization adapter; never silently clamp positions.
        if config.model_config.max_model_len > hf.max_position_embeddings:
            raise ValueError('Logical context exceeds the qualified draft position table')
        if config.max_concurrent_batches != 1 or config.scheduler_config.async_scheduling:
            raise ValueError('Initial bounded DFlash2 transactions require synchronous scheduling')
        if config.cache_config.enable_prefix_caching or config.cache_config.mamba_cache_mode != 'none':
            raise ValueError('Initial bounded DFlash2 transactions require APC-off GDN none state')
        if config.num_lookahead_tokens != query:
            raise ValueError('Native draft lookahead differs from the qualified query block')
        return cls(query, window, hf.max_position_embeddings, layers)

    def validate_native_groups(self, runner):
        import torch
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
        proposer = runner.speculator
        if self.method == 'mtp':
            names = set(proposer.draft_attn_layer_names)
            groups = tuple(i for i, group in enumerate(runner.kv_cache_config.kv_cache_groups)
                           if names.intersection(group.layer_names))
            covered = set()
            for i in groups:
                group = runner.kv_cache_config.kv_cache_groups[i]
                spec = group.kv_cache_spec
                specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
                from .cache_groups import _bounded_full_spec
                if any(not (type(s) is FullAttentionSpec or _bounded_full_spec(s)) or s.dtype not in (torch.float8_e4m3fn, torch.uint8) or
                       s.sliding_window is not None for s in specs):
                    raise ValueError('MTP requires lockstep full FP8 attention groups')
                covered.update(names.intersection(group.layer_names))
            if len(covered) != self.layers or covered != names or proposer.num_speculative_steps != self.query_tokens - 1:
                raise ValueError('MTP layer/query differs from the bounded contract')
            return groups
        groups = tuple(proposer.draft_kv_cache_group_ids)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError('Missing or duplicate native draft KV groups')
        layers = 0
        for group_id in groups:
            group = runner.kv_cache_config.kv_cache_groups[group_id]
            spec = group.kv_cache_spec
            specs = tuple(spec.kv_cache_specs.values()) if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,)
            if any(type(item) is not SlidingWindowSpec or item.sliding_window != self.sliding_window or
                   item.dtype != torch.bfloat16 for item in specs):
                raise ValueError('Draft cache must retain native BF16 sliding-window pages')
            layers += len(group.layer_names)
        if layers != self.layers or proposer.num_query_per_req != self.query_tokens:
            raise ValueError('Native draft groups/query differ from the model contract')
        return groups


MTP_SOURCES = {
    'v1/worker/gpu/spec_decode/mtp/speculator.py': '926d6c9ffdb368c971647911a60493820d056ce236fcae9d94a42fa2e006255d',
    'v1/worker/gpu/spec_decode/autoregressive/speculator.py': '5b97f54e20ddf65b98e939832d050745f67ef43cfa145786da300f1a8322e3c6',
    'model_executor/models/qwen3_5_mtp.py': 'a7550c4954899b38cecc536633c0fffd04820ed802610fb28cd4d79971b4f1f1',
}


@lru_cache(maxsize=1)
def verify_mtp_sources():
    from .integration import check_engine_version
    check_engine_version()
    package = distribution('vllm')
    for relative, expected in MTP_SOURCES.items():
        if hashlib.sha256(package.locate_file('vllm/' + relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Unqualified native MTP lifecycle source: {relative}')
