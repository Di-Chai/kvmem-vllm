from types import SimpleNamespace as NS

import pytest
import torch

pytest.importorskip("vllm")

from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.draft_contract import DraftContract


def config():
    hf = NS(architectures=['DFlash2DraftModel'], dflash_config={'block_size': 8},
            sliding_window=2048, num_hidden_layers=5,
            layer_types=['sliding_attention'] * 5, use_sliding_window=True,
            max_position_embeddings=262144)
    return NS(speculative_config=NS(method='dflash', num_speculative_tokens=7,
                  draft_model_config=NS(hf_config=hf, dtype=torch.bfloat16)),
              model_config=NS(max_model_len=262144), max_concurrent_batches=1,
              scheduler_config=NS(async_scheduling=False), num_lookahead_tokens=8,
              cache_config=NS(enable_prefix_caching=False, mamba_cache_mode='none'))


def test_draft_contract_rejects_unqualified_reselection_context_or_position_frames():
    cfg, memory = config(), KVMemConfig()
    contract = DraftContract.from_vllm(cfg, memory)
    assert contract.query_tokens == 8 and contract.sliding_window == 2048
    with pytest.raises(ValueError, match='recent band'):
        DraftContract.from_vllm(cfg, KVMemConfig(recent_blocks=1))
    cfg.model_config.max_model_len += 1
    with pytest.raises(ValueError, match='position table'):
        DraftContract.from_vllm(cfg, memory)
    cfg = config()
    cfg.speculative_config.draft_model_config.hf_config.layer_types[0] = 'full_attention'
    with pytest.raises(ValueError, match='geometry'):
        DraftContract.from_vllm(cfg, memory)
    cfg = config()
    cfg.num_lookahead_tokens = 7
    with pytest.raises(ValueError, match='lookahead'):
        DraftContract.from_vllm(cfg, memory)


def test_draft_contract_checks_actual_native_cache_specs_not_slot_mapping_flags():
    pytest.importorskip('vllm')
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec, FullAttentionSpec
    spec = SlidingWindowSpec(block_size=128, num_kv_heads=8, head_size=128,
                             dtype=torch.bfloat16, sliding_window=2048)
    group = KVCacheGroupSpec([f'draft{i}' for i in range(5)], spec)
    runner = NS(speculator=NS(draft_kv_cache_group_ids=[0], num_query_per_req=8),
                kv_cache_config=KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[group]))
    contract = DraftContract.from_vllm(config(), KVMemConfig())
    assert contract.validate_native_groups(runner) == (0,)
    group.kv_cache_spec = FullAttentionSpec(block_size=128, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='sliding-window'):
        contract.validate_native_groups(runner)
