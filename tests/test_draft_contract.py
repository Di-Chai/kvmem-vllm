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


@pytest.mark.parametrize('draft_tokens', [1, 3])
def test_shorter_draft_verifies_a_bounded_prefix_of_the_trained_block(draft_tokens):
    cfg = config()
    cfg.speculative_config.num_speculative_tokens = draft_tokens
    cfg.num_lookahead_tokens = draft_tokens + 1
    assert DraftContract.from_vllm(cfg, KVMemConfig()).query_tokens == draft_tokens + 1
    cfg.speculative_config.num_speculative_tokens = 8
    cfg.num_lookahead_tokens = 9
    with pytest.raises(ValueError, match='geometry'):
        DraftContract.from_vllm(cfg, KVMemConfig())


@pytest.mark.parametrize('depth', [2, 3])
@pytest.mark.parametrize('backing', [torch.float8_e4m3fn, torch.uint8])
def test_native_mtp_has_lockstep_full_fp8_kv_and_distinct_lookahead(depth, backing):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    cfg = config()
    cfg.speculative_config = NS(method='mtp', num_speculative_tokens=depth,
        kv_cache_dtype=None, draft_model_config=NS(
            hf_text_config=NS(model_type='qwen3_5_text', mtp_num_hidden_layers=1)))
    cfg.num_lookahead_tokens = depth
    cfg.cache_config.cache_dtype = 'fp8_e4m3'
    contract = DraftContract.from_vllm(cfg, KVMemConfig())
    assert contract.method == 'mtp' and contract.query_tokens == depth + 1
    group = KVCacheGroupSpec(['target', 'mtp'], FullAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=256, dtype=backing))
    runner = NS(speculator=NS(draft_attn_layer_names={'mtp'}, num_speculative_steps=depth),
                kv_cache_config=KVCacheConfig(num_blocks=32, kv_cache_tensors=[], kv_cache_groups=[group]))
    assert contract.validate_native_groups(runner) == (0,)
    cfg.speculative_config.kv_cache_dtype = 'bfloat16'
    with pytest.raises(ValueError, match='FP8 cache'):
        DraftContract.from_vllm(cfg, KVMemConfig())
    group.kv_cache_spec = FullAttentionSpec(block_size=128, num_kv_heads=4,
                                          head_size=256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='lockstep full FP8'):
        contract.validate_native_groups(runner)
