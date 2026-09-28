from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs
from vllm_kvmem.cache_groups import full_attention_groups
from vllm_kvmem.metadata import WorkingSetMetadata, WindowLengths


def test_native_mamba_slot_mapping_is_not_attention_window_eligibility():
    attention = FullAttentionSpec(block_size=1568, num_kv_heads=4, head_size=256, dtype=torch.float8_e4m3fn)
    recurrent = MambaSpec(block_size=1568, shapes=((4, 16),), dtypes=(torch.float32,))
    assert attention.uses_slot_mapping and recurrent.uses_slot_mapping
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s)
        for s in (recurrent, attention, recurrent, attention)])
    groups = full_attention_groups(config)
    assert groups == (1, 3)
    lens = torch.tensor([131073], dtype=torch.int32)
    metadata = WorkingSetMetadata(None, {g: WindowLengths(lens, lens, 131073) for g in groups})
    assert metadata.get_extra_common_attn_kwargs(0, 1) == {}
    assert metadata.get_extra_common_attn_kwargs(2, 1) == {}
    assert torch.equal(metadata.get_extra_common_attn_kwargs(1, 1)["seq_lens"], lens)
    wrapped = UniformTypeKVCacheSpecs(block_size=1568, kv_cache_specs={"attn": attention})
    assert full_attention_groups(SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=wrapped)])) == (0,)
