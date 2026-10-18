from types import SimpleNamespace as NS
import json

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')


def test_geometry_charges_shared_padded_pages_once_for_capacity(monkeypatch, tmp_path):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm_kvmem.cache_manifest import write_mean_k_geometry
    from vllm_kvmem.elastic import CacheGeometry
    monkeypatch.setattr('vllm_kvmem.options.RuntimeOptions.from_vllm', lambda _: NS(
        cpu_read_cache_bytes=100000, cpu_history_coordination=True, cpu_progress_reserve=True,
        memory=NS(select_budget=16, gen_budget=8,
            repository_blocks=2, block_tokens=128)))
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8,
                                  dtype=torch.float8_e4m3fn)
    recurrent = MambaSpec(block_size=256, shapes=((16,),), dtypes=(torch.float32,),
                          num_speculative_blocks=3)
    config = NS(compilation_config=NS(cudagraph_mode='FULL_DECODE_ONLY'))
    native = NS(num_blocks=100, kv_cache_tensors=[NS(size=32000), NS(size=32000)],
        kv_cache_groups=[NS(kv_cache_spec=attention, layer_names=['a']),
                         NS(kv_cache_spec=recurrent, layer_names=['r'])])
    write_mean_k_geometry(config, native, tmp_path)
    value = json.loads((tmp_path / 'cache-geometry.json').read_text())
    assert value['capacity_bytes'] == 32000  # one shared backing, not 64000
    assert value['schema'] == 2
    assert value['archive_codec'] == 'raw_fp8_e4m3fn_v_fp8_e4m3fn_v1'
    assert value['cpu_archive_bytes_per_token'] == 16
    assert [g['page_bytes'] for g in value['groups']] == [320, 320]
    assert [g['payload_page_bytes'] for g in value['groups']] == [64, 64]
    assert CacheGeometry(tuple(value['groups']), 32000).active_bytes(9) == (3 + 5) * 320
    native.kv_cache_tensors[1].size = 64000
    with pytest.raises(ValueError, match='one native'):
        write_mean_k_geometry(config, native, tmp_path)
