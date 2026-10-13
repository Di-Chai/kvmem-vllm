"""Full-context admission uses physical pool slots, including hybrid padding."""
import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('vllm')

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs

from vllm_kvmem.elastic import CacheGeometry
from vllm_kvmem.elastic_connector import KVMemElasticConnector


@pytest.fixture
def native_geometry(monkeypatch, tmp_path):
    # Exercise the real connector's manifest generation without starting an
    # offload worker or allocating a GPU. Specs and admission are native types.
    monkeypatch.setattr('vllm_kvmem.elastic_connector.check_engine_version', lambda: None)
    monkeypatch.setattr('vllm_kvmem.elastic_connector.OffloadingConnector.__init__',
                        lambda self, config, role, cache: None)
    attention = FullAttentionSpec(block_size=4, num_kv_heads=1, head_size=8,
                                  dtype=torch.float8_e4m3fn)
    recurrent = MambaSpec(block_size=256, shapes=((16,),), dtypes=(torch.float32,),
                          num_speculative_blocks=3)
    uniform = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs={'b': attention, 'c': attention})
    config = NS(cache_config=NS(mamba_cache_mode='align', enable_prefix_caching=True),
        kv_transfer_config=NS(kv_connector_extra_config={
            'kvmem_report_dir': str(tmp_path), 'cpu_bytes_to_use': 100000}),
        scheduler_config=NS(async_scheduling=True),
        compilation_config=NS(cudagraph_mode='FULL_AND_PIECEWISE'))
    native = NS(num_blocks=100, kv_cache_tensors=[NS(size=32000), NS(size=32000)],
        kv_cache_groups=[NS(kv_cache_spec=attention, layer_names=['a']),
                         NS(kv_cache_spec=uniform, layer_names=['b', 'c']),
                         NS(kv_cache_spec=recurrent, layer_names=['r'])])
    return config, native, tmp_path / 'cache-geometry.json'


def test_full_mode_reserves_shared_pool_padding(native_geometry):
    config, native, path = native_geometry
    KVMemElasticConnector(config, KVConnectorRole.SCHEDULER, native)
    value = json.loads(path.read_text())
    assert value['capacity_bytes'] == 32000  # Aliases do not add capacity.
    assert [g['page_bytes'] for g in value['groups']] == [320, 320, 320]
    assert [g['payload_page_bytes'] for g in value['groups']] == [64, 128, 64]
    # Nine tokens need three slots in each attention group, plus five GDN
    # slots for running state, boundary CoW and speculative rows.
    geometry = CacheGeometry(tuple(value['groups']), value['capacity_bytes'])
    assert geometry.active_bytes(9) == (3 + 3 + 5) * 320
    assert value['semantic_mode'] == 'full_context_native_aligned_checkpoint'
    assert value['cpu_offload_bytes'] == 100000
    assert value['async_scheduling'] is True


@pytest.mark.parametrize(('sizes', 'blocks', 'error'), [
    ([], 100, 'one native'),
    ([32000, 64000], 100, 'one native'),
    ([32000], 0, 'one native'),
    ([32001], 100, 'integral shared page pool'),
])
def test_full_mode_rejects_unsupported_pool_geometry(native_geometry, sizes, blocks, error):
    config, native, path = native_geometry
    native.kv_cache_tensors = [NS(size=size) for size in sizes]
    native.num_blocks = blocks
    with pytest.raises(ValueError, match=error):
        KVMemElasticConnector(config, KVConnectorRole.SCHEDULER, native)
    assert not path.exists()


def test_worker_does_not_publish_geometry(native_geometry):
    config, _, path = native_geometry
    KVMemElasticConnector(config, KVConnectorRole.WORKER, None)
    assert not path.exists()
