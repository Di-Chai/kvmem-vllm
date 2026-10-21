import importlib.metadata
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prefill_sample_cache import patch_kernel,patch_variant
from vllm_kvmem.native_attention_build import patch_prefill


def upstream():
    try:
        distribution=importlib.metadata.distribution('flashinfer-python')
    except importlib.metadata.PackageNotFoundError:
        pytest.skip('Frozen FlashInfer source is unavailable')
    path=distribution.locate_file('flashinfer/data/include/flashinfer/attention/prefill.cuh')
    if not path.is_file():
        pytest.skip('FlashInfer wheel omits its source headers')
    return path.read_text()


def test_cache_loads_after_pdl_wait_before_all_kv_loops():
    old=patch_prefill(upstream())
    new=patch_kernel(old)
    anchor=new.index('void BatchPrefillWithPagedKVCacheDevice(')
    wait=new.index('asm volatile("griddepcontrol.wait;");',anchor)
    allocation=new.index('__shared__ int32_t kvmem_sample_cache',anchor)
    fill=new.index('params.stat_rows[params.stat_offset',allocation)
    split=new.index('// Split-D over the VO dimension',allocation)
    assert wait < allocation < fill < split
    assert 'static_assert(CTA_TILE_Q == NUM_WARPS_Q * NUM_MMA_Q * 16)' in new
    assert 'kvmem_sample_cache, kvmem_tile_begin);' in new
    assert new.count('__shared__ int32_t kvmem_sample_cache')==1
    assert new[:anchor]==old[:anchor]
    with pytest.raises(ValueError,match='already patched'):
        patch_kernel(new)


def test_variant_preserves_entire_floating_body_and_original_classes():
    path=Path(__file__).resolve().parents[1]/'src/vllm_kvmem/native_prefill_stats.cuh'
    old=path.read_text();new=patch_variant(old)
    marker=old.index('struct KVMemStaticAlignedAttention :')
    assert old[:marker]==new[:marker]
    restored=new.replace('KVMemStaticCachedAttention','KVMemStaticAlignedAttention').replace(
        'uint32_t kv_head, const int32_t* kvmem_samples, uint32_t kvmem_tile_begin) const {',
        'uint32_t kv_head) const {').replace(
        'int sample = kvmem_samples[packed + m * 16 + tid.x / 4 + 8 * r - kvmem_tile_begin];',
        'int sample = stats.rows[p.q_indptr[batch] + q];')
    assert restored==old
    with pytest.raises(ValueError):
        patch_variant(new)


@pytest.mark.parametrize('source',['','void BatchPrefillWithPagedKVCacheDevice( nothing',
                                  'void BatchPrefillWithPagedKVCacheDevice( '*2])
def test_unknown_kernel_source_rejected(source):
    with pytest.raises(ValueError):
        patch_kernel(source)
