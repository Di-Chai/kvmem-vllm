"""Construction-device qualification on real FP8 physical-block geometry."""
import os

import pytest
import torch

from vllm_kvmem.serving.recall_scoring import build_block_features

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit GPU qualification only')


@pytest.mark.parametrize('method', ['fixed4', 'fps4', 'min_max', 'exact_mass'])
@pytest.mark.parametrize('seed', [317, 5931, 1024])
def test_fp8_feature_construction_matches_cpu_bytes_on_real_geometry(method, seed):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(16, 16, 32, 4, 256, generator=generator).to(torch.float8_e4m3fn)
    counts = [32] * 12 + [13, 3, 1, 0]
    expected = build_block_features(keys, counts, method)
    # Match the layer/block stride produced by the K-only archive gather.
    slab = keys.transpose(0, 1).contiguous()
    source = slab.permute(1, 0, 2, 3, 4).cuda()
    torch.cuda.reset_peak_memory_stats()
    actual = build_block_features(source, counts, method).to('cpu')
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.valid, expected.valid)
    assert torch.equal(actual.counts, expected.counts)
    assert torch.cuda.max_memory_allocated() < 512 * 1024**2


@pytest.mark.parametrize('method', ['fixed4', 'fps4', 'min_max', 'exact_mass'])
def test_zero_duplicate_ties_and_invalid_padding_keep_cpu_features(method):
    values = torch.zeros(2, 4, 32, 2, 256)
    values[:, 1, :2, :, :4] = torch.tensor([1., -1., 1., -1.])
    values[:, 2, :, :, :4] = torch.tensor([1., -1., 1., -1.])
    counts = [0, 2, 32, 1]
    for block, count in enumerate(counts):
        values[:, block, count:] = torch.nan
    keys = values.to(torch.float8_e4m3fn)
    expected = build_block_features(keys, counts, method)
    actual = build_block_features(keys.cuda(), counts, method).to('cpu')
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.valid, expected.valid)
