"""Compare serving scores directly against qw3's unmodified CUDA definition."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='explicit GPU opt-in required')


@pytest.fixture(scope='module')
def original_kernel(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    source = root / 'tests/cuda/meank_reference.cu'
    nvcc = shutil.which('nvcc') or '/usr/local/cuda/bin/nvcc'
    if not Path(nvcc).is_file():
        pytest.skip('CUDA compiler unavailable')
    output = tmp_path_factory.mktemp('meank-cuda') / 'reference.so'
    subprocess.run([nvcc, '-shared', '-Xcompiler', '-fPIC', '-O2',
        '-arch=sm_120', '-I' + str(root / 'third_party'), str(source),
        '-o', str(output)], check=True, capture_output=True, text=True)
    library = ctypes.CDLL(str(output))
    function = library.meank_reference
    function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_uint32] * 8 + [ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


@pytest.mark.parametrize('blocks,tokens,sink,recent', [(7, 1, 1, 1), (257, 7, 8, 16),
                                                      (2048, 17, 8, 16)])
def test_mean_scores_and_topk_match_original_cuda(original_kernel, blocks, tokens, sink, recent):
    import torch
    from vllm_kvmem.retrieval import mean_key_scores
    generator = torch.Generator(device='cuda').manual_seed(7341)
    layers, heads, kv_heads, dim = 3, 24, 4, 256
    queries = torch.randn(layers, tokens, heads, dim, generator=generator, device='cuda')
    means = torch.randn(layers, blocks, kv_heads, dim, generator=generator, device='cuda')
    expected = torch.empty(blocks, device='cuda')
    error = original_kernel(expected.data_ptr(), queries.data_ptr(), means.data_ptr(),
        layers, tokens, blocks, heads, kv_heads, dim, sink, recent,
        torch.cuda.current_stream().cuda_stream)
    assert error == 0
    eligible = torch.zeros(blocks, dtype=torch.bool, device='cuda')
    eligible[sink:blocks-recent] = True
    actual = mean_key_scores(queries, means, eligible)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    topk = min(32, blocks - sink - recent)
    assert set(actual.topk(topk).indices.tolist()) == set(expected.topk(topk).indices.tolist())
