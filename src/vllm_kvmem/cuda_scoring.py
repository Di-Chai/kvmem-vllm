"""Original bounded qw3 FP16 page-softmax kernel, on the current Torch stream."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import tempfile
import shutil

import torch


@lru_cache(maxsize=1)
def kernel():
    if shutil.which('ninja') is None:
        import ninja
        os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ.get('PATH','')
    os.environ.setdefault('MAX_JOBS','1')
    from torch.utils.cpp_extension import load
    source = Path(__file__).parent / 'cuda_sources'
    if not source.exists():
        source = Path(__file__).resolve().parents[2] / 'csrc'
    digest = hashlib.sha256(b''.join((source / n).read_bytes()
        for n in ('qw3_mean_k.cu','qw3_mean_k_kernel.cuh'))).hexdigest()[:12]
    directory = Path(os.getenv('KVMEM_CUDA_BUILD_DIR', tempfile.gettempdir())) / ('kvmem_meank_' + digest)
    directory.mkdir(parents=True, exist_ok=True)
    return load(name='kvmem_meank_' + digest, sources=[str(source / 'qw3_mean_k.cu')],
        build_directory=str(directory), extra_cuda_cflags=['-O3'], verbose=False)


def mean_key_scores(queries, means, eligible, *, scale=None):
    if queries.ndim != 4 or means.ndim != 4 or eligible.shape != (means.shape[1],) or eligible.dtype != torch.bool:
        raise ValueError('Invalid MeanK Query/index/mask')
    if queries.device != means.device or eligible.device != queries.device:
        raise ValueError('MeanK tensors must share a device')
    indices = eligible.nonzero().flatten()
    scores = torch.zeros(means.shape[1], device=queries.device, dtype=torch.float32)
    if not indices.numel():
        return scores
    # Compact only eligible pages before softmax, exactly excluding sink/recent
    # and mandatory pages. Scatter scores back without changing history order.
    values = kernel().score(queries.to(torch.float16).contiguous(),
        means.index_select(1, indices).to(torch.float16).contiguous(),
        queries.shape[-1] ** -.5 if scale is None else scale)
    scores.index_copy_(0, indices, values)
    return scores
