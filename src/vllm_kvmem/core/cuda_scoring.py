"""Original bounded qw3 FP16 page-softmax kernel, on the current Torch stream."""
from functools import lru_cache
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import shutil

import torch
from . import settings as _settings


_SOURCE_NAMES = ('qw3_mean_k.cu', 'qw3_mean_k_kernel.cuh', 'qw3_mean_k_tiled.cuh')
QUERY_CONVERSION_BYTES = 64 * 1024 * 1024


def _publish_source(path, content):
    """Publish one complete immutable source without touching an existing copy."""
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        pass
    else:
        if existing != content:
            raise RuntimeError(f'MeanK cached source content differs: {path}')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Another process may have published these same bytes meanwhile.
            # Unlike replace(), link-if-absent preserves its inode and mtime.
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise RuntimeError(f'MeanK cached source content differs: {path}')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _prepare_build_sources(source, build_root):
    # Snapshot all inputs before publishing any of them. Including boundaries
    # prevents different distributions across the three files sharing a key.
    contents = [(name, (source / name).read_bytes()) for name in _SOURCE_NAMES]
    key = hashlib.sha256()
    for name, content in contents:
        key.update(name.encode() + b'\0' + len(content).to_bytes(8, 'big') + content)
    digest = key.hexdigest()
    directory = build_root / ('kvmem_meank_' + digest[:12])
    # Keep the full content key in the input path even if module names share a
    # truncated prefix. All quoted local headers remain adjacent to the .cu.
    stable = directory / 'sources' / digest
    stable.mkdir(parents=True, exist_ok=True)
    for name, content in contents:
        _publish_source(stable / name, content)
    return directory, stable / _SOURCE_NAMES[0], digest[:12]


@lru_cache(maxsize=1)
def kernel():
    if shutil.which('ninja') is None:
        import ninja
        os.environ['PATH'] = str(ninja.BIN_DIR) + os.pathsep + os.environ.get('PATH','')
    os.environ.setdefault('MAX_JOBS','1')
    from torch.utils.cpp_extension import load
    source = Path(__file__).resolve().parents[1] / 'cuda_sources'   # the package root
    if not source.exists():
        source = Path(__file__).resolve().parents[3] / 'csrc'
    directory, stable_source, digest = _prepare_build_sources(source,
        Path(_settings.get('KVMEM_CUDA_BUILD_DIR', tempfile.gettempdir())).resolve())
    return load(name='kvmem_meank_' + digest, sources=[str(stable_source)],
        build_directory=str(directory), extra_cuda_cflags=['-O3'], verbose=False)


def global_backend():
    """Opt-in KVMEM_MEANK_GLOBAL_BACKEND for the global Query MeanK scoring.

    ``tiled`` uses qw3's own scalable kernels (already used above 8192 pages) for
    every index size. It is ~4x faster at final-Query sizes, but its per-page sums
    differ from the fused kernel's by ~2e-4 relative (the fused kernel adds each
    tiny term into a large FP32 total one at a time), so selections near the
    budget boundary can change: qualify end to end before use.
    """
    value = _settings.get('KVMEM_MEANK_GLOBAL_BACKEND', 'legacy')
    if value not in ('legacy', 'tiled'):
        raise ValueError('KVMEM_MEANK_GLOBAL_BACKEND must be legacy or tiled')
    return value


def mean_key_scores(queries, means, eligible, *, scale=None):
    if queries.ndim != 4 or means.ndim != 4 or eligible.shape != (means.shape[1],) or eligible.dtype != torch.bool:
        raise ValueError('Invalid MeanK Query/index/mask')
    if queries.device != means.device or eligible.device != queries.device:
        raise ValueError('MeanK tensors must share a device')
    if any(size <= 0 for size in queries.shape):
        raise ValueError('MeanK queries must have nonempty geometry')
    indices = eligible.nonzero().flatten()
    scores = torch.zeros(means.shape[1], device=queries.device, dtype=torch.float32)
    if not indices.numel():
        return scores
    # Compact only eligible pages before softmax, exactly excluding sink/recent
    # and mandatory pages. Scatter scores back without changing history order.
    compact_means = means.index_select(1, indices).to(
        dtype=torch.float16, memory_format=torch.contiguous_format)
    backend = global_backend()
    values = _compact_values(queries, compact_means, scale,
                             **({'backend': backend} if backend != 'legacy' else {}))
    scores.index_copy_(0, indices, values)
    return scores


def _compact_values(queries, compact_means, scale, *, backend='legacy'):
    if backend not in ('legacy', 'tiled'):
        raise ValueError('MeanK scoring backend must be legacy or tiled')
    attention_scale = queries.shape[-1] ** -.5 if scale is None else scale
    scorer = kernel()
    score = scorer.score if backend == 'legacy' else getattr(scorer, 'score_tiled', None)
    if not callable(score):
        raise RuntimeError('Explicit tiled MeanK requires the updated CUDA extension')
    if queries.dtype == torch.float16 and queries.is_contiguous():
        values = score(queries, compact_means, attention_scale)
    else:
        # The kernel sums over Query tokens and averages only layers/heads.
        # Keep its full page softmax domain while bounding dtype/layout copies;
        # a 2304-token BF16 Query otherwise duplicates 432 MiB at handoff.
        bytes_per_token = queries.shape[0] * queries.shape[2] * queries.shape[3] * 2
        chunk_tokens = max(1, QUERY_CONVERSION_BYTES // bytes_per_token)
        values = torch.zeros(compact_means.shape[1], device=queries.device, dtype=torch.float32)
        for begin in range(0, queries.shape[1], chunk_tokens):
            chunk = queries[:, begin:begin+chunk_tokens].to(
                dtype=torch.float16, memory_format=torch.contiguous_format)
            values.add_(score(chunk, compact_means, attention_scale))
            del chunk
    return values


@dataclass(frozen=True)
class PreparedMeanKeyIndex:
    indices: torch.Tensor
    means: torch.Tensor
    pages: int


def prepare_mean_key_index(means, eligible):
    """Freeze only the immutable candidate domain; never cache a Query or scores."""
    if (means.ndim!=4 or eligible.shape!=(means.shape[1],) or eligible.dtype!=torch.bool or
            eligible.device!=means.device):
        raise ValueError('Invalid static MeanK index domain')
    indices=eligible.nonzero().flatten()
    compact=means.index_select(1,indices).to(dtype=torch.float16,memory_format=torch.contiguous_format)
    return PreparedMeanKeyIndex(indices,compact,means.shape[1])


def prepared_layer_scores(queries, index, layer, *, scale=None, backend='legacy'):
    # This choice changes only execution. The immutable candidate mask, FP16
    # operands, query-token sum and head/layer averaging are retained.
    if backend not in ('legacy', 'tiled'):
        raise ValueError('MeanK scoring backend must be legacy or tiled')
    if (queries.ndim!=4 or queries.shape[0]!=1 or any(n<=0 for n in queries.shape) or
            not 0<=layer<index.means.shape[0] or queries.device!=index.means.device):
        raise ValueError('Current layer Query does not match its prepared index')
    scores=torch.zeros(index.pages,device=queries.device,dtype=torch.float32)
    if index.indices.numel():
        values=_compact_values(queries,index.means[layer:layer+1],scale,
                               **({'backend': backend} if backend != 'legacy' else {}))
        scores.index_copy_(0,index.indices,values)
    return scores
