"""Opt-in KVMEM_FAST_REQUEST_STAGING=1: stage a request's RoPE positions as arrays.

vLLM's GPU runner stages the M-RoPE positions of every new request as Python
lists: one ``tolist()`` per position channel, a list ``extend``, and one
element-by-element conversion of the whole list back to int32 when the staged
writes are applied. That is one Python object per prompt token and channel on
the engine thread, for every request (about 30 ms per turn at 131K tokens).

With the flag the runner's ``RopeState`` keeps those rows as int32 arrays. The
staged indices, starts, cumulative lengths and contents that reach vLLM's
write kernel are the same values; ``apply_write`` and the kernel are vLLM's
own, unmodified code. Rows the fast path does not cover (anything but a 1-D
CPU int32/int64 tensor whose values fit int32) go through vLLM's original
``stage_write`` with the same list as before.

The stand-in is installed only when the vLLM code it relies on is the code it
was written against (source digests below); otherwise nothing is changed.
"""
import hashlib
import inspect
import os

import numpy as np
import torch


# sha256 of inspect.getsource() of the vLLM functions whose behaviour the
# stand-in reproduces or relies on.
_SOURCES = {
    'RopeState.init_prefill_positions': '9ebad64a77a6a6b55fd9e40fffa84817e0a0a7c5793f7ac946f9e0f4f2cc376c',
    'StagedWriteTensor.stage_write': 'f069a415b9f7fc4376fa77c3380d12250e519e2b946748f3a7a7d6b03a1a139e',
    'StagedWriteTensor.apply_write': '45d692434b6d5c8372506db93e9c64fa87b1b10a3937ab40235fd9d690c659fd',
    'StagedWriteTensor.clear_staged_writes': '7c768de4feb398833be1a722be960958ebffb8aaf77986da47b4231e9f13b022',
    'UvaBufferPool.copy_to_uva': '564fd057d5775acc5b529c8b519637d0c54b28a6c9f508ce7f7e7edb1db1a907',
}


def enabled():
    value = os.getenv('KVMEM_FAST_REQUEST_STAGING', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_FAST_REQUEST_STAGING must be 0 or 1')
    return value == '1'


class StagedChunks:
    """List stand-in for ``StagedWriteTensor._staged_write_contents``.

    Supports exactly what vLLM does with that list: ``extend``, ``append``,
    ``len``, ``clear``, iteration, and assignment into a NumPy array of the
    tensor's dtype (``__array__``). Array chunks are stored as they are; other
    values are kept as Python lists and converted by NumPy at the same point,
    with the same element conversion, as the original list.
    """
    __slots__ = ('dtype', 'parts', 'count')

    def __init__(self, dtype):
        self.dtype, self.parts, self.count = np.dtype(dtype), [], 0

    def __len__(self):
        return self.count

    def extend(self, values):
        if not (type(values) is np.ndarray and values.dtype == self.dtype and values.ndim == 1):
            values = list(values)
        self.parts.append(values)
        self.count += len(values)

    def append(self, value):
        self.parts.append([value])
        self.count += 1

    def clear(self):
        self.parts.clear()
        self.count = 0

    def __iter__(self):
        for part in self.parts:
            yield from (part.tolist() if type(part) is np.ndarray else part)

    def __array__(self, dtype=None, copy=None):
        if all(type(part) is np.ndarray for part in self.parts):
            out = np.empty(self.count, self.dtype)
            offset = 0
            for part in self.parts:
                out[offset:offset+len(part)] = part
                offset += len(part)
        else:
            # NumPy converts the complete list, as it did for the plain list.
            out = np.empty(self.count, self.dtype)
            out[:] = list(self)
        return out if dtype is None else out.astype(dtype, copy=False)


def _int32_row(row):
    """An owned int32 copy of a position row, or None for the list path."""
    if type(row) is not torch.Tensor or row.dim() != 1 or row.device.type != 'cpu' or row.requires_grad:
        return None
    if row.dtype == torch.int32:
        return row.numpy().astype(np.int32)
    if row.dtype != torch.int64:
        return None
    values = row.numpy()
    if values.size and (int(values.min()) < -2**31 or int(values.max()) > 2**31-1):
        return None                     # the list conversion reports the overflow
    return values.astype(np.int32)


def _digest(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def source_digests():
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBufferPool
    from vllm.v1.worker.gpu.mm.rope import RopeState
    return {
        'RopeState.init_prefill_positions': _digest(RopeState.init_prefill_positions),
        'StagedWriteTensor.stage_write': _digest(StagedWriteTensor.stage_write),
        'StagedWriteTensor.apply_write': _digest(StagedWriteTensor.apply_write),
        'StagedWriteTensor.clear_staged_writes': _digest(StagedWriteTensor.clear_staged_writes),
        'UvaBufferPool.copy_to_uva': _digest(UvaBufferPool.copy_to_uva),
    }


def install_rope(rope, check_sources=True):
    """Install the array staging on one RopeState; returns whether it did."""
    from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor
    from vllm.v1.worker.gpu.mm.rope import RopeState
    if type(rope) is not RopeState:
        return False
    tensor = rope.prefill_positions
    if (type(tensor) is not StagedWriteTensor or tensor.write_contents is None or
            tensor.dtype != torch.int32 or type(tensor._staged_write_contents) is not list or
            tensor._staged_write_contents or 'init_prefill_positions' in vars(rope)):
        return False
    if check_sources:
        try:
            found = source_digests()
        except (OSError, TypeError):
            return False
        if found != _SOURCES:
            return False
    contents = tensor._staged_write_contents = StagedChunks(np.int32)
    original = rope.init_prefill_positions
    counts = rope._kvmem_fast_staging = {'array_rows': 0, 'list_rows': 0}

    def init_prefill_positions(req_idx, model, prefill_token_ids, mm_features):
        if tensor._staged_write_contents is not contents:
            return original(req_idx, model, prefill_token_ids, mm_features)
        positions, delta = model.get_mrope_input_positions(prefill_token_ids, mm_features)
        rope.prefill_delta.np[req_idx] = delta
        for i in range(rope.num_dims):
            row = positions[i]
            index = rope.num_dims*req_idx + i
            values = _int32_row(row)
            if values is None:
                counts['list_rows'] += 1
                tensor.stage_write(index, 0, row.tolist())
                continue
            # StagedWriteTensor.stage_write(index, 0, values.tolist())
            assert index >= 0
            if not len(values):
                continue
            tensor._staged_write_indices.append(index)
            tensor._staged_write_starts.append(0)
            contents.extend(values)
            tensor._staged_write_cu_lens.append(len(contents))
            counts['array_rows'] += 1

    rope.init_prefill_positions = init_prefill_positions
    return True


def install(runner):
    """Called once after the model state exists; a no-op unless opted in."""
    if not enabled():
        return False
    rope = getattr(getattr(runner, 'model_state', None), 'rope_state', None)
    installed = rope is not None and install_rope(rope)
    from vllm.logger import init_logger
    logger = init_logger('vllm.kvmem.fast_request_staging')
    if installed:
        logger.info('KVMEM_FAST_REQUEST_STAGING: RoPE prefill positions are staged as int32 arrays')
    else:
        logger.warning('KVMEM_FAST_REQUEST_STAGING: not installed (no M-RoPE state, or the vLLM staging '
                       'code differs from the validated version); requests are staged by vLLM as usual')
    return installed
