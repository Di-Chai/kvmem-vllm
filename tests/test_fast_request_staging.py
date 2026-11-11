"""KVMEM_FAST_REQUEST_STAGING: RoPE prefill positions staged as int32 arrays.

The CPU tests drive vLLM's own ``stage_write`` and ``UvaBufferPool.copy_to_uva``
on host buffers and compare everything the write kernel would receive.
"""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBufferPool
from vllm.v1.worker.gpu.mm.rope import RopeState
from vllm_kvmem import fast_request_staging as frs


class HostBuffer:
    def __init__(self, size, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()

    def uva(self, n=None):
        return self.cpu[:n].clone()


def host_pool(dtype):
    pool = UvaBufferPool.__new__(UvaBufferPool)
    pool.size, pool.dtype, pool.max_concurrency = 1, dtype, 2
    pool._buffer_cls = HostBuffer
    pool._uva_bufs = [HostBuffer(1, dtype) for _ in range(2)]
    pool._curr = 0
    return pool


def host_rope(num_dims=3):
    """A RopeState with vLLM's staging lists and host-memory pools (no CUDA)."""
    tensor = StagedWriteTensor.__new__(StagedWriteTensor)
    tensor.dtype, tensor.num_rows = torch.int32, 64
    tensor._staged_write_indices, tensor._staged_write_starts = [], []
    tensor._staged_write_contents, tensor._staged_write_cu_lens = [], []
    tensor.write_contents = host_pool(torch.int32)
    rope = RopeState.__new__(RopeState)
    rope.num_dims, rope.prefill_positions = num_dims, tensor
    rope.prefill_delta = NS(np=np.zeros(16, np.int32))
    return rope


class Model:
    def __init__(self, positions, delta=5):
        self.positions, self.delta, self.calls = positions, delta, []

    def get_mrope_input_positions(self, tokens, features):
        self.calls.append((tokens, features))
        return self.positions, self.delta


def staged(rope):
    """What apply_write hands to the kernel, then cleared like apply_write."""
    tensor = rope.prefill_positions
    contents = tensor.write_contents.copy_to_uva(tensor._staged_write_contents)
    result = (list(tensor._staged_write_indices), list(tensor._staged_write_starts),
              list(tensor._staged_write_cu_lens), contents.dtype, contents.tolist(),
              list(tensor._staged_write_contents), rope.prefill_delta.np.tolist())
    tensor.clear_staged_writes()
    assert len(tensor._staged_write_contents) == 0 and not list(tensor._staged_write_contents)
    return result


def pair():
    plain, fast = host_rope(), host_rope()
    assert frs.install_rope(fast, check_sources=False)
    return plain, fast


def text_positions(length, begin=0):
    return torch.from_numpy(np.broadcast_to(np.arange(length), (3, length)) + begin)


def test_flag_defaults_off_and_is_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_FAST_REQUEST_STAGING', raising=False)
    assert frs.enabled() is False and frs.install(NS()) is False
    monkeypatch.setenv('KVMEM_FAST_REQUEST_STAGING', '1')
    assert frs.enabled() is True
    monkeypatch.setenv('KVMEM_FAST_REQUEST_STAGING', 'on')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        frs.enabled()


def test_validated_vllm_sources_are_the_installed_ones():
    assert frs.source_digests() == frs._SOURCES


@pytest.mark.parametrize('positions', [
    text_positions(1), text_positions(1000), text_positions(70000, 3),
    torch.arange(3 * 257, dtype=torch.int32).reshape(3, 257),
    torch.arange(3 * 64).reshape(64, 3).T,                       # strided rows
    torch.tensor([[2 ** 31 - 1, 0], [-2 ** 31, 1], [5, 6]]),     # int32 limits
    torch.zeros((3, 0), dtype=torch.int64),                      # nothing to stage
    torch.arange(12, dtype=torch.int16).reshape(3, 4),           # list path
    torch.arange(12, dtype=torch.float32).reshape(3, 4),         # list path
])
def test_staged_writes_match_the_list_staging(positions):
    plain, fast = pair()
    for req_idx in (3, 0):
        for rope in (plain, fast):
            model = Model(positions, delta=7 + req_idx)
            rope.init_prefill_positions(req_idx, model, [1, 2, 3], ['f'])
            assert model.calls == [([1, 2, 3], ['f'])]
    assert staged(plain) == staged(fast)
    expected_arrays = positions.dtype in (torch.int32, torch.int64) and positions.shape[1] > 0
    assert (fast._kvmem_fast_staging['array_rows'] == 6) is bool(expected_arrays)


def test_several_requests_and_vllm_writes_share_one_batch():
    plain, fast = pair()
    for rope in (plain, fast):
        rope.init_prefill_positions(1, Model(text_positions(300)), [0], [])
        rope.prefill_positions.stage_write(40, 2, [7, 8, 9])             # vLLM's own entry points
        rope.prefill_positions.stage_write(41, 0, [])
        rope.prefill_positions.stage_write_elem(42, 11)
        rope.init_prefill_positions(2, Model(text_positions(5, 100)), [0], [])
        rope.prefill_positions.stage_write(43, 1, range(4))
    first = staged(plain)
    assert first == staged(fast) and first[2][-1] == 3 * 300 + 3 + 1 + 3 * 5 + 4
    for rope in (plain, fast):                                           # the next batch starts empty
        rope.init_prefill_positions(0, Model(text_positions(9)), [0], [])
    assert staged(plain) == staged(fast)


def test_out_of_range_positions_fail_like_the_list():
    errors = []
    for rope in pair():
        rope.init_prefill_positions(0, Model(torch.tensor([[1, 2 ** 31], [0, 0], [0, 0]])), [0], [])
        with pytest.raises(OverflowError) as error:
            staged(rope)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_staged_arrays_do_not_alias_the_model_positions():
    plain, fast = pair()
    positions = text_positions(50).clone()
    for rope in (plain, fast):
        rope.init_prefill_positions(0, Model(positions), [0], [])
    positions.zero_()
    result = staged(fast)
    assert result == staged(plain) and result[4][:50] == list(range(50))


def test_install_refuses_anything_but_the_validated_state(monkeypatch):
    rope = host_rope()
    rope.prefill_positions._staged_write_contents.append(1)              # writes already staged
    assert not frs.install_rope(rope, check_sources=False)
    rope = host_rope()
    rope.prefill_positions.write_contents = None                         # GPU-resident tensor
    assert not frs.install_rope(rope, check_sources=False)
    rope = host_rope()
    rope.prefill_positions.dtype = torch.int64
    assert not frs.install_rope(rope, check_sources=False)
    assert not frs.install_rope(NS(prefill_positions=host_rope().prefill_positions), check_sources=False)
    rope = host_rope()
    monkeypatch.setitem(frs._SOURCES, 'StagedWriteTensor.apply_write', 'another vLLM')
    assert not frs.install_rope(rope) and type(rope.prefill_positions._staged_write_contents) is list
    monkeypatch.undo()
    assert frs.install_rope(rope) and not frs.install_rope(rope)         # once


def test_replaced_staging_list_returns_to_vllm():
    plain, fast = pair()
    fast.prefill_positions._staged_write_contents = []                   # someone restored a list
    for rope in (plain, fast):
        rope.init_prefill_positions(2, Model(text_positions(40)), [0], [])
    assert staged(plain) == staged(fast) and fast._kvmem_fast_staging['array_rows'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_gpu_positions_are_identical_after_vllm_apply():
    device = torch.device('cuda')
    ropes = [RopeState(3, 4, 4096, 3000, device) for _ in range(2)]
    assert frs.install_rope(ropes[1])
    lengths = {0: 3000, 2: 1, 3: 1777}
    for rope in ropes:
        for req_idx, length in lengths.items():
            rope.init_prefill_positions(req_idx, Model(text_positions(length, req_idx), delta=req_idx + 9), [0], [])
        rope.apply_staged_writes()
        rope.init_prefill_positions(1, Model(text_positions(2048, 5), delta=1), [0], [])
        rope.apply_staged_writes()
    torch.cuda.synchronize()
    assert torch.equal(ropes[0].prefill_positions.gpu.cpu(), ropes[1].prefill_positions.gpu.cpu())
    assert torch.equal(ropes[0].prefill_delta.gpu.cpu(), ropes[1].prefill_delta.gpu.cpu())
    assert ropes[1]._kvmem_fast_staging == {'array_rows': 12, 'list_rows': 0}
    assert ropes[0].prefill_positions.gpu[9, :1777].tolist() == list(range(3, 1780))
