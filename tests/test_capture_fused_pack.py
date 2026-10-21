"""Host qualification for bounded, opt-in capture packing; no CUDA allocation."""
from dataclasses import fields, replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.capture_pack import PackLimits, SourceBinding, eligibility, source_signature, try_pack
from vllm_kvmem.capture_runtime import CaptureRuntime
from vllm_kvmem.options import RuntimeOptions
from test_options import config


class MetadataTensor:
    def __init__(self, shape, dtype, strides=None, pointer=128):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.device = len(shape), torch.device('cuda:0')
        contiguous, size = [], 1
        for dim in reversed(shape):
            contiguous.insert(0, size)
            size *= dim
        self.strides, self.size, self.pointer = tuple(strides or contiguous), size, pointer

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self):
        return self.size

    def data_ptr(self):
        return self.pointer

    def item(self):
        pytest.fail('Eligibility must not inspect CUDA values')


def metadata(tokens=128, requests=3):
    limits = PackLimits(17, 4, 256, 2048, 8)
    local = MetadataTensor((2048,), torch.int64)
    captured = [NS(key=MetadataTensor((2048, 4, 256), torch.bfloat16, pointer=i*1024+128),
                   value=MetadataTensor((2048, 4, 256), torch.bfloat16, pointer=i*1024+256),
                   positions=local) for i in range(17)]
    scales = MetadataTensor((17,), torch.float32)
    batch = NS(num_tokens=tokens, num_reqs=requests,
        query_start_loc=MetadataTensor((9,), torch.int32),
        positions=MetadataTensor((2048,), torch.int64))
    rejected = MetadataTensor((8,), torch.int32)
    return captured, batch, rejected, scales, limits


def check(parts, indices=None, raw_dtype=torch.float8_e4m3fn):
    captured, batch, rejected, scales, limits = parts
    return eligibility(captured, batch, rejected,
        tuple(range(batch.num_reqs)) if indices is None else indices, scales, raw_dtype, limits)


def test_exact_whole_batch_and_bounded_metadata_only_eligibility():
    parts = metadata()
    assert check(parts) is None
    assert check(parts, (0, 2)) == 'split_or_reordered_owners'
    assert check(parts, (2, 1, 0)) == 'split_or_reordered_owners'
    assert check(parts, (0, 0, 2)) == 'split_or_reordered_owners'
    assert check(parts, raw_dtype=torch.bfloat16) == 'raw_dtype'
    assert check(metadata(tokens=2049)) == 'batch_capacity'
    assert check(metadata(tokens=2)) == 'batch_capacity'
    assert check(metadata(requests=9)) == 'batch_capacity'
    assert check(metadata(tokens=0, requests=0)) == 'batch_capacity'


@pytest.mark.parametrize('field,value', [('dtype', torch.float32),
    ('shape', (127, 4, 256)), ('shape', (2048, 8, 128)),
    ('strides', (1024, 256, 2)), ('strides', (1024, 128, 1)),
    ('strides', (512, 256, 1)), ('device', torch.device('cuda:1'))])
def test_unsupported_source_is_explicit_fallback(field, value):
    parts = metadata()
    setattr(parts[0][-1].value, field, value)
    assert check(parts) == 'source_layout'


def test_noncontiguous_rows_qualify_but_two_dimensional_position_header_does_not():
    parts = metadata()
    parts[0][-1].key.strides = (3072, 256, 1)
    assert check(parts) is None
    parts[0][0].positions = MetadataTensor((3, 2048), torch.int64)
    assert check(parts) == 'header_layout'


@pytest.mark.parametrize('target,field,value,reason', [
    ('scales', 'dtype', torch.bfloat16, 'scale_layout'),
    ('scales', 'strides', (2,), 'scale_layout'),
    ('scales', 'size', 16, 'scale_layout'),
    ('scales', 'device', torch.device('cpu'), 'scale_layout'),
    ('starts', 'dtype', torch.float32, 'header_layout'),
    ('starts', 'size', 3, 'header_layout'),
    ('global', 'strides', (2,), 'header_layout'),
    ('rejected', 'device', torch.device('cuda:1'), 'header_layout'),
])
def test_header_and_scale_metadata_qualification(target, field, value, reason):
    parts = metadata()
    tensors = dict(scales=parts[3], starts=parts[1].query_start_loc,
                   global_=parts[1].positions, rejected=parts[2])
    setattr(tensors['global_' if target == 'global' else target], field, value)
    assert check(parts) == reason


def test_one_generation_signature_covers_capacity_pointer_stride_and_scale():
    captured, _, _, scales, _ = metadata()
    original = source_signature(captured, scales)
    for tensor, field, changed in [(captured[-1].key, 'shape', (1024, 4, 256)),
            (captured[0].value, 'pointer', 2048),
            (captured[2].key, 'strides', (2048, 256, 1)), (scales, 'pointer', 4096)]:
        old = getattr(tensor, field)
        setattr(tensor, field, changed)
        assert source_signature(captured, scales) != original
        setattr(tensor, field, old)
    binding = SourceBinding()
    binding.signature, binding.stream, binding.owners = original, 7, tuple(captured)
    binding.event = NS(query=lambda: False,
        synchronize=lambda: pytest.fail('Generation changes may not wait on GPU'))
    assert binding.can_replace(original, 7)
    assert not binding.can_replace(original, 8)
    assert not binding.can_replace(('changed',), 7)
    assert len(binding.owners) == 17  # A rejected replacement keeps the previous generation.
    binding.event = NS(query=lambda: True)
    assert binding.can_replace(('changed',), 8)


def test_capacity_accounts_for_single_pinned_and_gpu_table_and_largest_payload():
    limits = PackLimits(17, 4, 256, 2048, 8)
    assert limits.metadata_bytes == 1088
    assert limits.payload_bytes == 2*17*4*256*2048 + 33*8
    for field in ('layers', 'heads', 'dim', 'max_tokens', 'max_requests'):
        for bad in (0, -1, True, 1.5):
            with pytest.raises(ValueError, match='positive integers'):
                replace(limits, **{field: bad})


def test_cache_is_never_reallocated_after_geometry_change(monkeypatch):
    import vllm_kvmem.capture_pack as pack
    captured, batch, rejected, scales, limits = metadata()
    existing = NS(limits=replace(limits, max_tokens=4096), device=scales.device)
    runtime = NS(scales=scales, raw_dtype=torch.float8_e4m3fn,
                 _capture_pack_limits=limits, _capture_pack_cache=existing)
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: False)
    monkeypatch.setattr(pack, 'FusedCapturePacker', lambda *_: pytest.fail('Unbounded replacement cache'))
    assert try_pack(runtime, captured, batch, rejected, (0, 1, 2)) == (None, 'cache_geometry')
    assert runtime._capture_pack_cache is existing
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: True)
    assert try_pack(runtime, captured, batch, rejected, (0, 1, 2)) == (None, 'graph_capture')


@pytest.mark.parametrize('enabled', [False, True])
def test_opt_in_round_trip_preserves_profile_and_old_positional_parameters(enabled):
    payload = {'kvmem': {'capture_fused_pack': enabled}}
    assert RuntimeOptions.from_vllm(config(payload)).capture_fused_pack is enabled
    assert payload == {'kvmem': {'capture_fused_pack': enabled}}
    baseline = RuntimeOptions()
    names = [field.name for field in fields(RuntimeOptions)]
    assert names.index('capture_fused_pack') == names.index('cpu_seal_hash_workers')+1
    assert not baseline.capture_fused_pack
    assert RuntimeOptions(*(getattr(baseline, key) for key in names[:-1])) == baseline


@pytest.mark.parametrize('bad', [1, 0, None, 'true'])
def test_capture_pack_invalid_option_is_rejected(bad):
    with pytest.raises(ValueError, match='capture_fused_pack must be boolean'):
        RuntimeOptions.from_vllm(config({'kvmem': {'capture_fused_pack': bad}}))


def test_cpu_fallback_preserves_reference_bytes_and_legacy_fixture_without_options():
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.scales, runtime.raw_dtype = torch.tensor([.3]), torch.float8_e4m3fn
    positions = torch.arange(8, dtype=torch.int64) + 11
    captured = [NS(key=torch.arange(16).reshape(8, 1, 2).bfloat16(),
                   value=torch.arange(16).reshape(8, 1, 2).bfloat16(), positions=positions)]
    batch = NS(num_reqs=2, num_tokens=8, query_start_loc=torch.tensor([0, 3, 8]), positions=positions+10)
    rejected = torch.tensor([1, 0])
    expected, count = runtime._capture_payload(captured, batch, rejected, (0, 1))
    runtime.options = NS(capture_fused_pack=True)
    runtime._capture_pack_limits = PackLimits(1, 1, 2, 8, 2)
    actual, actual_count = runtime._capture_payload(captured, batch, rejected, (0, 1))
    assert torch.equal(actual, expected) and actual_count == count == 8
    assert runtime.capture_pack_stats == {'reference_batches': 2, 'fallback:scale_layout': 1}
