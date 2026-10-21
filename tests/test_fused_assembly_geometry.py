"""CPU-only qualification of opaque layer tables; no CUDA context or allocation."""
from copy import copy
from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem import split_assembly


class TensorMetadata:
    """Tensor geometry and byte addresses only; GPU values are never readable."""
    def __init__(self, shape, dtype, pointer, *, strides=None, device='cuda:0'):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.pointer = pointer
        if strides is None:
            strides = tuple(prod(self.shape[i+1:]) for i in range(self.ndim))
        self.strides = tuple(strides)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self.dtype.itemsize

    def is_contiguous(self):
        return self.strides == tuple(prod(self.shape[i+1:]) for i in range(self.ndim))

    def item(self):
        pytest.fail('Fused layer eligibility must not read device values')

    def cpu(self):
        pytest.fail('Fused layer eligibility must not download device values')


def layers(count=3, *, table_dtype=torch.int32, trig_dtype=torch.float32):
    result = []
    for i in range(count):
        base = 0x10000000 + i * 0x1000000
        result.append(SimpleNamespace(
            key=TensorMetadata((64, 16, 4, 256), torch.float8_e4m3fn, base),
            value=TensorMetadata((64, 16, 4, 256), torch.float8_e4m3fn, base + 0x200000),
            kernel_page_table=TensorMetadata((32,), table_dtype, 0x50000000 + i * 0x1000),
            cos_sin=TensorMetadata((1024, 64), trig_dtype, 0x60000000 + i * 0x100000),
            key_scale=TensorMetadata((), torch.float32, 0x70000000 + i * 0x1000),
        ))
    return tuple(result)


@pytest.mark.parametrize('table_dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('trig_dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_uniform_layer_table_has_exact_ptr_stride_order_and_strong_owners(table_dtype, trig_dtype):
    native = layers(17, table_dtype=table_dtype, trig_dtype=trig_dtype)
    spec, reason = split_assembly.prepare_fused_layer_rows(native)
    assert reason is None and spec is not None
    geometry = spec.geometry
    assert (geometry.heads, geometry.dim, geometry.half, geometry.page,
            geometry.table_dtype, geometry.trig_dtype) == (4, 256, 32, 16, table_dtype, trig_dtype)
    assert len(spec.rows) == 17 and all(len(row) == 12 for row in spec.rows)
    for layer, row in zip(native, spec.rows):
        pointers = (layer.kernel_page_table, layer.cos_sin, layer.key, layer.value, layer.key_scale)
        assert row == tuple(value.data_ptr() for value in pointers) + layer.key.stride()[:3] + layer.value.stride()[:3] + (layer.cos_sin.stride(0),)
        assert all(any(owner is value for owner in spec.owners) for value in pointers)
    repeated, _ = split_assembly.prepare_fused_layer_rows(native)
    assert repeated.signature == spec.signature and repeated.rows == spec.rows


def test_heterogeneous_positive_outer_strides_and_capacities_are_represented_per_layer():
    native = layers()
    native[1].key = TensorMetadata((48, 16, 4, 256), torch.float8_e4m3fn, 0x11000000,
                                    strides=(40000, 2000, 400, 1))
    native[1].value = TensorMetadata((48, 16, 4, 256), torch.float8_e4m3fn, 0x11400000,
                                      strides=(50000, 2500, 500, 1))
    native[1].kernel_page_table = TensorMetadata((20,), torch.int32, 0x50001000)
    native[1].cos_sin = TensorMetadata((700, 64), torch.float32, 0x60100000, strides=(96, 1))
    spec, reason = split_assembly.prepare_fused_layer_rows(native)
    assert reason is None
    assert spec.rows[1][5:] == (40000, 2000, 400, 50000, 2500, 500, 96)


def test_normal_same_layer_interleaved_key_value_views_do_not_false_alias():
    native = layers()
    for i, layer in enumerate(native):
        base = 0x10000000 + i * 0x1000000
        layer.key = TensorMetadata((64, 16, 4, 256), torch.float8_e4m3fn, base,
                                   strides=(32768, 2048, 512, 1))
        layer.value = TensorMetadata((64, 16, 4, 256), torch.float8_e4m3fn, base + 256,
                                     strides=(32768, 2048, 512, 1))
    spec, reason = split_assembly.prepare_fused_layer_rows(native)
    assert spec is not None and reason is None


@pytest.mark.parametrize('change', ['key_pointer', 'value_stride', 'capacity', 'page_pointer',
                                   'trig_pointer', 'scale_pointer'])
def test_signature_changes_for_each_bound_source_layout_or_address(change):
    native = layers()
    original, _ = split_assembly.prepare_fused_layer_rows(native)
    changed = list(native)
    changed[1] = copy(native[1])
    field = {'key_pointer': 'key', 'value_stride': 'value', 'capacity': 'key',
             'page_pointer': 'kernel_page_table', 'trig_pointer': 'cos_sin',
             'scale_pointer': 'key_scale'}[change]
    tensor = copy(getattr(changed[1], field))
    setattr(changed[1], field, tensor)
    if change == 'value_stride':
        tensor.strides = (20000, 1200, 256, 1)
    elif change == 'capacity':
        tensor.shape = (32, 16, 4, 256)
        changed[1].value = copy(changed[1].value)
        changed[1].value.shape = tensor.shape
    else:
        tensor.pointer += 0x400
    updated, reason = split_assembly.prepare_fused_layer_rows(tuple(changed))
    assert reason is None and updated.signature != original.signature


@pytest.mark.parametrize('fault', ['cpu', 'other_device', 'key_dtype', 'value_dtype',
    'scale_dtype', 'scale_count', 'page_dtype', 'mixed_page_dtype', 'page_stride', 'page_rank',
    'trig_dtype', 'mixed_trig_dtype', 'trig_inner_stride', 'trig_zero_stride', 'trig_half',
    'trig_odd_width', 'half_exceeds_dim', 'heads', 'dim', 'page_size', 'rank',
    'key_value_shape', 'inner_stride', 'zero_stride', 'negative_stride', 'overlapping_view'])
def test_unsupported_metadata_declines_the_entire_layer_bundle(fault):
    native = layers()
    layer = native[-1]
    if fault == 'cpu': layer.key.device = torch.device('cpu')
    elif fault == 'other_device': layer.key.device = torch.device('cuda:1')
    elif fault == 'key_dtype': layer.key.dtype = torch.bfloat16
    elif fault == 'value_dtype': layer.value.dtype = torch.uint8
    elif fault == 'scale_dtype': layer.key_scale.dtype = torch.float64
    elif fault == 'scale_count': layer.key_scale.shape, layer.key_scale.ndim = (2,), 1
    elif fault == 'page_dtype': layer.kernel_page_table.dtype = torch.float32
    elif fault == 'mixed_page_dtype': layer.kernel_page_table.dtype = torch.int64
    elif fault == 'page_stride': layer.kernel_page_table.strides = (2,)
    elif fault == 'page_rank': layer.kernel_page_table.shape, layer.kernel_page_table.ndim = (4, 8), 2
    elif fault == 'trig_dtype': layer.cos_sin.dtype = torch.float64
    elif fault == 'mixed_trig_dtype': layer.cos_sin.dtype = torch.bfloat16
    elif fault == 'trig_inner_stride': layer.cos_sin.strides = (128, 2)
    elif fault == 'trig_zero_stride': layer.cos_sin.strides = (0, 1)
    elif fault == 'trig_half': layer.cos_sin.shape = (1024, 128)
    elif fault == 'trig_odd_width': layer.cos_sin.shape = (1024, 63)
    elif fault == 'half_exceeds_dim': layer.cos_sin.shape, layer.cos_sin.strides = (1024, 1024), (1024, 1)
    elif fault == 'heads': layer.key.shape = layer.value.shape = (64, 16, 3, 256)
    elif fault == 'dim': layer.key.shape = layer.value.shape = (64, 16, 4, 128)
    elif fault == 'page_size': layer.key.shape = layer.value.shape = (64, 8, 4, 256)
    elif fault == 'rank': layer.key.ndim = 3
    elif fault == 'key_value_shape': layer.value.shape = (64, 16, 4, 128)
    elif fault == 'inner_stride': layer.key.strides = (32768, 2048, 512, 2)
    elif fault == 'zero_stride': layer.key.strides = (0, 1024, 256, 1)
    elif fault == 'negative_stride': layer.key.strides = (-16384, 1024, 256, 1)
    else: layer.key.strides = (16384, 1024, 128, 1)
    spec, reason = split_assembly.prepare_fused_layer_rows(native)
    assert spec is None and isinstance(reason, str) and reason


@pytest.mark.parametrize('alias', ['same_layer', 'cross_key', 'cross_value', 'partial_cross',
                                  'page', 'trig', 'scale'])
def test_cross_layer_writes_or_metadata_reads_cannot_alias_destinations(alias):
    native = layers()
    if alias == 'same_layer': native[0].value.pointer = native[0].key.pointer
    elif alias == 'cross_key': native[1].key.pointer = native[0].key.pointer
    elif alias == 'cross_value': native[1].value.pointer = native[0].key.pointer
    elif alias == 'partial_cross': native[1].key.pointer = native[0].value.pointer + 0x100
    else:
        name = {'page': 'kernel_page_table', 'trig': 'cos_sin', 'scale': 'key_scale'}[alias]
        getattr(native[1], name).pointer = native[0].key.pointer + 0x100
    spec, reason = split_assembly.prepare_fused_layer_rows(native)
    assert spec is None and reason == 'alias'


def test_empty_bundle_declines_without_allocation():
    spec, reason = split_assembly.prepare_fused_layer_rows(())
    assert spec is None and reason == 'empty'
