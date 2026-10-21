"""CPU metadata/source proof for the opt-in component-only readonly overlay."""
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import native_attention_build as build
from vllm_kvmem.native_prefill_readonly import ReadonlyMetadataIneligible, qualify_readonly_arguments


class Device:
    type = 'cuda'
    def __init__(self, index=0): self.index = index
    def __str__(self): return 'cuda:' + str(self.index)


class Tensor:
    def __init__(self, pointer, shape=(128,), strides=None, *, dtype='torch.int32', size=4, device=None):
        self.pointer, self.shape, self.dtype, self.size = pointer, shape, dtype, size
        self.strides = strides or (1,) * len(shape)
        self.device = device or Device()
    def stride(self): return self.strides
    def element_size(self): return self.size
    def data_ptr(self): return self.pointer
    def cpu(self): raise AssertionError('No content download')
    def item(self): raise AssertionError('No device scalar read')


def arguments():
    values = [None] * 24
    for index in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19):
        values[index] = Tensor((index + 1) * 100000)
    values[2] = [0] * 15
    values[14] = -1
    values[20:24] = [1000, 200, 0, .125]
    return values


def test_only_static_class_and_two_loads_change():
    source = Path(build.__file__).with_name('native_prefill_stats.cuh').read_text()
    candidate = build.patch_static_readonly(source)
    prefix = source[:source.index('struct KVMemStaticAlignedAttention :')]
    assert candidate.startswith(prefix)
    assert candidate.count('__ldg(') == 2
    restored = candidate.replace('KVMemStaticReadonlyAttention', 'KVMemStaticAlignedAttention')
    restored = restored.replace('int sample = __ldg(stats.rows + __ldg(p.q_indptr + batch) + q);',
                                'int sample = stats.rows[p.q_indptr[batch] + q];')
    assert restored == source
    assert '__restrict__' not in candidate


def test_patch_fails_closed_if_static_structure_changes():
    with pytest.raises(build.UnsupportedNativeStats):
        build.patch_static_readonly('missing class')
    source = Path(build.__file__).with_name('native_prefill_stats.cuh').read_text()
    with pytest.raises(build.UnsupportedNativeStats):
        build.patch_static_readonly(source.replace('int sample = stats.rows[p.q_indptr[batch] + q];', 'changed'))


@pytest.mark.parametrize('flags', [dict(prefill_static_readonly=True),
    dict(prefill_static_readonly=True, prefill_static_aligned=True, prefill_cta_gate=True)])
def test_readonly_requires_exact_static_component_before_flashinfer_import(flags):
    with pytest.raises(ValueError, match='requires only'):
        build.source_overlay(None, **flags)


def test_disjoint_metadata_and_ordinary_interleaved_kv_are_allowed():
    args = arguments()
    args[5] = Tensor(args[4].data_ptr() + 256, (4, 16, 4, 256), (32768, 2048, 512, 1), size=1)
    proof = qualify_readonly_arguments(args)
    assert len(proof['readonly_regions']) == 2
    assert len(proof['writable_regions']) == 5


@pytest.mark.parametrize('read', [6, 16])
@pytest.mark.parametrize('write', [0, 1, 10, 11, 19])
def test_every_possible_write_alias_is_rejected(read, write):
    args = arguments(); args[write] = Tensor(args[read].pointer + 4)
    with pytest.raises(ReadonlyMetadataIneligible, match='overlaps writable'):
        qualify_readonly_arguments(args)


def test_read_read_alias_is_allowed_and_optional_native_lse_can_be_absent():
    args = arguments(); args[6] = Tensor(args[16].pointer + 4); args[11] = None
    assert qualify_readonly_arguments(args)


@pytest.mark.parametrize('kind', ['empty', 'strided', 'negative', 'dtype', 'cpu', 'other-device', 'graph', 'offset', 'window', 'abi'])
def test_unproved_geometry_never_launches_readonly(kind):
    args = arguments()
    if kind == 'empty': args[6].shape = (0,)
    elif kind == 'strided': args[6].strides = (2,)
    elif kind == 'negative': args[16].strides = (-1,)
    elif kind == 'dtype': args[16].dtype = 'torch.float32'
    elif kind == 'cpu': args[6].device = NS(type='cpu')
    elif kind == 'other-device': args[19].device = Device(1)
    elif kind == 'graph': args[2][13] = 1
    elif kind == 'offset': args[22] = 1
    elif kind == 'window': args[14] = 0
    else: args.pop()
    with pytest.raises(ReadonlyMetadataIneligible):
        qualify_readonly_arguments(args)


def test_noncontiguous_writable_enclosure_rejects_uncertain_alias():
    args = arguments()
    args[10] = Tensor(args[6].pointer - 16, (4, 4), (128, 2), size=2)
    with pytest.raises(ReadonlyMetadataIneligible, match='overlaps writable'):
        qualify_readonly_arguments(args)


def test_i64_q_indptr_supported_but_rows_remain_i32():
    args = arguments(); args[6].dtype = 'torch.int64'; args[6].size = 8
    assert qualify_readonly_arguments(args)
    args[16].dtype = 'torch.int64'
    with pytest.raises(ReadonlyMetadataIneligible, match='integer vector'):
        qualify_readonly_arguments(args)
