"""SIMD changes independent FP8 channels, never token reduction order."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from vllm_kvmem import _native


def compare(raw, count, scored_layers=None):
    shape = (scored_layers or raw.shape[0], raw.shape[2], raw.shape[3])
    reference = np.full(shape, -123., np.float32)
    actual = reference.copy()
    assert _native.finalize_mean_batch([(raw, reference, count, True)]) is None
    used = _native.finalize_mean_batch([(raw, actual, count, True)], vectorized=True)
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    # The downstream index stores FP16 after division, not the raw sum alone.
    a = (torch.from_numpy(actual)/count).half().view(torch.int16)
    b = (torch.from_numpy(reference)/count).half().view(torch.int16)
    assert torch.equal(a, b)
    return used


@pytest.mark.parametrize('channels', [1,7,8,9,17,64,1024])
@pytest.mark.parametrize('count', [1,7,32,257])
def test_fp8_bitwise_with_padded_strides_offset_and_channel_tail(channels, count):
    rng = np.random.default_rng(871)
    # Extra physical layer and token rows must not affect the scored prefix.
    storage = rng.integers(0, 256, (4,count+5,1,channels), dtype=np.uint8)
    storage[storage == 127] = 126
    storage[storage == 255] = 254
    raw = storage[:, 2:count+3]
    used = compare(raw, count, 3)
    assert used == int(_native.finalize_mean_simd_supported() and channels >= 8)


@pytest.mark.parametrize('count', [1,2,32])
def test_all_fp8_codes_signed_nan_and_zero_bitwise(count):
    # The first row includes every exact FP8 bit pattern. Later rows exercise
    # NaN signs in both operands and cancellation to signed zero.
    codes = np.arange(256, dtype=np.uint8)
    raw = np.stack([np.roll(codes, token*37) for token in range(count)]).reshape(1,count,1,256)
    assert compare(raw,count) == int(_native.finalize_mean_simd_supported())
    zero = np.zeros((1,count,1,17),dtype=np.uint8)
    zero[:,::2] = 128
    assert compare(zero,count) == int(_native.finalize_mean_simd_supported())


@pytest.mark.parametrize('channels', [8,9,17,32,33,64])
@pytest.mark.parametrize('codes', [(127,255),(255,127),(0,127,255,0),(128,255,127,128)])
def test_opposite_signed_nans_in_same_channel_preserve_payload_bits(channels,codes):
    raw = np.broadcast_to(np.asarray(codes,dtype=np.uint8)[None,:,None,None],
                          (2,len(codes),1,channels)).copy()
    assert compare(raw,len(codes)) == int(_native.finalize_mean_simd_supported())


def test_mixed_fp8_and_bf16_records_report_actual_vectorized_count():
    fp8 = np.arange(3*32*34,dtype=np.uint8).reshape(3,32,2,17)
    bf16 = torch.arange(3*32*34).reshape(3,32,2,17).bfloat16().view(torch.uint8).numpy()
    expected = [np.empty((2,2,17),dtype=np.float32) for _ in range(2)]
    actual = [np.empty_like(s) for s in expected]
    inputs = [(fp8,True),(bf16,False)]
    assert _native.finalize_mean_batch([(r,s,32,k) for (r,k),s in zip(inputs,expected)]) is None
    used = _native.finalize_mean_batch([(r,s,32,k) for (r,k),s in zip(inputs,actual)], vectorized=True)
    assert used == int(_native.finalize_mean_simd_supported())
    for a,b in zip(actual,expected):
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
    assert _native.finalize_mean_batch([(bf16,actual[1],32,False)],vectorized=True) == 0
    assert _native.finalize_mean_batch([],vectorized=True) == 0
    assert _native.finalize_mean_batch([]) is None


def test_overlapping_raw_and_output_keep_original_scalar_write_order():
    snapshots = []
    for vectorized in (False,True):
        storage = np.arange(256,dtype=np.uint8)
        raw = storage[:128].reshape(1,8,1,16)
        out = np.ndarray((1,1,16),dtype=np.float32,buffer=storage)
        used = _native.finalize_mean_batch([(raw,out,8,True)],vectorized=vectorized)
        assert used == (0 if vectorized else None)
        snapshots.append(storage.copy())
    np.testing.assert_array_equal(*snapshots)


@pytest.mark.parametrize('fault', ['count','shape','readonly','stride','arity'])
def test_all_jobs_validated_before_any_simd_write(fault):
    raw = np.ones((2,32,1,64),dtype=np.uint8)
    earlier = np.full((1,1,64),123.,np.float32)
    later = earlier.copy()
    bad = (raw,later,32,True)
    if fault == 'count': bad = (raw,later,33,True)
    elif fault == 'shape': bad = (raw,later[...,:63],32,True)
    elif fault == 'readonly': later.flags.writeable = False
    elif fault == 'stride': bad = (raw[...,::-1],later,32,True)
    else: bad = (raw,later,32)
    with pytest.raises((ValueError,BufferError)):
        _native.finalize_mean_batch([(raw,earlier,32,True),bad],vectorized=True)
    assert np.all(earlier == 123.) and np.all(later == 123.)


def test_independent_owner_calls_have_no_shared_accumulator():
    def run(seed):
        raw = np.random.default_rng(seed).integers(0,127,(3,32,4,256),dtype=np.uint8)
        return compare(raw,32,2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run,range(8))) == [int(_native.finalize_mean_simd_supported())]*8
