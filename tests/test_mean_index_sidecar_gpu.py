"""Explicit CUDA qualification of diagnostic-only ordered archive FP8 sums."""
import os
from pathlib import Path
import sys
import weakref

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import mean_index_sidecar as sidecar
from benchmark_mean_index_sidecar import fixture,index_once,append_once
from vllm_kvmem import _native


pytestmark = pytest.mark.skipif(os.getenv('KVMEM_RUN_GPU_TESTS') != '1',
                               reason='Explicit experimental MeanK CUDA qualification')


def pool(g):
    names = tuple(f'target.{i}' for i in range(g.scoring_layers))
    archive = names+tuple(f'mtp.{i}' for i in range(g.layers-g.scoring_layers))
    return sidecar.OrderedMeanSidecar(g,archive_names=archive,target_names=names)


def compare(raw,worker):
    expected = torch.empty(raw.shape[1]//32,worker.geometry.scoring_layers,
                           worker.geometry.heads,worker.geometry.dim)
    cpu = raw.cpu()
    _native.finalize_mean_batch(sidecar.cpu_jobs(cpu,expected))
    ticket = worker.submit(raw)
    sums,invalid = ticket.resolve()
    actual = torch.empty_like(expected)
    consume = sidecar.SumConsumer(sums,invalid,_native.finalize_mean_batch)
    consume(sidecar.cpu_jobs(cpu,actual))
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert torch.equal((actual/32).half().view(torch.int16),(expected/32).half().view(torch.int16))
    failures = int(invalid.sum())
    fallback = consume.fallback_records
    ticket.release()
    return failures,fallback


@pytest.mark.parametrize('tokens',[32,1984,2048,14336])
def test_real_fused_capture_bytes_and_full_authority_lifecycle(tokens):
    env = fixture(tokens)
    timing,sums,means = index_once(env,'gpu')
    assert timing['fallback_records'] == 0
    assert torch.equal(sums.view(torch.int32),env.expected.view(torch.int32))
    assert torch.equal(means.view(torch.int16),env.mean.view(torch.int16))
    for mode in ('cpu','gpu'):
        _,usage = append_once(env,mode)
        assert usage['seal_records'] == tokens//32
        assert usage['seal_direct_bytes'] == tokens*17*4*256*2
    assert env.sidecar.pending is None


def test_all_finite_fp8_codes_cancellation_and_signed_zero():
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    worker = pool(g)
    codes = torch.arange(256,device='cuda',dtype=torch.int32)
    codes = torch.where(codes.remainder(128) == 127,codes-1,codes).byte()
    raw = torch.stack([codes.roll(i*37) for i in range(32)]).reshape(1,32,1,256).repeat(3,1,1,1)
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (0,0)
    raw[:,:] = 128
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (0,0)
    raw[:,1::2] = 0
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (0,0)
    raw[:,::2] = 126
    raw[:,1::2] = 254
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (0,0)


@pytest.mark.parametrize('codes',[(127,255),(255,127),(0,127,255,128)])
def test_nonfinite_codes_take_original_cpu_batch_preserving_nan_bits(codes):
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    raw = torch.zeros(3,32,1,256,device='cuda',dtype=torch.uint8)
    for index,code in enumerate(codes): raw[0,index] = code
    assert compare(raw.view(torch.float8_e4m3fn),pool(g)) == (1,1)


def test_mtp_nan_is_excluded_and_small_next_packet_has_no_stale_flag():
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    worker = pool(g)
    raw = torch.full((3,64,1,256),127,device='cuda',dtype=torch.uint8)
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (4,2)
    raw[:2] = 0
    assert compare(raw.view(torch.float8_e4m3fn),worker) == (0,0)
    small = raw[:,:32].contiguous()
    assert compare(small.view(torch.float8_e4m3fn),worker) == (0,0)


def test_single_fixed_slot_waits_for_gpu_and_cpu_consumer():
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    worker = pool(g)
    raw = torch.zeros(3,32,1,256,device='cuda').to(torch.float8_e4m3fn)
    pointers = (worker.sums.data_ptr(),worker.host_sums.data_ptr())
    ticket = worker.submit(raw)
    with pytest.raises(RuntimeError,match='busy'): worker.submit(raw)
    with pytest.raises(RuntimeError,match='Unresolved'): ticket.release()
    sums,flags = ticket.resolve()
    assert ticket.resolve()[0].data_ptr() == sums.data_ptr()
    assert not flags.any()
    ticket.release()
    with pytest.raises(RuntimeError): ticket.resolve()
    compare(raw,worker)
    assert pointers == (worker.sums.data_ptr(),worker.host_sums.data_ptr())


def test_changed_stream_and_partial_page_are_rejected_before_launch(monkeypatch):
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    worker = pool(g)
    raw = torch.zeros(3,32,1,256,device='cuda').to(torch.float8_e4m3fn)
    with pytest.raises(ValueError,match='partial'): worker.submit(raw,history_begin=1)
    with pytest.raises(ValueError,match='FP8'): worker.submit(raw.bfloat16())
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with pytest.raises(ValueError,match='producer'): worker.submit(raw)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda,'current_device',lambda: raw.device.index+1)
        with pytest.raises(ValueError,match='current'): worker.submit(raw)
    assert worker.pending is None


def test_wait_failure_quarantines_ticket_and_retains_exact_source():
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=256)
    worker = pool(g)
    original = worker.finished
    class FailedWait:
        def record(self): original.record()
        def synchronize(self): raise RuntimeError('injected completion failure')
    worker.finished = FailedWait()
    raw = torch.zeros(3,32,1,256,device='cuda').to(torch.float8_e4m3fn)
    source = weakref.ref(raw)
    ticket = worker.submit(raw)
    del raw
    with pytest.raises(RuntimeError,match='completion'): ticket.resolve()
    assert worker.poisoned and source() is not None and worker.pending is ticket
    with pytest.raises(RuntimeError,match='quarantined'): ticket.release()
    original.synchronize()  # test-only cleanup; production experiment stays poisoned
