"""CPU contracts for the diagnostic-only GPU sum adapter."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from vllm_kvmem import _native


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'mean_index_sidecar.py'
spec = importlib.util.spec_from_file_location('mean_index_sidecar', SCRIPT)
sidecar = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sidecar
spec.loader.exec_module(sidecar)


@pytest.mark.parametrize('tokens,pages', [(32,1),(1984,62),(2048,64),(14336,448)])
def test_exact_capacity_and_target_only_bytes(tokens, pages):
    g = sidecar.Geometry()
    sizes = g.sizes(tokens)
    assert sizes['pages'] == pages
    assert sizes['sum_bytes'] == pages*16*4*256*4
    assert sizes['d2h_bytes'] == sizes['sum_bytes']+pages*16*4
    assert g.sizes(16416)['pages'] == 513


@pytest.mark.parametrize('tokens', [0,1,31,33,1985,16424,16448])
def test_partial_or_oversize_pages_are_not_silently_used(tokens):
    with pytest.raises(ValueError):
        sidecar.Geometry().sizes(tokens)


@pytest.mark.parametrize('change', [dict(scoring_layers=17),dict(capacity_tokens=31),
    dict(capacity_tokens=20000),dict(layers=33),dict(heads=32),dict(dim=True)])
def test_unsupported_geometry_is_explicit(change):
    with pytest.raises(ValueError):
        sidecar.Geometry(**change)


def test_no_dtype_or_partial_history_shortcut():
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=16)
    raw = torch.zeros(3,32,1,16).to(torch.float8_e4m3fn)
    assert sidecar.qualify(raw,g)['pages'] == 1
    for data,begin in [(raw.bfloat16(),0),(raw,1),(raw.transpose(0,1),0)]:
        with pytest.raises(ValueError):
            sidecar.qualify(data,g,history_begin=begin)


@pytest.mark.parametrize('archive,target', [
    (('a','b','mtp'),('b','a')), (('a','b','mtp'),('a','mtp')),
    (('a','a','mtp'),('a','a')), (('a','mtp','b'),('a','b'))])
def test_target_mapping_cannot_be_inferred_from_layer_count(archive,target):
    g = sidecar.Geometry(layers=3,scoring_layers=2,heads=1,dim=16)
    with pytest.raises(ValueError,match='order'):
        sidecar.qualify_target_prefix(archive,target,g)


def data(pages=2):
    raw = torch.arange(3*pages*32*16).remainder(127).byte().reshape(3,pages*32,1,16)
    raw = raw.view(torch.float8_e4m3fn)
    sums = torch.empty(pages,2,1,16)
    _native.finalize_mean_batch(sidecar.cpu_jobs(raw,sums))
    return raw,sums


def test_adapter_preserves_exact_sum_division_and_logical_cursor():
    raw,expected = data()
    actual = torch.full_like(expected,999.)
    consume = sidecar.SumConsumer(expected,torch.zeros(2,2,dtype=torch.int32),_native.finalize_mean_batch)
    jobs = sidecar.cpu_jobs(raw,actual)
    with consume.installed():
        assert _native.finalize_mean_batch(jobs[:1]) is None
        assert _native.finalize_mean_batch(jobs[1:]) is None
    assert consume.consumed == 2 and consume.fallback_records == 0
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
    assert torch.equal((actual/32).half().view(torch.int16),(expected/32).half().view(torch.int16))


@pytest.mark.parametrize('codes', [(127,255),(255,127),(0,128)])
def test_nonfinite_batch_uses_old_routine_including_nan_signs(codes):
    raw,_ = data()
    for i,value in enumerate(codes):
        raw.view(torch.uint8)[0,i,0] = value
    expected = torch.empty(2,2,1,16)
    _native.finalize_mean_batch(sidecar.cpu_jobs(raw,expected))
    actual = torch.zeros_like(expected)
    consumer = sidecar.SumConsumer(torch.full_like(expected,999.),
        torch.ones(2,2,dtype=torch.int32),_native.finalize_mean_batch)
    consumer(sidecar.cpu_jobs(raw,actual))
    assert consumer.fallback_records == 2
    assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))


@pytest.mark.parametrize('fault',['count','shape','readonly','fp8','short','overrun'])
def test_adapter_prevalidates_every_job_before_any_write(fault):
    raw,sums = data()
    actual = torch.full_like(sums,123.)
    consumer = sidecar.SumConsumer(sums,torch.zeros(2,2,dtype=torch.int32),_native.finalize_mean_batch)
    jobs = sidecar.cpu_jobs(raw,actual)
    r,d,c,f = jobs[-1]
    if fault == 'count': c = 31
    elif fault == 'shape': d = d[:1]
    elif fault == 'readonly': d.flags.writeable = False
    elif fault == 'fp8': f = False
    elif fault == 'short': r = r[:, :31]
    elif fault == 'overrun': jobs.append(jobs[-1])
    jobs[1] = r,d,c,f
    with pytest.raises(ValueError): consumer(jobs)
    assert torch.all(actual == 123.) and consumer.consumed == 0


def test_adapter_restores_original_after_consumer_exception_or_unused_pages():
    raw,sums = data()
    original = _native.finalize_mean_batch
    consumer = sidecar.SumConsumer(sums,torch.zeros(2,2,dtype=torch.int32),original)
    with pytest.raises(RuntimeError,match='Not all'):
        with consumer.installed(): pass
    assert _native.finalize_mean_batch is original
    with pytest.raises(LookupError):
        with consumer.installed(): raise LookupError('injected')
    assert _native.finalize_mean_batch is original


def test_real_append_keeps_same_means_and_archive_without_changing_production():
    from vllm_kvmem.authority import RequestAuthority,append_authority_batch
    from vllm_kvmem.config import KVMemConfig
    from vllm_kvmem.memory_storage import MemoryArchive
    raw,sums = data()
    values = raw.view(torch.uint8).clone()
    results = []
    for candidate in (False,True):
        archive = MemoryArchive('/tmp/mean-sidecar-cpu-only',3*32*16*2,16,
                                cpu_direct_record_seal=True)
        archive.async_seal = True
        cfg = KVMemConfig(block_tokens=32,select_budget=64,gen_budget=64,repository_blocks=16,
                          sink_blocks=0,recent_blocks=0)
        authority = RequestAuthority('owner',cfg,archive,3,1,16,raw_dtype=raw.dtype,
                                     scoring_layers=2,mean_k_backend='qw3_cuda')
        consumer = sidecar.SumConsumer(sums,torch.zeros(2,2,dtype=torch.int32),_native.finalize_mean_batch)
        from contextlib import nullcontext
        with consumer.installed() if candidate else nullcontext():
            for begin in (0,32):
                assert append_authority_batch([(authority,raw[:,begin:begin+32],values[:,begin:begin+32],
                    dict(history_begin=begin,local_rope_begin=begin,accepted=32))]) == 1
        authority.publish_sealed(lambda: archive.wait_for_capacity(.01))
        assert archive.io_usage()['seal_direct_bytes'] == 3*64*16*2
        records = list(authority.records.values())
        results.append((authority.history_end,[r.digest for r in records],
                        torch.stack([r.mean_key for r in records]).view(torch.int16).clone()))
        assert authority.close() and archive.close()
    assert results[0][:2] == results[1][:2]
    assert torch.equal(results[0][2],results[1][2])
