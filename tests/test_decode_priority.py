"""Opt-in KVMEM_LAYERWISE_DECODE_PRIORITY: lane admission order only."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import decode_priority as dp
from test_maintenance import policy, job as maintenance_job


def job(stage='assembly_wait', pressure_stage='prefill', selection='pressure', done=False, cancelled=False):
    return NS(stage=stage, done=done, cancelled=cancelled,
              command=NS(selection=selection, pressure_stage=pressure_stage, request_id=f'{pressure_stage}-{stage}'))


def test_flag_defaults_off_and_is_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_DECODE_PRIORITY', raising=False)
    assert dp.enabled() is False
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', 'yes')
    with pytest.raises(ValueError):
        dp.enabled()


def test_only_non_decode_jobs_yield_to_a_waiting_decode_pressure(monkeypatch):
    decode, prefill, query = job(pressure_stage='decode'), job(), job(selection='query', pressure_stage=None)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', '0')
    assert not dp.should_yield([decode, prefill], prefill)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', '1')
    assert dp.should_yield([decode, prefill], prefill) and dp.should_yield([decode, query], query)
    assert not dp.should_yield([decode, prefill], decode)            # decode never yields
    other = job(pressure_stage='decode')
    assert not dp.should_yield([decode, other], other)               # decode jobs compete normally
    for idle in (job(pressure_stage='decode', stage='assembly'), job(pressure_stage='decode', done=True),
                 job(pressure_stage='decode', cancelled=True), job(pressure_stage='decode', stage='prepare')):
        assert not dp.should_yield([idle, prefill], prefill)       # only a decode job waiting for the lane


def test_maintenance_order_tries_a_waiting_decode_pressure_first(monkeypatch):
    p = policy()
    reap = maintenance_job('reap', stage='assembly')
    big = maintenance_job('big', 32, stage='assembly_wait'); big.command.selection = 'pressure'
    big.command.pressure_stage = 'prefill'
    decode = maintenance_job('decode', 8, stage='assembly_wait'); decode.command.selection = 'pressure'
    decode.command.pressure_stage = 'decode'
    old = maintenance_job('old', 16, 1.)
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', '0')
    assert p.order([big, old, decode, reap])[:2] == [reap, old]
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', '1')
    assert p.order([big, old, decode, reap])[:3] == [reap, decode, old]


def test_refresh_admission_yields_before_touching_lanes_or_credits(monkeypatch):
    from vllm_kvmem.async_refresh import AsyncRefresh
    monkeypatch.setenv('KVMEM_LAYERWISE_DECODE_PRIORITY', '1')
    refresh = AsyncRefresh.__new__(AsyncRefresh)
    decode, prefill = job(pressure_stage='decode'), job()
    prefill.guided_pressure = False
    refresh.jobs = {'d': decode, 'p': prefill}
    touched = []
    refresh.maintenance = NS(acquire=lambda *a, **k: touched.append('acquire') or True)
    assert refresh.step(prefill) is False and prefill.stage == 'assembly_wait' and touched == []
