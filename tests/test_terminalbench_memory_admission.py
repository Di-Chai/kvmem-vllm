import json
from types import SimpleNamespace

import pytest

from scripts.run_terminalbench_job import MemoryAdmission, GIB


def gate(tmp_path, monkeypatch, budget=48):
    monkeypatch.setattr('scripts.run_terminalbench_job.psutil.virtual_memory',
                        lambda:SimpleNamespace(total=128*GIB))
    events=[]
    value=MemoryAdmission(tmp_path,dict(container_memory_admission_bytes=budget*GIB),events.append)
    return value,events


def control(tmp_path, budget):
    (tmp_path/'memory-admission-control.json').write_text(json.dumps(dict(schema_version=1,budget_bytes=budget)))


def test_more_tasks_fit_without_changing_the_host_reserve(tmp_path,monkeypatch):
    value,_=gate(tmp_path,monkeypatch)
    assert value.admits(24*GIB,8*GIB,48*GIB)
    assert value.admits(40*GIB,8*GIB,20*GIB)
    assert not value.admits(40*GIB,8*GIB,20*GIB-1)
    assert not value.admits(48*GIB,2*GIB,48*GIB)


def test_online_budget_changes_apply_to_waiters_without_cancelling_running_tasks(tmp_path,monkeypatch):
    value,events=gate(tmp_path,monkeypatch,24)
    assert not value.admits(24*GIB,8*GIB,48*GIB)
    control(tmp_path,48*GIB)
    assert value.admits(24*GIB,8*GIB,48*GIB)
    assert events[-1]['previous_budget_bytes']==24*GIB and events[-1]['budget_bytes']==48*GIB
    control(tmp_path,16*GIB)
    assert not value.admits(32*GIB,2*GIB,48*GIB)
    # Admission has no cancellation/release side effect for existing owners.
    assert value.budget==16*GIB
    (tmp_path/'memory-admission-control.json').unlink()
    assert value.admits(16*GIB,8*GIB,48*GIB) and value.budget==24*GIB


@pytest.mark.parametrize('bad', [True,0,-1,48.5,'48',117*GIB])
def test_invalid_online_budget_keeps_last_valid_policy(tmp_path,monkeypatch,bad):
    value,events=gate(tmp_path,monkeypatch)
    control(tmp_path,bad)
    assert value.admits(24*GIB,8*GIB,48*GIB) and value.budget==48*GIB
    assert events[-1]['event']=='memory_admission_control_rejected'
    count=len(events)
    value.admits(24*GIB,8*GIB,48*GIB)
    assert len(events)==count


@pytest.mark.parametrize('raw', ['{', 'x'*4097,'[]','{"schema_version":true,"budget_bytes":1}',
    '{"schema_version":1,"budget_bytes":1,"extra":1}'])
def test_partial_or_malformed_control_does_not_abort_the_job(tmp_path,monkeypatch,raw):
    value,_=gate(tmp_path,monkeypatch)
    (tmp_path/'memory-admission-control.json').write_text(raw)
    assert value.admits(24*GIB,8*GIB,48*GIB) and value.budget==48*GIB
    control(tmp_path,32*GIB)
    assert value.admits(24*GIB,8*GIB,48*GIB) and value.budget==32*GIB


def test_default_and_invalid_initial_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.run_terminalbench_job.psutil.virtual_memory',
                        lambda:SimpleNamespace(total=128*GIB))
    assert MemoryAdmission(tmp_path,{},lambda _:None).budget==48*GIB
    for value in [True,0,117*GIB]:
        with pytest.raises(ValueError):
            MemoryAdmission(tmp_path,dict(container_memory_admission_bytes=value),lambda _:None)
