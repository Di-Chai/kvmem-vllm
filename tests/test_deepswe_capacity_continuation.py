"""A continuation must preserve completed evidence and wait for its dependency."""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('deepswe_capacity_continuation',
    Path(__file__).parents[1]/'scripts/benchmark_deepswe_capacity.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


@pytest.mark.parametrize('condition',('ends','reused_pid','changed_command'))
def test_dependency_wait_obeys_process_identity_without_signalling(tmp_path,monkeypatch,condition):
    class Missing(Exception):pass
    live=[True];sleeps=[]
    class Process:
        def __init__(self,pid):
            if not live[0]:raise Missing()
        def create_time(self):return 2 if condition=='reused_pid' else 1
        def status(self):return 'sleeping'
        def cmdline(self):return ['different'] if condition=='changed_command' else ['owned-existing-controller']
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=Process,NoSuchProcess=Missing,STATUS_ZOMBIE='zombie'))
    async def sleep(seconds):sleeps.append(seconds);live[0]=False
    monkeypatch.setattr(runner.asyncio,'sleep',sleep)
    cap=SimpleNamespace(save=lambda p,value:p.write_text(json.dumps(value)))
    owner=dict(pid=123,create_time=1,argv=['owned-existing-controller'])
    if condition=='changed_command':
        with pytest.raises(ValueError,match='command changed'):
            asyncio.run(runner.wait_dependency(tmp_path,owner,cap))
        assert not (tmp_path/'dependency-ended.json').exists()
    else:
        asyncio.run(runner.wait_dependency(tmp_path,owner,cap))
        assert (tmp_path/'dependency-ended.json').exists()
        assert sleeps==([10] if condition=='ends' else [])


@pytest.mark.parametrize('condition',('incomplete','different_profile'))
def test_carry_rejects_incomplete_or_incompatible_measurement_before_reuse(tmp_path,condition):
    path=tmp_path/'native-c4';path.mkdir()
    def save(p,value):p.write_text(json.dumps(value))
    controls={'environment':{'MAX_JOBS':'4'}}
    save(tmp_path/'manifest.json',dict(build_controls=controls,duration_seconds=1800))
    save(path/'run.json',dict(workers=4,status='running' if condition=='incomplete' else 'window_complete',
        cleanup_finished_at=1,cleanup_errors=[]))
    save(path/'reviewed-window.json',dict(result=dict(whole_window=dict(seconds=1800),errors=[])))
    save(path/'profile.json',dict(model='incompatible'))
    cap=SimpleNamespace(read=lambda p:json.loads(p.read_text()))
    with pytest.raises(ValueError,match='complete|Carried model'):
        runner.carry_native_window(path,cap,dict(model='expected'),controls,1800,tmp_path/'reference')
