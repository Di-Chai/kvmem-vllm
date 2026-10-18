import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


spec=importlib.util.spec_from_file_location('ready_agent_monitor',
    Path(__file__).parents[1]/'scripts/benchmark_ready_agents.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def test_monitor_failure_cancels_workload_and_preserves_original_exception():
    async def exercise():
        started=asyncio.Event();cancelled=asyncio.Event()
        error=RuntimeError('archive contract failed')
        async def workload():
            started.set()
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        async def monitor():
            await started.wait()
            raise error
        with pytest.raises(RuntimeError) as caught:
            await bench.supervise_monitored(workload(),monitor())
        assert caught.value is error and cancelled.is_set()
    asyncio.run(exercise())


def test_monitor_normal_exit_is_not_successful_monitoring():
    async def exercise():
        async def worker():await asyncio.Event().wait()
        async def observer():return None
        with pytest.raises(RuntimeError,match='stopped before workload'):
            await bench.supervise_monitored(worker(),observer())
    asyncio.run(exercise())


@pytest.mark.parametrize('fail_worker',[False,True])
def test_workload_completion_drains_monitor(fail_worker):
    async def exercise():
        observed=asyncio.Event();cancelled=asyncio.Event()
        error=ValueError('request failed')
        async def worker():
            await observed.wait()
            if fail_worker:raise error
            return 42
        async def observer():
            observed.set()
            try:await asyncio.Event().wait()
            finally:cancelled.set()
        if fail_worker:
            with pytest.raises(ValueError) as caught:
                await bench.supervise_monitored(worker(),observer())
            assert caught.value is error
        else:
            assert await bench.supervise_monitored(worker(),observer())==42
        assert cancelled.is_set()
    asyncio.run(exercise())


def test_parent_cancellation_drains_worker_and_monitor():
    async def exercise():
        started=[asyncio.Event(),asyncio.Event()];closed=[]
        async def blocked(i):
            started[i].set()
            try:await asyncio.Event().wait()
            finally:closed.append(i)
        task=asyncio.create_task(bench.supervise_monitored(blocked(0),blocked(1)))
        await asyncio.gather(*(event.wait() for event in started))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert sorted(closed)==[0,1]
    asyncio.run(exercise())


@pytest.mark.parametrize('failure',['foreign_gpu','memory_floor','telemetry'])
def test_run_arm_stops_owned_server_on_guard_or_telemetry_failure(tmp_path,monkeypatch,failure):
    """Exercise the real arm cleanup path without processes, network, or CUDA."""
    async def immediate(function,*args,**kwargs):return function(*args,**kwargs)
    monkeypatch.setattr(bench.asyncio,'to_thread',immediate)
    monkeypatch.setattr(bench,'verify',lambda root:None)
    async def idle(*args):pass
    monkeypatch.setattr(bench.bench.q,'wait_idle',idle)
    gpu_calls=0
    def gpu_state():
        nonlocal gpu_calls
        gpu_calls+=1
        return dict(pids=[999] if failure=='foreign_gpu' and gpu_calls>1 else [],
                    memory_mib=0,utilization=0)
    monkeypatch.setattr(bench.bench.q,'gpu_state',gpu_state)
    memory_calls=0
    def memory():
        nonlocal memory_calls
        memory_calls+=1
        return dict(MemAvailable=0 if failure=='memory_floor' and memory_calls>1 else 100)
    monkeypatch.setattr(bench.cap,'host_memory',memory)
    server=SimpleNamespace(pid=123,poll=lambda:None)
    stopped=[]
    monkeypatch.setattr(bench.subprocess,'Popen',lambda *args,**kwargs:server)
    monkeypatch.setattr(bench.bench.q,'owned_pids',lambda value:{123})
    monkeypatch.setattr(bench.bench.q,'stop_owned',stopped.append)
    monkeypatch.setattr(bench.native.launch,'build_launch',lambda *args,**kwargs:
                        dict(argv=['fake-server'],environment={}))
    telemetry_error=RuntimeError('lost telemetry')
    workload_cancelled=[]
    class Client:
        def __init__(self,*args,**kwargs):self.metrics_calls=0
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def get(self,url,**kwargs):
            if failure!='telemetry':await asyncio.Event().wait()
            if url.endswith('/metrics'):
                self.metrics_calls+=1
                if self.metrics_calls>1:raise telemetry_error
            return SimpleNamespace(status_code=200,text='',raise_for_status=lambda:None)
    monkeypatch.setitem(sys.modules,'httpx',SimpleNamespace(AsyncClient=Client,
        Timeout=lambda *args,**kwargs:None,Limits=lambda **kwargs:None,HTTPError=OSError))
    monkeypatch.setattr(bench.native,'metrics',lambda text:dict(
        **{'vllm:num_requests_running':0,'vllm:num_requests_waiting':0}))
    async def workload(*args):
        try:await asyncio.Event().wait()
        finally:workload_cancelled.append(True)
    monkeypatch.setattr(bench.work,'workload',workload)
    (tmp_path/'native.json').write_text('{}')
    (tmp_path/'fixture.json').write_text('{}')
    manifest=dict(idle_seconds=0,port=1,workload={},
        budget=dict(required_available_bytes=1,minimum_free_bytes=1))
    with pytest.raises((ValueError,RuntimeError)) as caught:
        asyncio.run(bench.run_arm(tmp_path,'native',manifest))
    if failure=='telemetry':
        assert caught.value is telemetry_error and workload_cancelled
    elif failure=='foreign_gpu':
        assert str(caught.value)=='Foreign GPU process appeared'
    else:
        assert str(caught.value)=='Host available memory below8GiB'
    assert stopped==[server]
    timing=json.loads((tmp_path/'native/timing.json').read_text())
    assert timing['elapsed_seconds']>=0 and timing['finished_wall']>=timing['started_wall']
    assert not(tmp_path/'native/results.json').exists()
