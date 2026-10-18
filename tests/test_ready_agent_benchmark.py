import importlib.util
import asyncio
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('ready_agent_benchmark',
    Path(__file__).parents[1]/'scripts/benchmark_ready_agents.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


@pytest.mark.parametrize('foreign,after_wait',[(False,110),(False,90),(True,110)])
def test_prelaunch_wait_follows_gpu_probe_and_never_bypasses_resource_checks(tmp_path,monkeypatch,foreign,after_wait):
    order=[];available=[110]
    monkeypatch.setattr(bench,'verify',lambda root:order.append('verify'))
    async def idle(*args):order.append('idle')
    monkeypatch.setattr(bench.bench.q,'wait_idle',idle)
    def gpu():
        order.append('gpu_process_probe');available[0]=90
        return dict(pids=[42] if foreign else [])
    monkeypatch.setattr(bench.bench.q,'gpu_state',gpu)
    async def admission(*args):
        assert order[-1]=='gpu_process_probe'
        order.append('host_wait');available[0]=after_wait
    def memory():order.append('final_host_check');return dict(MemAvailable=available[0])
    monkeypatch.setattr(bench.cap,'host_memory',memory)
    manifest=dict(idle_seconds=10,budget=dict(required_available_bytes=100))
    if foreign or after_wait<100:
        with pytest.raises(ValueError,match='Foreign GPU' if foreign else 'Insufficient host'):
            asyncio.run(bench.await_launch_admission(tmp_path,'native',manifest,prelaunch_check=admission))
    else:
        asyncio.run(bench.await_launch_admission(tmp_path,'native',manifest,prelaunch_check=admission))
    assert order==(['verify','idle','gpu_process_probe'] if foreign else
                   ['verify','idle','gpu_process_probe','host_wait','final_host_check'])
    assert manifest['budget']['required_available_bytes']==100


def fixture(users=2, initial=40):
    return dict(eos=0,agent_plans=[dict(initial_tokens=[i+1]*initial,
        query_plan=dict(kvmem_query_span=[0,2]),turns=[dict(output_tokens=8,
            tool_tokens=[] if t==0 else [10],tool_wait_seconds=0 if t==0 else .1)
            for t in range(4)]) for i in range(users)])


def test_memory_bounds_include_entire_finite_history_and_system_margin():
    b=bench.fixture_bounds(fixture(),2,4)
    assert b['history_limits']==[75,75]
    assert b['archive_bytes']==2*(96+64)*34816
    assert b['required_available_bytes']==b['archive_bytes']+2*805306368+32*1024**3


def test_full_history_ceiling_rejects_workload_before_any_gpu_launch():
    with pytest.raises(ValueError,match='historical ceiling'):
        bench.fixture_bounds(fixture(initial=262144),2,4)


def test_missing_eos_is_rejected_before_gpu_admission():
    f=fixture();del f['eos']
    with pytest.raises(ValueError,match='EOS'):
        bench.fixture_bounds(f,2,4)


def test_cpu_cap_is_checked_independently_of_native_gpu_slots():
    with pytest.raises(ValueError,match='CPU history'):
        bench.fixture_bounds(fixture(users=20,initial=200000),20,4)


def test_bad_tool_delay_or_output_quota_cannot_be_silently_clipped():
    f=fixture();f['agent_plans'][0]['turns'][2]['tool_wait_seconds']=-1
    with pytest.raises(ValueError,match='wait'):
        bench.fixture_bounds(f,2,4)
    f=fixture();f['agent_plans'][0]['turns'][2]['output_tokens']=32769
    with pytest.raises(ValueError,match='gen_budget'):
        bench.fixture_bounds(f,2,4)


def test_live_checkout_cannot_execute_a_prepared_experiment(tmp_path):
    bench.cap.save(tmp_path/'manifest.json',dict(kind='ready_agents_v1'))
    with pytest.raises(ValueError,match='frozen entry'):
        bench.verify(tmp_path)


def test_native_slot_tuning_preserves_original_default():
    assert bench.native_sequence_limit(None,12)==12
    assert bench.native_sequence_limit(8,12)==8
    for value in (0,13,True,8.0):
        with pytest.raises(ValueError,match='active sequence limit'):
            bench.native_sequence_limit(value,12)
