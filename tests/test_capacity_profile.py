import asyncio
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

spec=importlib.util.spec_from_file_location('capacity_profile_test',
    Path(__file__).parents[1]/'scripts/profile_capacity_chain.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def test_reference_verifier_runs_in_its_own_frozen_tree(tmp_path):
    entry=tmp_path/'frozen/scripts/benchmark_capacity_chain.py'
    entry.parent.mkdir(parents=True)
    entry.write_text('from pathlib import Path\n'
                    'def verify(root):\n'
                    '    assert Path(__file__).parents[1] == root / "frozen"\n'
                    '    return {"verified_reference": str(root)}\n')
    assert p.verified_reference(tmp_path)=={'verified_reference':str(tmp_path)}


def test_profile_changes_only_port_and_bounded_profiler_config(tmp_path):
    base=dict(port=18576,max_num_seqs=8,profiler_config=None,
              max_model_len=262144,kvmem={'select_budget':65536,'gen_budget':32768})
    before=copy.deepcopy(base);actual=p.configuration(base,tmp_path,18578)
    assert base==before
    assert {k for k in actual if actual[k]!=before[k]}=={'port','profiler_config'}
    assert actual['profiler_config']['max_iterations']==32
    assert actual['profiler_config']['ignore_frontend'] is True
    assert actual['profiler_config']['torch_profiler_with_stack'] is False


def test_freeze_replaces_stale_qualification_server_as_declared_source(tmp_path):
    reference=tmp_path/'reference';root=tmp_path/'diagnostic'
    source=reference/'frozen';(source/'scripts').mkdir(parents=True)
    (source/p.SERVER_ENTRY).write_text('select_budget = 32768\ngen_budget = 4096\n')
    (source/'scripts/keep.py').write_text('unchanged\n')
    delta=p.freeze_diagnostic_sources(reference,root)
    assert set(delta)=={p.ENTRY,p.SERVER_ENTRY}
    assert (root/'frozen'/p.SERVER_ENTRY).read_bytes()==(p.ROOT/p.SERVER_ENTRY).read_bytes()
    assert (root/'frozen/scripts/keep.py').read_bytes()==b'unchanged\n'
    assert (source/p.SERVER_ENTRY).read_text().startswith('select_budget = 32768')
    assert delta[p.SERVER_ENTRY]==p.cap.sha(root/'frozen'/p.SERVER_ENTRY)


@pytest.mark.parametrize('failed',[False,True])
def test_server_exit_is_recorded_after_startup_failure_too(tmp_path,monkeypatch,failed):
    events=[]
    async def server(*_):
        events.append('owned_model_stopped')
        if failed:raise ValueError('geometry mismatch')
        return {'passed':True}
    def proof(root,arm):
        assert events==['owned_model_stopped']
        assert root==tmp_path and arm=='diagnostic' and (root/arm).is_dir()
        events.append('exit_observed')
    monkeypatch.setattr(p.shared,'run_server',server)
    monkeypatch.setattr(p.chain,'post_stop_proof',proof)
    if failed:
        with pytest.raises(ValueError,match='geometry mismatch'):
            asyncio.run(p.run_diagnostic_server(tmp_path,{}))
    else:
        assert asyncio.run(p.run_diagnostic_server(tmp_path,{}))=={'passed':True}
    assert events==['owned_model_stopped','exit_observed']


def trace(tmp_path,*,mixed=True,kernel=True,scope=True,steps=32):
    events=[dict(ph='X',name='execute_context_1(2048)_generation_7(35)' if mixed and i==3
                 else 'execute_context_0(0)_generation_8(40)',ts=i*100,dur=100,cat='user_annotation')
            for i in range(steps)]
    if kernel:events.append(dict(ph='X',name='kernel',cat='kernel',ts=100,dur=80))
    if scope:
        for name in ('capture.submit','capture.pack_fp8','capture.drain_requests','action.capture'):
            events.append(dict(ph='X',name='kvmem::'+name,cat='user_annotation',ts=100,dur=10))
    path=tmp_path/'worker.pt.trace.json';path.write_text(json.dumps({'traceEvents':events}));return path


def test_trace_requires_actual_mixed_work_and_does_not_claim_performance(tmp_path):
    result=p.trace_coverage([trace(tmp_path)])
    assert result['mixed_steps']==1 and result['native_steps']==32 and result['kernels']==1
    assert result['performance_eligible'] is False
    assert result['cpu_scopes']['kvmem::capture.pack_fp8']['summed_cpu_us']==10


def test_gpu_annotations_do_not_duplicate_cpu_scopes(tmp_path):
    path=trace(tmp_path);body=json.loads(path.read_text())
    body['traceEvents'].append(dict(ph='X',name='kvmem::capture.pack_fp8',
                                   cat='gpu_user_annotation',ts=100,dur=500))
    path.write_text(json.dumps(body))
    assert p.trace_coverage([path])['cpu_scopes']['kvmem::capture.pack_fp8']==dict(calls=1,summed_cpu_us=10)


def test_gpu_only_scope_names_do_not_prove_host_coverage(tmp_path):
    path=trace(tmp_path);body=json.loads(path.read_text())
    for row in body['traceEvents']:
        if row['name'].startswith('kvmem::'):row['cat']='gpu_user_annotation'
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='scopes missing'):p.trace_coverage([path])


@pytest.mark.parametrize('change,reason',[
    ({'mixed':False},'no mixed'),({'kernel':False},'CUDA kernels'),
    ({'scope':False},'scopes missing'),({'steps':31},'32 native steps')])
def test_incomplete_or_wrong_phase_trace_is_not_a_success(tmp_path,change,reason):
    with pytest.raises(ValueError,match=reason):p.trace_coverage([trace(tmp_path,**change)])


@pytest.mark.parametrize('change',[{'ph':'B'},{'dur':float('inf')},{'dur':-1},{'ts':None}])
def test_invalid_native_timeline_cannot_pass_coverage(tmp_path,change):
    path=trace(tmp_path);body=json.loads(path.read_text());body['traceEvents'][0].update(change)
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='Incomplete'):p.trace_coverage([path])


@pytest.mark.parametrize('fail',[False,True])
def test_profiler_starts_only_after_full_warmup_and_restores_hook(tmp_path,monkeypatch,fail):
    phases=[];posts=[];directory=tmp_path/'workload';directory.mkdir()
    async def original_retirement(*args,**kwargs):
        phases.append('retired');return {'valid':True}
    async def workload(client,profile,fixture,where,settings):
        assert settings.duration_seconds==60 and settings.generation_limit==16384
        assert not posts
        await p.chain.work.wait_retirement(None,expected_views={})
        assert not posts
        await p.chain.work.wait_retirement(None,expected_views={str(i):100000 for i in range(8)})
        assert phases==['retired','retired','profile_started']
        if fail:raise RuntimeError('request failed')
        await p.chain.work.wait_retirement(None,expected_views={})
        p.cap.save(where/'traces.json',[])
        return {'total_stream_output_tokens':10}
    async def released(original,*args):return await original(*args[:-1])
    # released_workload has directory/settings followed by the qualification.
    monkeypatch.setattr(p.chain.work,'workload',workload)
    monkeypatch.setattr(p.chain.work,'wait_retirement',original_retirement)
    monkeypatch.setattr(p.chain,'released_workload',released)
    monkeypatch.setattr(p.chain,'own_history_proof',lambda *args:{'passed':True})
    p.cap.save(tmp_path/'manifest.json',{'release_qualification':{}})
    class Client:
        async def post(self,url,**kwargs):
            posts.append(url)
            if url.endswith('/start_profile'):phases.append('profile_started')
            return NS(raise_for_status=lambda:None)
    m=dict(reference=str(tmp_path),workload=dict(duration_seconds=60,generation_limit=16384))
    coro=p.measured_workload(Client(),{'port':18578},{},directory,tmp_path/'capture',manifest=m)
    if fail:
        with pytest.raises(RuntimeError,match='request failed'):asyncio.run(coro)
    else:
        assert asyncio.run(coro)['performance_eligible'] is False
    assert posts==['http://127.0.0.1:18578/start_profile','http://127.0.0.1:18578/stop_profile']
    assert p.chain.work.wait_retirement is original_retirement
