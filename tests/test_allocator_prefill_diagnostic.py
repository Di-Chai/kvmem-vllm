"""CPU-only identity, exact warmup transformation and cleanup checks."""
import asyncio
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

spec = importlib.util.spec_from_file_location('allocator_prefill_test',
    Path(__file__).parents[1]/'scripts/diagnose_allocator_prefill.py')
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def fixture():
    return dict(eos=0,agent_plans=[dict(initial_tokens=[i+1,2,3,4],
        query_plan={'kvmem_query_span':[0,3]},turns=[dict(output_tokens=2,
            tool_tokens=[] if turn == 0 else [90+i,turn],tool_wait_seconds=0. if turn == 0 else .25)
            for turn in range(16)]) for i in range(12)])


def resources(views, at=1000.):
    return dict(at=at,session_views=[dict(namespace=[s,s],history=h,window=h,active_records=[[0,h]])
                                   for s,h in views.items()])


def test_transform_changes_only_post_barrier_and_retains_original_warm_body():
    source = inspect.getsource(diag.ready.work.workload)
    changed = diag.transform_workload(source)
    before = ast.parse(source).body[0]
    after = ast.parse(changed).body[0]
    nested = lambda tree,name:next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name == name)
    assert ast.dump(nested(before,'warm')) == ast.dump(nested(after,'warm'))
    assert ast.dump(nested(before,'request')) == ast.dump(nested(after,'request'))
    assert changed.index('allocator_phase("drain")') > changed.index("save(directory / 'resources-before.json'")
    assert 'await asyncio.gather(*(agent(i)' not in changed
    assert 'await asyncio.gather(*(request(i, [fixture[\'eos\']]' in changed
    with pytest.raises(ValueError,match='transformation boundary'):
        diag.transform_workload(changed)


def test_zero_window_summary_has_no_rate_or_fake_utilization():
    result = diag.diagnostic_summary([dict(kind='warmup',status='completed',output_ids=[8,9])],
                                    dict(begin=7.,end=7.,duration_seconds=0),[])
    assert result['total_stream_output_tokens'] == 2
    assert not any('second' in key or 'tps' in key for key in result if key != 'duration_seconds')
    assert diag.no_window()['available'] is False
    with pytest.raises(ValueError,match='no measured window'):
        diag.diagnostic_summary([],dict(begin=7.,end=8.,duration_seconds=1),[])
    with pytest.raises(ValueError,match='producer'):
        diag.diagnostic_summary([dict(kind='workload',status='completed')],dict(begin=7.,end=7.,duration_seconds=0),[])


def test_server_entry_survives_original_serve_environment_cleaning(tmp_path,monkeypatch):
    root = tmp_path
    package = root/'runtime'/diag.ARM/'src/vllm_kvmem'
    original = dict(profile={'model':'unchanged'},vllm_argv=['serve','exact-model'],
                    environment={'KVMEM_SERVING':'1','VLLM_USE_V2_MODEL_RUNNER':'1'})
    seen = []
    def build(*args,**kwargs):
        assert 'KVMEM_UNRELATED' not in os.environ and 'VLLM_UNRELATED' not in os.environ
        return copy.deepcopy(original)
    serve = NS(__file__=str(package/'serve.py'),build_launch=build)
    def main(args):
        plan = serve.build_launch(None,None,None)
        # Exact environment clearing performed by the unchanged serve.main.
        for name in list(os.environ):
            if name.startswith('KVMEM_'):
                del os.environ[name]
        os.environ.update(plan['environment'])
        assert os.environ['KVMEM_ALLOCATOR_DIAGNOSTIC'] == '1'
        assert os.environ['KVMEM_ALLOCATOR_DIAGNOSTIC_PHASE_FILE'] == str(root/'allocator_phase.json')
        assert os.environ['VLLM_LOG_STATS_INTERVAL'] == '1'
        assert plan['profile'] == original['profile'] and plan['vllm_argv'] == original['vllm_argv']
        assert set(plan['environment'])-set(original['environment']) == {
            'KVMEM_ALLOCATOR_DIAGNOSTIC','KVMEM_ALLOCATOR_DIAGNOSTIC_PHASE_FILE'}
        seen.append(args)
    serve.main = main
    monkeypatch.setattr(diag,'ROOT',root/'frozen')
    monkeypatch.setitem(sys.modules,'vllm_kvmem',NS(__file__=str(package/'__init__.py'),serve=serve))
    monkeypatch.setattr(os,'environ',dict(os.environ,KVMEM_UNRELATED='bad',VLLM_UNRELATED='bad'))
    diag.server_entry(root)
    assert len(seen) == 1 and serve.build_launch is build
    assert diag.cap.read(root/'server-plan-identity.json')['original'] == original


def test_original_warmup_and_cleanup_run_48_requests_without_producers(tmp_path,monkeypatch):
    source = diag.transform_workload(inspect.getsource(diag.ready.work.workload))
    calls,phases,retirements = [],[],[]
    async def stream(client,profile,tokens,session,count,recorder,**kw):
        assert kw['kind'] != 'workload'
        row = dict(begin=float(len(calls)+1),end=float(len(calls)+1)+.1,session=session,
            agent=kw['agent'],epoch=kw['epoch'],turn=kw['turn'],kind=kw['kind'],request_id=kw['request_id'],
            input_tokens=len(tokens),input_sha256=diag.ready.work.digest(tokens),
            output_ids=[100+kw['agent']]*count,usage=dict(completion_tokens=count),status='completed',timeline=[])
        row['output_sha256'] = diag.ready.work.digest(row['output_ids'])
        recorder.rows.append(row)
        calls.append(dict(tokens=list(tokens),count=count,**kw))
        return row
    async def retired(capture,*,not_before,expected_views):
        retirements.append(dict(expected_views))
        return resources(expected_views)
    namespace = dict(diag.ready.work.__dict__,summarize=diag.diagnostic_summary,allocator_phase=phases.append,
                     stream_call=stream,wait_retirement=retired)
    exec(compile(source,'<allocator-test>','exec'),namespace)
    root = tmp_path
    directory = root/diag.ARM/'tools'
    frozen = fixture()
    diag.cap.save(root/'fixture.json',frozen)
    result = asyncio.run(namespace['workload'](None,dict(kvmem={'gen_budget':32768}),frozen,directory,
        NS(concurrency=12,warmup_turns=2,epoch_turns=16,duration_seconds=300.,mode='tools',run_tag=diag.ARM)))
    assert result['drained'] and result['total_stream_output_tokens'] == 72
    assert phases == ['drain'] and len(calls) == 48
    assert sum(c['kind'] == 'warmup' for c in calls) == 24
    assert sum(c['kind'] == 'cleanup' for c in calls) == 24
    assert len(retirements) == 2 and set(retirements[-1].values()) == {1}
    for i,plan in enumerate(frozen['agent_plans']):
        warm = [c for c in calls if c['agent'] == i and c['kind'] == 'warmup']
        assert warm[0]['tokens'] == plan['initial_tokens']
        assert warm[1]['tokens'] == plan['initial_tokens']+[100+i]*2+plan['turns'][1]['tool_tokens']
    assert diag.replay_proof(root)['own_output_tool_chains_verified'] == 12
    rows = diag.cap.read(directory/'traces.json')
    changed = next(r for r in rows if r['kind'] == 'warmup' and r['turn'] == 1)
    changed['input_sha256'] = 'foreign output or clipped history'
    diag.cap.save(directory/'traces.json',rows)
    with pytest.raises(ValueError,match='substituted'):
        diag.replay_proof(root)


def test_original_warm_failure_is_not_a_diagnostic_success(tmp_path):
    source = diag.transform_workload(inspect.getsource(diag.ready.work.workload))
    async def fail(*args,**kw):
        raise RuntimeError('real stream failed')
    def never(phase):
        pytest.fail('Reached drain after failed warmup')
    namespace = dict(diag.ready.work.__dict__,summarize=diag.diagnostic_summary,allocator_phase=never,stream_call=fail)
    exec(compile(source,'<allocator-failure-test>','exec'),namespace)
    with pytest.raises(RuntimeError,match='real stream failed'):
        asyncio.run(namespace['workload'](None,dict(kvmem={'gen_budget':32768}),fixture(),tmp_path,
            NS(concurrency=12,warmup_turns=2,epoch_turns=16,duration_seconds=300.,mode='tools')))
    assert diag.cap.read(tmp_path/'results.json')['status'] == 'failed'


def test_phase_file_is_atomic_small_and_loading_is_unknown(tmp_path):
    write = diag.phase_writer(tmp_path)
    for phase in ('unknown','warmup','drain'):
        write(phase)
        assert diag.cap.read(tmp_path/'allocator_phase.json')['phase'] == phase
    assert not (tmp_path/'allocator_phase.tmp').exists()
    assert len((tmp_path/'phase_events.jsonl').read_text().splitlines()) == 3
    with pytest.raises(ValueError,match='Unexpected allocator phase'):
        write('formal')


def test_overlay_allowlist_and_before_hashes_are_strict(tmp_path):
    base,overlay = tmp_path/'base',tmp_path/'overlay'
    rows = {}
    for name in diag.FILES:
        target = overlay/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('diagnostic '+name)
        if name.endswith('allocator_diagnostic.py'):
            before = None
        else:
            source = base/name
            source.parent.mkdir(parents=True,exist_ok=True)
            source.write_text('original '+name)
            before = diag.cap.sha(source)
        rows[name] = dict(before_sha256=before,after_sha256=diag.cap.sha(target))
    manifest = dict(kind='isolated_allocator_diagnostic_overlay_v1',files=rows)
    assert set(diag.overlay_contract(base,overlay,manifest)) == diag.FILES
    (overlay/'unrelated.py').write_text('extra')
    with pytest.raises(ValueError,match='extra diagnostic'):
        diag.overlay_contract(base,overlay,manifest)
    (overlay/'unrelated.py').unlink()
    next(iter(rows.values()))['before_sha256'] = 'changed'
    with pytest.raises(ValueError,match='base changed'):
        diag.overlay_contract(base,overlay,manifest)


def test_profile_keeps_known_fraction_and_never_auto_retries_or_tunes():
    profile = dict(port=18542,gpu_memory_utilization=.82,max_num_seqs=12,
                   kvmem=dict(select_budget=65536,gen_budget=32768),other=['exact'])
    result = diag.diagnostic_profile(profile,18568)
    assert result == dict(profile,port=18568) and profile['port'] == 18542
    for key,value in (('gpu_memory_utilization',.81),('max_num_seqs',8)):
        with pytest.raises(ValueError,match='original twelve'):
            diag.diagnostic_profile(dict(profile,**{key:value}),18568)


def test_negative_headroom_is_observation_not_successful_deployment(tmp_path):
    directory = tmp_path/diag.ARM
    capture = directory/'server/capture'
    capture.mkdir(parents=True)
    (capture/'allocator_diagnostic.jsonl').write_text(json.dumps(dict(event='allocator',kind='prefill',phase='warmup',stage='capture.raw.end'))+'\n'+
        json.dumps(dict(event='summary',counts={'record_limit_drops':2}))+'\n')
    (directory/'gpu_samples.jsonl').write_text(json.dumps(dict(memory_mib=96646,wall_at=11))+'\n')
    result = diag.observations(tmp_path,dict(device_memory_identity=dict(total_mib=97887,driver_reserved_mib=638)))
    assert result['sampled_headroom_mib'] == 603 and result['headroom_below_four_gib']
    assert result['performance_claim_allowed'] is False and result['deployment_acceptance_allowed'] is False
    assert 'accepted' not in result and 'output_tokens_per_second' not in result
    assert result['allocator_final_summary']['counts']['record_limit_drops'] == 2
    reasons = {gap['reason'] for gap in result['coverage_gaps']}
    assert {'warmup_prefill_stages_missing','dropped_records_or_unfinished_batches',
            'final_post_cleanup_flush_not_provable'} <= reasons
    assert result['complete_process_high_water_proven'] is False


def test_drain_peak_covers_warmup_without_claiming_process_tail(tmp_path):
    directory = tmp_path/diag.ARM
    capture = directory/'server/capture'
    capture.mkdir(parents=True)
    rows = [dict(event='allocator',kind='prefill',phase='warmup',stage='capture.raw.end'),
            dict(event='allocator',kind='decode',phase='drain',host_monotonic=22.,
                 memory={'allocated_bytes.all.peak':12345}),dict(event='summary')]
    (capture/'allocator_diagnostic.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (directory/'gpu_samples.jsonl').write_text(json.dumps(dict(memory_mib=90000,wall_at=11))+'\n')
    (tmp_path/'phase_events.jsonl').write_text(json.dumps(dict(phase='drain',monotonic=21.))+'\n')
    result = diag.observations(tmp_path,dict(device_memory_identity=dict(total_mib=97887,driver_reserved_mib=638)))
    assert result['warmup_scope']['lifetime_high_water_includes_complete_warmup'] is True
    assert result['warmup_scope']['lifetime_peak_bytes_observed_after_barrier'] == {'allocated_bytes.all.peak':12345}
    assert result['complete_process_high_water_proven'] is False


def test_run_gate_fails_before_wait_or_popen(tmp_path,monkeypatch):
    monkeypatch.setattr(diag,'ROOT',tmp_path/'frozen')
    def rejected(root):
        raise ValueError('Exact overlay changed')
    monkeypatch.setattr(diag,'verify',rejected)
    monkeypatch.setattr(diag.ready,'run_arm',lambda *args:pytest.fail('Launch before gate'))
    with pytest.raises(ValueError,match='overlay changed'):
        asyncio.run(diag.run(NS(output=tmp_path)))


@pytest.mark.parametrize('mutation',['raise','manifest'])
def test_run_rechecks_after_idle_and_restores_adapters(tmp_path,monkeypatch,mutation):
    monkeypatch.setattr(diag,'ROOT',tmp_path/'frozen')
    diag.cap.save(tmp_path/'status.json',dict(status='prepared'))
    (tmp_path/'executed-workload.py').write_text('async def workload(*args,**kwargs):\n    pass\n')
    checks = []
    def verify(root):
        checks.append(1)
        if len(checks) > 1:
            if mutation == 'raise':
                raise ValueError('Source changed while waiting')
            return dict(changed=True)
        return {}
    async def idle(*args,**kw):
        return None
    async def arm(*args):
        await diag.ready.bench.q.wait_idle(None,1)
        pytest.fail('Identity changed but continued to model launch')
    monkeypatch.setattr(diag,'verify',verify)
    monkeypatch.setattr(diag.ready.bench.q,'wait_idle',idle)
    monkeypatch.setattr(diag.ready,'run_arm',arm)
    monkeypatch.setattr(diag.fcntl,'flock',lambda *args:None)
    original = diag.ready.work.workload
    with pytest.raises(ValueError,match='changed'):
        asyncio.run(diag.run(NS(output=tmp_path)))
    assert diag.ready.bench.q.wait_idle is idle and diag.ready.work.workload is original
    assert diag.cap.read(tmp_path/'status.json')['status'] == 'failed'
