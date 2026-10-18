import importlib.util
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('aligned_fp8_benchmark_test',
    ROOT/'scripts/benchmark_aligned_fp8_agents.py')
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def profile():
    return dict(model='/fp8', port=18588, profiler_config=None, max_num_seqs=8,
        max_model_len=262144, draft_method='mtp', draft_tokens=4, execution_mode='compiled_graph',
        opaque_setting={'preserve': [3]}, kvmem=dict(select_budget=65536, gen_budget=32768,
        archive_backend='cpu', cpu_read_cache_bytes=100_000_000_000))


@pytest.mark.parametrize('duration', [0, 179, 301, float('nan'), float('inf'), True])
def test_invalid_or_adaptive_window_is_rejected(duration):
    with pytest.raises(ValueError):
        bench.checked_workload(duration, 0)


def test_natural_g16_and_capacity_are_never_renamed_or_reduced(tmp_path):
    original = profile()
    result = bench.selected_profile(original, 18608, tmp_path, 0)
    assert result == {**original, 'port': 18608}
    result['opaque_setting']['preserve'].append(4)
    assert original['opaque_setting']['preserve'] == [3]
    settings = bench.checked_workload(300, 0)
    assert settings == dict(bench.chain.WORKLOAD, duration_seconds=300.)
    assert settings['generation_limit'] == 16384
    assert result['kvmem']['gen_budget'] == 32768


def test_native_gc_ablation_only_changes_declared_allocator_control(tmp_path):
    p = profile()
    enabled = bench.selected_profile(p,18608,tmp_path,0,cuda_allocator_policy='native_gc_90')
    assert enabled == dict(p,port=18608,cuda_allocator_conf=bench.CUDA_CACHE_GC)
    assert bench.context_profile(enabled) == dict(p,port=18608)
    enabled['cuda_allocator_conf']['garbage_collection_threshold'] = .8
    with pytest.raises(ValueError,match='declared native90%'):
        bench.context_profile(enabled)
    with pytest.raises(ValueError):
        bench.selected_profile(p,18608,tmp_path,32,cuda_allocator_policy='native_gc_90')
    with pytest.raises(ValueError):
        bench.selected_profile(p,18608,tmp_path,0,cuda_allocator_policy='unrecorded')


def test_diagnostic_is_separate_60_seconds_with_bounded_profiler(tmp_path):
    result = bench.selected_profile(profile(), 18608, tmp_path, 32)
    assert result['profiler_config']['max_iterations'] == 32
    assert result['profiler_config']['ignore_frontend'] is True
    assert bench.checked_workload(300, 32)['duration_seconds'] == 60
    with pytest.raises(ValueError):
        bench.checked_workload(300, 33)


@pytest.mark.parametrize('failure', [None, 'runtime', 'profile', 'fixture'])
def test_diagnostic_uses_exact_completed_runtime_without_requalifying_weights(tmp_path, failure):
    p = profile(); runtime = {'src/a.py':'same','scripts/worker.py':'same'}
    baseline = {'port':p['port'], 'runtime_sha256':{bench.ARM:dict(runtime)},'fixture_sha256':'same'}
    diagnostic = bench.selected_profile(p, 18608, tmp_path, 32)
    fixture = 'same'
    if failure == 'runtime':runtime['src/a.py'] = 'other'
    if failure == 'profile':diagnostic['opaque_setting']['preserve'].append(9)
    if failure == 'fixture':fixture = 'other'
    if failure is None:
        bench.diagnostic_contract(baseline,p,diagnostic,runtime,fixture)
    else:
        with pytest.raises(ValueError, match='exact accepted baseline'):
            bench.diagnostic_contract(baseline,p,diagnostic,runtime,fixture)


@pytest.mark.parametrize('field,value', [('select_budget', 32768), ('gen_budget', 16384),
    ('archive_backend', 'ssd'), ('cpu_read_cache_bytes', 50_000_000_000),
    ('graph_mode', 'multi'), ('repository_gc', True)])
def test_profile_drift_is_rejected(tmp_path, field, value):
    p = profile(); p['kvmem'][field] = value
    with pytest.raises(ValueError):
        bench.selected_profile(p, 18608, tmp_path, 0)


def test_candidate_requires_complete_explicit_production_delta():
    before = {'src/a.py':'old', 'csrc/old.cc':'removed', 'scripts/harness.py':'h1'}
    after = {'src/a.py':'new', 'csrc/new.cc':'added', 'scripts/harness.py':'h2'}
    expected = {'src/a.py':['old','new'], 'csrc/old.cc':['removed',None],
        'csrc/new.cc':[None,'added']}
    assert bench.declared_delta(before, after, expected) == expected
    with pytest.raises(ValueError):
        bench.declared_delta(before, after, ['src/a.py'])
    with pytest.raises(ValueError):
        bench.declared_delta(after, after, [])


def test_runtime_hashes_do_not_include_artifacts_or_bytecode(tmp_path):
    for name in ('src/a.py', 'scripts/runner.py', 'csrc/a.cc', 'src/__pycache__/a.pyc', 'artifacts/large.json'):
        file = tmp_path/name; file.parent.mkdir(parents=True, exist_ok=True); file.write_text(name)
    assert set(bench.runtime_hashes(tmp_path)) == {'src/a.py', 'scripts/runner.py', 'csrc/a.cc'}


def make_alignment(tmp_path):
    source = tmp_path/'runtime/src/a.py'; source.parent.mkdir(parents=True); source.write_text('source')
    save(tmp_path/'fixture.json', {'fixture':True}); save(tmp_path/'profile.json', {'model':'NVFP4'})
    mapping = {'src/a.py':bench.cap.sha(source)}
    complete = dict(mapping)
    for name in ('qualify_context_alignment.py', 'qualify_block_graph.py'):
        helper = tmp_path/'runtime/scripts'/name; helper.parent.mkdir(exist_ok=True); helper.write_text(name)
        complete['scripts/'+name] = bench.cap.sha(helper)
    save(tmp_path/'manifest.json', dict(qualification_script='qualify_context_alignment.py',
        runtime_sha256=complete, fixture=str(tmp_path/'fixture.json'),
        fixture_sha256=bench.cap.sha(tmp_path/'fixture.json'), profile_sha256=bench.cap.sha(tmp_path/'profile.json')))
    save(tmp_path/'status.json', {'status':'passed'})
    save(tmp_path/'qualification/results.json', {'status':'passed','all_owned_sessions_released':True})
    save(tmp_path/'cleanup.json', {'remaining_owned_process_groups':{}, 'remaining_gpu_pids':[]})
    return mapping


def test_alignment_is_source_proof_not_fp8_model_qualification(tmp_path):
    mapping = make_alignment(tmp_path)
    proof = bench.alignment_proof(tmp_path, mapping, completed=True)
    assert 'separate exact-runtime FP8' in proof['scope']
    with pytest.raises(ValueError):
        bench.alignment_proof(tmp_path, {**mapping, 'src/new.py':'different'}, completed=True)


@pytest.mark.parametrize('failure', ['running', 'sessions', 'gpu', 'source', 'fixture', 'client'])
def test_alignment_requires_complete_immutable_qualification(tmp_path, failure):
    mapping = make_alignment(tmp_path)
    if failure == 'running':save(tmp_path/'status.json', {'status':'running'})
    if failure == 'sessions':save(tmp_path/'qualification/results.json', {'status':'passed'})
    if failure == 'gpu':save(tmp_path/'cleanup.json', {'remaining_owned_process_groups':{},'remaining_gpu_pids':[123]})
    if failure == 'source':(tmp_path/'runtime/src/a.py').write_text('changed')
    if failure == 'fixture':save(tmp_path/'fixture.json', {'fixture':False})
    if failure == 'client':
        helper = tmp_path/'runtime/scripts/qualify_context_alignment.py'
        helper.write_text('unfrozen client')
    with pytest.raises(ValueError):
        bench.alignment_proof(tmp_path, mapping, completed=True)


@pytest.mark.parametrize('failure', ['runtime', 'model', 'status', 'history', 'cleanup', 'oracle', 'counts', 'preemption'])
def test_new_runtime_fp8_gate_cannot_reuse_unproven_or_incomplete_result(tmp_path, monkeypatch, failure):
    runtime = {'src/a.py':'new'}
    qualified = dict(runtime)
    p = dict(profile(), port=18609)
    oracle = {'passed':True}
    result = dict(status='passed',cleanup_completed=True,history_proofs=[{}]*6,oracle=oracle,
        native_reconciliation={'passed':True,'preemptions':0,'native_tokens':12,'client_tokens':12})
    status = 'completed_pending_review'
    if failure == 'runtime':qualified['src/a.py'] = 'old'
    if failure == 'model':p['model'] = '/nvfp4'
    if failure == 'status':status = 'running'
    if failure == 'history':result['history_proofs'] = []
    if failure == 'cleanup':result['cleanup_completed'] = False
    if failure == 'oracle':result['oracle'] = {'passed':False}
    if failure == 'counts':result['native_reconciliation']['client_tokens'] = 11
    if failure == 'preemption':result['native_reconciliation']['preemptions'] = 1
    for name, value in [('profile.json',p),('results.json',result),('status.json',{'status':status}),
        ('manifest.json',{}),('workload/traces.json',[])]:save(tmp_path/name,value)
    gate = SimpleNamespace(verify=lambda _: {'runtime_sha256':qualified},
        qualification_profile=lambda p,port:dict(p,port=port), oracle_evidence=lambda *_:oracle)
    monkeypatch.setattr(bench, 'module', lambda *_:gate)
    with pytest.raises(ValueError):
        bench.fp8_qualification(tmp_path, profile(), runtime, completed=True)


@pytest.mark.parametrize('failure', [None, 'missing', 'live_gpu', 'wrong_performance', 'changed_post_stop'])
def test_fp8_qualification_requires_protected_final_gpu_cleanup(tmp_path, monkeypatch, failure):
    root = tmp_path/'qualification'; runtime = {'src/a.py':'new'}
    oracle = {'passed':True}
    result = dict(status='passed',cleanup_completed=True,history_proofs=[{}]*6,oracle=oracle,
        native_reconciliation={'passed':True,'preemptions':0,'native_tokens':12,'client_tokens':12})
    for name,value in [('profile.json',dict(profile(),port=18609)),('manifest.json',{}),
        ('results.json',result),('status.json',{'status':'completed_pending_review'}),('workload/traces.json',[])]:
        save(root/name,value)
    save(tmp_path/'manifest.json', {'performance':'binding'})
    gate = SimpleNamespace(verify=lambda _: {'runtime_sha256':runtime},
        qualification_profile=lambda p,port:dict(p,port=port), oracle_evidence=lambda *_:oracle)
    monkeypatch.setattr(bench, 'module', lambda *_:gate)
    if failure != 'missing':
        save(root/'process/post-stop.json', {'gpu_empty':failure != 'live_gpu',
            'observed_gpu_pids':[123] if failure == 'live_gpu' else []})
        binding = dict(protocol='aligned_fp8_qualification_cleanup_v1',
            qualification_manifest_sha256=bench.cap.sha(root/'manifest.json'),
            performance_manifest_sha256=bench.cap.sha(tmp_path/'manifest.json'),
            post_stop_sha256=bench.cap.sha(root/'process/post-stop.json'))
        if failure == 'wrong_performance':binding['performance_manifest_sha256'] = 'other'
        save(root/'process/qualification-cleanup.json',binding)
        if failure == 'changed_post_stop':save(root/'process/post-stop.json', {'gpu_empty':True,'observed_gpu_pids':[],'changed':True})
    if failure is None:
        assert bench.fp8_qualification(root, profile(), runtime, completed=True)['path'] == str(root)
    else:
        with pytest.raises(ValueError, match='protected owned-process cleanup'):
            bench.fp8_qualification(root, profile(), runtime, completed=True)


def test_separate_metrics_use_executed_prefill_not_completed_prompt_credit(tmp_path, monkeypatch):
    save(tmp_path/'manifest.json', {'performance_eligible':False})
    save(tmp_path/bench.ARM/'tools/window.json', {'begin':1.,'end':301.})
    executed = {'completed_prefill_tokens_per_second_bounds':[2063.44,2083.92]}
    save(tmp_path/bench.ARM/'results.json', {'output_tokens_per_second':134.,
        'requests_completed_in_window':95, 'public_work':executed})
    monkeypatch.setattr(bench, 'module', lambda *_:SimpleNamespace(analyze=lambda *a,**kw:{'steps':'separate'}))
    report = bench.separated_report(tmp_path)
    assert report['actual_public_prefill'] == executed
    assert report['accepted_output_tokens_per_wall_second'] == 134.
    assert report['completed_turns'] == 95
    assert report['performance_eligible'] is False


@pytest.mark.parametrize('failure', ['running', 'diagnostic', 'headroom', 'reconciliation', 'history', 'release'])
def test_candidate_cannot_select_an_unaccepted_baseline(tmp_path, monkeypatch, failure):
    manifest = {'experiment_kind':bench.KIND,'performance_eligible':True}
    status = {'status':'completed_pending_review'}
    deployment = {'accepted':True}
    result = {'drained':True, 'native_client_tokens_equal':True,
        'explicit_session_release_verified':True, 'preemptions':0}
    history = {'passed':True, 'complete_own_outputs':True}
    if failure == 'running':status['status'] = 'running'
    if failure == 'diagnostic':manifest['performance_eligible'] = False
    if failure == 'headroom':deployment['accepted'] = False
    if failure == 'reconciliation':result['native_client_tokens_equal'] = False
    if failure == 'history':history['complete_own_outputs'] = False
    if failure == 'release':result['explicit_session_release_verified'] = False
    for name,value in [('status.json',status), (bench.ARM+'/results.json',result),
        (bench.ARM+'/deployment-acceptance.json',deployment), (bench.ARM+'/own-history-proof.json',history)]:
        save(tmp_path/name,value)
    calls = []
    def verify(root, *, completed):
        calls.append((root, completed)); return manifest
    monkeypatch.setattr(bench, 'module', lambda *_:SimpleNamespace(verify=verify))
    with pytest.raises(ValueError, match='completed, accepted'):
        bench.baseline_proof(tmp_path)
    assert calls == [(tmp_path,False)]


def test_failed_workload_still_retains_post_stop_proof(tmp_path, monkeypatch):
    events = []
    async def workload(*args):
        try:
            raise ValueError('HTTP failed')
        finally:
            events.append('stopped')
    monkeypatch.setattr(bench.chain, 'post_stop_proof', lambda *a:events.append('proof'))
    with pytest.raises(ValueError, match='HTTP failed'):
        asyncio.run(bench.guarded_arm(workload, tmp_path, 'arm'))
    assert events == ['stopped', 'proof']


def test_repeated_cancel_does_not_abandon_owned_stop_or_post_stop(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(bench.chain, 'post_stop_proof', lambda *a:events.append('proof'))
    async def scenario():
        running = asyncio.Event(); stopping = asyncio.Event(); released = asyncio.Event()
        async def workload(*args):
            try:
                running.set()
                await asyncio.Event().wait()
            finally:
                events.append('stop_started'); stopping.set()
                await released.wait()
                events.append('stopped')
        task = asyncio.create_task(bench.guarded_arm(workload, tmp_path, 'arm'))
        await running.wait(); task.cancel(); await stopping.wait()
        task.cancel(); await asyncio.sleep(0)
        released.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(scenario())
    assert events == ['stop_started', 'stopped', 'proof']
