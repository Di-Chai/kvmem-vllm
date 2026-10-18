import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


spec = importlib.util.spec_from_file_location('qualification', Path(__file__).parents[1]/'scripts/qualify_coordination.py')
qualification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualification)


def native_profile(tmp_path):
    from test_serving_launcher import nvfp4_profile
    base = nvfp4_profile(tmp_path)
    base.update(draft_method='mtp',draft_tokens=4,max_num_seqs=8)
    base.pop('draft_model',None)
    base['kvmem'].update(archive_backend='cpu',cpu_read_cache_bytes=80*1024**3,
        request_planning=True,async_refresh=True,session_capacity=8,
        session_capture_terminal=True,session_checkpoint_policy='pm')
    source=tmp_path/'profile.json'
    source.write_text(json.dumps(base))
    return source


def test_prepared_profiles_change_only_coordination_not_model_or_memory(tmp_path):
    source = native_profile(tmp_path)
    profiles=qualification.make_profiles(source,tmp_path)
    from vllm_kvmem.serve import build_launch
    for arm in profiles.values():
        build_launch(arm,tmp_path,tmp_path/'roundtrip')
        arm['kvmem'].pop('cpu_history_coordination')
        arm['kvmem'].pop('maintenance_max_steps')
    assert profiles['B0'] == profiles['B1'] == profiles['B2']
    assert profiles['B0']['kvmem']['step_latency_interval'] == 0
    assert not profiles['B0']['kvmem']['async_restore']
    assert not profiles['B0']['kvmem']['adaptive_prefill']


@pytest.mark.parametrize('interval', [0, 4, 8])
def test_v2_profiles_keep_fair_timing_and_disable_features_in_baseline(tmp_path, interval):
    source = native_profile(tmp_path)
    base = json.loads(source.read_text())
    base['kvmem'].update(async_restore=True, adaptive_prefill=True, step_latency_interval=16)
    source.write_text(json.dumps(base))
    profiles = qualification.make_profiles(source, tmp_path, arms=('B0', 'B2', 'B3', 'B4'),
                                           step_latency_interval=interval)
    from vllm_kvmem.serve import build_launch
    for name, profile in profiles.items():
        build_launch(profile, tmp_path, tmp_path/'roundtrip')
        options = profile['kvmem']
        assert options['step_latency_interval'] == (interval or 4)
        assert options['async_restore'] == (name in ('B3', 'B4'))
        assert options['adaptive_prefill'] == (name == 'B4')
        assert options['maintenance_max_steps'] == (8 if name in ('B2', 'B3', 'B4') else 0)
        assert options['cpu_history_coordination'] == (name != 'B0')
        for key in ('cpu_history_coordination', 'maintenance_max_steps', 'async_restore', 'adaptive_prefill'):
            options.pop(key)
    assert all(profile == profiles['B0'] for profile in profiles.values())


def test_v1_profiles_override_v2_base_switches_without_enabling_timing(tmp_path):
    source = native_profile(tmp_path)
    base = json.loads(source.read_text())
    base['kvmem'].update(async_restore=True, adaptive_prefill=True, step_latency_interval=4)
    source.write_text(json.dumps(base))
    profiles = qualification.make_profiles(source, tmp_path)
    assert set(profiles) == {'B0', 'B1', 'B2'}
    for profile in profiles.values():
        assert not profile['kvmem']['async_restore']
        assert not profile['kvmem']['adaptive_prefill']
        assert profile['kvmem']['step_latency_interval'] == 0


def test_v2_requires_a_real_pm_restore_profile(tmp_path):
    source = native_profile(tmp_path)
    base = json.loads(source.read_text())
    base['kvmem'].update(session_checkpoint_policy='sealed', session_capture_terminal=False,
                         session_capture_decode=True)
    source.write_text(json.dumps(base))
    with pytest.raises(ValueError, match='P/M sessions'):
        qualification.make_profiles(source, tmp_path, arms=('B0', 'B3'))


def test_v2_elastic_example_validates_and_propagates_flags_without_loading_model(tmp_path):
    from test_serving_launcher import nvfp4_profile
    from vllm_kvmem.elastic_serve import build_launch as build_elastic
    from vllm_kvmem.serve import build_launch
    model = nvfp4_profile(tmp_path)
    example = Path(__file__).parents[1]/'examples/qwen-nvfp4-coordinated-v2-elastic.json'
    profile = json.loads(example.read_text())
    profile.update(model=model['model'], tokenizer=model['tokenizer'])
    (tmp_path/'users.json').write_text(json.dumps({'users':[{'id':'alice','api_key':'test-only'}]}))
    resolved, _ = build_elastic(profile, tmp_path, tmp_path/'elastic')
    backend = resolved['mean_k_backend_profile']
    options = backend['kvmem']
    assert options['async_restore'] and options['adaptive_prefill']
    assert options['step_latency_interval'] == 4
    assert options['session_checkpoint_policy'] == 'pm'
    build_launch(backend, tmp_path, tmp_path/'backend')


def test_prepare_only_ignores_busy_execution_lock_and_never_queries_gpu(tmp_path, monkeypatch):
    source = native_profile(tmp_path)
    output = tmp_path/'preparation'
    output.mkdir()
    fixture = dict(prompt=[1, 2, 3], tool=[4], eos=0)
    (output/'fixture.json').write_text(json.dumps(fixture))
    lock_attempts = []

    def busy_lock(*args, **kwargs):
        lock_attempts.append(args)
        raise BlockingIOError('Existing qualification owns the execution lock')

    def forbidden_gpu(*args, **kwargs):
        pytest.fail('CPU-only preparation must not query GPU or execute an arm')

    monkeypatch.setattr(qualification, 'open', busy_lock, raising=False)
    monkeypatch.setattr(qualification, 'gpu_state', forbidden_gpu)
    monkeypatch.setattr(qualification, 'run_arm', forbidden_gpu)
    monkeypatch.setattr(qualification, 'wait_idle', forbidden_gpu)
    args = SimpleNamespace(config=source, output=output, arms=['B0', 'B2', 'B3', 'B4'],
                           step_latency_interval=4, pairs=5, cases=['decode', 'agents'], prepare_only=True)
    asyncio.run(qualification.main(args))
    assert lock_attempts == []
    assert json.loads((output/'status.json').read_text())['status'] == 'prepared'
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['fixture_sha256'] == qualification.digest(fixture)
    assert manifest['arms'] == args.arms
    assert manifest['effective_step_latency_interval'] == 4
    assert all((output/f'{arm}.json').is_file() for arm in args.arms)
    # Actual execution still takes the same exclusive lock before preparing or
    # launching work, and must fail while that lock is held by the older run.
    args.prepare_only = False
    with pytest.raises(BlockingIOError, match='owns the execution lock'):
        asyncio.run(qualification.main(args))
    assert len(lock_attempts) == 1


def test_stream_counts_are_checked_and_synthetic_requests_keep_prefix_ids():
    import json
    seen=[]
    def serve(request):
        seen.append(json.loads(request.content))
        events=[dict(id='id',choices=[dict(token_ids=[31,32])]),
                dict(id='id',choices=[],usage=dict(completion_tokens=2,prompt_tokens=256))]
        text=''.join('data: '+json.dumps(e)+'\n\n' for e in events)+'data: [DONE]\n\n'
        return httpx.Response(200,text=text)
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            return await qualification.call(client,dict(served_model_name='test',port=1234),
                list(range(256)), 'session', 2)
    result=asyncio.run(run())
    assert result['output_ids'] == [31,32]
    assert seen[0]['prompt'] == list(range(256))
    assert seen[0]['kvmem_query_span'] == [128,255]


def test_session_reset_explicitly_clears_old_mandatory_spans(monkeypatch):
    from vllm_kvmem.http_sessions import normalize_query_plan
    from vllm_kvmem.request_plan import RequestPlan
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', json.dumps(dict(
        block_tokens=32, gen_budget=16384, select_budget=36864, max_model_len=1048576)))
    plans = []
    def serve(request):
        payload = json.loads(request.content)
        assert payload['prompt'] == [0]
        assert payload['kvmem_mandatory_spans'] == []
        assert normalize_query_plan(payload)
        plan = RequestPlan.parse(payload['kv_transfer_params']['kvmem_request_plan'], 1, 32, 16384)
        continued, changed = plan.continuation(1, 1, 32768, 32768, 53248,
            previous_query=(32640, 32767), previous_mandatory=((32640, 32767, 'query'),))
        plans.append(RequestPlan.parse(continued.payload(), 1, 32, 16384))
        event = dict(id='reset', choices=[dict(token_ids=[1])], usage=dict(completion_tokens=1, prompt_tokens=1))
        return httpx.Response(200, text='data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n')
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
            return await qualification.call(client, dict(served_model_name='test', port=1234),
                [0], 'session', 1, query=False)
    assert asyncio.run(run())['output_ids'] == [1]
    assert plans[0].query is None and not plans[0].mandatory


def test_paired_comparison_does_not_call_one_run_proven_improvement():
    def measured(pair, arm, rate):
        return dict(pair=pair,arm=arm,results={'agents':dict(output_tokens_per_second=rate,
            output_event_gap_p95_seconds=.1)})
    results=[measured(0,'B0',100),measured(0,'B2',120)]
    assert not qualification.compare(results)['B2/agents']['passes_proposed_gate']
    results=[measured(i,arm,rate) for i in range(5) for arm,rate in [('B0',100),('B2',120)]]
    assert qualification.compare(results)['B2/agents']['passes_proposed_gate']


def test_v2_comparisons_preserve_paired_baseline_gates():
    results = [dict(pair=pair, arm=arm, results={'agents':dict(
        output_tokens_per_second=rate, output_event_gap_p95_seconds=.1)})
        for pair in range(5) for arm, rate in [('B0', 100), ('B2', 110), ('B3', 120), ('B4', 130)]]
    comparison = qualification.compare(results)
    assert comparison['B3/agents']['throughput_ratio_median'] == 1.2
    assert comparison['B4/agents']['throughput_ratio_median'] == 1.3
    assert comparison['B4/agents']['passes_proposed_gate']


def test_diagnostics_are_optional_completed_samples_and_keep_latency_separate(tmp_path):
    assert 'model_step_latency' not in qualification.collect_diagnostics(tmp_path)
    capture = tmp_path/'capture'
    capture.mkdir()
    timing = [dict(started_at=start, prefill_tokens=prefill, decode_requests=decode,
                   step_ms=ms, host_submit_ms=1.)
              for start, prefill, decode, ms in [(0., 0, 8, 100.), (1., 0, 8, 20.),
                  (2., 128, 7, 30.), (3., 512, 0, 40.), (4., 0, 8, 200.)]]
    (capture/'model_step_latency.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in timing) + '{')
    restores = [dict(started=2., finished=2.5, host_prepare_seconds=.1,
                     uploaded_bytes=123, model_bytes=45, public_peer_count=7, cancelled=False)]
    (capture/'async_restore.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in restores))
    result = qualification.collect_diagnostics(tmp_path, started=1., finished=4.)
    assert result['model_step_latency']['samples'] == 3
    assert result['model_step_latency']['incomplete_lines'] == 1
    assert result['model_step_latency']['all']['step_ms_p50'] == 30.
    assert result['model_step_latency']['decode_only']['step_ms_p95'] == 20.
    assert result['model_step_latency']['mixed']['step_ms_p95'] == 30.
    assert result['model_step_latency']['prefill_only']['step_ms_p95'] == 40.
    assert result['async_restore']['duration_seconds_p95'] == .5
    assert result['async_restore']['uploaded_bytes'] == 123
    assert result['async_restore']['model_bytes'] == 45
    assert result['async_restore']['with_public_peers'] == 1
    assert 'output_tokens_per_second' not in result


def test_gpu_idle_wait_requires_no_foreign_pid_even_at_zero_utilization(tmp_path,monkeypatch):
    samples=iter([dict(pids=[42],memory_mib=20,utilization=0),
                  dict(pids=[],memory_mib=20,utilization=0),
                  dict(pids=[],memory_mib=20,utilization=0)])
    sleeps=[]
    monkeypatch.setattr(qualification,'gpu_state',lambda:next(samples))
    async def sleep(_): sleeps.append(1)
    async def to_thread(fn): return fn()
    monkeypatch.setattr(qualification,'asyncio',SimpleNamespace(sleep=sleep,to_thread=to_thread))
    # Isolate the clock from asyncio's own event-loop clock.
    monkeypatch.setattr(qualification,'time',SimpleNamespace(monotonic=lambda:10.+120.*len(sleeps),time=lambda:0.))
    asyncio.run(qualification.wait_idle(tmp_path,120))
    assert len(sleeps) == 2
