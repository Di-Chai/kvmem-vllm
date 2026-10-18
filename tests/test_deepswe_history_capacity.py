import asyncio
import importlib.util
from pathlib import Path
import time
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'scripts'/filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


work = load('history_capacity_test_work', 'deepswe_history_capacity_workload.py')
probe = load('history_capacity_test_probe', 'benchmark_capacity_scaling.py')


def test_sampling_proxy_keeps_transport_and_quota_identical():
    class Client:
        def stream(self, method, url, **kwargs):
            return method, url, kwargs['json']
    original = dict(model='model', max_tokens=8192, ignore_eos=True, request_id='id', temperature=0)
    _, _, sent = work.SampledClient(Client()).stream('POST', '/v1/completions', json=original)
    assert sent == dict(original, **work.SAMPLING)
    assert original['temperature'] == 0 and sent['max_tokens'] == 8192 and sent['ignore_eos'] is True


def test_native_preemptions_are_measured_without_rejecting_the_baseline():
    before = dict(started_at=0, values={'vllm:generation_tokens_total': 1, 'vllm:num_preemptions_total': 2})
    after = dict(started_at=1, values={'vllm:generation_tokens_total': 9, 'vllm:num_preemptions_total': 5})
    async def snapshot(): return after
    result = asyncio.run(probe.reconcile_outputs(snapshot, before, 8,
        allow_preemptions=True, include_snapshot=True))
    assert result['preemptions'] == 3 and result['last_snapshot'] == after
    with pytest.raises(ValueError, match='preempted'):
        asyncio.run(probe.reconcile_outputs(snapshot, before, 8))


@pytest.mark.parametrize('wrong_prompt_usage', [False, True])
def test_prime_barrier_and_fixed_continuation_despite_different_prime_predictions(tmp_path, monkeypatch, wrong_prompt_usage):
    monkeypatch.setattr(work, 'validate_fixture', lambda fixture: None)
    monkeypatch.setattr(work, 'OUTPUT_TOKENS', 4)
    values = dict(generated=0, prefill=0, preemptions=0, primed=0)
    calls = []
    async def snapshot():
        at = time.perf_counter()
        return dict(started_at=at, finished_at=at,
            values={'vllm:generation_tokens_total': values['generated'],
                    'vllm:num_preemptions_total': values['preemptions']},
            raw_metrics=f"vllm:prompt_tokens_total {values['prefill']}\n"
                        'vllm:prompt_tokens_cached_total 0\n')
    async def quiescent(**kwargs): return dict(native_metrics=await snapshot())
    async def stream(client, profile, tokens, session, count, recorder, **kwargs):
        phase = kwargs['kind']
        if phase == 'resume':
            assert values['primed'] == 8
            assert tokens == [1, 2, 17]  # Recorded17, never the prime prediction999.
        calls.append(phase)
        await asyncio.sleep(0)
        if phase == 'prime': values['primed'] += 1
        if kwargs['agent'] == 0: values['preemptions'] += 1
        values['generated'] += count; values['prefill'] += len(tokens)
        at = time.perf_counter()
        row = dict(begin=at, end=at+.01, timeline=[(at, 1), (at+.01, count-1)] if count>1 else [(at,1)],
            output_ids=[999]*count, input_tokens=len(tokens), session=session,
            request_id=kwargs['request_id'], kind=phase, status='completed',
            usage={'prompt_tokens':len(tokens)-int(wrong_prompt_usage)})
        recorder.rows.append(row)
        return row
    monkeypatch.setattr(work.timed, 'stream_call', stream)
    fixture = dict(eos=0, samples=[dict(id=str(i), input_ids=[1,2], continuation_token_id=17,
                                     request_plan={}, sha256='test') for i in range(8)])
    args = SimpleNamespace(offered_concurrency=8, admitted_concurrency=8,
        capture_directory=tmp_path/'unused', run_tag='native', native_quiescent=quiescent,
        snapshot=snapshot, reconcile_outputs=probe.reconcile_outputs)
    if wrong_prompt_usage:
        with pytest.raises(ValueError, match='complete recorded input'):
            asyncio.run(work.workload(None, {'native_vllm':True}, fixture, tmp_path/'workload', args))
        assert 'resume' not in calls
        return
    result = asyncio.run(work.workload(None, {'native_vllm':True}, fixture, tmp_path/'workload', args))
    assert calls == ['prime']*8+['resume']*8
    assert result['prime_counter_reconciliation']['preemptions'] == 1
    assert result['resume_counter_reconciliation']['preemptions'] == 1
    assert result['computed_public_prefill']['cold_prime_tokens'] == 16
    assert result['computed_public_prefill']['resume_tokens'] == 24
    assert result['total_stream_output_tokens'] == 40 and result['fixed_work_output_tokens'] == 32


def test_real_history_rejects_an_external_native_five_user_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(work, 'validate_fixture', lambda fixture: None)
    args = SimpleNamespace(offered_concurrency=8, admitted_concurrency=5)
    with pytest.raises(ValueError, match='no external5-user gate'):
        asyncio.run(work.workload(None, {'native_vllm':True}, {}, tmp_path, args))


def test_full_history_storage_gate_cannot_pass_with_only_selected_cpu_pages():
    prime = [dict(response_id='prime', request_id='prime')]
    rows = [dict(response_id='resume', request_id='resume', session='user', input_tokens=200000,
                 output_ids=[1]*4)]
    expected = 200003
    resources = dict(session_views=[dict(namespace=['user','user'], history=expected, window=65536)])
    archive = dict(completed=[dict(request_id='resume-0-00000000', history_end=expected,
        checkpoint_tokens=expected, checkpoint_kind='M')], archive_backend='cpu', repository_gc=False,
        disk_read_bytes=0, disk_write_bytes=0, pending_downloads=0, session_retiring_requests=0,
        cpu_authority_bytes=expected*34816)
    geometry = dict(cpu_archive_bytes_per_token=34816, archive_codec='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1',
        select_budget=65536, gen_budget=32768, historical_tokens=262144, cpu_offload_bytes=100_000_000_000)
    refreshes = [dict(request_id='prime-0-00000000', selection='pressure', selected_tokens=65536, history_begin=100000,
                     mean_k_backend='qw3_cuda', action_attention_enabled=True, cancelled=False)]
    profile = dict(kvmem=dict(select_budget=65536, gen_budget=32768))
    assert work.compression_evidence(prime, rows, resources, archive, geometry, refreshes, profile)['passed']
    archive['cpu_authority_bytes'] = 65536*34816
    assert not work.compression_evidence(prime, rows, resources, archive, geometry, refreshes, profile)['passed']
