"""CPU-only protocol tests; no model/server/native extension import."""
import asyncio
import array
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location('capacity_workload_test',
    Path(__file__).resolve().parents[1]/'scripts/capacity_scaling_workload.py')
work = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(work)


def test_crop_preserves_prefix_query_and_assistant_suffix_without_gold():
    original = list(range(24))
    tokens, span, plan = work.crop_preserving_query(original, 20, 22, 16)
    assert tokens == original[:12]+original[20:]
    assert tokens[span[0]:span[1]] == original[20:22]
    assert tokens[-2:] == original[-2:]
    assert plan['kept_ranges'] == [[0, 12], [20, 24]]
    assert plan['removed_range'] == [12, 20]
    assert plan['removed_tokens'] == 8
    assert plan['selection_uses_gold'] is False
    assert original == list(range(24))


@pytest.mark.parametrize('query,target', [((0, 3), 8), ((9, 9), 8), ((8, 12), 8), ((4, 5), 5), ((8, 9), 11)])
def test_crop_rejects_ambiguous_or_impossible_extent(query, target):
    with pytest.raises(ValueError):
        work.crop_preserving_query(list(range(10)), *query, target)


def test_no_crop_has_identical_token_fingerprint():
    tokens = list(range(12))
    cropped, span, plan = work.crop_preserving_query(tokens, 9, 11, 12)
    assert cropped == tokens and span == [9, 11]
    assert plan['removed_tokens'] == 0
    assert work.token_bytes_sha256(cropped) == work.token_bytes_sha256(tokens)


def test_preparation_pins_exact_source_bytes_and_declared_cohort(monkeypatch, tmp_path):
    monkeypatch.setattr(work, 'INPUT_TOKENS', 8)
    monkeypatch.setattr(work, 'OUTPUT_TOKENS', 2)
    monkeypatch.setattr(work, 'validate_fixture', lambda fixture: None)
    (tmp_path/'inputs').mkdir()
    rows = []
    for seed in (1000, 2000):
        for task in work.TASKS:
            identity = f'{task}-{seed}'
            tokens = list(range(len(rows)*20, len(rows)*20+10))
            data = array.array('i', tokens)
            if work.sys.byteorder != 'little': data.byteswap()
            raw = data.tobytes()
            fingerprint = hashlib.sha256(raw).hexdigest()
            (tmp_path/'inputs'/f'{identity}.i32').write_bytes(raw)
            rows.append(dict(id=identity, task=task, seed=seed, position_percent=10,
                nominal_input_tokens=262144, served_sha256=fingerprint, token_sha256=fingerprint,
                served_input_tokens=10, query_begin=8, query_end=9,
                expected=['gold is deliberately never interpreted']))
    source = tmp_path/'manifest.json'
    source.write_text(json.dumps({'rows': list(reversed(rows))}))
    fixture = work.prepare_fixture(source, context_tokens=8, output_tokens=2, eos_token_id=7)
    assert [r['id'] for r in fixture['samples']] == [r['id'] for r in rows]
    assert [r['source']['task'] for r in fixture['samples'][:4]] == list(work.TASKS)
    assert fixture['source_manifest_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert all(r['crop']['removed_range'] == [6, 8] for r in fixture['samples'])
    assert all('expected' not in r for r in fixture['samples'])
    first = tmp_path/'inputs'/f"{rows[0]['id']}.i32"
    first.write_bytes(first.read_bytes()+b'\x00\x00\x00\x00')
    with pytest.raises(ValueError, match='differs from its manifest'):
        work.prepare_fixture(source, context_tokens=8, output_tokens=2, eos_token_id=7)


def test_decode_window_uses_last_token_not_http_close_and_counts_boundaries_once():
    rows = [dict(request_id='a', timeline=[[1., 1], [2., 2], [3., 3]], end=20.),
            dict(request_id='b', timeline=[[2., 1], [3., 2], [4., 3]], end=30.)]
    result = work.decode_summary(rows, 2)
    assert result['seconds'] == result['all_offered_common_seconds'] == 1.
    assert result['accepted_output_tokens'] == 5
    assert result['tokens_per_second'] == 5
    assert result['intervals'] == [dict(begin=2., end=3., request_ids=['a', 'b'])]


def test_offered_queue_is_not_claimed_as_common_eight_decode():
    rows = [dict(request_id=str(i), timeline=[[float(i//4*10), 1], [float(i//4*10+5), 4]])
            for i in range(8)]
    report = work.decode_summary(rows, 4)
    assert report['seconds'] == 10.
    assert report['all_offered_common_seconds'] == 0.
    assert work.decode_summary(rows, 8)['seconds'] == 0.
    evidence = work.gpu_decode_evidence([
        dict(started_at=2., decode_requests=4, prefill_tokens=0),
        dict(started_at=12., decode_requests=4, prefill_tokens=128)], report)
    assert evidence['passed'] and evidence['observed_steps'] == 1
    assert not work.gpu_decode_evidence([], report)['passed']


def evidence_fixture():
    row = dict(request_id='client', response_id='cmpl-one', session='user', agent=0,
               input_tokens=229376, output_ids=[1]*8192)
    rid, history = 'cmpl-one-0-abcdef12', 237567
    fixture = dict(samples=[dict(query_begin=229351, query_end=229369)])
    profile = dict(kvmem=dict(select_budget=65536, gen_budget=32768, block_tokens=32))
    state = dict(session_views=[dict(namespace=['user', 'user'], history=history, window=73751)])
    geometry = dict(cpu_archive_bytes_per_token=34816, historical_tokens=262144,
                    archive_codec='raw_fp8_e4m3fn_v_fp8_e4m3fn_v1')
    archive = dict(completed=[dict(request_id=rid, history_end=history, checkpoint_tokens=history,
                                   checkpoint_kind='M')], archive_backend='cpu', repository_gc=False,
                   cpu_authority_bytes=(history//32)*32*34816, disk_read_bytes=0, disk_write_bytes=0,
                   pending_downloads=0, session_retiring_requests=0)
    refresh = [dict(request_id=rid, selection='query', query_span=[229351, 229369],
                    selected_tokens=65536, history_begin=229344, mean_k_backend='qw3_cuda',
                    action_attention_enabled=True, cancelled=False)]
    return row, fixture, profile, state, archive, geometry, refresh


def test_full_cpu_authority_and_smaller_active_both_required():
    row, fixture, profile, state, archive, geometry, refresh = evidence_fixture()
    report = work.compression_evidence([row], fixture, profile, state, archive, geometry, refresh)
    assert report['passed']
    assert report['requests'][0]['expected_kv_history'] == 237567
    assert report['requests'][0]['accepted_token_ledger'] == 237568
    # Actual active includes Query/new generation and may exceed selected K.
    assert state['session_views'][0]['window'] > 65536
    state['session_views'][0]['window'] = 237567
    assert not work.compression_evidence([row], fixture, profile, state, archive, geometry, refresh)['passed']


@pytest.mark.parametrize('mutation', ['missing_cpu', 'short_cpu', 'gc', 'disk', 'wrong_identity', 'missing_query', 'wrong_history', 'wrong_backend'])
def test_compression_evidence_fails_closed(mutation):
    row, fixture, profile, state, archive, geometry, refresh = evidence_fixture()
    if mutation == 'missing_cpu': archive.pop('cpu_authority_bytes')
    elif mutation == 'short_cpu': archive['cpu_authority_bytes'] -= 1
    elif mutation == 'gc': archive['repository_gc'] = True
    elif mutation == 'disk': archive['disk_read_bytes'] = 1
    elif mutation == 'wrong_identity': archive['completed'][0]['request_id'] += '-extra'
    elif mutation == 'missing_query': refresh.clear()
    elif mutation == 'wrong_history': archive['completed'][0]['history_end'] += 1
    elif mutation == 'wrong_backend': refresh[0]['mean_k_backend'] = 'bm25'
    assert not work.compression_evidence([row], fixture, profile, state, archive, geometry, refresh)['passed']


def test_partial_jsonl_tail_is_not_success_evidence(tmp_path):
    path = tmp_path/'evidence.jsonl'
    path.write_text('{"a":1}\n{"a":')
    assert work.read_records(path) == [{'a': 1}]
    path.write_text('{"a":1}\n{"a":\n')
    with pytest.raises(json.JSONDecodeError): work.read_records(path)


def test_native_fixed_work_drains_all_queued_requests(monkeypatch, tmp_path):
    # Small fixture keeps this lifecycle regression CPU-only and fast; separate
    # protocol tests cover fixed production lengths/crop identity.
    monkeypatch.setattr(work, 'validate_fixture', lambda fixture: None)
    monkeypatch.setattr(work, 'INPUT_TOKENS', 4)
    monkeypatch.setattr(work, 'OUTPUT_TOKENS', 3)
    fixture = dict(samples=[dict(id=str(i), input_ids=[i]*4, query_begin=2, query_end=3,
        source={'sha256':str(i)}, input_bytes_sha256=str(i)) for i in range(4)])
    active, peak, finished = 0, 0, []

    async def fake_stream(client, profile, tokens, session, count, recorder, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        began = work.time.perf_counter()
        await asyncio.sleep(.001)
        first = work.time.perf_counter()
        await asyncio.sleep(.001)
        end = work.time.perf_counter()
        row = dict(request_id=kwargs['request_id'], response_id='response', session=session,
            agent=kwargs['agent'], begin=began, end=end, kind='workload', output_ids=[1]*count,
            timeline=[[first, 1], [end, count-1]], status='completed', usage={'prompt_tokens':4},
            input_tokens=4)
        recorder.rows.append(row)
        finished.append(kwargs['agent'])
        active -= 1
        return row

    async def quiescent(**kwargs):
        assert len(finished) == 4 and active == 0
        return {'passed': True}

    monkeypatch.setattr(work.timed, 'stream_call', fake_stream)
    args = SimpleNamespace(offered_concurrency=4, admitted_concurrency=2, output_tokens=3,
                           run_tag='test', native_quiescent=quiescent)
    result = asyncio.run(work.workload(None, dict(native_vllm=True, max_model_len=262144), fixture, tmp_path, args))
    assert peak == 2 and sorted(finished) == [0, 1, 2, 3]
    assert result['drained'] and result['cleanup_complete']
    assert result['fixed_work_output_tokens'] == result['total_stream_output_tokens'] == 12
    assert result['cancelled_requests'] == []
    assert result['gpu_decode_concurrency']['passed'] is None
    assert len(result['ingress_queue']) == 4


def test_fixture_protocol_refuses_small_synthetic_or_replicated_input():
    with pytest.raises(ValueError): work.validate_fixture({'format':'capacity_scaling_v1'})
    with pytest.raises(ValueError): work.prepare_fixture(Path('unused'), context_tokens=128)


def test_warmup_requires_history_above_k_and_nonempty_actual_query_score_path():
    row = dict(response_id='cmpl-warm')
    profile = dict(kvmem=dict(select_budget=65536, block_tokens=32, sink_blocks=41, recent_blocks=0))
    query = dict(request_id='cmpl-warm-0-abcdef12', selection='query', query_span=[67456, 67584],
                 history_begin=67456, selected_tokens=65536, query_score_tokens=128,
                 mean_k_backend='qw3_cuda', mandatory_fit={'hard_blocks':list(range(41))},
                 refresh_started_at=1., refresh_finished_at=2., cancelled=False)
    proof = work.warmup_selection_evidence(row, profile, [query], [67456, 67584])
    assert proof['passed'] and proof['eligible_complete_pages'] == 2067
    # A backend configuration label by itself cannot establish extension use.
    query['query_score_tokens'] = 0
    assert not work.warmup_selection_evidence(row, profile, [query], [67456, 67584])['passed']
    query['query_score_tokens'] = 128
    query['history_begin'] = 65536
    assert not work.warmup_selection_evidence(row, profile, [query], [67456, 67584])['passed']
    query['history_begin'] = 67456
    query['mandatory_fit']['hard_blocks'] = list(range(2108))
    assert not work.warmup_selection_evidence(row, profile, [query], [67456, 67584])['passed']


@pytest.mark.parametrize('native', [False, True])
def test_shared_stream_payload_uses_same_ids_without_native_kvmem_keys(native, tmp_path):
    payloads = []
    class Response:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def raise_for_status(self): pass
        async def aiter_lines(self):
            yield 'data: '+json.dumps(dict(id='cmpl-fixed', choices=[dict(index=0, token_ids=[5])],
                usage=dict(prompt_tokens=3, completion_tokens=1)))
            yield 'data: [DONE]'
    class Client:
        def stream(self, method, url, json):
            payloads.append(json)
            return Response()
    recorder = work.timed.Recorder(tmp_path, 1)
    try:
        row = asyncio.run(work.timed.stream_call(Client(),
            dict(native_vllm=native, served_model_name='test', port=1), [1, 2, 3], 'isolated-user', 1,
            recorder, agent=0, epoch=0, turn=0, kind='workload', request_id='fixed', new_input_tokens=3,
            query_plan={'kvmem_query_span':[1, 2], 'kvmem_reselect':'force'}))
    finally:
        recorder.stream.close()
    assert row['output_ids'] == [5] and payloads[0]['prompt'] == [1, 2, 3]
    assert payloads[0]['ignore_eos'] is True
    if native:
        assert not any(key.startswith('kvmem_') for key in payloads[0])
        assert 'session_id' not in payloads[0]
        assert payloads[0]['cache_salt'] == 'isolated-user:epoch:0'
    else:
        assert payloads[0]['kvmem_query_span'] == [1, 2]
        assert payloads[0]['kvmem_reselect'] == 'force'
        assert payloads[0]['session_id'] == 'isolated-user'
