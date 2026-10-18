"""CPU-only coverage/provenance probes: never spawn a server or call CUDA."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import time

import pytest

from test_benchmark_rolling_agents import fixture, profile as original_profile, qualification

spec = importlib.util.spec_from_file_location('capture_reuse_qualification',
    Path(__file__).parents[1]/'scripts/qualify_capture_reuse.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def profile():
    value = probe.natural.profiles(original_profile(), 18546)[probe.ARM]
    value['kvmem']['async_cpu_scatter'] = True
    return probe.profile(value, 18546)


def byte_qualification(tmp_path, completed=False):
    path, hashes = qualification(tmp_path, True, completed)
    p = probe.cap.read(path/'profile.json')
    p['kvmem'].update(early_capture_slot_release=True,
        early_capture_feedback_bytes=probe.FEEDBACK_BYTES, split_capture_dma=True, async_cpu_scatter=True)
    probe.cap.save(path/'profile.json', p)
    m = probe.cap.read(path/'manifest.json')
    m['profile_sha256'] = probe.cap.sha(path/'profile.json')
    probe.cap.save(path/'manifest.json', m)
    return path, hashes


def test_profile_keeps_original_natural_controls_and_budget_does_not_crop_fixture():
    p = profile()
    source = copy.deepcopy(p)
    data = fixture()
    before = copy.deepcopy(data)
    budget = probe.fixture_budget(data, p)
    assert data == before and p == source
    assert budget['history_limits'] == [262144]*8
    assert budget['required_available_bytes'] == 113834459136
    assert p['max_num_seqs'] == 8 and p['kvmem']['early_capture_feedback_bytes'] == 268435456
    assert p['kvmem']['select_budget'] == 65536 and p['kvmem']['gen_budget'] == 32768
    assert probe.natural.SAMPLING['temperature'] == .6 and not probe.natural.SAMPLING['ignore_eos']


@pytest.mark.parametrize('change', [dict(max_num_seqs=7), dict(draft_tokens=3), dict(max_model_len=300000)])
def test_qualification_rejects_changed_native_deployment(change):
    p = profile()
    p.update(change)
    with pytest.raises(ValueError, match='eight-slot'):
        probe.profile(p, 18546)


def test_pending_byte_qualification_can_be_identified_but_cannot_authorize_run(tmp_path):
    path, hashes = byte_qualification(tmp_path)
    identity = probe.byte_identity(path, hashes, {'version':'test'})
    m = dict(byte_qualification=identity, runtime_sha256=hashes, native_source={'version':'test'})
    with pytest.raises(ValueError, match='has not completed'):
        probe.completed_byte_proof(m)
    probe.cap.save(path/'status.json', dict(status='completed_pending_review'))
    assert probe.completed_byte_proof(m)['evidence']['oracle']['cold_restore_required']
    (path/'runtime/runtime.py').write_text('different implementation')
    with pytest.raises(ValueError, match='runtime changed'):
        probe.completed_byte_proof(m)


def test_unenabled_byte_qualification_or_missing_gdn_proof_cannot_authorize(tmp_path):
    path, hashes = byte_qualification(tmp_path, True)
    identity = probe.byte_identity(path, hashes, {'version':'test'})
    m = dict(byte_qualification=identity, runtime_sha256=hashes, native_source={'version':'test'})
    (path/'server/capture/verified_state_restores.jsonl').write_text('')
    with pytest.raises(ValueError, match='Each continuation'):
        probe.completed_byte_proof(m)
    p = probe.cap.read(path/'profile.json')
    p['kvmem']['early_capture_slot_release'] = False
    probe.cap.save(path/'profile.json', p)
    manifest = probe.cap.read(path/'manifest.json')
    manifest['profile_sha256'] = probe.cap.sha(path/'profile.json')
    probe.cap.save(path/'manifest.json', manifest)
    with pytest.raises(ValueError, match='did not enable'):
        probe.byte_identity(path, hashes, {'version':'test'})


def test_run_checks_completed_byte_evidence_before_idle_wait_or_gpu_launch(tmp_path, monkeypatch):
    probe.cap.save(tmp_path/'status.json', dict(status='prepared'))
    monkeypatch.setattr(probe, 'verify', lambda root:{})
    failure = ValueError('byte qualification pending')
    def reject(m):
        raise failure
    monkeypatch.setattr(probe, 'completed_byte_proof', reject)
    async def forbidden(*args):
        pytest.fail('No GPU wait or launch is permitted before byte qualification')
    monkeypatch.setattr(probe, 'run_server', forbidden)
    with pytest.raises(ValueError) as error:
        asyncio.run(probe.run(NS(output=tmp_path)))
    assert error.value is failure
    assert probe.cap.read(tmp_path/'status.json')['status'] == 'prepared'


@pytest.mark.parametrize('change,match', [
    ('profile_source', 'Original natural profile'), ('fixture_source', 'Original natural fixture'),
    ('derived_profile', 'original natural deployment'), ('provenance', 'Original provenance')])
def test_run_verify_rechecks_original_sources_exact_derivation_and_provenance(tmp_path, monkeypatch, change, match):
    root = tmp_path/'probe'
    frozen, runtime = root/'frozen', root/'runtime'
    frozen.mkdir(parents=True)
    runtime.mkdir()
    (frozen/'entry.py').write_text('frozen entry')
    (runtime/'entry.py').write_text('qualified runtime')
    p, data = profile(), fixture()
    source_profile, source_fixture = tmp_path/'original-profile.json', tmp_path/'original-fixture.json'
    for path in (source_profile, root/'profile.json'):
        probe.cap.save(path, p)
    for path in (source_fixture, root/'fixture.json'):
        probe.cap.save(path, data)
    manifest = dict(kind=probe.KIND, source_sha256=probe.cap.hashes(frozen),
        runtime_sha256=probe.cap.hashes(runtime), profile_sha256=probe.cap.sha(root/'profile.json'),
        fixture_sha256=probe.cap.sha(root/'fixture.json'), profile_source=str(source_profile),
        fixture_source=str(source_fixture), profile_source_sha256=probe.cap.sha(source_profile),
        fixture_source_sha256=probe.cap.sha(source_fixture), native_source={'test':True}, port=18546,
        budget=probe.fixture_budget(data,p), workload=dict(concurrency=8,complete_rounds=2,
            generation_limit=32000,generation_mode='natural',sampling=probe.natural.SAMPLING))
    probe.cap.save(root/'manifest.json', manifest)
    monkeypatch.setattr(probe, 'ROOT', frozen)
    monkeypatch.setattr(probe.cap, 'native_source_identity', lambda:{'test':True})
    checked = []
    monkeypatch.setattr(probe.natural, 'verify_provenance', lambda f:checked.append(True))
    assert probe.verify(root) == manifest and checked
    if change in ('profile_source', 'fixture_source'):
        (source_profile if change == 'profile_source' else source_fixture).write_text('{}')
    elif change == 'derived_profile':
        p['kvmem']['action_interval'] = 99
        probe.cap.save(root/'profile.json', p)
        manifest['profile_sha256'] = probe.cap.sha(root/'profile.json')
        probe.cap.save(root/'manifest.json', manifest)
    else:
        def reject(f):
            raise ValueError('Original provenance changed')
        monkeypatch.setattr(probe.natural, 'verify_provenance', reject)
    with pytest.raises(ValueError, match=match):
        probe.verify(root)


def reuse_records():
    rows = [dict(kind='workload', request_id=name, response_id=name) for name in ('a', 'b', 'c')]
    def event(stage, at, generation=1, ids=('a-0-12345678','b-0-12345678'), **kw):
        return dict(stage=stage, at=at, lane='bulk', slot=0, generation=generation, request_ids=list(ids), **kw)
    events = [event('source_submitted', 1, pending_previous_generations=[]),
        event('source_copied', 2, copied_indices=[0]), event('source_copied', 3, copied_indices=[1]),
        event('source_released', 4, semantic_pending=2, feedback_bytes=128),
        event('source_submitted', 5, 2, ('c-0-12345678',), pending_previous_generations=[1]),
        event('semantic_complete', 6)]
    samples = [dict(at=5., feedback=dict(detached_packets=1, packet_limit=2, detached_owners=2,
        owner_limit=8, detached_bytes=128, byte_limit=probe.FEEDBACK_BYTES), publication_metrics={
            'main/bulk':dict(source_releases_before_semantic=1, shared_source_reuse_before_semantic_packets=1)})]
    final = copy.deepcopy(samples[0])
    final['at'] = 7.
    final['feedback'].update(detached_packets=0, detached_owners=0, detached_bytes=0)
    samples.append(final)
    return events, samples, rows


def test_shared_physical_reuse_requires_source_copy_then_release_then_new_generation_then_semantics():
    events, samples, rows = reuse_records()
    evidence = probe.reuse_evidence(events, samples, rows)
    assert evidence['covered'] and len(evidence['shared_slot_reuse_proofs']) == 1
    assert not probe.reuse_evidence([], [], rows)['covered']
    events[4]['pending_previous_generations'] = []
    assert not probe.reuse_evidence(events, samples, rows)['covered']


@pytest.mark.parametrize('mutation,match', [
    (lambda e,s,r:e[2].update(copied_indices=[]), 'before complete copy'),
    (lambda e,s,r:e[-1].update(at=4.5), 'after semantic completion'),
    (lambda e,s,r:e[4].update(generation=1), 'duplicate source'),
    (lambda e,s,r:s[0]['feedback'].update(detached_bytes=probe.FEEDBACK_BYTES+1), 'credit bound')])
def test_invalid_reuse_or_unbounded_credit_is_a_failure(mutation, match):
    events, samples, rows = reuse_records()
    mutation(events, samples, rows)
    with pytest.raises(ValueError, match=match):
        probe.reuse_evidence(events, samples, rows)


def test_cleanup_only_reuse_cannot_qualify_real_agent_work():
    events, samples, rows = reuse_records()
    rows[0]['kind'] = 'cleanup'
    assert not probe.reuse_evidence(events, samples, rows)['covered']


def test_nonzero_or_stale_final_feedback_balance_cannot_pass():
    events, samples, rows = reuse_records()
    with pytest.raises(ValueError, match='stale or not released'):
        probe.reuse_evidence(events, samples, rows, not_before=8.)
    samples[-1]['feedback'].update(detached_packets=1, detached_owners=1, detached_bytes=128)
    with pytest.raises(ValueError, match='stale or not released'):
        probe.reuse_evidence(events, samples, rows)


def fake_workload(tmp_path, monkeypatch, *, long_first=False, covered=False):
    data = fixture()
    if long_first:
        data['agent_plans'][0]['initial_tokens'] = [1]*230000
    capture = tmp_path/'capture'
    capture.mkdir()
    requests, publications = [], []
    async def stream(client, p, tokens, session, count, recorder, **kw):
        ids = [900+kw['agent']]*(1 if kw['kind'] == 'cleanup' else
            32000 if long_first and kw['agent'] == 0 else 2+kw['agent'])
        begin = time.monotonic()
        row = dict(**kw, begin=begin, end=begin, session=session, input_tokens=len(tokens),
            input_sha256=probe.work.digest(tokens), input_ids=list(tokens), output_ids=ids,
            output_sha256=probe.work.digest(ids), timeline=[[begin,len(ids)]],
            usage=dict(completion_tokens=len(ids)), status='completed', response_id=kw['request_id'],
            declared_generation_ceiling=count, finish_reason='stop')
        requests.append(row)
        recorder.rows.append(row)
        recorder.event('stream_chunk', request_id=row['request_id'], token_ids=ids,
            response_id=row['response_id'], accepted_tokens=len(ids), output_offset=len(ids), usage=row['usage'])
        recorder.event('request_finished', request_id=row['request_id'], status='completed', accepted_tokens=len(ids))
        publications.append(dict(request_id=row['request_id']+'-0-12345678',
            history_end=len(tokens)+len(ids)-1, checkpoint_tokens=len(tokens)+len(ids)-1, checkpoint_kind='M'))
        probe.cap.save(capture/'archive.json', dict(completed=publications))
        probe.cap.save(capture/'progress.json', dict(publication_observed_at=time.monotonic(),
            early_capture_feedback=dict(detached_packets=0, detached_owners=0, detached_bytes=0)))
        await asyncio.sleep(0)
        return row
    async def retirement(*args, **kw):
        return dict(expected_views=kw['expected_views'])
    monkeypatch.setattr(probe.work, 'stream_call', stream)
    monkeypatch.setattr(probe.work, 'wait_retirement', retirement)
    if covered:
        monkeypatch.setattr(probe, 'reuse_evidence', lambda *args,**kw:dict(covered=True))
    result = asyncio.run(probe.workload(None, profile(), data, tmp_path/'workload', capture))
    return result, requests, data


@pytest.mark.parametrize('covered', [False, True])
def test_two_natural_rounds_keep_own_ids_tools_query_and_normal_cleanup(tmp_path, monkeypatch, covered):
    result, rows, data = fake_workload(tmp_path, monkeypatch, covered=covered)
    assert result['status'] == ('passed' if covered else 'coverage_insufficient')
    assert result['complete_rounds'] and result['cleanup_completed'] and len(rows) == 24
    assert len(result['output_proofs']) == len(result['history_proofs']) == 24
    assert result['total_stream_output_tokens'] == 96
    for user in range(8):
        turns = [r for r in rows if r['agent'] == user]
        assert turns[0]['input_ids'] == data['agent_plans'][user]['initial_tokens']
        assert turns[1]['input_ids'] == turns[0]['input_ids']+turns[0]['output_ids']+data['agent_plans'][user]['turns'][1]['tool_tokens']
        assert turns[0]['declared_generation_ceiling'] == turns[1]['declared_generation_ceiling'] == 32000
        assert turns[-1]['input_ids'] == [0] and turns[-1]['kind'] == 'cleanup'


def test_context_capacity_does_not_clip_reduce_limits_or_cancel_other_agents(tmp_path, monkeypatch):
    result, rows, data = fake_workload(tmp_path, monkeypatch, long_first=True, covered=True)
    assert result['status'] == 'coverage_insufficient' and not result['complete_rounds']
    assert result['cleanup_completed'] and len(rows) == 23
    assert result['context_capacity_shortfalls'][0]['agent'] == 0
    assert sum(r['kind'] == 'workload' for r in rows if r['agent'] != 0) == 14
    assert len(data['agent_plans'][0]['initial_tokens']) == 230000
    assert all(r['declared_generation_ceiling'] == 32000 for r in rows if r['kind'] == 'workload')


@pytest.mark.parametrize('field,value', [('usage',dict(completion_tokens=999)), ('output_sha256','bad')])
def test_token_usage_or_identity_mismatch_cannot_pass(tmp_path, monkeypatch, field, value):
    _, rows, data = fake_workload(tmp_path, monkeypatch)
    rows[0][field] = value
    with pytest.raises(ValueError, match='Stream/token IDs/usage'):
        probe.output_evidence(rows, probe.ready.records(tmp_path/'workload/events.jsonl'), data)


@pytest.mark.parametrize('failure', ['foreign_gpu', 'memory_floor', 'history_contract'])
def test_guard_failure_stops_only_owned_server_and_preserves_failure(tmp_path, monkeypatch, failure):
    manifest = dict(idle_seconds=0, budget=dict(required_available_bytes=1, minimum_free_bytes=1))
    monkeypatch.setattr(probe, 'verify', lambda root:manifest)
    monkeypatch.setattr(probe, 'completed_byte_proof', lambda m:{})
    async def immediate(fn, *args, **kw):
        return fn(*args, **kw)
    monkeypatch.setattr(probe.asyncio, 'to_thread', immediate)
    async def idle(*args):
        pass
    monkeypatch.setattr(probe.ready.bench.q, 'wait_idle', idle)
    calls = 0
    def gpu():
        nonlocal calls
        calls += 1
        return dict(pids=[999] if calls > 1 and failure == 'foreign_gpu' else [], utilization=0, memory_mib=0)
    monkeypatch.setattr(probe.ready.bench.q, 'gpu_state', gpu)
    memory_calls = 0
    def memory():
        nonlocal memory_calls
        memory_calls += 1
        return dict(MemAvailable=0 if memory_calls > 1 and failure == 'memory_floor' else 100)
    monkeypatch.setattr(probe.cap, 'host_memory', memory)
    server = NS(pid=123, poll=lambda:None)
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *args, **kw:server)
    monkeypatch.setattr(probe.ready.bench.q, 'owned_pids', lambda value:{123})
    stopped = []
    monkeypatch.setattr(probe.ready.bench.q, 'stop_owned', stopped.append)
    monkeypatch.setitem(sys.modules, 'psutil', NS(Process=lambda *args:NS(create_time=lambda:1.)))
    class Client:
        def __init__(self, *args, **kw):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get(self, *args, **kw):
            await asyncio.Event().wait()
    monkeypatch.setitem(sys.modules, 'httpx', NS(AsyncClient=Client, Timeout=lambda *a,**k:None,
        Limits=lambda **k:None, HTTPError=OSError))
    probe.cap.save(tmp_path/'profile.json', dict(port=18546))
    if failure == 'history_contract':
        path = tmp_path/'server/capture'
        path.mkdir(parents=True)
        probe.cap.save(path/'archive.json', dict(archive_backend='ssd'))
    with pytest.raises(ValueError, match={'foreign_gpu':'Foreign GPU', 'memory_floor':'below8GiB',
        'history_contract':'CPU-only history'}[failure]):
        asyncio.run(probe.run_server(tmp_path, manifest))
    assert stopped == [server]
    assert probe.cap.read(tmp_path/'timing.json')['elapsed_seconds'] >= 0
    assert not (tmp_path/'results.json').exists()


@pytest.mark.parametrize('failure', ['runtime', 'byte_qualification'])
def test_waiting_for_gpu_cannot_bypass_changed_identity_or_byte_proof(tmp_path, monkeypatch, failure):
    waited = []
    async def idle(*args):
        waited.append(True)
    monkeypatch.setattr(probe.ready.bench.q, 'wait_idle', idle)
    manifest = dict(idle_seconds=0)
    original = ValueError('Changed '+failure)
    def identity(root):
        assert waited
        if failure == 'runtime':
            raise original
        return manifest
    def byte(m):
        assert waited
        raise original
    monkeypatch.setattr(probe, 'verify', identity)
    monkeypatch.setattr(probe, 'completed_byte_proof', byte)
    monkeypatch.setattr(probe.subprocess, 'Popen', lambda *a,**k:pytest.fail('Changed input must not launch'))
    with pytest.raises(ValueError) as error:
        asyncio.run(probe.run_server(tmp_path, manifest))
    assert error.value is original and waited


def test_feedback_cleanup_requires_fresh_zero_balance(tmp_path):
    probe.cap.save(tmp_path/'progress.json', dict(publication_observed_at=10.,
        early_capture_feedback=dict(detached_packets=1, detached_owners=1, detached_bytes=128)))
    with pytest.raises(TimeoutError, match='fresh zero balance'):
        asyncio.run(probe.wait_feedback_released(tmp_path, not_before=9., timeout=0))
    probe.cap.save(tmp_path/'progress.json', dict(publication_observed_at=10.,
        early_capture_feedback=dict(detached_packets=0, detached_owners=0, detached_bytes=0)))
    assert asyncio.run(probe.wait_feedback_released(tmp_path, not_before=9., timeout=0))
