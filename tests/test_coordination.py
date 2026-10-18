from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.coordination import HistoryAdmission
from vllm_kvmem.session_control import SessionLedger, SessionNotice


def request(rid, tokens=80, session=None, salt='tenant'):
    return SimpleNamespace(request_id=rid, session_id=session or rid, cache_salt=salt,
        num_prompt_tokens=tokens, sampling_params=SimpleNamespace(max_tokens=0))


def retired(req, generation=1, history=72):
    return SessionNotice('retired', (req.cache_salt, req.session_id), generation, req.request_id, history)


def test_shared_credits_allow_eight_long_trajectories_and_reserve_one_full_history():
    gate = HistoryAdmission(72 * 1024**3, 32, 52224, 1048576)
    for i in range(8):
        req = request(str(i), 50000)
        assert gate.allows(req)
        gate.allocated(req, i + 1)
    state = gate.snapshot()
    assert state['active'] == 8 and state['progress_reserve_bytes'] > 48 * 1024**3
    assert state['reserved_bytes'] + state['progress_reserve_bytes'] < 72 * 1024**3


def test_blocked_large_request_does_not_block_a_small_peer_or_reserved_growth():
    gate = HistoryAdmission(208, 4, 1, 100)
    a, b = request('a'), request('b')
    gate.allocated(a, 1); gate.allocated(b, 2)
    assert not gate.allows(request('large', 16))
    assert gate.allows(request('small', 4))
    gate.allocated(request('small', 4), 3)
    assert gate.snapshot()['available_bytes'] == 0
    gate.finished(a)
    assert not gate.allows(request('next', 100, session='a'))
    gate.consume(retired(a))
    assert gate.allows(request('next', 100, session='a'))
    gate.allocated(request('next', 100, session='a'), 1)
    assert gate.snapshot()['reserved_bytes'] == 208


def test_credits_wait_for_exact_generation_and_retirement_fences():
    gate = HistoryAdmission(200, 4, 1, 100)
    req = request('a'); gate.allocated(req, 7)
    gate.finished(req)
    with pytest.raises(ValueError, match='Missing session retirement'):
        gate.released('a')
    with pytest.raises(ValueError, match='Stale'):
        gate.consume(retired(req, 6))
    with pytest.raises(ValueError, match='another request'):
        gate.consume(replace(retired(req, 7), request_id='foreign'))
    assert gate.snapshot()['reserved_bytes'] == 88
    gate.consume(retired(req, 7))
    gate.released('a')  # Native ack may share the same packet.
    assert gate.snapshot()['reserved_bytes'] == 88
    gate.consume(SessionNotice('evicted', ('tenant', 'a'), 7, ''))
    assert gate.snapshot()['reserved_bytes'] == 0
    with pytest.raises(ValueError, match='Stale'):
        gate.consume(SessionNotice('evicted', ('tenant', 'a'), 7, ''))


def test_lease_view_distinguishes_active_retiring_retained_and_released_history():
    gate = HistoryAdmission(200, 4, 1, 100)
    req = request('a')
    gate.allocated(req, 7)
    view = gate.snapshot()['lease_views']
    assert view == [dict(key=['session','tenant','a'],generation=7,
                         request_id='a',phase='active',bytes=88)]
    gate.finished(req)
    assert gate.snapshot()['lease_views'][0]['phase'] == 'retiring'
    gate.consume(retired(req, 7))
    view = gate.snapshot()['lease_views'][0]
    assert view['phase'] == 'idle' and view['request_id'] is None and view['bytes'] == 88
    gate.consume(SessionNotice('evicted', ('tenant','a'), 7, ''))
    assert gate.snapshot()['lease_views'] == []


def test_cancelled_waiter_never_takes_credits_and_stateless_release_is_fenced():
    gate = HistoryAdmission(200, 4, 1, 100)
    for key in ('a', 'b'):
        gate.allocated(request(key), 1)
    waiter = request('waiter')
    assert not gate.allows(waiter)
    gate.finished(waiter)
    assert not gate.waiting and len(gate.leases) == 2
    gate = HistoryAdmission(200, 4, 1, 100)
    plain = request('plain'); plain.session_id = None
    gate.allocated(plain)
    with pytest.raises(ValueError):
        gate.released('plain')
    gate.finished(plain)
    assert gate.snapshot()['retiring'] == 1
    gate.released('plain')
    assert not gate.leases


def test_salts_isolate_histories_and_lookup_checks_do_not_reserve_memory():
    gate = HistoryAdmission(200, 4, 1, 100)
    a = request('a', session='same', salt='a')
    b = request('b', session='same', salt='b')
    for _ in range(3):
        assert gate.allows(a) and not gate.leases
    gate.allocated(a, 1); gate.allocated(b, 2)
    assert len(gate.leases) == 2
    assert not gate.allows(request('c', session='same', salt='a'))


def test_preserved_session_capacity_neither_evicts_nor_runs_uncached():
    ledger = SessionLedger(1, 4, lambda *a: None, lambda *a: None, lambda *a: None,
                           preserve_history=True)
    a = request('a'); a.prompt_token_ids = list(range(80))
    b = request('b'); b.prompt_token_ids = list(range(80))
    assert ledger.lookup(a, 0) == 0
    assert ledger.lookup(b, 0) is None
    assert not ledger.uncached_sessions and not ledger.commands.evictions
    ledger.entries[('tenant', 'a')].phase = 'idle'
    ledger.entries[('tenant', 'a')].request_id = None
    assert ledger.lookup(b, 0) is None
    assert not ledger.commands.evictions


def test_impossible_full_trajectory_fails_at_startup():
    with pytest.raises(ValueError, match='one full trajectory'):
        HistoryAdmission(99, 4, 1, 100)


def test_actual_archive_watermark_cannot_exceed_reserved_bytes():
    gate=HistoryAdmission(200,4,1,100)
    req=request('a'); gate.allocated(req,1)
    gate.observe(80)
    assert gate.snapshot()['last_observed_archive_bytes'] == 80
    assert gate.total_wait_seconds == 0
    with pytest.raises(ValueError,match='exceeded'):
        gate.observe(89)


def test_cpu_geometry_counts_native_mtp_but_not_recurrent_or_sliding_cache():
    torch = pytest.importorskip('torch')
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
    from vllm_kvmem.cache_manifest import archive_bytes_per_token
    spec = FullAttentionSpec(block_size=32, num_kv_heads=4, head_size=256, dtype=torch.uint8)
    groups = [SimpleNamespace(layer_names=list(map(str, range(16))), kv_cache_spec=spec),
              SimpleNamespace(layer_names=['mtp'], kv_cache_spec=UniformTypeKVCacheSpecs(
                  block_size=32, kv_cache_specs={'mtp':spec})),
              SimpleNamespace(layer_names=['draft'], kv_cache_spec=SlidingWindowSpec(
                  block_size=32, num_kv_heads=4, head_size=256, dtype=torch.bfloat16, sliding_window=32))]
    assert archive_bytes_per_token(SimpleNamespace(kv_cache_groups=groups)) == 34816


def test_options_require_cpu_fenced_history_and_validate_maintenance_without_cuda():
    from vllm_kvmem.options import RuntimeOptions
    base = RuntimeOptions()
    with pytest.raises(ValueError, match='CPU history coordination'):
        replace(base, cpu_history_coordination=True)
    with pytest.raises(ValueError, match='asynchronous refresh'):
        replace(base, maintenance_max_steps=4)
    with pytest.raises(ValueError, match='positive and finite'):
        replace(base, maintenance_busy_ms=float('nan'))


def test_native_connector_gates_individual_requests_and_settles_session_before_release():
    from test_connector import pair
    from test_session_control import ledger, request as session_request
    from vllm.v1.outputs import KVConnectorOutput
    from vllm_kvmem.connector import ControlCompletions
    connector, _ = pair()
    connector.async_retirement = True
    connector.history_admission = gate = HistoryAdmission(200, 4, 1, 100)
    connector.sessions, _ = ledger()
    req = session_request()
    assert connector.get_num_new_matched_tokens(req, 0) == (0, False)
    assert not gate.leases  # Native GPU allocation has not succeeded yet.
    connector.update_state_after_alloc(req, None, 0)
    assert gate.snapshot()['active'] == 1
    connector.sessions.capture_before_final_prefill(req, 9)
    commands = connector.sessions.take_commands()
    binding = commands.bindings[0]
    connector.request_finished_all_groups(req, ())
    assert gate.snapshot()['retiring'] == 1
    connector.update_connector_output(KVConnectorOutput(finished_sending={req.request_id},
        kv_connector_worker_meta=ControlCompletions(sessions=[SessionNotice('retired',
            binding.namespace, binding.generation, req.request_id, 16, 12)])))
    assert gate.snapshot()['retiring'] == 0 and gate.snapshot()['leases'] == 1
    entry = connector.sessions.entries[binding.namespace]
    connector.sessions._evict(entry)
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        sessions=[SessionNotice('evicted', binding.namespace, binding.generation, '')])))
    assert not gate.leases


def test_explicit_close_at_full_cpu_budget_reaches_eviction_and_uses_request_retirement():
    from test_connector import pair
    from test_session_control import ledger, request as session_request, close_request
    from vllm.v1.outputs import KVConnectorOutput
    from vllm_kvmem.connector import ControlCompletions
    connector,_=pair();connector.async_retirement=True
    connector.sessions,_=ledger()
    connector.history_admission=gate=HistoryAdmission(108,4,1,100)
    old=session_request()
    assert connector.get_num_new_matched_tokens(old,0)==(0,False)
    connector.update_state_after_alloc(old,None,0)
    connector.sessions.capture_before_final_prefill(old,9)
    binding=connector.sessions.take_commands().bindings[0]
    close=close_request()
    assert gate.snapshot()['available_bytes']==0
    assert gate.can_admit(close)  # No new allocation; same-session serialization still blocks.
    assert connector.get_num_new_matched_tokens(close,0)==(None,False)
    assert not connector.sessions.take_commands().evictions
    assert not gate.can_admit(close_request('absent',session='absent'))
    connector.request_finished_all_groups(old,())
    connector.update_connector_output(KVConnectorOutput(finished_sending={old.request_id},
        kv_connector_worker_meta=ControlCompletions(sessions=[SessionNotice('retired',
            binding.namespace,binding.generation,old.request_id,16,12)])))
    assert connector.get_num_new_matched_tokens(close,0)==(None,False)
    eviction,=connector.sessions.take_commands().evictions
    with pytest.raises(ValueError,match='fenced eviction'):
        gate.allocated(close)
    assert gate.snapshot()['leases']==1
    connector.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=ControlCompletions(
        sessions=[SessionNotice('evicted',eviction.namespace,eviction.generation,'')])))
    assert connector.get_num_new_matched_tokens(close,0)==(0,False)
    connector.update_state_after_alloc(close,None,0)
    assert gate.requests[close.request_id]==('request',close.request_id)
    assert gate.snapshot()['reserved_bytes']==12 and gate.snapshot()['progress_reserve_bytes']==0
    connector.request_finished_all_groups(close,())
    assert not connector.sessions.take_commands().terminals
    assert gate.snapshot()['retiring']==1  # Keep credit through ordinary worker retirement.
    connector.update_connector_output(KVConnectorOutput(finished_sending={close.request_id}))
    assert not gate.leases and not gate.requests and not connector.sessions.uncached_sessions


def test_close_ephemeral_credits_do_not_release_a_peer_or_accept_session_notice():
    from test_session_control import close_request
    gate=HistoryAdmission(220,4,1,100)
    peer=request('peer');gate.allocated(peer,7)
    close=close_request()
    gate.allocated(close)
    gate.finished(close)
    with pytest.raises(ValueError,match='Stale'):
        gate.consume(SessionNotice('retired',('tenant','s'),0,close.request_id,0))
    gate.released(close.request_id)
    assert set(gate.leases)=={('session','tenant','peer')}
    assert gate.leases[('session','tenant','peer')].generation==7


def test_native_connector_skips_only_cpu_blocked_request():
    from test_connector import pair
    connector, _ = pair()
    gate = connector.history_admission = HistoryAdmission(208, 4, 1, 100)
    gate.allocated(request('a'), 1); gate.allocated(request('b'), 2)
    assert connector.get_num_new_matched_tokens(request('large', 16), 0) == (None, False)
    assert connector.get_num_new_matched_tokens(request('small', 4), 0) == (0, False)
