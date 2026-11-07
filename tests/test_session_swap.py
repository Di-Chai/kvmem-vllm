from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.session_control import SessionBinding, SessionCapture, SessionCommands, SessionEviction
from vllm_kvmem.session_swap import (
    SessionSwapIdentity, session_swap_candidate, validate_session_swap, owned_session_admission,
)


def scheduler_fixture():
    request = NS(request_id='a', session_id='session', cache_salt='salt', resumable=False,
                 prompt_embeds=None, mm_features=None, lora_request=None,
                 prompt_token_ids=[1, 2, 3], max_tokens=10, kv_transfer_params=None)
    entry = NS(namespace=('salt', 'session'), generation=7, request_id='a', phase='active',
               allocated=True, history=0, candidate_position=32)
    ledger = NS(entries={entry.namespace: entry}, requests={'a': entry}, commands=SessionCommands(),
                capacity=2, uncached_sessions={}, waits=0)
    return NS(connector=NS(sessions=ledger)), request, entry


def test_scheduler_identity_preserves_owned_sealed_p_and_finished_request_lease():
    scheduler, request, entry = scheduler_fixture()
    allowed, identity = session_swap_candidate(scheduler, request)
    assert allowed and identity == SessionSwapIdentity(('salt', 'session'), 7)
    request.is_finished = lambda: True
    validate_session_swap(scheduler, request, identity)
    assert entry.phase == 'active' and entry.candidate_position == 32


@pytest.mark.parametrize('change', ['generation', 'namespace', 'request_id', 'phase', 'history', 'allocated'])
def test_saved_namespace_witness_rejects_reuse_or_import(change):
    scheduler, request, entry = scheduler_fixture()
    _, identity = session_swap_candidate(scheduler, request)
    setattr(entry, change, {'generation': 8, 'namespace': ('other', 'session'), 'request_id': 'b',
        'phase': 'retiring', 'history': 16, 'allocated': False}[change])
    with pytest.raises(ValueError, match='changed'):
        validate_session_swap(scheduler, request, identity)


@pytest.mark.parametrize('kind', ['bindings', 'captures', 'terminals'])
def test_pending_session_commands_defer_new_swap(kind):
    scheduler, request, _ = scheduler_fixture()
    getattr(scheduler.connector.sessions.commands, kind).append(NS(request_id='a'))
    assert session_swap_candidate(scheduler, request) == (False, None)


def test_import_and_unowned_session_defer_but_stateless_is_unchanged():
    scheduler, request, entry = scheduler_fixture()
    entry.history = 16
    assert session_swap_candidate(scheduler, request) == (False, None)
    scheduler.connector.sessions.requests.clear()
    assert session_swap_candidate(scheduler, request) == (False, None)
    request.session_id = None
    assert session_swap_candidate(scheduler, request) == (True, None)
    validate_session_swap(scheduler, request, None)


def test_full_owned_pool_waits_instead_of_falling_back_to_uncached_session():
    scheduler, request, entry = scheduler_fixture()
    ledger = scheduler.connector.sessions
    ledger.capacity = 1
    request.session_id = 'new'
    assert not owned_session_admission(ledger, request) and ledger.waits == 1
    assert not ledger.uncached_sessions
    entry.phase = 'idle'
    assert owned_session_admission(ledger, request)  # Normal lookup may evict this idle entry.
    entry.phase = 'active'
    request.kv_transfer_params = {'kvmem_release_session': True}
    request.prompt_token_ids, request.max_tokens = [1], 1
    assert owned_session_admission(ledger, request)


def test_worker_p_snapshot_and_authority_stay_owned_until_explicit_swap_release(tmp_path):
    from test_session_worker import worker_fixture
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    identity = SessionSwapIdentity(binding.namespace, binding.generation)
    checkpoint = worker.pool.entries[binding.namespace]
    assert worker.swap_ready('a', identity)
    worker.begin_swap('a', identity)
    worker.validate_swap('a', identity)
    assert worker.pool.entries[binding.namespace] is checkpoint
    assert worker.runtime.authorities['a'] is authority
    assert not worker.close()
    with pytest.raises(RuntimeError, match='swap release'):
        worker.detach('a', authority, {})
    with pytest.raises(RuntimeError, match='swap release'):
        worker.before_attention(NS(req_ids=['a']))
    with pytest.raises(RuntimeError, match='swap release'):
        worker.receive(SessionCommands(captures=[SessionCapture('a', binding.generation, 8)]))
    with pytest.raises(RuntimeError, match='swap release'):
        worker.receive(SessionCommands(bindings=[replace(binding, request_id='new', history=8)]))
    assert 'new' not in worker.bindings
    with pytest.raises(RuntimeError, match='swap release'):
        worker.evict([SessionEviction(binding.namespace, binding.generation)])
    worker.end_swap('a', identity)
    assert not worker._swap_fences
    assert worker.close() and authority.close() and archive.close()


def test_worker_waits_for_pm_downloads_and_pending_capture_before_page_detach(tmp_path):
    from test_core import Event
    from test_session_worker import worker_fixture
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    identity = SessionSwapIdentity(binding.namespace, binding.generation)
    checkpoint = worker.pool.entries[binding.namespace]
    event = Event()
    checkpoint.model = replace(checkpoint.model, completion=event, source_owners=(object(),))
    assert not worker.swap_ready('a', identity)
    with pytest.raises(RuntimeError, match='completion'): worker.begin_swap('a', identity)
    event.done = True
    assert worker.swap_ready('a', identity)
    worker.captures['a'] = SessionCapture('a', binding.generation, 8)
    assert not worker.swap_ready('a', identity)
    worker.captures.clear()
    worker.imports.add('a')
    assert not worker.swap_ready('a', identity)
    worker.imports.clear()
    assert worker.close() and authority.close() and archive.close()


@pytest.mark.parametrize('change', ['binding', 'generation', 'authority', 'checkpoint'])
def test_worker_detects_rebound_namespace_or_replaced_retained_state(tmp_path, change):
    from test_session_worker import worker_fixture
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    identity = SessionSwapIdentity(binding.namespace, binding.generation)
    checkpoint = worker.pool.entries[binding.namespace]
    worker.begin_swap('a', identity)
    if change == 'binding': worker.bindings['a'] = replace(binding)
    if change == 'generation': worker.identities[binding.namespace] += 1
    if change == 'authority': worker.runtime.authorities['a'] = object()
    if change == 'checkpoint': worker.pool.entries.pop(binding.namespace)
    with pytest.raises(ValueError): worker.validate_swap('a', identity)
    worker.bindings['a'] = binding
    worker.identities[binding.namespace] = binding.generation
    worker.runtime.authorities['a'] = authority
    worker.pool.entries[binding.namespace] = checkpoint
    worker.end_swap('a', identity)
    assert worker.close() and authority.close() and archive.close()


def test_missing_session_identity_never_silently_uses_stateless_path(tmp_path):
    from test_session_worker import worker_fixture
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    with pytest.raises(ValueError, match='omitted'): worker.swap_ready('a', None)
    assert worker.swap_ready('other', None)
    worker.begin_swap('other', None)
    worker.validate_swap('other', None)
    worker.end_swap('other', None)
    assert worker.close() and authority.close() and archive.close()


def test_incremental_connector_admits_owned_session_but_preserves_namespace_serialization():
    from test_connector import pair
    from test_session_control import ledger, request
    connector, _ = pair()
    connector.gpu_memory_admission = 'incremental'
    connector.sessions, _ = ledger(capacity=1)
    first = request('first', 'session')
    assert connector.get_num_new_matched_tokens(first, 0) == (0, False)
    connector.update_state_after_alloc(first, None, 0)
    assert connector.sessions.requests['first'].phase == 'active'
    assert connector.get_num_new_matched_tokens(request('same', 'session'), 0) == (None, False)
    assert connector.get_num_new_matched_tokens(request('other', 'other'), 0) == (None, False)
    assert not connector.sessions.uncached_sessions
    assert connector.get_num_new_matched_tokens(request('plain', None), 0) == (0, False)


def test_swap_fence_preserves_both_prompt_and_terminal_checkpoint_versions(tmp_path):
    from test_core import Event
    from test_session_worker import worker_fixture
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    worker.pool.retain_prompt = True
    prompt = worker.pool.entries[binding.namespace]
    worker.runtime.last_steps['a'] = [dict(history_begin=6, accepted=2, scheduled=2)]
    worker.model_adapter.recurrent.conv_dim_first = False
    worker.model_adapter.recurrent.conv_widths = {'gdn': 4}
    terminal = worker.pool.capture(binding.namespace, 'a', authority, list(range(11)),
                                  worker.model_adapter, replace=True, terminal=True)
    assert terminal.kind == 'M' and worker.pool.prompts[binding.namespace] is prompt
    identity = SessionSwapIdentity(binding.namespace, binding.generation)
    fence = Event()
    prompt.model = replace(prompt.model, completion=fence)
    assert not worker.swap_ready('a', identity)
    fence.done = True
    worker.begin_swap('a', identity)
    worker.validate_swap('a', identity)
    assert worker.pool.entries[binding.namespace] is terminal
    assert worker.pool.prompts[binding.namespace] is prompt
    worker.end_swap('a', identity)
    assert worker.close() and authority.close() and archive.close()
