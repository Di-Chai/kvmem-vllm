"""CPU lifecycle regressions for protected terminal frames under GC contention.

The native store and checkpoint/model snapshots are real. Only the historical
policy gate is controlled, so a retry cannot silently become ordinary retirement.
"""
from dataclasses import replace
from types import SimpleNamespace as NS
import weakref

import pytest

from vllm.v1.core.sched.output import SchedulerOutput
from vllm_kvmem.serving.connector import ControlMetadata
from vllm_kvmem.core.options import RuntimeOptions
from vllm_kvmem.serving.session_control import SessionCapture, SessionCommands
from vllm_kvmem.serving.terminal import TerminalPrefix
from test_session_worker import worker_fixture, wait_retirement


@pytest.fixture
def owner(tmp_path, monkeypatch):
    import test_session_worker as helper
    setup = helper.setup
    def protected_setup(*args, **kwargs):
        authority, archive = setup(*args, **kwargs)
        authority.memory._store.set_protect_current_window(True)
        return authority, archive
    monkeypatch.setattr(helper, 'setup', protected_setup)
    worker, binding, authority, archive, _ = worker_fixture(tmp_path)
    worker.options.verify_archive_on_finish = False
    runtime = worker.runtime
    runtime.options = RuntimeOptions(memory=authority.memory.config)
    runtime.last_steps['a'] = [dict(history_begin=4, accepted=4, scheduled=4)]
    worker.model_adapter.recurrent.conv_dim_first = False
    worker.model_adapter.recurrent.conv_widths = {'gdn': 4}
    worker.runner._kvmem_sessions = worker
    command = SessionCapture('a', binding.generation, authority.history_end, binding.tokens)
    yield NS(worker=worker, binding=binding, authority=authority, archive=archive,
             runtime=runtime, command=command)
    assert worker.close()
    assert authority.close()
    assert archive.tier.used_slots == 0
    assert archive.close()


def drop_prompt(owner):
    owner.worker.pool.drop_live(owner.worker.pool.entries[owner.binding.namespace])


def output(owner, *, commands=True):
    prefix = TerminalPrefix(owner.authority.history_end, owner.authority.history_end, 1, 'FINISHED_LENGTH_CAPPED')
    return replace(SchedulerOutput.make_empty(), finished_req_ids={'a'},
        num_scheduled_tokens={'peer': 1}, total_num_scheduled_tokens=1,
        kv_connector_metadata=ControlMetadata(terminal_prefixes={'a': prefix},
            sessions=SessionCommands(terminals=[owner.command] if commands else [])))


def empty_output():
    return replace(SchedulerOutput.make_empty(), kv_connector_metadata=ControlMetadata())


@pytest.mark.parametrize('command_already_received', [False, True])
def test_busy_policy_gate_defers_once_then_next_step_reuses_prepared_frame(owner, monkeypatch, command_already_received):
    drop_prompt(owner)
    worker, memory = owner.worker, owner.authority.memory
    original = memory.checkpoint_window
    attempts = []
    busy = True
    def checkpoint(**kwargs):
        attempts.append(kwargs)
        if busy:
            raise BufferError('Historical checkpoint admission must retry before model capture')
        return original(**kwargs)
    monkeypatch.setattr(memory, 'checkpoint_window', checkpoint)
    if command_already_received:
        worker.receive(SessionCommands(terminals=[owner.command]))
    initial = output(owner, commands=not command_already_received)
    held = owner.runtime.defer_terminals(initial)
    assert len(attempts) == 1  # No same-step second attempt from the deferred loop.
    assert held.finished_req_ids == set()
    assert held.num_scheduled_tokens == {'peer': 1}
    assert not held.kv_connector_metadata.terminal_prefixes
    assert 'a' in worker.runner.req_states.req_id_to_index
    assert 'a' in owner.runtime.authorities and not worker.pool.entries
    assert not worker.prepared_terminal_windows and not worker.notices()
    assert worker.checkpoint_deferrals == 1
    if not command_already_received:
        worker.receive(held.kv_connector_metadata.sessions)
    busy = False
    resumed = owner.runtime.defer_terminals(empty_output())
    assert resumed.finished_req_ids == {'a'} and len(attempts) == 2
    assert resumed.kv_connector_metadata.terminal_prefixes['a'] == initial.kv_connector_metadata.terminal_prefixes['a']
    prepared = weakref.ref(worker.prepared_terminal_windows['a'])
    assert worker.prepare_terminal_checkpoint('a', owner.command) and len(attempts) == 2
    # detach runs after native terminal state release; reacquiring a policy gate
    # here would again turn a transient busy result into a discarded session.
    monkeypatch.setattr(memory, 'checkpoint_window', lambda **_: pytest.fail('Prepared frame was reacquired'))
    assert worker.detach('a', owner.authority, {'terminal_prefix': {'status': 'FINISHED_LENGTH_CAPPED'}})
    assert not worker.prepared_terminal_windows
    assert worker.pool.entries[owner.binding.namespace].window is prepared()
    del owner.runtime.authorities['a']
    wait_retirement(worker, 'a')
    assert [notice.kind for notice in worker.notices()] == ['captured', 'retired']
    assert owner.runtime.completed[-1]['checkpoint_kind'] == 'M'
    assert owner.runtime.completed[-1]['history_end'] == owner.command.position
    assert not owner.runtime.defer_terminals(empty_output()).finished_req_ids
    assert worker.close() and prepared() is None


def test_terminal_preparation_waits_for_accepted_publication_without_blocking_peer(owner, monkeypatch):
    drop_prompt(owner)
    ready, calls = False, []
    prepare = owner.worker.prepare_terminal_checkpoint
    monkeypatch.setattr(owner.runtime, 'publication_ready', lambda ids: ready)
    monkeypatch.setattr(owner.worker, 'prepare_terminal_checkpoint',
        lambda rid, command: calls.append(rid) or prepare(rid, command))
    held = owner.runtime.defer_terminals(output(owner))
    assert not held.finished_req_ids and not calls
    owner.worker.receive(held.kv_connector_metadata.sessions)
    ready = True
    resumed = owner.runtime.defer_terminals(empty_output())
    assert resumed.finished_req_ids == {'a'} and calls == ['a']
    assert 'a' in owner.worker.prepared_terminal_windows


@pytest.mark.parametrize('keep_previous', [False, True])
def test_real_checkpoint_capacity_failure_keeps_ordinary_fallback_and_releases_prepared_pin(owner, keep_previous):
    worker = owner.worker
    if not keep_previous:
        drop_prompt(owner)
    previous = worker.pool.entries.get(owner.binding.namespace)
    assert worker.prepare_terminal_checkpoint('a', owner.command)
    prepared = weakref.ref(worker.prepared_terminal_windows['a'])
    worker.receive(SessionCommands(terminals=[owner.command]))
    worker.pool.max_payload_bytes = 1
    assert worker.detach('a', owner.authority, {}) is keep_previous
    assert not worker.prepared_terminal_windows and prepared() is None
    assert worker.checkpoint_deferrals == 1
    assert worker.notices()[0].kind == 'capture_deferred'
    if keep_previous:
        assert worker.pool.entries[owner.binding.namespace] is previous
        wait_retirement(worker, 'a')
        assert worker.notices()[0].kind == 'retired'
    else:
        assert not worker.pool.entries
        assert owner.authority.close()
        worker.ordinary_retired('a')
        notice, = worker.notices()
        assert notice.kind == 'retired' and notice.history == notice.window == 0
        assert not worker.bindings and not worker.identities


@pytest.mark.parametrize('kind', ['no_terminal', 'release', 'unbound'])
def test_requests_without_terminal_capture_are_never_held_or_given_phantom_checkpoints(owner, monkeypatch, kind):
    drop_prompt(owner)
    if kind == 'release':
        owner.worker.release_requests.add('a')
    elif kind == 'unbound':
        owner.worker.bindings.clear()
        owner.worker.identities.clear()
    monkeypatch.setattr(owner.worker, 'prepare_terminal_checkpoint',
        lambda *args: pytest.fail('A request without a terminal command was prepared'))
    result = owner.runtime.defer_terminals(output(owner, commands=False))
    assert result.finished_req_ids == {'a'}
    assert not owner.worker.prepared_terminal_windows
    assert not owner.worker.detach('a', owner.authority, {})
    assert owner.authority.close()
    owner.worker.ordinary_retired('a')
    assert not owner.worker.release_requests and not owner.worker.bindings


@pytest.mark.parametrize('damage', ['generation', 'history'])
def test_terminal_preparation_rejects_stale_identity_before_pin_acquisition(owner, monkeypatch, damage):
    monkeypatch.setattr(owner.authority.memory, 'checkpoint_window',
        lambda **_: pytest.fail('Invalid terminal metadata acquired a frame'))
    command = replace(owner.command, **({'generation': 2} if damage == 'generation' else {'position': 4}))
    with pytest.raises(ValueError, match='stale session|accepted history'):
        owner.worker.prepare_terminal_checkpoint('a', command)
    assert not owner.worker.prepared_terminal_windows


@pytest.mark.parametrize('damage', ['not_terminal', 'foreign_owner', 'different_records'])
def test_pool_rejects_a_prepared_frame_from_another_boundary(owner, damage):
    assert owner.worker.prepare_terminal_checkpoint('a', owner.command)
    prepared = owner.worker.prepared_terminal_windows['a']
    if damage == 'foreign_owner': prepared = replace(prepared, request_id='foreign')
    if damage == 'different_records': prepared = replace(prepared, records=prepared.records[:-1])
    with pytest.raises(ValueError, match='Prepared terminal checkpoint'):
        owner.worker.pool.capture(owner.binding.namespace, 'a', owner.authority, owner.binding.tokens,
            owner.worker.model_adapter, replace=True, terminal=damage != 'not_terminal', prepared_window=prepared)
    assert owner.worker.pool.entries[owner.binding.namespace].kind == 'P'


def test_shutdown_releases_unconsumed_prepared_frame_before_pool_eviction(owner):
    drop_prompt(owner)
    assert owner.worker.prepare_terminal_checkpoint('a', owner.command)
    prepared = weakref.ref(owner.worker.prepared_terminal_windows['a'])
    # A shutdown between native preparation and detach must release these
    # prefix leases even though no model checkpoint ever entered the pool.
    assert owner.worker.close()
    assert prepared() is None and not owner.worker.prepared_terminal_windows
    assert owner.authority.close() and owner.archive.tier.used_slots == 0
