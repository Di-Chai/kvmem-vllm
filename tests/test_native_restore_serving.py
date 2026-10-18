"""Opt-in native frame preparation retains the serving transaction contract."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vllm_kvmem.memory import Phase
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool
from vllm_kvmem.restore_timing import RestoreTiming
from vllm_kvmem.session_worker import SessionWorker
from test_native_restore_frame import Event, commit, fixture, plan_value
from test_options import config


def test_native_restore_option_defaults_off_and_round_trips_without_input_mutation():
    assert RuntimeOptions().native_restore_frame is False
    assert RuntimeOptions.from_vllm(config({})).native_restore_frame is False
    payload = {'kvmem': {'native_restore_frame': True}}
    options = RuntimeOptions.from_vllm(config(payload))
    assert options.native_restore_frame is True
    assert payload == {'kvmem': {'native_restore_frame': True}}
    disabled = RuntimeOptions.from_vllm(config({'kvmem': {'native_restore_frame': False}}))
    assert disabled.native_restore_frame is False


@pytest.mark.parametrize('invalid', [0, 1, 'false', None])
def test_native_restore_option_rejects_non_boolean(invalid):
    with pytest.raises(ValueError, match='native_restore_frame must be boolean'):
        RuntimeOptions.from_vllm(config({'kvmem': {'native_restore_frame': invalid}}))
    with pytest.raises(ValueError, match='native_restore_frame must be boolean'):
        PromptCheckpointPool(1, 64, native_restore_frame=invalid)


@pytest.mark.parametrize('enabled', [False, True])
def test_worker_propagates_explicit_option_to_owned_checkpoint_pool(enabled):
    worker = SessionWorker(SimpleNamespace(), SimpleNamespace(session_capacity=1,
        session_checkpoint_bytes=64, native_restore_frame=enabled))
    assert worker.pool.native_restore_frame is enabled


@pytest.mark.parametrize('kind', ['dense', 'sparse', 'partial'])
def test_enabled_serving_restore_matches_default_path_and_preserves_fence(kind):
    old, old_checkpoint = fixture(kind)
    native, checkpoint = fixture(kind)
    expected = old.restore_window(old_checkpoint)
    timing = RestoreTiming(enabled=True)
    actual = native.restore_window(checkpoint, native_restore_frame=True, timing=timing)
    assert plan_value(actual) == plan_value(expected)
    assert native.phase is Phase.ASSEMBLING and native._plan is actual
    assert native._submitted is False and native._fence is None
    assert [row['name'] for row in timing.summary()['stages']] == ['native_restore_frame']
    with pytest.raises(RuntimeError, match='expected READY'):
        native.restore_window(checkpoint, native_restore_frame=True)
    assert native._plan is actual
    with pytest.raises(RuntimeError, match='No completion fence'):
        native.commit_selection()
    fence = Event(False)
    native.attach_completion(fence)
    assert not native.commit_selection() and not native.close()
    assert native._plan is actual and native.phase is Phase.ASSEMBLING
    fence.done = True
    assert native.commit_selection()
    commit(old)
    assert native.phase is Phase.READY and native._plan is native._fence is None
    assert tuple(native._store.working_records()) == tuple(old._store.working_records()) == checkpoint.records
    assert old.close() and native.close()


class StoreProxy:
    def __init__(self, store, native):
        self.store, self.native = store, native

    def __getattr__(self, name):
        if name == 'restore_frame':
            return self.native()
        return getattr(self.store, name)


def test_default_path_never_resolves_optional_native_adapter():
    memory, checkpoint = fixture()
    memory._store = StoreProxy(memory._store,
        lambda: pytest.fail('Default path consulted native adapter'))
    memory.restore_window(checkpoint)
    commit(memory)
    assert memory.close()


@pytest.mark.parametrize('malformed', ['wrong_type', 'wrong_request', 'bad_record', 'foreign_prefix'])
def test_native_serving_validation_failure_keeps_ready_transaction(malformed):
    memory, checkpoint = fixture(current='same')
    foreign, other = fixture(current='same')
    if malformed == 'wrong_type':
        checkpoint = object()
    elif malformed == 'wrong_request':
        checkpoint = replace(checkpoint, request_id='other-request')
    elif malformed == 'foreign_prefix':
        checkpoint = replace(checkpoint, prefix=other.prefix)
    else:
        row = checkpoint.records[-1]
        checkpoint = replace(checkpoint, records=(*checkpoint.records[:-1], (*row[:4], row[4]+1)))
    before = tuple(memory._store.working_records()), memory.repository.stats['leases']
    timing = RestoreTiming(enabled=True)
    with pytest.raises(ValueError):
        memory.restore_window(checkpoint, native_restore_frame=True, timing=timing)
    assert memory.phase is Phase.READY and memory._plan is memory._fence is None
    assert memory._submitted is False
    assert (tuple(memory._store.working_records()), memory.repository.stats['leases']) == before
    if malformed in ('wrong_type', 'wrong_request'):
        assert not timing.summary()['stages']  # Reject before entering C++.
    else:
        assert timing.summary()['stages'][0]['status'] == 'error'
    assert memory.close() and foreign.close()


def test_enabled_path_fails_explicitly_if_native_extension_is_missing():
    memory, checkpoint = fixture()

    def absent():
        raise AttributeError('restore_frame')

    memory._store = StoreProxy(memory._store, absent)
    with pytest.raises(RuntimeError, match='refusing silent fallback'):
        memory.restore_window(checkpoint, native_restore_frame=True)
    assert memory.phase is Phase.READY and memory._plan is None
    assert memory.close()


def test_enabled_restore_with_disabled_timing_never_reads_observer_clocks():
    memory, checkpoint = fixture()

    def forbidden_clock():
        pytest.fail('Disabled restore timing read a clock')

    timing = RestoreTiming(enabled=False, clock=forbidden_clock, thread_clock=forbidden_clock)
    memory.restore_window(checkpoint, native_restore_frame=True, timing=timing)
    assert not timing.summary()['stages']
    commit(memory)
    assert memory.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_pool_claim_restore_forwards_opt_in_only_and_preserves_exact_frame(tmp_path, monkeypatch, enabled):
    from test_prompt_checkpoints import ready, saved
    pool = PromptCheckpointPool(2, 4096, native_restore_frame=enabled)
    pool, checkpoint, authority, archive, _, _ = saved(tmp_path, pool=pool)
    pool.retire(checkpoint, Event())
    ready(pool, checkpoint)
    assert pool.claim(checkpoint.namespace, 'new', list(range(11))) is checkpoint
    original, calls = authority.memory.restore_window, []

    def restore(window, **kwargs):
        calls.append(dict(kwargs))
        return original(window, **kwargs)

    monkeypatch.setattr(authority.memory, 'restore_window', restore)
    timing = RestoreTiming(enabled=True)
    plan = pool.prepare_claim_window(checkpoint, wait=False, timing=timing)
    assert calls == [dict(timing=timing, **({'native_restore_frame': True} if enabled else {}))]
    assert plan.total_window_tokens == checkpoint.window.window_tokens
    assert checkpoint.restore_started and authority.memory.phase is Phase.ASSEMBLING
    commit(authority.memory)
    assert pool.finish_claim(checkpoint, Event()) is authority
    del plan
    assert authority.close() and archive.close()
