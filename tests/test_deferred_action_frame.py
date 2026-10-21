"""A frozen whole-frame decision gates graph and graph-off archive mutation."""
from copy import deepcopy
from threading import Event
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.deferred_action_frame import DeferredActionFrame
from vllm_kvmem.options import RuntimeOptions
from test_options import config
from test_prefill_pipeline_options import profile
from test_request_publication import fixture, packet, close


def deferred(*, valid=True, resolve=None, fallback=None):
    log = []
    slot = object()
    def release(value):
        assert value is slot
        log.append('release')
    def score(value):
        assert value is slot
        log.append('fallback')
        if fallback:
            fallback()
    def validate():
        log.append('validate')
        if resolve:
            resolve()
        return NS(valid=valid)
    pool = NS(seal=lambda value: log.append('seal'), release=release, fallback=score)
    frame = DeferredActionFrame(pool)
    frame.slot = slot
    frame.seal(slot, NS(resolve=validate))
    return frame, log


def test_poll_never_runs_validation_or_fallback_and_engine_applies_once():
    frame, log = deferred(valid=False)
    assert not frame.resolve(wait=False) and not frame.ready and log == ['seal']
    assert frame.resolve(wait=True) and frame.ready and frame.slot is None
    assert log == ['seal', 'validate', 'fallback', 'release']
    assert frame.resolve(wait=True) and frame.resolve(wait=False)
    bank = NS(apply_detached_validation=lambda ticket, result: log.append(('apply', result.valid)))
    assert frame.apply(bank) and log[-1] == ('apply', False)
    with pytest.raises(RuntimeError, match='already applied'):
        frame.apply(bank)


@pytest.mark.parametrize('graph', [False, True])
def test_entire_shared_frame_resolves_before_either_owner_append(tmp_path, monkeypatch, graph):
    runtime, a, b, archive = fixture(tmp_path, threshold=1024)
    shared = packet(runtime, [('a', 4, 1), ('b', 4, 1)])
    entered, proceed = Event(), Event()
    def validate():
        entered.set()
        assert proceed.wait(10)
    frame, log = deferred(valid=False, resolve=validate)
    shared.action_validation = frame
    if graph:
        from vllm_kvmem.block_graph import BlockGraph
        for name, owner in (('a', a), ('b', b)):
            owner.block_graph = BlockGraph(name, owner.memory.config.block_tokens, initial_position=4)
    observed = []
    original = runtime._prepare_graph_acceptance
    def reserve(*args):
        assert frame.ready and log == ['seal', 'validate', 'fallback', 'release']
        observed.append(args[0].request_ids[args[1]])
        return original(*args)
    monkeypatch.setattr(runtime, '_prepare_graph_acceptance', reserve)
    try:
        assert not runtime.drain(wait=False)
        assert log == ['seal'] and (a.history_end, b.history_end) == (4, 4)
        # A tiny deferred frame still starts the worker; size alone cannot
        # leave its validation parked until a foreground pool-capacity wait.
        assert runtime.start_cpu_scatter() and entered.wait(5)
        assert not runtime.publication_ready({'a', 'b'})
        assert not runtime.drain_requests({'b'}, wait=False)
        assert (a.history_end, b.history_end) == (4, 4) and not observed
        proceed.set()
        runtime.join_cpu_scatter()
        assert observed == (['a', 'b'] if graph else ['a', 'a', 'b', 'b'])
        assert (a.history_end, b.history_end) == (5, 5)
        assert log.count('validate') == log.count('fallback') == log.count('release') == 1
    finally:
        proceed.set()
        close(runtime, (a, b), archive)


@pytest.mark.parametrize('stage', ['validate', 'fallback'])
def test_failure_keeps_snapshot_packet_and_history_unmodified(tmp_path, stage):
    runtime, a, b, archive = fixture(tmp_path)
    shared = packet(runtime, [('a', 4, 4), ('b', 4, 4)])
    def fail():
        raise OSError('injected deferred failure')
    frame, log = deferred(valid=False, **{('resolve' if stage == 'validate' else stage): fail})
    shared.action_validation = frame
    try:
        with pytest.raises(OSError, match='deferred failure'):
            runtime.drain(wait=True)
        assert frame.slot is not None and 'release' not in log
        assert (a.history_end, b.history_end) == (4, 4)
        assert shared.download.owners and runtime.queue == [shared]
        assert not shared.published and not shared.graph_updates
        with pytest.raises(RuntimeError, match='quarantined'):
            frame.resolve(wait=True)
    finally:
        # No pending GPU work in this CPU fault oracle. Production retains the
        # poisoned publication sources and never emits a retirement ACK.
        close(runtime, (a, b), archive)


def test_option_rejects_missing_physical_prefix_lifetime_or_unbounded_settings():
    assert not RuntimeOptions().prefill_deferred_validation
    value = profile()
    value['kvmem'].update(prefill_deferred_validation=True,
        prefill_native_plan_arm=True, prefill_action_before_mtp=True, action_native_scores='reuse')
    before = deepcopy(value)
    assert RuntimeOptions.from_vllm(config(value)).prefill_deferred_validation
    assert value == before
    for name, bad in [('prefill_deferred_validation', 1), ('prefill_action_before_mtp', False),
                      ('prefill_native_plan_arm', False), ('async_archive_retirement', False),
                      ('prefill_deferred_validation_max_bytes', 0),
                      ('prefill_deferred_validation_scratch_bytes', True)]:
        changed = deepcopy(value)
        changed['kvmem'][name] = bad
        with pytest.raises(ValueError):
            RuntimeOptions.from_vllm(config(changed))


def test_native_preempt_and_forced_reset_cannot_release_a_frozen_prefix(monkeypatch):
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_kvmem.scheduler import KVMemScheduler
    scheduler = KVMemScheduler.__new__(KVMemScheduler)
    scheduler._deferred_prefill_validation = True
    owner = object()
    scheduler.requests = {'r': owner}
    scheduler.running = [owner]
    calls = []
    monkeypatch.setattr(Scheduler, '_preempt_request', lambda *a: calls.append('preempt'))
    monkeypatch.setattr(Scheduler, 'reset_prefix_cache', lambda *a: calls.append('reset') or True)
    assert not scheduler.reset_prefix_cache(reset_running_requests=True)
    assert scheduler.requests == {'r': owner} and scheduler.running == [owner] and not calls
    with pytest.raises(RuntimeError, match='forbids native'):
        scheduler._preempt_request(owner, 0)
    assert not calls
    scheduler.requests.clear()
    scheduler.running.clear()
    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert calls == ['reset']
