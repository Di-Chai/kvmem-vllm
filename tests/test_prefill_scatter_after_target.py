"""CPU proof that delaying worker startup changes no publication authority."""
from concurrent.futures import Future
from dataclasses import replace
from threading import Event
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem.action_attention import ActionRuntime
from vllm_kvmem.capture_runtime import CaptureRuntime, CapturedPacket
from vllm_kvmem.connector import ControlMetadata
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.prefill_planning import PrefillPlanFrame
from vllm_kvmem.runner import make_probe_runner
from test_options import config
from test_prefill_pipeline_options import profile


def fixture(*, owners=('a',), enabled=True):
    runtime = CaptureRuntime.__new__(CaptureRuntime)
    runtime.options = NS(prefill_scatter_after_target=enabled, async_cpu_scatter=True,
        prefill_publication_pipeline=True, cpu_scatter_min_tokens=128)
    runtime.queue = []
    runtime.terminal_prefixes, runtime.retiring = {}, {}
    runtime.runner = NS(req_states=NS(req_id_to_index={rid: i for i, rid in enumerate(owners)},
        prompt_len=NS(np=np.full(len(owners), 32768))),
        _kvmem_window_adapter=NS(offsets={}), _kvmem_sessions=NS())
    runtime.authorities = {rid: NS(action_history=NS(prefill_replay=None, prefill_planning_epoch=0))
                           for rid in owners}
    action = runtime.action = ActionRuntime.__new__(ActionRuntime)
    action.runner, action.runtime = runtime.runner, runtime
    action.native = NS(mode='reuse', disabled=None)
    action.prefill_plan_frames = {rid: PrefillPlanFrame(runtime.authorities[rid],
        runtime.authorities[rid].action_history, (0, 0, 0, 0, None), 2048, 2048, 2048, ())
        for rid in owners}
    old = CapturedPacket(owners, 2048 * len(owners), NS(owners=(object(),), query=lambda: True),
        request_tokens=(2048,) * len(owners), pipeline_requests=frozenset(owners))
    runtime.queue = [old]
    runtime.downloads = NS(pending={0: old.download}, buffers=[object(), object()])
    output = NS(num_scheduled_tokens=dict.fromkeys(owners, 2048), total_num_scheduled_tokens=2048*len(owners),
        kv_connector_metadata=ControlMetadata(), finished_req_ids=set(), scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={}, scheduled_cached_reqs=NS(req_ids=list(owners),
            resumed_req_ids=set(), num_computed_tokens=[2048]*len(owners), num_output_tokens=[0]*len(owners)))
    batch = NS(req_ids=list(owners), num_reqs=len(owners), is_prefilling_np=np.ones(len(owners), dtype=bool),
        query_start_loc_np=np.arange(len(owners)+1)*2048, num_computed_tokens_np=np.full(len(owners), 2048))
    action.pipeline_requests = set(owners)
    plans = {rid: dict(history_begin=2048, local_begin=2048, count=2048, index=i,
                      begin=i*2048, pipeline=True) for i, rid in enumerate(owners)}
    action.prefill_plans = action.native_plans = (batch, plans)
    return runtime, output, batch, old


def test_default_off_and_explicit_dependencies_preserve_other_options():
    assert not RuntimeOptions().prefill_scatter_after_target
    payload = profile()['kvmem']
    payload.update(prefill_scatter_after_target=True, prefill_native_plan_arm=True,
                   action_native_scores='reuse')
    result = RuntimeOptions.from_vllm(config({'kvmem': payload}))
    assert result.prefill_scatter_after_target and not result.prefill_deferred_validation
    for key, value in [('prefill_scatter_after_target', 1), ('async_cpu_scatter', False),
                       ('split_capture_dma', False), ('prefill_publication_pipeline', False),
                       ('prefill_native_plan_arm', False)]:
        with pytest.raises(ValueError):
            RuntimeOptions.from_vllm(config({'kvmem': {**payload, key: value}}))


@pytest.mark.parametrize('owners', [('a',), ('a', 'b')])
def test_one_confirmed_packet_keeps_its_sources_and_empty_slot_until_target(owners):
    runtime, output, batch, old = fixture(owners=owners)
    calls = []
    runtime.start_cpu_scatter = lambda: calls.append('start') or True
    assert runtime.defer_prefill_scatter(output)
    runtime.advance_publication(output.num_scheduled_tokens, defer_scatter=True)
    assert not calls and old.confirmed and old.download.owners and not old.published
    assert len(runtime.downloads.pending) == 1
    runtime.recheck_prefill_scatter(batch)
    assert not calls
    assert runtime.resume_prefill_scatter(after_target=True, batch=batch)
    assert calls == ['start'] and runtime._prefill_scatter_deferred is None
    assert runtime.prefill_scatter_after_target_stats == {
        'deferred_batches': 1, 'after_target_started_batches': 1}
    assert not runtime.resume_prefill_scatter(after_target=True, batch=batch)


@pytest.mark.parametrize('changed', ['off', 'no_tokens', 'commands', 'cancelled', 'terminal',
    'finished', 'preempted', 'retiring', 'session_import', 'session_capture', 'session_terminal',
    'session_retiring', 'session_eviction', 'new_owner', 'resumed', 'speculative', 'encoder',
    'sync_load', 'disabled_native', 'verify', 'mixed_decode', 'last_prefill', 'missing_plan',
    'frame_generation', 'replay_overlap', 'unconfirmed', 'wrong_lane', 'published', 'consuming',
    'different_owner', 'nonpipeline', 'full_slots', 'two_packets', 'small_packet', 'offset'])
def test_lifecycle_and_nonordinary_inputs_keep_immediate_start(changed):
    runtime, output, batch, old = fixture()
    if changed == 'off': runtime.options.prefill_scatter_after_target = False
    elif changed == 'no_tokens': output.num_scheduled_tokens = {}
    elif changed == 'commands': output.kv_connector_metadata.commands = [object()]
    elif changed == 'cancelled': output.kv_connector_metadata.cancelled = {'other'}
    elif changed == 'terminal': output.kv_connector_metadata.terminal_prefixes = {'other': object()}
    elif changed == 'finished': output.finished_req_ids = {'other'}
    elif changed == 'preempted': output.preempted_req_ids = {'other'}
    elif changed == 'retiring': runtime.retiring = {'other': object()}
    elif changed.startswith('session_') and changed != 'session_eviction':
        field = dict(session_import='imports', session_capture='captures',
                     session_terminal='terminal_captures', session_retiring='retiring')[changed]
        setattr(runtime.runner._kvmem_sessions, field, {'other': object()})
    elif changed == 'session_eviction': output.kv_connector_metadata.sessions.evictions = [object()]
    elif changed == 'new_owner': output.scheduled_new_reqs = [object()]
    elif changed == 'resumed': output.scheduled_cached_reqs.resumed_req_ids = {'a'}
    elif changed == 'speculative': output.scheduled_spec_decode_tokens = {'a': [1]}
    elif changed == 'encoder': output.scheduled_encoder_inputs = {'a': [0]}
    elif changed == 'sync_load': output.has_sync_kv_loads = True
    elif changed == 'disabled_native': runtime.action.native.disabled = 'unsupported'
    elif changed == 'verify': runtime.action.native.mode = 'verify'
    elif changed == 'mixed_decode': output.scheduled_cached_reqs.num_output_tokens = [1]
    elif changed == 'last_prefill': runtime.runner.req_states.prompt_len.np[0] = 4096
    elif changed == 'missing_plan': runtime.action.prefill_plan_frames.clear()
    elif changed == 'frame_generation': runtime.runner._kvmem_request_generations = {'a': 1}
    elif changed == 'replay_overlap': runtime.authorities['a'].action_history.prefill_replay = (2048, 4096)
    elif changed == 'unconfirmed': old.confirmed = False
    elif changed == 'wrong_lane': old.lane = 'decode'
    elif changed == 'published': old.published.add(0)
    elif changed == 'consuming': old.consuming.add(0)
    elif changed == 'different_owner': old.request_ids = ('other',)
    elif changed == 'nonpipeline': old.pipeline_requests = frozenset()
    elif changed == 'full_slots': runtime.downloads.pending[1] = object()
    elif changed == 'two_packets': runtime.queue.append(replace(old))
    elif changed == 'small_packet': old.tokens = 1
    elif changed == 'offset': runtime.runner._kvmem_window_adapter.offsets['a'] = 32
    assert not runtime.defer_prefill_scatter(output)
    assert getattr(runtime, '_prefill_scatter_deferred', None) is None
    calls = []
    runtime.start_cpu_scatter = lambda: calls.append('start')
    runtime.drain_requests = lambda *a, **k: None
    runtime.advance_publication(output.num_scheduled_tokens)
    assert calls == ['start']
    if changed != 'off':
        assert runtime.prefill_scatter_after_target_stats['eligibility_declines'] == 1


def test_active_worker_is_neither_paused_nor_joined():
    runtime, output, batch, old = fixture()
    future = runtime._cpu_scatter_future = Future()
    assert not runtime.defer_prefill_scatter(output)
    runtime.advance_publication(output.num_scheduled_tokens)
    assert runtime._cpu_scatter_future is future and not future.done()
    assert runtime.prefill_scatter_after_target_stats['already_running_batches'] == 1


def test_completed_worker_error_is_collected_before_any_new_forward():
    runtime, output, batch, old = fixture()
    future = runtime._cpu_scatter_future = Future()
    future.set_exception(OSError('old publication failed'))
    with pytest.raises(OSError, match='publication failed'):
        runtime.defer_prefill_scatter(output)
    assert runtime._publication_error is future.exception()
    assert old.download.owners and not old.published


@pytest.mark.parametrize('change', ['pipeline', 'count', 'history', 'local', 'index', 'begin', 'native', 'decode'])
def test_prepared_batch_mismatch_resumes_immediately(change):
    runtime, output, batch, old = fixture()
    calls = []
    runtime.start_cpu_scatter = lambda: calls.append('start') or True
    assert runtime.defer_prefill_scatter(output)
    if change == 'native': runtime.action.native_plans = None
    elif change == 'decode': batch.is_prefilling_np[0] = False
    else:
        key = dict(history='history_begin', local='local_begin').get(change, change)
        runtime.action.prefill_plans[1]['a'][key] = False if change == 'pipeline' else -1
    runtime.recheck_prefill_scatter(batch)
    assert calls == ['start'] and runtime._prefill_scatter_deferred is None
    assert runtime.prefill_scatter_after_target_stats['resumed_before_target_batches'] == 1


def runner_fixture(*, fail=None, recheck=True, fresh=True, owners=('a',)):
    runtime, output, batch, old = fixture(owners=owners)
    calls = []

    class Base:
        def execute_model(self, *args, **kwargs):
            calls.append('target_enter')
            if recheck:
                self._kvmem_prepare_action_metadata(batch, None)
            calls.extend(('embedding', 'first_fa2', 'last_layer'))
            if fail == 'target':
                raise OSError('target failed')
            if fresh:
                self.execute_model_state = NS(input_batch=batch)
            return 'queued'

    runner = make_probe_runner(Base)()
    runner.__dict__.update(vars(runtime.runner))
    runner._kvmem_capture_runtime = runtime
    runtime.runner = runtime.action.runner = runner
    runtime.action.prepare = lambda *a: calls.append('prepare')
    runner._kvmem_public_work = NS(submit=lambda *a: calls.append('work'))

    def start():
        calls.append('scatter')
        if fail == 'start':
            raise OSError('scatter startup failed')
        return True
    runtime.start_cpu_scatter = start
    assert runtime.defer_prefill_scatter(output)
    runtime.advance_publication(output.num_scheduled_tokens, defer_scatter=True)
    return runner, runtime, output, batch, old, calls


@pytest.mark.parametrize('owners', [('a',), ('a', 'b')])
def test_real_runner_hook_starts_once_after_complete_target_before_sampling(owners):
    runner, runtime, output, batch, old, calls = runner_fixture(owners=owners)
    assert runner._kvmem_execute_public_forward(output, observe_work=True) == 'queued'
    calls.append('sample_mtp_submit')
    assert calls == ['target_enter', 'prepare', 'embedding', 'first_fa2', 'last_layer',
                     'work', 'scatter', 'sample_mtp_submit']
    assert not runner._kvmem_public_action_forward and runtime._prefill_scatter_deferred is None
    assert not old.published and old.download.owners


@pytest.mark.parametrize('fail,recheck,fresh', [('target', True, True), ('start', True, True),
                                              (None, False, True), (None, True, False)])
def test_real_runner_failure_quarantines_old_sources_without_publication_or_ack(fail,recheck,fresh):
    runner, runtime, output, batch, old, calls = runner_fixture(fail=fail,recheck=recheck,fresh=fresh)
    with pytest.raises((OSError, RuntimeError)):
        runner._kvmem_execute_public_forward(output, observe_work=True)
    assert runtime._publication_error is not None and runtime._prefill_scatter_deferred is None
    assert runtime.prefill_scatter_after_target_stats['quarantined_batches'] == 1
    assert not old.published and old.download.owners and runtime.queue == [old]
    with pytest.raises(type(runtime._publication_error)): runtime.publication_ready({'a'})
    if fail != 'start': assert 'scatter' not in calls


def test_deferred_worker_runs_once_and_still_stops_at_next_unconfirmed_packet():
    runtime, output, batch, old = fixture()
    entered = Event()
    def publish(*, wait):
        assert wait
        # Existing drain, not the delayed-start feature, owns confirmation and
        # FIFO. Exercise the real executor handoff with a deterministic consumer.
        assert runtime.queue[0] is old and old.confirmed
        old.published.add(0)
        runtime.queue.pop(0)
        entered.set()
    runtime.drain = publish
    assert runtime.defer_prefill_scatter(output)
    runtime.recheck_prefill_scatter(batch)
    following = replace(old, confirmed=False, published=set(), consuming=set())
    runtime.queue.append(following)
    try:
        assert runtime.resume_prefill_scatter(after_target=True, batch=batch)
        assert entered.wait(5)
        runtime.join_cpu_scatter()
        assert runtime.queue == [following] and not following.published
        assert runtime.cpu_scatter_batches == 1
    finally:
        runtime._cpu_scatter_executor.shutdown(wait=True)


def test_maintenance_still_pumps_and_zero_token_step_never_needs_later_target():
    runtime, output, batch, old = fixture()
    calls = []
    refresh = NS(jobs={'other': object()}, pump=lambda peers: calls.append(('pump', peers)))
    assert not runtime.defer_prefill_scatter(output, refresh)
    runtime.start_cpu_scatter = lambda: calls.append('start')
    runtime.drain_requests = lambda ids, **kw: calls.append(('drain', ids))
    runtime.advance_publication({}, refresh)
    assert calls == ['start', ('drain', set()), ('drain', set()), ('pump', ())]


def test_unconsumed_intent_fails_closed_instead_of_deferring_another_step():
    runtime, output, batch, old = fixture()
    assert runtime.defer_prefill_scatter(output)
    with pytest.raises(RuntimeError, match='escaped'):
        runtime.defer_prefill_scatter(output)
    assert runtime._publication_error is not None and runtime._prefill_scatter_deferred is None
    assert old.download.owners and not old.published


def test_prepared_fallback_synchronous_publication_consumes_intent_as_no_work():
    runtime, output, batch, old = fixture()
    assert runtime.defer_prefill_scatter(output)
    # Planner's existing owner drain already published the predecessor before
    # noticing that this batch must return to the ordinary synchronous path.
    runtime.queue.clear()
    runtime.action.native_plans = None
    runtime.recheck_prefill_scatter(batch)
    assert runtime._prefill_scatter_deferred is None
    assert runtime.prefill_scatter_after_target_stats['no_work_batches'] == 1
    assert getattr(runtime, '_cpu_scatter_executor', None) is None


@pytest.mark.parametrize('connector_failure', [False, True])
def test_complete_runner_entry_confirm_defer_and_connector_exception_guard(tmp_path, monkeypatch, connector_failure):
    import vllm.distributed.kv_transfer as transfer
    runner, runtime, output, batch, old, calls = runner_fixture()
    # runner_fixture qualifies the lower hook directly. Start the full entry
    # fresh so the real execute_model admission owns this test's intent.
    runtime._prefill_scatter_deferred = None
    runtime.prefill_scatter_after_target_stats.clear()
    runner._kvmem_probe_output = tmp_path
    (tmp_path / 'armed').touch()
    runner._kvmem_probe_done, runner._kvmem_continuous = True, False
    runner._kvmem_bounded_target = True
    runner._kvmem_query_refresh = lambda *a: None
    runner._kvmem_window_adapter.free = lambda *a: None
    runner._kvmem_sessions.receive = lambda *a: calls.append('sessions')
    runner._kvmem_public_work.poll = lambda **kw: None
    runtime.begin_step = lambda *a: calls.append('confirm')
    runtime.defer_terminals = lambda value: value
    runtime.finish = lambda *a: calls.append('finish')
    def connector():
        if connector_failure:
            assert runtime._prefill_scatter_deferred is not None
            raise OSError('connector setup failed before target wrapper')
        return NS()
    monkeypatch.setattr(transfer, 'get_kv_transfer_group', connector)
    if connector_failure:
        with pytest.raises(OSError, match='connector setup'):
            runner.execute_model(output)
        assert calls == ['confirm', 'sessions', 'finish']
        assert runtime._publication_error is not None
        assert runtime.prefill_scatter_after_target_stats['quarantined_batches'] == 1
        assert not old.published and old.download.owners
    else:
        assert runner.execute_model(output) == 'queued'
        assert calls == ['confirm', 'sessions', 'finish', 'target_enter', 'prepare',
                         'embedding', 'first_fa2', 'last_layer', 'work', 'scatter']
        assert runtime.prefill_scatter_after_target_stats['after_target_started_batches'] == 1
    assert runtime._prefill_scatter_deferred is None


def test_actual_archive_publication_keeps_two_slot_fifo_and_next_confirmation(tmp_path):
    from test_request_publication import fixture as archive_fixture, packet, close
    runtime, a, b, archive = archive_fixture(tmp_path)
    template, output, batch, _ = fixture()
    runtime.options = NS(**{**vars(runtime.options), 'prefill_scatter_after_target': True,
                            'prefill_publication_pipeline': True})
    template.runner.vllm_config = runtime.runner.vllm_config
    runtime.runner, runtime.action = template.runner, template.action
    runtime.action.runtime = runtime
    a.action_history = template.authorities['a'].action_history
    runtime.action.prefill_plan_frames = {'a': PrefillPlanFrame(a, a.action_history,
        (0, 0, 0, 0, None), 8, 8, 8, ())}
    # No Action observation is attached to these archive-only packets. The
    # existing Action/graph tests independently qualify those transactions.
    runtime.action.accept = lambda *args: None
    output.num_scheduled_tokens = {'a': 4}
    output.total_num_scheduled_tokens = 4
    output.scheduled_cached_reqs.num_computed_tokens = [8]
    batch.num_computed_tokens_np[0] = 8
    batch.query_start_loc_np[:] = [0, 4]
    plan = runtime.action.prefill_plans[1]['a']
    plan.update(history_begin=8, local_begin=8, count=4)
    old = packet(runtime, [('a', 4, 4)])
    old.pipeline_requests = frozenset(('a',))
    try:
        assert runtime.defer_prefill_scatter(output)
        runtime.advance_publication(output.num_scheduled_tokens, defer_scatter=True)
        assert a.history_end == 4 and old.download.owners and not old.published
        runtime.recheck_prefill_scatter(batch)
        following = packet(runtime, [('a', 8, 4)], confirmed=False)
        following.pipeline_requests = frozenset(('a',))
        assert runtime.publication_blocked_requests() == {'a'}
        assert runtime.resume_prefill_scatter(after_target=True, batch=batch)
        runtime.join_cpu_scatter()
        assert a.history_end == 8 and runtime.released == [old.download]
        assert runtime.queue == [following] and following.download.owners and not following.published
        assert len(runtime.downloads.pending) == 1 and len(runtime.downloads.buffers) == 2
        runtime.begin_step()
        assert runtime.drain(wait=True)
        assert a.history_end == 12 and runtime.released == [old.download, following.download]
        assert [step['history_begin'] for step in runtime.last_steps['a']] == [4, 8]
        assert runtime.archive_capacity()['a'].history_end == 12
    finally:
        close(runtime, (a, b), archive)
