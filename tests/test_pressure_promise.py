"""KVMEM_LAYERWISE_PRESSURE_PROMISE: ACK before the plan, exact plans afterwards."""
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
import threading

import pytest

from vllm_kvmem.action_attention import ActionRuntime
from vllm_kvmem.logical_action import plan_logical_scores
from vllm_kvmem.logical_prefill import LogicalPrefillGroup, LogicalPrefillSpan
from vllm_kvmem.prefill_planning import PrefillPlanFrame, freeze_prefill_frame
from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem import pressure_promise

BLOCK, EXTENT, HISTORY = 32, 36864, 81920


def options():
    return NS(prefill_attention_stride=32, prefill_attention_interval=1024,
              memory=NS(select_budget=EXTENT, gen_budget=4096, block_tokens=BLOCK))


def records(seed=0):
    # A real selected prefix: arbitrary, non-contiguous history pages.
    import random
    pages = sorted(random.Random(seed).sample(range(HISTORY // BLOCK), EXTENT // BLOCK))
    return tuple((1000+i, 3, page*BLOCK, BLOCK, i*BLOCK) for i, page in enumerate(pages))


def frames(sampled_end=HISTORY):
    state = NS(options=options(), prefill_replay=None, prefill=NS(end=HISTORY))
    identity = (0, 1, HISTORY-EXTENT, 0, None)
    exact = freeze_prefill_frame('auth', state, identity, HISTORY, EXTENT, records())
    exact = replace(exact, sampled_end=HISTORY)
    synthetic = PrefillPlanFrame('auth', state, identity, HISTORY, EXTENT, sampled_end,
        tuple((-2, -2, ActionRuntime.PROMISE_KEY_BASE + i*BLOCK, BLOCK, i*BLOCK)
              for i in range(EXTENT // BLOCK)))
    return exact, synthetic


def group(chunks=4):
    spans = tuple(LogicalPrefillSpan(i, HISTORY+i*1024, EXTENT+i*1024, 1024, i*1024) for i in range(chunks))
    return LogicalPrefillGroup(request_id='r', owner_epoch=(0, 1, HISTORY-EXTENT, 0, None), scheduler_step=9,
        offset=HISTORY-EXTENT, prompt_tokens=HISTORY+1_000_000, capacity=EXTENT+4096, spans=spans,
        request_plan=RequestPlan(None, block_tokens=BLOCK, gen_budget=4096), coalescing_authorization_id=None)


def test_geometry_frame_plans_the_same_rows_and_segments_as_the_exact_frame():
    exact, synthetic = frames()
    a = exact.extend(HISTORY, EXTENT, 1024, BLOCK)
    b = synthetic.extend(HISTORY, EXTENT, 1024, BLOCK)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]          # rows, starts, counts
    assert a[3][EXTENT//BLOCK:] == b[3][EXTENT//BLOCK:]            # new-chunk pages are real in both
    assert a[3][:EXTENT//BLOCK] != b[3][:EXTENT//BLOCK]            # prefix keys are placeholders
    assert all(k >= ActionRuntime.PROMISE_KEY_BASE for k in b[3][:EXTENT//BLOCK])


def test_logical_groups_validate_with_placeholders_and_match_exact_geometry():
    exact, synthetic = frames()
    pa, ra, na = plan_logical_scores(group(), exact)
    pb, rb, nb = plan_logical_scores(group(), synthetic)
    assert ra == rb and len(pa) == len(pb) == 4
    for x, y in zip(pa, pb):
        assert (x.span, x.rows, x.starts, x.counts) == (y.span, y.rows, y.starts, y.counts)
        assert x.history != y.history
    assert na.records[EXTENT//BLOCK:] == nb.records[EXTENT//BLOCK:]


def test_wrong_sampled_end_changes_rows_and_is_detectable():
    # A mispredicted sampled end inside the interval suppresses the first
    # span's rows; the capture-time comparison must see a different plan.
    exact, synthetic = frames(sampled_end=HISTORY+512)
    a = exact.extend(HISTORY, EXTENT, 1024, BLOCK)
    b = synthetic.extend(HISTORY, EXTENT, 1024, BLOCK)
    assert a[0] and not b[0]
    _, ra, _ = plan_logical_scores(group(), exact)
    _, rb, _ = plan_logical_scores(group(), synthetic)
    assert ra != rb


def job(**changes):
    ticket = NS(window_tokens=EXTENT, replay_tokens=0, original_block_ids=(1, 2))
    command = NS(request_id='r', selection='pressure', pressure_stage='prefill', guided_reason='',
                 query_span=None, input_tokens=(), groups=((0, ticket),), command_id=5)
    j = NS(command=command, done=False, cancelled=False, stage='prepare', guided_pressure=False,
           history=HISTORY, selected_tokens=None, future=None, authority=None)
    for k, v in changes.items():
        setattr(j, k, v)
    return j


def refresh_fixture(history_end=HISTORY - 4096):
    authority = NS(history_end=history_end, block_graph=None, query_history=None, memory=NS())
    runner = NS(req_states=NS(req_id_to_index={'r': 0}), _kvmem_sessions=None,
                _kvmem_capture_runtime=NS(authorities={'r': authority}),
                _kvmem_window_adapter=NS(commits=[], commit=lambda *a: runner._kvmem_window_adapter.commits.append(a)),
                device='cpu')
    refresh = NS(runner=runner, options=NS(memory=options().memory), maintenance=None, emitted=[], jobs={},
                 emit=lambda j, e, **f: refresh.emitted.append(e), stream=object(),
                 _ensure_window_adapter=lambda c: None)
    return refresh, authority


def pipeline_fixture():
    from vllm_kvmem.pressure_pipeline import PressurePipeline
    p = PressurePipeline.__new__(PressurePipeline)
    p.active, p.installed, p.names = {}, True, ('a', 'b', 'mtp')
    p.pressure = NS(lanes=1, single_assembly=True)
    return p


@pytest.mark.parametrize('change,reason', [
    (dict(done=True), 'command'), (dict(stage='assembly'), 'command'),
    (dict(guided_pressure=True), 'command'), (dict(history=EXTENT-32), 'extent'),
    (dict(history=HISTORY+16), 'extent')])
def test_ineligible_jobs_take_the_original_path(change, reason):
    refresh, authority = refresh_fixture(history_end=0)
    p = pipeline_fixture()
    assert pressure_promise.ineligible(p, refresh, job(**change), refresh.runner._kvmem_capture_runtime) == reason


def test_busy_lane_graph_and_query_history_decline():
    refresh, authority = refresh_fixture()
    capture = refresh.runner._kvmem_capture_runtime
    p = pipeline_fixture()
    p.active['other'] = object()
    assert pressure_promise.ineligible(p, refresh, job(), capture) == 'lane_busy'
    p.active.clear()
    authority.block_graph = object()   # Graph metadata does not affect pressure selection.
    assert pressure_promise.ineligible(p, refresh, job(), capture) is None
    authority.query_history = object()
    assert pressure_promise.ineligible(p, refresh, job(), capture) == 'authority'
    authority.query_history = None
    assert pressure_promise.ineligible(p, refresh, job(), capture) is None


def test_promise_acknowledges_with_fixed_extent_and_registers_a_guarding_lease(monkeypatch):
    import torch
    refresh, authority = refresh_fixture()
    p = pipeline_fixture()
    submitted = []
    refresh.executor = NS(submit=lambda fn, *a: submitted.append((fn, a)) or Future())
    monkeypatch.setattr(torch.cuda, 'Event', lambda: NS(record=lambda *a: None))
    monkeypatch.setattr(pressure_promise, 'ineligible', lambda *a: None)
    import vllm_kvmem.async_refresh as ar
    monkeypatch.setattr(ar, 'stored_expansions', lambda tables: {0: 1})
    refresh.runner.block_tables = NS(num_blocks=NS(np={(0, 0): 2}),
        block_tables={0: NS(gpu={(0, slice(None, 2)): None})})
    # Page reservation check: replace with an exact table.
    class Table:
        def __getitem__(self, key):
            return NS(cpu=lambda: NS(tolist=lambda: [1, 2]))
    refresh.runner.block_tables.block_tables = {0: NS(gpu=Table())}
    refresh.runner.block_tables.num_blocks = NS(np={(0, 0): 2})
    j = job()
    assert pressure_promise.promise(p, refresh, j)
    lease = p.active['r']
    assert j.done and j.selected_tokens == EXTENT and lease.promised and lease.plan is None
    assert j.promised_lease is lease and not lease.committed.is_set()
    assert refresh.runner._kvmem_window_adapter.commits == [('r', HISTORY, EXTENT)]
    assert submitted and submitted[0][0] is pressure_promise.producer
    assert 'pipeline_promised' in refresh.emitted
    # The forward's guard blocks on the plan-less lease until a layer event exists.
    released = threading.Event()
    def guard():
        lease.wait_layer(0, NS(cuda_stream=1, wait_event=lambda e: None))
        released.set()
    t = threading.Thread(target=guard); t.start()
    assert not released.wait(.05)
    lease.publish_layer(0, NS(query=lambda: True))
    assert released.wait(2); t.join()


def test_begin_binds_plan_once_and_commits(monkeypatch):
    from vllm_kvmem.pressure_pipeline import LayerMaterialization
    p = pipeline_fixture()
    j = job()
    lease = LayerMaterialization(p, j, None, p.names)
    lease.promised, lease.committed, lease.promise_extent = True, threading.Event(), EXTENT
    j.promised_lease = lease; p.active['r'] = lease
    calls = []
    monkeypatch.setattr(pressure_promise, 'commit', lambda pipe, jj, l: calls.append(l.plan))
    plan = NS(total_window_tokens=EXTENT)
    assert p.begin(j, plan, p.names) is lease and calls == [plan]
    with pytest.raises(RuntimeError, match='bound twice'):
        p.begin(j, plan, p.names)


def test_commit_refuses_a_different_extent():
    from vllm_kvmem.pressure_pipeline import LayerMaterialization
    p = pipeline_fixture()
    j = job()
    lease = LayerMaterialization(p, j, NS(total_window_tokens=EXTENT-32), p.names)
    lease.promised, lease.committed, lease.promise_extent = True, threading.Event(), EXTENT
    with pytest.raises(RuntimeError, match='different extent'):
        pressure_promise.commit(p, j, lease)


def test_scheduler_requests_promise_only_when_idle(monkeypatch):
    from vllm.distributed import kv_transfer
    from vllm_kvmem.scheduler import KVMemScheduler
    calls = []
    def take(wait=0., idle=False):
        calls.append((wait, idle)); return None
    w = NS(take_schedule_completions=take)
    monkeypatch.setattr(kv_transfer, 'has_kv_transfer_group', lambda: True)
    monkeypatch.setattr(kv_transfer, 'get_kv_transfer_group', lambda: w)
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '1')
    for peers, expected in (([], (0., True)), ([NS(request_id='peer')], (0., False))):
        calls.clear()
        sched = NS(transactions={'r': object()}, connector=NS(update_connector_output=lambda o: None),
                   running=[NS(request_id='r')] + peers, _waiting_queues=lambda: ())
        KVMemScheduler._schedule_time_acks(sched)
        assert calls == [expected]


def action_fixture(monkeypatch, working):
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '1')
    state = NS(options=options(), prefill_replay=None, prefill=NS(end=HISTORY))
    authority = NS(action_history=state, memory=NS(_store=NS(working_records=lambda: working)))
    lease = NS(promised=True, committed=threading.Event(), error=None, promise_extent=EXTENT,
               job=NS(history=HISTORY, authority=authority))
    a = ActionRuntime.__new__(ActionRuntime)
    a.runner = NS(_kvmem_layerwise_pressure=NS(pipeline=NS(active={'r': lease})),
                  _kvmem_window_adapter=NS(generations={'r': 1}, offsets={'r': HISTORY-EXTENT}))
    a.runtime = NS(authorities={'r': authority}, config=NS(block_tokens=BLOCK),
                   drain_requests=lambda ids, wait: True, ensure_authority=lambda rid: authority)
    return a, authority, state, lease


def previous_frame(authority, state, sampled_end=HISTORY):
    return PrefillPlanFrame(authority, state, ('old',), HISTORY, 40960, sampled_end, ())


def promised_plan(a, logical, sampled_end=HISTORY):
    authority = a.runtime.authorities['r']
    frame, lease = a._promised_frame('r', HISTORY, 4096 if logical else 1024,
                                     previous_frame(authority, authority.action_history, sampled_end))
    plan = dict(authority=frame.authority, state=frame.state, history_begin=HISTORY, local_begin=EXTENT,
                index=0, begin=0, count=4096 if logical else 1024, pipeline=True)
    if logical:
        receipt = group()
        g = replace(receipt, owner_epoch=frame.identity)
        plans, rows, nxt = plan_logical_scores(g, frame)
        plan.update(rows=list(rows), starts=list(plans[-1].starts), counts=list(plans[-1].counts),
                    history=list(plans[-1].history), logical_group=g, logical_score_plans=plans)
    else:
        receipt = None
        rows, starts, counts, history, nxt = frame.extend(HISTORY, EXTENT, 1024, BLOCK)
        plan.update(rows=rows, starts=starts, counts=counts, history=history)
    plan.update(promise_lease=lease, promise_identity=frame.identity, promise_receipt=receipt,
                promise_next_frame=nxt)
    a.prefill_plan_frames = {'r': nxt}
    return plan, frame


@pytest.mark.parametrize('logical', [False, True])
def test_capture_resolution_equals_the_original_blocking_plan(monkeypatch, logical):
    working = records(seed=3)
    a, authority, state, lease = action_fixture(monkeypatch, working)
    plan, frame = promised_plan(a, logical)
    batch = object()
    a.prefill_plans = (batch, {'r': plan})
    lease.committed.set()
    a._resolve_promised_plans(batch)
    exact = replace(freeze_prefill_frame(authority, state, frame.identity, HISTORY, EXTENT, working),
                    sampled_end=HISTORY)
    if logical:
        g = replace(group(), owner_epoch=frame.identity)
        plans, rows, nxt = plan_logical_scores(g, exact)
        assert plan['logical_score_plans'] == plans and plan['history'] == list(plans[-1].history)
    else:
        rows, starts, counts, history, nxt = exact.extend(HISTORY, EXTENT, 1024, BLOCK)
        assert plan['history'] == list(history)
    assert a.prefill_plan_frames['r'] == nxt
    assert not any(k.startswith('promise_') for k in plan)
    assert all(k < ActionRuntime.PROMISE_KEY_BASE for k in plan['history'])


def test_capture_resolution_fails_closed_on_changed_geometry(monkeypatch):
    working = list(records(seed=4))
    a, authority, state, lease = action_fixture(monkeypatch, tuple(working))
    plan, frame = promised_plan(a, True)
    # A committed prefix with a partial page cannot match uniform geometry.
    slot, gen, begin, count, pos = working[-1]
    working[-1] = (slot, gen, begin, 16, pos)
    authority.memory._store.working_records = lambda: tuple(working)
    batch = object()
    a.prefill_plans = (batch, {'r': plan})
    lease.committed.set()
    with pytest.raises((RuntimeError, ValueError)):
        a._resolve_promised_plans(batch)


def test_geometry_frame_only_while_uncommitted_and_for_the_promised_prefix(monkeypatch):
    a, authority, state, lease = action_fixture(monkeypatch, records())
    old = previous_frame(authority, state)
    assert a._promised_frame('r', HISTORY, 1024, old) is not None
    assert a._promised_frame('r', HISTORY, 1024, None) is None            # no planning frame
    assert a._promised_frame('r', HISTORY+32, 1024, old) is None
    lease.committed.set()
    assert a._promised_frame('r', HISTORY, 1024, old) is None
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_PROMISE', '0')
    lease.committed.clear()
    assert a._promised_frame('r', HISTORY, 1024, old) is None


@pytest.mark.parametrize('sampled_end', [HISTORY, HISTORY - 160, HISTORY - 900, HISTORY - 1024])
def test_small_unsampled_chunks_take_the_sampled_end_from_the_previous_frame(monkeypatch, sampled_end):
    # A 160-token chunk is sampled only when the interval since the last
    # sampled end is reached; the prediction must reproduce the exact rows.
    working = records(seed=5)
    a, authority, state, lease = action_fixture(monkeypatch, working)
    state.prefill.end = sampled_end
    frame, _ = a._promised_frame('r', HISTORY, 160, previous_frame(authority, state, sampled_end))
    exact = replace(freeze_prefill_frame(authority, state, frame.identity, HISTORY, EXTENT, working),
                    sampled_end=sampled_end)
    a_rows = frame.extend(HISTORY, EXTENT, 160, BLOCK)
    b_rows = exact.extend(HISTORY, EXTENT, 160, BLOCK)
    assert a_rows[:3] == b_rows[:3]


def test_publication_identity_keeps_the_pre_promise_window_for_older_chunks():
    a = ActionRuntime.__new__(ActionRuntime)
    adapter = NS(generations={'r': 7}, offsets={'r': HISTORY-EXTENT})
    a.runner = NS(_kvmem_window_adapter=adapter)
    state = NS(prefill_planning_epoch=0, prefill_replay=None)
    before = a._prefill_identity('r', state)
    a.promise_epochs = {'r': (HISTORY, 7, HISTORY-EXTENT)}
    adapter.generations['r'], adapter.offsets['r'] = 8, HISTORY + 4096 - EXTENT   # promised commit
    after = a._prefill_identity('r', state)
    assert after != before
    assert a._publication_identity('r', state, HISTORY) == before          # chunk that filled the window
    assert a._publication_identity('r', state, HISTORY - 1024) == before
    assert a._publication_identity('r', state, HISTORY + 4096) == after    # first chunk after the refresh
    a.promise_epochs = {}
    assert a._publication_identity('r', state, HISTORY) == after


def test_store_reading_planners_wait_for_the_promised_commit(monkeypatch):
    a, authority, state, lease = action_fixture(monkeypatch, records())
    order = []
    def commit_later():
        import time
        time.sleep(.05); order.append('commit'); lease.committed.set()
    t = threading.Thread(target=commit_later); t.start()
    a._await_promised_commit('r')
    order.append('read')
    t.join()
    assert order == ['commit', 'read']
    a._await_promised_commit('other')   # No lease: no wait.
