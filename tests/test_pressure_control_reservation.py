"""A nearly full restored tail must still reach its private pressure replay."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip('vllm')
from vllm.v1.request import RequestStatus
from vllm.v1.core.sched.output import SchedulerOutput
from vllm_kvmem.request_plan import RequestPlan
from vllm_kvmem.speculative_admission import SpeculativeQueryPolicy
from vllm_kvmem.window_manager import RefreshRequired
from test_capacity import make_gate
from test_scheduler_transactions import setup_scheduler


def prepared(monkeypatch, tmp_path):
    # Scaled version of DeepSWE H=98254 -> 98272, K+G=98304:
    # the old tail is beyond the new prefill policy's two-block boundary.
    monkeypatch.setenv('KVMEM_REQUEST_PLAN_CONFIG', '{"rolling_active_window":true}')
    monkeypatch.setenv('KVMEM_RUNNER_PROBE_DIR', str(tmp_path))
    gate, manager, pool = make_gate(17)
    calls = []
    def allocate(request, count, reserved_blocks=0, num_lookahead_tokens=0):
        history = request.num_computed_tokens
        end = history + count
        calls.append(num_lookahead_tokens)
        needed = manager.get_num_blocks_to_allocate(request.request_id,
            end + num_lookahead_tokens, (), history, history, end)
        if needed > pool.get_num_free_blocks() - reserved_blocks:
            return None
        return manager.allocate_new_blocks(request.request_id, end + num_lookahead_tokens, end)
    gate.native.allocate_slots = allocate
    plan = replace(RequestPlan.parse({}, 80, 16, 32), rolling_active_window=True)
    plan, _ = plan.continuation(80, 32, 39, 39, 64)
    assert plan.prefill_window == 'pressure' and plan.admit(39, 32, 80, 0, 64) == 9
    request = SimpleNamespace(request_id='a', num_computed_tokens=0, num_prompt_tokens=80,
        max_tokens=32, status=RequestStatus.RUNNING, num_in_flight_tokens=0,
        next_decode_eligible_step=7, all_token_ids=list(range(80)), is_finished=lambda:False,
        kv_transfer_params={'kvmem_request_plan':plan.payload()})
    gate.allocate_slots(request, 39)
    request.num_computed_tokens = 39
    gate.allocate_slots(request, 9, num_lookahead_tokens=4)
    request.num_computed_tokens = 48
    scheduler, _, _ = setup_scheduler()
    scheduler._bounded_target = scheduler.async_refresh = True
    scheduler.prepared_refreshes = {}
    scheduler.window_managers = {0:manager}
    scheduler.kv_cache_manager = gate
    scheduler.query_policy = SpeculativeQueryPolicy(16,32,32,5)
    scheduler.requests = {'a':request, 'b':SimpleNamespace(request_id='b',
        num_computed_tokens=0, num_prompt_tokens=80)}
    gate.prefill_refresh_control = scheduler._prefill_refresh_control
    return scheduler, request, gate, manager, calls


def test_real_trace_boundary_and_full_public_mtp_after_private_replay(monkeypatch,tmp_path):
    scheduler, request, gate, manager, calls = prepared(monkeypatch,tmp_path)
    # The original native lookahead produces the observed fatal exception.
    callback = gate.prefill_refresh_control
    gate.prefill_refresh_control = None
    with pytest.raises(RefreshRequired):
        gate.allocate_slots(request,16,num_lookahead_tokens=4)
    gate.prefill_refresh_control = callback
    quota = gate.reservations['a'][1]
    gate.begin_step()
    assert gate.allocate_slots(request,16,num_lookahead_tokens=4) is not None
    assert calls[-1] == 0 and gate.reservations['a'][1] == quota
    assert gate.refresh_control_allocations
    out = replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'a':16,'b':1},
                  total_num_scheduled_tokens=17)
    command, = scheduler._build_kv_connector_meta(scheduler.connector,out).commands
    assert out.num_scheduled_tokens == {'b':1} and out.total_num_scheduled_tokens == 1
    assert command.input_tokens == tuple(range(48,64)) and command.selection == 'pressure'
    assert request.num_computed_tokens == 48 and not gate.refresh_control_allocations
    gate.end_step()
    scheduler._complete_refresh('a',command.command_id,selected_tokens=32)
    assert manager.offsets['a'] == 16
    assert scheduler.prepared_refreshes == {'a':(48,16)}
    assert gate.allocate_slots(request,16,num_lookahead_tokens=4) is not None
    assert calls[-1] == 4 and not gate.refresh_control_allocations
    replay = replace(SchedulerOutput.make_empty(),num_scheduled_tokens={'a':16},
                     total_num_scheduled_tokens=16)
    assert not scheduler._build_kv_connector_meta(scheduler.connector,replay).commands
    assert replay.num_scheduled_tokens == {'a':16}


@pytest.mark.parametrize('kind',['public_escape','missing_transaction','wrong_replay'])
def test_no_lookahead_control_cannot_escape_verification(monkeypatch,tmp_path,kind):
    scheduler, request, gate, _, _ = prepared(monkeypatch,tmp_path)
    gate.allocate_slots(request,16,num_lookahead_tokens=4)
    if kind == 'missing_transaction':
        transactions = {}
    else:
        scheduler.begin_refresh(request,32,replay_tokens=16,selection='pressure',
            input_tokens=tuple(range(48,64)),asynchronous=True)
        transactions = dict(scheduler.transactions)
        if kind == 'wrong_replay':
            command, step = transactions['a']
            transactions['a'] = (replace(command,input_tokens=(1,)),step)
    with pytest.raises(RuntimeError,match='escaped'):
        gate.validate_refresh_controls({'a':16} if kind == 'public_escape' else {},transactions)
    assert gate.refresh_control_allocations
    with pytest.raises(RuntimeError,match='Unverified'):
        gate.begin_step()


@pytest.mark.parametrize('kind',['synchronous','prepared','inflight','waiting','decode','query','unaligned'])
def test_only_exact_quiescent_prefill_pressure_is_eligible(monkeypatch,tmp_path,kind):
    scheduler, request, _, _, _ = prepared(monkeypatch,tmp_path)
    assert scheduler._prefill_refresh_control(request,16)
    if kind == 'synchronous':scheduler.async_refresh = False
    elif kind == 'prepared':scheduler.prepared_refreshes['a'] = (48,16)
    elif kind == 'inflight':request.num_in_flight_tokens = 1
    elif kind == 'waiting':request.status = RequestStatus.WAITING
    elif kind == 'decode':request.num_prompt_tokens = 48
    elif kind == 'query':request.kv_transfer_params['kvmem_request_plan']={'kvmem_query_span':[48,60]}
    elif kind == 'unaligned':request.num_computed_tokens = 47
    assert not scheduler._prefill_refresh_control(request,16)
