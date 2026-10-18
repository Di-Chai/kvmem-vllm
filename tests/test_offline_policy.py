from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.offline_policy import OfflineAgentPolicy, ServiceCosts
from vllm.v1.core.sched.request_queue import FCFSRequestQueue
from vllm_kvmem.session_control import SessionLedger


def request(rid, prompt=512, computed=0, arrival=0.):
    return NS(request_id=rid, num_prompt_tokens=prompt, num_computed_tokens=computed,
        num_in_flight_tokens=0, next_decode_eligible_step=0, is_finished=lambda:False,
        arrival_time=arrival, status='ready', kv_transfer_params={}, session_id=rid,
        cache_salt='tenant', prompt_token_ids=list(range(prompt)), max_tokens=16)


def fixture():
    now = [10.]
    ledger = SessionLedger(8, 4, lambda *a:None, lambda *a:None, lambda *a:None)
    s = NS(running=[], waiting=FCFSRequestQueue(), skipped_waiting=FCFSRequestQueue(),
        current_step=0, transactions={}, window_managers={0:NS(offsets={})},
        connector=NS(sessions=ledger, active_residency=NS(entries={}),
            history_admission=NS(can_admit=lambda r:True)),
        kv_cache_manager=NS(admission_preview=lambda r:None),
        query_policy=NS(capacity=262144), _is_blocked_waiting_status=lambda status:status=='blocked')
    options = NS(maintenance_aging_ms=250, prefill_max_tokens=2048)
    return s, OfflineAgentPolicy(s, options, None, clock=lambda:now[0]), now


def calibrated_prefill(costs):
    costs.observe_steps([dict(sequence=i, model_ms=10., prefill_requests=1,
        decode_requests=0, prefill_tokens=100, prefill_context_bucket=1024) for i in range(1,4)])


def test_ready_decode_precedes_prefill_without_unblocking_a_restore():
    s,p,now=fixture()
    prefill, suspended, decode = request('prefill'), request('restore'), request('decode',128,128)
    s.running = [prefill,suspended,decode]
    s.transactions['restore'] = object()
    p.prepare()
    assert s.running == [decode,prefill,suspended]
    assert p.rows['restore'].reason == 'completion_fence'
    assert suspended.num_computed_tokens == 0 and suspended in s.running
    assert p.reorders == 1


def test_unknown_costs_keep_fifo_and_do_not_create_artificial_zero_time_work():
    s,p,_=fixture()
    long,short=request('long',800,arrival=1),request('short',32,arrival=2)
    s.waiting.extend([long,short]); p.prepare()
    assert list(s.waiting) == [long,short]
    assert all(r.estimated_ms is None for r in p.rows.values())
    assert not s.connector.sessions.entries


def test_measured_short_work_wins_then_aged_long_work_gets_a_turn():
    s,p,now=fixture(); calibrated_prefill(p.costs)
    long,short=request('long',800,arrival=1),request('short',32,arrival=2)
    s.waiting.extend([long,short]); p.prepare()
    assert list(s.waiting) == [short,long]
    s.waiting.pop_request()
    now[0] += .3
    newer = request('newer',16,arrival=3); s.waiting.add_request(newer)
    p.prepare()
    assert list(s.waiting) == [long,newer]
    assert p.rows['long'].waiting_seconds == pytest.approx(.3)


def test_native_skipped_queue_does_not_put_a_blocked_request_ahead_of_ready_work():
    s,p,_=fixture()
    blocked, ready = request('blocked',arrival=0),request('ready',arrival=1)
    blocked.status = 'blocked'
    s.skipped_waiting.add_request(blocked); s.waiting.add_request(ready)
    p.prepare()
    assert p.waiting_queue() is s.waiting
    s.waiting.pop_request()
    assert p.waiting_queue() is s.skipped_waiting  # Native may promote it now.
    assert blocked.status == 'blocked'


def test_admission_rechecks_credits_after_a_peer_allocates_in_the_same_step():
    s,p,_=fixture()
    large,small=request('large',800,arrival=0),request('small',32,arrival=1)
    s.waiting.extend([large,small]);p.prepare()
    assert s.waiting.peek_request() is large
    s.kv_cache_manager.admission_preview=lambda r:'gpu_lifetime_budget' if r is large else None
    assert p.waiting_queue().peek_request() is small
    assert p.rows['large'].reason=='gpu_lifetime_budget'


def test_cpu_and_gpu_constraints_are_read_only_ordering_hints():
    s,p,_=fixture()
    cpu,gpu,ready=[request(rid,arrival=i) for i,rid in enumerate(('cpu','gpu','ready'))]
    s.connector.history_admission.can_admit=lambda r:r is not cpu
    s.kv_cache_manager.admission_preview=lambda r:'gpu_lifetime_budget' if r is gpu else None
    s.waiting.extend([cpu,gpu,ready]);p.prepare()
    assert s.waiting.peek_request() is ready
    assert p.rows['cpu'].reason=='cpu_history_budget'
    assert p.rows['gpu'].reason=='gpu_lifetime_budget'
    assert not s.connector.sessions.requests


def test_exact_checkpoint_hint_never_changes_session_or_bypasses_prefix_identity():
    from test_session_control import first_turn
    s,p,_=fixture()
    binding,notice=first_turn(s.connector.sessions)
    entry=s.connector.sessions.entries[binding.namespace]
    assert entry.phase=='idle'
    s.connector.active_residency.entries[entry.namespace]=NS(
        generation=entry.generation,history=entry.history,window=entry.window,expires=11.)
    new=request('next',32);new.session_id='s'
    before=deepcopy(entry)
    history,window,mode,reason=p._hint(new)
    assert (history,window,mode,reason)==(entry.history,entry.window,'resident',None)
    assert entry==before and 'next' not in s.connector.sessions.requests
    s.connector.active_residency.entries[entry.namespace].expires=9.
    assert p._hint(new)==(entry.history,entry.window,'cpu',None)
    assert entry.namespace in s.connector.active_residency.entries  # Preview does not free native pages.
    changed=request('changed',32);changed.session_id='s';changed.prompt_token_ids[0]=9876
    assert p._hint(changed)==(0,0,'unknown',None)
    changed.cache_salt='another-tenant'
    assert p._hint(changed)==(0,0,'none',None)
    assert entry==before


def test_cost_model_excludes_fixed_maintenance_and_requires_matching_decode_context():
    costs=ServiceCosts()
    for i in range(1,4):
        costs.observe_steps([dict(sequence=i,model_ms=20.,prefill_tokens=0,
            prefill_requests=0,decode_requests=2,decode_context_bucket=1024)])
    for i in range(4,7):
        costs.observe_steps([dict(sequence=i,model_ms=40.,maintenance_ms=2000.,prefill_tokens=100,
            prefill_requests=1,decode_requests=2,decode_context_bucket=1024,prefill_context_bucket=1024)])
    assert costs.estimate(200,1024,'none',0)==pytest.approx(40.)
    costs.observe_steps([dict(sequence=7,model_ms=1000.,prefill_tokens=100,
        prefill_requests=1,decode_requests=2,decode_context_bucket=2048,prefill_context_bucket=4096)])
    assert costs.estimate(200,4096,'none',0) is None
    for _ in range(3):costs.observe_restores([dict(service_seconds=.1,active_tokens=1024,resident=True)])
    assert costs.estimate(200,1024,'resident',1024)==pytest.approx(140.)
    assert costs.estimate(200,1024,'cpu',1024) is None


def test_cpu_preview_does_not_accumulate_waits_and_native_gpu_preview_counts_growth():
    from test_coordination import request as cpu_request
    from vllm_kvmem.coordination import HistoryAdmission
    from test_capacity import make_gate,request as gpu_request
    cpu=HistoryAdmission(208,4,1,100)
    cpu.allocated(cpu_request('a'),1);cpu.allocated(cpu_request('b'),2)
    before=cpu.snapshot()
    assert not cpu.can_admit(cpu_request('large',16))
    assert cpu.snapshot()==before
    gpu,manager,pool=make_gate()
    a,b,c=[gpu_request(x) for x in 'abc']
    gpu.allocate_slots(a,16);gpu.allocate_slots(b,16)
    before=pool.get_num_free_blocks()
    assert gpu.admission_preview(c)=='gpu_lifetime_budget'
    assert pool.get_num_free_blocks()==before and c.request_id not in gpu.reservations
    gpu.free(a);gpu.free(b)


def test_worker_uses_the_same_measured_readiness_costs_and_fifo_when_incomplete():
    from test_maintenance import policy,job
    p=policy(offline_agent_policy=True)
    old,new=job('old',32),job('new',8)
    old.started=9.99;new.started=10.
    p.readiness_costs={'old':100.,'new':10.}
    assert p.order([old,new])==[new,old]
    p.readiness_costs={'old':100.,'new':None}
    assert p.order([old,new])==[old,new]
    old.started=9.
    p.readiness_costs={'old':100.,'new':10.}
    assert p.order([new,old])==[old,new]


def test_active_context_cost_bucket_uses_dense_window_not_full_cpu_history():
    from vllm_kvmem.step_timing import StepTiming
    class Event:
        def record(self):pass
    t=StepTiming(1,None,event_factory=Event,clock=lambda:10.)
    t.begin()
    t.model_begin(NS(req_ids=['long','decode'],num_computed_tokens_np=[900000,4096],
        num_scheduled_tokens=[512,5],is_prefilling_np=[True,False]),{'long':898000})
    assert t.active['prefill_context_bucket']==2048
    assert t.active['decode_context_bucket']==4096


def test_restore_only_feedback_crosses_connector_without_waiting_for_a_model_step():
    from test_connector import pair,poll
    scheduler,worker=pair();seen=[]
    scheduler.scheduler_restore_feedback=seen.extend
    rows=[dict(service_seconds=.1,active_tokens=1024,resident=True)]
    worker.restore_timings=lambda:list(rows)
    assert not poll(scheduler,worker).is_empty() and seen==rows


def test_unified_policy_rejects_competing_controllers_before_model_start():
    from vllm_kvmem.options import RuntimeOptions
    from vllm_kvmem.config import KVMemConfig
    settings=dict(memory=KVMemConfig(),offline_agent_policy=True,cpu_history_coordination=True,
        async_restore=True,active_resident_ttl_ms=5000,step_latency_interval=4,
        async_refresh=True,maintenance_max_steps=8,archive_backend='cpu',request_planning=True,
        async_archive_retirement=True,session_capacity=32,session_checkpoint_bytes=4*1024**3,
        cpu_read_cache_bytes=100*1024**3,session_capture_terminal=True,session_checkpoint_policy='pm')
    RuntimeOptions(**settings)
    shared = RuntimeOptions(**(settings | dict(prefill_batch_tokens=2048)))
    assert shared.offline_agent_policy and shared.prefill_batch_tokens == 2048
    pressure = settings | dict(active_resident_ttl_ms=0, active_resident_pressure_only=True)
    assert RuntimeOptions(**pressure).active_residency_enabled
    with pytest.raises(ValueError, match='mutually exclusive'):
        RuntimeOptions(**(settings | dict(active_resident_pressure_only=True)))
    for change in (dict(active_resident_pressure_only=1), dict(active_resident_pressure_only=False),
                   dict(async_restore=False), dict(session_capture_terminal=False)):
        with pytest.raises(ValueError):
            RuntimeOptions(**(pressure | change))
    for changes in (dict(adaptive_prefill=True),
                    dict(adaptive_prefill=True, prefill_batch_tokens=2048),
                    dict(maintenance_transfer_bytes_per_second=1),dict(prefill_cadence=2)):
        with pytest.raises(ValueError,match='independent latency'):
            RuntimeOptions(**(settings|changes))


def test_efficiency_gate_requires_agent_improvement_and_pure_decode_evidence():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('agent_sequence',Path(__file__).parents[1]/'scripts/run_agent_policy_sequence.py')
    sequence=importlib.util.module_from_spec(spec);spec.loader.exec_module(sequence)
    row=dict(pairs=3,throughput_ratio_median=1.2,throughput_ratios=[1.2,1.2,1.2])
    results={'P1/agents':dict(row),'P1/mixed':dict(row),'P1/decode':dict(row,
        common_decode_ratios=[1.,1.,1.],common_decode_ratio_median=1.)}
    assert sequence.efficiency_gate(results)['passed']
    results['P1/decode']['common_decode_ratio_median']=.90
    assert not sequence.efficiency_gate(results)['passed']
    results['P1/decode']['common_decode_ratio_median']=1.
    results['P1/agents']['throughput_ratios']=[.95,1.2,1.2]
    assert not sequence.efficiency_gate(results)['passed']
