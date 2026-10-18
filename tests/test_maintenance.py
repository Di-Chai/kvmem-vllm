from types import SimpleNamespace

import pytest

from vllm_kvmem.maintenance import MaintenancePolicy


def policy(clock=lambda: 10., **changes):
    options = dict(maintenance_max_steps=8, maintenance_busy_ms=2., maintenance_idle_ms=8.,
        maintenance_aging_ms=250., maintenance_inflight_bytes=0, memory=SimpleNamespace(select_budget=32,block_tokens=4))
    return MaintenancePolicy(SimpleNamespace(**(options | changes)), 3, clock=clock)


def job(rid, tokens=16, age=0., stage='prepare'):
    return SimpleNamespace(command=SimpleNamespace(request_id=rid, groups=((0, SimpleNamespace(window_tokens=tokens)),)),
        history=64, started=10.-age, last_progress=None, done=False, cancelled=False, steps=0, stage=stage)


def test_cost_ordering_ages_into_fairness_and_reaps_completions_first():
    p = policy()
    big, small, old, completing = job('big',32), job('small',8), job('old',32,1.), job('done',stage='assembly')
    assert p.order([big,small,old,completing]) == [completing,old,small,big]
    old.last_progress = 10.
    assert p.order([old,small])[0] is small


def test_busy_batch_gets_one_private_forward_but_chains_host_transitions():
    p = policy(); j = job('a'); events=[]
    def step(j):
        events.append(j.stage)
        if j.stage == 'prepare': j.stage = 'query'
        elif j.stage == 'query': j.steps += 1; j.stage = 'guided_decode'
        else: j.steps += 1
        return True
    p.pump([j], step, busy=True)
    assert events == ['prepare','query'] and p.forward_quanta == 1
    p.pump([j], step, busy=False)
    assert p.forward_quanta == 9  # Idle GPU can consume the bounded step budget.


def test_unready_owner_does_not_block_peer_or_spin_on_futures():
    p=policy(); a,b=job('a',stage='assembly'),job('b'); calls=[]
    def step(j):
        calls.append(j.command.request_id)
        if j is a: return False
        j.done=True
        return True
    p.pump([a,b],step,busy=True)
    assert calls == ['a','b','a'] and p.advances == 1


def test_soft_time_budget_yields_after_nonpreemptible_step():
    now=[10.]; p=policy(clock=lambda:now[0]); j=job('a')
    def step(j): now[0]+=.003; return True
    p.pump([j],step,busy=True)
    assert p.advances == 1 and p.time_yields == 1


def test_transfer_credit_is_exclusive_until_owner_fence_is_committed():
    p=policy(); a,b=job('a',32),job('b',16)
    assert p.acquire(a) and p.inflight_bytes == 120
    assert not p.acquire(b)
    with pytest.raises(ValueError, match='Foreign'):
        p.release(b)
    p.release(a)
    assert p.acquire(b) and p.inflight_bytes == 72
    with pytest.raises(ValueError, match='complete active view'):
        policy(maintenance_inflight_bytes=95)


def test_transfer_observations_affect_priority_but_never_reduce_reserved_bytes():
    p=policy(); a,b=job('a',32),job('b',16)
    a.authority=SimpleNamespace(memory=SimpleNamespace(request_id='retained-session'))
    p.observe(a,12)
    assert p.order([a,b])[0] is a
    assert p.acquire(a) and p.inflight_bytes == 120
    a.last_progress=8.
    assert p.order([b,a])[0] is a  # An aged request wins even without cost advantage.


def test_async_refresh_cancel_keeps_transfer_credit_until_cuda_completion():
    from test_async_refresh import manager, command
    obj=manager(); obj.maintenance=p=policy()
    first=obj.enqueue(command('a')); second=obj.enqueue(command('b'))
    for j in (first,second):
        j.command.groups[0][1].window_tokens=32
        j.stage='assembly_wait'; j.assembly_inputs=('queries','ready')
        j.authority=SimpleNamespace(memory=SimpleNamespace(attach_completion=lambda _:None,
            commit_selection=lambda:True))
    ready=[False]
    assembly=SimpleNamespace(event=SimpleNamespace(query=lambda:ready[0]),uploaded_bytes=96)
    result=(SimpleNamespace(close=lambda:True),assembly,SimpleNamespace(total_window_tokens=32),{},0.)
    obj.executor=SimpleNamespace(submit=lambda *a:SimpleNamespace(done=lambda:True,result=lambda:result))
    obj.report=lambda *a:None
    assert obj.step(first) and p.owner=='a'
    first.cancelled=True
    assert not obj.step(first) and not obj.step(second)
    assert p.inflight_bytes==120
    ready[0]=True
    assert obj.step(first) and first.done and p.inflight_bytes==0
    assert obj.step(second) and p.owner=='b'


def test_checkpoint_restore_reserves_full_active_view_and_model_payload():
    # M can include K plus accepted generation, whereas a reselection uses K.
    p = policy(async_restore=True, session_checkpoint_bytes=256,
        memory=SimpleNamespace(select_budget=32, gen_budget=16, block_tokens=4))
    restore = job('restore')
    restore.command = SimpleNamespace(request_id='restore', window_tokens=48)
    restore.model_bytes = 200
    assert p.capacity == (48 + 8) * 3 + 256
    assert p.acquire(restore) and p.inflight_bytes == (48 + 8) * 3 + 200
    p.observe(restore, 48 * 3 + 200)
    assert not p.acquire(job('neighbour'))
    p.release(restore)
    assert p.inflight_bytes == 0


def test_resident_pipeline_enforces_slots_bytes_and_individual_completion():
    p = policy(async_restore=True, resident_restore_slots=2, session_checkpoint_bytes=256,
        memory=SimpleNamespace(select_budget=32, gen_budget=16, block_tokens=4))
    a, b, c = [job(rid) for rid in ('a', 'b', 'c')]
    for j in (a, b, c):
        j.command.resident = True
        j.model_bytes = 200
    assert p.acquire(a) and p.acquire(b) and not p.acquire(c)
    assert p.resident_inflight_bytes == 400 and p.resident_capacity == 512
    with pytest.raises(ValueError, match='Duplicate'):
        p.acquire(a)
    b.cancelled = True
    assert not p.acquire(c)
    p.release(a)
    assert p.acquire(c) and p.resident_inflight_bytes == 400
    p.release(c)
    assert p.resident_owners == {'b': 200}
    p.release(b)
    assert p.resident_inflight_bytes == 0 and p.resident_peak_inflight == 2


def test_resident_credit_is_independent_bounded_and_does_not_distort_cold_cost():
    p = policy(async_restore=True, session_checkpoint_bytes=256,
        memory=SimpleNamespace(select_budget=32, gen_budget=16, block_tokens=4))
    cold, hot, next_hot = job('cold'), job('hot'), job('next-hot')
    for item in (hot, next_hot):
        item.command.resident = True
        item.model_bytes = 200
        item.authority = SimpleNamespace(memory=SimpleNamespace(request_id='same-session'))
    assert p.acquire(cold) and p.acquire(hot)
    assert p.inflight_bytes == 72 and p.resident_inflight_bytes == 200
    assert p.cost(hot) == 200 and not p.acquire(next_hot)
    with pytest.raises(ValueError, match='Foreign resident'):
        p.release(next_hot)
    p.observe(hot, 200)
    assert not p.upload_ratios
    hot.cancelled = True
    assert not p.acquire(next_hot)  # Cancellation does not return DMA credit.
    p.release(hot)
    next_hot.model_bytes = 257
    with pytest.raises(ValueError, match='Resident model restore exceeds'):
        p.acquire(next_hot)
    assert p.resident_owner is None and p.resident_inflight_bytes == 0
    next_hot.model_bytes = 200
    assert p.acquire(next_hot) and p.owner == 'cold'
    assert p.snapshot()['resident_inflight_bytes'] == 200
    p.release(cold)
    assert p.resident_owner == 'next-hot'
    p.release(next_hot)


def test_maintenance_waits_for_native_cache_allocation_during_model_load(monkeypatch):
    from vllm_kvmem.async_refresh import AsyncRefresh
    import vllm_kvmem.cache_manifest as manifest
    from vllm_kvmem.runner import make_probe_runner
    opts = SimpleNamespace(maintenance_max_steps=8)
    runner = SimpleNamespace(model=SimpleNamespace(forward=lambda:None))
    refresh = AsyncRefresh(runner, opts)  # kv_cache_config intentionally absent.
    assert refresh.maintenance is None
    calls = []
    monkeypatch.setattr(manifest, 'archive_bytes_per_token', lambda geometry:calls.append(geometry) or 123)
    monkeypatch.setattr('vllm_kvmem.maintenance.MaintenancePolicy', lambda o,n:('budget',o,n))
    class Base:
        def initialize_kv_cache(self):
            runner.kv_cache_config = 'allocated-geometry'
            return 'native-result'
    wrapper = make_probe_runner(Base)()
    wrapper._kvmem_async_refresh = refresh
    monkeypatch.setattr('vllm_kvmem.profile_scopes.install_profile_scopes', lambda _:None)
    monkeypatch.setattr('vllm_kvmem.recurrent_table_journal.install_recurrent_table_journal', lambda *a, **kw:None)
    assert wrapper.initialize_kv_cache() == 'native-result'
    assert calls == ['allocated-geometry'] and refresh.maintenance == ('budget',opts,123)
    refresh.initialize_cache()
    assert calls == ['allocated-geometry']  # No budget/lease reset.
    refresh.shutdown()
