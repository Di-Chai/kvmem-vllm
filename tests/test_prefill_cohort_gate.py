from types import SimpleNamespace as NS
import pytest
from vllm_kvmem.serving.prefill_cohort_gate import CohortGate


def fixture(monkeypatch):
    import vllm_kvmem.serving.request_plan as module
    monkeypatch.setattr(module,'request_plan',lambda r,*_:NS(query=(8000,8100),replay_begin=8000))
    def owner(rid,n):
        return NS(request_id=rid,num_prompt_tokens=n,num_computed_tokens=0,finished=False,
                  is_finished=lambda:False)
    a,b=owner('a',9000),owner('b',10000)
    waiting=[];ready={'a','b'};maintenance=set()
    scheduler=NS(requests={'a':a,'b':b},running=[b,a],query_policy=NS(block_tokens=32,gen_budget=16384),
        _waiting_queues=lambda:[waiting],_phase_eligible=lambda r:r.request_id in ready,
        _phase_refresh_only=lambda r:r.request_id in maintenance)
    return CohortGate([9000,10000]),scheduler,a,b,ready,maintenance


def test_initial_arrivals_and_ready_wave(monkeypatch):
    gate,s,a,b,ready,maintenance=fixture(monkeypatch)
    del s.requests['b'];gate.prepare(s);assert not gate.permits(a)
    s.requests['b']=b;gate.prepare(s)
    assert gate.permits(a) and gate.permits(b) and s.running==[a,b]


def test_publication_wait_and_maintenance_are_separate(monkeypatch):
    gate,s,a,b,ready,maintenance=fixture(monkeypatch)
    ready.remove('b');gate.prepare(s);assert not gate.allowed
    ready.add('b');maintenance.add('b');gate.prepare(s);assert gate.allowed=={'b'}
    maintenance.clear();gate.prepare(s);assert gate.allowed=={'a','b'}


def test_final_query_and_decode_fixed_order(monkeypatch):
    gate,s,a,b,ready,maintenance=fixture(monkeypatch)
    a.num_computed_tokens=b.num_computed_tokens=8000
    maintenance.add('b');gate.prepare(s);assert gate.allowed=={'a','b'}
    maintenance.clear();gate.prepare(s);assert gate.allowed=={'a'}
    a.is_finished=lambda:True;gate.prepare(s);assert gate.allowed=={'b'}


def test_duplicate_identity_rejected(monkeypatch):
    gate,s,a,b,ready,maintenance=fixture(monkeypatch)
    gate.prepare(s);s.requests['a']=NS(num_prompt_tokens=9000)
    with pytest.raises(RuntimeError,match='repeats'):gate.prepare(s)


@pytest.mark.parametrize('lengths',[[],[1,1],[0],[True],['20']])
def test_identity_must_be_unambiguous(lengths):
    with pytest.raises(ValueError):CohortGate(lengths)
