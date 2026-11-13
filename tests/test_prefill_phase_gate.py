from types import SimpleNamespace as NS
from copy import deepcopy
import json
import sys

import pytest

from vllm_kvmem.serving.prefill_phase_gate import PrefillPhaseGate


def job(rid, selection='query', **kw):
    return NS(command=NS(request_id=rid, selection=selection), history=100,
              stage=kw.get('stage', 'prepare'), future=kw.get('future'),
              done=False, cancelled=kw.get('cancelled', False))


def test_gate_holds_query_without_holding_pressure_and_releases_exact_cohort():
    events = []
    gate = PrefillPhaseGate(2, events.append, clock=lambda: 1.)
    a, b = job('a'), job('b')
    gate.arrive(a)
    assert not gate.permits(a)
    assert gate.permits(job('pressure', selection='pressure', stage='assembly'))
    assert len(events) == 1
    gate.arrive(b)
    assert gate.permits(a) and gate.permits(b)
    assert events[-1]['event'] == 'history_prefill_released'
    assert events[-1]['history_tokens'] == 200
    a.stage = 'assembly'
    assert gate.permits(a)
    assert len(events) == 3


@pytest.mark.parametrize('fault', ['duplicate', 'extra', 'early_work', 'cancelled', 'unknown'])
def test_invalid_cohort_fails_without_silently_releasing(fault):
    events = []
    gate = PrefillPhaseGate(2, events.append)
    a = job('a')
    gate.arrive(a)
    with pytest.raises(RuntimeError):
        if fault == 'duplicate': gate.arrive(a)
        elif fault == 'extra':
            gate.arrive(job('b'))
            gate.arrive(job('c'))
        elif fault == 'early_work':
            a.stage = 'assembly'
            gate.permits(a)
        elif fault == 'cancelled':
            a.cancelled = True
            gate.permits(a)
        else: gate.permits(job('unknown'))


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_finite_owner_count(value):
    with pytest.raises(ValueError): PrefillPhaseGate(value, lambda _: None)


def completed_work(tokens=200):
    counts = dict(public_batches=2, public_prefill_tokens=tokens,
                  public_non_prefill_tokens=0, decode_request_batches=0)
    return dict(counter_id='test-counter',submitted=counts,completed=deepcopy(counts),
                all_submitted_completed=True,pending_fence=None,
                completed_fence=dict(completion_lower_bound=2.,completion_upper_bound=3.))


def test_gpu_endpoint_waits_for_final_fence_without_blocking_pressure():
    events, polls = [], []
    snapshots = iter((None, completed_work()))
    def completion():
        polls.append(1)
        return next(snapshots)
    gate = PrefillPhaseGate(2,events.append,completion=completion,clock=lambda:4.)
    a,b = job('a'),job('b')
    gate.arrive(a)
    assert not gate.permits(a) and not polls
    gate.arrive(b)
    assert not gate.released and len(events)==2
    assert gate.permits(job('pressure',selection='pressure',stage='assembly'))
    assert not polls
    assert not gate.permits(a)
    assert gate.permits(b) and len(polls)==2
    assert events[-1]['gpu_completion_confirmed']
    assert events[-1]['public_work']['completed']['public_prefill_tokens']==200
    assert gate.permits(a) and len(polls)==2


@pytest.mark.parametrize('fault',('tokens','decode','nonprefill','pending','counter','fence'))
def test_gpu_endpoint_rejects_incomplete_or_contaminated_work(fault):
    work=completed_work()
    if fault=='tokens':
        for key in ('submitted','completed'):work[key]['public_prefill_tokens']=201
    elif fault in ('decode','nonprefill'):
        field='decode_request_batches' if fault=='decode' else 'public_non_prefill_tokens'
        for key in ('submitted','completed'):work[key][field]=1
    elif fault=='pending':work['pending_fence']={'sequence':3}
    elif fault=='counter':work['counter_id']=None
    else:work['completed_fence']=None
    gate=PrefillPhaseGate(2,lambda _:None,completion=lambda:work)
    a,b=job('a'),job('b');gate.arrive(a);gate.arrive(b)
    with pytest.raises(RuntimeError):gate.permits(a)
    assert not gate.released


def test_install_gpu_endpoint_uses_nonblocking_flush_only(tmp_path,monkeypatch):
    from vllm_kvmem.serving import prefill_phase_gate
    class Manager:
        def enqueue(self,command):return job(command.request_id,command.selection)
        def step(self,task):return 'query-executed'
    class Work:
        submitted={'public_batches':2}
        completed={'public_batches':1}
        pending=object()
        ready=False
        calls=[]
        def poll(self,**kwargs):
            self.calls.append(kwargs)
            if self.ready:self.pending=None;self.completed=self.submitted.copy()
        def snapshot(self):return completed_work()
    work=Work();manager=Manager()
    manager.runner=NS(vllm_config=NS(scheduler_config=NS(max_num_seqs=2)),
                      _kvmem_probe_output=tmp_path,_kvmem_public_work=work)
    monkeypatch.setitem(sys.modules,'vllm_kvmem.serving.async_refresh',NS(AsyncRefresh=Manager))
    monkeypatch.setenv('KVMEM_DIAGNOSTIC_PREFILL_PHASE_OWNERS','2')
    monkeypatch.setenv('KVMEM_DIAGNOSTIC_PREFILL_GPU_COMPLETION','1')
    prefill_phase_gate.install()
    a=manager.enqueue(NS(request_id='a',selection='query'))
    b=manager.enqueue(NS(request_id='b',selection='query'))
    assert manager.step(a) is False
    assert work.calls==[{'flush':True}]
    work.ready=True
    assert manager.step(b)=='query-executed'
    release=json.loads((tmp_path/'prefill_phase_gate.jsonl').read_text().splitlines()[-1])
    assert release['public_work']['counter_id']=='test-counter'
    assert work.calls==[{'flush':True},{'flush':True}]
