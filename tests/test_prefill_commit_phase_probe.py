"""The timing wrapper must not redirect a staged transaction to live state."""
from copy import copy
from types import SimpleNamespace as NS

import numpy as np
import pytest

from prefill_commit_phase_probe import CommitTimings
from prefill_phase_probe import PhaseScopes
from vllm_kvmem.buffered_prefill_attention import BufferedPrefillHistory
from vllm_kvmem._prefill_state_native import State


def test_native_and_python_scopes_keep_copy_isolation_and_exact_scores(monkeypatch):
    events=[];worker=NS(profiler=NS(is_running=False))
    scopes=PhaseScopes(worker,push=events.append,pop=lambda:events.append('pop'))
    timing=CommitTimings(scopes)
    options=NS(native_prefill_commit=True,prefill_attention_half_life=8192,
        prefill_attention_interval=512,prefill_attention_stride=32,
        memory=NS(select_budget=36864,gen_budget=16384,block_tokens=32))
    original=BufferedPrefillHistory(options);control=BufferedPrefillHistory(options)
    args=(512,512,512,[511],[0,32],[0,32],0,np.array([[.3,.7]],dtype=np.float32))
    assert control.accept(*args)
    # monkeypatch retains the originals and restores both class descriptors.
    for owner,name,label in ((BufferedPrefillHistory,'accept','cpu_commit.buffered_accept'),
                              (State,'committed','cpu_commit.native_committed')):
        monkeypatch.setattr(owner,name,getattr(owner,name))
        timing.wrap(owner,name,label)
        wrapped=getattr(owner,name);timing.wrap(owner,name,label)
        assert getattr(owner,name) is wrapped
    for owner,name,label in ((BufferedPrefillHistory,'scores','buffered_history.scores'),
                              (State,'values','buffered_state.values')):
        monkeypatch.setattr(owner,name,getattr(owner,name))
        assert scopes.wrap(owner,name,label)
    assert original.accept(*args)
    assert not timing.rows and not events
    assert original.snapshot()==control.snapshot()
    worker.profiler.is_running=True
    staged=copy(original);before=original.snapshot()
    assert staged.accept(1024,*args[1:])
    assert original.snapshot()==before
    assert control.accept(1024,*args[1:])
    assert staged.snapshot()==control.snapshot() and staged.scores()==control.scores()
    assert 'accept' not in staged.__dict__ and 'scores' not in staged.__dict__
    assert 'prefill_phase::buffered_history.scores' in events
    assert 'prefill_phase::buffered_state.values' in events
    assert {r['label'] for r in timing.rows}=={'cpu_commit.buffered_accept','cpu_commit.native_committed'}
    assert all(r['end_ns']>=r['start_ns'] and r['cpu_ns']>=0 for r in timing.rows)
    assert events.count('pop')==len(events)//2


def test_failed_call_balances_ranges_and_keeps_exception():
    worker=NS(profiler=NS(is_running=True));events=[]
    scopes=PhaseScopes(worker,push=events.append,pop=lambda:events.append('pop'))
    timing=CommitTimings(scopes);failure=ValueError('original')
    class Owner:
        def call(self):
            worker.profiler.is_running=False
            raise failure
    timing.wrap(Owner,'call','failure')
    with pytest.raises(ValueError) as caught:Owner().call()
    assert caught.value is failure and events==['prefill_phase::failure','pop']
    assert len(timing.rows)==1
