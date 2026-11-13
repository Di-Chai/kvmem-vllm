from copy import copy
import struct
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vllm_kvmem.serving.buffered_prefill_attention import BufferedPrefillHistory
from vllm_kvmem.serving.prefill_attention import PrefillHistory


def histories():
    options=NS(native_prefill_commit=True,prefill_attention_half_life=8192,
        prefill_attention_interval=512,prefill_attention_stride=32,
        memory=NS(select_budget=36864,gen_budget=16384,block_tokens=32))
    return PrefillHistory(options),BufferedPrefillHistory(options)


def exact(a,b):
    assert a.snapshot()==b.snapshot() and a.scores()==b.scores()
    assert list(a.values)==list(b.values)
    assert b''.join(struct.pack('<dd',*v) for v in a.values.values())==b''.join(
        struct.pack('<dd',*v) for v in b.values.values())


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('layout',['contiguous','strided','reversed'])
def test_reordered_repeated_and_retired_keys_preserve_every_double(dtype,layout):
    old,new=histories();rng=np.random.default_rng(317)
    for step in range(1,65):
        keys=rng.integers(0,90,size=81).tolist()
        starts=list(range(0,81*32,32))
        if step%5==0:starts=starts[:70]  # Preserve the original zip truncation.
        matrix=rng.random((16,81));matrix/=matrix.sum(axis=1,keepdims=True)
        matrix=matrix.astype(dtype)
        if layout=='strided':
            backing=np.zeros((16,162),dtype=dtype);backing[:,::2]=matrix;matrix=backing[:,::2]
        elif layout=='reversed':matrix=matrix[::-1,::-1]
        args=(step*512,512,512,list(range(31,512,32)),starts,keys,2304,matrix)
        assert old.accept(*args)==new.accept(*args)
        exact(old,new)


@pytest.mark.parametrize('value',[np.nextafter(.995,-np.inf),.995,np.nextafter(.995,np.inf),
    np.nextafter(1.005,-np.inf),1.005,np.nextafter(1.005,np.inf)])
def test_mass_validation_boundary_matches_the_original(value):
    states=histories();outcomes=[]
    for state in states:
        try:state.accept(512,512,512,[511],[0],[0],0,np.array([[value]]))
        except RuntimeError:outcomes.append('error')
        else:outcomes.append(state.snapshot())
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('rows,pages',[(32,1664),(64,5375)])
def test_real_prefill_shapes_preserve_all_scores(rows,pages):
    old,new=histories();rng=np.random.default_rng(128)
    masses=rng.random((rows,pages)).astype(np.float32);masses/=masses.sum(axis=1,keepdims=True)
    history=list(range(0,pages*32,32))
    for state in (old,new):
        state.end=170000;state.values={key:[.123456789,.875] for key in history}
        assert state.accept(172048,2048,2048,list(range(31,rows*32,32)),history,history,170000,masses)
    exact(old,new)


@pytest.mark.parametrize('fault',['nan','negative','sum','rows','columns','dtype','dimension','overflow'])
def test_failed_commit_preserves_original_and_staged_state(fault):
    old,new=histories();args=[512,512,512,[31,63],[0,32],[100,200],0,np.array([[.5,.5],[.2,.8]])]
    old.accept(*args);new.accept(*args);saved=new.snapshot();staged=copy(new)
    args[0]=1024
    if fault=='nan':args[-1][0,0]=np.nan
    elif fault=='negative':args[-1][0,0]=-.5
    elif fault=='sum':args[-1][0,0]=1.
    elif fault=='rows':args[3]=[31]
    elif fault=='columns':args[5]=[100]
    elif fault=='dtype':args[-1]=args[-1].astype(np.float16)
    elif fault=='dimension':args[-1]=args[-1].reshape(4)
    else:args[6]=2**63-1
    with pytest.raises((ValueError,RuntimeError,TypeError)):staged.accept(*args)
    assert staged.snapshot()==saved and new.snapshot()==saved
    exact(old,new)


def test_shallow_staging_and_checkpoint_restore_remain_independent():
    old,new=histories();matrix=np.array([[.3,.7]],dtype=np.float32)
    args=(512,512,512,[511],[0,32],[0,32],0,matrix)
    old.accept(*args);new.accept(*args);saved=new.snapshot();staged=copy(new)
    staged.accept(1024,*args[1:]);assert new.snapshot()==saved
    # Materialized values are snapshots, not mutable aliases into native state.
    exported=staged.values;exported[0][0]=999;assert staged.values[0][0]!=999
    restored=histories()[1];restored.restore(saved,512);exact(old,restored)
    restored.restore(saved,0);assert restored.snapshot()==({},0,0,0)
    old.accept(1024,*args[1:]);exact(old,staged)


@pytest.mark.parametrize('end,count,accepted,rows',[(512,512,511,[31]),(512,512,512,[]),(0,512,512,[31])])
def test_rejected_packets_do_not_materialize_or_mutate_scores(end,count,accepted,rows):
    old,new=histories()
    args=(end,count,accepted,rows,[0],[0],0,np.ones((1,1)))
    assert not old.accept(*args) and not new.accept(*args);exact(old,new)


def test_layerwise_staging_is_atomic_and_reports_actual_native_updates():
    from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure
    c=LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    c.names=('a','b');c.buffered_history=True;c.histories={};c.options=None
    options=NS(native_prefill_commit=True,prefill_attention_half_life=8192,
        pressure_policy='attention',guided_query_tokens=0,query_history=False,prefill_attention=True)
    c._history('owner',options)
    masses=np.array([[[.3,.7]],[[.8,.2]]],dtype=np.float32)
    pending=NS(prefill=(512,512,[0,32],0),rows=[511],
        layerwise_pressure=(c,'owner',[[0,32],[0,64]],masses))
    previous=[h.snapshot() for h in c.histories['owner']]
    prepared=c.prepare_accept(pending,512)
    assert [h.snapshot() for h in c.histories['owner']]==previous
    c.commit_accept(prepared)
    assert [h.successful_buffered_commits for h in c.histories['owner']]==[1,1]
    assert c.history_execution(c.histories['owner'][0])==dict(
        attention_state_backend='native_buffered',buffered_commits=1,buffered_restores=0)
    with pytest.raises(RuntimeError,match='configured'):
        c.history_execution(PrefillHistory(options))
    previous=[h.snapshot() for h in c.histories['owner']]
    pending.prefill=(1024,512,[0,32],0);masses[1,0,0]=np.nan
    with pytest.raises(RuntimeError):c.prepare_accept(pending,512)
    assert [h.snapshot() for h in c.histories['owner']]==previous


def test_frozen_prefix_restore_keeps_the_buffered_backend():
    from vllm_kvmem.serving.layerwise_pressure import LayerwisePrefillPressure
    from vllm_kvmem.serving.layerwise_checkpoints import LayerwisePrefix,prepare_restore
    c=LayerwisePrefillPressure.__new__(LayerwisePrefillPressure)
    c.names=('a','b');c.buffered_history=True;c.maps={};c.histories={};c.query=NS(finished=set())
    mapping=((0,1,0,32,0),(1,1,32,32,32))
    history=(((0,.3,1.),(32,.7,1.)),64,1,1)
    checkpoint=LayerwisePrefix(64,c.names,(mapping,mapping),(history,history))
    authority=NS(history_end=64,records={(i,1):NS(history_begin=i*32,count=32,handle=NS(slot=i,generation=1)) for i in range(2)},
        memory=NS(config=NS(block_tokens=32),_store=NS(working_records=lambda:mapping)),
        action_history=NS(prefill=NS(snapshot=lambda:({0:(.3,1.),32:(.7,1.)},64,1,1))))
    options=NS(native_prefill_commit=True,prefill_attention_half_life=8192)
    chosen,frame,restored=prepare_restore(c,'restored',authority,checkpoint,options)
    assert all(isinstance(h,BufferedPrefillHistory) for h in restored)
    assert all(h.snapshot()==authority.action_history.prefill.snapshot() for h in restored)
    assert all(c.history_execution(h)==dict(attention_state_backend='native_buffered',
        buffered_commits=0,buffered_restores=1) for h in restored)
    assert [r.history_begin for r in chosen[0]]==[0,32]


def test_direct_native_score_export_preserves_order_and_float_bits(monkeypatch):
    from vllm_kvmem._prefill_state_native import State
    if not hasattr(State,'scores'):
        pytest.skip('Rebuild the native prefill extension for direct-score qualification')
    monkeypatch.setenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT','1')
    old,new=histories()
    rng=np.random.default_rng(3421)
    # Include tiny/subnormal inputs, signed zero and excluded zero/negative
    # exposure. Exact score ties must not change native picker's tie-breaking.
    values={int(k):[float(m),float(e)] for k,m,e in zip(rng.permutation(2048),
        rng.random(2048),rng.random(2048))}
    values.update({3000:[-0.,1.],3001:[1e-320,1e-310],3002:[1.,0.],3003:[1.,-1.]})
    old.values=values;new.values=values
    expected,actual=old.scores(),new.scores()
    assert list(actual)==list(expected)
    assert b''.join(struct.pack('<d',v) for v in actual.values())==b''.join(
        struct.pack('<d',v) for v in expected.values())
    # Shallow staging/checkpoints may share State, but exported dictionaries
    # must remain detached from it.
    actual[3000]=123.
    assert struct.pack('<d',new.scores()[3000])==struct.pack('<d',-0.)


def test_older_extension_score_fallback_remains_exact():
    _,history=histories()
    history._buffered_state=NS(values=lambda:{64:[.5,2.],0:[1.,0.]})
    assert history.scores()=={64:.25}


def test_native_score_capability_does_not_enable_unqualified_export_by_itself(monkeypatch):
    monkeypatch.delenv('KVMEM_LAYERWISE_PRESSURE_NATIVE_SCORE_EXPORT',raising=False)
    _,history=histories()
    history._buffered_state=NS(values=lambda:{64:[.5,2.]},
        scores=lambda:pytest.fail('Direct score export was not enabled'))
    assert not history.native_score_export and history.scores()=={64:.25}
