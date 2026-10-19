"""Compare native V2 with the unchanged sparse and independent matrix oracles."""
import math
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.graph_scoring import score_graph
from test_graph_scoring_v2 import matrix_oracle


def graph(owner='g',seed=17):
    g=BlockGraph(owner,4,topk=4)
    rng=np.random.default_rng(seed)
    masses=np.tril(rng.random((12,12)))
    masses/=masses.sum(axis=1,keepdims=True)
    with g.prepare_accept(0,48,list(range(3,48,4)),list(range(0,48,4)),masses) as update:update.commit()
    g.update_message_spans([(0,13,'user'),(13,37,'tool')])
    return g


@pytest.mark.parametrize('mode',['locality','attention','multi'])
@pytest.mark.parametrize('alpha',[0.,.2,1.])
@pytest.mark.parametrize('cross',[0.,.25,1.])
def test_native_matches_python_scores_diagnostics_and_selected_pages(mode,alpha,cross):
    g=graph()
    try:
        with g.snapshot() as snap:
            starts=list(range(0,52,4));raw=[0.,13.,1.,0.,3.,5.,0.,.5,.25,.125,.01,1.,2.]
            for mask in [[True]*13,[False,True,False,True]*3+[True], [True]*2+[False]*11]:
                kwargs=dict(mode=mode,scoring_version='v2',alpha=alpha,cross_message_weight=cross)
                with score_graph(raw,starts,mask,snap,**kwargs) as ref,score_graph(raw,starts,mask,snap,**kwargs,native=True) as fast:
                    assert fast.scores==ref.scores
                    for key,value in ref.diagnostics.items():
                        if key!='scoring_seconds':assert fast.diagnostics[key]==pytest.approx(value) if isinstance(value,float) else fast.diagnostics[key]==value
                    assert sorted(starts,key=lambda p:(-fast.scores[starts.index(p)],p))[:4]==sorted(starts,key=lambda p:(-ref.scores[starts.index(p)],p))[:4]
                    if cross==.25:
                        assert fast.scores==pytest.approx(matrix_oracle(raw,starts,mask,snap,mode,alpha),abs=1e-12)
    finally:g.close()
    assert g.budget.used==0


def test_cached_index_recomputes_mask_normalization_and_restore_binding():
    g=graph()
    try:
        raw=[0.]*11+[1.];starts=list(range(0,48,4))
        saved=g.snapshot()
        with saved:
            for mask in [[True]*12,[False]*9+[True]*3,[True]*12]:
                with score_graph(raw,starts,mask,saved,mode='attention',scoring_version='v2',native=True,alpha=1.) as fast,score_graph(raw,starts,mask,saved,mode='attention',scoring_version='v2',alpha=1.) as ref:
                    assert fast.scores==ref.scores
            assert g._native_scoring_cache.builds==1 and g._native_scoring_cache.hits==2
            with g.prepare_accept(48,52,[51],[0,4,8],np.array([[.2,.3,.5]])) as update:update.commit()
            with g.snapshot() as now:
                with score_graph(raw,starts,[True]*12,now,scoring_version='v2',native=True):pass
            assert g._native_scoring_cache.builds==2
            g.restore(saved)
            with g.snapshot() as restored:
                with score_graph(raw,starts,[True]*12,restored,scoring_version='v2',native=True):pass
            assert g._native_scoring_cache.builds==3
    finally:g.close()
    assert not g.budget.used


def test_owner_isolation_equal_names_and_concurrent_masked_scoring():
    a,b=graph('same',17),graph('same',71)
    try:
        with a.snapshot() as sa,b.snapshot() as sb:
            def run(snapshot,index):
                mask=[(i+index)%3!=0 for i in range(12)]
                with score_graph([1.]*12,list(range(0,48,4)),mask,snapshot,scoring_version='v2',native=True) as fast,score_graph([1.]*12,list(range(0,48,4)),mask,snapshot,scoring_version='v2') as ref:
                    assert fast.scores==ref.scores
            with ThreadPoolExecutor(4) as pool:
                futures=[pool.submit(run,s,i) for i in range(8) for s in (sa,sb)]
                for f in futures:f.result(timeout=5)
            assert a._native_scoring_cache.entry is not b._native_scoring_cache.entry
    finally:a.close();b.close()
    assert not a.budget.used and not b.budget.used


def test_index_budget_failure_releases_scratch_and_does_not_publish_cache():
    g=graph()
    try:
        with g.snapshot() as snap:
            before=g.binding,g.bytes
            snap._limit=g.bytes+65536+12*320
            with pytest.raises(BufferError):score_graph([1.]*12,list(range(0,48,4)),[True]*12,snap,scoring_version='v2',native=True)
            assert (g.binding,g.bytes)==before and g._native_scoring_cache.entry is None
    finally:g.close()
    assert not g.budget.used


def test_trace_and_v1_have_explicit_reference_fallback():
    g=graph()
    try:
        with g.snapshot() as snap:
            for version,trace,reason in [('v1',False,'scoring_version'),('v2',True,'trace')]:
                with score_graph([1.]*12,list(range(0,48,4)),[True]*12,snap,scoring_version=version,trace=trace,native=True) as result:
                    assert result.diagnostics['backend']=='python_reference'
                    assert result.diagnostics['native_fallback']==reason
    finally:g.close()
