"""Raw-source max pooling against an independent incoming-neighbor oracle."""
from types import SimpleNamespace as NS
import math
import random

import pytest

from vllm_kvmem.block_graph import GraphRow
from vllm_kvmem.graph_scoring import score_graph
from test_graph_scoring_v2 import snapshot


def pool(raw, rows=(), **kwargs):
    return score_graph(raw, list(range(0,len(raw)*4,4)), [True]*len(raw),
        snapshot(rows), scoring_version='v2', aggregation='max', **kwargs)


def incoming_oracle(raw, starts, mask, snap, mode, alpha, cross):
    """Build incoming sets independently; never run sparse transfers."""
    weights = (alpha/2,alpha/2) if mode=='multi' else (alpha,0) if mode=='locality' else (0,alpha)
    boundaries={a for a,b,role in snap.message_spans}
    boundaries.update(b for (a,b,role),known in zip(snap.message_spans,snap.message_end_known) if known)
    output=[]
    for j,target in enumerate(starts):
        if not mask[j]:
            output.append(raw[j]);continue
        adjacent=[];attention=[]
        for i,source in enumerate(starts):
            if not mask[i] or i==j:continue
            if abs(source-target)==snap.block_tokens:
                lo,hi=min(source,target),max(source,target)+snap.block_tokens
                contained=any(a<=lo and hi<=b for a,b,role in snap.message_spans)
                factor=cross if not contained and any(lo<b<hi for b in boundaries) else 1.
                if factor>0:adjacent.append(factor*raw[i])
            for row in snap.rows:
                if row.source!=source or row.samples<=0 or source+snap.block_tokens>snap.completed_through:continue
                if any(t==target and w>0 for t,w in zip(row.targets,row.weights)):
                    attention.append(raw[i])
        l=max(adjacent,default=raw[j]);a=max(attention,default=raw[j]);wl,wa=weights
        output.append((1-wl-wa)*raw[j]+wl*l+wa*a)
    return output


def test_shared_target_does_not_gain_from_hundreds_of_equal_sources():
    rows=[GraphRow(i*4,(0,),(.001,),1) for i in range(1,101)]
    with pool([1.]*101,rows,mode='multi',alpha=.2) as out:
        assert out.scores==[1.]*101
        assert out.diagnostics['identity'] and not out.diagnostics['score_sum_conserved']


def test_raw_max_ignores_attention_edge_magnitude_and_incoming_multiplicity():
    rows=[GraphRow(4,(0,),(.0001,),1),GraphRow(8,(0,),(.9,),1)]
    with pool([0.,10.,8.],rows,mode='attention',alpha=.2,trace=True) as out:
        assert out.scores==[2.,10.,8.]
        target=next(r for r in out.trace if r['target']==0)
        assert target['sources'][0]['source']==4
        assert target['sources'][0]['raw_edge_weight']==.0001
        assert target['sources'][0]['raw_source_score']==10.


def test_one_hop_never_forwards_received_scores():
    rows=[GraphRow(8,(4,),(.1,),1),GraphRow(4,(0,),(.1,),1)]
    with pool([0.,0.,10.],rows,mode='attention',alpha=.2) as out:
        assert out.scores==[0.,2.,10.]
        assert out.diagnostics['propagated_score_sum']==12.


def test_pool_then_mix_retains_original_alpha_and_equal_graph_shares():
    with pool([.1,.2,.7],[GraphRow(8,(0,),(.8,),1)],mode='multi',alpha=.2) as out:
        assert out.scores==pytest.approx([.17,.25,.65])
        assert sum(out.scores)>1.
        assert out.diagnostics['locality_weight']==out.diagnostics['attention_weight']==.1
    with pool([10.,1.],[GraphRow(4,(0,),(.8,),1)],mode='attention',alpha=.2) as out:
        assert out.scores==pytest.approx([8.2,1.])


def test_no_relation_is_exact_identity_and_zero_alpha_uses_original_object():
    raw=[.1,.2,.7]
    with pool(raw,mode='attention',alpha=.2) as out:assert out.scores is raw
    with pool(raw,mode='multi',alpha=0.) as out:assert out.scores is raw


@pytest.mark.parametrize('mode',['attention','locality','multi'])
@pytest.mark.parametrize('alpha',[0.,.2,1.])
@pytest.mark.parametrize('cross',[0.,.25,1.])
def test_random_graphs_masks_and_message_boundaries_match_incoming_oracle(mode,alpha,cross):
    rng=random.Random(814)
    raw=[rng.random()*10 for _ in range(13)]
    starts=[i*4 for i in range(13)]
    rows=[GraphRow(i*4,tuple(j*4 for j in range(max(0,i-4),i)),
        tuple(.01+rng.random()*.1 for _ in range(min(4,i))),1) for i in range(13)]
    snap=snapshot(rows,spans=((0,13,'user'),(13,37,'tool')),completed=48)
    for mask in [[True]*13, [i%3!=0 for i in range(13)],[True]*2+[False]*11]:
        with score_graph(raw,starts,mask,snap,mode=mode,alpha=alpha,aggregation='max',
                         scoring_version='v2',cross_message_weight=cross) as out:
            assert out.scores==pytest.approx(incoming_oracle(raw,starts,mask,snap,mode,alpha,cross))
            assert all(a==b for a,b,keep in zip(raw,out.scores,mask) if not keep)
            assert min(out.scores)>=0 and max(out.scores)<=max(raw)+1e-12


def test_masked_sources_and_targets_cannot_donate_or_receive():
    snap=snapshot([GraphRow(8,(0,4),(.5,.5),1)])
    for mask in ([True,True,False],[False,True,True]):
        with score_graph([0.,0.,10.],[0,4,8],mask,snap,mode='attention',alpha=.2,
                         scoring_version='v2',aggregation='max') as out:
            assert out.scores==([0.,0.,10.] if not mask[2] else [0.,2.,10.])


def test_unsampled_and_incomplete_source_rows_do_not_propagate():
    snap=snapshot([GraphRow(4,(0,),(.1,),0),GraphRow(8,(0,),(.1,),1)],completed=8)
    raw=[0.,8.,10.]
    with score_graph(raw,[0,4,8],[True]*3,snap,mode='attention',alpha=1.,
                     scoring_version='v2',aggregation='max') as out:assert out.scores is raw


@pytest.mark.parametrize('aggregation',[None,False,[],{},'mean'])
def test_invalid_aggregators_fail_explicitly(aggregation):
    with pytest.raises(ValueError,match='graph_aggregation'):
        score_graph([1.,2.],[0,4],[True,True],snapshot(),aggregation=aggregation)


def test_max_requires_v2_and_sum_default_remains_unchanged():
    with pytest.raises(ValueError,match='requires'):
        score_graph([1.,2.],[0,4],[True,True],snapshot(),aggregation='max')
    args=([.1,.2,.7],[0,4,8],[True]*3,snapshot([GraphRow(8,(0,),(.8,),1)]))
    with score_graph(*args,scoring_version='v2') as default,score_graph(*args,scoring_version='v2',aggregation='sum') as explicit:
        assert default.scores==explicit.scores
        assert default.diagnostics['aggregation']=='sum'


def test_max_trace_respects_target_bound_and_highest_raw_source_tie_break():
    snap=snapshot([GraphRow(8,(0,),(.8,),1),GraphRow(4,(0,),(.001,),1)])
    with score_graph([0.,10.,10.],[0,4,8],[True]*3,snap,mode='attention',alpha=.2,
                     scoring_version='v2',aggregation='max',trace=True,trace_targets=[0]) as out:
        assert out.trace[0]['sources'][0]['source']==4
        assert len(out.trace)<=64 and all(len(r['sources'])<=2 for r in out.trace)
