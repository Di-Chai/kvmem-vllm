"""Owner/version-bound, budgeted CSR caches for the native V2 operator."""
from dataclasses import dataclass
from threading import RLock
import time
from . import _native


@dataclass
class _Entry:
    binding: tuple
    index: object
    charge: object
    edges: int


class IndexCache:
    def __init__(self):
        self.entry=None
        self.lock=RLock()
        self.builds=self.hits=0

    def get(self,snapshot):
        binding=(snapshot._account,snapshot.binding,id(snapshot._data))
        with self.lock:
            if self.entry is not None and self.entry.binding==binding:
                self.hits+=1
                return self.entry
            self.entry=None
            edges=sum(len(row.targets) for row in snapshot.rows)
            maximum=8192+len(snapshot.rows)*128+edges*48+len(snapshot.message_spans)*128
            charge=snapshot.reserve_scoring_scratch(maximum)
            try:
                index=_native.GraphPropagationIndex(snapshot.rows,snapshot.message_spans,snapshot.message_end_known)
                if index.bytes>maximum:
                    raise BufferError('Native graph index exceeded its reserved capacity')
                self.entry=_Entry(binding,index,charge,edges)
                self.builds+=1
                return self.entry
            except BaseException:
                charge.close()
                raise

    def close(self):
        with self.lock:
            # An active scorer retains its entry/charge independently.
            self.entry=None


def _cache(snapshot):
    owner=snapshot._graph_owner()
    host=owner if owner is not None and not owner.closed else snapshot
    with snapshot._budget._lock:
        cache=getattr(host,'_native_scoring_cache',None)
        if cache is None:
            cache=host._native_scoring_cache=IndexCache()
    return cache


def score_native(raw,starts,eligible,snapshot,*,mode,alpha,cross_message_weight,block_tokens=None):
    from .graph_scoring import GraphScores,_operator_weights
    started=time.perf_counter()
    block=snapshot.block_tokens if block_tokens is None else block_tokens
    if type(block) is not int or block<=0 or any(type(p) is not int for p in starts):
        raise ValueError('Graph scoring requires distinct original block starts')
    cache=_cache(snapshot);entry=cache.get(snapshot)
    wl,wa=_operator_weights(mode,'v2',alpha,0.,0.,cross_message_weight)
    scores,values=entry.index.score(raw,starts,eligible,block,snapshot.completed_through,wl,wa,cross_message_weight)
    diagnostics=dict(mode=mode,scoring_version='v2',alpha=alpha,locality_weight=wl,attention_weight=wa,
        locality_share=.5 if mode=='multi' else 1. if mode=='locality' else 0.,
        attention_share=.5 if mode=='multi' else 1. if mode=='attention' else 0.,
        revision=snapshot.revision,runtime_epoch=snapshot.epoch,accepted_through=snapshot.accepted_through,
        completed_through=snapshot.completed_through,observed_sources=len(snapshot.rows),edges=entry.edges,
        no_sample_sources=snapshot.completed_through//block-len(snapshot.rows),
        partial_source=None if snapshot.partial is None else snapshot.partial.source,
        partial_samples=0 if snapshot.partial is None else snapshot.partial.samples,snapshot_bytes=snapshot.bytes,
        normalized_l1_change=0.,raw_l1_change=0.)
    counts=('locality_fallback_count','locality_missing_count','locality_same_message_count',
        'locality_cross_message_count','locality_masked_count','attention_missing_count','attention_masked_count')
    for name in counts:diagnostics[name]=0
    for name in ('masked_self_mass','missing_self_mass','attention_external_mass','attention_seed_coverage',
            'attention_retained_raw_mass','attention_effective_raw_mass','attention_excluded_raw_mass',
            'locality_external_mass'):
        diagnostics[name]=0.
    diagnostics.update(values)
    for name in (*counts,'eligible'):diagnostics[name]=int(diagnostics[name])
    diagnostics['identity']=bool(diagnostics['identity'])
    total=diagnostics['raw_eligible_sum']
    diagnostics['attention_identity_mass']=(wa if total else 0.)-diagnostics['attention_external_mass']
    diagnostics['locality_identity_mass']=(wl if total else 0.)-diagnostics['locality_external_mass']
    for kind in ('attention_external','locality_external','self_retained'):
        diagnostics['raw_'+kind+'_mass']=diagnostics[kind+'_mass']*total
    edge_count=sum(diagnostics['locality_'+k+'_count'] for k in ('fallback','same_message','cross_message'))
    for category,key in (('unknown','fallback'),('same_message','same_message'),('cross_message','cross_message')):
        diagnostics['locality_'+category+'_fraction']=diagnostics['locality_'+key+'_count']/edge_count if edge_count else 0.
    diagnostics.update(scoring_seconds=time.perf_counter()-started,backend='cpp_v2',
        native_index_bytes=entry.charge.amount,native_index_builds=cache.builds,native_index_hits=cache.hits)
    return GraphScores(raw if diagnostics['identity'] else scores,diagnostics)
