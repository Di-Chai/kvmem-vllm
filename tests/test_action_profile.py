"""CPU diagnostics tests: synthetic events never touch CUDA or a model."""
import copy
import json
from threading import RLock
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.action_profile import ActionProfiler
from vllm_kvmem.options import RuntimeOptions
from vllm_kvmem.serve import build_launch
from test_options import config
from test_serving_launcher import profile as serving_profile


class Events:
    def __init__(self):
        self.ready = False
        self.created = self.recorded = self.queried = 0

    def event(self):
        self.created += 1
        parent = self
        class Event:
            def record(self): parent.recorded += 1
            def query(self):
                parent.queried += 1
                return parent.ready
            def elapsed_time(self, other):
                assert parent.ready
                return 1.25
            def synchronize(self): pytest.fail('Diagnostic introduced a CUDA fence')
        return Event()


def batch(cuda=False):
    return NS(input_ids=NS(device=NS(type='cuda' if cuda else 'cpu')), req_ids=['a','b'])


def profiler(tmp_path, **kwargs):
    events = Events()
    p = ActionProfiler(tmp_path/'profile.jsonl', event_factory=events.event,
                       capturing=lambda:False, **kwargs)
    return p, events


def records(p):
    return [json.loads(line) for line in p.output.read_text().splitlines()]


def test_nested_event_capacity_drops_only_measurement_and_recovers_without_sync(tmp_path):
    p,e = profiler(tmp_path, max_events=1)
    t = p.begin_batch(batch(True)).child(request_id='a',kind='output',layer='L3')
    scored=[]
    with t.stage('score.layer_total',gpu=True):
        with t.stage('score.softmax',gpu=True):scored.append('unchanged score')
    p.finish_batch()
    assert scored==['unchanged score'] and p.reserved_events==1 and len(p.pending)==1
    assert e.created==2 and e.recorded==2
    assert p.counters['gpu_event_drops']==1
    assert not [r for r in records(p) if r['event']=='gpu_stage']
    e.ready=True;p.poll()
    assert p.reserved_events==0 and not p.pending
    gpu=[r for r in records(p) if r['event']=='gpu_stage']
    assert len(gpu)==1 and gpu[0]['stage']=='score.layer_total' and gpu[0]['gpu_elapsed_ms']==1.25


def test_graph_capture_never_allocates_records_or_queries_events(tmp_path):
    p,e=profiler(tmp_path)
    capturing=[False];p.capturing=lambda:capturing[0]
    t=p.begin_batch(batch(True))
    with t.stage('previous',gpu=True):pass
    assert p.pending
    capturing[0]=True
    before=(e.created,e.recorded,e.queried)
    assert p.begin_batch(batch(True)) is None
    with t.stage('inside_graph',gpu=True):pass
    p.poll();p.finish_batch()
    assert (e.created,e.recorded,e.queried)==before
    assert p.counters['graph_capture_skips']>=3
    assert not p.output.exists()
    capturing[0]=False;e.ready=True;p.poll()
    assert p.reserved_events==0


def test_interval_keeps_prepare_submit_capture_same_batch_and_separates_owner_kind(tmp_path):
    p,e=profiler(tmp_path,interval=2)
    b=batch();t=p.begin_batch(b)
    assert p.begin_batch(b) is t
    with t.stage('publication.batch_drain',request_ids=['a','b']):pass
    for rid,kind in [('a','output'),('b','prefill')]:
        with t.child(request_id=rid,kind=kind).stage('metadata.starts_h2d',gpu=True,
                transfer_bytes=32,direction='H2D'):pass
    p.finish_batch()
    assert p.begin_batch(batch()) is None
    p.finish_batch()
    assert p.counters['batches_seen']==2 and p.counters['batches_sampled']==1
    assert p.counters['interval_skips']==1 and e.created==0
    rows=[r for r in records(p) if r['event']=='batch']
    assert len(rows)==1
    assert len([r for r in records(p) if r['event']=='summary'])==1
    p.flush()
    assert len([r for r in records(p) if r['event']=='summary'])==2
    assert {r['kind'] for r in rows[0]['stages'] if 'kind' in r}=={'output','prefill'}
    assert sum(r['transfer_bytes'] for r in p.summary()['totals'])==64


def test_stage_exception_does_not_swallow_work_failure_or_leave_active_reservation(tmp_path):
    p,e=profiler(tmp_path)
    t=p.begin_batch(batch(True))
    with pytest.raises(ValueError,match='real error'):
        with t.stage('score',gpu=True):raise ValueError('real error')
    p.finish_batch(status='error')
    e.ready=True;p.poll()
    assert p.reserved_events==0
    row=next(r for r in records(p) if r['event']=='gpu_stage')
    assert row['status']=='error' and row['error_type']=='ValueError'


def test_default_off_and_explicit_serve_connector_roundtrip(tmp_path):
    assert RuntimeOptions().action_profiling is False
    cfg=serving_profile(tmp_path)
    cfg['kvmem'].update(action_attention=True,action_profiling=True,
                        action_profile_interval=7,action_profile_max_events=31)
    before=copy.deepcopy(cfg)
    plan=build_launch(cfg,tmp_path,tmp_path/'run')
    assert cfg==before
    argv=plan['vllm_argv'];transfer=json.loads(argv[argv.index('--kv-transfer-config')+1])
    opts=RuntimeOptions.from_vllm(config(transfer['kv_connector_extra_config']))
    assert opts.action_profiling and opts.action_profile_interval==7 and opts.action_profile_max_events==31
    assert build_launch(plan['profile'],tmp_path,tmp_path/'again')['profile']==plan['profile']


@pytest.mark.parametrize('values,match',[
    ({'action_profiling':1},'boolean'),
    ({'action_profiling':True},'requires action_attention'),
    ({'action_profile_interval':False},'positive integer'),
    ({'action_profile_interval':0},'positive integer'),
    ({'action_profile_max_events':0},'positive integer'),
    ({'action_profile_max_events':1.5},'positive integer'),
])
def test_invalid_profile_options_fail_closed(values,match):
    with pytest.raises(ValueError,match=match):RuntimeOptions.from_vllm(config({'kvmem':values}))


def test_submit_batch_fence_is_preserved_and_distinct_from_owner_measurement(tmp_path):
    from vllm_kvmem.capture_runtime import CaptureRuntime
    p,_=profiler(tmp_path)
    b=batch();b.num_reqs=2
    r=CaptureRuntime.__new__(CaptureRuntime)
    r._raise_publication_error=lambda:None
    calls=[]
    r.drain_requests=lambda ids,wait:calls.append((tuple(ids),wait)) or True
    r.names=();r._capture_groups=lambda _:[];r.split_capture_dma=True
    r._publication_lock=lambda:RLock();r.batches=0;r.options=NS(query_history=False)
    def capture(*_):
        t=p.begin_batch(b)
        with t.child(request_id='a',kind='output').stage('publication.owner_drain'):
            r.drain_requests(('a',),wait=True)
        p.finish_batch();return {}
    r.action=NS(profile=p,capture=capture)
    r.submit([],b,None)
    assert calls==[(('a','b'),True),(('a',),True)] and r.batches==1
    stages=next(x['stages'] for x in records(p) if x['event']=='batch')
    assert [s['stage'] for s in stages]==['publication.batch_drain','publication.owner_drain']


def test_page_attention_instrumentation_preserves_math_and_disabled_uses_no_clocks(tmp_path,monkeypatch):
    # CPU replacements only for the two Triton launches; all surrounding tensor
    # preparation, RoPE, softmax and profile branches are the production code.
    import vllm_kvmem.action_scoring as scoring
    class Kernel:
        def __init__(self,fn):self.fn=fn
        def __getitem__(self,grid):return self.fn
    def logits(q,key,table,pos,scale,out,*args,**kwargs):
        # Deterministic masked logits are sufficient to check unchanged dispatch
        # arguments/reductions, without pretending to qualify the CUDA kernel.
        length=args[9]
        out.fill_(-torch.inf)
        for row in range(q.shape[0]):
            upto=min(length,int(pos[row])+1)
            out[row,:,:upto]=q[row].float().sum(-1)[:,None]+torch.arange(upto)[None,:]/10
    def mass(prob,starts,counts,out,heads,width,*args):
        for i,(a,n) in enumerate(zip(starts,counts)):
            out[:,i]=prob[:,:,int(a):int(a+n)].sum(-1).mean(1)
    monkeypatch.setattr(scoring,'_paged_logits',Kernel(logits))
    monkeypatch.setattr(scoring,'_segment_mass',Kernel(mass))
    q=torch.arange(16,dtype=torch.float32).reshape(2,2,4).to(torch.bfloat16)/8
    pos=torch.tensor([3,6]);key=torch.zeros((2,4,1,4));table=torch.tensor([0,1])
    trig=torch.cat((torch.ones(7,2),torch.zeros(7,2)),1)
    starts=torch.tensor([0,4],dtype=torch.int32);counts=torch.tensor([4,3],dtype=torch.int32)
    args=(q,pos,key,table,trig,torch.tensor(1.),starts,counts,7)
    kw=dict(scale=.5,q_dtype=torch.bfloat16,q_scale=torch.tensor(1.),block_tokens=4)
    expected=scoring.page_attention(*args,**kw)
    p,e=profiler(tmp_path);t=p.begin_batch(batch()).child(request_id='a',kind='prefill',layer='L3')
    actual=scoring.page_attention(*args,**kw,profile=t);p.finish_batch()
    assert torch.equal(expected,actual) and e.created==0
    stages={s['stage'] for r in records(p) if r['event']=='batch' for s in r['stages']}
    assert {'score.rope','score.q_codec','score.logits','score.softmax','score.segment_mass'} <= stages
    def no_clock():pytest.fail('Disabled scoring used the diagnostic clock')
    monkeypatch.setattr('vllm_kvmem.action_profile.time.perf_counter',no_clock)
    monkeypatch.setattr('vllm_kvmem.action_profile.time.thread_time',no_clock)
    assert torch.equal(scoring.page_attention(*args,**kw),expected)


def test_mixed_owner_capture_preserves_packets_and_accepted_feedback(tmp_path,monkeypatch):
    from vllm_kvmem.action_attention import ActionRuntime
    import vllm_kvmem.action_attention as action
    import vllm_kvmem.action_scoring as scoring
    # Preserve actual parser, metadata, window mapping and feedback commit.
    # Replace GPU-only storage/calls; GPU numerical qualification is separate.
    empty=torch.empty
    monkeypatch.setattr(action.torch,'empty',lambda *a,**k:empty(*a,**{n:v for n,v in k.items() if n!='pin_memory'}))
    fences=[]
    class Event:
        def record(self):fences.append('record')
        def synchronize(self):fences.append('original accept fence')
    monkeypatch.setattr(action.torch.cuda,'Event',Event)
    monkeypatch.setattr(action,'native_vllm_fp8_views',lambda *_:(torch.zeros(1,4,1,4),None))
    launches=[]
    def score(q,pos,key,table,trig,ks,starts,counts,length,**kwargs):
        launches.append((q.tolist(),pos.tolist(),starts.tolist(),counts.tolist(),length))
        return torch.ones((q.shape[0],starts.numel()))/starts.numel()
    monkeypatch.setattr(scoring,'page_attention',score)
    opts=NS(prefill_attention=True,prefill_attention_interval=1,prefill_attention_stride=1,
            prefill_attention_half_life=128,native_prefill_commit=False,
            memory=NS(select_budget=16,gen_budget=8,block_tokens=4),
            action_scope='non-thinking',action_interval=1,action_decay=.5)
    results=[]
    for enabled in (False,True):
        calls=[]
        owners={rid:NS(memory=NS(_store=NS(working_records=lambda:[(0,1,0,4,0)])))
                for rid in ('a','b')}
        r=ActionRuntime.__new__(ActionRuntime)
        r.options=opts;r.profile=profiler(tmp_path)[0] if enabled else None
        r.runtime=NS(options=opts,config=NS(block_tokens=4),output=tmp_path,
            drain_requests=lambda ids,wait:calls.append((tuple(ids),wait)) or True,
            ensure_authority=owners.__getitem__)
        r.runner=NS(req_states=NS(req_id_to_index={'a':0,'b':1}),
            _kvmem_sessions=NS(bindings={'b':NS(tokens=[14,13])}),
            block_tables=NS(block_tables=[NS(gpu=torch.tensor([[0,1],[0,1]]))]))
        module=NS(attn=NS(layer_name='L3',kv_cache=None,_k_scale=torch.tensor(1.),
            _q_scale=torch.tensor(1.),impl=NS(scale=.5)),num_kv_heads=1,head_dim=4,
            rotary_emb=NS(cos_sin_cache=None))
        r.layers=[(0,module)];r.groups={'L3':0}
        r.q_dtypes=r.prefill_q_dtypes={'L3':torch.bfloat16}
        r.ids=[10,11,12,13,14];r.metadata_early_batches=0;r.metadata=None
        r.candidate_samples=r.packets=0;r.capture_seconds=0.
        b=NS(num_reqs=2,req_ids=['a','b'],is_prefilling_np=[True,False],
            query_start_loc_np=[0,2,4],input_ids=torch.tensor([20,21,22,23]),
            positions=torch.tensor([8,9,8,9]))
        captured=[NS(query=torch.arange(32).reshape(4,2,4).to(torch.bfloat16),
                     positions=torch.tensor([4,5,4,5]))]
        r.prepare(b,captured[0].positions)
        packets=r.capture(captured,b)
        frozen=[(rid,p.rows,p.history,p.masses.tolist(),p.prefill) for rid,p in packets.items()]
        for rid in ('a','b'):r.accept(NS(action_inputs=packets),rid,2)
        results.append((frozen,calls,owners['a'].action_history.prefill.snapshot(),
                        owners['b'].action_history.mass,owners['b'].action_history.samples))
    assert results[0]==results[1]
    assert launches[:2]==launches[2:] and fences.count('original accept fence')==4
    rows=records(r.profile)
    record=next(x for x in rows if x['event']=='batch')
    assert record['request_kinds']=={'a':'prefill','b':'output'}
    stages=record['stages']
    assert {s['kind'] for s in stages if s['stage']=='score.layer_total'}=={'prefill','output'}
    assert len([s for s in stages if s['stage']=='publication.owner_drain'])==2
    assert sum(s.get('transfer_bytes',0) for s in stages if s['stage']=='mass.d2h')==32
