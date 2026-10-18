"""Private rewrites publish only qw3-compatible, naturally terminated Query Q."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.guided_query import GuidedQueryPolicy, collect_guided_query


class Tokenizer:
    eos_token_id = 1000
    unk_token_id = -1

    def __init__(self,pieces):
        self.pieces = {index+1:piece for index,piece in enumerate(pieces)}

    def convert_tokens_to_ids(self,token):
        return {'<|im_end|>':1001,'</think>':1002}.get(token,self.unk_token_id)

    def decode(self,tokens,skip_special_tokens=False):
        return ''.join(self.pieces[token] for token in tokens)


def stream(pieces,stop=1000):
    tokenizer = Tokenizer(pieces)
    sequence = list(tokenizer.pieces)+[stop]
    calls = []
    def forward(tokens,begin):
        assert tokens==[sequence[len(calls)]]
        calls.append((tokens[0],begin))
        return sequence[len(calls)],torch.full((2,1,1),float(tokens[0]))
    return tokenizer,sequence[0],calls,forward


@pytest.mark.parametrize('leading',['\n',' ', '"',"'",'`','“','‘',''])
def test_direct_prefix_is_consumed_but_not_scored_and_internal_newline_is_content(leading):
    tokenizer,first,calls,forward=stream([leading,'Recover','\n',' ledger-71'])
    query,report=collect_guided_query(tokenizer,8,first,120,forward,direct=True)
    assert query[0,:,0].tolist()==[2.,3.,4.]
    assert calls==[(1,120),(2,121),(3,122),(4,123)]
    assert report['guided_query_tokens']==report['guided_consumed_tokens']==4
    assert report['guided_captured_tokens']==report['guided_published_tokens']==3
    assert report['guided_ignored_prefix_tokens']==1
    assert report['guided_natural_termination'] and report['guided_fallback'] is None


def test_existing_non_direct_prompt_keeps_quote_rows_like_qw3_zero_thinking_mode():
    tokenizer,first,calls,forward=stream(['"','Recover the ledger','"'])
    query,report=collect_guided_query(tokenizer,8,first,120,forward)
    assert query[0,:,0].tolist()==[1.,2.,3.]
    assert report['guided_ignored_prefix_tokens']==0 and report['guided_fallback'] is None


@pytest.mark.parametrize('stop',[1000,1001,1002])
def test_all_natural_boundaries_stop_before_consumption(stop):
    tokenizer,first,calls,forward=stream(['Recover the original ledger'],stop)
    query,report=collect_guided_query(tokenizer,8,first,64,forward)
    assert len(calls)==1 and query.shape[1]==1
    assert report['guided_natural_termination'] and report['guided_fallback'] is None


@pytest.mark.parametrize('pieces',[
    ['  ','{','"answer": 1}'],['[','"answer"',']'],['<ANSWER','>text</answer>'],
    ['tool_CALL','(arg)'],['<Tool','_call>'],['```','python']])
def test_structured_rewrite_discards_every_staged_row(pieces):
    tokenizer,first,calls,forward=stream(pieces)
    query,report=collect_guided_query(tokenizer,32,first,64,forward)
    assert query is None and len(calls)==len(pieces)
    assert report['guided_fallback']=='structured-output'
    assert report['guided_published_tokens']==0 and report['guided_captured_tokens']>0


@pytest.mark.parametrize('pieces',[['  '],[''],['"',' ']])
def test_empty_or_separator_only_rewrite_falls_back(pieces):
    tokenizer,first,calls,forward=stream(pieces)
    query,report=collect_guided_query(tokenizer,32,first,64,forward,direct=True)
    assert query is None and report['guided_fallback']=='empty-output'
    assert report['guided_natural_termination'] and not report['guided_captured_tokens']


@pytest.mark.parametrize('pieces,limit',[
    (['Recover',' records'],1),(['Recover'],1),(['\n','Recover',' records'],2)])
def test_cap_includes_ignored_prefix_and_never_publishes_truncated_or_cap_boundary_query(pieces,limit):
    tokenizer,first,calls,forward=stream(pieces)
    query,report=collect_guided_query(tokenizer,limit,first,64,forward)
    assert query is None and len(calls)==limit
    assert report['guided_fallback']=='unterminated-output'
    assert not report['guided_natural_termination'] and report['guided_published_tokens']==0
    assert max(position for _,position in calls)<64+limit


def test_prefix_guard_stops_without_forwarding_seventeenth_separator():
    tokenizer,first,calls,forward=stream(['\n']*17+['Recover'])
    query,report=collect_guided_query(tokenizer,64,first,64,forward)
    assert query is None and len(calls)==16
    assert report['guided_fallback']=='empty-output' and report['guided_ignored_prefix_tokens']==16


def test_only_successful_private_transition_commits_policy_state():
    policy=GuidedQueryPolicy(Tokenizer(['Recover']),8)
    assert policy.prepare(1)
    assert policy.consumed==policy.captured==[]
    with pytest.raises(RuntimeError,match='prepared'):
        policy.prepare(1)
    with pytest.raises(RuntimeError,match='completed'):
        policy.report()
    policy.commit()
    with pytest.raises(RuntimeError,match='no pending'):
        policy.commit()
    assert policy.prepare(1000) is None and policy.report()['guided_published_tokens']==1


def test_unknown_token_id_is_not_an_accidental_stop_marker():
    tokenizer=Tokenizer(['Recover','ordinary unknown token'])
    tokenizer.unk_token_id=2
    tokenizer.convert_tokens_to_ids=lambda marker:2
    policy=GuidedQueryPolicy(tokenizer,8)
    assert policy.prepare(2)
    policy.commit()
    assert policy.prepare(1000) is None and policy.report()['guided_published_tokens']==1


@pytest.mark.parametrize('pieces,limit,stop',[
    (['\n','"','Recover','\n',' ledger'],16,1000),
    (['Recover',' ledger'],1,1000),
    (['{','"answer": true}'],16,1001),
    (['  ']*17+['Recover'],32,1000),
    (['Recover'],16,1002)])
def test_cooperative_and_sync_policy_publish_identical_rows_after_restore_fence(monkeypatch,pieces,limit,stop):
    pytest.importorskip('vllm')
    from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
    from vllm_kvmem.connector import RefreshCommand
    tokenizer,first,sync_calls,forward=stream(pieces,stop)
    expected,diagnostics=collect_guided_query(tokenizer,limit,first,68,forward)
    tokenizer,first,async_calls,forward=stream(pieces,stop)
    public=torch.tensor([[[700.],[701.]],[[800.],[801.]]])
    original=public.clone()
    events=[]
    class Event:
        def record(self):events.append('ready')
    monkeypatch.setattr(torch.cuda,'Event',Event)
    obj=AsyncRefresh.__new__(AsyncRefresh)
    obj.stream=object();obj.guided_prompt=[90,91]
    obj.options=NS(guided_query_tokens=limit)
    obj.emit=lambda *a,**k:None
    obj.adapter=NS(validate=lambda bundle:events.append('validated'))
    obj._forward=lambda job,tokens,begin,**kwargs:forward(tokens,begin)
    ticket=NS(history_end=64)
    command=RefreshCommand('r',1,((0,ticket),),input_tokens=(10,20),asynchronous=True)
    job=RefreshJob(command,stage='guided_decode',next_token=first,
        query_rows=[public],guided_policy=GuidedQueryPolicy(tokenizer,limit))
    def restore(before,tensors):
        events.append('restored')
        return object()
    job.state=NS(tensors=object()),NS(restore=restore),object()
    for _ in range(len(pieces)+2):
        assert obj.step(job)
        if job.stage=='assembly_wait':break
        assert not events and torch.equal(public,original)
    assert job.stage=='assembly_wait' and events==['validated','restored','ready']
    assert job.guided_diagnostics==diagnostics and async_calls==sync_calls
    assert torch.equal(public,original)
    assert torch.equal(job.assembly_inputs[0],original if expected is None else expected)
    if expected is None:assert not job.guided_rows


def test_cooperative_report_preserves_consumed_count_and_adds_distinct_capture_counts(tmp_path):
    pytest.importorskip('vllm')
    from vllm_kvmem.async_refresh import AsyncRefresh, RefreshJob
    from vllm_kvmem.connector import RefreshCommand
    import json
    obj=AsyncRefresh.__new__(AsyncRefresh)
    obj.options=NS(memory=NS(action_attention=False,action_combine='quota'),mean_k_backend='torch_reference')
    obj.runner=NS(_kvmem_probe_output=tmp_path)
    authority=NS(records={},memory=NS(_store=NS(working_records=lambda:[])))
    job=RefreshJob(RefreshCommand('r',1,((0,NS(history_end=64)),),input_tokens=(10,20)),
        authority=authority,query_rows=[torch.ones((1,2,1))],generated=[1,2,3],
        guided_diagnostics=dict(guided_query_tokens=3,guided_consumed_tokens=3,
            guided_captured_tokens=2,guided_published_tokens=0,guided_ignored_prefix_tokens=1,
            guided_natural_termination=False,guided_fallback='unterminated-output'))
    assembly=NS(host_prepare_seconds=0,uploaded_bytes=0,raw_cache_hit_tokens=0,resident_v_reused_tokens=0)
    obj.report(job,assembly,{},0)
    report=json.loads((tmp_path/'bounded_refresh.jsonl').read_text())
    assert report['guided_query_tokens']==3 and report['guided_captured_tokens']==2
    assert report['guided_fallback']=='unterminated-output' and report['query_score_tokens']==2
