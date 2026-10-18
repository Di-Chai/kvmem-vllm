"""CPU fakes only: never import vLLM, torch, or a GPU driver."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import weakref

import pytest


PATH=Path(__file__).resolve().parents[1]/'diagnostics/native_executed_batch_observer.py'
SPEC=importlib.util.spec_from_file_location('native_batch_observer_test',PATH)
o=importlib.util.module_from_spec(SPEC);sys.modules[SPEC.name]=o;SPEC.loader.exec_module(o)


class Sink:
    def __init__(self,directory):
        self.directory=directory;directory.mkdir(parents=True);self.events=[]
    def emit(self,event):self.events.append(event)
    def snapshot(self):return dict(dropped=0,io_errors=0,queued=0,writer_alive=False)
    def close(self):return self.snapshot()


@pytest.fixture
def fixture(tmp_path):
    class Timing:
        def __init__(self):self.collecting=False;self.calls=0
        def record_batch(self,input_batch,full_cudagraph):self.calls+=1
    class State:
        def __init__(self):self.req_id_to_index={};self.prompt_len=NS(np=[0]*32)
        def add_request(self,req_id,prompt_len):
            index=len(self.req_id_to_index);self.req_id_to_index[req_id]=index;self.prompt_len.np[index]=prompt_len
    class Output:
        def __init__(self,result,fail=False):self.result=result;self.fail=fail;self.calls=0
        def get_output(self):
            self.calls+=1
            if self.fail:raise RuntimeError('native output failure')
            return self.result
    class Runner:
        def __init__(self):
            self.parallel_config=NS(tensor_parallel_size=1,pipeline_parallel_size=1,data_parallel_size=1)
            self.dcp_size=1;self.pcp_manager=None;self.ubatch_runner=None;self.is_pooling_model=False
            self.lora_config=None;self.uses_inputs_embeds=False;self.adaptive_verification=None
            self.vllm_config=NS(kv_transfer_config=None);self.req_states=State();self.step_timing=Timing()
            self.execute_model_state=None;self.fail_sample=False;self.fail_output=False
        def execute_model(self,batch,intermediate_tensors=None,dummy_run=False,skip_attn_for_dummy_run=False,
                          is_profile=False,context_len=0,valid_dummy_state_slots=False):
            if batch is None:return NS(no_forward=True)
            self.step_timing.record_batch(batch,True)
            if getattr(batch,'fail_execute',False):raise RuntimeError('native execute failure')
            self.execute_model_state=NS(input_batch=batch)
            return None
        def sample_tokens(self,grammar=None):
            state=self.execute_model_state;self.execute_model_state=None
            if self.fail_sample:raise RuntimeError('native sample failure')
            if state is None:return None
            return Output(NS(req_ids=state.input_batch.req_ids.copy(),
                sampled_token_ids=[[7,8] for _ in state.input_batch.req_ids]),self.fail_output)
    class Worker:
        def __init__(self):self.model_runner=Runner();self.closed=False
        def compile_or_warm_up_model(self):
            # Native startup uses actual execute/sample calls, with dummy=False.
            self.model_runner.req_states.add_request('synthetic',1)
            self.model_runner.execute_model(batch(self.model_runner,('synthetic',),(False,)))
            self.model_runner.sample_tokens().get_output()
            return 'compiled'
        def shutdown(self):self.closed=True;return 'shutdown'
    classes=NS(Worker=Worker,Runner=Runner,Timing=Timing,AsyncOutput=Output,RequestState=State)
    undo=o.install(classes,dict(enabled=True,output_directory=str(tmp_path/'logs')),Sink)
    worker=Worker()
    assert not hasattr(worker.model_runner,o.ATTR)
    assert worker.compile_or_warm_up_model()=='compiled'
    runner=worker.model_runner
    # Retire the fake synthetic state before public requests.
    runner.req_states.req_id_to_index.clear()
    for req,length in [('a',100),('b',200),('c',300)]:runner.req_states.add_request(req,length)
    try:yield worker,runner,getattr(runner,o.ATTR)
    finally:undo()


def batch(runner,ids=('a','b'),prefill=(False,True)):
    slots=[runner.req_states.req_id_to_index[x] for x in ids]
    lengths=[runner.req_states.prompt_len.np[s] for s in slots]
    return NS(req_ids=list(ids),idx_mapping_np=slots,num_reqs=len(ids),num_reqs_after_padding=4,
        is_prefilling_np=list(prefill),prefill_len_np=lengths,
        num_computed_prefill_tokens_np=[n-2 if p else n for n,p in zip(lengths,prefill)],
        num_scheduled_tokens=[2 if p else 5 for p in prefill],num_draft_tokens_per_req=[0 if p else 4 for p in prefill],
        num_tokens=sum(2 if p else 5 for p in prefill),num_tokens_after_padding=16)


def test_two_inflight_reverse_completion_immutable_ids_no_getter_cycle(fixture):
    worker,runner,obs=fixture
    first=batch(runner);runner.execute_model(first);a=runner.sample_tokens()
    frozen=obs.writer.events[0]['batch']
    first.req_ids[:]=['changed','changed'];first.is_prefilling_np[:]=[True,False]
    first.num_scheduled_tokens[:]=[99,99]
    runner.execute_model(batch(runner,('c',),(False,)));b=runner.sample_tokens()
    assert obs.snapshot()['completed']==0 and obs.snapshot()['pending']==2
    assert runner.step_timing.collecting is False and runner.step_timing.calls==3
    assert frozen.req_ids==('a','b') and frozen.is_prefilling==(False,True)
    assert frozen.scheduled_token_upper_counts==(5,2) and frozen.slot_generations==(1,1)
    b.get_output();a.get_output();a.get_output()  # Native repeated read does not count twice.
    assert [e['execution_id'] for e in obs.writer.events if e['kind']=='completed']==[2,1]
    assert obs.snapshot()['completed']==2 and not obs.pending
    assert 'get_output' not in a.__dict__ and not hasattr(a,o.TICKET)
    ref=weakref.ref(a);del a;gc.collect();assert ref() is None
    assert worker.shutdown()=='shutdown'
    assert json.loads((obs.writer.directory/'summary.json').read_text())['complete_coverage'] is True


@pytest.mark.parametrize('stage',['execute','sample','output'])
def test_native_failure_is_propagated_and_never_counted_executed(fixture,stage):
    _,runner,obs=fixture
    request=batch(runner)
    if stage=='execute':request.fail_execute=True
    if stage=='sample':runner.fail_sample=True
    if stage=='output':runner.fail_output=True
    with pytest.raises(RuntimeError,match='native '+stage+' failure'):
        runner.execute_model(request);output=runner.sample_tokens();output.get_output()
    assert obs.snapshot()['completed']==0 and obs.snapshot()['failed']==1 and not obs.pending
    assert all(e.get('executed') is not True for e in obs.writer.events)


def test_dummy_profile_and_no_forward_are_excluded_without_enabling_step_timing(fixture):
    _,runner,obs=fixture
    runner.execute_model(batch(runner),dummy_run=True);runner.sample_tokens().get_output()
    runner.execute_model(batch(runner),is_profile=True);runner.sample_tokens().get_output()
    runner.execute_model(None)
    assert obs.snapshot()['dispatches']==0 and runner.step_timing.calls==3
    assert not runner.step_timing.collecting


def test_wrong_state_and_wrong_output_fail_closed(fixture):
    _,runner,obs=fixture
    runner.execute_model(batch(runner));runner.execute_model_state=NS(input_batch=batch(runner))
    runner.sample_tokens().get_output()
    assert obs.snapshot()['failed']==1
    runner.execute_model(batch(runner));output=runner.sample_tokens();output.result.req_ids.reverse();output.get_output()
    assert obs.snapshot()['failed']==2 and obs.snapshot()['completed']==0


def test_bounded_inflight_drops_only_diagnostics_and_never_model_output(fixture):
    _,runner,obs=fixture
    obs.max_pending=1
    runner.execute_model(batch(runner));a=runner.sample_tokens()
    runner.execute_model(batch(runner,('c',),(False,)));b=runner.sample_tokens()
    assert b.get_output().sampled_token_ids==[[7,8]]
    assert obs.snapshot()['pending']==1 and obs.snapshot()['dropped']==1
    a.get_output();assert obs.snapshot()['completed']==1


def test_missing_sample_and_malformed_cpu_snapshot_do_not_change_native_result(fixture):
    _,runner,obs=fixture
    runner.execute_model(batch(runner))
    runner.execute_model(batch(runner));runner.sample_tokens().get_output()
    assert obs.snapshot()['failed']==1 and obs.snapshot()['completed']==1
    malformed=batch(runner);malformed.num_scheduled_tokens=[1]
    runner.execute_model(malformed);assert runner.sample_tokens().get_output().req_ids==['a','b']
    assert obs.snapshot()['observer_errors']==1 and obs.snapshot()['completed']==1


def test_reused_slot_generation_does_not_change_pending_request_identity(fixture):
    _,runner,obs=fixture
    runner.execute_model(batch(runner,('c',),(False,)));old=runner.sample_tokens()
    runner.req_states.req_id_to_index.pop('c')
    runner.req_states.add_request('replacement',400)
    runner.execute_model(batch(runner,('replacement',),(False,)));new=runner.sample_tokens()
    batches=[e['batch'] for e in obs.writer.events if e['kind']=='dispatch']
    assert batches[0].slots==batches[1].slots==(2,)
    assert batches[0].slot_generations==(1,) and batches[1].slot_generations==(2,)
    assert batches[0].req_ids==('c',)
    new.get_output();old.get_output();assert obs.snapshot()['completed']==2


def test_writer_disk_bound_and_closed_queue(tmp_path):
    writer=o.Writer(tmp_path/'writer',capacity=8,max_records=1,max_bytes=100,max_record_bytes=100)
    writer.emit({'kind':'one'});writer.emit({'kind':'two'})
    summary=writer.close()
    assert summary['written']==1 and summary['dropped']==1 and summary['bytes']<=100
    writer.emit({'kind':'after-close'})
    assert writer.snapshot()['dropped']==2 and writer.snapshot()['queued']==0


def test_disabled_and_unsupported_config_fail_before_observation(fixture):
    _,runner,_=fixture
    runner.adaptive_verification=object()
    with pytest.raises(ValueError,match='Adaptive'):o.validate_runner(runner)
    with pytest.raises(ValueError,match='explicitly enabled'):o.install(NS(),{'enabled':False})


def test_seal_rpc_rejects_pending_sample_and_pending_output_without_disarming(fixture):
    worker,runner,obs=fixture
    runner.execute_model(batch(runner))
    with pytest.raises(ValueError,match='pending'):
        o.Extension.native_executed_batch_seal(worker)
    assert obs.armed and obs.pending_sample is not None
    output=runner.sample_tokens()
    with pytest.raises(ValueError,match='pending'):
        o.Extension.native_executed_batch_seal(worker)
    assert obs.armed and not obs.explicit_seal
    output.get_output()
    summary=o.Extension.native_executed_batch_seal(worker)
    assert summary['complete_coverage'] and summary['explicit_seal'] and not summary['armed']
    assert json.loads((obs.writer.directory/'summary.json').read_text())==summary
    assert o.Extension.native_executed_batch_seal(worker)==summary


def test_seal_rejects_untracked_runner_state_and_execute_frame(fixture):
    worker,runner,obs=fixture
    runner.execute_model_state=NS(input_batch=None)
    with pytest.raises(ValueError,match='target execution'):
        o.Extension.native_executed_batch_seal(worker)
    runner.execute_model_state=None
    frame=obs.begin(True)
    try:
        with pytest.raises(ValueError,match='target execution'):obs.seal()
    finally:obs.end(frame)
    assert obs.armed and not obs.explicit_seal


def test_post_seal_target_invalidates_coverage_without_changing_native_output(fixture):
    worker,runner,obs=fixture
    runner.execute_model(batch(runner));runner.sample_tokens().get_output()
    assert o.Extension.native_executed_batch_seal(worker)['complete_coverage']
    runner.execute_model(batch(runner),dummy_run=True);runner.sample_tokens().get_output()
    runner.execute_model(None)
    assert o.Extension.native_executed_batch_snapshot(worker)['complete_coverage']
    runner.execute_model(batch(runner,('c',),(False,)))
    assert runner.sample_tokens().get_output().sampled_token_ids==[[7,8]]
    invalid=o.Extension.native_executed_batch_snapshot(worker)
    assert not invalid['complete_coverage'] and invalid['post_close_dispatches']==1
    assert invalid['dispatches']==invalid['completed']==1
    assert not o.Extension.native_executed_batch_seal(worker)['complete_coverage']
    worker.shutdown()
    assert not json.loads((obs.writer.directory/'summary.json').read_text())['complete_coverage']


def test_seal_writer_failure_is_not_reported_as_complete(fixture,monkeypatch):
    worker,runner,obs=fixture
    runner.execute_model(batch(runner));runner.sample_tokens().get_output()
    monkeypatch.setattr(obs.writer,'snapshot',lambda:dict(dropped=0,io_errors=0,queued=1,writer_alive=True))
    summary=o.Extension.native_executed_batch_seal(worker)
    assert summary['explicit_seal'] and not summary['complete_coverage']


def test_seal_summary_persist_failure_invalidates_proof(fixture,monkeypatch):
    worker,runner,obs=fixture
    runner.execute_model(batch(runner));runner.sample_tokens().get_output()
    def broken(*args,**kwargs):raise OSError('diagnostic disk failure')
    monkeypatch.setattr(Path,'replace',broken)
    with pytest.raises(OSError,match='diagnostic disk failure'):
        o.Extension.native_executed_batch_seal(worker)
    assert obs.snapshot()['observer_errors']==1 and not obs.snapshot()['complete_coverage']
