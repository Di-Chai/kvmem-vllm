"""Private decode inputs and graph safety, without loading a model."""
from dataclasses import dataclass
import os

import numpy as np
import pytest
import torch

from vllm_kvmem.guided_graph import (PrivateDecodeGraph, UnsupportedGuidedGraph,
                                      metadata_signature, restore_tensors)
from vllm_kvmem.guided_query import update_private_decode


@pytest.mark.parametrize('backend',['eager','cuda_graph'])
def test_backend_ablation_survives_launch_profile_and_worker_options(tmp_path,backend):
    import json
    from test_serving_launcher import nvfp4_profile
    from vllm_kvmem.serve import build_launch
    cfg=nvfp4_profile(tmp_path);cfg['kvmem']['guided_query_backend']=backend
    plan=build_launch(cfg,tmp_path,tmp_path/'run')
    argv=plan['vllm_argv']
    transfer=json.loads(argv[argv.index('--kv-transfer-config')+1])
    assert transfer['kv_connector_extra_config']['kvmem']['guided_query_backend']==backend
    replay=build_launch(plan['profile'],tmp_path,tmp_path/'replay')
    assert replay['profile']['kvmem']['guided_query_backend']==backend


@pytest.mark.parametrize('backend',[None,False,1,'auto','dflash'])
def test_backend_rejects_ambiguous_values(backend):
    from vllm_kvmem.options import RuntimeOptions
    with pytest.raises(ValueError,match='Guided Query backend'):
        RuntimeOptions(guided_query_backend=backend)


def test_reused_inputs_advance_actual_lengths_without_changing_capture_bound():
    from types import SimpleNamespace
    batch = SimpleNamespace(input_ids=torch.zeros(1,dtype=torch.int32),
        positions=torch.zeros(1,dtype=torch.int64),seq_lens=torch.zeros(1,dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.zeros(1,dtype=torch.int32),
        num_computed_tokens_np=np.zeros(1,dtype=np.int32),
        prefill_len_np=np.zeros(1,dtype=np.int32),
        num_computed_prefill_tokens_np=np.zeros(1,dtype=np.int32))
    addresses = [v.data_ptr() for v in vars(batch).values() if isinstance(v,torch.Tensor)]
    for token,position in ((17,100),(23,101),(31,132)):
        update_private_decode(batch,token,position,200)
        assert batch.input_ids.item()==token and batch.positions.item()==position
        assert batch.seq_lens.item()==position+1 and batch.seq_lens_cpu_upper_bound.item()==200
        assert batch.num_computed_tokens_np.tolist()==[position]
        assert batch.prefill_len_np.tolist()==[position+1]
        assert batch.num_computed_prefill_tokens_np.tolist()==[position]
        assert addresses==[v.data_ptr() for v in vars(batch).values() if isinstance(v,torch.Tensor)]


def test_graph_rejects_host_tensors_and_backend_planner_objects():
    with pytest.raises(UnsupportedGuidedGraph,match='cpu-attention-tensor'):
        metadata_signature({'lengths':torch.ones(1)})
    with pytest.raises(UnsupportedGuidedGraph,match='attention-object-object'):
        metadata_signature({'wrapper':object()})


def test_graph_layout_signature_includes_scalar_kernel_extent():
    @dataclass
    class Metadata:
        max_seq_len:int
        num_decodes:int
    assert metadata_signature(Metadata(200,1))!=metadata_signature(Metadata(201,1))
    assert metadata_signature({'a':Metadata(200,1)})==metadata_signature({'a':Metadata(200,1)})


def test_warmup_budget_falls_back_before_any_cuda_work(monkeypatch):
    graph=PrivateDecodeGraph()
    monkeypatch.setattr(graph,'MAX_SNAPSHOT_BYTES',0)
    state=torch.tensor([7.])
    graph.capture(lambda:pytest.fail('must not execute a model'),{}, {}, {}, {'state':state})
    assert graph.graph is None and graph.fallback=='warmup-state-memory-limit'
    assert state.item()==7.


def test_warmup_restore_is_byte_exact_and_schema_failure_is_atomic():
    live={'conv':torch.tensor([-0.,float('nan')]),'accepted':torch.tensor([3])}
    saved={k:v.clone() for k,v in live.items()}
    live['conv'].zero_();live['accepted'].zero_()
    restore_tensors(live,saved)
    assert all(torch.equal(v.view(torch.uint8),saved[k].view(torch.uint8)) for k,v in live.items())
    invalid={**saved,'accepted':torch.ones(2,dtype=torch.int64)}
    live['conv'].fill_(7)
    with pytest.raises(ValueError,match='recurrent schema'):
        restore_tensors(live,invalid)
    assert live['conv'].tolist()==[7.,7.]


def test_fresh_graph_has_no_previous_owner_outputs_or_replays():
    graph=PrivateDecodeGraph()
    assert graph.graph is None and graph.outputs is None and graph.replays==0
    assert graph.report()['guided_execution_mode']=='eager'
    graph.close()


@pytest.mark.skipif(os.environ.get('KVMEM_TEST_GUIDED_GPU')!='1',reason='explicit CUDA component gate')
@pytest.mark.parametrize('measure_gpu',[False,True])
def test_cuda_replay_restores_warmup_and_does_not_publish_stale_outputs(measure_gpu):
    value=torch.tensor([10.],device='cuda')
    token=torch.tensor([2],device='cuda')
    length=torch.tensor([1],device='cuda')
    metadata={'length':length}
    inputs={'token':token}
    def run():
        value.add_(token)
        return token+1,(value*length).reshape(1,1,1)
    graph=PrivateDecodeGraph(measure_gpu=measure_gpu)
    try:
        graph.capture(run,metadata,inputs,{}, {'recurrent':value})
        assert value.item()==10.,'warmup/capture must not consume a real transition'
        next_token,first=graph.replay(metadata,inputs,{})
        assert next_token==3 and first.item()==12.
        token.fill_(4);length.fill_(2)
        next_token,second=graph.replay(metadata,inputs,{})
        assert next_token==5 and second.item()==32. and first.item()==12.
        assert graph.replay({'length':length.clone()},inputs,{}) is None
        assert value.item()==16.,'a changed metadata address must not execute the graph'
        assert graph.report()['guided_graph_replays']==2
        assert (graph.report()['guided_graph_sampled_gpu_ms'] is not None)==measure_gpu
    finally:
        graph.close()
    assert graph.graph is None and graph.outputs is None


@pytest.mark.skipif(os.environ.get('KVMEM_TEST_GUIDED_GPU')!='1',reason='explicit CUDA component gate')
def test_cuda_failed_warmup_is_fenced_before_state_rollback():
    value=torch.tensor([17.],device='cuda')
    graph=PrivateDecodeGraph()
    def fail():
        torch.cuda._sleep(2_000_000)
        value.add_(13)
        raise RuntimeError('simulated model failure')
    with pytest.raises(RuntimeError,match='simulated model failure'):
        graph.capture(fail,{}, {}, {}, {'state':value})
    assert value.item()==17.
    graph.close()
