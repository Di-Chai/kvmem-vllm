"""Dense output staging preserves accepted-row arithmetic and rollback."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.action_attention import ActionHistory, ActionRuntime
from vllm_kvmem.block_graph import BlockGraph
from test_graph_capture import options, output_packet


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('strided',[False,True])
def test_dense_legacy_mass_has_identical_double_additions_and_key_order(dtype,strided):
    rng=np.random.default_rng(71)
    values=rng.random((5,173)).astype(dtype)
    values/=values.sum(axis=1,keepdims=True)
    if strided:
        backing=np.zeros((5,346),dtype=dtype);backing[:,::2]=values;values=backing[:,::2]
    targets=[(i%131)*32 for i in range(173)]
    old={32:1e10,96:1e-17,8192:2.}
    expected=old.copy()
    for row in [0,2,3]:
        for key,value in zip(targets,values[row]):
            value=float(value)
            if value:expected[key]=expected.get(key,0.)+value
    actual,samples=_native.prepare_decode_attention(old,4,[0,2,3],targets,values,1<<20)
    assert actual==expected and list(actual)==list(expected) and samples==7
    assert old=={32:1e10,96:1e-17,8192:2.}


def setup(tmp_path,native,owner='s'):
    cfg=options(action_native_decode_commit=native)
    state=ActionHistory(cfg,[10,11,12,13,14]);state.begin('r',[14])
    runtime=ActionRuntime.__new__(ActionRuntime);runtime.runtime=NS(output=tmp_path);runtime.options=cfg
    graph=BlockGraph(owner,32,initial_position=64)
    return state,runtime,graph


def test_partial_mtp_rejected_poison_and_graph_rows_match_reference(tmp_path):
    outcomes=[]
    for native in [False,True]:
        state,runtime,graph=setup(tmp_path,native)
        try:
            state.mass={0:1.25,32:2.}
            packet=output_packet(state,[20]*5,64,[0,0,32,64],
                [[.125,.125,.5,.25]]*3+[[float('nan'),0,0,0]]*2)
            before=state.mass.copy(),state.parser.state,graph.binding
            prepared=runtime.prepare_accept(packet,'r',3,graph=graph,history_begin=64)
            assert (state.mass,state.parser.state,graph.binding)==before
            prepared.commit()
            assert state.samples==3 and graph.partial.samples==3
            with graph.snapshot() as snap:
                graph.restore(snap)
            outcomes.append((state.mass,state.samples,graph.rows,graph.partial))
        finally:graph.close()
        assert graph.budget.used==0
    assert outcomes[0]==outcomes[1]


@pytest.mark.parametrize('failure',['capacity','invalid','abort','commit_capacity'])
def test_native_failure_is_before_parser_legacy_and_graph_publication(tmp_path,failure):
    state,runtime,graph=setup(tmp_path,True)
    try:
        state.mass={0:.125}
        packet=output_packet(state,[20]*5,64,[0,32,64],[[.25,.25,.5]]*5)
        baseline=state.mass.copy(),state.samples,state.parser.state,graph.binding,graph.bytes
        if failure=='capacity':graph.max_bytes_per_session=graph.bytes
        if failure=='invalid':packet.action_inputs['r'].masses[1,0]=float('inf')
        if failure=='commit_capacity':state.options.action_commit_max_bytes=4096
        if failure=='abort':
            prepared=runtime.prepare_accept(packet,'r',3,graph=graph,history_begin=64);prepared.abort()
        else:
            with pytest.raises((BufferError,ValueError,RuntimeError)):
                runtime.prepare_accept(packet,'r',3,graph=graph,history_begin=64)
        assert (state.mass,state.samples,state.parser.state,graph.binding,graph.bytes)==baseline
        assert 'r' in packet.action_inputs and graph._pending_update is None
    finally:graph.close()


def test_graph_off_stages_before_parser_and_owners_are_isolated(tmp_path):
    owners=[]
    for i in range(2):
        state,_,graph=setup(tmp_path,True,str(i));graph.close();owners.append(state)
    for state in owners:
        state.options.graph_mode='off';state.parser.prepare([20]*5)
    owners[0].accept(2,[0,1,2,3,4],[0,32],torch.tensor([[.25,.75]]*5))
    assert owners[0].mass=={0:.5,32:1.5} and owners[0].samples==2
    assert owners[1].mass=={} and owners[1].samples==0
    before=owners[1].parser.state
    with pytest.raises(ValueError):
        owners[1].accept(2,[0,1,2,3,4],[0,32],torch.tensor([[float('nan'),1.]]*5))
    assert owners[1].parser.state==before and owners[1].mass=={}


def test_dense_index_bounds_rejected_without_changing_graph():
    graph=BlockGraph('s',32,initial_position=64)
    try:
        before=graph.binding,graph.bytes
        with pytest.raises(ValueError):
            graph.prepare_accept(64,69,[65],[0,32],np.ones((1,2),dtype='float32')/2,
                row_indices=[2])
        assert (graph.binding,graph.bytes)==before and graph._pending_update is None
    finally:graph.close()
