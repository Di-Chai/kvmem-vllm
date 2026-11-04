from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.original_rope import record_positions, MTPPositions


def test_sparse_logical_positions_are_independent_of_packed_storage():
    records=[NS(history_begin=0,count=2),NS(history_begin=180000,count=3)]
    assert record_positions(records,'cpu').tolist()==[0,1,180000,180001,180002]


def test_bridge_keeps_original_rope_but_dense_attention_lengths(monkeypatch):
    from vllm_kvmem.window_bridge import install_window_bridge
    from vllm_kvmem.window_batch import _active_lengths
    monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','1')
    seen=[]
    state=NS(prepare_attn=lambda *a:seen.append(_active_lengths.get()),
             prepare_inputs=lambda *a:dict(positions=torch.tensor([100,101])))
    window=NS(positions=torch.tensor([4,5]),groups={7:'dense-lengths'})
    install_window_bridge(state,lambda _:window)
    state.prepare_attn(None)
    assert seen==[{7:'dense-lengths'}]
    assert state.prepare_inputs(None,None)['positions'].tolist()==[100,101]
    monkeypatch.delenv('KVMEM_LAYERWISE_ORIGINAL_ROPE')
    assert state.prepare_inputs(None,None)['positions'].tolist()==[4,5]


def test_mtp_rotates_original_per_owner_without_modifying_dense_slot_positions():
    seen=[]
    module=NS(_kvmem_mtp_prefix_active=True,
              _project_qkv_gate=lambda q,p:seen.append(p.clone()),
              rotary_emb=NS(cos_sin_cache=torch.zeros(4096,2)))
    runner=NS(model_config=NS(enforce_eager=True,max_model_len=4096),device='cpu',max_num_reqs=4,
              _kvmem_window_adapter=NS(offsets={'a':1000,'b':2000}))
    owner=MTPPositions(runner,[module])
    batch=NS(num_tokens_after_padding=8,req_ids=['b','a'],query_start_loc_np=[0,2,5])
    owner.begin(batch)
    dense=torch.tensor([9,10,20,21,22,0,0,0])
    module._project_qkv_gate(None,dense)
    assert dense.tolist()==[9,10,20,21,22,0,0,0]
    assert seen[-1].tolist()==[2009,2010,1020,1021,1022,0,0,0]
    module._kvmem_mtp_prefix_active=False
    module._project_qkv_gate(None,torch.tensor([11,23]))
    assert seen[-1].tolist()==[2011,1023]
    owner.end()
    module._project_qkv_gate(None,torch.tensor([0]))
    assert seen[-1].tolist()==[0]


def test_rope_extension_preserves_prefix_bytes_and_same_frequencies():
    from vllm_kvmem.original_rope import extend_rope_table
    old=torch.randn(8,4,dtype=torch.bfloat16)
    rope=NS(cos_sin_cache=old.clone(),base=10000.,max_position_embeddings=8,
            _compute_inv_freq=lambda base:torch.tensor([1.,.01]))
    extend_rope_table(rope,12)
    assert torch.equal(rope.cos_sin_cache[:8],old)
    phase=torch.arange(8,12)[:,None]*torch.tensor([1.,.01])
    torch.testing.assert_close(rope.cos_sin_cache[8:],torch.cat([phase.cos(),phase.sin()],1).bfloat16(),rtol=0,atol=0)
    assert rope.base==10000.


def test_original_rope_bounds_are_checked_separately_from_physical_slots(monkeypatch):
    from vllm_kvmem.window_assembly import _validate_destinations
    monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','1')
    remap=NS(block_id=1,n_tokens=32,to_base=0)
    layer=NS(kernel_page_table=torch.zeros(2),key=torch.zeros((2,32,1,2)),
             cos_sin=torch.zeros((1000,2)))
    _validate_destinations([remap],{1:NS(count=32,history_begin=900)},[layer])
    with pytest.raises(ValueError,match='RoPE'):
        _validate_destinations([remap],{1:NS(count=32,history_begin=990)},[layer])


@pytest.mark.parametrize('indices',[(0,1),(1,)])
def test_archive_keeps_dense_append_coordinates_after_original_rope(monkeypatch,indices):
    from vllm_kvmem.capture import CapturedLayer
    from vllm_kvmem.capture_runtime import CaptureRuntime
    monkeypatch.setenv('KVMEM_LAYERWISE_ORIGINAL_ROPE','1')
    runtime=CaptureRuntime.__new__(CaptureRuntime)
    runtime.runner=NS(_kvmem_window_batch=NS(positions=torch.tensor([36864,36865,40000,40001])))
    runtime.options=NS(capture_fused_pack=False);runtime.raw_dtype=torch.float8_e4m3fn
    runtime.scales=torch.tensor([1.])
    logical=torch.tensor([53152,53153,150000,150001])
    captured=[CapturedLayer('layer',logical,torch.zeros(4,1,2),torch.zeros(4,1,2),torch.zeros(4,1,2))]
    batch=NS(num_reqs=2,num_tokens=4,positions=logical,
              query_start_loc=torch.tensor([0,2,4]),query_start_loc_np=[0,2,4])
    payload,n=runtime._capture_payload(captured,batch,torch.zeros(2,dtype=torch.int64),indices)
    header=payload[:(4*len(indices)+1)*8].view(torch.int64).tolist()
    expected=[0,2,4,53152,150000,36864,40000,0,0] if len(indices)==2 else [0,2,150000,40000,0]
    assert header==expected
    assert captured[0].positions is logical
