from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.layerwise_pressure_efficiency import packed_layer_chunk


def test_full_block_addressing_reuses_existing_kernel_and_keeps_skipped_destinations(monkeypatch):
    from vllm_kvmem import split_assembly
    calls=[]
    def launch(raw,ri,values,vi,bases,layer,**kw):
        calls.append(dict(raw=raw.clone(),values=values.clone(),ri=ri.tolist(),vi=vi.tolist(),
                          bases=bases.tolist(),layer=layer,kw=kw))
    monkeypatch.setattr(split_assembly,'assemble_split_layer',launch)
    state=dict(selected_tokens=160)
    records=[NS(count=32),NS(count=32)]
    data=torch.arange(512).to(torch.uint8)
    assert packed_layer_chunk(state,'layer0',data,256,64,records,128,8)
    cached=state['packed_layer_addressing']
    assert packed_layer_chunk(state,'layer1',data,256,0,records,128,8)
    assert state['packed_layer_addressing'] is cached
    assert calls[0]['bases']==[64,96] and calls[1]['bases']==[0,32]
    for call in calls:
        assert call['ri']==call['vi']==[0,1]
        assert torch.equal(call['raw'],data[:256])
        assert torch.equal(call['values'],data[256:])
        assert call['kw']==dict(layer_index=0,layers=1,block_tokens=32,
            raw_record_bytes=128,value_record_bytes=128,raw_dtype=torch.float8_e4m3fn)


@pytest.mark.parametrize('cursor,sizes',[(1,[32]),(0,[17]),(32,[32,13])])
def test_partial_geometry_keeps_original_token_assembly(cursor,sizes):
    state=dict(selected_tokens=100)
    assert not packed_layer_chunk(state,None,None,0,cursor,[NS(count=n) for n in sizes],128,8)
    assert 'packed_layer_addressing' not in state


@pytest.mark.parametrize('count,half,cursor',[(0,0,0),(2,255,0),(2,256,128)])
def test_packed_geometry_fails_before_any_write(count,half,cursor):
    with pytest.raises(ValueError,match='extent'):
        packed_layer_chunk(dict(selected_tokens=160),None,None,half,cursor,
                           [NS(count=32) for _ in range(count)],128,8)


def test_packed_assembly_cannot_change_rope_or_unoptimized_path(monkeypatch):
    from vllm_kvmem.layerwise_query import LayerwiseQuery
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE','sequential')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES','1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PACKED_ASSEMBLY','1')
    monkeypatch.delenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED',raising=False)
    engine=NS(_kvmem_layers=())
    with pytest.raises(ValueError,match='compact-position'):LayerwiseQuery(engine)

