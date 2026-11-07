"""A Query-owned CPU reader must drain on success, failure and retirement."""
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.layerwise_query import LayerwiseQuery


@pytest.mark.parametrize('flag,attribute', [
    ('KVMEM_LAYERWISE_QUERY_RESIDENT_V', 'resident_values'),
    ('KVMEM_LAYERWISE_QUERY_LAZY_READER', 'lazy_reader')])
def test_query_transport_candidates_require_explicit_supported_scope(monkeypatch, flag, attribute):
    from test_layerwise_query import runner
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE', 'sequential')
    assert getattr(LayerwiseQuery(runner()), attribute) is False
    monkeypatch.setenv(flag, '1')
    with pytest.raises(ValueError, match='optimized public sequential'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PAIRED_READS', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PACKED_ASSEMBLY', '1')
    assert getattr(LayerwiseQuery(runner()), attribute) is True
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_PAIRED_GATHER', '1')
    with pytest.raises(ValueError, match='separate qualification'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_PAIRED_GATHER', '0')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MTP_SELECTION', 'layer0')
    with pytest.raises(ValueError, match='global MTP'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MTP_SELECTION', 'global')
    monkeypatch.setenv(flag, 'invalid')
    with pytest.raises(ValueError, match='must be 0 or 1'):
        LayerwiseQuery(runner())


def controller(state):
    query=LayerwiseQuery.__new__(LayerwiseQuery)
    query.pending={};query.active_owners={'r':state};query.active=None;query.finished=set()
    query.runner=NS(device='cpu',_kvmem_request_generations={'r':7},
                    _kvmem_capture_runtime=NS(authorities={'r':state.get('authority')}))
    return query


def test_retirement_keeps_owned_reader_until_producer_has_drained():
    closed=[];ready=[False]
    def close():
        closed.append(ready[0])
        return ready[0]
    reader=NS(close=close)
    state={'query_region_reader':reader}
    query=controller(state)
    with pytest.raises(RuntimeError,match='active CPU producer'):query.retire('r')
    assert query.active_owners['r'] is state and state['query_region_reader'] is reader
    ready[0]=True;query.retire('r')
    assert query.active_owners=={} and 'query_region_reader' not in state
    assert closed==[False,True]


def test_projection_failure_closes_only_the_failed_query_reader(monkeypatch):
    closed=[]
    state={'query_region_reader':NS(close=lambda:closed.append('r') or True)}
    query=controller(state)
    other={'query_region_reader':NS(close=lambda:closed.append('other') or True)}
    query.active_owners['other']=other
    def fail(*args):raise ValueError('projection failed')
    monkeypatch.setattr(query,'_projection_inner',fail)
    with pytest.raises(ValueError,match='projection failed'):query._projection(state,0,None)
    assert closed==['r'] and 'query_region_reader' not in state
    assert 'query_region_reader' in other


def test_resident_metadata_is_retained_until_existing_or_exception_fence(monkeypatch):
    sequence=[]
    import torch
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device:NS(synchronize=lambda:sequence.append('fence')))
    class Owners(list):
        def clear(self):sequence.append('clear');super().clear()
    state={'query_resident_metadata_owners':Owners([object()]),
           'query_region_reader':NS(close=lambda:sequence.append('close') or True)}
    query=controller(state)
    query.retire('r')
    assert sequence==['fence','close','clear']


def test_query_reader_is_lazy_and_bound_to_current_generation(monkeypatch):
    from vllm_kvmem import query_region_reader
    created=[]
    def factory(authority,**kwargs):
        reader=NS(authority=authority,**kwargs);created.append(reader);return reader
    monkeypatch.setattr(query_region_reader,'QueryRegionReader',factory)
    state={'request_id':'r','authority':object()};query=controller(state)
    assert not created
    reader=query._query_reader(state)
    assert query._query_reader(state) is reader and len(created)==1
    assert reader.request_generation==7 and reader.generation_getter()==7
    query.runner._kvmem_request_generations['r']=8
    assert reader.generation_getter()==8
    query.runner._kvmem_capture_runtime.authorities['r']=object()
    with pytest.raises(ValueError,match='authority binding'):
        reader.generation_getter()
