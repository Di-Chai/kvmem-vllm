"""Query gather candidate cannot silently alter pressure or P restoration."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.layerwise_query import LayerwiseQuery
from test_layerwise_query import runner


def test_default_off_and_rejects_missing_execution_prerequisites(monkeypatch):
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE', 'sequential')
    for key in ('KVMEM_LAYERWISE_QUERY_OPTIMIZED', 'KVMEM_LAYERWISE_PAIRED_READS',
                'KVMEM_LAYERWISE_QUERY_PAIRED_GATHER'):
        monkeypatch.delenv(key, raising=False)
    assert not LayerwiseQuery(runner()).paired_gather
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_PAIRED_GATHER', '1')
    with pytest.raises(ValueError, match='optimized paired'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED', '1')
    with pytest.raises(ValueError, match='optimized paired'):
        LayerwiseQuery(runner())
    monkeypatch.setenv('KVMEM_LAYERWISE_PAIRED_READS', '1')
    assert LayerwiseQuery(runner()).paired_gather
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_PAIRED_GATHER', 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        LayerwiseQuery(runner())


@pytest.mark.parametrize('kind', ['query', 'prefill_pressure', 'decode_pressure', 'restore'])
@pytest.mark.parametrize('enabled', [False, True])
def test_dispatch_and_counter_are_scoped_to_public_query(monkeypatch, kind, enabled):
    from vllm_kvmem import packed_assembly, window_assembly, layerwise_pressure_efficiency
    no_op = lambda *args: None
    stream = NS(wait_event=no_op, synchronize=no_op)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: stream)
    layer, calls = object(), []
    staging = NS(slot_bytes=128, claim=no_op, release=no_op, consumed=no_op,
        destination=lambda n: (torch.empty(n, dtype=torch.uint8), None, 0))
    def upload(size, destination, fill, **kwargs):
        fill(torch.empty(size, dtype=torch.uint8))
        return NS(event=object())
    assembler = NS(layers=[layer], staging=staging, close=no_op,
                   transfers=NS(upload_from_reader=upload))
    monkeypatch.setattr(window_assembly.NativeWindowAssembler, 'from_runner',
                        lambda *args, **kwargs: assembler)
    monkeypatch.setattr(layerwise_pressure_efficiency, 'packed_layer_chunk',
                        lambda *args: True)
    def read(authority, handles, *, destination, **kwargs):
        calls.append(kwargs)
        destination.fill_(53)
        return None, {'paired_native_gather': kwargs.get('paired_native', False)}
    monkeypatch.setattr(packed_assembly, 'read_packed_records', read)
    query = LayerwiseQuery.__new__(LayerwiseQuery)
    query.names = ('layer0',)
    query.runner = NS(device='cpu')
    query.batched_reads = False
    query.paired_reads = query.packed_assembly = True
    query.paired_gather = enabled
    state = dict(authority=NS(raw_element_bytes=1, shape=(1,32,1,2),
                             archive=NS(memory_only=True)),
                 request_id='q', selected_tokens=32)
    if kind == 'query':
        state.update(next_layer=0, span=(128,129))
    elif kind == 'restore':
        state['assembly_arena'] = 'layerwise'
    else:
        state.update(assembly_arena='pressure',
                     pressure_stage='prefill' if kind == 'prefill_pressure' else 'decode')
    stats = query.assemble_layer_pinned(state, 0,
        [NS(handle=object(), history_begin=32, count=32, archived=True)])
    assert len(calls) == 1
    expected = enabled and kind == 'query'
    assert ('paired_native' in calls[0]) is expected
    assert stats['native_paired_gather_chunks'] == int(expected)
    assert stats['paired_read_chunks'] == 1 and stats['uploaded_bytes'] == 128
    assert calls[0]['inline_cpu'] is (kind == 'prefill_pressure')
