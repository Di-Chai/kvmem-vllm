"""Query-local metadata preserves native ranking and original record order."""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem.core.config import KVMemConfig
from vllm_kvmem.serving.layerwise_query import LayerwiseQuery, top_set
from vllm_kvmem.core.memory import RequestMemory
from vllm_kvmem.serving.query_record_index import QueryRecordIndex, query_record_index
from test_core import append


class CountedHandle:
    def __init__(self, slot):
        self._slot, self.reads = slot, 0

    @property
    def slot(self):
        self.reads += 1
        return self._slot


def records_for_slots(slots):
    return tuple(NS(handle=CountedHandle(slot), history_begin=i*32, count=32)
                 for i, slot in enumerate(slots))


def test_score_items_and_record_order_match_legacy_without_repeated_handle_access():
    records = records_for_slots((9, 3, 12, 5))
    state = dict(records=records, partial_excluded={12})
    index = query_record_index(state)
    assert index.excluded_indices == (2,)
    assert [record.handle.reads for record in records] == [1]*4
    calls = []
    def preview(scores, fixed, **kwargs):
        calls.append((tuple(scores.items()), fixed, kwargs))
        return (5, 9)
    memory = NS(preview_selection=preview)
    for _ in range(16):
        assert query_record_index(state) is index
        selected = top_set(memory, records, [2, 2, -1, 3], (9,), 64, record_index=index)
        assert index.chosen(selected) == [records[0], records[3]]
        assert index.starts(selected) == [0, 96]
    assert all(c == calls[0] for c in calls)
    assert calls[0] == (((9, 2.), (3, 2.), (12, -1.), (5, 3.)), (9,),
                        dict(action=False, prefill_scores=None, max_tokens=64))
    assert [record.handle.reads for record in records] == [1]*4
    top_set(memory, records, [2, 2, -1, 3], (9,), 64)
    assert calls[-1] == calls[0]


@pytest.mark.parametrize('partial_fixed', [False, True])
@pytest.mark.parametrize('scores', [[1.]*8, [8., 3., 3., 9., 0., 3., 9., 100.]])
def test_native_picker_ties_and_partial_inclusion_are_identical(partial_fixed, scores):
    memory = RequestMemory('index', KVMemConfig(block_tokens=32, select_budget=96,
        gen_budget=256, repository_blocks=16, sink_blocks=1, recent_blocks=0))
    spans = append(memory, 233)
    records = tuple(NS(handle=span.handle, history_begin=i*32,
                       count=9 if i==7 else 32) for i, span in enumerate(spans))
    slots = [record.handle.slot for record in records]
    fixed = (slots[0], slots[7]) if partial_fixed else (slots[0],)
    excluded = set() if partial_fixed else {slots[7]}
    index = QueryRecordIndex(records, excluded)
    before, after = list(scores), list(scores)
    for i, record in enumerate(records):
        if record.handle.slot in excluded:before[i] = -1.
    for i in index.excluded_indices:after[i] = -1.
    try:
        expected = top_set(memory, records, before, fixed, 96)
        actual = top_set(memory, records, after, fixed, 96, record_index=index)
        assert actual == expected and set(fixed) <= actual
        assert index.chosen(actual) == [r for r in records if r.handle.slot in expected]
        assert index.starts(actual) == [r.history_begin for r in records if r.handle.slot in expected]
        if not partial_fixed:assert slots[7] not in actual
    finally:
        assert memory.close()


@pytest.mark.parametrize('fault', ['duplicate', 'excluded', 'scores', 'selected', 'rebound'])
def test_invalid_or_rebound_record_geometry_is_rejected(fault):
    records = records_for_slots((3, 7, 9))
    if fault == 'duplicate':
        with pytest.raises(ValueError, match='duplicate'):
            QueryRecordIndex(records_for_slots((3, 3)), set())
        return
    if fault == 'excluded':
        with pytest.raises(ValueError, match='absent'):
            QueryRecordIndex(records, {99})
        return
    state = dict(records=records, partial_excluded=set())
    index = query_record_index(state)
    if fault == 'scores':
        with pytest.raises(ValueError, match='extent'):index.score_dict([1.])
    elif fault == 'selected':
        with pytest.raises(ValueError, match='absent'):index.chosen({99})
    else:
        state['records'] = tuple(list(records))
        with pytest.raises(RuntimeError, match='changed'):query_record_index(state)


@pytest.mark.parametrize('mode', ['global', 'sequential'])
@pytest.mark.parametrize('diagnostics', [False, True])
def test_complete_query_hooks_preserve_selection_and_evidence(tmp_path, monkeypatch, mode, diagnostics):
    from vllm_kvmem.core import cuda_scoring
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_MODE', mode)
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_PREPARED_INDEX', '1')
    monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_DIAGNOSTICS', str(int(diagnostics)))
    for key in ('KVMEM_QUERY_STAGE_PROFILE', 'KVMEM_LAYERWISE_PREFILL_PRESSURE',
                'KVMEM_LAYERWISE_QUERY_NO_PROBE', 'KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES'):
        monkeypatch.delenv(key, raising=False)
    memory = RequestMemory('q', KVMemConfig(block_tokens=32, select_budget=96,
        gen_budget=256, repository_blocks=16, sink_blocks=1, recent_blocks=0))
    spans = append(memory, 256)
    slots = [span.handle.slot for span in spans]
    records = records_for_slots(slots)
    queries = torch.ones(16, 1, 1, 2)
    monkeypatch.setattr(cuda_scoring, 'prepared_layer_scores',
        lambda q, index, layer:torch.tensor([1., 3., 3., 2., 4.+(layer%2), 4., 3., -1.]))
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args:NS(synchronize=lambda:None))
    observed = []
    try:
        for enabled in (False, True):
            monkeypatch.setenv('KVMEM_LAYERWISE_QUERY_RECORD_INDEX', str(int(enabled)))
            engine = NS(model_config=NS(enforce_eager=True), _kvmem_graph=False,
                _kvmem_layers=tuple(f'layer{i}' for i in range(16)), device='cpu',
                _kvmem_probe_output=tmp_path/str(enabled),
                vllm_config=NS(scheduler_config=NS(max_num_seqs=1)),
                _kvmem_last_batch=NS(req_ids=['q'],num_computed_tokens_np=[256],num_tokens=1))
            controller = LayerwiseQuery(engine)
            assembled = []
            controller.assemble_layer_pinned = lambda state, layer, chosen: (
                assembled.append((layer, [r.history_begin for r in chosen])) or {})
            state = dict(request_id='q', next_layer=0, history=256, span=(256,257),
                old_q=queries, records=records, authority=NS(memory=memory), fixed=(slots[0],),
                partial_excluded={slots[-1]}, budget=96, selected_tokens=96,
                global_slots={slots[0],slots[1],slots[2]}, mean_index=object())
            controller.active = state
            for record in records:record.handle.reads=0
            for layer in range(16):
                controller.projection(layer, NS(query=torch.ones(1,1,2)))
            if enabled:assert [r.handle.reads for r in records]==[1]*len(records)
            events=[json.loads(line) for line in controller.path.read_text().splitlines()]
            rows=[{k:v for k,v in event.items() if k not in
                    ('at','monotonic_at','scoring_seconds','assembly_host_seconds')}
                  for event in events]
            assert len([e for e in rows if e['event']=='layer_selected'])==16
            assert rows[-1]['event']=='completed' and controller.active is None
            observed.append((assembled,rows))
        assert observed[0]==observed[1]
    finally:
        assert memory.close()
