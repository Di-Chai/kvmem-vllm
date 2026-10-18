"""Native qw3 score-domain alignment; actual authority/picker regressions."""
from types import SimpleNamespace as NS

import pytest
import torch

from vllm_kvmem import _native
from vllm_kvmem.authority import RequestAuthority, _Ready, query_score_frame
from vllm_kvmem.block_graph import BlockGraph
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.storage import AsyncArchive


def fixture(tmp_path, *, mode='off', budget=8, sink=0, recent=0, pages=3):
    cfg = KVMemConfig(block_tokens=4, select_budget=budget, gen_budget=16,
        repository_blocks=16, sink_blocks=sink, recent_blocks=recent)
    archive = AsyncArchive(str(tmp_path), 24, 16)
    authority = RequestAuthority('mask', cfg, archive, 1, 1, 2, resident_records=16)
    # Basis queries below yield exactly [[2,0,10],[0,1,0]] at scale=1/sqrt(2).
    vectors = [(2, 0), (0, 1), (10, 0), (1, 1), (3, 2), (0, 0)]
    raw = torch.tensor(vectors[:pages], dtype=torch.bfloat16).repeat_interleave(4, 0)[None, :, None, :]
    assert authority.append(raw, torch.zeros_like(raw, dtype=torch.uint8), history_begin=0, local_rope_begin=0)
    assert authority.flush()
    for job, _ in authority.writes:
        job.result(timeout=5)
    authority.poll()
    if mode != 'off':
        authority.block_graph = BlockGraph('mask', 4)
        authority.block_graph.prepare(0, pages*4).commit()
        authority.graph_options = NS(graph_mode=mode, graph_scoring_version='v2', graph_alpha=.2,
            graph_locality_weight=.1, graph_attention_weight=.2,
            graph_cross_message_weight=.25, graph_trace=False)
    return authority, archive


def finish(authority, archive, plan):
    authority.memory.attach_completion(_Ready())
    assert authority.memory.commit_selection()
    # The plan itself retains leases until released by the caller.
    plan = None


@pytest.mark.parametrize('mode', ['off', 'collect', 'attention'])
def test_mandatory_keeps_softmax_denominator_and_reverses_ranking(tmp_path, mode):
    a, archive = fixture(tmp_path, mode=mode)
    records = sorted(a.records.values(), key=lambda r: r.history_begin)
    q = torch.eye(2).reshape(1, 2, 1, 2) * 2**.5
    plan = a.select_for_queries(q, [records[2].handle])
    assert [row.block_id for row in plan.remaps] == [records[1].handle.slot, records[2].handle.slot]
    assert a.query_score_domain['score_mask'] == (True, True, True)
    assert a.query_score_domain['transport_mask'] == (True, True, False)
    finish(a, archive, plan)
    del plan
    assert a.close() and archive.close()


def test_small_history_scores_keep_bands_then_runtime_budget_masks_them(tmp_path):
    a, archive = fixture(tmp_path, budget=16, sink=1, recent=1)
    with a.index_snapshot() as (records, _):
        with query_score_frame(a.memory, records) as frame:
            assert frame.score_mask == [True]*3
            assert frame.transport_mask == [False]*3
        with query_score_frame(a.memory, records, max_tokens=8) as frame:
            assert frame.score_mask == [False, True, False]
            assert frame.transport_mask == [False, True, False]
        assert a.memory._store.runtime_select_budget == 0
        a.memory._store.set_runtime_select_budget(12)
        with query_score_frame(a.memory, records, max_tokens=8):
            pass
        assert a.memory._store.runtime_select_budget == 12
    assert a.close() and archive.close()


def test_sink_recent_overlap_and_mandatory_consumes_recent_capacity(tmp_path):
    a, archive = fixture(tmp_path, budget=16, sink=2, recent=2)
    with a.index_snapshot() as (records, _):
        with query_score_frame(a.memory, records) as frame:
            assert frame.score_mask == [True]*3
    assert a.close() and archive.close()
    a, archive = fixture(tmp_path/'second', budget=12, sink=1, recent=1, pages=4)
    with a.index_snapshot() as (records, _):
        mandatory = [records[1].handle.slot, records[2].handle.slot]
        with query_score_frame(a.memory, records, mandatory) as frame:
            # Recent is masked by native scoring, but is not actually fixed:
            # sink + mandatory have already exhausted the picker budget.
            assert frame.score_mask == [False, True, True, False]
            assert frame.transport_mask == [False, False, False, True]
    assert a.close() and archive.close()


def test_native_counts_generations_and_budget_exception_leave_scope_unchanged(tmp_path):
    a, archive = fixture(tmp_path, budget=16, sink=1, recent=1)
    with a.index_snapshot() as (records, _):
        handles = [r.handle for r in records]
        counts = [r.count for r in records]
        a.memory._store.set_runtime_select_budget(12)
        with pytest.raises(RuntimeError, match='coverage'):
            a.memory._store.query_score_frame(handles, [3,4,4], 12, [], 8)
        assert a.memory._store.runtime_select_budget == 12
        stale = _native.Handle(handles[0].slot, handles[0].generation+1)
        with pytest.raises(ValueError, match='Stale'):
            a.memory._store.query_score_frame([stale,*handles[1:]], counts, 12, [], 8)
        assert a.memory._store.runtime_select_budget == 12
        with pytest.raises(ValueError):
            a.memory._store.query_score_frame(handles, counts, 13, [], 8)
        with pytest.raises(ValueError):
            a.memory._store.query_score_frame(handles, counts, 12, [], 7)
        assert a.memory._store.runtime_select_budget == 12
    assert a.close() and archive.close()


def test_observer_keeps_query_history_preview_without_graph_transaction(tmp_path):
    a, archive = fixture(tmp_path, mode='collect')
    # Archive can lead graph during an accepted event; observer remains legal.
    a.block_graph.close()
    a.block_graph = BlockGraph('mask', 4)
    seen = []
    a.query_history = NS(policy=NS(open=True), preview=lambda x: seen.append(x) or x)
    a.select_for_queries(torch.eye(2).reshape(1,2,1,2)*2**.5, observe_only=True)
    assert len(seen) == 1 and seen[0][4] > seen[0][0]
    assert a.block_graph.bytes == a.block_graph.live_bytes
    assert a.close() and archive.close()
