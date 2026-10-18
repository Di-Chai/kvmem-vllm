from types import SimpleNamespace

import pytest
import torch

from vllm_kvmem.authority import RequestAuthority
from vllm_kvmem.block_graph import BlockGraph, GraphBudget, GraphSample
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.prompt_checkpoints import PromptCheckpointPool
from vllm_kvmem.storage import AsyncArchive
from test_authority import payload, drain
from test_core import Event
from test_prompt_checkpoints import saved, ready


def attach(authority, mode='attention', budget=None):
    graph = authority.block_graph = BlockGraph(('tenant', authority.memory.request_id),
        authority.memory.config.block_tokens, budget=budget)
    authority.graph_options = SimpleNamespace(graph_mode=mode, graph_locality_weight=.1,
        graph_attention_weight=.2, graph_cross_message_weight=.25, graph_trace=True)
    return graph


def fixture(tmp_path, mode='attention', *, sink=0, recent=0, budget=12):
    cfg = KVMemConfig(block_tokens=4, select_budget=budget, gen_budget=16,
        repository_blocks=16, sink_blocks=sink, recent_blocks=recent,
        action_attention=True, action_fraction=0.)
    archive = AsyncArchive(str(tmp_path), 48, 16)
    authority = RequestAuthority('selection', cfg, archive, 2, 1, 2, resident_records=16)
    graph = attach(authority, mode)
    assert authority.append(*payload(0, 24), history_begin=0, local_rope_begin=0)
    drain(authority)
    graph.prepare(0, 24, [GraphSample(20, (4, 20), (.9, .1), 'prefill')]).commit()
    return authority, archive, graph


def selected_starts(authority, plan):
    slots = {row.block_id for row in plan.remaps}
    return sorted(r.history_begin for r in authority.records.values() if r.handle.slot in slots)


def complete(authority):
    authority.memory.attach_completion(Event(True))
    assert authority.memory.commit_selection()


def test_query_graph_changes_native_selection_same_budget_and_skips_legacy(tmp_path, monkeypatch):
    authority, archive, graph = fixture(tmp_path)
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *a: torch.tensor([.01, .10, .15, .12, .02, .60]))
    class Forbidden:
        def scores(self):
            pytest.fail('Graph scoring must not apply legacy prefill fusion')
    authority.action_history = SimpleNamespace(scores={0: 999}, samples=0, prefill=Forbidden())
    plan = authority.select_for_queries(torch.zeros(2, 1, 1, 2), max_tokens=12)
    assert selected_starts(authority, plan) == [4, 8, 20]
    assert plan.total_window_tokens == 12
    assert authority.graph_selection_report['added_blocks'] == 1
    assert authority.graph_selection_report['displaced_blocks'] == 1
    complete(authority)
    del plan
    assert authority.close() and archive.close()
    assert graph.budget.used == 0


def test_collect_keeps_native_selection_and_observe_only_never_selects(tmp_path, monkeypatch):
    authority, archive, graph = fixture(tmp_path, 'collect')
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *a: torch.tensor([.01, .10, .15, .12, .02, .60]))
    before = authority.memory.phase
    assert authority.select_for_queries(torch.zeros(2, 1, 1, 2), observe_only=True) is None
    assert authority.memory.phase is before
    plan = authority.select_for_queries(torch.zeros(2, 1, 1, 2))
    assert selected_starts(authority, plan) == [8, 12, 20]
    assert authority.graph_selection_report['selection_unchanged']
    complete(authority)
    del plan
    assert authority.close() and archive.close()


@pytest.mark.parametrize('trace,extra_capacity', [(False, 0), (True, 65536)])
def test_scoring_scratch_exhaustion_precedes_native_assembly(tmp_path, monkeypatch, trace, extra_capacity):
    authority, archive, graph = fixture(tmp_path)
    authority.graph_options.graph_trace = trace
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *a: torch.tensor([.01, .10, .15, .12, .02, .60]))
    # Permit the immutable scoring snapshot, then reject either the first
    # propagation scratch or the additional bounded explanation pass.
    before = graph.bytes
    graph.max_bytes_per_session = 2 * graph.live_bytes + extra_capacity
    phase = authority.memory.phase
    with pytest.raises(BufferError):
        authority.select_for_queries(torch.zeros(2, 1, 1, 2))
    assert authority.memory.phase is phase
    assert graph.bytes == before
    assert graph.accepted_through == 24
    # A rejected scoring attempt remains retryable on the same owner.
    graph.max_bytes_per_session = 33554432
    plan = authority.select_for_queries(torch.zeros(2, 1, 1, 2))
    assert selected_starts(authority, plan) == [4, 8, 20]
    complete(authority)
    del plan
    assert authority.close() and archive.close()
    assert graph.budget.used == 0


def test_collect_observer_can_run_inside_uncommitted_graph_acceptance(tmp_path, monkeypatch):
    authority, archive, graph = fixture(tmp_path, 'collect')
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *a: torch.tensor([.01, .10, .15, .12, .02, .60, .01]))
    assert authority.append(*payload(24, 1), history_begin=24, local_rope_begin=24)
    # Archive publication precedes query-history observation and graph commit.
    # An observe-only call must not require the pending graph water mark.
    with graph.prepare(24, 25) as update:
        assert authority.select_for_queries(torch.zeros(2, 1, 1, 2), observe_only=True) is None
        update.commit()
    assert graph.accepted_through == 25
    assert authority.close() and archive.close()


def test_native_keep_bands_mandatory_and_runtime_budget_survive_graph(tmp_path, monkeypatch):
    authority, archive, graph = fixture(tmp_path, sink=1, recent=1, budget=16)
    def scores(q, means, eligible):
        assert eligible.tolist() == [False, True, False, True, True, False]
        return torch.tensor([0., .1, 0., .5, .4, 0.])
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores', scores)
    mandatory = [r.handle for r in authority.records.values() if r.history_begin == 8]
    plan = authority.select_for_queries(torch.zeros(2, 1, 1, 2), mandatory, max_tokens=16)
    assert selected_starts(authority, plan) == [0, 8, 12, 20]
    assert plan.total_window_tokens == 16
    complete(authority)
    del plan
    assert authority.close() and archive.close()


def test_old_async_plan_cannot_commit_after_epoch_invalidation(tmp_path, monkeypatch):
    authority, archive, graph = fixture(tmp_path)
    monkeypatch.setattr('vllm_kvmem.retrieval.mean_key_scores',
        lambda *a: torch.tensor([.01, .10, .15, .12, .02, .60]))
    plan = authority.select_for_queries(torch.zeros(2, 1, 1, 2))
    graph.invalidate()
    authority.memory.attach_completion(Event(True))
    with pytest.raises(RuntimeError, match='Stale'):
        authority.memory.commit_selection()
    # Failed assembly closes with its fence and never publishes a selected view.
    del plan
    assert authority.close() and archive.close()


def graph_checkpoints(tmp_path, *, partial=False):
    pool = PromptCheckpointPool(2, 262144, retain_prompt=True)
    pool, old, authority, archive, adapter, _ = saved(tmp_path, pool=pool)
    pool.drop_live(old)
    graph = attach(authority)
    graph.prepare(0, 8, [GraphSample(4, (0, 4), (.7, .3), 'prefill')]).commit()
    graph.update_message_spans(((0, 8, 'user'),), validated_prefix=8)
    prompt = pool.capture(('tenant', 'session'), 'a', authority, list(range(24)), adapter)
    end = 10 if partial else 12
    assert authority.append(*payload(8, end-8), history_begin=8, local_rope_begin=8)
    drain(authority)
    graph.prepare(8, end, [GraphSample(8, (0, 8), (.8, .2), 'output')]).commit()
    adapter.runner.req_states.num_computed_tokens_np[0] = end
    adapter.runner.req_states.prefill_len.np[0] = 24
    adapter.runner._kvmem_capture_runtime = SimpleNamespace(
        last_steps={'a': [dict(history_begin=8, accepted=end-8, scheduled=end-8)]},
        authorities={'a': authority})
    adapter.recurrent.conv_dim_first = False
    adapter.recurrent.conv_widths = {'gdn': 4}
    terminal = pool.capture(prompt.namespace, 'a', authority, list(range(24)),
        adapter, replace=True, terminal=True)
    pool.retire(terminal, Event(True))
    ready(pool, terminal)
    return pool, prompt, terminal, authority, archive, adapter, graph


@pytest.mark.parametrize('edit', [False, True])
def test_full_m_preservation_vs_true_branch_and_intermediate_p_snapshot(tmp_path, edit):
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path)
    old_binding = graph.binding
    tokens = list(range(24))
    if edit:
        tokens[9] = 999
    checkpoint = pool.claim(prompt.namespace, 'new', tokens, position=8)
    assert checkpoint is prompt
    assert (checkpoint.claim_graph_history is None) is edit
    plan = pool.prepare_claim_window(checkpoint)
    assert graph.accepted_through == (8 if edit else 12)
    with pytest.raises(RuntimeError, match='Stale'):
        graph.validate_binding(old_binding)
    with graph.snapshot(8) as earlier:
        assert earlier.accepted_through == 8
        assert [row.source for row in earlier.rows] == [4]
    if not edit:
        binding = graph.binding
        graph.prepare(8, 12, [GraphSample(8, (0, 8), (.1, .9), 'prefill')]).commit()
        assert graph.binding == binding
        assert graph.rows[-1].output_samples == 1
    complete(authority)
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    del plan
    assert pool.payload_bytes == 0
    assert authority.close() and archive.close()
    assert graph.budget.used == 0


def test_partial_m_restore_retains_accumulator_without_future_leak(tmp_path):
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path, partial=True)
    checkpoint = pool.claim(terminal.namespace, 'new', list(range(24)), position=10)
    plan = pool.prepare_claim_window(checkpoint)
    assert graph.partial.source == 8 and graph.partial.output_samples == 1
    assert graph.partial.sums == (.8, .2)
    complete(authority)
    assert pool.finish_claim(checkpoint, Event(True)) is authority
    del plan
    # A retained P still has its own exact graph; close it before retiring live authority.
    if pool.entries:
        pool.drop_live(next(iter(pool.entries.values())))
    assert authority.close() and archive.close()
    assert graph.budget.used == 0


def test_checkpoint_graph_capacity_failure_keeps_previous_snapshot_and_credits(tmp_path):
    pool, prompt, terminal, authority, archive, adapter, graph = graph_checkpoints(tmp_path)
    before, graph_before = pool.payload_bytes, graph.bytes
    pool.max_payload_bytes = before
    with pytest.raises(BufferError, match='M graph preservation'):
        pool.claim(prompt.namespace, 'new', list(range(24)), position=8)
    assert pool.entries[prompt.namespace] is terminal
    assert pool.payload_bytes == before and graph.bytes == graph_before
    assert pool.evict(terminal)
    assert pool.payload_bytes == 0 and graph.budget.used == 0 and archive.close()


def test_shared_budget_session_isolation_and_released_snapshot_accounting(tmp_path):
    budget = GraphBudget(65536)
    one = BlockGraph(('tenant', 'one'), 4, budget=budget)
    two = BlockGraph(('tenant', 'two'), 4, budget=budget)
    one.prepare(0, 8, [GraphSample(4, (0, 4), (.7, .3))]).commit()
    snap = one.snapshot()
    with pytest.raises(ValueError, match='foreign'):
        two.restore(snap)
    assert not two.rows and two.accepted_through == 0
    one.close()
    assert budget.used == snap.bytes + two.live_bytes
    snap.close()
    two.close()
    assert budget.used == 0
