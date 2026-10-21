from types import SimpleNamespace as NS

import pytest

from vllm_kvmem.action_attention import ActionHistory, ActionRuntime, window_segments
from vllm_kvmem.config import KVMemConfig
from vllm_kvmem.memory import RequestMemory
from vllm_kvmem.prefill_planning import freeze_prefill_frame
from test_core import Event, append


def options(**updates):
    values = dict(prefill_attention=True, prefill_attention_interval=7,
        prefill_attention_stride=2, prefill_attention_half_life=32,
        native_prefill_commit=False, prefill_publication_pipeline=True,
        memory=KVMemConfig(block_tokens=4, select_budget=12, gen_budget=12,
            repository_blocks=64, sink_blocks=1, recent_blocks=0,
            action_attention=True), action_scope='non-thinking', action_interval=1,
        action_decay=.5)
    values.update(updates)
    return NS(**values)


def history(opts):
    return ActionHistory(opts, [10, 11, 12, 13, 14])


@pytest.mark.parametrize('reordered', [False, True])
def test_frames_match_native_appends_and_serial_sampling_with_delayed_commits(reordered):
    """Compare the actual native append ledger and unchanged sampling policy."""
    opts = options()
    memory = RequestMemory('a', opts.memory)
    reference, delayed = history(opts), history(opts)
    local = begin = 0
    if reordered:
        spans = append(memory, 24)
        selected = {spans[i].handle.slot: 1. for i in (0, 2, 5)}
        plan = memory.select(selected)
        memory.attach_completion(Event(True))
        assert memory.commit_selection()
        del plan
        begin, local = 24, 12
        chunks = (3, 2, 1, 5)
    else:
        chunks = (3, 2, 5, 1, 4, 6)
    authority = NS(memory=memory)
    frame = freeze_prefill_frame(authority, delayed, ('generation',), begin,
        local, memory._store.working_records())
    pending = []
    for count in chunks:
        expected = window_segments(memory._store.working_records(), begin, local, count, 4)
        expected_rows = reference.prefill.rows(begin, count, local)
        rows, starts, counts, keys, frame = frame.extend(begin, local, count, 4)
        assert (starts, counts, keys) == expected
        assert rows == expected_rows
        # Use distinct normalized row masses and commit the pipeline one chunk
        # behind; the publication order and floating-point results stay exact.
        masses = []
        for row in rows:
            weights = [float(i + row + 1) if start <= local + row else 0.
                       for i, start in enumerate(starts)]
            masses.append([v / sum(weights) for v in weights])
        args = (begin + count, count, count, rows, starts, keys, local, masses)
        reference.prefill.accept(*args)
        pending.append(args)
        if len(pending) > 1:
            delayed.prefill.accept(*pending.pop(0))
        append(memory, count, local=local)
        begin += count
        local += count
        assert [(r[2], r[3], r[4]) for r in frame.records] == [
            (r[2], r[3], r[4]) for r in memory._store.working_records()]
    for args in pending:
        delayed.prefill.accept(*args)
    assert reference.prefill.snapshot() == delayed.prefill.snapshot()
    assert memory.close()


def runtime(owners=('a',), *, interval=7):
    opts = options(prefill_attention_interval=interval)
    reads, drains = [], []
    records = {rid: [] for rid in owners}
    authorities = {}
    for rid in owners:
        def working_records(rid=rid):
            reads.append(rid)
            return records[rid]
        authorities[rid] = NS(memory=NS(_store=NS(working_records=working_records)),
                             action_history=history(opts), block_graph=object())
    action = ActionRuntime.__new__(ActionRuntime)
    action.options, action.ids = opts, [10, 11, 12, 13, 14]
    action.runner = NS(req_states=NS(req_id_to_index={rid: i for i, rid in enumerate(owners)},
        prompt_len=NS(np=[100] * len(owners))),
        _kvmem_request_generations={rid: 1 for rid in owners},
        _kvmem_window_adapter=NS(generations={}, offsets={}))
    action.runtime = NS(options=opts, config=opts.memory, authorities=authorities,
        drain_requests=lambda ids, wait: drains.append(tuple(ids)) or True,
        ensure_authority=authorities.__getitem__, prefill_pipeline_can_advance=lambda rid: True)
    return action, records, reads, drains


def prepare(action, begin, count, rid='a', *, prefill=True):
    batch = NS(req_ids=[rid], query_start_loc_np=[0, count],
        num_computed_tokens_np=[begin], is_prefilling_np=[prefill])
    action.pipeline_requests = set()
    action.prefill_plans = None
    action._prepare_prefill_pipeline(batch)
    return action.prefill_plans[1].get(rid)


def test_second_plan_uses_frozen_frame_while_scores_and_native_store_remain_unpublished():
    action, records, reads, drains = runtime()
    first = prepare(action, 0, 3)
    assert first['rows'] == [0, 2]
    assert action.runtime.authorities['a'].action_history.prefill.end == 0
    assert records['a'] == []  # Simulate a blocked CPU publisher.
    second = prepare(action, 3, 2)
    assert second['rows'] == []  # Planned sample end 3, not mutable committed end 0.
    assert second['starts'] == [0, 3, 4]
    assert second['counts'] == [3, 1, 1]
    assert second['history'] == [0, 0, 4]
    assert reads == ['a'] and drains == [('a',)]
    assert action.pipeline_requests == {'a'}
    assert action.prefill_pipeline_ready('a', 5, 5)
    assert not action.prefill_pipeline_ready('a', 4, 4)


@pytest.mark.parametrize('invalidate', ['window_generation', 'request_generation',
                                      'restore_epoch', 'state', 'offset', 'runtime_fence'])
def test_changed_owner_identity_or_runtime_fence_drains_and_refreshes_snapshot(invalidate):
    action, records, reads, drains = runtime()
    prepare(action, 0, 4)
    records['a'] = [(7, 3, 0, 4, 0)]
    authority = action.runtime.authorities['a']
    authority.action_history.prefill.end = 4
    if invalidate == 'window_generation':
        action.runner._kvmem_window_adapter.generations['a'] = 2
    elif invalidate == 'request_generation':
        action.runner._kvmem_request_generations['a'] += 1
    elif invalidate == 'restore_epoch':
        authority.action_history.prefill_planning_epoch += 1
    elif invalidate == 'state':
        authority.action_history = history(action.options)
    elif invalidate == 'offset':
        action.runner._kvmem_window_adapter.offsets['a'] = 2
        records['a'] = [(7, 3, 2, 2, 0)]
    elif invalidate == 'runtime_fence':
        action.runtime.prefill_pipeline_can_advance = lambda _: False
    plan = prepare(action, 4, 3)
    assert plan is not None and action.pipeline_requests == {'a'}
    assert reads == ['a', 'a'] and drains == [('a',), ('a',)]
    assert plan['local_begin'] == (2 if invalidate == 'offset' else 4)


def test_replay_decode_and_terminal_prompt_chunks_never_pipeline():
    for kind in ('replay', 'decode', 'terminal'):
        action, records, _, _ = runtime()
        prepare(action, 0, 4)
        records['a'] = [(7, 1, 0, 4, 0)]
        if kind == 'replay':
            action.runtime.authorities['a'].action_history.prefill_replay = (4, 8)
        if kind == 'terminal':
            action.runner.req_states.prompt_len.np[0] = 8
        assert prepare(action, 4, 4, prefill=kind != 'decode') is None
        assert action.pipeline_requests == set()
        assert 'a' not in action.prefill_plan_frames


def test_cached_frame_cannot_enter_a_preexisting_future_query_replay_interval():
    action, _, _, _ = runtime()
    action.runtime.authorities['a'].action_history.prefill_replay = (4, 8)
    assert prepare(action, 0, 4) is not None
    assert prepare(action, 4, 3) is None
    assert action.pipeline_requests == set() and not action.prefill_plan_frames


def test_validated_future_sampling_watermark_survives_replayed_prefix():
    action, records, _, _ = runtime()
    state = action.runtime.authorities['a'].action_history
    state.prefill.end = 8  # Validated M evidence survives physical restore to P.
    reference = history(action.options)
    reference.prefill.end = 8
    for begin, count in ((0, 3), (3, 3), (6, 3), (9, 6)):
        plan = prepare(action, begin, count)
        expected = reference.prefill.rows(begin, count, begin)
        assert plan['rows'] == expected
        if expected:
            reference.prefill.end = begin + count
        assert action.prefill_plan_frames['a'].sampled_end == reference.prefill.end
    assert records['a'] == [] and state.prefill.end == 8


def test_owner_frames_are_isolated_and_invalidation_does_not_discard_other_owner():
    action, _, reads, drains = runtime(('a', 'b'))
    prepare(action, 0, 3, 'a')
    prepare(action, 0, 4, 'b')
    prepare(action, 3, 2, 'a')
    assert reads == ['a', 'b'] and drains == [('a',), ('b',)]
    action.invalidate_prefill_plans(('a',))
    assert set(action.prefill_plan_frames) == {'b'}
    prepare(action, 4, 3, 'b')
    assert reads == ['a', 'b']
    action.invalidate_prefill_plans()
    assert not action.prefill_plan_frames


def test_exact_gpu_metadata_must_confirm_cpu_prefill_plan():
    action, _, _, _ = runtime()
    plan = prepare(action, 0, 3)
    action._validate_prefill_plan(plan, [0, 1, 2], [0, 1, 2], 3)
    for glob, local, count in (([0, 1, 3], [0, 1, 2], 3),
                               ([0, 1, 2], [1, 2, 3], 3),
                               ([0, 1], [0, 1], 2)):
        with pytest.raises(RuntimeError, match='coordinates differ'):
            action._validate_prefill_plan(plan, glob, local, count)


def test_missing_stable_prefix_and_noncontiguous_advances_fail_closed():
    action, _, _, _ = runtime()
    authority = action.runtime.authorities['a']
    with pytest.raises(ValueError, match='unpublished prefix'):
        freeze_prefill_frame(authority, authority.action_history, (), 4, 4, [])
    frame = freeze_prefill_frame(authority, authority.action_history, (), 0, 0, [])
    with pytest.raises(ValueError, match='Noncontiguous'):
        frame.extend(4, 4, 4, 4)


@pytest.mark.parametrize('native_retry', [False, True])
def test_real_prepare_capture_and_delayed_accept_match_serial_packets(tmp_path, monkeypatch, native_retry):
    """Exercise production metadata/planning/capture with CPU-backed GPU calls."""
    import torch
    from collections import Counter
    import vllm_kvmem.action_attention as module
    import vllm_kvmem.action_scoring as scoring
    original_empty = torch.empty
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: original_empty(
        *a, **{k: v for k, v in kw.items() if k != 'pin_memory'}))
    class Fence:
        def record(self):
            pass
        def synchronize(self):
            pass
    monkeypatch.setattr(torch.cuda, 'Event', Fence)
    monkeypatch.setattr(module, 'native_vllm_fp8_views', lambda *_: (None, None))
    def score(q, positions, key, table, trig, ks, starts, counts, length, **kwargs):
        # Fixed complete causal softmax; only the GPU operators are replaced.
        logits = torch.arange(length).float()[None, :] * (q.float().sum((1, 2))[:, None] / 1000)
        logits.masked_fill_(torch.arange(length)[None, :] > positions[:, None], -torch.inf)
        probabilities = logits.softmax(-1)
        return torch.stack([probabilities[:, int(s):int(s+n)].sum(-1)
                            for s, n in zip(starts, counts)], 1)
    monkeypatch.setattr(scoring, 'page_attention', score)
    results = []
    for enabled in (False, True):
        action, _, _, drains = runtime(interval=7)
        action.options.prefill_publication_pipeline = enabled
        authority = action.runtime.authorities['a']
        memory = authority.memory = RequestMemory('a', action.options.memory)
        action.runtime.output = tmp_path
        action.runner.block_tables = NS(block_tables=[NS(gpu=torch.tensor([[0, 1, 2, 3, 4, 5]]))])
        action.layers = [(i, NS(attn=NS(layer_name=f'L{i}', kv_cache=None,
            _k_scale=None, _q_scale=None, impl=NS(scale=.5)), num_kv_heads=1,
            head_dim=4, rotary_emb=NS(cos_sin_cache=None))) for i in range(3)]
        action.groups = {f'L{i}': 0 for i in range(3)}
        action.q_dtypes = action.prefill_q_dtypes = {f'L{i}': torch.bfloat16 for i in range(3)}
        action.metadata_early_batches = action.candidate_samples = action.packets = 0
        action.capture_seconds = 0.
        action.profile = action.metadata = None
        class NativeRetry:
            disabled = False
            counts = Counter()
            max_segments = 64
            context = 24
            mode = 'reuse'
            def arm(self, plans):
                pass
            def score(self, *a, **kw):
                return None
            def validate(self):
                if self.disabled:
                    return True
                self.disabled = True
                return False
            def disarm(self):
                pass
            def report(self):
                pass
        action.native = NativeRetry() if native_retry else None
        pending, saved = [], []
        begin = 0
        def publish(item):
            start, count, packets = item
            append(memory, count, local=start)
            action.accept(NS(action_inputs=packets), 'a', count)
        for count in (3, 2, 5, 1):
            positions = torch.arange(begin, begin + count)
            batch = NS(req_ids=['a'], num_reqs=1, is_prefilling_np=[True],
                query_start_loc_np=[0, count], num_computed_tokens_np=[begin],
                positions=positions, input_ids=positions + 20)
            captured = [NS(query=(torch.arange(count * 4).reshape(count, 1, 4) + begin + i).to(torch.bfloat16),
                           positions=positions) for i in range(3)]
            action.prepare(batch, positions)
            packets = action.capture(captured, batch)
            saved.append([(p.rows, p.history, p.prefill, p.masses.tolist()) for p in packets.values()])
            assert action.prefill_plans is None
            if enabled:
                assert action.prefill_plan_frames['a'].history_end == begin + count
            pending.append((begin, count, packets))
            if not enabled or len(pending) > 1:
                publish(pending.pop(0))
            begin += count
        for item in pending:
            publish(item)
        results.append((saved, authority.action_history.prefill.snapshot()))
        assert len(drains) == (1 if enabled else 4)
        assert memory.close()
    assert results[0] == results[1]
