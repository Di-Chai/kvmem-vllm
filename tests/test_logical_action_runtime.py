"""Physical Action capture with CPU-backed score/DMA operators only."""
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

import vllm_kvmem.action_attention as module
import vllm_kvmem.action_scoring as scoring
from vllm_kvmem.action_attention import ActionRuntime
from vllm_kvmem.action_download_pool import ActionDownloadPool
from vllm_kvmem.logical_action import LogicalActionResult
from test_action_download_pool import Backend
from test_logical_action import fixture, state


def runtime(monkeypatch, *, retry=False, interval=2048, pooled_metadata=False, chunks=2):
    group, _, _, source = fixture(chunks=chunks)
    current = state(True)
    opts = current.options = current.prefill.options
    opts.prefill_attention = opts.prefill_publication_pipeline = True
    opts.prefill_physical_coalescing = opts.prefill_native_plan_arm = True
    opts.prefill_physical_max_chunks = chunks
    opts.prefill_attention_interval = interval
    authority = NS(action_history=current, block_graph=object(), memory=NS(_store=NS(
        working_records=lambda: [(0, 0, 0, 32, 0), (1, 0, 4096, 32, 32)])))
    action = ActionRuntime.__new__(ActionRuntime)
    action.options, action.ids = opts, (1, 2, 3, 4, 5)
    action.runner = NS(req_states=NS(req_id_to_index={'r': 0}, prompt_len=NS(np=[group.prompt_tokens])),
        _kvmem_request_generations={'r': 2},
        _kvmem_window_adapter=NS(generations={'r': 3}, offsets={'r': 8128}),
        _kvmem_logical_prefill_groups={'r': replace(group, owner_epoch=('scheduler', 7, 'r', 8128))},
        _kvmem_logical_prefill_step=7,
        block_tables=NS(block_tables=[NS(gpu=torch.zeros((1, 4096), dtype=torch.int32))]))
    action.runtime = NS(config=opts.memory, authorities={'r': authority},
        drain_requests=lambda *_a, **_kw: True, ensure_authority=lambda _: authority,
        prefill_pipeline_can_advance=lambda _: True)
    action.download_pool = ActionDownloadPool(524288 if chunks == 2 else (8<<20 if chunks == 8 else 4<<20),
                                              4, backend=Backend())
    if pooled_metadata:
        original_metadata = module.action_metadata
        def metadata(*args, **kwargs):
            value = original_metadata(*args, **kwargs)
            value.host, value.event = action.download_pool.copy(value.host)
            return value
        monkeypatch.setattr(module, 'action_metadata', metadata)
    action.metadata_upload = None
    action.profile = action.metadata = None
    action.candidate_samples = action.packets = action.metadata_early_batches = 0
    action.capture_seconds = 0.
    action.layers = [(i, NS(attn=NS(layer_name=f'L{i}', kv_cache=None,
        _k_scale=None, _q_scale=None, impl=NS(scale=.5)), num_kv_heads=1,
        head_dim=4, rotary_emb=NS(cos_sin_cache=None))) for i in range(3)]
    action.groups = {f'L{i}': 0 for i in range(3)}
    action.q_dtypes = action.prefill_q_dtypes = {f'L{i}': torch.bfloat16 for i in range(3)}
    calls = []

    class Native:
        disabled = False
        mode = 'reuse'
        max_rows = max(256, chunks*64)
        max_segments = 2202
        context = 65576
        width = 6301
        rows = torch.empty(16424, dtype=torch.int32)
        counts = Counter()
        invalid = False
        def arm(self, plans):
            self.plans = plans
            calls.append('arm')
        def score(self, *_a, **_kw):
            calls.append('native score')
            if self.invalid:
                return None
            plan = self.plans['r']
            indices = [row//32 for row in plan['rows']]
            return torch.from_numpy(source[indices].copy())
        def validate(self):
            calls.append('validate')
            # The real validation event follows all score D2H on the same
            # stream. Model that completion, not merely flag availability.
            for event in action.download_pool.backend.events:
                event.ready = True
            if retry and not self.invalid:
                self.invalid = True
                return False
            return True
        def enqueue_validation(self):
            self._validation_ticket = NS(resolved=False)
            calls.append('record validation')
            return self._validation_ticket
        def resolve_validation(self, ticket):
            assert ticket is self._validation_ticket and not ticket.resolved
            valid = self.validate()
            ticket.resolved = True
            self._validation_ticket = None
            return valid
        def disarm(self):
            calls.append('disarm')
        def report(self):
            pass
    action.native = Native()
    monkeypatch.setattr(module, 'native_vllm_fp8_views', lambda *_: (None, None))
    def reference(query, positions, key, table, trig, key_scale, starts, counts, length, **kwargs):
        calls.append('reference score')
        assert (length, len(starts), len(counts)) in tuple(
            (64+(i+1)*2048, 2+(i+1)*64, 2+(i+1)*64) for i in range(chunks))
        assert len(query) == 64
        indices = [(int(position)-64)//32 for position in positions]
        return torch.from_numpy(source[indices, :len(starts)].copy())
    monkeypatch.setattr(scoring, 'page_attention', reference)
    batch = NS(req_ids=['r'], num_reqs=1, query_start_loc_np=[0, group.count],
        num_computed_tokens_np=[8192], is_prefilling_np=[True],
        positions=torch.arange(8192, group.history_end), input_ids=torch.arange(group.count))
    captured = [NS(query=torch.ones((group.count, 1, 4), dtype=torch.bfloat16),
                   positions=torch.arange(64, 64+group.count)) for _ in range(3)]
    return action, batch, captured, current, calls


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('interval', [2048, 4096, 8192])
@pytest.mark.parametrize('pooled_metadata', [False, True])
def test_capture_preserves_two_logical_packets_native_fallback_and_no_sampled_rows(monkeypatch, retry, interval, pooled_metadata):
    action, batch, captured, current, calls = runtime(monkeypatch, retry=retry, interval=interval,
                                                    pooled_metadata=pooled_metadata)
    before = current.prefill.snapshot()
    action.prepare(batch, captured[0].positions)
    plan = action.prefill_plans[1]['r']
    assert plan['count'] == 4096 and len(plan['history']) == 130
    assert plan['rows'] == [row for offset in (0, 2048)
                           if interval == 2048 or interval == 4096 and offset == 2048
                           for row in range(offset+31, offset+2048, 32)]
    bound = plan['logical_group']
    assert bound.owner_epoch == (2, 3, 8128, 0, None)
    assert bound.owner_epoch != action.runner._kvmem_logical_prefill_groups['r'].owner_epoch
    result = action.capture(captured, batch)
    assert isinstance(result, LogicalActionResult) and set(result) == {'r'}
    parts = result.logical_slices['r']
    assert result.logical_groups == {'r': bound}
    assert [part.plan.span.count for part in parts] == [2048, 2048]
    assert [part.masses.shape[1] for part in parts] == [66, 130]
    for part in parts:
        assert part.source_owner is result['r'].masses
        assert part.event is result['r'].event
        assert part.masses.untyped_storage().data_ptr() == result['r'].masses.untyped_storage().data_ptr()
    assert current.prefill.snapshot() == before
    assert action.candidate_samples == sum(len(part.plan.rows) for part in parts)
    assert action.packets == 1 and action.prefill_plans is None
    assert action.prefill_plan_frames['r'].history_end == 12288
    assert action.logical_prefill_stats == dict(planned_physical_groups=1, planned_logical_chunks=2,
        projected_physical_groups=1, projected_logical_chunks=2, sampled_rows=len(plan['rows']),
        **({'validation_fallback_groups': 1} if retry else {}))
    if retry and interval < 8192:
        assert calls.count('reference score') == (6 if interval == 2048 else 3)
        assert action.download_pool.snapshot()['reuses'] == 1
    for call in (lambda: action.accept(NS(action_inputs=result), 'r', 4096),
                 lambda: action.prepare_accept(NS(action_inputs=result), 'r', 4096,
                    graph=None, history_begin=8192)):
        with pytest.raises(RuntimeError, match='ordered logical publication'):
            call()
        assert 'r' in result and current.prefill.snapshot() == before
    result.clear()
    assert result == {} and not result.logical_slices and not result.logical_groups


def test_first_logical_chunk_sampled_second_skipped_uses_original_two_boundaries(monkeypatch):
    action, batch, captured, current, _ = runtime(monkeypatch, interval=8192)
    current.prefill.end = 0
    action.prepare(batch, captured[0].positions)
    result = action.capture(captured, batch)
    first, second = result.logical_slices['r']
    assert len(first.plan.rows) == 64 and second.plan.rows == ()
    assert first.masses.shape == (64, 66) and second.masses.shape == (0, 130)
    assert action.prefill_plan_frames['r'].sampled_end == 10240


@pytest.mark.parametrize('retry', [False, True])
@pytest.mark.parametrize('cancel', [False, True])
def test_actual_early_capture_preserves_logical_projection_and_drops_all_tentative_leases(monkeypatch, retry, cancel):
    action, batch, captured, current, calls = runtime(monkeypatch, retry=retry, pooled_metadata=True)
    action.options.prefill_action_before_mtp = True
    action.runner._kvmem_layers = [f'L{i}' for i in range(3)]
    action.runtime._raise_publication_error = lambda: None
    for i, value in enumerate(captured):
        value.name, value.key = f'L{i}', value.query
    action.prepare(batch, captured[0].positions)
    before = current.prefill.snapshot()
    assert action.stage_before_mtp(captured, batch)
    refs = [weakref.ref(part.masses) for part in action._mtp_capture_frame.packets.logical_slices['r']]
    assert action.logical_prefill_stats == dict(planned_physical_groups=1, planned_logical_chunks=2)
    calls.append('MTP submitted')
    if cancel:
        action.cancel_before_mtp()
        assert all(ref() is None for ref in refs)
        assert action.candidate_samples == action.packets == 0
        assert 'projected_physical_groups' not in action.logical_prefill_stats
    else:
        result = action.capture(captured, batch)
        assert len(result.logical_slices['r']) == 2
        assert action.logical_prefill_stats['projected_physical_groups'] == 1
        assert action.logical_prefill_stats['sampled_rows'] == 128
        assert action.candidate_samples == 128 and action.packets == 1
        if retry:
            assert all(ref() is None for ref in refs)
            assert action.logical_prefill_stats['validation_fallback_groups'] == 1
            assert action.download_pool.snapshot()['reuses'] == 1
    assert action._mtp_capture_frame is None and current.prefill.snapshot() == before
    assert calls.index('record validation') < calls.index('MTP submitted') < calls.index('validate')


@pytest.mark.parametrize('change', ['step', 'history', 'offset', 'prompt', 'capacity', 'policy',
    'owner', 'decode', 'terminal', 'replay', 'disabled', 'mode', 'rows', 'segments', 'context',
    'width', 'slots', 'bytes', 'metadata_capacity', 'flag'])
def test_invalid_physical_receipt_cannot_arm_or_advance_a_frame(monkeypatch, change):
    action, batch, captured, current, calls = runtime(monkeypatch)
    group = action.runner._kvmem_logical_prefill_groups['r']
    if change == 'step': action.runner._kvmem_logical_prefill_step = 8
    elif change == 'history': batch.num_computed_tokens_np[0] += 32
    elif change == 'offset': action.runner._kvmem_window_adapter.offsets['r'] += 32
    elif change == 'prompt': action.runner.req_states.prompt_len.np[0] += 32
    elif change == 'capacity': action.runtime.config.select_budget += 32
    elif change == 'policy': action.runner._kvmem_logical_prefill_groups['r'] = replace(
        group, request_plan=replace(group.request_plan, gen_budget=8192))
    elif change == 'owner': action.runner._kvmem_logical_prefill_groups = {'other': group}
    elif change == 'decode': batch.is_prefilling_np[0] = False
    elif change == 'terminal': action.runner.req_states.prompt_len.np[0] = 12288
    elif change == 'replay': current.prefill_replay = (10240, 12288)
    elif change == 'disabled': action.native.disabled = True
    elif change == 'mode': action.native.mode = 'verify'
    elif change == 'rows': action.native.max_rows = 127
    elif change == 'segments': action.native.max_segments = 129
    elif change == 'context': action.native.context = 4159
    elif change == 'width': action.native.width = 259
    elif change == 'slots': action.download_pool.max_slots = 3
    elif change == 'bytes': action.download_pool.max_bytes = 524287
    elif change == 'metadata_capacity': action.metadata_upload = NS(capacity=1)
    elif change == 'flag': action.options.prefill_physical_coalescing = False
    before = current.prefill.snapshot()
    with pytest.raises((RuntimeError, ValueError)):
        action.prepare(batch, captured[0].positions)
    assert 'arm' not in calls and current.prefill.snapshot() == before
    assert not getattr(action, 'prefill_plan_frames', {})


def test_physical_gpu_coordinates_still_must_confirm_the_logical_cpu_receipt(monkeypatch):
    action, batch, captured, current, calls = runtime(monkeypatch)
    batch.positions[-1] += 1
    action.prepare(batch, captured[0].positions)
    before = current.prefill.snapshot()
    with pytest.raises(RuntimeError, match='coordinates differ'):
        action.capture(captured, batch)
    assert 'native score' not in calls and current.prefill.snapshot() == before


def test_receipt_cannot_silently_become_one_4k_transaction_without_pre_forward_plan(monkeypatch):
    action, batch, captured, current, calls = runtime(monkeypatch)
    action.prepare(batch, captured[0].positions)
    action.prefill_plans = action.native_plans = None
    action._plan_owner = lambda *_: pytest.fail('Receipt loss must fail before original owner preparation')
    before = current.prefill.snapshot()
    with pytest.raises(RuntimeError, match='pre-forward worker plan'):
        action.capture(captured, batch)
    assert 'native score' not in calls and current.prefill.snapshot() == before
