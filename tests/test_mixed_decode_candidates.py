"""KVMEM_MIXED_DECODE_CANDIDATES: prefill+decode batches arm without a pre-forward fence."""
from collections import Counter
from types import SimpleNamespace as NS

import pytest

from vllm_kvmem import action_attention
from vllm_kvmem.action_attention import ActionRuntime, mixed_decode_candidates
from vllm_kvmem.native_attention_planning import make_decode_frame
from vllm_kvmem.window_batch import WindowBatchAdapter


def fixture():
    adapter = WindowBatchAdapter((0,), 128)
    adapter.commit('d', 100, 64)
    action = ActionRuntime.__new__(ActionRuntime)
    action.options = NS(prefill_native_plan_arm=True, prefill_attention=True,
                        prefill_publication_pipeline=False, prefill_action_before_mtp=False)
    action.runner = NS(_kvmem_window_adapter=adapter)
    action.runtime = NS(config=NS(block_tokens=32),
        ensure_authority=lambda r: pytest.fail('candidate planning must not read mutable authority'),
        drain_requests=lambda *a, **k: pytest.fail('candidate planning must not wait for publication'))
    armed = []
    action.native = NS(disabled=None, arm=armed.append, max_rows=16, counts=Counter())
    action.native_cache = {'d': dict(generation=1, offset=36,
        **make_decode_frame([0, 32, 64], [32, 32, 5], [0, 1024, 96], 100, 64, 5, 128, 32))}
    batch = NS(req_ids=['p', 'd'], num_reqs=2, is_prefilling_np=[True, False],
        num_computed_tokens_np=[1000, 100], seq_lens_cpu_upper_bound=[1064, 105],
        query_start_loc_np=[0, 64, 69])
    prefill = dict(pipeline=True, index=0, begin=0, count=64, rows=[31, 63], local_begin=200,
                   starts=[0, 256], counts=[256, 8], history=[0, 992])
    action.prefill_plans = (batch, {'p': prefill})
    action.metadata = NS(batch=batch, indices=(0, 1), slices=((0, 64), (64, 69)), event=None)
    action.native_candidates = action.native_plans = None
    action.mixed_candidates = True
    return action, batch, prefill, armed


def test_mixed_batch_arms_exact_prefill_plans_and_decode_candidates():
    action, batch, prefill, armed = fixture()
    assert action._prepare_mixed_candidates(batch)
    plans, = armed
    assert list(plans) == ['p', 'd'] and plans['p'] is prefill and '_layout' not in prefill
    decode = plans['d']
    assert (decode['index'], decode['begin'], decode['rows']) == (1, 64, [0, 1, 2, 3, 4])
    assert decode['local_begin'] + decode['count'] == 69 and '_layout' in decode
    assert action.native_candidates == (batch, plans) and action.native_plans is None


@pytest.mark.parametrize('change', ['arm_off', 'native_off', 'no_prefill_plans', 'other_batch',
    'pure_decode', 'pure_prefill', 'partial_metadata', 'reordered_metadata', 'not_pipeline',
    'missing_plan', 'plan_index', 'plan_begin', 'plan_count', 'no_cache', 'stale_generation',
    'stale_offset', 'decode_slice', 'extent', 'capacity', 'no_bounds'])
def test_anything_else_keeps_the_eager_planner(change):
    action, batch, prefill, armed = fixture()
    if change == 'arm_off': action.options.prefill_native_plan_arm = False
    elif change == 'native_off': action.native.disabled = 'unsupported'
    elif change == 'no_prefill_plans': action.prefill_plans = None
    elif change == 'other_batch': action.prefill_plans = (object(), {'p': prefill})
    elif change == 'pure_decode': batch.is_prefilling_np[0] = False
    elif change == 'pure_prefill': batch.is_prefilling_np[1] = True
    elif change == 'partial_metadata': action.metadata.indices = (0,)
    elif change == 'reordered_metadata': action.metadata.indices = (1, 0)
    elif change == 'not_pipeline': prefill['pipeline'] = False
    elif change == 'missing_plan': action.prefill_plans = (batch, {})
    elif change == 'plan_index': prefill['index'] = 1
    elif change == 'plan_begin': prefill['begin'] = 1
    elif change == 'plan_count': prefill['count'] = 63
    elif change == 'no_cache': action.native_cache = {}
    elif change == 'stale_generation': action.native_cache['d']['generation'] = 0
    elif change == 'stale_offset': action.native_cache['d']['offset'] = 35
    elif change == 'decode_slice': action.metadata.slices = ((0, 64), (64, 68))
    elif change == 'extent': batch.seq_lens_cpu_upper_bound[1] = 104
    elif change == 'capacity': action.native.max_rows = 6
    elif change == 'no_bounds': del batch.seq_lens_cpu_upper_bound
    assert not action._prepare_mixed_candidates(batch)
    assert not armed and action.native_candidates is None and action.native_plans is None
    reason = {'pure_decode': None, 'pure_prefill': None, 'partial_metadata': 'partial_metadata',
        'reordered_metadata': 'partial_metadata', 'missing_plan': 'prefill_unplanned',
        'not_pipeline': 'prefill_plan_mismatch', 'plan_index': 'prefill_plan_mismatch',
        'plan_begin': 'prefill_plan_mismatch', 'plan_count': 'prefill_plan_mismatch',
        'no_cache': 'decode_frame', 'stale_generation': 'decode_frame', 'stale_offset': 'decode_frame',
        'decode_slice': 'decode_frame', 'extent': 'decode_frame', 'capacity': 'rows'}.get(change, 'unprepared')
    assert action.native.counts == ({} if reason is None else {'mixed_skip:' + reason: 1})


@pytest.mark.parametrize('enabled', [True, False])
def test_prepare_takes_the_candidate_path_only_when_enabled(monkeypatch, enabled):
    action, batch, prefill, armed = fixture()
    action.mixed_candidates = enabled
    metadata, prepared = action.metadata, action.prefill_plans
    class Fence:
        def synchronize(self): raise OSError('eager metadata fence')
    metadata.event = Fence()
    monkeypatch.setattr(action_attention, 'action_metadata', lambda *a, **k: metadata)
    action._collect_deferred_scores = lambda: None
    action.options.prefill_publication_pipeline = True
    action._prepare_prefill_pipeline = lambda b: setattr(action, 'prefill_plans', prepared)
    action.native.disarm = lambda: pytest.fail('a supported batch is never disarmed here')
    if enabled:
        action.prepare(batch, None)
        assert action.native.counts == {'prepare:mixed_candidates': 1} and len(armed) == 1
    else:
        with pytest.raises(OSError, match='eager metadata fence'):
            action.prepare(batch, None)
        assert action.native.counts == {'prepare:eager': 1} and not armed


class Resolved(Exception):
    pass


def capture_fixture():
    action, batch, prefill, armed = fixture()
    assert action._prepare_mixed_candidates(batch)
    action.metadata.host = NS(tolist=lambda: (list(range(69)), list(range(1000, 1064)) + list(range(100, 105)),
                                              list(range(200, 264)) + list(range(64, 69))))
    action.metadata_early_batches = 0
    planned, resolved = [], []
    def plan_owner(request_id, prefilling, tokens, global_positions, local_positions, count):
        planned.append((request_id, prefilling, tokens, global_positions, local_positions, count))
        return dict(rows=[], sources=(), history_begin=100, local_begin=64)
    action._plan_owner = plan_owner
    def resolve(plans):
        resolved.append(plans)
        raise Resolved
    action.native.resolve = resolve
    return action, batch, prefill, planned, resolved


def test_capture_reuses_the_armed_prefill_plan_and_plans_only_decode_rows():
    action, batch, prefill, planned, resolved = capture_fixture()
    with pytest.raises(Resolved):
        action._capture([], batch)
    assert planned == [('d', False, [64, 65, 66, 67, 68], [100, 101, 102, 103, 104],
                        [64, 65, 66, 67, 68], 5)]
    plans, = resolved
    assert list(plans) == ['p', 'd'] and plans['p'] is prefill
    assert (plans['d']['index'], plans['d']['begin'], plans['d']['count']) == (1, 64, 5)


@pytest.mark.parametrize('change', ['index', 'begin', 'count', 'decode'])
def test_capture_rejects_a_prefill_plan_that_differs_from_the_batch(change):
    action, batch, prefill, planned, resolved = capture_fixture()
    if change == 'decode': batch.is_prefilling_np[0] = False
    else: prefill[change] += 1
    with pytest.raises(RuntimeError, match='differs from the captured batch'):
        action._capture([], batch)
    assert not planned and not resolved


def test_flag_is_off_by_default_and_validated(monkeypatch):
    monkeypatch.delenv('KVMEM_MIXED_DECODE_CANDIDATES', raising=False)
    assert not mixed_decode_candidates()
    monkeypatch.setenv('KVMEM_MIXED_DECODE_CANDIDATES', '1')
    assert mixed_decode_candidates()
    monkeypatch.setenv('KVMEM_MIXED_DECODE_CANDIDATES', 'yes')
    with pytest.raises(ValueError, match='0 or 1'):
        mixed_decode_candidates()
