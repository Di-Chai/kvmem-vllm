"""CPU control-flow checks for native prefill arming from frozen plans."""
from collections import Counter
from types import SimpleNamespace as NS

import pytest

import vllm_kvmem.action_attention as module
from vllm_kvmem.action_attention import ActionMetadata
from test_prefill_planning import runtime


def prepared_runtime(monkeypatch, owners=('a',), *, enabled=True, actual_positions=None):
    action, records, reads, drains = runtime(owners)
    action.options.prefill_native_plan_arm = enabled
    action.profile = None
    action.candidate_samples = action.packets = action.metadata_early_batches = 0
    action.capture_seconds = 0.
    calls = []
    class Fence:
        def synchronize(self):
            calls.append('metadata fence')
    class Host:
        def tolist(self):
            calls.append('metadata read')
            positions = actual_positions or [0, 1, 2] * len(owners)
            return [list(range(20, 20 + len(positions))), positions, positions]
    def metadata(batch, *_a, **_kw):
        return ActionMetadata(batch, tuple(range(len(owners))),
            tuple((i * 3, (i + 1) * 3) for i in range(len(owners))), Host(), Fence())
    monkeypatch.setattr(module, 'action_metadata', metadata)
    class Native:
        disabled = False
        max_segments = 0  # Avoid unrelated decode-cache construction in host tests.
        counts = Counter()
        def arm(self, plans):
            calls.append(('arm', tuple(plans)))
        def disarm(self):
            calls.append('disarm')
        def report(self):
            pass
        def validate(self):
            calls.append('validate')
            return True
    action.native = Native()
    batch = NS(req_ids=list(owners), num_reqs=len(owners),
        query_start_loc_np=[3 * i for i in range(len(owners) + 1)],
        num_computed_tokens_np=[0] * len(owners), is_prefilling_np=[True] * len(owners))
    return action, batch, calls, records, reads, drains


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('owners', [('a',), ('a', 'b')])
def test_prepare_defers_only_metadata_fence_and_capture_still_checks_actual_coordinates(
        monkeypatch, enabled, owners):
    action, batch, calls, _, _, _ = prepared_runtime(monkeypatch, owners, enabled=enabled)
    for authority in action.runtime.authorities.values():
        authority.action_history.prefill.end = 100  # No sampled rows/kernel calls required.
    action.prepare(batch, None)
    assert ('arm', owners) in calls
    assert ('metadata fence' in calls) is not enabled
    assert ('metadata read' in calls) is not enabled
    assert action.native.counts['prepare:prefill_plan'] == int(enabled)
    assert action.native.counts['prepare:eager'] == int(not enabled)
    calls.clear()
    assert action.capture([], batch) == {}
    assert calls == ['metadata fence', 'metadata read', 'validate', 'disarm']


def test_gpu_coordinate_mismatch_cannot_escape_as_native_score_packet(monkeypatch):
    action, batch, calls, _, _, _ = prepared_runtime(monkeypatch, actual_positions=[0, 1, 9])
    action.prepare(batch, None)
    assert calls == [('arm', ('a',))]
    with pytest.raises(RuntimeError, match='coordinates differ'):
        action.capture([], batch)
    assert 'validate' not in calls
    assert action.candidate_samples == action.packets == 0
    assert action.runtime.authorities['a'].action_history.prefill.end == 0


@pytest.mark.parametrize('missing', ['owner', 'batch', 'count', 'index', 'begin', 'pipeline', 'decode'])
def test_incomplete_or_changed_plan_never_arms_without_original_metadata_fence(monkeypatch, missing):
    action, batch, calls, _, _, _ = prepared_runtime(monkeypatch, ('a', 'b'))
    action.pipeline_requests = set()
    action._prepare_prefill_pipeline(batch)
    action.metadata = module.action_metadata(batch, None, True)
    if missing == 'owner':
        action.prefill_plans[1].pop('b')
    elif missing == 'batch':
        action.prefill_plans = (NS(), action.prefill_plans[1])
    elif missing == 'decode':
        batch.is_prefilling_np[1] = False
    else:
        action.prefill_plans[1]['b'][missing] = False if missing == 'pipeline' else 999
    assert not action._arm_prefill_plans(batch)
    assert calls == []
    assert action.native.counts['prepare:prefill_plan'] == 0


@pytest.mark.parametrize('excluded', ['replay', 'terminal'])
def test_replay_and_terminal_prefill_keep_eager_validation(monkeypatch, excluded):
    action, batch, calls, _, _, _ = prepared_runtime(monkeypatch)
    authority = action.runtime.authorities['a']
    if excluded == 'replay':
        authority.action_history.prefill_replay = (0, 3)
    else:
        action.runner.req_states.prompt_len.np[0] = 3
    action.prepare(batch, None)
    assert calls[:2] == ['metadata fence', 'metadata read']
    assert action.native.counts['prepare:eager'] == 1
    assert not action.native.counts['prepare:prefill_plan']


def test_prefill_plan_arm_preserves_packets_on_native_validation_fallback(tmp_path, monkeypatch):
    import test_prefill_planning as planning
    original_options = planning.options
    monkeypatch.setattr(planning, 'options', lambda **kw: original_options(
        prefill_native_plan_arm=True, **kw))
    # This integration exercises actual prepare/capture and native failure
    # recomputation, then compares all packets and accepted scores to serial.
    planning.test_real_prepare_capture_and_delayed_accept_match_serial_packets(
        tmp_path, monkeypatch, native_retry=True)
