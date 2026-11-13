"""Offline replay of decision traces (vllm_kvmem.strategy.replay) and eviction traces."""
import json
import random

import pytest

from vllm_kvmem.strategy import replay, trace
from test_recall_experiment import native_authority


def starts_of(authority, slots):
    by_slot = {r.handle.slot: r.history_begin for r in authority.records.values()}
    return {by_slot[s] for s in slots}


@pytest.mark.parametrize('seed', range(6))
def test_reference_query_picker_matches_the_native_store(seed):
    rng = random.Random(seed)
    authority = native_authority(count=512, budget=256, sink=2)
    records = sorted(authority.records.values(), key=lambda r: r.history_begin)
    mandatory = [r.handle.slot for r in rng.sample(records[2:], 2)]
    tie = seed % 2 == 0
    scores = {r.handle.slot: float(rng.randint(0, 3)) if tie else rng.random() for r in records}
    native = starts_of(authority, authority.memory.preview_selection(scores, tuple(mandatory), action=False,
                                                                     max_tokens=256))
    fixed = {r.history_begin for r in records if r.history_begin < 64} | starts_of(authority, mandatory)
    mine = replay.pick_query([r.history_begin for r in records], [scores[r.handle.slot] for r in records],
                             fixed, 256)
    assert mine == native


@pytest.mark.parametrize('seed', range(6))
def test_reference_pressure_picker_matches_the_native_store(seed):
    rng = random.Random(seed)
    authority = native_authority(count=512, budget=256, sink=2)
    records = sorted(authority.records.values(), key=lambda r: r.history_begin)
    window = records[:2] + rng.sample(records[2:], 10)
    mandatory = [window[3].handle.slot]
    tie = seed % 2 == 0
    pairs = [(r.handle.slot, float(rng.randint(0, 2)) if tie else rng.random() * (seed % 3 > 0)) for r in window]
    pairs[-1] = (pairs[-1][0], 1.)     # at least one positive score
    native = starts_of(authority, authority.memory.preview_selection(
        {}, tuple(mandatory), action=False, pressure_scores=pairs, max_tokens=256))
    by_slot = {r.handle.slot: r.history_begin for r in records}
    mine = replay.pick_pressure([by_slot[s] for s, _ in pairs], [v for _, v in pairs],
                                starts_of(authority, mandatory), 256, 64)
    assert mine == native


def write_run(tmp_path, *, pressure_selected, query_selected):
    capture = tmp_path / 'server' / 'capture'
    capture.mkdir(parents=True)
    events = [
        dict(event='pressure_selected', request_id='r', accepted_history=320, event_type='decode_pressure', layers=[
            dict(layer=0, actual_method='accumulated_attention_per_layer', candidate_starts=[0, 32, 64, 96, 128],
                 selected_starts=pressure_selected, candidate_scores=[0., 5., 1., 0., 2.],
                 mandatory_starts=[64], budget_tokens=96, sink_tokens=32)]),
        dict(event='query_candidates', request_id='r', budget=96, history=320, candidate_starts=[0, 32, 64, 96],
             candidate_counts=[32, 32, 32, 32], eligible=[False, True, True, True], fixed_starts=[0],
             excluded_starts=[]),
        dict(event='layer_selected', request_id='r', layer=0, selected_starts=query_selected,
             candidate_scores=[0., .9, .1, .5]),
    ]
    (capture / trace.FILE).write_text(''.join(json.dumps(e) + '\n' for e in events))
    return tmp_path


def test_replay_repeats_the_recorded_choice_and_scores_other_strategies(tmp_path):
    run = write_run(tmp_path, pressure_selected=[0, 32, 64], query_selected=[0, 32, 96])
    result = replay.replay(trace.read(run), evidence={'0': [[100, 120]]})
    recorded = result['strategies']['recorded']
    assert result['replayable'] == 2
    assert recorded['pressure']['exact'] == recorded['query']['exact'] == 1
    recency = result['strategies']['recency']
    # Recency keeps the newest blocks: pressure 0 (sink), 64 (mandatory), 128; Query 0, 64, 96.
    assert recency['pressure']['exact'] == 0 and recency['pressure']['overlap'] == pytest.approx(2 / 3)
    assert recency['query']['overlap'] == pytest.approx(2 / 3)
    assert recency['query']['evidence_coverage'] == 1.0 and recorded['query']['evidence_coverage'] == 1.0
    assert replay.main([str(run)]) == 0


def test_replay_reports_a_record_it_cannot_repeat(tmp_path):
    run = write_run(tmp_path, pressure_selected=[0, 64, 96], query_selected=[0, 32, 96])
    result = replay.replay(trace.read(run), strategies=['recency'])
    assert result['strategies']['recorded']['pressure']['first_difference'] == dict(
        request=0, layer=0, history=320, missing=1, extra=1)
    assert replay.main([str(run)]) == 1


def test_evictions_join_the_trace_once_per_request(tmp_path):
    capture = tmp_path / 'capture'
    capture.mkdir()
    rows = [dict(event='evicted', request_id='a', policy='lru', history_begins=[64, 32]),
            dict(event='evicted', request_id='a', policy='lru', history_begins=[96]),
            dict(event='evicted', request_id='b', policy='lru', history_begins=[0])]
    (capture / trace.EVICTION_FILE).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (capture / trace.GLOBAL_FILE).write_text('')
    decisions = trace.read(tmp_path)
    assert [(d.kind, d.request, d.selected, d.method) for d in decisions] == [
        ('evict', 0, (32, 64, 96), 'lru'), ('evict', 1, (0,), 'lru')]
    assert trace.summary(decisions)['evict']['selected_blocks'] == 4


def test_a_run_with_trace_scores_compares_equal_to_one_without(tmp_path):
    with_scores = trace.read(write_run(tmp_path / 'a', pressure_selected=[0, 32, 64], query_selected=[0, 32, 96]))
    plain = [d if d.kind != 'query' else trace.Decision(d.kind, d.request, d.layer, -1, d.method, d.selected, (), ())
             for d in with_scores]
    assert trace.diff(with_scores, plain)['same']
    moved = [trace.Decision(d.kind, d.request, d.layer, d.history + 32, d.method, d.selected, (), ()) for d in with_scores]
    assert not trace.diff(with_scores, moved)['same']


def test_only_policy_retirements_are_logged_as_evictions(tmp_path):
    from types import SimpleNamespace as NS
    from vllm_kvmem.serving.authority import RequestAuthority
    stats = {'retired': 0}
    auth = NS(archive=NS(history_scope_metrics_path=tmp_path / 'historical_gc_scopes.jsonl', _history_policy='lru'),
              gc=NS(scope=NS(stats=stats)), memory=NS(request_id='r'))
    log = tmp_path / trace.EVICTION_FILE
    RequestAuthority._trace_evictions(auth, [(0, 1, 0)])          # a release at hand-off: counter unchanged
    assert not log.exists()
    stats['retired'] = 2
    RequestAuthority._trace_evictions(auth, [(64, 3, 0), (32, 2, 0)])
    RequestAuthority._trace_evictions(auth, [(96, 4, 0)])          # no new policy retirement since
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(r['request_id'], r['policy'], r['history_begins']) for r in rows] == [('r', 'lru', [32, 64])]
