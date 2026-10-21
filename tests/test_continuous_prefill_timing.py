"""Completed-event continuity without extra CUDA events or blocking fences."""
import copy
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import weakref

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from benchmark_active_prefill import (STEADY_SHAPES, continuous_phase_span,
                                      continuous_steady_phases)
from native_prefill_shape_probe import NativeTiming
from vllm_kvmem.step_timing import StepTiming


class EventClock:
    def __init__(self):
        self.now = 0
        self.completed = -1
        self.events = []

    def event(self):
        event = Event(self)
        self.events.append(weakref.ref(event))
        return event


class Event:
    def __init__(self, clock):
        self.clock = clock
        self.at = None

    def record(self):
        assert self.at is None
        self.at = self.clock.now

    def query(self):
        return self.at is not None and self.at <= self.clock.completed

    def elapsed_time(self, end):
        assert self.query() and end.query(), 'Elapsed time queried before event completion'
        return end.at - self.at

    def synchronize(self):
        pytest.fail('Timing must never synchronize')


def batch(history=0):
    return NS(req_ids=['owner'], num_computed_tokens_np=[history],
              num_scheduled_tokens=[32], is_prefilling_np=[True],
              num_computed_prefill_tokens_np=[history], prefill_len_np=[1000000])


def record(timing, clock, start, model, end, history=0):
    clock.now = start
    timing.begin()
    if model is not None:
        clock.now = model
        timing.model_begin(batch(history))
    clock.now = end
    timing.end()


@pytest.fixture(params=['kvmem', 'native'])
def timer(request):
    clock = EventClock()
    timing = (StepTiming(1, None, event_factory=clock.event) if request.param == 'kvmem'
              else NativeTiming(None, event_factory=clock.event))
    return timing, clock


def test_cross_poll_links_keep_exactly_one_completed_event(timer):
    timing, clock = timer
    record(timing, clock, 0, 1, 5)
    assert timing.poll() == []
    assert timing.previous_completed is None
    clock.completed = 5
    first, = timing.poll()
    assert first['previous_sequence'] is None and first['previous_end_to_model_ms'] is None
    assert first['call_index'] == 1
    assert sum(ref() is not None for ref in clock.events) == 1
    record(timing, clock, 6, 8, 11, 32)
    assert timing.poll() == []
    assert timing.previous_completed[:2] == (1, 1)
    clock.completed = 11
    second, = timing.poll()
    assert second['previous_sequence'] == second['previous_call_index'] == 1
    assert second['previous_end_to_model_ms'] == 3
    assert first['model_ms'] + second['model_ms'] + second['previous_end_to_model_ms'] == 10
    assert len(clock.events) == 6
    assert sum(ref() is not None for ref in clock.events) == 1


def test_multiple_completed_rows_in_one_poll_are_chained(timer):
    timing, clock = timer
    record(timing, clock, 0, 1, 5)
    record(timing, clock, 6, 9, 12, 32)
    clock.completed = 12
    first, second = timing.poll()
    assert second['previous_sequence'] == first['sequence']
    assert second['previous_end_to_model_ms'] == 4


def test_missing_model_event_is_not_read_or_mislabeled(timer):
    timing, clock = timer
    record(timing, clock, 0, 1, 5)
    record(timing, clock, 6, None, 7, 32)
    clock.completed = 7
    first, second = timing.poll()
    assert second['previous_sequence'] == first['sequence']
    assert second['previous_end_to_model_ms'] is None and 'model_ms' not in second


def test_negative_event_gap_is_error_not_clipped(timer):
    timing, clock = timer
    record(timing, clock, 0, 1, 5)
    record(timing, clock, 6, 4, 9, 32)
    clock.completed = 9
    with pytest.raises(RuntimeError, match='CUDA timing gap'):
        timing.poll()


def test_sampling_interval_retains_actual_call_index():
    clock = EventClock()
    timing = StepTiming(2, None, event_factory=clock.event)
    for i in range(3):
        record(timing, clock, 10*i, 10*i+1, 10*i+5, 32*i)
    clock.completed = 30
    first, second = timing.poll()
    assert [first['sequence'], second['sequence']] == [1, 2]
    assert [first['call_index'], second['call_index']] == [1, 3]
    assert second['previous_call_index'] == 1
    assert len(clock.events) == 6


def test_capacity_drops_are_visible_in_call_index(timer):
    timing, clock = timer
    for i in range(20):
        record(timing, clock, 10*i, 10*i+1, 10*i+5, 32*i)
    assert timing.dropped == 4 and len(clock.events) == 48
    clock.completed = 200
    assert len(timing.poll()) == 16
    record(timing, clock, 201, 202, 205, 640)
    clock.completed = 205
    current, = timing.poll()
    assert current['sequence'] == 17 and current['call_index'] == 21
    assert current['previous_call_index'] == 16
    assert sum(ref() is not None for ref in clock.events) == 1


def phase(offset=16320):
    return [dict(sequence=i+50, call_index=i+80, previous_sequence=i+49,
                 previous_call_index=i+79, previous_end_to_model_ms=999 if i == 0 else i,
                 model_ms=100+i, decode_requests=0,
                 batch_geometry=[dict(request_id='owner', query_tokens=q, active_seq_len=k,
                     logical_seq_len=k+offset, offset=offset, prefill=True, prefill_terminal=False)])
            for i, (q, k) in enumerate(STEADY_SHAPES)]


def test_continuous_phase_uses_interior_gaps_and_excludes_pressure():
    result = continuous_phase_span(phase())
    assert result['new_input_tokens'] == 16320 and result['steps'] == 9
    assert result['model_seconds'] == .936 and result['interior_gap_seconds'] == .036
    assert result['model_start_to_last_end_seconds'] == .972
    assert result['new_tokens_per_second'] == pytest.approx(16320/.972)
    assert continuous_steady_phases(phase())['matched_phases'] == 1


@pytest.mark.parametrize('fault', ['owner', 'offset', 'logical_gap', 'shape', 'terminal', 'decode',
                                  'atomic', 'sequence', 'previous_sequence', 'call_index',
                                  'previous_call_index', 'negative_gap', 'nan_gap', 'none_gap', 'nan_model'])
def test_continuous_phase_rejects_ambiguous_or_broken_spans(fault):
    rows = phase()
    row, geometry = rows[4], rows[4]['batch_geometry'][0]
    if fault == 'owner': geometry['request_id'] = 'other'
    elif fault == 'offset': geometry['offset'] += 1
    elif fault == 'logical_gap': geometry['logical_seq_len'] += 1
    elif fault == 'shape': geometry['query_tokens'] += 1
    elif fault == 'terminal': geometry['prefill_terminal'] = True
    elif fault == 'decode': row['decode_requests'] = 1
    elif fault == 'atomic': row['atomic_prefill_tokens'] = 32
    elif fault in ('sequence', 'previous_sequence', 'call_index', 'previous_call_index'): row[fault] += 1
    elif fault == 'negative_gap': row['previous_end_to_model_ms'] = -1
    elif fault == 'nan_gap': row['previous_end_to_model_ms'] = float('nan')
    elif fault == 'none_gap': row['previous_end_to_model_ms'] = None
    else: row['model_ms'] = float('nan')
    with pytest.raises(ValueError):
        continuous_phase_span(rows)


def test_historical_rows_remain_diagnostic_without_claimed_span():
    rows = phase()
    for row in rows:
        for field in ('call_index', 'previous_call_index', 'previous_sequence', 'previous_end_to_model_ms'):
            del row[field]
    result = continuous_steady_phases(rows)
    assert result['status'] == 'unavailable' and result['matched_phases'] == 1
    assert 'model_start_to_last_end_seconds' not in result['phases'][0]


def test_short_tail_and_full32k_do_not_masquerade_as_steady_phase():
    assert continuous_steady_phases(phase()[:-1])['status'] == 'no_complete_steady_phase'
    rows = copy.deepcopy(phase())
    rows[1]['batch_geometry'][0].update(query_tokens=736, active_seq_len=33536)
    assert continuous_steady_phases(rows)['matched_phases'] == 0
