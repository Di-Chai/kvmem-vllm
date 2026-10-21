from types import SimpleNamespace as NS

from vllm_kvmem.step_timing import StepTiming


class Event:
    def record(self):
        pass

    def query(self):
        return True

    def elapsed_time(self, other):
        return 1.0


def test_geometry_retains_partial_chunks_and_per_owner_active_offsets():
    timing = StepTiming(1, None, event_factory=Event, clock=lambda: 10.0)
    timing.begin()
    timing.model_begin(NS(req_ids=['fresh', 'window', 'decode'],
        num_computed_tokens_np=[30720, 261248, 100003],
        num_scheduled_tokens=[2048, 736, 5], is_prefilling_np=[True, True, False]),
        {'window': 228480, 'decode': 67232})
    timing.end()
    row, = timing.poll()
    assert row['batch_geometry'] == [
        dict(request_id='fresh', query_tokens=2048, active_seq_len=32768,
             logical_seq_len=32768, offset=0, prefill=True),
        dict(request_id='window', query_tokens=736, active_seq_len=33504,
             logical_seq_len=261984, offset=228480, prefill=True),
        dict(request_id='decode', query_tokens=5, active_seq_len=32776,
             logical_seq_len=100008, offset=67232, prefill=False),
    ]
    assert row['prefill_tokens'] == 2784
    assert row['decode_requests'] == 1


def test_geometry_is_not_reused_between_batches():
    timing = StepTiming(1, None, event_factory=Event, clock=lambda: 10.0)
    timing.begin()
    timing.model_begin(NS(req_ids=['window'], num_computed_tokens_np=[100000],
        num_scheduled_tokens=[32], is_prefilling_np=[True]), {'window': 67232})
    timing.end()
    timing.begin()
    timing.model_begin(NS(req_ids=['fresh'], num_computed_tokens_np=[0],
        num_scheduled_tokens=[2048], is_prefilling_np=[True]))
    timing.end()
    rows = timing.poll()
    assert rows[0]['batch_geometry'][0]['active_seq_len'] == 32800
    assert rows[1]['batch_geometry'][0]['active_seq_len'] == 2048
