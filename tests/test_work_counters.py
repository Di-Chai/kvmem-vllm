import json
from types import SimpleNamespace

import pytest

from vllm_kvmem.work_counters import PublicWorkCounters, classify_public_batch


class Clock:
    now = 0.

    def __call__(self):
        return self.now


class Event:
    def __init__(self):
        self.ready = False
        self.recorded = 0

    def record(self):
        self.recorded += 1

    def query(self):
        return self.ready

    def synchronize(self):
        raise AssertionError('Public work observations must never synchronize')


def batch(scheduled=(128,), lengths=(1024,), computed=(0,), *, history=None,
          tokens=None, padded_tokens=4096, padded_requests=8):
    return SimpleNamespace(num_reqs=len(scheduled), req_ids=[f'r{i}' for i in range(len(scheduled))],
        num_scheduled_tokens=scheduled, prefill_len_np=lengths,
        num_computed_prefill_tokens_np=computed, num_computed_tokens_np=history or computed,
        num_tokens=sum(scheduled) if tokens is None else tokens,
        num_tokens_after_padding=padded_tokens, num_reqs_after_padding=padded_requests)


def observer(tmp_path):
    clock = Clock()
    events = []

    def factory():
        event = Event()
        events.append(event)
        return event

    work = PublicWorkCounters(tmp_path/'public_work.json', event_factory=factory,
        clock=clock, wall_clock=lambda: 1000+clock())
    return work, clock, events


def test_mixed_last_prefill_excludes_mtp_lookahead_padding_and_restored_history():
    row = classify_public_batch(batch((21, 5), (131089, 90000), (131072, 90000),
        history=(131072, 103000)), {'r0':65536, 'r1':70000})
    assert row['public_prefill_tokens'] == 17
    assert row['public_non_prefill_tokens'] == 9
    assert row['prefill_request_batches'] == row['decode_request_batches'] == 1
    assert row['prefill_context_bucket'] == 65536
    assert row['decode_context_bucket'] == 32768


def test_compacted_draft_positions_use_actual_token_total():
    row = classify_public_batch(batch((128, 5, 5), (1024, 500, 500), (0, 500, 500),
        history=(0, 600, 700), tokens=134))
    assert row['public_prefill_tokens'] == 128
    assert row['public_non_prefill_tokens'] == 6
    assert row['decode_request_batches'] == 2


@pytest.mark.parametrize('value', [
    batch((5,), (10,), (11,)),
    batch((128,), tokens=127),
    batch((128,), tokens=129),
    batch((0,)),
])
def test_invalid_metadata_cannot_silently_publish_false_measurements(value):
    with pytest.raises(ValueError):
        classify_public_batch(value)


def test_slow_fence_preserves_every_batch_and_idle_flush_confirms_tail(tmp_path):
    work, clock, events = observer(tmp_path)
    work.poll(force=True)
    work.submit(batch())
    for i in range(1, 101):
        clock.now = i / 10
        work.submit(batch())
    assert len(events) == 1  # No unbounded pending queue or dropped work.
    row = work.snapshot()
    assert row['submitted']['public_batches'] == 101
    assert row['submitted']['public_prefill_tokens'] == 12928
    assert row['completed']['public_batches'] == 0
    assert row['not_yet_confirmed']['public_batches'] == 101
    events[0].ready = True
    clock.now = 10.1
    row = work.poll(force=True, flush=True)
    assert row['completed']['public_batches'] == 1
    assert row['completed_fence']['completion_lower_bound'] == 10.
    assert row['completed_fence']['completion_upper_bound'] == 10.1
    assert len(events) == 2
    assert row['pending_fence']['totals']['public_batches'] == 101
    events[1].ready = True
    clock.now = 10.2
    row = work.poll(force=True, flush=True)
    assert row['all_submitted_completed']
    assert row['completed']['public_prefill_tokens'] == 12928
    assert row['not_yet_confirmed']['public_batches'] == 0
    assert row['completed']['batch_composition'][0]['batches'] == 101
    assert json.loads(work.output.read_text()) == row
    snapshots = [json.loads(line) for line in work.events_output.read_text().splitlines()]
    assert len(snapshots) <= 14  # Low-frequency publication, not one row per batch.
    assert all(r['counter_id'] == row['counter_id'] for r in snapshots)


def test_completion_and_submit_counters_have_no_shared_mutable_histogram(tmp_path):
    work, clock, events = observer(tmp_path)
    work.submit(batch())
    events[0].ready = True
    clock.now = .1
    work.poll()
    work.submit(batch((5,), (1000,), (1000,), history=(2000,)))
    row = work.snapshot()
    assert row['completed']['public_batches'] == 1
    assert len(row['completed']['batch_composition']) == 1
    assert row['submitted']['public_batches'] == 2
    assert len(row['submitted']['batch_composition']) == 2
    # The final subsecond interval is fenced by an idle connector poll.
    work.poll(flush=True)
    assert len(events) == 2
    events[1].ready = True
    clock.now = .2
    work.poll(force=True, flush=True)
    assert work.snapshot()['all_submitted_completed']


def test_short_idle_flush_publishes_all_submitted_tail_once_without_waiting(tmp_path):
    work, clock, events = observer(tmp_path)
    work.submit(batch())
    assert json.loads(work.output.read_text())['submitted']['public_batches'] == 1
    clock.now = .1
    for _ in range(4):
        work.submit(batch())
    assert json.loads(work.output.read_text())['submitted']['public_batches'] == 1
    assert not events[0].ready
    row = work.poll(flush=True)
    assert row['submitted']['public_batches'] == 5
    assert row['completed']['public_batches'] == 0
    assert row['not_yet_confirmed']['public_batches'] == 5
    assert row['pending_fence']['totals']['public_batches'] == 1
    published = work.events_output.read_text()
    for _ in range(50):
        clock.now += .001
        assert work.poll(flush=True) is None
    assert work.events_output.read_text() == published
    # Completing the old fence and fencing the tail is observable immediately,
    # even though neither publication interval has expired.
    events[0].ready = True
    row = work.poll(flush=True)
    assert row['completed']['public_batches'] == 1
    assert row['pending_fence']['totals']['public_batches'] == 5
    assert len(events) == 2
    events[1].ready = True
    row = work.poll(flush=True)
    assert row['all_submitted_completed']
    assert row['completed']['public_batches'] == 5
    published = work.events_output.read_text()
    for _ in range(50):
        assert work.poll(flush=True) is None
    assert work.events_output.read_text() == published


def test_idle_flush_publishes_new_pending_fence_without_changing_counts(tmp_path):
    work, clock, events = observer(tmp_path)
    work.submit(batch())
    events[0].ready = True
    clock.now = .1
    work.poll()
    work.submit(batch())
    # A forced boundary snapshot can have submitted > completed but no pending
    # fence yet, because the publication interval has not elapsed.
    row = work.poll(force=True)
    assert row['pending_fence'] is None
    assert row['submitted']['public_batches'] == 2
    assert row['completed']['public_batches'] == 1
    row = work.poll(flush=True)
    assert row['submitted']['public_batches'] == 2
    assert row['completed']['public_batches'] == 1
    assert row['pending_fence']['sequence'] == 2
    assert work.poll(flush=True) is None


def _runner(tmp_path, work):
    pytest.importorskip('vllm')
    from vllm_kvmem.runner import make_probe_runner

    class Base:
        def prepare_attn(self, input_batch):
            return 'tables', 'slots'

        def execute_model(self, scheduler_output, **kwargs):
            if getattr(scheduler_output, 'raise_forward', False):
                raise RuntimeError('forward failed')
            if scheduler_output.total_num_scheduled_tokens:
                self.prepare_attn(scheduler_output.input_batch)
                self.execute_model_state = SimpleNamespace(input_batch=scheduler_output.input_batch)
            return 'native'

        def shutdown(self):
            self.native_shutdown = True

    runner = make_probe_runner(Base)()
    runner._kvmem_probe_output = tmp_path
    runner._kvmem_probe_done = True
    runner._kvmem_continuous = False
    runner._kvmem_bounded_target = False
    runner._kvmem_window_adapter = None
    runner._kvmem_public_work = work
    return runner


def test_runner_counts_each_intermediate_forward_without_sample_and_not_private(tmp_path):
    work, clock, events = observer(tmp_path)
    runner = _runner(tmp_path, work)
    (tmp_path/'armed').touch()
    for i in range(3):
        # Same attention preparation hook is used by private Query forwards.
        runner.prepare_attn(batch((333,), (333,), (0,)))
        public = batch((128,), (1024,), (128*i,))
        output = SimpleNamespace(total_num_scheduled_tokens=128, input_batch=public, finished_req_ids=set())
        assert runner.execute_model(output) == 'native'
    assert work.submitted['public_batches'] == 3
    assert work.submitted['public_prefill_tokens'] == 384
    assert work.submitted['public_non_prefill_tokens'] == 0
    assert len(events) == 1


def test_runner_excludes_unarmed_warmup_dummy_profile_and_zero_work(tmp_path):
    work, clock, events = observer(tmp_path)
    runner = _runner(tmp_path, work)
    public = batch()
    output = SimpleNamespace(total_num_scheduled_tokens=128, input_batch=public, finished_req_ids=set())
    runner.execute_model(output)  # Unarmed native warmup.
    (tmp_path/'armed').touch()
    runner.execute_model(output, dummy_run=True)
    runner.execute_model(output, is_profile=True)
    assert work.submitted['public_batches'] == 0
    assert not events
    runner.execute_model(output)
    assert work.submitted['public_batches'] == 1
    events[0].ready = True
    clock.now = 2.
    runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0, finished_req_ids=set()))
    assert work.submitted['public_batches'] == 1
    assert work.snapshot()['all_submitted_completed']
    assert json.loads(work.output.read_text())['completed']['public_batches'] == 1


def test_failed_forward_is_not_published_as_completed_work(tmp_path):
    work, clock, events = observer(tmp_path)
    runner = _runner(tmp_path, work)
    (tmp_path/'armed').touch()
    output = SimpleNamespace(total_num_scheduled_tokens=128, input_batch=batch(),
                             finished_req_ids=set(), raise_forward=True)
    with pytest.raises(RuntimeError, match='forward failed'):
        runner.execute_model(output)
    assert work.submitted['public_batches'] == 0
    assert not events


def test_shutdown_publishes_pending_status_without_waiting(tmp_path):
    work, clock, events = observer(tmp_path)
    runner = _runner(tmp_path, work)
    work.submit(batch())
    runner.shutdown()
    assert runner.native_shutdown
    row = json.loads(work.output.read_text())
    assert not row['all_submitted_completed']
    assert row['not_yet_confirmed']['public_batches'] == 1
