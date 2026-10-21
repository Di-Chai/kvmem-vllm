"""Host lifecycle proof for the same-forward complete-score CUDA fence.

The fake event can recover after an injected error; production must still keep
that failed ticket quarantined rather than retrying it into reusable state.
"""
from collections import Counter

import pytest

from vllm_kvmem.native_attention import NativeAttentionStats


class Flags:
    def __init__(self, values, events, *, copy_error=False, read_error=False):
        self.values = list(values)
        self.events = events
        self.copy_error = copy_error
        self.read_error = read_error

    def copy_(self, source, *, non_blocking):
        assert non_blocking
        self.events.append('flags-copy')
        if self.copy_error:
            raise OSError('injected flags copy failure')
        self.values = list(source)
        return self

    def tolist(self):
        self.events.append('flags-read')
        if self.read_error:
            raise OSError('injected flags read failure')
        return list(self.values)


class Event:
    def __init__(self, events, *, record_error=False, first_wait_error=False):
        self.events = events
        self.record_error = record_error
        self.first_wait_error = first_wait_error
        self.waits = 0
        self.recorded = False

    def record(self):
        self.events.append('event-record')
        if self.record_error:
            raise OSError('injected event record failure')
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.waits += 1
        self.events.append('event-wait')
        if self.first_wait_error and self.waits == 1:
            raise OSError('injected first event wait failure')


def bank_fixture(monkeypatch, *, dirty=True, values=(1, 1, 1),
                 copy_error=False, read_error=False,
                 record_error=False, first_wait_error=False):
    events = []
    created = []

    def event_factory():
        events.append('event-create')
        event = Event(events, record_error=record_error, first_wait_error=first_wait_error)
        created.append(event)
        return event

    monkeypatch.setattr('vllm_kvmem.native_attention.torch.cuda.Event', event_factory)
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.rows = None
    bank.names = ('layer-a', 'layer-b', 'layer-c')
    bank.valid_flags = list(values)
    bank.host = {'valid_flags': Flags((0, 0, 0), events,
        copy_error=copy_error, read_error=read_error)}
    bank.validation_dirty = dirty
    bank.frame_invalid = False
    bank.invalid = set()
    bank.validation_failures = 0
    bank.counts = Counter({'reuse:prefill': 9, 'fallback:prefill:other': 2})
    bank.frame_success = Counter({'reuse:prefill': 3})
    return bank, events, created


def test_enqueue_is_nonblocking_and_resolve_reads_only_after_complete_fence(monkeypatch):
    bank, events, created = bank_fixture(monkeypatch)
    ticket = bank.enqueue_validation()
    assert events == ['flags-copy', 'event-create', 'event-record']
    assert bank._validation_ticket is ticket and not ticket.resolved
    assert bank.resolve_validation(ticket)
    assert events[-2:] == ['event-wait', 'flags-read']
    assert created[0].waits == 1 and ticket.resolved
    assert bank._validation_ticket is None and not bank.validation_dirty
    assert bank.counts['reuse:prefill'] == 9 and bank.validation_failures == 0


@pytest.mark.parametrize('invalid_before', [False, True])
def test_clean_ticket_still_fences_all_scores_without_reading_stale_flags(monkeypatch, invalid_before):
    bank, events, _ = bank_fixture(monkeypatch, dirty=False)
    bank.frame_invalid = invalid_before
    ticket = bank.enqueue_validation()
    assert not ticket.dirty
    assert events == ['event-create', 'event-record']
    assert bank.resolve_validation(ticket) is (not invalid_before)
    assert events == ['event-create', 'event-record', 'event-wait']
    assert not bank.invalid and bank.validation_failures == 0


def test_missing_layer_retracts_only_this_frames_success_once(monkeypatch):
    bank, events, _ = bank_fixture(monkeypatch, values=(1, 0, 1))
    ticket = bank.enqueue_validation()
    assert not bank.resolve_validation(ticket)
    assert bank.invalid == {'layer-b'} and bank.frame_invalid
    assert bank.counts == {'reuse:prefill': 6, 'fallback:prefill:other': 2}
    assert bank.frame_success == {} and bank.validation_failures == 1
    # The reference fallback has no new dirty flags but still gets a full fence.
    fallback = bank.enqueue_validation()
    assert not fallback.dirty and not bank.resolve_validation(fallback)
    assert events.count('flags-read') == 1 and bank.validation_failures == 1
    assert bank.counts['reuse:prefill'] == 6


@pytest.mark.parametrize('operation', ['enqueue', 'validate', 'disarm', 'arm', 'score', 'planning'])
def test_one_live_ticket_rejects_overwrite_and_synchronous_shortcuts(monkeypatch, operation):
    bank, events, _ = bank_fixture(monkeypatch)
    ticket = bank.enqueue_validation()
    before = list(events)
    with pytest.raises(RuntimeError):
        if operation == 'enqueue':
            bank.enqueue_validation()
        elif operation == 'validate':
            bank.validate()
        elif operation == 'disarm':
            bank.disarm()
        elif operation == 'arm':
            bank.arm({})
        elif operation == 'score':
            bank.score('layer-a', 'owner', True, defer_validation=True)
        else:
            bank.resolve({})
    assert bank._validation_ticket is ticket and events == before
    assert bank.resolve_validation(ticket)


def test_stale_foreign_and_duplicate_resolve_never_wait_or_read(monkeypatch):
    bank, events, _ = bank_fixture(monkeypatch)
    ticket = bank.enqueue_validation()
    before = list(events)
    with pytest.raises(RuntimeError):
        bank.resolve_validation(object())
    assert events == before and bank._validation_ticket is ticket
    assert bank.resolve_validation(ticket)
    before = list(events)
    with pytest.raises(RuntimeError):
        bank.resolve_validation(ticket)
    assert events == before


@pytest.mark.parametrize('dirty', [False, True])
def test_original_validate_preserves_clean_no_fence_and_dirty_synchronous_behavior(monkeypatch, dirty):
    bank, events, _ = bank_fixture(monkeypatch, dirty=dirty)
    assert bank.validate()
    assert events == (['flags-copy', 'event-create', 'event-record', 'event-wait', 'flags-read']
                      if dirty else [])
    assert getattr(bank, '_validation_ticket', None) is None
    bank.disarm()
    assert not bank.frame_invalid and not bank.validation_dirty


@pytest.mark.parametrize('failure', ['copy', 'record', 'wait', 'read'])
def test_any_fence_failure_remains_quarantined_even_if_retry_could_succeed(monkeypatch, failure):
    bank, events, created = bank_fixture(monkeypatch,
        copy_error=failure == 'copy', record_error=failure == 'record',
        first_wait_error=failure == 'wait', read_error=failure == 'read')
    with pytest.raises(OSError):
        ticket = bank.enqueue_validation()
        bank.resolve_validation(ticket)
    ticket = bank._validation_ticket
    assert ticket is not None and not ticket.resolved
    before = list(events)
    # Make every fake backend operation succeed from here on. A failed CUDA
    # fence is still fatal: neither a cleanup retry nor rearm may recycle it.
    bank.host['valid_flags'].copy_error = False
    bank.host['valid_flags'].read_error = False
    for event in created:
        event.record_error = False
    with pytest.raises(RuntimeError):
        bank.resolve_validation(ticket)
    with pytest.raises(RuntimeError):
        bank.disarm()
    assert events == before
    assert bank._validation_ticket is ticket and not ticket.resolved
    assert bank.counts['reuse:prefill'] == 9
