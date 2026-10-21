"""CPU lifecycle proof for detached flags, independent of live native scratch."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from threading import Event as ThreadEvent, get_ident
from types import SimpleNamespace
import weakref

import pytest
import torch

from vllm_kvmem.native_attention import (
    DetachedValidationResult,
    NativeAttentionStats,
)


_pointers = count(1024, 256)


class Flags:
    def __init__(self, values, trace, *, pointer=None, copy_error=False, read_error=False):
        self.values = list(values)
        self.trace = trace
        self.pointer = next(_pointers) if pointer is None else pointer
        self.copy_error, self.read_error = copy_error, read_error
        self.device = SimpleNamespace(type='cpu')
        self.dtype = torch.int32
        self.shape = (len(values),)
        self.pinned = self.contiguous = True

    def data_ptr(self):
        return self.pointer

    def is_pinned(self):
        return self.pinned

    def is_contiguous(self):
        return self.contiguous

    def copy_(self, source, *, non_blocking):
        assert non_blocking
        self.trace.append('copy')
        if self.copy_error:
            raise OSError('flags copy failed')
        self.values = list(source)

    def tolist(self):
        self.trace.append('read')
        if self.read_error:
            raise OSError('flags read failed')
        return list(self.values)

    def zero_(self):
        self.values = [0] * len(self.values)

    def fill_(self, value):
        self.values = [value] * len(self.values)


class Event:
    def __init__(self, trace, *, record_error=False, first_wait_error=False):
        self.trace = trace
        self.record_error, self.first_wait_error = record_error, first_wait_error
        self.recorded = False
        self.waits = 0

    def record(self):
        self.trace.append('record')
        if self.record_error:
            raise OSError('record failed')
        self.recorded = True

    def synchronize(self):
        assert self.recorded
        self.waits += 1
        self.trace.append('wait')
        if self.first_wait_error and self.waits == 1:
            raise OSError('wait failed')


def set_frame(bank, *, owners=1, values=(1, 1, 1)):
    bank.plans = {f'owner-{i}': {'rows': [0, 31]} for i in range(owners)}
    bank.slots = {rid: (0, 2, 16) for rid in bank.plans}
    bank.reduced = set(bank.names) if owners else set()
    bank.valid_flags = list(values)
    bank.validation_dirty = bool(owners)
    bank.frame_success = Counter({'reuse:prefill': owners * len(bank.names)}) if owners else Counter()
    bank.counts.update(bank.frame_success)


def fixture(*, owners=1, values=(1, 1, 1)):
    trace = []
    bank = NativeAttentionStats.__new__(NativeAttentionStats)
    bank.rows = None
    bank.mode, bank.disabled, bank.batch_reason = 'reuse', None, None
    bank.names = ('a', 'b', 'c')
    bank.host = {name: Flags((0, 0, 0), []) for name in ('valid_flags', 'ranges', 'rows')}
    bank.max_rows = 256
    bank.upload_event = None
    bank.counts = Counter({'reuse:prefill': 6, 'fallback:prefill:other': 2})
    bank.validation_failures = 0
    bank._detached_owner_thread = get_ident()
    bank.disarm()
    set_frame(bank, owners=owners, values=values)
    return bank, Flags((9, 9, 9), trace), Event(trace), trace


def test_detach_enqueues_complete_fence_without_wait_and_old_host_is_untouched():
    bank, host, event, trace = fixture()
    ticket = bank.detach_validation(host, event)
    assert trace == ['copy', 'record']
    assert bank.host['valid_flags'].values == [0, 0, 0]
    assert ticket.names == ('a', 'b', 'c')
    assert ticket.frame_success == (('reuse:prefill', 3),)
    assert ticket.recorded and not ticket.resolved and not bank.validation_dirty
    result = ticket.resolve()
    assert trace == ['copy', 'record', 'wait', 'read']
    assert result.valid and result.invalid_names == ()
    assert bank.apply_detached_validation(ticket, result)
    assert bank.counts['reuse:prefill'] == 9 and bank.validation_failures == 0


def test_empty_frame_still_fences_without_reading_flags():
    bank, host, event, trace = fixture(owners=0)
    ticket = bank.detach_validation(host, event)
    assert not ticket.dirty and trace == ['record']
    result = ticket.resolve()
    assert result.valid and trace == ['record', 'wait']
    assert bank.apply_detached_validation(ticket, result)


def test_late_invalid_frame_cannot_change_rearmed_frame_or_its_flags():
    bank, host, event, _ = fixture(owners=2, values=(1, 0, 1))
    old = bank.detach_validation(host, event)
    # Genuine arm() may proceed before the old flags are inspected.
    bank.arm({})
    set_frame(bank, values=(1, 1, 1))
    live_plans, live_success = bank.plans, bank.frame_success.copy()
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(old.resolve).result()
    assert result.invalid_names == ('b',)
    assert bank.plans is live_plans and bank.frame_success == live_success
    assert bank.validation_dirty and not bank.frame_invalid and not bank.invalid
    assert not bank.apply_detached_validation(old, result)
    assert bank.counts == {'reuse:prefill': 9, 'fallback:prefill:other': 2,
                           'fallback:prefill:invalid_native_mass': 6}
    assert bank.validation_failures == 1
    assert bank.plans is live_plans and bank.frame_success == live_success
    assert bank.validation_dirty and not bank.frame_invalid and not bank.invalid
    newer = bank.detach_validation(Flags((0, 0, 0), []), Event([]))
    assert bank.apply_detached_validation(newer, newer.resolve())
    assert bank.validation_failures == 1


def test_multiple_frames_resolve_out_of_order_and_each_apply_once():
    bank, host, event, _ = fixture(values=(0, 1, 1))
    first = bank.detach_validation(host, event)
    bank.arm({})
    set_frame(bank, owners=2, values=(1, 1, 0))
    second = bank.detach_validation(Flags((0, 0, 0), []), Event([]))
    bank.arm({})
    assert not bank.apply_detached_validation(second, second.resolve())
    assert not bank.apply_detached_validation(first, first.resolve())
    assert bank.validation_failures == 2
    assert bank.counts['reuse:prefill'] == 6
    assert bank.counts['fallback:prefill:invalid_native_mass'] == 9
    assert not bank._detached_pending
    with pytest.raises(RuntimeError):
        bank.apply_detached_validation(first, first.result)
    assert bank.validation_failures == 2


@pytest.mark.parametrize('operation', ['detach', 'enqueue', 'validate', 'score', 'planning'])
def test_detached_current_frame_rejects_duplicate_consumption_until_rearm(operation):
    bank, host, event, trace = fixture()
    ticket = bank.detach_validation(host, event)
    before = list(trace)
    with pytest.raises(RuntimeError):
        if operation == 'detach':
            bank.detach_validation(Flags((0, 0, 0), []), Event([]))
        elif operation == 'enqueue':
            bank.enqueue_validation()
        elif operation == 'validate':
            bank.validate()
        elif operation == 'score':
            bank.score('a', 'owner-0', True)
        else:
            bank.resolve({})
    assert trace == before and not ticket.resolved
    bank.disarm()


@pytest.mark.parametrize('fault', ['copy', 'record'])
def test_enqueue_error_keeps_bank_and_original_lease_poisoned(fault):
    bank, host, event, trace = fixture()
    host.copy_error = fault == 'copy'
    event.record_error = fault == 'record'
    with pytest.raises(OSError):
        bank.detach_validation(host, event)
    ticket = bank._detached_enqueue_failure
    assert ticket.host_flags is host and ticket.event is event and not ticket.recorded
    host.copy_error = event.record_error = False
    before = list(trace)
    for operation in (bank.disarm, lambda: bank.arm({}), bank.validate, ticket.resolve):
        with pytest.raises(RuntimeError):
            operation()
    assert trace == before and not ticket.resolved
    assert bank.counts['reuse:prefill'] == 9 and bank.validation_failures == 0


@pytest.mark.parametrize('fault', ['wait', 'read', 'shape'])
def test_resolution_error_is_permanent_and_cannot_be_applied(fault):
    bank, host, event, trace = fixture()
    event.first_wait_error = fault == 'wait'
    host.read_error = fault == 'read'
    ticket = bank.detach_validation(host, event)
    if fault == 'shape':
        host.values.pop()
    with pytest.raises((OSError, ValueError)):
        ticket.resolve()
    host.read_error = False
    before = list(trace)
    with pytest.raises(RuntimeError):
        ticket.resolve()
    with pytest.raises(RuntimeError):
        bank.apply_detached_validation(ticket, DetachedValidationResult(()))
    assert trace == before and ticket.error is not None and not ticket.resolved
    assert bank._detached_pending[ticket.sequence] is ticket
    assert bank.counts['reuse:prefill'] == 9 and bank.validation_failures == 0


def test_foreign_forged_and_duplicate_results_cannot_change_counts():
    bank, host, event, trace = fixture(values=(0, 0, 0))
    ticket = bank.detach_validation(host, event)
    result = ticket.resolve()
    before = list(trace)
    with pytest.raises(RuntimeError):
        ticket.resolve()
    with pytest.raises(RuntimeError):
        bank.apply_detached_validation(ticket, DetachedValidationResult(('a', 'b', 'c')))
    other, other_host, other_event, _ = fixture()
    other.detach_validation(other_host, other_event)
    with pytest.raises(RuntimeError):
        other.apply_detached_validation(ticket, result)
    assert bank.counts['reuse:prefill'] == 9 and trace == before
    assert not bank.apply_detached_validation(ticket, result)
    with pytest.raises(RuntimeError):
        bank.apply_detached_validation(ticket, result)


def test_resolver_does_not_retain_or_require_native_bank():
    bank, host, event, _ = fixture()
    ticket = bank.detach_validation(host, event)
    reference = weakref.ref(bank)
    del bank
    assert reference() is None
    assert ticket.resolve().valid


@pytest.mark.parametrize('field,value', [
    ('pinned', False), ('contiguous', False), ('dtype', torch.float32),
    ('shape', (1, 3)), ('device', SimpleNamespace(type='cuda')),
])
def test_unqualified_flags_fail_before_copy(field, value):
    bank, host, event, trace = fixture()
    setattr(host, field, value)
    with pytest.raises(ValueError):
        bank.detach_validation(host, event)
    assert trace == [] and not getattr(bank, '_detached_pending', {})
    bank.disarm()


@pytest.mark.parametrize('alias', ['shared', 'overlap_shared', 'pending', 'overlap_pending', 'event'])
def test_host_and_event_exclusivity_is_checked_before_enqueue(alias):
    bank, host, event, trace = fixture()
    if alias in ('shared', 'overlap_shared'):
        host.pointer = bank.host['valid_flags'].data_ptr() + (4 if alias == 'overlap_shared' else 0)
    else:
        old = bank.detach_validation(host, event)
        bank.arm({})
        set_frame(bank)
        host = Flags((0, 0, 0), trace, pointer=(old.host_flags.data_ptr() +
            (4 if alias == 'overlap_pending' else 0)) if alias != 'event' else None)
        event = old.event if alias == 'event' else Event(trace)
    before = list(trace)
    with pytest.raises(ValueError):
        bank.detach_validation(host, event)
    assert trace == before


@pytest.mark.parametrize('fault', ['mode', 'layer', 'owner', 'counter', 'reason', 'known_invalid', 'legacy_ticket'])
def test_mixed_or_incomplete_native_frames_keep_synchronous_route(fault):
    bank, host, event, trace = fixture()
    if fault == 'mode': bank.mode = 'verify'
    elif fault == 'layer': bank.reduced.remove('b')
    elif fault == 'owner': bank.slots.clear()
    elif fault == 'counter': bank.frame_success['reuse:prefill'] -= 1
    elif fault == 'reason': bank.request_reasons['owner-0'] = 'missing'
    elif fault == 'known_invalid': bank.frame_invalid = True
    else: bank._validation_ticket = object()
    assert not bank.can_detach_validation()
    with pytest.raises(RuntimeError):
        bank.detach_validation(host, event)
    assert trace == []


def test_only_submission_and_counter_application_are_engine_thread_owned():
    bank, host, event, _ = fixture()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError):
            executor.submit(bank.detach_validation, host, event).result()
    assert not event.recorded
    ticket = bank.detach_validation(host, event)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(ticket.resolve).result()
        with pytest.raises(RuntimeError):
            executor.submit(bank.apply_detached_validation, ticket, result).result()
    assert not ticket.applied and bank.apply_detached_validation(ticket, result)
    bank.arm({})
    set_frame(bank)
    other_trace = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError):
            executor.submit(bank.detach_validation, Flags((0, 0, 0), other_trace), Event(other_trace)).result()
    assert other_trace == []


def test_applied_buffers_can_be_reused_without_reviving_old_ticket():
    bank, host, event, trace = fixture(values=(0, 1, 1))
    old = bank.detach_validation(host, event)
    old_result = old.resolve()
    bank.apply_detached_validation(old, old_result)
    bank.arm({})
    set_frame(bank)
    new = bank.detach_validation(host, event)
    assert new.sequence == old.sequence + 1
    assert new.resolve().valid and old_result.invalid_names == ('a',)
    before = list(trace)
    with pytest.raises(RuntimeError): old.resolve()
    with pytest.raises(RuntimeError): bank.apply_detached_validation(old, old_result)
    assert trace == before and bank.apply_detached_validation(new, new.result)


def test_resolved_buffers_reuse_before_worker_ready_and_counter_application():
    bank, host, event, _ = fixture(values=(1, 0, 1))
    old = bank.detach_validation(host, event)
    bank.arm({})
    set_frame(bank)
    released, may_finish = ThreadEvent(), ThreadEvent()
    result_holder = []

    def finish_old_frame():
        result_holder.append(old.resolve())
        # A bounded publication pool may release its slot here, before the
        # frame's ready flag becomes visible to the next main-thread scan.
        released.set()
        assert may_finish.wait(5)

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker = executor.submit(finish_old_frame)
        try:
            assert released.wait(5)
            assert old.resolved and not old.applied and not worker.done()
            newer = bank.detach_validation(host, event)
            assert host.values == [1, 1, 1]
            assert result_holder[0].invalid_names == ('b',)
            assert not bank.apply_detached_validation(old, result_holder[0])
            assert bank.frame_success == {'reuse:prefill': 3}
            assert not bank.frame_invalid and not bank.invalid
            assert bank.apply_detached_validation(newer, newer.resolve())
        finally:
            may_finish.set()
        worker.result()
    assert bank.validation_failures == 1
    assert bank.counts['reuse:prefill'] == 9
    assert not bank._detached_pending
