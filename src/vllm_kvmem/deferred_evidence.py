"""Opt-in: write large pressure evidence rows after the plan/ACK, same bytes and order.

KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE=1 lets LayerwiseQuery.defer() queue a
record whose timestamps are taken at the original call. A background writer
serializes it (per layer, yielding the GIL between layers) after
KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE_DELAY_MS, normally while the next
forward runs. Every row keeps its original position: emit() first writes all
queued rows synchronously, and the writer only appends the oldest queued row.
Serialized lines are byte-identical to json.dumps(record, allow_nan=False).
"""
import atexit
from collections import deque
import json
import os
import threading
import time

_SENTINEL = '\x00kvmem-deferred-layers\x00'
_TOKEN = json.dumps(_SENTINEL)


def delay_seconds():
    value = os.getenv('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE_DELAY_MS', '100')
    try:
        delay = float(value)
    except ValueError:
        delay = -1.
    if not 0 <= delay <= 2000:
        raise ValueError('KVMEM_LAYERWISE_PRESSURE_DEFERRED_EVIDENCE_DELAY_MS must be in [0, 2000]')
    return delay / 1000


def serialize(record, *, yield_gil=False):
    """Exactly json.dumps(record, allow_nan=False) + newline, optionally per layer."""
    layers = record.get('layers')
    if not yield_gil or type(layers) is not list or len(layers) < 2:
        return json.dumps(record, allow_nan=False) + '\n'
    outer = dict(record)
    outer['layers'] = _SENTINEL
    head = json.dumps(outer, allow_nan=False)
    if head.count(_TOKEN) != 1:
        return json.dumps(record, allow_nan=False) + '\n'
    parts = []
    for layer in layers:
        parts.append(json.dumps(layer, allow_nan=False))
        time.sleep(0)  # Let the engine thread take the GIL between layers.
    return head.replace(_TOKEN, '[' + ', '.join(parts) + ']') + '\n'


class _Entry:
    __slots__ = ('record', 'chunked', 'line', 'resolve')

    def __init__(self, record, chunked, resolve=None):
        self.record, self.chunked, self.line, self.resolve = record, chunked, None, resolve

    def resolved(self, yield_gil=False):
        # A lazy row's trailing fields are computed here, on whichever thread
        # writes it. The queued record itself is never mutated.
        if self.resolve is None:
            return self.record
        return {**self.record, **self.resolve(yield_gil)}

    def text(self):
        if self.line is None:
            self.line = serialize(self.resolved())
        return self.line


class DeferredEvidence:
    def __init__(self, query):
        self.query = query
        self.delay = delay_seconds()
        self.queue = deque()
        self.error = None
        self.condition = threading.Condition(query._emit_lock)
        self.written = self.deferred = self.synchronous = 0
        self.thread = threading.Thread(target=self._run, name='kvmem-evidence', daemon=True)
        self.thread.start()
        atexit.register(self._flush_at_exit)

    def defer(self, record, buffered, resolve=None):
        """Caller holds query._emit_lock. buffered: earlier unwritten plain rows."""
        for pending in buffered:
            self.queue.append(_Entry(pending, False))
        # Shallow copies: later key replacement on the live evidence cannot
        # change what is written; list contents are never mutated after commit.
        record = dict(record)
        if type(record.get('layers')) is list:
            record['layers'] = [dict(layer) if type(layer) is dict else layer for layer in record['layers']]
        self.queue.append(_Entry(record, True, resolve))
        self.deferred += 1
        self.condition.notify_all()

    def take_lines(self):
        """Caller holds query._emit_lock: serialize every queued row now, in order."""
        if self.error is not None:
            raise RuntimeError('Deferred pressure evidence writer failed') from self.error
        lines = [entry.text() for entry in self.queue]
        self.synchronous += sum(entry.chunked for entry in self.queue)
        self.queue.clear()
        return lines

    def flush(self):
        with self.query._emit_lock:
            lines = self.take_lines()
            if lines:
                self.query._append_lines(lines)
            self.written += len(lines)

    def _flush_at_exit(self):
        try:
            self.flush()
        except BaseException as error:  # Already surfaced to the engine if it emitted again.
            import sys
            print(f'kvmem: deferred pressure evidence was not flushed: {error!r}', file=sys.stderr)

    def _run(self):
        lock = self.query._emit_lock
        while True:
            with lock:
                while not self.queue:
                    self.condition.wait()
                entry = self.queue[0]
            if self.delay:
                time.sleep(self.delay)
            try:
                line = serialize(entry.resolved(entry.chunked), yield_gil=entry.chunked)
                with lock:
                    if not self.queue or self.queue[0] is not entry:
                        continue  # emit() already wrote it, in order.
                    entry.line = line
                    self.query._append_lines([line])
                    self.queue.popleft()
                    self.written += 1
            except BaseException as error:  # Surface at the next emit/flush.
                with lock:
                    self.error = error
                return
