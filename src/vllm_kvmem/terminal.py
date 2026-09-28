"""Scheduler-owned terminal prefix, before native request fields are discarded.

Sampling acceptance precedes the scheduler's EOS/length truncation. The reusable
input prefix excludes the last emitted token (which is normally not evaluated).
In-flight rows of an aborted request are never acknowledged as durable history.
This describes KV capture, not a reusable recurrent-state checkpoint.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalPrefix:
    history_end: int
    computed_tokens: int
    output_tokens: int
    status: str

    @classmethod
    def from_request(cls, request):
        computed = request.num_computed_tokens
        inflight = request.num_in_flight_tokens
        total = request.num_tokens
        outputs = request.num_output_tokens
        if any(type(x) is not int or x < 0 for x in (computed, inflight, total, outputs)):
            raise ValueError('Invalid terminal request counters')
        if inflight > computed or outputs > total or not request.is_finished():
            raise ValueError('Terminal prefix requires consistent finished request state')
        end = min(computed - inflight, total - bool(outputs))
        return cls(end, computed, outputs, request.status.name)
