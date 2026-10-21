"""Immutable logical prefill frames independent of CPU publication progress.

Only full, scheduler-confirmed ordinary prefill chunks may extend a frame.
The capture runtime bounds that lookahead and fences reselection/restoration;
these frames never commit scores, parser state or an accepted history boundary.
"""
from dataclasses import dataclass

from .prefill_attention import PrefillHistory


@dataclass(frozen=True)
class SamplingSnapshot:
    options: object
    end: int

    def rows(self, begin, count, local_begin):
        # Use the policy itself, including interval and near-capacity forcing.
        # No mutable values/samples/chunks from PrefillHistory are consulted.
        return PrefillHistory.rows(self, begin, count, local_begin)


@dataclass(frozen=True)
class PrefillPlanFrame:
    authority: object
    state: object
    identity: tuple
    history_end: int
    local_end: int
    sampled_end: int
    records: tuple

    def matches(self, authority, identity, history_begin, local_begin):
        return (self.authority is authority and self.identity == identity and
                self.history_end == history_begin and self.local_end == local_begin)

    def extend(self, history_begin, local_begin, count, block_tokens):
        if (history_begin != self.history_end or local_begin != self.local_end or
                count <= 0):
            raise ValueError('Noncontiguous pipelined prefill frame')
        from .action_attention import window_segments
        rows = SamplingSnapshot(self.state.options, self.sampled_end).rows(
            history_begin, count, local_begin)
        starts, counts, history = window_segments(
            self.records, history_begin, local_begin, count, block_tokens)
        # A committed append grows the existing partial tail in place. Keep
        # that exact record boundary for the next chunk; scoring this chunk
        # still sees the original historical-tail/new-suffix split above.
        records = list(self.records)
        cursor, logical, remaining = local_begin, history_begin, count
        while remaining:
            n = min(block_tokens - logical % block_tokens, remaining)
            if (records and records[-1][2] + records[-1][3] == logical and
                    records[-1][4] + records[-1][3] == cursor and
                    records[-1][2] // block_tokens == logical // block_tokens):
                slot, generation, begin, size, position = records[-1]
                records[-1] = (slot, generation, begin, size + n, position)
            else:
                records.append((-1, -1, logical, n, cursor))
            cursor += n
            logical += n
            remaining -= n
        next_frame = PrefillPlanFrame(self.authority, self.state, self.identity,
            history_begin + count, local_begin + count,
            history_begin + count if rows else self.sampled_end, tuple(records))
        return rows, starts, counts, history, next_frame


def freeze_prefill_frame(authority, state, identity, history_begin, local_begin, records):
    """Snapshot only after this owner's prior publication has drained."""
    frozen = []
    cursor = 0
    for slot, generation, begin, count, position in records:
        if position >= local_begin:
            break
        count = min(count, local_begin - position)
        if position != cursor or count <= 0:
            raise ValueError('Pipelined prefill snapshot has a stale dense mapping')
        frozen.append((int(slot), int(generation), int(begin), int(count), int(position)))
        cursor += count
    if cursor != local_begin:
        raise ValueError('Pipelined prefill snapshot has an unpublished prefix')
    return PrefillPlanFrame(authority, state, identity, history_begin, local_begin,
                           state.prefill.end, tuple(frozen))
