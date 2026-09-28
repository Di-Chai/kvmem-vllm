"""Accepted-prefix alignment for synchronous speculative target verification.

A verifier can reject any suffix. Clip its scheduled extent before the refresh
boundary so accepted history cannot jump over the checkpoint, without publishing
rejected candidates or padding the history to an artificial boundary.
"""
from dataclasses import dataclass

from .admission import QueryPolicy


@dataclass(frozen=True)
class SpeculativeQueryPolicy(QueryPolicy):
    max_query_tokens: int

    def __post_init__(self):
        if not 1 <= self.max_query_tokens <= self.block_tokens:
            raise ValueError("Speculative query must fit the reserved refresh headroom")

    def admit(self, history, desired, prompt_tokens, offset):
        tokens = super().admit(history, desired, prompt_tokens, offset)
        if history < prompt_tokens or not desired:
            return tokens
        if desired > self.max_query_tokens or offset % self.block_tokens:
            raise ValueError("Unqualified speculative query extent or checkpoint mapping")
        local = history - offset
        boundary = self.capacity - self.block_tokens
        if local > boundary:
            raise RuntimeError("Accepted history crossed the speculative refresh checkpoint")
        if local < boundary:
            tokens = min(tokens, boundary - local)
        # At the boundary, the current query is replayed after selection; the
        # old working set has one complete block of provisional headroom.
        return tokens

    @staticmethod
    def authoritative_probe_tokens(history, prompt_tokens, scheduled):
        if min(history, prompt_tokens) < 0 or scheduled <= 0:
            raise ValueError("Invalid query probe extent")
        # Known input tokens may all condition selection. During decode only
        # the first (already sampled) input is authoritative. Future unverified
        # draft tokens must not choose the target's historical working set.
        return min(scheduled, max(1, prompt_tokens - history))
