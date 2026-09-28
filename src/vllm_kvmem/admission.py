"""Token admission for aligned semantic prefill and target-only decode refresh."""
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPolicy:
    block_tokens: int
    select_budget: int
    gen_budget: int

    @property
    def capacity(self):
        return self.select_budget + self.gen_budget

    def admit(self, history, desired, prompt_tokens, offset):
        local = history - offset
        if min(history, desired, prompt_tokens, local) < 0 or local > self.capacity:
            raise ValueError("Invalid native history or working-set extent")
        tokens = min(desired, self.capacity - local)
        if history < prompt_tokens:
            tokens = min(tokens, prompt_tokens - history)
            # Install the all-history bootstrap exactly at K before semantic
            # query-conditioned chunks. Non-final chunks preserve checkpoints.
            if history < self.select_budget:
                tokens = min(tokens, self.select_budget - history)
            if history + tokens < prompt_tokens:
                tokens = max(0, (history + tokens) // self.block_tokens * self.block_tokens - history)
        if desired and not tokens and local == self.capacity:
            raise RuntimeError("Working set filled without a preceding query refresh")
        return tokens

    def refresh(self, history, prompt_tokens, offset):
        if history < self.select_budget or history % self.block_tokens:
            return False
        if history < prompt_tokens:
            return True
        # Leave one block for the provisional target query. Speculative batches
        # require their own accepted-boundary checkpoint policy, not this rule.
        return history - offset >= self.capacity - self.block_tokens
