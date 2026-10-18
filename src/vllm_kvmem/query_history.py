"""Per-authority logical-input relevance, using the unmodified qw3 policy.

Only a completed public prefill commits an observation. Guided scratch tokens
never enter the input identity. Replays replace the same round's preview; a
regeneration starts from its strict ancestor, so neither rewards evidence twice.
"""
from array import array

from . import _native


class QueryRelevance:
    def __init__(self, decay=0.5):
        self.policy = _native.QueryHistory()
        self.decay = decay
        self.tokens = array('I')
        self.pending_tokens = None
        self.request_id = None
        self.query = None
        self.observation = None

    def begin(self, request_id, tokens):
        if self.request_id == request_id:
            return
        self.policy.abort()
        current = array('I', tokens)
        common = 0
        for a, b in zip(self.tokens, current):
            if a != b:
                break
            common += 1
        self.policy.begin(len(current), common, self.decay)
        self.pending_tokens = current
        self.request_id = request_id
        self.query = self.observation = None

    def preview(self, raw):
        self.observation = dict(raw)
        return self.policy.preview(raw)

    def commit(self):
        if self.policy.commit():
            self.tokens = self.pending_tokens
            self.pending_tokens = None
            self.query = None
            return True
        return False

    def abort(self):
        self.policy.abort()
        self.pending_tokens = self.query = self.observation = None
        self.request_id = None


def begin_round(runner, authority, request_id):
    sessions = getattr(runner, '_kvmem_sessions', None)
    binding = sessions.bindings.get(request_id) if sessions is not None else None
    if binding is None:
        raise ValueError('Query relevance history requires an owned prompt session')
    authority.query_history.begin(request_id, binding.tokens)
    return binding
