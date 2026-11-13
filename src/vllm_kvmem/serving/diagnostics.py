"""Bounded serving diagnostics with cumulative counts for long-lived workers."""
from collections import deque


SERVING_COMPLETION_LIMIT = 256


class CompletionHistory(deque):
    """Keep recent reports while counting completions across evicted reports.

    Qualification runs use an unlimited history so existing evidence remains
    complete. Serving runs use a fixed bound independent of total requests.
    All completion producers append on the engine thread.
    """

    def __init__(self, *, serving=False):
        super().__init__(maxlen=SERVING_COMPLETION_LIMIT if serving else None)
        self.total = 0

    def append(self, report):
        super().append(report)
        self.total += 1


def completion_counts(history):
    retained = len(history)
    total = getattr(history, 'total', retained)
    return {
        'completed_requests': total,
        'completed_requests_retained': retained,
        'completed_requests_dropped': total - retained,
        'completion_history_limit': getattr(history, 'maxlen', None),
    }
