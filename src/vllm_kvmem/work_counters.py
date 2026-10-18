"""Unsampled public-forward work, with nonblocking completion brackets.

These counters describe executed target-model inputs, not accepted output tokens.
The caller submits only the fresh ExecuteModelState of a real public forward.
Private Query forwards and restored KV never pass through this hook. CUDA fences
are recorded at most once per publication interval (or an idle flush), queried
without synchronization, and carry cumulative counters so slow fences lose no
work. Completion observation times are bounds, not CUDA wall-clock timestamps.
"""
from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time
import uuid


def _bucket(tokens):
    return 1 << max(10, max(1, int(tokens)).bit_length() - 1)


def classify_public_batch(batch, offsets=None):
    """Use CPU metadata only; never read a CUDA tensor or padded request row."""
    count = int(batch.num_reqs)
    if count <= 0 or len(batch.req_ids) != count:
        raise ValueError('Public work requires a nonempty unsharded real batch')
    fields = (batch.num_scheduled_tokens, batch.prefill_len_np,
              batch.num_computed_prefill_tokens_np, batch.num_computed_tokens_np)
    if any(len(values) < count for values in fields):
        raise ValueError('Incomplete public batch metadata')
    offsets = offsets or {}
    prefills = decodes = prefill_tokens = 0
    prefill_contexts, decode_contexts = [], []
    for i, request_id in enumerate(batch.req_ids):
        scheduled, length, computed, history = (int(values[i]) for values in fields)
        if scheduled <= 0 or min(length, computed, history) < 0 or computed > length:
            raise ValueError('Invalid public batch token extents')
        remaining = length - computed
        # A final prefill step can also verify MTP lookahead positions. Those
        # positions are work, but must not inflate the prompt-token counter.
        prefill_tokens += min(scheduled, remaining)
        context = history + scheduled - int(offsets.get(request_id, 0))
        if context <= 0:
            raise ValueError('Invalid dense public context extent')
        if remaining:
            prefills += 1
            prefill_contexts.append(context)
        else:
            decodes += 1
            decode_contexts.append(context)
    tokens = int(batch.num_tokens)  # Excludes graph padding; can compact drafts.
    if not prefill_tokens <= tokens <= sum(int(n) for n in batch.num_scheduled_tokens[:count]):
        raise ValueError('Public token work is inconsistent with the executed batch')
    return dict(public_batches=1, public_prefill_tokens=prefill_tokens,
                public_non_prefill_tokens=tokens-prefill_tokens,
                prefill_request_batches=prefills, decode_request_batches=decodes,
                prefill_context_bucket=_bucket(max(prefill_contexts)) if prefills else None,
                decode_context_bucket=_bucket(max(decode_contexts)) if decodes else None)


_COUNTERS = ('public_batches', 'public_prefill_tokens', 'public_non_prefill_tokens',
             'prefill_request_batches', 'decode_request_batches')


class PublicWorkCounters:
    def __init__(self, output, *, interval=1., event_factory=None,
                 clock=time.monotonic, wall_clock=time.time):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError('Public work publication interval must be positive')
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=False)
        self.output = Path(output)
        self.events_output = self.output.with_suffix('.jsonl')
        self.interval, self.event_factory = interval, event_factory
        self.clock, self.wall_clock = clock, wall_clock
        self.counter_id = uuid.uuid4().hex
        self.created_at, self.created_wall_time = clock(), wall_clock()
        self.submitted = {key: 0 for key in _COUNTERS}
        self.histogram = Counter()
        self.completed = self._totals()
        self.pending = self.completed_fence = self.last_batch = None
        self.last_fence_at = self.last_publish_at = -math.inf
        self.snapshot_sequence = self.fence_sequence = 0
        self.last_published_progress = None

    def _totals(self):
        return dict(self.submitted, batch_composition=[dict(
            prefill_requests=key[0], decode_requests=key[1],
            prefill_context_bucket=key[2],
            decode_context_bucket=key[3], batches=value)
            for key, value in sorted(self.histogram.items(), key=lambda pair: repr(pair[0]))])

    def submit(self, batch, offsets=None):
        row = classify_public_batch(batch, offsets)
        for key in _COUNTERS:
            self.submitted[key] += row[key]
        self.histogram[(row['prefill_request_batches'], row['decode_request_batches'],
                        row['prefill_context_bucket'],
                        row['decode_context_bucket'])] += 1
        self.last_batch = dict(row, submitted_at=self.clock())
        self.poll()

    def poll(self, *, force=False, flush=False):
        """Publish low-frequency cumulative snapshots; never wait on a fence.

        An idle scheduler call may request ``flush`` to fence the final partial
        interval. If another fence is pending, its later poll will carry the
        entire accumulated tail. A stale idle snapshot is not a zero-work proof;
        consumers retain submitted/completed brackets until the next poll.
        """
        if self.pending is not None:
            query_started = self.clock()
            if self.pending['event'].query():
                pending = self.pending
                self.completed = pending['totals']
                self.completed_fence = dict(sequence=pending['sequence'],
                    recorded_started_at=pending['recorded_started_at'],
                    recorded_at=pending['recorded_at'],
                    completion_lower_bound=pending['last_not_ready_at'],
                    completion_upper_bound=self.clock())
                self.pending = None
            else:
                # query() was false at some point after query_started; this is
                # a conservative lower bound, without reading a GPU clock.
                self.pending['last_not_ready_at'] = query_started
        now = self.clock()
        if (self.pending is None and
                self.submitted['public_batches'] > self.completed['public_batches'] and
                (flush or now-self.last_fence_at >= self.interval)):
            event = self.event_factory()
            recorded_started_at = self.clock()
            event.record()
            recorded_at = self.clock()
            self.fence_sequence += 1
            self.pending = dict(event=event, totals=self._totals(), sequence=self.fence_sequence,
                recorded_started_at=recorded_started_at, recorded_at=recorded_at,
                last_not_ready_at=recorded_started_at)
            self.last_fence_at = recorded_at
        progress = (self.submitted['public_batches'], self.completed['public_batches'],
                    self.pending['sequence'] if self.pending is not None else None)
        # A short final interval must remain visible even if inference becomes
        # idle before the normal one-second publication deadline. Repeated
        # no-work polls with unchanged progress do not write per engine step.
        idle_progress = flush and progress != self.last_published_progress
        if force or idle_progress or now-self.last_publish_at >= self.interval:
            row = self.snapshot()
            self.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output.with_suffix(self.output.suffix+'.tmp')
            temporary.write_text(json.dumps(row, allow_nan=False)+'\n')
            temporary.replace(self.output)
            with self.events_output.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
            self.last_publish_at = self.clock()
            self.last_published_progress = progress
            return row
        return None

    def snapshot(self):
        self.snapshot_sequence += 1
        submitted = self._totals()
        pending = self.pending
        return dict(schema_version=1, counter_id=self.counter_id, pid=os.getpid(),
            sequence=self.snapshot_sequence, created_at=self.created_at,
            created_wall_time=self.created_wall_time, at=self.clock(), wall_time=self.wall_clock(),
            interval_seconds=self.interval, submitted=submitted,
            completed=deepcopy(self.completed), completed_fence=self.completed_fence,
            pending_fence=(dict(sequence=pending['sequence'], recorded_at=pending['recorded_at'],
                totals=deepcopy(pending['totals'])) if pending is not None else None),
            not_yet_confirmed={key: submitted[key]-self.completed[key] for key in _COUNTERS},
            all_submitted_completed=submitted['public_batches']==self.completed['public_batches'],
            last_public_batch=self.last_batch,
            scope='All real public target forwards, including intermediate prefill chunks. '
                  'No private Query forwards, restored KV, CUDA graph padding or warmup. '
                  'Prefill counts existing prompt positions actually forwarded, including any re-prefill; '
                  'non-prefill positions include MTP verification and are not accepted output tokens. '
                  'Batch composition is per-forward frequency, not time-weighted GPU occupancy. '
                  'Completion is observed asynchronously; fence brackets are not exact CUDA wall times.')
