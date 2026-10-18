"""Bounded, nonblocking CUDA-event feedback outside compiled model graphs."""
from collections import deque
import json
import time


class StepTiming:
    def __init__(self, interval, output, *, event_factory=None, clock=time.monotonic):
        if event_factory is None:
            import torch
            event_factory = lambda: torch.cuda.Event(enable_timing=True)
        self.event_factory, self.clock = event_factory, clock
        self.interval, self.output = interval, output
        self.calls = self.sequence = self.dropped = 0
        self.active = None
        self.pending = deque()

    def begin(self, cost_classification=None):
        if self.active is not None:
            raise RuntimeError('Previous measured public batch never reached sampling')
        self.calls += 1
        if (self.calls - 1) % self.interval:
            return
        if len(self.pending) >= 16:
            self.dropped += 1
            return
        start, model, end = self.event_factory(), self.event_factory(), self.event_factory()
        start.record()
        self.sequence += 1
        self.active = dict(sequence=self.sequence, start=start, model=model, end=end,
                           started_at=self.clock(), prefill_tokens=0, decode_requests=0,
                           cost_classification=dict(cost_classification or {}))

    def classify(self, batch):
        if self.active is not None:
            self.active['prefill_tokens'] = sum(int(n) for n, p in
                zip(batch.num_scheduled_tokens, batch.is_prefilling_np) if p)
            self.active['decode_requests'] = sum(not bool(p) for p in batch.is_prefilling_np)
            self.active['prefill_requests'] = sum(bool(p) for p in batch.is_prefilling_np)
            source = self.active['cost_classification']
            known = bool(source) and hasattr(batch, 'req_ids') and hasattr(batch, 'num_computed_tokens_np')
            ordinary = atomic = unknown = 0
            receipt = None
            if known:
                for i, rid in enumerate(batch.req_ids):
                    n, h = int(batch.num_scheduled_tokens[i]), int(batch.num_computed_tokens_np[i])
                    entry = source.get(rid)
                    valid = (entry is not None and entry['history'] == h and
                             entry['scheduled_tokens'] == n)
                    if hasattr(batch, 'prefill_len_np') and hasattr(batch, 'num_computed_prefill_tokens_np'):
                        remaining = max(0, int(batch.prefill_len_np[i])-int(batch.num_computed_prefill_tokens_np[i]))
                    else:
                        remaining = max(0, entry['prompt_tokens']-h) if entry else n if batch.is_prefilling_np[i] else 0
                    public = min(n, remaining)
                    if public and entry is not None:
                        receipt = entry.get('authorization_id')
                    if bool(public) != bool(batch.is_prefilling_np[i]):
                        valid = False
                    known &= valid
                    span = entry.get('atomic_range') if entry else None
                    replay = (entry is not None and entry.get('prepared_replay', False) or
                              span is not None and h < span[1] and h + public > span[0])
                    if not valid:
                        unknown += public
                    elif replay:
                        atomic += public
                    else:
                        ordinary += public
                self.active['prefill_tokens'] = ordinary + atomic + unknown
            else:
                unknown = self.active['prefill_tokens']
            self.active.update(ordinary_prefill_only=bool(known and not atomic and not unknown),
                ordinary_prefill_tokens=ordinary, atomic_prefill_tokens=atomic,
                unknown_prefill_tokens=unknown,
                coalescing_authorization_id=receipt if self.active['prefill_requests'] == 1 else None)

    def model_begin(self, batch, offsets=None):
        """Fence after request maintenance/attention preparation, before forward.

        Neither this boundary nor end synchronizes the GPU. Model cost includes
        sampling, MTP and subsequent capture work; it is not pure kernel time.
        """
        if self.active is not None:
            self.classify(batch)
            if hasattr(batch, 'num_computed_tokens_np'):
                from .offline_policy import context_bucket
                offsets = offsets or {}
                extents = [int(h) + int(n) - offsets.get(rid, 0) for rid,h,n in
                           zip(batch.req_ids, batch.num_computed_tokens_np, batch.num_scheduled_tokens)]
                for prefill, name in ((True, 'prefill'), (False, 'decode')):
                    lengths = [n for n,p in zip(extents, batch.is_prefilling_np) if bool(p) == prefill]
                    self.active[name + '_context_bucket'] = context_bucket(max(lengths)) if lengths else None
            self.active['model'].record()
            self.active['model_submitted_at'] = self.clock()

    def end(self):
        if self.active is None:
            return
        row = self.active
        row['end'].record()  # After native sampling, GDN postprocess and MTP.
        row['submitted_at'] = self.clock()
        self.pending.append(row)
        self.active = None

    def poll(self):
        rows = []
        while self.pending and self.pending[0]['end'].query():
            row = self.pending.popleft()
            row.pop('cost_classification', None)
            start, model, end = row.pop('start'), row.pop('model'), row.pop('end')
            row['step_ms'] = start.elapsed_time(end)
            if 'model_submitted_at' in row:
                row['maintenance_ms'] = start.elapsed_time(model)
                row['model_ms'] = model.elapsed_time(end)
                row['maintenance_host_ms'] = (row['model_submitted_at'] - row['started_at']) * 1000
            row['host_submit_ms'] = (row['submitted_at'] - row['started_at']) * 1000
            rows.append(row)
        if rows and self.output is not None:
            with self.output.open('a') as stream:
                stream.writelines(json.dumps(row) + '\n' for row in rows)
        return rows
