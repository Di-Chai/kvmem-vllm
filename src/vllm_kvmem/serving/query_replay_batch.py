"""One private Query suffix in a public target forward, without private sampling.

The native runner executes a normal mixed prefill/decode batch. Before sampling
or MTP, only public hidden rows and public metadata are exposed again. The
private owner keeps its existing refresh lease and recurrent rollback fence.
No extra request slot or token beyond the native fixed capacities is admitted.
"""
from collections import Counter
from dataclasses import dataclass, replace
import time

import numpy as np
import torch


class UnsupportedQueryBatch(ValueError):
    """A pre-execution admission rejection; the original path remains usable."""


def _check_batch(batch):
    n, t = batch.num_reqs, batch.num_tokens
    if (not n or len(batch.req_ids) != n or len(set(batch.req_ids)) != n or
            batch.num_reqs_after_padding != n or batch.num_tokens_after_padding != t or
            len(batch.num_scheduled_tokens) != n or np.any(batch.num_scheduled_tokens <= 0) or
            int(batch.num_scheduled_tokens.sum()) != t or
            len(batch.query_start_loc_np) != n + 1 or batch.query_start_loc_np[0] != 0 or
            not np.array_equal(np.diff(batch.query_start_loc_np), batch.num_scheduled_tokens)):
        raise UnsupportedQueryBatch('padded-or-inconsistent-batch')
    if (batch.dcp_local_seq_lens is not None or
            getattr(batch, 'dcp_local_seq_lens_cpu_upper_bound', None) is not None or
            getattr(batch, 'fast_prefill', None) is not None or batch.prompt_lens is not None):
        raise UnsupportedQueryBatch('unqualified-batch-layout')


def merge_query_batch(public, private):
    """Keep all public token/logit coordinates; append one non-speculative owner."""
    _check_batch(public)
    _check_batch(private)
    if (set(public.req_ids) & set(private.req_ids) or
            set(public.idx_mapping_np.tolist()) & set(private.idx_mapping_np.tolist()) or private.num_reqs != 1 or
            private.num_draft_tokens or private.num_draft_tokens_per_req is not None or
            not private.has_prefill or not bool(private.is_prefilling_np[0]) or
            private.has_structured_output_reqs):
        raise UnsupportedQueryBatch('private-owner-or-speculation')
    t, n = public.num_tokens, public.num_reqs
    cat = lambda a, b: torch.cat((a, b.to(dtype=a.dtype)), dim=0)
    arrays = {name: np.concatenate((getattr(public, name), getattr(private, name))) for name in
        ('idx_mapping_np', 'num_scheduled_tokens', 'num_computed_tokens_np',
         'prefill_len_np', 'num_computed_prefill_tokens_np', 'is_prefilling_np')}
    tensors = {name: cat(getattr(public, name), getattr(private, name)) for name in
        ('idx_mapping', 'expanded_idx_mapping', 'expanded_local_pos', 'seq_lens',
         'seq_lens_cpu_upper_bound', 'input_ids', 'positions', 'is_padding')}
    drafts = (None if public.num_draft_tokens_per_req is None else
              np.concatenate((public.num_draft_tokens_per_req, np.zeros(1, dtype=np.int32))))
    return replace(public, **arrays, **tensors, req_ids=[*public.req_ids, *private.req_ids],
        num_reqs=n + 1, num_reqs_after_padding=n + 1,
        num_tokens=t + private.num_tokens, num_tokens_after_padding=t + private.num_tokens,
        num_draft_tokens_per_req=drafts, has_prefill=True,
        query_start_loc=cat(public.query_start_loc, private.query_start_loc[1:] + t),
        query_start_loc_np=np.concatenate((public.query_start_loc_np, private.query_start_loc_np[1:] + t)),
        logits_indices=cat(public.logits_indices, private.logits_indices + t),
        cu_num_logits=cat(public.cu_num_logits, private.cu_num_logits[1:] + int(public.cu_num_logits_np[-1])),
        cu_num_logits_np=np.concatenate((public.cu_num_logits_np,
            private.cu_num_logits_np[1:] + public.cu_num_logits_np[-1])),
        max_query_len=max(int(public.num_scheduled_tokens.max()), private.num_tokens))


def public_hidden(value, tokens):
    """Project tensor/auxiliary outputs without exposing a private suffix."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[0] < tokens:
            raise ValueError('Mixed Query output does not cover its public prefix')
        return value[:tokens]
    if isinstance(value, (list, tuple)):
        return type(value)(public_hidden(item, tokens) for item in value)
    raise ValueError('Unsupported mixed Query hidden-state output')


@dataclass
class QueryFrame:
    job: object
    generation: int
    offset: int
    end: int
    started: float
    command: object = None
    state: object = None
    public: object = None
    combined: object = None
    bytes: int = 0


class QueryReplayCoBatch:
    MAX_DECLINES = 2

    def __init__(self, refresh):
        self.refresh = refresh
        self.ready = ()
        self.frame = self.failed_frame = None
        self.declines = Counter()
        self.counts = Counter()
        self.poisoned = False

    @property
    def runner(self):
        return self.refresh.runner

    def begin_turn(self, ready):
        if self.frame is not None or self.poisoned:
            raise RuntimeError('Previous mixed Query frame is not retired')
        self.ready = tuple(ready)
        live = set(self.refresh.jobs)
        self.declines = Counter({k: v for k, v in self.declines.items() if k in live})

    def defer(self, job):
        return (bool(self.ready) and job.command.request_id not in self.ready and
                not job.cancelled and self.declines[job.command.request_id] < self.MAX_DECLINES)

    def _reject(self, job, reason):
        self.declines[job.command.request_id] += 1
        self.counts['decline:' + reason] += 1
        self.frame = None
        return False

    def arm(self, scheduled):
        if self.poisoned or self.frame is not None:
            raise RuntimeError('Mixed Query execution is not reusable')
        public_ids = set(scheduled.num_scheduled_tokens)
        if not public_ids:
            return False
        jobs = [job for job in self.refresh.jobs.values() if job.stage == 'query' and
                not job.done and not job.cancelled and job.command.request_id not in public_ids and
                self.declines[job.command.request_id] < self.MAX_DECLINES]
        if not jobs:
            return False
        job = min(jobs, key=lambda j: j.last_progress if j.last_progress is not None else j.started)
        end = min(len(job.command.input_tokens), job.offset + self.refresh.options.refresh_quantum_tokens)
        count = end - job.offset
        if getattr(self.runner, '_kvmem_logical_prefill_groups', None):
            return self._reject(job, 'physical-prefill-group')
        if getattr(self.runner, 'routed_experts_capturer', None) is not None:
            return self._reject(job, 'routed-experts-output')
        if len(public_ids) + 1 > self.runner.max_num_reqs:
            return self._reject(job, 'request-capacity')
        if count <= 0 or scheduled.total_num_scheduled_tokens + count > self.runner.max_num_tokens:
            return self._reject(job, 'token-capacity')
        if job.state is None or job.restore_event is not None:
            return self._reject(job, 'state-not-borrowed')
        if not self.refresh.runtime.publication_ready((job.command.request_id,)):
            return self._reject(job, 'publication-not-ready')
        self.refresh.adapter.validate(job.state[0])
        generation = self.runner._kvmem_request_generations[job.command.request_id]
        self.frame = QueryFrame(job, generation, job.offset, end, time.perf_counter(),
                                command=job.command, state=job.state)
        self.counts['armed'] += 1
        return True

    def _validate(self, frame):
        job = frame.job
        rid = job.command.request_id
        if (job.cancelled or job.done or job.stage != 'query' or job.offset != frame.offset or
                job.command is not frame.command or job.state is not frame.state or
                self.refresh.jobs.get(rid) is not job or
                self.runner._kvmem_request_generations.get(rid) != frame.generation):
            raise RuntimeError('Mixed Query frame lost its owner or accepted prefix')
        self.refresh.adapter.validate(job.state[0])

    def prepare(self, public):
        frame = self.frame
        if frame is None:
            return public
        self._validate(frame)
        count = frame.end - frame.offset
        records = getattr(self.runner, '_kvmem_graph_captures', ())
        if not records:
            self._reject(frame.job, 'persistent-capture-required')
            return public
        # Account for the retained Q copy and its transient stack, plus all
        # merged input arrays. Target hidden storage stays within native capacity.
        required = (2 * count * sum(r.query[0].numel() * r.query.element_size() for r in records)
                    + (public.num_tokens + count) * 64 + (public.num_reqs + 1) * 256)
        if required > self.refresh.options.query_replay_max_bytes:
            self._reject(frame.job, 'memory-capacity')
            return public
        from .async_refresh import maintenance_batch
        private = maintenance_batch(self.runner, frame.job.command.request_id,
            frame.job.command.input_tokens[frame.offset:frame.end], frame.job.history + frame.offset)
        try:
            combined = merge_query_batch(public, private)
        except UnsupportedQueryBatch as error:
            self._reject(frame.job, str(error))
            return public
        frame.public, frame.combined, frame.bytes = public, combined, required
        self.counts['peak_bytes'] = max(self.counts['peak_bytes'], required)
        return combined

    def public_batch(self, batch):
        frame = self.frame
        return frame.public if frame is not None and batch is frame.combined else batch

    def complete(self, state):
        frame = self.frame
        if frame is None:
            return state
        self._validate(frame)
        if frame.combined is None or state is None or state.input_batch is not frame.combined:
            raise RuntimeError('Native target did not execute the admitted mixed Query batch')
        public = frame.public
        records = self.runner._kvmem_graph_captures
        if tuple(r.name for r in records) != tuple(self.runner._kvmem_layers):
            raise RuntimeError('Mixed Query target lost its complete Q capture')
        q = torch.stack([r.query[public.num_tokens:frame.combined.num_tokens] for r in records])
        projected = state._replace(input_batch=public,
            hidden_states=public_hidden(state.hidden_states, public.num_tokens),
            aux_hidden_states=public_hidden(state.aux_hidden_states, public.num_tokens))
        self.refresh.accept_query(frame.job, q, frame.end)
        frame.job.steps += 1
        frame.job.last_progress = time.perf_counter()
        frame.job.query_execution_counts['cobatch'] = frame.job.query_execution_counts.get('cobatch', 0) + 1
        # Native MTP receives public-only attention and slot metadata. Rebuilding
        # these must not reinitialize state, clear scores, or overwrite capture Q.
        metadata, slots = self.runner._kvmem_rebuild_public_metadata(public)
        projected = projected._replace(attn_metadata=metadata, slot_mappings_by_layer=slots)
        self.refresh.emit(frame.job, 'query_cobatch', tokens=frame.end - frame.offset,
            public_tokens=public.num_tokens, public_requests=public.num_reqs,
            host_seconds=time.perf_counter()-frame.started,
            timing_scope='Shared target forward and projection, not exclusive private cost.')
        self.counts['completed'] += 1
        self.counts['private_tokens'] += frame.end - frame.offset
        self.counts['public_tokens'] += public.num_tokens
        self.frame = None
        return projected

    def abort(self, error):
        frame = self.frame
        if frame is None:
            return
        self.poisoned = True
        self.failed_frame = frame  # Retain owner, snapshot and buffers until shutdown.
        self.frame = None
        self.counts['quarantined'] += 1
        try:
            self.refresh._restore_state(frame.job)
        finally:
            self.refresh.runtime.quarantine_publication(error)

    def snapshot(self):
        return dict(enabled=True, poisoned=self.poisoned, active=self.frame is not None,
                    max_declines=self.MAX_DECLINES, **self.counts)
