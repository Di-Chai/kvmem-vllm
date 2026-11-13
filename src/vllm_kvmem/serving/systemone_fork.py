"""Bounded eager System One forks in native public target batches.

The scheduler reserves physical scratch and a native request slot at startup.
Full allocator pages are leased through the connector before terminal frees;
partial tails, appended questions and GDN rows are private. Before an in-place
parent refresh, the reader migrates its shared prefix into its reserved pages.
All copies and model work use the native stream. No parent rollback is involved.
"""
from dataclasses import replace
import json
import os
from pathlib import Path
import time

from .systemone_active import ActiveDecisionError
from .systemone_worker import ActiveDecisionQueue
from ..core import settings as _settings

MAX_QUESTION = 544


class ForkPagePool:
    """Scheduler-only ownership. Reserve worst-case COW before public admission."""
    def __init__(self, scheduler, memory):
        from .window_manager import WorkingSetManager
        cfg = scheduler.vllm_config
        self.branch_count = int(_settings.get('KVMEM_SYSTEMONE_BRANCHES', '1'))
        if (not scheduler._bounded_target or scheduler.async_refresh or scheduler.async_restore or
                cfg.max_concurrent_batches != 1 or cfg.scheduler_config.async_scheduling or
                self.branch_count not in (1, 2, 4) or
                cfg.scheduler_config.max_num_seqs != self.branch_count + 1 or not cfg.model_config.enforce_eager or
                cfg.cache_config.enable_prefix_caching or cfg.cache_config.mamba_cache_mode != 'none' or
                getattr(cfg.speculative_config, 'method', None) not in (None, 'mtp')):
            raise ValueError('System One fork requires eager synchronous bounded execution, '
                             'APC off, Mamba none, optional MTP, and max_num_seqs=branches+1')
        manager = scheduler.kv_cache_manager
        self.pool, self.groups = manager.block_pool, manager.groups
        self.capacity = memory.select_budget + memory.gen_budget + MAX_QUESTION
        counts = [((self.capacity + m.block_size - 1) // m.block_size
                   if isinstance(m, WorkingSetManager) else manager.static_blocks[i])
                  for i, m in enumerate(self.groups)]
        if self.pool.get_num_free_blocks() < self.branch_count * sum(counts) + manager._demand(manager.max_model_len):
            raise ValueError('Insufficient native pages for a fork and full foreground growth')
        self.verify_refresh = _settings.get('KVMEM_SYSTEMONE_VERIFY_REFRESH') == '1'
        if self.verify_refresh and _settings.get('KVMEM_VERIFY_SESSION_RESTORE') != '1':
            raise ValueError('Forced refresh is verification-only and requires byte audit')
        if self.capacity > manager.max_model_len:
            raise ValueError('Observer scratch extent exceeds native maximum model length')
        self.lane_rows = [[self.pool.get_new_blocks(n) for n in counts] for _ in range(self.branch_count)]
        self.rows = self.lane_rows[0]
        self.attention = [i for i, m in enumerate(self.groups) if isinstance(m, WorkingSetManager)]
        self.leases = {}
        self.lease_parents = {}
        self.refresh_probes = {}
        self.acquired = self.released = 0
        scheduler.max_num_running_reqs = 1
        # A retiring foreground must relinquish its worker state before a new
        # foreground can consume the sole public slot.
        manager.max_authorities = 1
        self.report()

    def metadata(self):
        return dict(rows=[[b.block_id for b in row] for row in self.rows],
                    capacity=self.capacity, attention=self.attention,
                    block_sizes=[m.block_size for m in self.groups], slot=0,
                    lanes=[dict(slot=i, rows=[[b.block_id for b in row] for row in rows])
                           for i, rows in enumerate(self.lane_rows)])

    def consume(self, notices):
        for notice in notices:
            operation = notice['operation']
            if notice['action'] == 'acquire':
                if operation in self.leases or len(self.leases) >= self.branch_count:
                    raise RuntimeError('Duplicate or overlapping observer page lease')
                if any(parent != notice['parent'] for parent in self.lease_parents.values()):
                    raise RuntimeError('Observer leases must share one foreground owner')
                blocks = []
                for group, ids in notice['groups']:
                    if group not in self.attention:
                        raise RuntimeError('Observer may only share attention pages')
                    owned = {b.block_id: b for b in self.groups[group].req_to_blocks.get(notice['parent'], ())}
                    if len(set(ids)) != len(ids) or not set(ids) <= owned.keys():
                        raise RuntimeError('Observer lease no longer belongs to its parent')
                    blocks.extend(owned[i] for i in ids)
                if (len({b.block_id for b in blocks}) != len(blocks) or
                        any(b.is_null or b.ref_cnt != 1 + sum(
                            any(owned is b for owned in lease) for lease in self.leases.values()) for b in blocks)):
                    raise RuntimeError('Observer pages have unaccounted references')
                self.pool.touch(blocks)
                self.leases[operation] = blocks
                self.lease_parents[operation] = notice['parent']
                if self.verify_refresh and blocks:
                    self.refresh_probes[operation] = notice['parent']
                self.acquired += 1
            elif notice['action'] == 'release':
                if operation not in self.leases:
                    raise RuntimeError('Unknown observer lease release')
                self.pool.free_blocks(reversed(self.leases.pop(operation)))
                self.lease_parents.pop(operation)
                self.refresh_probes.pop(operation, None)
                self.released += 1
            else:
                raise RuntimeError('Unknown observer ownership notice')
        if notices:
            self.report()

    def take_refresh_probe(self, parent):
        matches = [operation for operation, rid in self.refresh_probes.items() if rid == parent]
        for operation in matches:
            del self.refresh_probes[operation]
        return bool(matches)

    def report(self):
        directory = _settings.get('KVMEM_RUNNER_PROBE_DIR')
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            report = Path(directory) / 'systemone_fork_pool.json'
            temporary = report.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(
                branches=self.branch_count,
                reserved_blocks=sum(len(row) for rows in self.lane_rows for row in rows), leases=len(self.leases),
                pinned_blocks=sum(map(len, self.leases.values())), acquired=self.acquired,
                released=self.released, updated_at=time.perf_counter(), free_blocks_at_last_notice=self.pool.get_num_free_blocks())))
            temporary.replace(report)


def before_step(runner, output):
    metadata = getattr(output.kv_connector_metadata, 'systemone_pool', None)
    if metadata is None:
        return
    fork = getattr(runner, '_kvmem_systemone_fork', None)
    if fork is None:
        fork = ForkDecisions(runner, metadata)
        runner._kvmem_systemone_fork = fork
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    get_kv_transfer_group().systemone_worker = fork
    fork.before_step(output)


class ForkLane(ActiveDecisionQueue):
    def __init__(self, runner, pool):
        super().__init__(runner)
        self.pool, self.active, self.frame = pool, None, None
        if runner.lora_config is not None or getattr(runner.model_state, 'prompt_embeds_state', None) is not None:
            raise ValueError('System One fork requires plain text without LoRA or prompt embeddings')
        self.notices = []
        self.capture_tokens = None
        self.slot = pool['slot']
        self.quantum = int(_settings.get('KVMEM_SYSTEMONE_QUANTUM', '16'))
        if not 2 <= self.quantum <= 64:
            raise ValueError('System One quantum must be between 2 and 64 tokens')
        states = runner.req_states
        if self.slot not in states.free_indices:
            raise RuntimeError('Reserved observer state slot already occupied')
        states.free_indices.remove(self.slot)
        if runner.kv_block_zeroer is None:
            raise RuntimeError('Observer reservation requires native KV page zeroing')
        runner.kv_block_zeroer.zero_block_ids([p for row in pool['rows'] for p in row])
        self.serial = 0

    def emit(self, kind, **fields):
        out = getattr(self.runner, '_kvmem_probe_output', None)
        if out is not None:
            with (out / 'systemone_fork.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(kind=kind, at=time.perf_counter(), **fields)) + '\n')

    def rpc(self, action, operation, reference=None, reads=None):
        if action == 'cancel':
            if self.active is not None and self.active['operation'] == operation:
                self.finish(error='Decision cancelled')
            self.pending.pop(operation, None)
            return {'status': 'cancelled'}
        if action in ('submit', 'poll'):
            # Expiry must release resources, never just remove a dictionary key.
            self.expire()
        result = super().rpc(action, operation, reference, reads)
        if action == 'submit' and not result.get('error'):
            self.pending[operation]['operation'] = operation
        return result

    def expire(self):
        if self.active is not None and self.active['expires'] <= time.monotonic():
            self.finish(error='Decision expired')

    def take_notices(self):
        notices, self.notices = self.notices, []
        return notices

    def before_step(self, output):
        self.capture_tokens = None
        self.blocked_step = bool(output.kv_connector_metadata.commands or output.kv_connector_metadata.logical_prefill_groups)
        self.expire()
        job = self.active
        if job is None:
            return
        rid = job['binding'].request_id
        if rid in output.finished_req_ids or rid in output.kv_connector_metadata.cancelled:
            self.finish(error='Foreground completed or cancelled during observation')
            return
        # Refresh can overwrite retained pages even when ref_cnt > 1. Detach
        # before connector.pre_forward starts that write. Pins are released only
        # after this stream's copy completes, before scheduler refresh commit.
        if any(c.request_id == rid for c in output.kv_connector_metadata.commands):
            self.detach()

    def _copy_pages(self, group, sources, destinations):
        import torch
        from ..core.codec import native_vllm_fp8_views
        from ..core.native_pages import kernel_expansions
        runner = self.runner
        expansion = kernel_expansions(runner)[group]
        offsets = torch.arange(expansion, device=runner.device)
        src = (torch.tensor(sources, device=runner.device)[:, None] * expansion + offsets).reshape(-1).long()
        dst = (torch.tensor(destinations, device=runner.device)[:, None] * expansion + offsets).reshape(-1).long()
        if not len(sources):
            return 0
        modules = dict(runner.model.named_modules())
        names = set(runner.kv_cache_config.kv_cache_groups[group].layer_names)
        size = 0
        for name in runner._kvmem_layers:
            module = modules[name]
            if module.attn.layer_name not in names:
                continue
            if module.attn.kv_cache_dtype != 'fp8_e4m3':
                raise RuntimeError('Fork requires qualified FP8 KV')
            for cache in native_vllm_fp8_views(module.attn.kv_cache, module.num_kv_heads, module.head_dim):
                cache = cache.view(torch.uint8)
                cache[dst] = cache[src]
                size += len(src) * cache[0].numel()
        return size

    def start(self, job, public, index):
        import torch
        from .native_state import NativeRecurrentState
        from .publication_step import native_publication_step
        from .window_batch import WindowBatchAdapter
        runner = self.runner
        parent = job['binding'].request_id
        history = int(public.num_computed_tokens_np[index])
        runtime = runner._kvmem_capture_runtime
        if not runtime.drain_requests([parent], wait=True):
            return False
        if runtime.authorities[parent].history_end != history:
            raise ActiveDecisionError('Accepted history and fork boundary disagree')
        if job['reference'].get('history_end', history) != history:
            raise ActiveDecisionError('Requested history boundary is no longer active')
        steps = runtime.last_steps.get(parent, ())
        step = native_publication_step(steps[-1]) if steps else None
        if step is None or step['history_begin'] + step['accepted'] != history:
            raise ActiveDecisionError('No exact accepted recurrent-state boundary')
        maximum = max(len(r['tokens']) for r in job['reads'])
        if history + maximum > runner.model_config.max_model_len:
            raise ActiveDecisionError('Observer suffix exceeds model limit')
        if runner._kvmem_window_adapter is None:
            runner._kvmem_window_adapter = WindowBatchAdapter(tuple(self.pool['attention']), self.pool['capacity'] - MAX_QUESTION)
        adapter = runner._kvmem_window_adapter
        offset = adapter.offsets.get(parent, 0)
        window = history - offset
        if not 0 < window <= self.pool['capacity'] - MAX_QUESTION:
            raise ActiveDecisionError('Invalid active window at fork')
        self.serial += 1
        child = '__systemone_fork_' + job['operation']
        started = time.perf_counter()
        parent_index = runner.req_states.req_id_to_index[parent]
        # Register only the private native state; it never enters the scheduler's
        # public Request history, sampler or CPU archive.
        runner.req_states.free_indices.append(self.slot)
        runner.req_states.add_request(child, history, [0] * history, history, maximum)
        if runner.req_states.req_id_to_index[child] != self.slot:
            raise RuntimeError('Private request did not receive reserved slot')
        if runner.encoder_cache is not None:
            runner.encoder_cache.add_request(child, [])
        from vllm.v1.core.sched.output import NewRequestData
        initial = [0] * history
        runner.model_state.add_request(self.slot, NewRequestData(
            req_id=child, prompt_token_ids=initial, prefill_token_ids=initial, mm_features=[],
            sampling_params=None, pooling_params=None, block_ids=tuple(self.pool['rows']),
            num_computed_tokens=history, lora_request=None))
        runner.lora_state.add_request(child, self.slot, None)
        runner.req_states.apply_staged_writes()
        runner.model_state.apply_staged_writes()
        runner._kvmem_request_generations[child] = self.serial
        rows = [list(row) for row in self.pool['rows']]
        shared = []
        copied = 0
        for group in self.pool['attention']:
            block_size = self.pool['block_sizes'][group]
            full, tail = divmod(window, block_size)
            count = full + bool(tail)
            old = runner.block_tables.block_tables[group].gpu[parent_index, :count].cpu().tolist()
            if len(old) != count or any(p <= 0 for p in old):
                raise RuntimeError('Fork prefix has missing native pages')
            rows[group][:full] = old[:full]
            shared.append((group, old[:full]))
            if tail:
                copied += self._copy_pages(group, old[full:count], rows[group][full:count])
        runner.block_tables.append_block_ids(self.slot, tuple(rows), overwrite=True)
        runner.block_tables.apply_staged_writes()
        state = NativeRecurrentState(runner)
        both = state.borrow([parent, child])  # Also proves recurrent rows disjoint.
        saved = []
        audit = _settings.get('KVMEM_VERIFY_SESSION_RESTORE') == '1'
        before = {k: v.cpu() for k, v in both.tensors.items() if k.startswith(parent + '/')} if audit else {}
        speculative = step['history_begin'] + step['scheduled'] > int(public.prefill_len_np[index])
        accepted = step['accepted'] - 1 if speculative else 0
        counter = int(runner.model_state.num_accepted_tokens_gpu[parent_index].item())
        if counter == 1:
            accepted = 0  # Already canonicalized by a prior native boundary.
        for group, layers in state.groups.items():
            source = both.page_rows[parent, group]
            dest = both.page_rows[child, group][0]
            if accepted >= len(source):
                raise RuntimeError('Accepted state exceeds native speculation rows')
            for name, caches in layers.items():
                conv = caches[0][source[0]].clone()
                temporal = conv.transpose(0, 1) if state.conv_dim_first else conv
                width = state.conv_widths[name] - 1
                if accepted:
                    temporal[:width].copy_(temporal[accepted:accepted + width].clone())
                for cache, initial in zip(caches, (conv, caches[1][source[accepted]].clone())):
                    cache[dest].copy_(initial)
                    saved.append((cache[dest], initial))
        runner.model_state.num_accepted_tokens_gpu[self.slot].fill_(1)
        if audit and not all(torch.equal(both.tensors[k].cpu().view(torch.uint8), v.view(torch.uint8)) for k, v in before.items()):
            raise RuntimeError('Fork initialization mutated parent recurrent bytes')
        adapter.offsets[child] = offset
        adapter.private_capacities[child] = self.pool['capacity']
        prefix_audit = None
        if audit:
            from .systemone_worker import tail_views
            from .systemone_fork_audit import HostTailSnapshot as TailSnapshot
            prefix_audit = TailSnapshot(tail_views(runner, child, 0, window))
        job.update(child=child, history=history, window=window, rows=rows, shared=shared,
                   saved=saved, qindex=0, consumed=0, answers=[], started=started,
                   copied_bytes=copied, cow_bytes=0, mixed_steps=0, public_tokens=0,
                   shared_pages=sum(len(ids) for _, ids in shared), audit=audit,
                   prefix_audit=prefix_audit, recurrent_disjoint=True, detached=False,
                   parent_generation=runner._kvmem_request_generations[parent],
                   parent_window_generation=adapter.generations.get(parent, 0),
                   fork_ms=(time.perf_counter() - started) * 1000)
        self.active = job
        self.notices.append(dict(action='acquire', operation=job['operation'], parent=parent, groups=shared))
        self.emit('fork', operation=job['operation'], parent=parent, child=child, history=history,
                  window=window, shared_pages=job['shared_pages'], fork_ms=job['fork_ms'],
                  parent_slot=parent_index, child_slot=self.slot, rows=rows, shared=shared,
                  fork_group=job['reference'].get('fork_group'))
        return True

    def detach(self):
        job = self.active
        if job is None or job['detached']:
            return
        started = time.perf_counter()
        for group, ids in job['shared']:
            private = self.pool['rows'][group][:len(ids)]
            job['cow_bytes'] += self._copy_pages(group, ids, private)
            job['rows'][group][:len(ids)] = private
        self.runner.block_tables.append_block_ids(self.slot, tuple(job['rows']), overwrite=True)
        self.runner.block_tables.apply_staged_writes()
        self.release_pin(job)
        job['detached'] = True
        # The byte oracle's locations must follow the new private table.
        if job['prefix_audit'] is not None:
            from .systemone_worker import tail_views
            views = tail_views(self.runner, job['child'], 0, job['window'])
            old = job['prefix_audit'].rows
            job['prefix_audit'].rows = [(t, pages, offsets, saved) for (t, pages, offsets), (_, _, _, saved) in zip(views, old)]
            if not job['prefix_audit'].verify():
                raise RuntimeError('COW migration changed observer prefix')
            from .native_state import NativeRecurrentState
            from .systemone_fork_audit import HostTailSnapshot as TailSnapshot
            bundle = NativeRecurrentState(self.runner).borrow([job['child']])
            job['refresh_guard'] = (bundle, {k:v.cpu() for k,v in bundle.tensors.items()},
                TailSnapshot(tail_views(self.runner, job['child'], 0, job['window'] + job['consumed'])))
        self.emit('detach', operation=job['operation'], bytes=job['cow_bytes'],
                  elapsed_ms=(time.perf_counter() - started) * 1000)

    def release_pin(self, job):
        import torch
        if job.get('pin_released'):
            return
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.runner.device))
        event.synchronize()
        self.notices.append(dict(action='release', operation=job['operation']))
        job['pin_released'] = True

    def prepare(self, public, *, private_only=False, quantum=None):
        import numpy as np
        import torch
        from .guided_query import private_batch
        from .query_replay_batch import merge_query_batch
        if self.frame is not None:
            raise RuntimeError('Observer forward has not completed')
        if getattr(self, 'blocked_step', False) or public.num_reqs != 1 or bool(public.is_prefilling_np.any()):
            return public
        if public.num_tokens + self.quantum > self.runner.max_num_tokens:
            return public
        if self.active is None:
            for job in list(self.pending.values()):
                if job['result'].get('status') != 'pending':
                    continue
                try:
                    binding = self._binding(job['reference'])
                    if binding != job['binding']:
                        raise ActiveDecisionError('Active session owner changed before fork')
                    if binding.request_id not in public.req_ids or binding.request_id in self.runner._kvmem_sessions.imports:
                        continue
                    if self.start(job, public, public.req_ids.index(binding.request_id)):
                        break
                except ActiveDecisionError as error:
                    if self.slot in self.runner.req_states.index_to_req_id:
                        raise RuntimeError('Fork failed after acquiring its state slot; refusing reuse') from error
                    job['result'] = {'error': str(error)}
        job = self.active
        if job is None:
            return public
        if (public.req_ids != [job['binding'].request_id] or self.runner._kvmem_request_generations.get(job['binding'].request_id) != job['parent_generation']):
            self.finish(error='Foreground owner changed during observation')
            return public
        guard = job.pop('refresh_guard', None)
        if guard is not None:
            from .native_state import NativeRecurrentState
            bundle, saved, prefix = guard
            NativeRecurrentState(self.runner).validate(bundle)
            unchanged = prefix.verify() and all(torch.equal(bundle.tensors[k].cpu().contiguous().view(torch.uint8), v.contiguous().view(torch.uint8)) for k,v in saved.items())
            if not unchanged:
                raise RuntimeError('Parent refresh changed the suspended observer state')
            job['refresh_state_byte_verified'] = True
            self.emit('refresh_isolation', operation=job['operation'], child_state_and_prefix_byte_equal=True)
            del guard, bundle, saved, prefix
        read = job['reads'][job['qindex']]
        start = job['consumed']
        end = min(len(read['tokens']), start + (quantum or self.quantum))
        # Avoid one-token prefill tails: some native GDN paths classify these
        # as decode and apply different state advancement rules.
        if len(read['tokens']) - end == 1:
            end -= 1
        if start == 0:
            for dest, initial in job['saved']:
                dest.copy_(initial)
            self.runner.model_state.num_accepted_tokens_gpu[self.slot].fill_(1)
        self.runner.req_states.num_computed_tokens.gpu[self.slot] = job['history'] + start
        self.runner.req_states.num_computed_tokens_np[self.slot] = job['history'] + start
        source = replace(public, req_ids=[job['child']], idx_mapping=torch.tensor([self.slot], device=self.runner.device, dtype=public.idx_mapping.dtype), idx_mapping_np=np.array([self.slot], dtype=np.intp))
        private = private_batch(source, 0, read['tokens'][start:end], job['history'] + start, self.runner.device)
        private = replace(private, decode_graph_eligible=False, prefill_runs_as_decode_np=None, max_seq_len_np=None)
        if private_only:
            self.frame = dict(public=public, end=end, started=time.perf_counter(), private=private)
            return private
        combined = merge_query_batch(public, private)
        flags = public.prefill_runs_as_decode_np
        combined = replace(combined, decode_graph_eligible=False, prefill_runs_as_decode_np=(None if flags is None else np.concatenate((flags, [False]))))
        self.frame = dict(public=public, combined=combined, end=end, started=time.perf_counter())
        return combined

    def public_batch(self, batch):
        return self.frame['public'] if self.frame is not None and batch is self.frame['combined'] else batch

    def complete(self, state, *, project=True):
        import torch
        from .query_replay_batch import public_hidden
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs
        frame, job = self.frame, self.active
        if job is None or state is None or state.input_batch is not frame['combined']:
            raise RuntimeError('Native model did not execute the admitted observer batch')
        public = frame['public']
        read = job['reads'][job['qindex']]
        job['mixed_steps'] += 1
        job['public_tokens'] += public.num_tokens
        job['consumed'] = frame['end']
        self.emit('cobatch', operation=job['operation'], question=job['qindex'], end=frame['end'],
                  public_tokens=public.num_tokens, private_tokens=frame.get('private_count', frame['combined'].num_tokens-public.num_tokens),
                  batch_id=frame.get('batch_id'), concurrent_branches=frame.get('concurrent_branches', 1),
                  parent_history=int(public.num_computed_tokens_np[0]))
        if job['consumed'] == len(read['tokens']):
            last = frame.get('last_token_index', frame['combined'].num_tokens - 1)
            logits = self.runner.model.compute_logits(state.hidden_states[last:last+1])
            labels = torch.tensor([read['labels']], device=logits.device, dtype=torch.int64)
            job['answers'].append(dict(argmax=int(logits.argmax(-1).item()), logprobs=compute_token_logprobs(logits, labels)[0].cpu().tolist()))
            job['qindex'] += 1
            job['consumed'] = 0
        if not project:
            self.frame = None
            if job['qindex'] == len(job['reads']):
                self.finish()
            return state
        projected = state._replace(input_batch=public, hidden_states=public_hidden(state.hidden_states, public.num_tokens), aux_hidden_states=public_hidden(state.aux_hidden_states, public.num_tokens))
        metadata, slots = self.runner._kvmem_rebuild_public_metadata(public)
        self.runner._kvmem_last_batch = public
        self.capture_tokens = public.num_tokens
        self.frame = None
        if job['qindex'] == len(job['reads']):
            self.finish()
        return projected._replace(attn_metadata=metadata, slot_mappings_by_layer=slots)

    def project_capture(self, captured):
        if self.capture_tokens is None:
            return captured
        n = self.capture_tokens
        return [replace(c, positions=c.positions[:n], query=c.query[:n], key=c.key[:n], value=c.value[:n], archive_value_bytes=None if c.archive_value_bytes is None else c.archive_value_bytes[:n]) for c in captured]

    def finish(self, error=None):
        job = self.active
        if job is None:
            return
        if self.frame is not None:
            raise RuntimeError('Cannot retire an in-flight observer forward')
        self.release_pin(job)
        if job['prefix_audit'] is not None and not job['prefix_audit'].verify():
            raise RuntimeError('Observation changed its immutable prefix bytes')
        runner = self.runner
        child = job['child']
        runner.model_state.remove_request(child)
        if runner.encoder_cache is not None:
            runner.encoder_cache.remove_request(child)
        runner.lora_state.remove_request(child)
        runner.req_states.remove_request(child)
        runner.req_states.free_indices.remove(self.slot)
        runner._kvmem_request_generations.pop(child, None)
        runner._kvmem_window_adapter.free(child)
        if error is None:
            job['result'] = dict(status='done', reads=job['answers'], snapshot=dict(
                mode='fork_cobatch', session_id=job['binding'].namespace[1], generation=job['binding'].generation,
                history_end=job['history'], active_tokens=job['window'], question_tokens=sum(len(r['tokens']) for r in job['reads']),
                shared_pages=job['shared_pages'], mixed_steps=job['mixed_steps'], public_tokens_during_forward=job['public_tokens'],
                fork_ms=job['fork_ms'], execution_ms=(time.perf_counter()-job['started'])*1000,
                tail_copy_bytes=job['copied_bytes'], refresh_cow_bytes=job['cow_bytes'],
                refresh_state_byte_verified=job.get('refresh_state_byte_verified', False),
                recurrent_copy_bytes=sum(t.numel()*t.element_size() for _, t in job['saved']),
                prefix_byte_verified=job['audit'], parent_fork_state_byte_verified=job['audit'],
                counterfactual_checks=job.get('counterfactual_checks', []),
                recurrent_rows_disjoint=True, parent_rollback=bool(job.get('counterfactual_checks')),
                normal_execution_parent_rollback=False,
                parent_window_generation=job['parent_window_generation'], quantum=self.quantum))
            job['result']['snapshot'].update(observer_slot=self.slot,
                concurrent_branches=job.get('max_concurrent_branches', 1),
                fork_group=job['reference'].get('fork_group'))
        else:
            job['result'] = {'error': error}
        self.emit('finish', operation=job['operation'], error=error, mixed_steps=job['mixed_steps'], refresh_cow_bytes=job['cow_bytes'])
        for key in ('saved', 'prefix_audit', 'refresh_guard'):
            job.pop(key, None)
        self.active = None


class ForkDecisions(ActiveDecisionQueue):
    """One foreground and up to four independently owned observer lanes."""

    def __init__(self, runner, pool):
        super().__init__(runner)
        self.lanes = [ForkLane(runner, {**pool, **lane}) for lane in pool['lanes']]
        self.frame = None
        self.capture_tokens = None
        self.batch_id = 0
        self.quantum = self.lanes[0].quantum
        self.token_budget = int(_settings.get('KVMEM_SYSTEMONE_TOKEN_BUDGET', str(self.quantum * len(self.lanes))))
        if not 4 <= self.quantum <= 64 or not 4 * len(self.lanes) <= self.token_budget <= 64 * len(self.lanes):
            raise ValueError('Observer quantum must be 4..64, total budget 4..64 tokens per reserved branch')
        from .systemone_fork_audit import install
        install(self)

    def emit(self, kind, **fields):
        self.lanes[0].emit(kind, **fields)

    def memory(self, phase, **fields):
        import torch
        self.emit('memory', phase=phase, allocated_bytes=torch.cuda.memory_allocated(self.runner.device),
                  reserved_bytes=torch.cuda.memory_reserved(self.runner.device),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(self.runner.device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(self.runner.device), **fields)

    @staticmethod
    def group_key(job):
        group = job['reference'].get('fork_group')
        return (tuple(job['reference']['namespace']), group['id']) if group else None

    def expire(self):
        for lane in self.lanes:
            lane.expire()

    def rpc(self, action, operation, reference=None, reads=None):
        self.expire()
        if action == 'cancel':
            job = self.pending.get(operation)
            owner = next((lane for lane in self.lanes if lane.active is job and job is not None), None)
            if owner is not None:
                owner.finish(error='Decision cancelled')
            elif job is not None and job['result'].get('status') == 'pending' and self.group_key(job) is not None:
                for other in self.pending.values():
                    if self.group_key(other) == self.group_key(job) and other['result'].get('status') == 'pending':
                        other['result'] = {'error': 'Fork group cancelled before admission'}
            self.pending.pop(operation, None)
            return {'status': 'cancelled'}
        if action == 'submit':
            group = (reference or {}).get('fork_group')
            if group:
                if group['size'] > len(self.lanes):
                    return {'error': 'Fork group exceeds reserved observer branches'}
                key = (tuple(reference['namespace']), group['id'])
                peers = [job for job in self.pending.values() if self.group_key(job) == key]
                if len(peers) >= group['size'] or any(
                        job['reference']['fork_group']['size'] != group['size'] or
                        job['result'].get('status') != 'pending' or
                        any(lane.active is job for lane in self.lanes) for job in peers):
                    return {'error': 'Fork group is already admitted or has inconsistent size'}
        result = super().rpc(action, operation, reference, reads)
        if action == 'submit' and not result.get('error'):
            self.pending[operation]['operation'] = operation
        return result

    def before_step(self, output):
        self.capture_tokens = None
        self.expire()
        for lane in self.lanes:
            lane.before_step(output)

    def take_notices(self):
        return [notice for lane in self.lanes for notice in lane.take_notices()]

    def admit(self, public):
        free = [lane for lane in self.lanes if lane.active is None]
        waiting = [job for job in self.pending.values() if job['result'].get('status') == 'pending'
                   and not any(lane.active is job for lane in self.lanes)]
        considered = set()
        for job in waiting:
            if job['operation'] in considered:
                continue
            key = self.group_key(job)
            cohort = [other for other in waiting if self.group_key(other) == key] if key else [job]
            considered.update(other['operation'] for other in cohort)
            size = job['reference'].get('fork_group', {}).get('size', 1)
            if len(cohort) != size or len(free) < size:
                continue
            parent = job['binding'].request_id
            if parent not in public.req_ids or parent in self.runner._kvmem_sessions.imports:
                continue
            history = int(public.num_computed_tokens_np[0])
            try:
                for other in cohort:
                    if self._binding(other['reference']) != job['binding'] or other['binding'] != job['binding']:
                        raise ActiveDecisionError('Fork group foreground owner changed')
                    if other['reference'].get('history_end', history) != history:
                        raise ActiveDecisionError('Requested history boundary is no longer active')
                    if history + max(len(read['tokens']) for read in other['reads']) > self.runner.model_config.max_model_len:
                        raise ActiveDecisionError('Observer suffix exceeds model limit')
            except ActiveDecisionError as error:
                for other in cohort:
                    other['result'] = {'error': str(error)}
                continue
            if not self.runner._kvmem_capture_runtime.drain_requests([parent], wait=True):
                continue
            if all(lane.active is None for lane in self.lanes):
                import torch
                torch.cuda.reset_peak_memory_stats(self.runner.device)
                self.memory('before_admission', requested_branches=size)
            # No model forward or scheduler mutation is permitted between starts.
            for other, lane in zip(cohort, free[:size]):
                lane.pending = {}
                if not lane.start(other, public, 0):
                    raise RuntimeError('Accepted fork-group boundary lost during atomic admission')
            free = free[size:]
            self.emit('admitted', parent=parent, history=history,
                      operations=[other['operation'] for other in cohort],
                      fork_group=job['reference'].get('fork_group'))
            self.memory('after_admission', active_branches=sum(lane.active is not None for lane in self.lanes))

    def prepare(self, public):
        import numpy as np
        from .query_replay_batch import merge_query_batch
        if self.frame is not None:
            raise RuntimeError('Observer batch has not completed')
        if (public.num_reqs != 1 or bool(public.is_prefilling_np.any()) or
                any(getattr(lane, 'blocked_step', False) for lane in self.lanes) or
                public.num_tokens + self.token_budget > self.runner.max_num_tokens):
            return public
        self.admit(public)
        active = [lane for lane in self.lanes if lane.active is not None]
        if not active:
            return public
        quantum = min(self.quantum, self.token_budget // len(active))
        combined = public
        frames = []
        self.batch_id += 1
        for lane in active:
            private = lane.prepare(public, private_only=True, quantum=quantum)
            if lane.frame is None:
                continue
            begin = combined.num_tokens
            combined = merge_query_batch(combined, private)
            lane.frame.update(begin=begin, last_token_index=combined.num_tokens-1,
                              private_count=private.num_tokens, batch_id=self.batch_id)
            frames.append(lane)
        if not frames:
            return public
        flags = public.prefill_runs_as_decode_np
        combined = replace(combined, decode_graph_eligible=False,
            prefill_runs_as_decode_np=None if flags is None else np.concatenate((flags, [False]*len(frames))))
        for lane in frames:
            lane.frame.update(combined=combined, concurrent_branches=len(frames))
            lane.active['max_concurrent_branches'] = max(lane.active.get('max_concurrent_branches', 0), len(frames))
        self.frame = dict(public=public, combined=combined, lanes=frames, batch_id=self.batch_id)
        return combined

    def public_batch(self, batch):
        return self.frame['public'] if self.frame is not None and batch is self.frame['combined'] else batch

    def complete(self, state):
        from .query_replay_batch import public_hidden
        frame = self.frame
        if state is None or state.input_batch is not frame['combined']:
            raise RuntimeError('Native model did not execute the admitted multi-observer batch')
        public = frame['public']
        self.memory('after_forward', batch_id=frame['batch_id'], active_branches=len(frame['lanes']))
        for lane in frame['lanes']:
            lane.complete(state, project=False)
        metadata, slots = self.runner._kvmem_rebuild_public_metadata(public)
        self.runner._kvmem_last_batch = public
        self.capture_tokens = public.num_tokens
        self.frame = None
        return state._replace(input_batch=public, hidden_states=public_hidden(state.hidden_states, public.num_tokens),
            aux_hidden_states=public_hidden(state.aux_hidden_states, public.num_tokens),
            attn_metadata=metadata, slot_mappings_by_layer=slots)

    def project_capture(self, captured):
        return ForkLane.project_capture(self, captured)
