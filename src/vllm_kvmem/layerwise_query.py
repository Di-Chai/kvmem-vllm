"""Serial, eager diagnostic: select a layer's history immediately before attention.

All layers retain the original number of native slots. Their historical contents
may differ after the final public Query. This is deliberately a one-turn probe:
another refresh or session continuation is rejected, never silently interpreted
using the shared historical mapping. The ordinary serving path is unaffected.
"""
import hashlib
import json
import os
import time
from types import MethodType

import torch
from .original_rope import enabled as original_rope, record_positions


def top_set(memory, records, scores, fixed, budget):
    selected = set(memory.preview_selection(
        {r.handle.slot: float(s) for r, s in zip(records, scores)},
        tuple(fixed), action=False, prefill_scores=None, max_tokens=budget))
    if not set(fixed) <= selected or len(selected) * 32 > budget:
        raise RuntimeError('Layer selection lost mandatory content or exceeded its budget')
    return selected


class LayerwiseQuery:
    def __init__(self, runner):
        self.runner = runner
        self.mode = os.environ['KVMEM_LAYERWISE_QUERY_MODE']
        if self.mode not in ('global', 'sequential'):
            raise ValueError('Expected global or sequential layerwise diagnostic')
        self.native_batches = os.getenv('KVMEM_LAYERWISE_QUERY_NATIVE_BATCHES') == '1'
        if not self.native_batches and (not runner.model_config.enforce_eager or runner._kvmem_graph):
            raise ValueError('Layerwise diagnostic requires an eager native forward')
        if not self.native_batches and runner.vllm_config.scheduler_config.max_num_seqs != 1:
            raise ValueError('Layerwise diagnostic requires one serial request')
        self.names = tuple(runner._kvmem_layers)
        if original_rope():
            from .original_rope import extend_rope_table
            if not runner.model_config.enforce_eager:
                raise ValueError('Original RoPE qualification requires eager execution')
            modules = dict(runner.model.named_modules())
            for name in self.names:
                extend_rope_table(modules[name].rotary_emb, runner.model_config.max_model_len)
        self.pending = {}
        self.active = None
        self.finished = set()
        self.path = None
        self.optimized = os.getenv('KVMEM_LAYERWISE_QUERY_OPTIMIZED') == '1'
        self.packed_assembly = os.getenv('KVMEM_LAYERWISE_PACKED_ASSEMBLY') == '1'
        self.prepared_index = os.getenv('KVMEM_LAYERWISE_PREPARED_INDEX') == '1'
        self.batched_reads = os.getenv('KVMEM_LAYERWISE_BATCHED_READS') == '1'
        self.paired_reads = os.getenv('KVMEM_LAYERWISE_PAIRED_READS') == '1'
        if self.paired_reads and not self.optimized:
            raise ValueError('Paired layer reads require optimized Query execution')
        if self.batched_reads and not self.optimized:
            raise ValueError('Batched layer reads require optimized Query execution')
        if self.prepared_index and not self.optimized:
            raise ValueError('Prepared layer index requires optimized Query execution')
        if self.packed_assembly and (not self.optimized or original_rope()):
            raise ValueError('Packed layer assembly requires optimized compact-position execution')
        self.diagnostics = not self.optimized or os.getenv('KVMEM_LAYERWISE_QUERY_DIAGNOSTICS') == '1'
        self.buffered_events = []
        self.active_owners = {}
        self.pressure = None
        if os.getenv('KVMEM_LAYERWISE_PREFILL_PRESSURE') == '1':
            from .layerwise_pressure import LayerwisePrefillPressure
            self.pressure = runner._kvmem_layerwise_pressure = LayerwisePrefillPressure(self)

    @property
    def has_active(self):
        return self.active is not None or bool(self.active_owners)

    def emit(self, event, **fields):
        if self.path is None:
            self.path = self.runner._kvmem_probe_output / 'layerwise_query.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = dict(event=event, mode=self.mode, at=time.time(), monotonic_at=time.perf_counter(), **fields)
        if self.optimized and event == 'layer_selected':
            self.buffered_events.append(record)
            return
        with self.path.open('a') as stream:
            for pending in self.buffered_events:
                stream.write(json.dumps(pending, allow_nan=False) + '\n')
            self.buffered_events.clear()
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def prepare(self, refresh, job, queries):
        rid = job.command.request_id
        if rid in self.finished:
            raise RuntimeError('Layerwise one-turn diagnostic prohibits another refresh')
        if job.command.selection != 'query':
            if self.pressure is not None:
                self.pressure.prepare(refresh, job)
                return
            if refresh.options.pressure_policy != 'attention':
                raise ValueError('Layerwise diagnostic requires accumulated-attention pressure')
            from .prefill_attention import pressure_seed
            _, job.layerwise_pressure_source = pressure_seed(
                job.authority, tuple(job.authority.memory._store.working_records()))
            return
        if ((not self.native_batches and (self.pending or self.active is not None)) or
                rid in self.pending or rid in self.active_owners):
            raise RuntimeError('Overlapping layerwise Query owners')
        from .authority import query_score_frame
        from .request_plan import RequestPlan, selection_config
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
                               selection='query', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=32, gen_budget=cfg.gen_budget,
                          mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        authority = job.authority
        authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        with authority.index_snapshot() as (records, means):
            required = set(fit['hard_blocks'])
            pins = [r.handle.slot for r in records if r.history_begin // 32 in required]
            with query_score_frame(authority.memory, records, pins, cfg.select_budget) as frame:
                score_mask = tuple(frame.score_mask)
                fixed = tuple(authority.memory._store.fixed_selection_blocks(
                    pins, pressure=False, budget=cfg.select_budget))
            state = dict(authority=authority, records=tuple(records), means=means[:len(self.names)].clone(),
                         old_q=queries.detach().clone(), score_mask=score_mask, fixed=fixed,
                         budget=cfg.select_budget, history=job.history, span=job.command.query_span,
                         request_id=rid, next_layer=0, query_tokens=len(job.command.input_tokens))
        if state['old_q'].shape[0] != len(self.names):
            raise ValueError('Frozen Query does not cover every full-attention layer')
        if self.optimized:
            state['means'] = state['means'].to(self.runner.device)
            state['gpu_mask'] = torch.tensor(state['score_mask'], device=self.runner.device, dtype=torch.bool)
            if self.prepared_index:
                from .cuda_scoring import prepare_mean_key_index
                state['mean_index'] = prepare_mean_key_index(state['means'],state['gpu_mask'])
                # Keep one index copy after preparation, as in the old path.
                state['means'] = None
        self.pending[rid] = state

    def committed(self, job):
        if job.command.selection != 'query':
            if self.pressure is not None:
                self.pressure.committed(job)
                return
            self.emit('pressure_selected', request_id=job.command.request_id,
                      actual_method='accumulated_attention',
                      score_source=job.layerwise_pressure_source,
                      accepted_history=job.history, selected_tokens=job.selected_tokens)
            return
        state = self.pending.pop(job.command.request_id)
        authority = state['authority']
        by_slot = {r.handle.slot: r for r in state['records']}
        working = list(authority.memory._store.working_records())
        original = {r[0] for r in working}
        # Partial records must have identical inclusion in every layer so native
        # causal lengths and Query positions remain exactly the baseline's.
        state['fixed'] = tuple(set(state['fixed']) | {
            s for s in original if by_slot[s].count != 32})
        state['partial_excluded'] = {r.handle.slot for r in state['records']
                                     if r.count != 32 and r.handle.slot not in original}
        state['selected_tokens'] = job.selected_tokens
        state['global_slots'] = original
        if self.native_batches:
            self.active_owners[state['request_id']] = state
        else:
            self.active = state
        self.emit('armed', request_id=state['request_id'], layers=len(self.names),
                  budget=state['budget'], gen_budget=authority.memory.config.gen_budget,
                  selected_tokens=job.selected_tokens, query_span=state['span'],
                  fixed_blocks=len(state['fixed']), candidate_blocks=len(state['records']),
                  actual_method='mean_k_global' if self.mode == 'global' else 'mean_k_sequential_layerwise')

    def projection(self, layer_index, record):
        if self.native_batches:
            batch = self.runner._kvmem_last_batch
            # Every request keeps its own logical history, mandatory set and
            # layer cursor even when native scheduling combines public rows.
            for state in tuple(self.active_owners.values()):
                if state['request_id'] in batch.req_ids:
                    self._projection(state, layer_index, record)
        else:
            self._projection(self.active, layer_index, record)

    def _projection(self, state, layer_index, record):
        if state is None:
            return
        batch = self.runner._kvmem_last_batch
        if not self.native_batches and list(batch.req_ids) != [state['request_id']]:
            raise RuntimeError('Layerwise diagnostic reached a foreign batch')
        if layer_index != state['next_layer']:
            raise RuntimeError('Full-attention layers did not execute in causal order')
        owner_index = list(batch.req_ids).index(state['request_id'])
        begin = int(batch.num_computed_tokens_np[owner_index])
        row_begin, row_end = (map(int,batch.query_start_loc_np[owner_index:owner_index+2])
                              if self.native_batches else (0,batch.num_tokens))
        end = begin + row_end-row_begin
        qb, qe = state['span']
        if begin != state['history'] or not begin <= qb < qe <= end:
            raise RuntimeError('The complete public Query must fit in one native batch')
        q = record.query[row_begin+qb-begin:row_begin+qe-begin].unsqueeze(0)
        position_evidence = {}
        if original_rope():
            observed = record.positions[..., row_begin+qb-begin:row_begin+qe-begin]
            expected = torch.arange(qb, qe, device=observed.device, dtype=observed.dtype)
            if not torch.equal(observed, expected.expand_as(observed)):
                raise RuntimeError('Layer Query did not use original-history RoPE positions')
            position_evidence = dict(rope_position_policy='original', query_rope_span=[qb,qe],
                                     query_rope_verified=True)
        if q.shape != state['old_q'][layer_index:layer_index+1].shape:
            raise RuntimeError('Fresh and frozen layer Queries have different rows')
        from .cuda_scoring import mean_key_scores
        records = state['records']
        started = time.perf_counter()
        old_q = state['old_q'][layer_index:layer_index+1]
        if self.prepared_index:
            from .cuda_scoring import prepared_layer_scores
            scores = prepared_layer_scores(q,state['mean_index'],layer_index).cpu().tolist()
            old_scores = (prepared_layer_scores(old_q,state['mean_index'],layer_index).cpu().tolist()
                          if self.diagnostics else None)
        else:
            means = state['means'][layer_index:layer_index+1].to(q.device)
            mask = state['gpu_mask'] if self.optimized else torch.tensor(state['score_mask'], device=q.device, dtype=torch.bool)
            scores = mean_key_scores(q, means, mask).cpu().tolist()
            old_scores = mean_key_scores(old_q, means, mask).cpu().tolist() if self.diagnostics else None
        for i, r in enumerate(records):
            if r.handle.slot in state['partial_excluded']:
                scores[i] = -1.
                if old_scores is not None:
                    old_scores[i] = -1.
        memory = state['authority'].memory
        selected = top_set(memory, records, scores, state['fixed'], state['budget'])
        frozen_selected = (top_set(memory, records, old_scores, state['fixed'], state['budget'])
                           if old_scores is not None else None)
        chosen = [r for r in records if r.handle.slot in selected]
        if sum(r.count for r in chosen) != state['selected_tokens']:
            raise RuntimeError('Layer-specific selection changed the native causal length')
        scoring_seconds = time.perf_counter() - started
        actual = selected if self.mode == 'sequential' else state['global_slots']
        assembly_started = time.perf_counter()
        assembly_stats = {}
        if self.mode == 'sequential':
            if self.optimized:
                assembly_stats = self.assemble_layer_pinned(state, layer_index, chosen)
            else:
                self.assemble_layer(state, layer_index, chosen)
        assembly_seconds = time.perf_counter() - assembly_started
        diagnostic = {}
        if self.diagnostics:
            fresh, old = q.float(), old_q.float()
            norm = old.norm().clamp_min(1e-12)
            cosine = torch.nn.functional.cosine_similarity(fresh.flatten(), old.flatten(), dim=0)
            diagnostic = dict(query_cosine=float(cosine), query_relative_l2=float((fresh-old).norm()/norm),
                fresh_frozen_set_overlap=len(selected & frozen_selected)/max(1,len(selected)),
                frozen_layer_selected_starts=[r.history_begin for r in records if r.handle.slot in frozen_selected],
                q_sha256=hashlib.sha256(q.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        self.emit('layer_selected', request_id=state['request_id'], layer=layer_index,
                  layer_name=self.names[layer_index], query_source='same_forward_current_layer',
                  predecessor_layers_completed=layer_index, scoring_seconds=scoring_seconds,
                  assembly_host_seconds=assembly_seconds, optimized=self.optimized, diagnostics=self.diagnostics,
                  assembly_backend=assembly_stats,
                  prepared_candidate_index=self.prepared_index,
                  selected_tokens=state['selected_tokens'], charged_tokens=32*len(actual),
                  selected_starts=[r.history_begin for r in records if r.handle.slot in actual],
                  **diagnostic, **position_evidence)
        state['next_layer'] += 1
        if state['next_layer'] == len(self.names):
            if self.optimized:
                # One final fence also makes the phase timing comparable to
                # the original synchronized path; queued last-layer DMA must
                # not be reported as completed work.
                torch.cuda.current_stream(self.runner.device).synchronize()
            self.finished.add(state['request_id'])
            self.emit('completed', request_id=state['request_id'], layers=len(self.names))
            if self.native_batches:
                self.active_owners.pop(state['request_id'])
            else:
                self.active = None

    def assemble_layer_pinned(self, state, layer_index, chosen, *, layer_name=None,
                              skip_history_begins=()):
        """Reuse native bounded pinned slots, archive leases, and DMA fences.

        Only this layer's byte regions are read. K/V packing, FP8 rounding,
        dense positions and record order are identical to the diagnostic path.
        The following attention runs on the consuming stream, so it requires
        the DMA event dependency rather than a host synchronization per layer.
        """
        from .assembly import assemble_fp8
        from .packed_assembly import read_packed_records
        from .window_assembly import NativeWindowAssembler
        authority = state['authority']
        if authority.raw_element_bytes != 1:
            raise ValueError('Layerwise assembly requires original FP8 content Keys')
        read_blocks = None
        if self.batched_reads:
            from .layerwise_pressure_efficiency import layer_read_blocks
            options = getattr(getattr(self.runner, '_kvmem_capture_runtime', None), 'options', None)
            read_blocks = layer_read_blocks(getattr(options, 'assembly_chunk_blocks', 32), authority.shape[0])
        assembler = NativeWindowAssembler.from_runner(
            self.runner, state['request_id'],
            (self.names[layer_index] if layer_name is None else layer_name,), 32,
            chunk_blocks=read_blocks,
            arena=state.get('assembly_arena', 'layerwise'))
        layer, staging = assembler.layers[0], assembler.staging
        row_bytes = 32 * authority.shape[2] * authority.shape[3]
        total_raw = authority.shape[0] * row_bytes
        chunk_blocks = min(256, staging.slot_bytes // (2 * row_bytes))
        if chunk_blocks < 1:
            raise ValueError('Bounded native staging cannot hold one layer block')
        from .layerwise_pressure_efficiency import assembly_groups
        groups, total = assembly_groups(chosen, chunk_blocks, skip_history_begins)
        uploaded_bytes = chunks = packed_chunks = paired_read_chunks = 0
        staging.claim(assembler)
        try:
            for cursor, group in groups:
                n = sum(r.count for r in group)
                half = n * authority.shape[2] * authority.shape[3]
                gpu, ready, slot = staging.destination(2 * half)
                def fill(host):
                    nonlocal paired_read_chunks
                    if all(r.count == 32 and r.archived for r in group):
                        handles = [r.handle for r in group]
                        if getattr(self, 'paired_reads', False):
                            read_packed_records(authority, handles, destination=host,
                                regions=((layer_index*row_bytes, row_bytes),
                                         (total_raw+layer_index*row_bytes, row_bytes)))
                            paired_read_chunks += 1
                        else:
                            read_packed_records(authority, handles, destination=host[:half],
                                region=(layer_index*row_bytes, row_bytes))
                            read_packed_records(authority, handles, destination=host[half:],
                                region=(total_raw+layer_index*row_bytes, row_bytes))
                    else:
                        operands = authority.read_records([r.handle for r in group])
                        raw = torch.cat([k[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                        values = torch.cat([v[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                        host[:half].copy_(raw.contiguous().view(torch.uint8).flatten())
                        host[half:].copy_(values.contiguous().view(torch.uint8).flatten())
                transfer = assembler.transfers.upload_from_reader(2*half, gpu, fill, ready=ready)
                if transfer is None:
                    next(iter(assembler.transfers.pending.values())).event.synchronize()
                    transfer = assembler.transfers.upload_from_reader(2*half, gpu, fill, ready=ready)
                if transfer is None:
                    raise RuntimeError('Pinned layerwise transfer made no progress')
                torch.cuda.current_stream(self.runner.device).wait_event(transfer.event)
                packed = False
                if getattr(self, 'packed_assembly', False):
                    from .layerwise_pressure_efficiency import packed_layer_chunk
                    packed = packed_layer_chunk(state,layer,gpu,half,cursor,group,row_bytes,chunk_blocks)
                if packed:
                    staging.consumed(slot)
                    uploaded_bytes += 2 * half
                    chunks += 1
                    packed_chunks += 1
                    continue
                raw = gpu[:half].view(torch.float8_e4m3fn).reshape(n, authority.shape[2], authority.shape[3])
                values = gpu[half:].view(torch.float8_e4m3fn).reshape_as(raw)
                positions = torch.arange(cursor, cursor+n, device=gpu.device, dtype=torch.int64)
                page_tokens = layer.key.shape[1]
                slots = layer.kernel_page_table[positions//page_tokens].long()*page_tokens + positions%page_tokens
                rp = record_positions(group, gpu.device) if original_rope() else positions
                if original_rope() and max(r.history_begin+r.count for r in group) > layer.cos_sin.shape[0]:
                    raise ValueError('Original Key position exceeds RoPE table')
                assemble_fp8(raw, values, rp, slots, layer.cos_sin, layer.key, layer.value, layer.key_scale)
                staging.consumed(slot)
                uploaded_bytes += 2 * half
                chunks += 1
            if total != state['selected_tokens']:
                raise RuntimeError('Layer assembly did not cover the original selected-token extent')
        except BaseException:
            # Native scratch cannot be reused until every submitted copy and
            # consumer has drained, including a partially submitted chunk.
            assembler.transfers.stream.synchronize() if assembler.transfers is not None else None
            torch.cuda.current_stream(self.runner.device).synchronize()
            raise
        finally:
            staging.release(assembler)
            assembler.close()
        return dict(uploaded_bytes=uploaded_bytes, chunks=chunks, packed_chunks=packed_chunks,
                    chunk_blocks=chunk_blocks, staging_slot_bytes=staging.slot_bytes,
                    paired_read_chunks=paired_read_chunks)

    def assemble_layer(self, state, layer_index, chosen):
        from .assembly import assemble_fp8
        from .packed_assembly import read_packed_records
        from .window_assembly import NativeWindowAssembler
        authority = state['authority']
        assembler = NativeWindowAssembler.from_runner(
            self.runner, state['request_id'], (self.names[layer_index],), 32, arena='layerwise')
        layer = assembler.layers[0]
        row_bytes = 32*authority.shape[2]*authority.shape[3]
        total_raw = authority.shape[0]*row_bytes*authority.raw_element_bytes
        if authority.raw_element_bytes != 1:
            raise ValueError('Layerwise diagnostic requires original FP8 content Keys')
        cursor = 0
        try:
            for begin in range(0, len(chosen), 32):
                group = chosen[begin:begin+32]
                if all(r.count == 32 and r.archived for r in group):
                    handles = [r.handle for r in group]
                    kb, _ = read_packed_records(authority, handles,
                        region=(layer_index*row_bytes, row_bytes))
                    vb, _ = read_packed_records(authority, handles,
                        region=(total_raw+layer_index*row_bytes, row_bytes))
                    raw = kb.view(torch.float8_e4m3fn).reshape(-1, authority.shape[2], authority.shape[3])
                    values = vb.reshape_as(raw)
                else:
                    operands = authority.read_records([r.handle for r in group])
                    raw = torch.cat([k[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                    values = torch.cat([v[layer_index, :r.count] for r,(k,v) in zip(group,operands)])
                raw = raw.to(self.runner.device)
                values = values.view(torch.float8_e4m3fn).to(self.runner.device)
                positions = torch.arange(cursor, cursor+len(raw), device=raw.device, dtype=torch.int64)
                page_tokens = layer.key.shape[1]
                slots = layer.kernel_page_table[positions//page_tokens].long()*page_tokens + positions%page_tokens
                rp = record_positions(chosen[begin:begin+32], raw.device) if original_rope() else positions
                assemble_fp8(raw, values, rp, slots, layer.cos_sin, layer.key, layer.value, layer.key_scale)
                cursor += len(raw)
            torch.cuda.current_stream().synchronize()
            if cursor != state['selected_tokens']:
                raise RuntimeError('Layer assembly did not cover the original selected-token extent')
        finally:
            assembler.close()


def install_layerwise_query(runner):
    from .async_refresh import AsyncRefresh
    from .capture import _sink, capture_to
    controller = runner._kvmem_layerwise_query = LayerwiseQuery(runner)
    modules = dict(runner.model.named_modules())
    for index, name in enumerate(controller.names):
        module = modules[name]
        original = module._project_qkv_gate
        def project(self, qkv, positions, _index=index, _original=original):
            if not controller.has_active:
                return _original(qkv, positions)
            buffers = getattr(self, '_kvmem_capture_buffers', None)
            if buffers is not None:
                # Persistent capture deliberately bypasses ContextVar sinks.
                # Its fused native projection has just written these buffers
                # on the consuming stream; use the current rows immediately.
                from .capture import CapturedLayer
                result = _original(qkv, positions)
                rows = qkv.shape[0]
                if rows > buffers.query.shape[0]:
                    raise RuntimeError('Current Query exceeds native capture capacity')
                record = CapturedLayer(buffers.name, positions, buffers.query[:rows],
                                       buffers.key[:rows], buffers.value[:rows])
                controller.projection(_index, record)
                return result
            parent = _sink.get()
            captured = []
            def receive(record):
                captured.append(record)
                if parent is not None:
                    parent(record)
            with capture_to(receive):
                result = _original(qkv, positions)
            if len(captured) != 1:
                raise RuntimeError('Layerwise projection did not capture exactly one real Q')
            controller.projection(_index, captured[0])
            return result
        module._project_qkv_gate = MethodType(project, module)
    if not getattr(AsyncRefresh, '_layerwise_diagnostic_installed', False):
        assemble, report = AsyncRefresh._assemble, AsyncRefresh.report
        def assemble_with_probe(self, job, queries, ready):
            diagnostic = getattr(self.runner, '_kvmem_layerwise_query', None)
            if diagnostic is not None:
                with torch.inference_mode(), torch.cuda.device(self.runner.device), torch.cuda.stream(self.stream):
                    self.stream.wait_event(ready)
                    diagnostic.prepare(self, job, queries)
            return assemble(self, job, queries, ready)
        def report_with_probe(self, job, assembly, fit, seconds):
            result = report(self, job, assembly, fit, seconds)
            diagnostic = getattr(self.runner, '_kvmem_layerwise_query', None)
            if diagnostic is not None:
                diagnostic.committed(job)
            return result
        AsyncRefresh._assemble, AsyncRefresh.report = assemble_with_probe, report_with_probe
        AsyncRefresh._layerwise_diagnostic_installed = True
    if controller.native_batches:
        install_native_query_dispatch(controller)


def install_native_query_dispatch(controller):
    """Keep native compiled/graph execution except an armed layerwise Query.

    The Query requires Python-controlled archive access between layers. Native
    dispatch and its existing skip_compiled contract provide that boundary;
    normal prefill and decode retain the configured batch/compiled execution.
    """
    from vllm.v1.worker.gpu import model_runner
    original_dispatch = model_runner.dispatch_cg_and_sync_dp
    original_context = model_runner.set_forward_context
    if getattr(original_dispatch, '_kvmem_layerwise_owner', None) is not None:
        raise RuntimeError('A second layerwise dispatcher cannot share the native worker')
    def dispatch(*args, **kwargs):
        if controller.has_active:
            kwargs['need_eager'] = True
        return original_dispatch(*args, **kwargs)
    def context(*args, **kwargs):
        if controller.has_active:
            kwargs['skip_compiled'] = True
        return original_context(*args, **kwargs)
    dispatch._kvmem_layerwise_owner = controller
    model_runner.dispatch_cg_and_sync_dp = dispatch
    model_runner.set_forward_context = context
