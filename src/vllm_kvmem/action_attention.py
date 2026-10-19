"""Request-owned action feedback with qw3's transactional scope parser.

Output feedback uses only scheduler-confirmed, consumed rows. Optional input
feedback separately tracks sampled prefill mass and causal exposure. Private
guided Q, public Query replay, MTP draft layers and rejected rows contribute
neither. Both score histories are checkpointed with the exact historical prefix.
"""
from dataclasses import dataclass
from contextlib import nullcontext
import json
import math
import re
import time

import torch

from . import _native
from .codec import native_vllm_fp8_views


class _AcceptanceParser:
    """Read-only acceptance preview around the unchanged native scope parser.

    Graph reservations must succeed before native commit advances thinking,
    interval and accepted-prefix state. Only enabled graph sessions pay for this
    wrapper; both legacy feedback and graph then consume the same commit mask.
    """
    def __init__(self, parser):
        self.parser = parser
        self.mask = self.count = self.accepted = 0

    @property
    def state(self):
        return self.parser.state

    def prepare(self, tokens):
        mask = self.parser.prepare(tokens)
        self.mask, self.count, self.accepted = mask, len(tokens), 0
        return mask

    def preview(self, accepted):
        if not isinstance(accepted, int) or not self.accepted <= accepted <= self.count:
            raise ValueError('Invalid action attention acceptance prefix')
        return self.mask & ((1 << accepted)-1) & ~((1 << self.accepted)-1)

    def commit(self, accepted):
        expected = self.preview(accepted)
        result = self.parser.commit(accepted)
        if result != expected:
            raise RuntimeError('Action acceptance preview differs from native parser')
        self.accepted = accepted
        return result


class ActionHistory:
    def __init__(self, options, token_ids):
        self.options,self.token_ids=options,token_ids
        self.scores={}
        self.request_id=None
        self.parser=None
        self.mass={}
        self.samples=0
        self.rounds=0
        self.replay_snapshot=None
        from .prefill_attention import PrefillHistory
        self.prefill = PrefillHistory(options) if getattr(options,'prefill_attention',False) else None
        self.prefill_replay = None

    def begin(self, request_id, prompt):
        if self.request_id==request_id:
            return
        thinking=False
        # The last message header resets historical assistant thinking markers.
        # Only the last scope marker determines the initial state. Agent
        # prompts usually end in a fresh message header; do not rescan the
        # complete retained history on every tool return.
        for token in reversed(prompt):
            if token in (self.token_ids[4], self.token_ids[2], self.token_ids[3]):
                thinking = token == self.token_ids[2]
                break
        self.parser=_native.ActionParser(self.options.action_scope,self.options.action_interval,
                                        self.token_ids[:4],thinking)
        if (getattr(self.options, 'graph_mode', 'off') != 'off' or
                getattr(self.options, 'action_native_decode_commit', False)):
            self.parser = _AcceptanceParser(self.parser)
        self.request_id=request_id
        self.mass={};self.samples=0

    def accept(self, accepted, rows, history, masses):
        if getattr(self.options, 'action_native_decode_commit', False):
            mask = self.parser.preview(accepted)
            indices = [i for i,row in enumerate(rows) if (mask >> row) & 1]
            if not indices:
                if self.parser.commit(accepted) != mask:
                    raise RuntimeError('Action transaction lost its accepted prefix')
                return
            if hasattr(masses, 'numpy'):
                masses = masses.numpy()
            elif not hasattr(masses, 'shape'):
                import numpy as np
                masses = np.asarray([] if masses is None else masses, dtype='float64').reshape(len(rows),len(history))
            updated, samples = _native.prepare_decode_attention(self.mass,self.samples,indices,
                history,masses,getattr(self.options,'action_commit_max_bytes',16777216))
            if self.parser.commit(accepted) != mask:
                raise RuntimeError('Action transaction lost its accepted prefix')
            self.mass, self.samples = updated, samples
            return
        mask=self.parser.commit(accepted)
        for i,row in enumerate(rows):
            if not (mask>>row)&1:
                continue
            values=masses[i]
            if any(not math.isfinite(v) or v<0 for v in values) or abs(sum(values)-1)>.005:
                raise RuntimeError('Action attention mass is invalid or does not sum to one')
            for key,value in zip(history,values):
                if value: self.mass[key]=self.mass.get(key,0.0)+value
            self.samples+=1

    def finish(self, authority):
        live={r.history_begin for r in authority.records.values()}
        if self.samples:
            self.scores={key:self.options.action_decay*self.scores.get(key,0.0)+
                         self.mass.get(key,0.0)/self.samples for key in live}
            self.rounds+=1
        else:
            self.scores={key:value for key,value in self.scores.items() if key in live}
        authority.memory.set_action_scores(self.scores)
        report={'samples':self.samples,'rounds':self.rounds,
                'positive_pages':sum(v>0 for v in self.scores.values()),
                'score_sum':sum(self.scores.values()),'scope':self.options.action_scope}
        self.parser=None; self.mass={}
        return report

    def snapshot(self, position=None):
        if self.replay_snapshot is not None and position is not None and position<self.replay_snapshot[0]:
            # A replay-only P checkpoint cannot publish feedback whose accepted
            # source extends beyond that checkpoint to a future edited branch.
            from copy import deepcopy
            return deepcopy(self.replay_snapshot[1])
        result = (dict(self.scores),self.rounds)
        return result + (self.prefill.snapshot(),) if self.prefill is not None else result

    def restore(self, snapshot, authority, *, preserve_future=False, validated_end=None):
        # An M->P model rewind can retain feedback through M only when the
        # checkpoint claim compared the entire M token prefix. The physical
        # authority ends at P here; it is not the feedback validity boundary.
        if validated_end is not None:
            if (not preserve_future or type(validated_end) is not int or
                    validated_end < authority.history_end):
                raise ValueError('Invalid validated action-history prefix')
        prefill_snapshot = snapshot[2] if len(snapshot)>2 else ({},0,0,0)
        if self.prefill is not None:
            prefill_end = authority.history_end if validated_end is None else validated_end
            if preserve_future and prefill_snapshot[1] > prefill_end:
                raise ValueError('Future prefill history requires its complete validated prefix')
        scores,self.rounds=snapshot[:2]
        live={r.history_begin for r in authority.records.values()}
        self.scores={key:value for key,value in scores.items() if preserve_future or key in live}
        self.request_id=None;self.parser=None;self.mass={};self.samples=0
        self.replay_snapshot=None
        self.prefill_replay=None
        if self.prefill is not None:
            self.prefill.restore(prefill_snapshot, prefill_end)
        authority.memory.set_action_scores(self.scores)


def window_segments(records, history_begin, local_begin, count, block_tokens):
    """Freeze logical identities, including a partial historical tail."""
    starts=[];counts=[];history=[];cursor=0
    for _,_,begin,n,position in records:
        if position>=local_begin: break
        n=min(n,local_begin-position)
        if position!=cursor or n<=0 or n>block_tokens:
            raise ValueError('Action window has a gap or stale dense mapping')
        starts.append(cursor);counts.append(n);history.append(begin-begin%block_tokens)
        cursor+=n
    logical=history_begin+cursor-local_begin
    if logical<0: raise ValueError('Invalid action suffix origin')
    while cursor<local_begin+count:
        n=min(block_tokens-logical%block_tokens,local_begin+count-cursor)
        starts.append(cursor);counts.append(n);history.append(logical-logical%block_tokens)
        cursor+=n;logical+=n
    return starts,counts,history


@dataclass
class ActionPacket:
    state: ActionHistory
    rows: list
    history: list
    masses: object
    event: object
    prefill: tuple | None = None
    # Original consumed-Q coordinates, never predicted-token (+1), dense
    # working positions, GPU page indices or a temporary request identity.
    source_positions: tuple = ()
    history_begin: int | None = None
    candidate_count: int = 0


class _PreparedActionAcceptance:
    """A fully validated, reserved graph/legacy publication transaction."""
    def __init__(self, runtime, packet, request_id, pending, graph_update,
                 *, accepted, output=None, prefill=None):
        self.runtime, self.packet, self.request_id = runtime, packet, request_id
        self.pending, self.graph_update = pending, graph_update
        self.accepted, self.output, self.prefill = accepted, output, prefill
        self.done = False

    def commit(self):
        if self.done:
            raise RuntimeError('Action acceptance transaction already finished')
        self.graph_update.resume()
        pending = self.pending
        if self.output is not None:
            mask, mass, samples = self.output
            # All row validation and memory reservation precede this sole
            # parser commit. The graph must not call commit independently.
            if pending.state.parser.commit(self.accepted) != mask:
                raise RuntimeError('Action transaction lost its accepted prefix')
            pending.state.mass, pending.state.samples = mass, samples
        elif self.prefill is not None:
            original, staged, committed = self.prefill
            original.__dict__ = staged.__dict__
        started = time.perf_counter()
        self.graph_update.commit()
        self.runtime.graph_commit_seconds = getattr(self.runtime,'graph_commit_seconds',0.) + time.perf_counter()-started
        self.runtime.graph_commit_calls = getattr(self.runtime,'graph_commit_calls',0) + 1
        self.packet.action_inputs.pop(self.request_id, None)
        if self.output is not None:
            backend = 'native' if getattr(pending.state.options,'action_native_decode_commit',False) else 'python'
            key = 'decode_commit_'+backend+'_calls'
            setattr(self.runtime,key,getattr(self.runtime,key,0)+1)
        self.done = True
        if self.prefill is not None and self.prefill[2]:
            self.runtime._report_prefill(self.request_id, pending)
        self.output = self.prefill = None

    def abort(self):
        if not self.done:
            self.output = self.prefill = None
            self.graph_update.abort()
            self.done = True


@dataclass
class ActionMetadata:
    batch: object
    indices: tuple
    slices: tuple
    host: object
    event: object


def action_metadata(batch, local_positions, prefill_attention, profile=None, pool=None):
    """Snapshot exact GPU coordinates before the target forward is enqueued.

    CPU scheduler counters are only upper bounds under speculative decoding.
    This copies the actual IDs and positions, and records a fence immediately
    after that copy, independently of the later target and draft kernels.
    """
    indices = tuple(i for i in range(batch.num_reqs)
                    if not batch.is_prefilling_np[i] or prefill_attention)
    if not indices:
        return None
    slices = tuple((int(batch.query_start_loc_np[i]), int(batch.query_start_loc_np[i+1]))
                   for i in indices)
    if local_positions.ndim == 2:
        local_positions = local_positions[0]
    with profile.stage('metadata.pack', gpu=True,
                       shape=[3, sum(b-a for a, b in slices)]) if profile else nullcontext():
        packed = torch.cat([torch.stack((batch.input_ids[a:b].long(), batch.positions[a:b],
                                        local_positions[a:b])) for a, b in slices], dim=1)
    event = None
    if packed.device.type == 'cuda' and pool is not None:
        host,event=pool.copy(packed)
    elif packed.device.type == 'cuda':
        with profile.stage('metadata.pinned_allocate', shape=list(packed.shape),
                           allocation_bytes=packed.numel()*packed.element_size()) if profile else nullcontext():
            host = torch.empty(packed.shape, dtype=packed.dtype, pin_memory=True)
        with profile.stage('metadata.d2h', gpu=True, transfer_bytes=packed.numel()*packed.element_size(),
                           direction='D2H') if profile else nullcontext():
            host.copy_(packed, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(packed.device))
    else:
        host = packed.clone()
    return ActionMetadata(batch, indices, slices, host, event)


class ActionRuntime:
    def __init__(self, runtime):
        self.runtime=runtime
        self.runner=runtime.runner
        self.options=runtime.options
        self.capture_seconds=0.0
        self.candidate_samples=0
        self.packets=0
        self.metadata = None
        self.metadata_early_batches = 0
        self.metadata_fallback_batches = 0
        self.metadata_wait_seconds = 0.0
        self.metadata_upload = None
        self.download_pool = None
        if getattr(self.options,'action_download_pool',False):
            from .action_download_pool import ActionDownloadPool
            self.download_pool=ActionDownloadPool(self.options.action_download_max_bytes,
                self.options.action_download_max_slots)
        if getattr(self.options, 'async_action_metadata', False):
            from .action_metadata_upload import PackedActionMetadata
            self.metadata_upload = PackedActionMetadata()
        self.profile = None
        if getattr(self.options, 'action_profiling', False):
            from .action_profile import ActionProfiler
            self.profile = ActionProfiler(runtime.output/'action_profile.jsonl',
                interval=self.options.action_profile_interval,
                max_events=self.options.action_profile_max_events)
        from vllm.tokenizers import get_tokenizer
        cfg=self.runner.vllm_config.model_config
        tokenizer=get_tokenizer(cfg.tokenizer,tokenizer_mode=cfg.tokenizer_mode,
                                trust_remote_code=cfg.trust_remote_code)
        self.ids=[tokenizer.convert_tokens_to_ids(s) for s in
                  ('<tool_call>','</tool_call>','<think>','</think>','<|im_start|>')]
        if any(x is None or x<0 for x in self.ids) or len(set(self.ids))!=5:
            raise ValueError('Action attention requires distinct Qwen scope tokens')
        names=self.runner._kvmem_layers
        ids=[int(re.search(r'layers\.(\d+)\.',name).group(1)) for name in names]
        selected=_native.resolve_action_layers(self.options.action_layers,ids)
        self.layers=[(i,runtime.layers[i]) for i,number in enumerate(ids) if number in selected]
        self.groups={name:group for group,g in enumerate(self.runner.kv_cache_config.kv_cache_groups)
                     for name in g.layer_names}
        self.q_dtypes={}
        self.prefill_q_dtypes={}
        for groups in self.runner.attn_groups:
            for group in groups:
                builder=group.get_metadata_builder(0)
                for name in group.layer_names:
                    if hasattr(builder,'q_data_type_decode'):
                        self.q_dtypes[name]=builder.q_data_type_decode
                        self.prefill_q_dtypes[name]=builder.q_data_type_prefill
        for _,module in self.layers:
            impl=module.attn.impl
            if (module.attn.layer_name not in self.q_dtypes or
                    getattr(impl,'logits_soft_cap',None) not in (None,0) or
                    getattr(impl,'window_left',-1)!=-1 or getattr(impl,'sinks',None) is not None):
                raise ValueError('Action sampling requires qualified full causal FlashInfer attention')
        (runtime.output/'action_config.json').write_text(json.dumps({
            'layers':selected,'scope':self.options.action_scope,'interval':self.options.action_interval,
            'decay':self.options.action_decay,'fraction':runtime.config.action_fraction,
            'combine':runtime.config.action_combine,'q_dtype':{k:str(v) for k,v in self.q_dtypes.items()},
            'prefill_attention':self.options.prefill_attention,'action_pressure':self.options.action_pressure,
            'prefill_interval':self.options.prefill_attention_interval,
            'prefill_stride':self.options.prefill_attention_stride,
            'prefill_half_life':self.options.prefill_attention_half_life,
            'native_action_metadata':getattr(self.options,'native_action_metadata',False),
            'async_action_metadata':getattr(self.options,'async_action_metadata',False),
            'prefill_q_dtype':{k:str(v) for k,v in self.prefill_q_dtypes.items()},
            'native_decode_commit':getattr(self.options,'action_native_decode_commit',False),
            'download_pool':getattr(self.options,'action_download_pool',False),
            'fused_scoring':getattr(self.options,'action_fused_scoring',False),
            'implementation':'sampled target Q against native working K; original qw3 parser/fusion'},indent=2))

    def begin_response(self, state, request_id):
        if state.request_id == request_id:
            return
        sessions=getattr(self.runner,'_kvmem_sessions',None)
        binding=sessions.bindings.get(request_id) if sessions is not None else None
        if binding is not None:
            prompt=binding.tokens
        else:
            # Stateless requests and sessions admitted without a cache slot
            # still have an exact native prompt. Do not assume every request
            # owns a retained P/M binding; never include generated draft IDs.
            states=self.runner.req_states
            index=states.req_id_to_index[request_id]
            count=int(states.prompt_len.np[index])
            prompt=states.all_token_ids.gpu[index,:count].cpu().tolist()
        state.begin(request_id,prompt)

    def prepare(self, batch, local_positions):
        # Called only for public forwards. Private Query/guided replays must
        # not overwrite the packet associated with a public native batch.
        profiler = getattr(self, 'profile', None)
        trace = profiler.begin_batch(batch) if profiler is not None else None
        self.metadata = action_metadata(batch, local_positions, self.options.prefill_attention, trace,
            pool=getattr(self,'download_pool',None))

    def _window_segments(self, authority, history_begin, local_begin, count):
        store = authority.memory._store
        block_tokens = self.runtime.config.block_tokens
        if getattr(self.options, 'native_action_metadata', False):
            native = getattr(store, 'action_window_segments', None)
            if native is None:
                raise RuntimeError('native_action_metadata requires a rebuilt native extension')
            return native(history_begin, local_begin, count, block_tokens)
        return window_segments(store.working_records(), history_begin, local_begin, count, block_tokens)

    def capture(self, captured, batch):
        profiler = getattr(self, 'profile', None)
        if profiler is None:
            return self._capture(captured, batch)
        trace = profiler.begin_batch(batch)
        status = 'submitted'
        try:
            return self._capture(captured, batch, trace)
        except BaseException:
            status = 'error'
            raise
        finally:
            profiler.finish_batch(status=status)

    def _capture(self, captured, batch, trace=None):
        indices=[i for i in range(batch.num_reqs) if not batch.is_prefilling_np[i] or self.options.prefill_attention]
        if not indices:return {}
        started=time.perf_counter()
        metadata = getattr(self, 'metadata', None)
        self.metadata = None
        if metadata is not None and metadata.batch is batch:
            self.metadata_early_batches += 1
        else:
            # The capture runtime is constructed after the first forward.
            # Standalone diagnostic callers also use this exact fallback.
            metadata = action_metadata(batch, captured[0].positions, self.options.prefill_attention, trace,
                pool=getattr(self,'download_pool',None))
            self.metadata_fallback_batches = getattr(self, 'metadata_fallback_batches', 0) + 1
        waited = time.perf_counter()
        with trace.stage('metadata.existing_fence') if trace else nullcontext():
            if metadata.event is not None:
                metadata.event.synchronize()
        self.metadata_wait_seconds = getattr(self, 'metadata_wait_seconds', 0.0) + time.perf_counter() - waited
        slices = metadata.slices
        with trace.stage('metadata.tolist') if trace else nullcontext():
            data = metadata.host.tolist()
        tokens,global_positions,local_positions=data
        packets={};cursor=0
        from .action_scoring import page_attention
        for i,(begin,end) in zip(indices,slices):
            request_id=batch.req_ids[i]
            prefill=bool(batch.is_prefilling_np[i])
            owner = trace.child(request_id=request_id, kind='prefill' if prefill else 'output') if trace else None
            # Parser state and dense page identities must both reflect this
            # request's preceding accepted prefix. A shared DMA packet can
            # still have an unrelated owner publishing CPU history. Admission
            # holds large unfinished owners out of this public batch; tiny
            # decode dependencies may finish through this request-local fence.
            with owner.stage('publication.owner_drain') if owner else nullcontext():
                if not self.runtime.drain_requests((request_id,), wait=True):
                    raise RuntimeError('Action capture preceded scheduler confirmation')
            count=end-begin
            authority=self.runtime.ensure_authority(request_id)
            state=getattr(authority,'action_history',None)
            if state is None:
                state=authority.action_history=ActionHistory(self.options,self.ids)
            history_begin=int(global_positions[cursor])
            local_begin=int(local_positions[cursor])
            if prefill:
                excluded=state.prefill_replay
                replay=excluded is not None and history_begin < excluded[1] and history_begin+count > excluded[0]
                if replay and not (excluded[0]<=history_begin and history_begin+count<=excluded[1]):
                    raise RuntimeError('Prefill chunk partly overlaps a public Query replay')
                with owner.stage('prefill.rows') if owner else nullcontext():
                    rows=state.prefill.rows(history_begin,count,local_begin,replay=replay)
            else:
                with owner.stage('output.begin_response') if owner else nullcontext():
                    self.begin_response(state,request_id)
                with owner.stage('output.parser_prepare') if owner else nullcontext():
                    mask=state.parser.prepare(tokens[cursor:cursor+count])
                    rows=[r for r in range(count) if (mask>>r)&1]
            sources = (tuple(int(global_positions[cursor+r]) for r in rows)
                       if getattr(authority, 'block_graph', None) is not None else ())
            cursor+=count
            if not rows:
                if not prefill:
                    packets[request_id]=ActionPacket(state,[],[],None,None,
                        source_positions=sources,history_begin=history_begin,candidate_count=count)
                continue
            with owner.stage('metadata.window_segments', active_tokens=local_begin+count,
                             sample_rows=len(rows)) if owner else nullcontext():
                starts,counts,history=self._window_segments(authority,history_begin,local_begin,count)
            device=batch.input_ids.device
            if owner:
                owner = owner.child(active_tokens=local_begin+count, sample_rows=len(rows), segments=len(starts))
            upload = getattr(self, 'metadata_upload', None)
            staging = (upload.lease(starts,counts,rows,begin,local_begin,device,profile=owner)
                       if upload is not None else nullcontext(None))
            with staging as lease:
                if lease is None:
                    with owner.stage('metadata.starts_h2d', gpu=True, transfer_bytes=4*len(starts),
                                     direction='H2D', shape=[len(starts)], dtype='int32') if owner else nullcontext():
                        starts_gpu=torch.tensor(starts,device=device,dtype=torch.int32)
                    with owner.stage('metadata.counts_h2d', gpu=True, transfer_bytes=4*len(counts),
                                     direction='H2D', shape=[len(counts)], dtype='int32') if owner else nullcontext():
                        counts_gpu=torch.tensor(counts,device=device,dtype=torch.int32)
                    with owner.stage('metadata.rows_h2d', gpu=True, transfer_bytes=8*len(rows),
                                     direction='H2D', shape=[len(rows)], dtype='int64') if owner else nullcontext():
                        row_ids=torch.tensor([begin+r for r in rows],device=device,dtype=torch.long)
                    with owner.stage('metadata.positions_h2d', gpu=True, transfer_bytes=8*len(rows),
                                     direction='H2D', shape=[len(rows)], dtype='int64') if owner else nullcontext():
                        positions=torch.tensor([local_begin+r for r in rows],device=device,dtype=torch.int64)
                else:
                    starts_gpu,counts_gpu,row_ids,positions=lease.views
                native_index=self.runner.req_states.req_id_to_index[request_id]
                total=None
                for index,module in self.layers:
                    layer = owner.child(layer=module.attn.layer_name) if owner else None
                    group=self.groups[module.attn.layer_name]
                    table=self.runner.block_tables.block_tables[group].gpu[native_index]
                    key,_=native_vllm_fp8_views(module.attn.kv_cache,module.num_kv_heads,module.head_dim)
                    with layer.stage('score.layer_total', gpu=True) if layer else nullcontext():
                        with layer.stage('score.q_gather', gpu=True) if layer else nullcontext():
                            query = captured[index].query.index_select(0,row_ids)
                        mass=page_attention(query,positions,key,table,
                            module.rotary_emb.cos_sin_cache,module.attn._k_scale,starts_gpu,counts_gpu,
                            local_begin+count,scale=module.attn.impl.scale,
                            q_dtype=(self.prefill_q_dtypes if prefill else self.q_dtypes)[module.attn.layer_name],q_scale=module.attn._q_scale,
                            block_tokens=self.runtime.config.block_tokens, **({'profile': layer} if layer else {}),
                            **(dict(fused=True,complete_partition=True)
                               if getattr(self.options,'action_fused_scoring',False) else {}))
                        counts_by_backend=self.__dict__.setdefault('scoring_backend_counts',{})
                        backend=getattr(mass,'_action_scoring_backend','reference')
                        counts_by_backend[backend]=counts_by_backend.get(backend,0)+1
                    with layer.stage('score.layer_add', gpu=True) if layer else nullcontext():
                        total=mass if total is None else total+mass
                with owner.stage('score.layer_mean', gpu=True) if owner else nullcontext():
                    total=total/len(self.layers)
                pool=getattr(self,'download_pool',None)
                if pool is not None:
                    host,event=pool.copy(total)
                else:
                    with owner.stage('mass.pinned_allocate', shape=list(total.shape),
                                 allocation_bytes=total.numel()*total.element_size()) if owner else nullcontext():
                        host=torch.empty(total.shape,dtype=total.dtype,pin_memory=True)
                    with owner.stage('mass.d2h', gpu=True, transfer_bytes=total.numel()*total.element_size(),
                                 direction='D2H') if owner else nullcontext():
                        host.copy_(total,non_blocking=True)
                    event=torch.cuda.Event();event.record()
                if lease is not None:
                    # Covers every layer and recursive scoring chunk, not just
                    # the H2D. The runtime owns staging even if publication fails.
                    lease.complete(event)
                metadata=(history_begin+count,count,starts,local_begin) if prefill else None
                packets[request_id]=ActionPacket(state,rows,history,host,event,metadata,
                    sources,history_begin,count)
                self.candidate_samples+=len(rows)
        self.packets+=len(packets)
        self.capture_seconds+=time.perf_counter()-started
        return packets

    def _wait_packet(self, pending):
        if pending is None or pending.event is None:
            return
        waited = time.perf_counter()
        pending.event.synchronize()
        self.accept_event_wait_seconds = getattr(self, 'accept_event_wait_seconds', 0.) + time.perf_counter()-waited

    def _prefill_accept(self, history, pending, accepted):
        end,count,starts,local = pending.prefill
        started = time.perf_counter()
        try:
            return history.accept(end,count,accepted,pending.rows,starts,pending.history,local,pending.masses)
        finally:
            self.prefill_commit_host_seconds = getattr(self, 'prefill_commit_host_seconds', 0.) + time.perf_counter()-started
            self.prefill_commit_calls = getattr(self, 'prefill_commit_calls', 0) + 1
            backend = getattr(history, 'last_commit_backend', None)
            if backend is not None:
                key = 'prefill_commit_' + backend + '_calls'
                setattr(self, key, getattr(self, key, 0)+1)

    def _report_prefill(self, request_id, pending):
        history = pending.state.prefill
        with (self.runtime.output/'prefill_attention.jsonl').open('a') as stream:
            stream.write(json.dumps({'request_id':request_id,'end':pending.prefill[0],
                'query_samples':len(pending.rows),'samples':history.samples,
                'chunks':history.chunks,'pages':len(history.values)})+'\n')

    def prepare_accept(self, packet, request_id, accepted, *, graph, history_begin):
        """Reserve graph state before modifying any accepted feedback.

        Empty and partial-prefill callbacks still prepare an empty graph update,
        so their *accepted* logical boundary can finalize a partial source.
        """
        pending = packet.action_inputs.get(request_id)
        self._wait_packet(pending)
        output = prefill = None
        positions, masses, targets = (), (), ()
        kind = 'output'
        dense_output = None
        if pending is not None:
            if pending.history_begin is not None and pending.history_begin != history_begin:
                raise ValueError('Graph attention packet has stale source coordinates')
            if pending.prefill is not None:
                from copy import copy
                kind = 'prefill'
                original = pending.state.prefill
                staged = copy(original)
                committed = self._prefill_accept(staged, pending, accepted)
                prefill = original, staged, committed
                # Preserve the existing full-chunk rule. An accepted partial
                # prefill advances graph progress with no observed rows.
                end,count,_,_ = pending.prefill
                excluded = getattr(pending.state, 'prefill_replay', None)
                replay = excluded is not None and history_begin < excluded[1] and end > excluded[0]
                if accepted == count and not replay and pending.rows:
                    if len(pending.source_positions) != len(pending.rows):
                        raise ValueError('Graph prefill packet is missing source positions')
                    if any(position != history_begin+row for row,position in
                           zip(pending.rows,pending.source_positions)):
                        raise ValueError('Graph prefill source differs from consumed Q position')
                    positions = pending.source_positions
                    # The captured CPU tensor already owns this storage. Avoid
                    # expanding a large prefill matrix into Python floats
                    # outside graph staging admission.
                    masses = pending.masses.numpy() if hasattr(pending.masses, 'numpy') else pending.masses
                    targets = pending.history
            else:
                parser = pending.state.parser
                if not isinstance(parser, _AcceptanceParser):
                    raise RuntimeError('Graph output requires transactional action parser')
                mask = parser.preview(accepted)
                values = pending.masses.numpy() if hasattr(pending.masses, 'numpy') else pending.masses
                values = [] if values is None else values
                if len(pending.rows) != len(values) or len(pending.source_positions) != len(pending.rows):
                    raise ValueError('Graph output packet is missing attention rows or source positions')
                if any(not isinstance(row,int) or not 0 <= row < parser.count for row in pending.rows):
                    raise ValueError('Graph attention candidate row is out of bounds')
                if mask & ~sum(1 << row for row in set(pending.rows)):
                    raise ValueError('Graph output packet omitted an accepted sampled row')
                native_decode = getattr(getattr(self,'options',pending.state.options),'action_native_decode_commit',False)
                if native_decode:
                    selected = tuple(i for i,row in enumerate(pending.rows) if (mask >> row) & 1)
                    if selected:
                        import numpy as np
                        values = np.asarray(values).reshape(len(pending.rows),len(pending.history))
                        positions = tuple(pending.source_positions[i] for i in selected)
                        if any(pending.source_positions[i] != history_begin+pending.rows[i] for i in selected):
                            raise ValueError('Graph output source differs from consumed Q position')
                        dense_output = selected, (pending.state.mass,pending.state.samples,
                            getattr(pending.state.options,'action_commit_max_bytes',16777216)), mask
                        masses, targets = values, pending.history
                    else:
                        # Most decode frames have no accepted sampled row.
                        # Advancing their graph/parser boundary needs no copy
                        # or validation of the immutable published score map.
                        output=mask,pending.state.mass,pending.state.samples
                updated, samples = (None,None) if native_decode else (dict(pending.state.mass),pending.state.samples)
                kept_positions, kept_masses = [], []
                for row, position, row_mass in (() if native_decode else zip(pending.rows,pending.source_positions,values)):
                    if not (mask >> row) & 1:
                        continue
                    if (len(row_mass) != len(pending.history) or
                            any(not math.isfinite(v) or v < 0 for v in row_mass) or
                            abs(sum(float(v) for v in row_mass)-1) > .005):
                        raise RuntimeError('Action attention mass is invalid or does not sum to one')
                    if position != history_begin + row:
                        raise ValueError('Graph output source differs from consumed Q position')
                    for key,value in zip(pending.history,row_mass):
                        # Tensor.tolist(), used by the graph-off path, creates
                        # Python doubles. Preserve that exact legacy arithmetic
                        # in collect mode instead of NumPy float32 promotion.
                        value = float(value)
                        if value:
                            updated[key] = updated.get(key,0.) + value
                    samples += 1
                    kept_positions.append(position)
                    kept_masses.append(row_mass)
                if not native_decode:
                    output = mask, updated, samples
                    positions, masses, targets = kept_positions, kept_masses, pending.history
        started = time.perf_counter()
        try:
            graph_begin = min(history_begin, graph.accepted_through) if not accepted else history_begin
            update = graph.prepare_accept(graph_begin, graph_begin+accepted,
                source_positions=positions,target_starts=targets,mass_rows=masses,kind=kind,
                **({} if dense_output is None else dict(row_indices=dense_output[0],output_state=dense_output[1])))
            if dense_output is not None:
                mass,samples = getattr(update,'output_state',dense_output[1][:2])
                output = dense_output[2],mass,samples
        finally:
            self.graph_prepare_seconds = getattr(self,'graph_prepare_seconds',0.) + time.perf_counter()-started
            self.graph_prepare_calls = getattr(self,'graph_prepare_calls',0) + 1
        return _PreparedActionAcceptance(self,packet,request_id,pending,update,
            accepted=accepted,output=output,prefill=prefill)

    def accept(self, packet, request_id, accepted):
        pending=packet.action_inputs.pop(request_id,None)
        if pending is None:return
        self._wait_packet(pending)
        if pending.prefill is not None:
            history=pending.state.prefill
            committed = self._prefill_accept(history,pending,accepted)
            if committed:
                self._report_prefill(request_id,pending)
            return
        pending.state.accept(accepted,pending.rows,pending.history,
            pending.masses if getattr(pending.state.options,'action_native_decode_commit',False) else
            pending.masses.tolist() if pending.masses is not None else [])
        backend = 'native' if getattr(pending.state.options,'action_native_decode_commit',False) else 'python'
        key = 'decode_commit_'+backend+'_calls'
        setattr(self,key,getattr(self,key,0)+1)

    def finish(self, request_id, authority):
        profiler = getattr(self, 'profile', None)
        if profiler is not None:
            # Publication already used its original accepted-score fence.
            # Polling here only recovers completed diagnostic events.
            profiler.flush()
        state=getattr(authority,'action_history',None)
        if state is None:return None
        report=state.finish(authority)
        if state.prefill is not None:
            report.update(prefill_samples=state.prefill.samples,prefill_chunks=state.prefill.chunks,
                          prefill_end=state.prefill.end,prefill_pages=len(state.prefill.values))
        report.update(request_id=request_id,candidate_samples_total=self.candidate_samples,
                      capture_host_seconds_total=self.capture_seconds,
                      metadata_early_batches=getattr(self,'metadata_early_batches',0),
                      metadata_fallback_batches=getattr(self,'metadata_fallback_batches',0),
                      metadata_wait_seconds=getattr(self,'metadata_wait_seconds',0.0))
        report.update(prefill_commit_host_seconds=getattr(self,'prefill_commit_host_seconds',0.),
                      prefill_commit_calls=getattr(self,'prefill_commit_calls',0),
                      prefill_commit_native_calls=getattr(self,'prefill_commit_native_calls',0),
                      prefill_commit_python_calls=getattr(self,'prefill_commit_python_calls',0),
                      accept_event_wait_seconds=getattr(self,'accept_event_wait_seconds',0.))
        report.update(decode_commit_native_calls=getattr(self,'decode_commit_native_calls',0),
                      decode_commit_python_calls=getattr(self,'decode_commit_python_calls',0))
        if getattr(self,'download_pool',None) is not None:
            report['download_pool_total']=self.download_pool.snapshot()
        report['scoring_backend_counts']=dict(getattr(self,'scoring_backend_counts',{}))
        graph = getattr(authority,'block_graph',None)
        if graph is not None:
            rows = graph.rows
            partial = graph.partial
            report['block_graph'] = dict(graph.stats(),
                prefill_samples=sum(row.prefill_samples for row in rows) +
                    (partial.prefill_samples if partial is not None else 0),
                output_samples=sum(row.output_samples for row in rows) +
                    (partial.output_samples if partial is not None else 0))
            report.update(graph_prepare_seconds=getattr(self,'graph_prepare_seconds',0.),
                graph_prepare_calls=getattr(self,'graph_prepare_calls',0),
                graph_commit_seconds=getattr(self,'graph_commit_seconds',0.),
                graph_commit_calls=getattr(self,'graph_commit_calls',0))
        if getattr(self, 'metadata_upload', None) is not None:
            # Diagnostic snapshot only. Publication/cancellation never retires
            # staging: the engine thread polls its consumer completion events.
            report['metadata_upload_total'] = self.metadata_upload.snapshot()
        with (self.runtime.output/'action_attention.jsonl').open('a') as stream:
            stream.write(json.dumps(report)+'\n')
        return report
