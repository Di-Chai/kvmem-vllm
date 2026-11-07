"""Opt-in experiment: independent accepted prefill attention per owner/layer.

The native shared working frame represents layer zero only. Other layers keep
their own logical block maps; newly appended records still share original token
coordinates. No cold blocks enter a pressure candidate set. Final Query keeps
the existing causal per-layer Mean-K implementation.
"""
from copy import copy
import math
import os

import torch

from .action_attention import ActionRuntime, window_segments
from .prefill_attention import PrefillHistory
from .original_rope import enabled as original_rope


class _LayerwiseAcceptance:
    """Publish per-layer histories only with the accepted graph transaction."""
    def __init__(self, transaction, controller, staged):
        self.transaction, self.controller, self.staged = transaction, controller, staged

    def __getattr__(self, name):
        return getattr(self.transaction, name)

    def commit(self):
        self.transaction.commit()
        self.controller.commit_accept(self.staged)
        self.staged = None

    def abort(self):
        self.transaction.abort()
        self.staged = None


def extend_records(records, origin, end, block_tokens=32):
    """Extend a frozen layer map with accepted ordinary input, without handles."""
    if end < origin:
        raise ValueError('Layer map cannot move back across its pressure boundary')
    result = list(records)
    cursor = sum(r[3] for r in result)
    logical = origin
    while logical < end:
        n = min(block_tokens-logical % block_tokens, end-logical)
        if (result and result[-1][2]+result[-1][3] == logical and
                result[-1][4]+result[-1][3] == cursor and
                result[-1][2]//block_tokens == logical//block_tokens):
            slot, gen, begin, size, pos = result[-1]
            result[-1] = slot, gen, begin, size+n, pos
        else:
            result.append((-1, -1, logical, n, cursor))
        cursor += n
        logical += n
    return result


class LayerwisePrefillPressure:
    def __init__(self, query):
        self.query, self.runner, self.names = query, query.runner, query.names
        if not query.optimized:
            raise ValueError('Layerwise pressure requires optimized layerwise Query execution')
        if not query.native_batches and not self.runner.model_config.enforce_eager:
            raise ValueError('Compiled batches require the native layerwise Query dispatcher')
        self.maps = {}
        self.histories = {}
        self.rolling_decode = os.getenv('KVMEM_LAYERWISE_ROLLING_DECODE') == '1'
        self.decode_histories = {}
        self.options = None
        self.single_assembly = os.getenv('KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY') == '1'
        self.fast_metadata = os.getenv('KVMEM_LAYERWISE_PRESSURE_FAST_METADATA') == '1'
        self.buffered_history = os.getenv('KVMEM_LAYERWISE_BUFFERED_HISTORY') == '1'
        batch_accept = os.getenv('KVMEM_LAYERWISE_PRESSURE_BATCH_ACCEPT', '0')
        if batch_accept not in ('0', '1'):
            raise ValueError('Layerwise pressure batch acceptance must be 0 or 1')
        self.batch_accept = batch_accept == '1'
        if self.batch_accept:
            if not self.buffered_history:
                raise ValueError('Batched layer acceptance requires native buffered histories')
            from .buffered_prefill_attention import require_batch_accept
            require_batch_accept()
        # New execution candidates remain opt-in until full throughput gates
        # pass; a successful component test alone must not change defaults.
        self.prepared_reads = os.getenv('KVMEM_LAYERWISE_PRESSURE_PREPARED_READS') == '1'
        self.post_forward_reap = os.getenv('KVMEM_LAYERWISE_PRESSURE_POST_FORWARD_REAP') == '1'
        self.submit_on_prepare = os.getenv('KVMEM_LAYERWISE_PRESSURE_SUBMIT_ON_PREPARE') == '1'
        self.reuse_layer_views = os.getenv('KVMEM_LAYERWISE_PRESSURE_REUSE_LAYER_VIEWS') == '1'
        self.resident_values = os.getenv('KVMEM_LAYERWISE_PRESSURE_RESIDENT_VALUES') == '1'
        batch_selection = os.getenv('KVMEM_LAYERWISE_PRESSURE_BATCH_SELECTION', '0')
        if batch_selection not in ('0', '1'):
            raise ValueError('Layerwise pressure batch selection must be 0 or 1')
        self.batch_selection = batch_selection == '1'
        pipeline = os.getenv('KVMEM_LAYERWISE_PRESSURE_PIPELINE', '0')
        if pipeline not in ('0', '1'):
            raise ValueError('Layerwise pressure pipeline must be 0 or 1')
        handoff = os.getenv('KVMEM_LAYERWISE_PRESSURE_PIPELINE_HANDOFF', '0')
        if handoff not in ('0', '1'):
            raise ValueError('Layerwise pressure pipeline handoff must be 0 or 1')
        self.pipeline_handoff = handoff == '1'
        prepared_commit = os.getenv('KVMEM_LAYERWISE_PRESSURE_PREPARED_COMMIT', '0')
        preforward = os.getenv('KVMEM_LAYERWISE_PRESSURE_PIPELINE_PREFORWARD', '0')
        progress = os.getenv('KVMEM_LAYERWISE_PRESSURE_PIPELINE_PROGRESS', '0')
        if prepared_commit not in ('0', '1') or preforward not in ('0', '1') or progress not in ('0', '1'):
            raise ValueError('Prepared commit and pre-forward pressure flags must be 0 or 1')
        self.prepared_commit = prepared_commit == '1'
        self.pipeline_preforward = preforward == '1'
        self.pipeline_progress = progress == '1'
        if self.prepared_commit and not self.single_assembly:
            raise ValueError('Prepared pressure commit requires the single-assembly path')
        if self.pipeline_preforward and not self.pipeline_handoff:
            raise ValueError('Pre-forward pressure preparation requires guarded early handoff')
        if self.pipeline_progress and (not self.pipeline_handoff or self.pipeline_preforward):
            raise ValueError('Completion progress requires the original bounded handoff without early enqueue')
        if self.pipeline_handoff and (pipeline != '1' or not self.post_forward_reap or
                                      not self.submit_on_prepare):
            raise ValueError('Early handoff requires pipeline and original pressure prepare/reap paths')
        self.pipeline = None
        if pipeline == '1':
            if (not self.single_assembly or not self.prepared_reads or not self.reuse_layer_views or
                    not query.paired_reads or not query.native_batches or
                    os.getenv('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING', '0') != '0' or
                    self.resident_values):
                raise ValueError('Pipeline requires the qualified single-assembly immutable-reader path')
            from .pressure_pipeline import PressurePipeline
            self.pipeline = PressurePipeline(self)
        # Opt-in concurrent prefill-pressure producers. Each lane has its own
        # worker thread, CUDA stream, staging scratch and transfer credit; the
        # engine thread still owns every commit. Selection and bytes are the
        # same per owner; only how many owners assemble at once changes.
        lanes = os.getenv('KVMEM_LAYERWISE_PRESSURE_LANES', '1')
        if lanes not in ('1', '2'):
            raise ValueError('Layerwise pressure lanes must be 1 or 2')
        self.lanes = int(lanes)
        if self.lanes > 1 and (self.pipeline is None or not self.pipeline_handoff or
                               self.batch_selection or self.resident_values or self.parent_staging_requested()):
            raise ValueError('Concurrent pressure lanes require the guarded pipeline handoff '
                             'without batch selection, resident values or parent staging')
        self.install_acceptance()

    @staticmethod
    def parent_staging_requested():
        return os.getenv('KVMEM_LAYERWISE_PRESSURE_PARENT_STAGING', '0') != '0'

    def new_history(self, options):
        if getattr(self, 'buffered_history', False):
            from .buffered_prefill_attention import BufferedPrefillHistory
            return BufferedPrefillHistory(options)
        return PrefillHistory(options)

    def history_execution(self, history):
        backend = getattr(history, 'storage_backend', 'native_dictionary')
        commits = getattr(history, 'successful_buffered_commits', 0)
        restores = getattr(history, 'successful_buffered_restores', 0)
        if getattr(self, 'buffered_history', False) and (backend != 'native_buffered' or commits + restores <= 0):
            raise RuntimeError('Layer pressure did not execute the configured native history backend')
        result = dict(attention_state_backend=backend, buffered_commits=commits, buffered_restores=restores)
        if getattr(self, 'batch_accept', False):
            result['buffered_batch_commits'] = history.successful_buffered_batch_commits
        return result

    def _history(self, rid, options):
        if self.options is None:
            self.options = options
            # Memory suites retain their original graph metadata for the
            # initial Query path. Pressure itself still uses only the supplied
            # per-layer cumulative attention, never graph propagation.
            if (options.pressure_policy != 'attention' or
                    options.guided_query_tokens or options.query_history or
                    not options.prefill_attention):
                raise ValueError('Layerwise pressure requires cumulative attention without guidance/history fusion')
        if rid not in self.histories:
            self.histories[rid] = [self.new_history(options) for _ in self.names]
        return self.histories[rid]

    def scoring_layers(self, action):
        # CPU history also archives MTP's draft-attention layer. It has no
        # public target-layer Query and must never join this score tensor.
        if tuple(action.runtime.names[:len(self.names)]) != self.names:
            raise RuntimeError('Target attention layers changed their archive order')
        return [(i, action.runtime.layers[i]) for i in range(len(self.names))]

    def records(self, rid, authority, history_begin, local_begin, layer):
        frame = self.maps.get(rid)
        if frame is None:
            result = list(authority.memory._store.working_records())
        else:
            origin, layers = frame
            result = extend_records(layers[layer], origin, history_begin)
        if sum(r[3] for r in result) != local_begin:
            raise RuntimeError('Layer-specific physical history length differs from the native frame')
        return result

    def attach_packet(self, pending, rid, authority, starts, counts, masses):
        if len(masses.shape) != 3 or masses.shape[0] != len(self.names):
            raise RuntimeError('Missing independent prefill scores for a full-attention layer')
        self._history(rid, pending.state.options)
        if pending.prefill is None:
            if not self.rolling_decode:
                raise RuntimeError('Decode feedback requires explicit layerwise rolling mode')
            from .layerwise_decode import attach_packet
            return attach_packet(self, pending, rid, authority, starts, counts, masses)
        if self.fast_metadata and rid in self.maps:
            from .layerwise_pressure_efficiency import packet_histories
            histories = packet_histories(self.maps.get(rid), pending, starts, counts)
            pending.layerwise_pressure = (self, rid, histories, masses)
            return
        end, count, _, local = pending.prefill
        histories = []
        for layer in range(len(self.names)):
            if rid not in self.maps:
                # The original immutable pipeline plan can include a preceding
                # accepted chunk whose CPU archive publication is still pending.
                # All layers share that plan before the first pressure event.
                histories.append(list(pending.history))
                continue
            records = self.records(rid, authority, end-count, local, layer)
            ls, lc, lh = window_segments(records, end-count, local, count, 32)
            if list(ls) != list(starts) or list(lc) != list(counts):
                raise RuntimeError('Layer map changed causal segment geometry')
            histories.append(lh)
        if histories[0] != list(pending.history):
            raise RuntimeError('Shared metadata no longer describes physical layer zero')
        pending.layerwise_pressure = (self, rid, histories, masses)

    def prepare_accept(self, pending, accepted):
        if getattr(pending, 'layerwise_decode', None) is not None:
            from .layerwise_decode import prepare_accept
            return prepare_accept(self, pending, accepted)
        marker = getattr(pending, 'layerwise_pressure', None)
        if marker is None:
            return None
        owner, rid, logical, masses = marker
        if owner is not self:
            raise RuntimeError('Foreign layerwise attention packet')
        end, count, starts, local = pending.prefill
        # Caller waits for the original score-download fence before this reads.
        staged = [copy(history) for history in self.histories[rid]]
        if getattr(self, 'batch_accept', False):
            from .buffered_prefill_attention import accept_buffered_batch
            accepted_flags = accept_buffered_batch(staged, end, count, accepted, pending.rows,
                starts, logical, local, masses)
        else:
            accepted_flags = [history.accept(end, count, accepted, pending.rows,
                starts, logical[layer], local, masses[layer]) for layer, history in enumerate(staged)]
        if len(set(accepted_flags)) != 1:
            raise RuntimeError('Layer attention histories disagree on accepted sampling boundary')
        return rid, staged

    def commit_accept(self, prepared):
        if prepared is not None:
            rid, histories = prepared[:2]
            target = self.decode_histories if len(prepared) == 3 else self.histories
            target[rid] = histories

    def prepare(self, refresh, job):
        from .request_plan import RequestPlan, selection_config
        rid, authority = job.command.request_id, job.authority
        decode = job.command.pressure_stage == 'decode' and self.rolling_decode
        if job.command.pressure_stage != 'prefill' and not decode:
            raise RuntimeError('This short-output experiment cannot reinterpret decode pressure')
        authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
            selection='pressure', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=32, gen_budget=cfg.gen_budget,
            mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        required = set(fit['hard_blocks'])
        records = sorted(authority.records.values(), key=lambda r:r.history_begin)
        # Each handle.slot read crosses pybind; read once per pressure, not per layer.
        record_slots = [r.handle.slot for r in records]
        record_positions = {slot:i for i,slot in enumerate(record_slots)}
        if len(record_positions) != len(records):
            raise RuntimeError('Pressure archive contains duplicate physical slots')
        by_begin = {r.history_begin:(r, s) for r, s in zip(records, record_slots)}
        mandatory = tuple(s for r, s in zip(records, record_slots) if r.history_begin//32 in required)
        mandatory_set = set(mandatory)
        if {r.history_begin//32 for r, s in zip(records, record_slots) if s in mandatory_set} != required:
            raise RuntimeError('Missing original mandatory pressure blocks')
        global_history = authority.action_history.prefill
        histories = (self.decode_histories if decode else self.histories).get(rid)
        if histories is None or (not decode and global_history.snapshot() != histories[0].snapshot()):
            raise RuntimeError('Accepted layer-zero attention differs from the shared control state')
        current = list(authority.memory._store.working_records())
        local = sum(r[3] for r in current)
        chosen, evidence, source_maps = [], [], []
        prepare_commit = not decode and getattr(self, 'prepared_commit', False)
        selected_slot_groups = []
        capture_values = not decode and getattr(self, 'resident_values', False)
        batched = None
        if not decode and getattr(self, 'batch_selection', False):
            from .pressure_selection_batch import prepare_pressure_batch
            batched = prepare_pressure_batch(self, job, histories, local, by_begin,
                                             mandatory, cfg.select_budget)
        for layer, history in enumerate(histories):
            if batched is None:
                mapping = self.records(rid, authority, job.history, local, layer)
                candidates, candidate_slots = [], []
                for _, _, begin, n, _ in mapping:
                    entry = by_begin.get(begin)
                    if entry is None or entry[0].count != n:
                        raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                    candidates.append(entry[0])
                    candidate_slots.append(entry[1])
                scores = history.scores()
                supplied = [(s, scores.get(r.history_begin, 0.)) for r, s in zip(candidates, candidate_slots)]
                if any(not math.isfinite(s) or s < 0 for _,s in supplied):
                    raise RuntimeError('Invalid layer cumulative attention score')
                if not any(s > 0 for _,s in supplied):
                    raise RuntimeError('Layer pressure has no accepted attention; refusing a recency fallback')
                slots = set(authority.memory.preview_selection({}, mandatory, action=False,
                    pressure_scores=supplied, max_tokens=cfg.select_budget))
            else:
                mapping, candidates, candidate_slots, slots = batched[layer]
            if capture_values:
                source_maps.append(tuple(mapping))
            allowed = set(candidate_slots) | mandatory_set
            if not mandatory_set <= slots or not slots <= allowed or 32*len(slots) > cfg.select_budget:
                raise RuntimeError('Layer pressure violated candidates, mandatory blocks or Active budget')
            # Preserve the original archive order without scanning cold records
            # again for every layer. The native preview still owns all ranking.
            ordered_slots = sorted(slots, key=record_positions.__getitem__)
            picked = [records[record_positions[s]] for s in ordered_slots]
            if prepare_commit:
                selected_slot_groups.append(ordered_slots)
            chosen.append(picked)
            evidence.append(dict(layer=layer, layer_name=self.names[layer],
                candidate_block_count=len(candidates), candidate_starts=[r.history_begin for r in candidates],
                selected_starts=[r.history_begin for r in picked], selected_tokens=sum(r.count for r in picked),
                charged_tokens=32*len(picked), accepted_score_end=history.end,
                query_samples=history.samples, sampled_chunks=history.chunks,
                actual_method='accumulated_attention_per_layer', fallback_reason=None,
                **({'pressure_selection_backend': 'native_batch_snapshot'} if batched is not None else {}),
                **(dict(attention_state_backend='native_decode',
                        score_scope='all_accepted_including_thinking') if decode else
                   self.history_execution(history))))
        lengths = {sum(r.count for r in group) for group in chosen}
        partial = [{(r.history_begin,r.count) for r in group if r.count != 32} for group in chosen]
        if len(lengths) != 1 or any(p != partial[0] for p in partial):
            raise RuntimeError('Layer pressure cannot change native causal length or partial-block geometry')
        job.layerwise_pressure_plan = chosen, evidence
        if prepare_commit:
            from .pressure_commit import prepare_commit as build_commit
            job.layerwise_prepared_commit = build_commit(
                job, chosen, records, record_slots, selected_slot_groups)
        if capture_values:
            job.layerwise_pressure_value_sources = (rid, job.history, local, tuple(source_maps))
        if decode:
            if any(history.end > job.history for history in histories):
                raise RuntimeError('Rolling pressure contains future or rejected Query feedback')
            parser = getattr(authority.action_history, 'parser', None)
            job.layerwise_decode_thinking = bool(parser.state[0]) if parser is not None else None
            # The final Query can select a new layer-zero set while the native
            # host frame still describes its global preselection. Admit exactly
            # the independent preview; never rank stale shared-frame identities.
            job.layerwise_decode_selection = tuple(r.handle.slot for r in chosen[0])

    def committed(self, job):
        rid, authority = job.command.request_id, job.authority
        chosen, evidence = job.layerwise_pressure_plan
        actual = {r[2] for r in authority.memory._store.working_records()}
        prepared = None
        if job.command.pressure_stage == 'prefill' and getattr(self, 'prepared_commit', False):
            prepared = getattr(job, 'layerwise_prepared_commit', None)
            if prepared is None:
                raise RuntimeError('Missing the configured prepared pressure maps')
            prepared.validate(job)
            if not (getattr(job, 'layerwise_single_assembly_done', False) or
                    getattr(job, 'layerwise_pipeline_published', False)):
                raise RuntimeError('Prepared maps need completed assembly or guarded pipeline publication')
        if actual != (prepared.layer_zero_starts if prepared else {r.history_begin for r in chosen[0]}):
            raise RuntimeError('Native pressure selection disagrees with the layer-zero preview')
        state = dict(authority=authority, request_id=rid, selected_tokens=job.selected_tokens)
        maps = prepared.maps if prepared is not None else []
        for layer, picked in (() if prepared is not None else enumerate(chosen)):
            if layer and not (getattr(job, 'layerwise_single_assembly_done', False) or
                              getattr(job, 'layerwise_pipeline_published', False)):
                self.query.assemble_layer_pinned(state, layer, picked)
            cursor = 0
            mapping = []
            for r in picked:
                mapping.append((r.handle.slot,r.handle.generation,r.history_begin,r.count,cursor))
                cursor += r.count
            if cursor != job.selected_tokens:
                raise RuntimeError('Layer pressure assembly changed the unchanged native token extent')
            maps.append(tuple(mapping))
        # Single-pass assembly completes on the existing refresh fence before
        # the native transaction commits. It needs no second main-thread wait.
        if not (getattr(job, 'layerwise_single_assembly_done', False) or
                getattr(job, 'layerwise_pipeline_published', False)):
            torch.cuda.current_stream(self.runner.device).synchronize()
        self.maps[rid] = job.history, tuple(maps)
        self.query.emit('pressure_selected', request_id=rid, accepted_history=job.history,
            actual_method='accumulated_attention_per_layer', event_type=job.command.pressure_stage+'_pressure',
            configured_method='accumulated_attention_per_layer',
            query_source=('accepted_output_including_thinking' if job.command.pressure_stage == 'decode'
                          else 'accepted_public_prefill'),
            thinking_at_refresh=getattr(job, 'layerwise_decode_thinking', None),
            selected_tokens=job.selected_tokens, layers=evidence, fallback_reason=None)
        if original_rope():
            self.query.emit('pressure_rope', request_id=rid, accepted_history=job.history,
                rope_position_policy='original', key_source='record.history_begin',
                score_query_source='accepted_original_position', physical_slots='dense')

    @staticmethod
    def install_acceptance():
        if getattr(ActionRuntime, '_layerwise_pressure_accept_installed', False):
            return
        original, finish = ActionRuntime.accept, ActionRuntime.finish
        graph_prepare = ActionRuntime.prepare_accept
        def accept(runtime, packet, rid, accepted):
            pending = packet.action_inputs.get(rid)
            controller = getattr(runtime.runner, '_kvmem_layerwise_pressure', None)
            prepared = None
            if controller is not None and (getattr(pending, 'layerwise_pressure', None) is not None or
                                           getattr(pending, 'layerwise_decode', None) is not None):
                runtime._wait_packet(pending)
                prepared = controller.prepare_accept(pending, accepted)
            result = original(runtime, packet, rid, accepted)
            if controller is not None:
                controller.commit_accept(prepared)
            return result
        def prepare_graph(runtime, packet, rid, accepted, **kwargs):
            pending = packet.action_inputs.get(rid)
            controller = getattr(getattr(runtime, 'runner', None), '_kvmem_layerwise_pressure', None)
            staged = None
            if controller is not None and (getattr(pending, 'layerwise_pressure', None) is not None or
                                           getattr(pending, 'layerwise_decode', None) is not None):
                runtime._wait_packet(pending)
                staged = controller.prepare_accept(pending, accepted)
            transaction = graph_prepare(runtime, packet, rid, accepted, **kwargs)
            return (_LayerwiseAcceptance(transaction, controller, staged)
                    if staged is not None else transaction)
        def finished(runtime, rid, authority):
            result = finish(runtime, rid, authority)
            controller = getattr(runtime.runner, '_kvmem_layerwise_pressure', None)
            if controller is not None:
                controller.maps.pop(rid, None)
                controller.histories.pop(rid, None)
                controller.decode_histories.pop(rid, None)
            return result
        ActionRuntime.accept, ActionRuntime.finish = accept, finished
        ActionRuntime.prepare_accept = prepare_graph
        ActionRuntime._layerwise_pressure_accept_installed = True
