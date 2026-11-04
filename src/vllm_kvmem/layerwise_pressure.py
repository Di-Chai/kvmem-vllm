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
        self.options = None
        self.single_assembly = os.getenv('KVMEM_LAYERWISE_PRESSURE_SINGLE_ASSEMBLY') == '1'
        self.fast_metadata = os.getenv('KVMEM_LAYERWISE_PRESSURE_FAST_METADATA') == '1'
        self.buffered_history = os.getenv('KVMEM_LAYERWISE_BUFFERED_HISTORY') == '1'
        self.install_acceptance()

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
        return dict(attention_state_backend=backend, buffered_commits=commits, buffered_restores=restores)

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
        marker = getattr(pending, 'layerwise_pressure', None)
        if marker is None:
            return None
        owner, rid, logical, masses = marker
        if owner is not self:
            raise RuntimeError('Foreign layerwise attention packet')
        end, count, starts, local = pending.prefill
        # Caller waits for the original score-download fence before this reads.
        staged = [copy(history) for history in self.histories[rid]]
        accepted_flags = [history.accept(end, count, accepted, pending.rows,
            starts, logical[layer], local, masses[layer]) for layer, history in enumerate(staged)]
        if len(set(accepted_flags)) != 1:
            raise RuntimeError('Layer attention histories disagree on accepted sampling boundary')
        return rid, staged

    def commit_accept(self, prepared):
        if prepared is not None:
            rid, histories = prepared
            self.histories[rid] = histories

    def prepare(self, refresh, job):
        from .request_plan import RequestPlan, selection_config
        rid, authority = job.command.request_id, job.authority
        if job.command.pressure_stage != 'prefill':
            raise RuntimeError('This short-output experiment cannot reinterpret decode pressure')
        authority.publish_sealed(lambda: authority.archive.wait_for_capacity(.05))
        cfg = selection_config(refresh.options.memory, job.command.groups[0][1].window_tokens,
            selection='pressure', pressure_policy='attention')
        fit = RequestPlan(None, block_tokens=32, gen_budget=cfg.gen_budget,
            mandatory=job.command.mandatory_spans).mandatory_fit(job.history, cfg)
        required = set(fit['hard_blocks'])
        records = sorted(authority.records.values(), key=lambda r:r.history_begin)
        by_begin = {r.history_begin:r for r in records}
        mandatory = tuple(r.handle.slot for r in records if r.history_begin//32 in required)
        if {by_begin[r.history_begin].history_begin//32 for r in records if r.handle.slot in mandatory} != required:
            raise RuntimeError('Missing original mandatory pressure blocks')
        global_history = authority.action_history.prefill
        histories = self.histories.get(rid)
        if histories is None or global_history.snapshot() != histories[0].snapshot():
            raise RuntimeError('Accepted layer-zero attention differs from the shared control state')
        current = list(authority.memory._store.working_records())
        local = sum(r[3] for r in current)
        chosen, evidence = [], []
        for layer, history in enumerate(histories):
            mapping = self.records(rid, authority, job.history, local, layer)
            candidates = []
            for _, _, begin, n, _ in mapping:
                r = by_begin.get(begin)
                if r is None or r.count != n:
                    raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
                candidates.append(r)
            scores = history.scores()
            supplied = [(r.handle.slot, scores.get(r.history_begin, 0.)) for r in candidates]
            if any(not math.isfinite(s) or s < 0 for _,s in supplied):
                raise RuntimeError('Invalid layer cumulative attention score')
            if not any(s > 0 for _,s in supplied):
                raise RuntimeError('Layer pressure has no accepted attention; refusing a recency fallback')
            slots = set(authority.memory.preview_selection({}, mandatory, action=False,
                pressure_scores=supplied, max_tokens=cfg.select_budget))
            allowed = {r.handle.slot for r in candidates} | set(mandatory)
            if not set(mandatory) <= slots or not slots <= allowed or 32*len(slots) > cfg.select_budget:
                raise RuntimeError('Layer pressure violated candidates, mandatory blocks or Active budget')
            picked = [r for r in records if r.handle.slot in slots]
            chosen.append(picked)
            evidence.append(dict(layer=layer, layer_name=self.names[layer],
                candidate_block_count=len(candidates), candidate_starts=[r.history_begin for r in candidates],
                selected_starts=[r.history_begin for r in picked], selected_tokens=sum(r.count for r in picked),
                charged_tokens=32*len(picked), accepted_score_end=history.end,
                query_samples=history.samples, sampled_chunks=history.chunks,
                actual_method='accumulated_attention_per_layer', fallback_reason=None,
                **self.history_execution(history)))
        lengths = {sum(r.count for r in group) for group in chosen}
        partial = [{(r.history_begin,r.count) for r in group if r.count != 32} for group in chosen]
        if len(lengths) != 1 or any(p != partial[0] for p in partial):
            raise RuntimeError('Layer pressure cannot change native causal length or partial-block geometry')
        job.layerwise_pressure_plan = chosen, evidence

    def committed(self, job):
        rid, authority = job.command.request_id, job.authority
        chosen, evidence = job.layerwise_pressure_plan
        actual = {r[2] for r in authority.memory._store.working_records()}
        if actual != {r.history_begin for r in chosen[0]}:
            raise RuntimeError('Native pressure selection disagrees with the layer-zero preview')
        state = dict(authority=authority, request_id=rid, selected_tokens=job.selected_tokens)
        maps = []
        for layer, picked in enumerate(chosen):
            if layer and not getattr(job, 'layerwise_single_assembly_done', False):
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
        if not getattr(job, 'layerwise_single_assembly_done', False):
            torch.cuda.current_stream(self.runner.device).synchronize()
        self.maps[rid] = job.history, tuple(maps)
        self.query.emit('pressure_selected', request_id=rid, accepted_history=job.history,
            actual_method='accumulated_attention_per_layer', event_type='prefill_pressure',
            configured_method='accumulated_attention_per_layer', query_source='accepted_public_prefill',
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
            if controller is not None and getattr(pending, 'layerwise_pressure', None) is not None:
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
            if controller is not None and getattr(pending, 'layerwise_pressure', None) is not None:
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
            return result
        ActionRuntime.accept, ActionRuntime.finish = accept, finished
        ActionRuntime.prepare_accept = prepare_graph
        ActionRuntime._layerwise_pressure_accept_installed = True
