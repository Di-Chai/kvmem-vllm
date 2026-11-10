"""CPU identity/state snapshots for frozen P-prefix layerwise pressure reuse.

This module does not publish a restore or write GPU pages. The session owner
must rebuild the prepared per-layer records and wait for its completion fence
before installing the maps/histories or admitting a public Query.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
import time

from .prefill_attention import PrefillHistory


def sessions_enabled():
    """KVMEM_LAYERWISE_SESSIONS: per-layer checkpoints of multi-turn sessions.

    Without it only frozen pre-Query P prefixes are captured (MemoryAgentBench).
    With it, non-frozen P prefixes and terminal M checkpoints also carry every
    layer's map, prefill attention history and (rolling) decode history.
    """
    value = os.getenv('KVMEM_LAYERWISE_SESSIONS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_LAYERWISE_SESSIONS must be 0 or 1')
    return value == '1'


@dataclass(frozen=True)
class LayerwisePrefix:
    position: int
    names: tuple
    maps: tuple
    histories: tuple
    # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: native policy hold on every
    # record of ``maps`` for this checkpoint's lifetime (dropped with it).
    gc_hold: object = field(default=None, compare=False, repr=False)
    # Opt-in KVMEM_LAYERWISE_SESSIONS: rolling-decode attention per layer,
    # ((key, mass) rows, samples, end, chunks), or None when it never ran.
    decode: tuple = None
    terminal: bool = False

    @property
    def bytes(self):
        # Serialized integer identities and FP64 mass/exposure, matching the
        # checkpoint pool's explicit payload accounting (not Python overhead).
        return 8 + sum(len(rows) * 40 for rows in self.maps) + sum(
            24 + len(snapshot[0]) * 24 for snapshot in self.histories) + sum(
            32 + len(snapshot[0]) * 16 for snapshot in (self.decode or ()))

    def evidence(self):
        return dict(accepted_history=self.position, layer_names=self.names,
            layer_tokens=[sum(r[3] for r in rows) for rows in self.maps],
            # Handles can change on rewind. Identity is original text plus its
            # dense destination, not an implementation-specific archive slot.
            map_sha256=hashlib.sha256(json.dumps([_identity(rows) for rows in self.maps]).encode()).hexdigest(),
            attention_sha256=hashlib.sha256(json.dumps(self.histories).encode()).hexdigest())

    def evidence_resolver(self):
        """evidence() computed when called; holds the immutable maps/histories only."""
        position, names, maps, histories = self.position, self.names, self.maps, self.histories
        def resolve(yield_gil=False):
            return dict(accepted_history=position, layer_names=names,
                layer_tokens=[sum(r[3] for r in rows) for rows in maps],
                map_sha256=_list_sha256((_identity(rows) for rows in maps), yield_gil),
                attention_sha256=_list_sha256(histories, yield_gil))
        return resolve


def _identity(rows):
    return tuple(tuple(row[2:]) for row in rows)


def _list_sha256(items, yield_gil=False):
    """sha256(json.dumps(list(items))) hashed one item at a time."""
    digest, first = hashlib.sha256(b'['), True
    for item in items:
        digest.update((json.dumps(item) if first else ', ' + json.dumps(item)).encode())
        first = False
        if yield_gil:
            time.sleep(0)  # Let the engine thread take the GIL between layers.
    digest.update(b']')
    return digest.hexdigest()


def lazy_evidence():
    """KVMEM_CHECKPOINT_LAZY_EVIDENCE: hash checkpoint evidence off the engine step.

    The capture/restore row keeps its timestamps, bytes and position in
    layerwise_query.jsonl; its digests are computed by the deferred evidence
    writer (or at the next synchronous emit) instead of before the step runs.
    """
    value = os.getenv('KVMEM_CHECKPOINT_LAZY_EVIDENCE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_CHECKPOINT_LAZY_EVIDENCE must be 0 or 1')
    return value == '1'


def emit_prefix_evidence(query, event, snapshot, **fields):
    if lazy_evidence():
        query.defer_lazy(event, snapshot.evidence_resolver(), **fields)
    else:
        query.emit(event, **fields, **snapshot.evidence())


def layer0_diverged(shared, layer_zero):
    """Whether layer zero's own window differs from the native (shared) frame.

    A probe-mode final Query selects each target layer with its own same-forward
    Query while the native frame (and the MTP draft) keep the global selection;
    the next pressure realigns them. Session checkpoints may be taken in between
    (KVMEM_LAYERWISE_SESSIONS). Lengths and partial-block geometry must agree.
    """
    if _identity(shared) == _identity(layer_zero):
        return False
    partial = lambda rows: tuple((r[2], r[3], r[4]) for r in rows if r[3] != 32)
    if (sum(r[3] for r in shared) != sum(r[3] for r in layer_zero) or
            partial(shared) != partial(layer_zero)):
        raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
    return True


def _resolve(authority, maps, position, *, planned_records=None, allow_layer0_divergence=False):
    """Resolve original coordinates against the retained immutable archive."""
    by_begin = {record.history_begin: record for record in authority.records.values()}
    resolved, canonical = [], []
    for rows in maps:
        chosen, mapped, seen, cursor = [], [], set(), 0
        for _, _, begin, count, dense in rows:
            if (begin in seen or dense != cursor or begin < 0 or count <= 0 or
                    begin + count > position):
                raise ValueError('Invalid layerwise checkpoint coordinates')
            record = by_begin.get(begin)
            if record is None or record.count != count:
                raise ValueError('Layerwise checkpoint source was not retained')
            chosen.append(record)
            mapped.append((record.handle.slot, record.handle.generation, begin, count, dense))
            seen.add(begin)
            cursor += count
        resolved.append(tuple(chosen))
        canonical.append(tuple(mapped))
    if not canonical or len({sum(r[3] for r in rows) for rows in canonical}) != 1:
        raise ValueError('Layerwise checkpoint has inconsistent physical lengths')
    partial = [tuple((r[2], r[3], r[4]) for r in rows if r[3] != authority.memory.config.block_tokens)
               for rows in canonical]
    if any(rows != partial[0] for rows in partial):
        raise ValueError('Layerwise checkpoint has inconsistent partial-block geometry')
    shared = (tuple(authority.memory._store.working_records()) if planned_records is None
              else tuple(planned_records))
    if _identity(shared) != _identity(canonical[0]):
        if not allow_layer0_divergence:
            raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
        layer0_diverged(shared, canonical[0])
    return tuple(resolved), tuple(canonical)


MAX_RETIRED = 64


def retire_state(controller, request_id, authority=None):
    """Keep a finished request's per-layer state for its terminal capture.

    The native finish runs before the session's terminal (M) capture. The
    resolved maps, prefill and decode histories move out of the live tables
    into a bounded stash; the request's window hold stays until the capture
    takes its own checkpoint hold (or the stash entry is dropped).
    """
    from .layerwise_gc_holds import holds_of
    holds = holds_of(controller)
    frame = controller.resolved_maps(request_id) if request_id in controller.maps else None
    controller.maps.pop(request_id, None)
    histories = controller.histories.pop(request_id, None)
    decode = controller.decode_histories.pop(request_id, None)
    from .layerwise_pressure import _drop_record_index
    _drop_record_index(controller, request_id, authority)
    owners = controller.__dict__.get('_history_owners', {})
    if histories is None:
        holds.release(request_id)
        owners.pop(request_id, None)
        return
    holds.retired[request_id] = (frame, histories, decode)
    while len(holds.retired) > MAX_RETIRED:
        oldest = next(iter(holds.retired))
        holds.retired.pop(oldest)
        holds.release(oldest)
        owners.pop(oldest, None)


def _consume_retired(controller, request_id):
    from .layerwise_gc_holds import holds_of
    holds = holds_of(controller)
    if holds.retired.pop(request_id, None) is not None:
        holds.release(request_id)
    controller.__dict__.get('_history_owners', {}).pop(request_id, None)


def capture(controller, request_id, authority, position, *, terminal=False, committed=False):
    """Per-layer snapshot; a finished request's stashed state is used if live state is gone."""
    reinstated = False
    if sessions_enabled() and request_id not in controller.histories:
        from .layerwise_gc_holds import holds_of
        stash = holds_of(controller).retired.get(request_id)
        if stash is not None:
            frame, histories, decode = stash
            if frame is not None:
                controller.maps[request_id] = frame
            controller.histories[request_id] = histories
            if decode is not None:
                controller.decode_histories[request_id] = decode
            reinstated = True
    try:
        return _capture(controller, request_id, authority, position, terminal=terminal, committed=committed)
    finally:
        if reinstated:
            controller.maps.pop(request_id, None)
            controller.histories.pop(request_id, None)
            controller.decode_histories.pop(request_id, None)


def _capture(controller, request_id, authority, position, *, terminal=False, committed=False):
    if position != authority.history_end or position <= 0:
        raise ValueError('Layerwise checkpoint must use the exact accepted boundary')
    if (terminal or committed) and not sessions_enabled():
        raise ValueError('Layerwise session checkpoints require KVMEM_LAYERWISE_SESSIONS')
    if not (terminal or committed) and request_id in controller.query.finished:
        raise ValueError('Layerwise P checkpoint must precede the final Query selection')
    histories = controller.histories.get(request_id)
    if histories is None or len(histories) != len(controller.names):
        raise ValueError('Missing independent cumulative attention histories')
    shared = tuple(authority.memory._store.working_records())
    extent = sum(r[3] for r in shared)
    maps = tuple(tuple(controller.records(request_id, authority, position, extent, i))
                 for i in range(len(controller.names)))
    _, maps = _resolve(authority, maps, position, allow_layer0_divergence=sessions_enabled())
    snapshots = []
    for history in histories:
        values, end, samples, chunks = history.snapshot()
        if end > position or any(begin >= position for begin in values):
            raise ValueError('Layerwise checkpoint contains future attention evidence')
        snapshots.append((tuple((key, mass, exposure) for key, (mass, exposure)
                                in sorted(values.items())), end, samples, chunks))
    if authority.action_history.prefill.snapshot() != histories[0].snapshot():
        raise ValueError('Layerwise checkpoint differs from shared layer-zero scores')
    decode = None
    decode_histories = controller.decode_histories.get(request_id)
    if decode_histories is not None:
        if not terminal and not sessions_enabled():
            raise ValueError('A frozen P checkpoint cannot carry decode attention')
        if len(decode_histories) != len(controller.names):
            raise ValueError('Missing per-layer rolling decode histories')
        decode = []
        for history in decode_histories:
            if history.end > position or any(begin >= position for begin in history.mass):
                raise ValueError('Layerwise checkpoint contains future decode attention')
            decode.append((tuple(sorted(history.mass.items())), history.samples, history.end, history.chunks))
        decode = tuple(decode)
    from .layerwise_gc_holds import checkpoint_hold
    # A diverged layer zero leaves the native frame (restored into the MTP draft
    # and, before the per-layer rebuild, every layer) outside the layer maps.
    held = maps + ((shared,) if _identity(shared) != _identity(maps[0]) else ())
    return LayerwisePrefix(position, tuple(controller.names), maps, tuple(snapshots),
                           gc_hold=checkpoint_hold(authority, held), decode=decode, terminal=terminal)


def prepare_restore(controller, request_id, authority, snapshot, options, *, planned_records=None):
    """Validate all CPU state without publishing partially restored owners."""
    if snapshot is None:
        raise ValueError('Layerwise restore requires an explicit per-layer checkpoint')
    if (tuple(controller.names) != snapshot.names or
            len(snapshot.maps) != len(snapshot.names) or
            len(snapshot.histories) != len(snapshot.names)):
        raise ValueError('Layerwise checkpoint layer order or count changed')
    if authority.history_end != snapshot.position:
        raise ValueError('Layerwise restore must first rewind to its exact P prefix')
    if (request_id in controller.maps or request_id in controller.histories or
            request_id in controller.query.finished):
        raise ValueError('Layerwise restore requires a fresh native request owner')
    chosen, maps = _resolve(authority, snapshot.maps, snapshot.position, planned_records=planned_records,
                            allow_layer0_divergence=sessions_enabled())
    histories = []
    for rows, end, samples, chunks in snapshot.histories:
        if end > snapshot.position or any(begin >= snapshot.position for begin, _, _ in rows):
            raise ValueError('Layerwise restore cannot reuse future attention evidence')
        history = controller.new_history(options)
        history.restore(({begin: (mass, exposure) for begin, mass, exposure in rows},
                         end, samples, chunks), snapshot.position)
        histories.append(history)
    if authority.action_history.prefill.snapshot() != histories[0].snapshot():
        raise ValueError('Restored layer-zero scores differ from the shared checkpoint')
    return chosen, (snapshot.position, maps), histories


def restored_decode(snapshot):
    """Per-layer rolling decode histories of a session checkpoint, or None."""
    if snapshot.decode is None:
        return None
    from .layerwise_decode import DecodeHistory
    if len(snapshot.decode) != len(snapshot.names):
        raise ValueError('Layerwise checkpoint decode history count changed')
    return [DecodeHistory(dict(rows), samples, end, chunks)
            for rows, samples, end, chunks in snapshot.decode]


def install_restored(controller, request_id, authority, frame, histories, snapshot):
    """Publish restored per-layer CPU state after every GPU write completed."""
    bind = getattr(controller, 'bind_history_owner', None)
    if bind is not None:
        bind(request_id, authority)
    controller.maps[request_id] = frame
    controller.histories[request_id] = histories
    decode = restored_decode(snapshot)
    if decode is not None:
        controller.decode_histories[request_id] = decode
    # Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: hold every restored layer window.
    # The selection pause first excludes an in-flight retirement decision.
    from .layerwise_gc_holds import holds_of, wait_paused
    wait_paused(authority)
    holds_of(controller).protect(request_id, authority, frame[1])


def capture_for_pool(adapter, request_id, authority, position, *, terminal=False):
    runner = getattr(adapter, 'runner', None)
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is None:
        return None
    worker = getattr(runner, '_kvmem_sessions', None)
    options = getattr(worker, 'options', None)
    if not sessions_enabled() and (terminal or not getattr(options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise checkpoints currently require frozen pre-Query P prefixes')
    snapshot = capture(controller, request_id, authority, position, terminal=terminal)
    if terminal and sessions_enabled():
        # The checkpoint now holds its records; the finished request's stash goes.
        _consume_retired(controller, request_id)
    return snapshot


def _rebuild_draft(controller, state, records):
    """Rebuild the MTP draft layers from layer zero's window; returns the count.

    A probe-mode final Query leaves the draft (and the native frame) on the
    global selection while layer zero has its own. A session restore makes the
    new turn start aligned: draft and native frame follow layer zero.
    """
    runner = getattr(controller, 'runner', None)
    names = tuple(getattr(runner, '_kvmem_archive_layers', controller.names))
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Draft rebuild changed the target archive layer order')
    count = 0
    for index, name in enumerate(names[len(controller.names):], len(controller.names)):
        controller.query.assemble_layer_pinned(state, index, records, layer_name=name)
        count += 1
    return count


def realign_after_restore(worker, request_id, checkpoint):
    """KVMEM_LAYERWISE_SESSIONS: after a restore is published, commit layer
    zero's window as the native frame (bookkeeping only; its pages, and the
    draft's, already hold it) and let the retained checkpoint adopt it.

    The shared attention state, fast metadata and native skip hints all
    describe the native frame, so the new turn must not start diverged.
    """
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None or snapshot is None or not sessions_enabled():
        return False
    frame = controller.maps.get(request_id)
    if frame is None or isinstance(frame, PendingMapsType()) or worker.bindings[request_id].pressure_restore:
        return False
    authority = checkpoint.authority
    if not layer0_diverged(tuple(authority.memory._store.working_records()), frame[1][0]):
        return False
    from .layerwise_no_probe import commit_layer_zero
    records = [authority.records[(row[0], row[1])] for row in frame[1][0]]
    commit_layer_zero(controller, dict(authority=authority, layer_zero_records=records,
                                       selected_tokens=sum(r.count for r in records)))
    worker.selection_committed(request_id, checkpoint.position)
    controller.query.emit('pressure_prefix_realigned', request_id=request_id,
                          accepted_history=checkpoint.position)
    return True


def query_frame_committed(runner, request_id):
    """KVMEM_LAYERWISE_SESSIONS: a committed Query (non-pressure) selection
    replaced the whole native frame. Until the per-layer Query completes, every
    layer resumes from that frame, so earlier per-layer maps are stale; drop
    them before a P checkpoint at this boundary re-takes its snapshot.
    """
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is None or not sessions_enabled():
        return False
    frame = controller.maps.pop(request_id, None)
    if isinstance(frame, PendingMapsType()):
        frame.deferred.discard()
    return frame is not None


def PendingMapsType():
    from .layerwise_pressure import PendingMaps
    return PendingMaps


def restore_sync(worker, request_id, checkpoint):
    """Rebuild distinct target-layer pages before publishing a synchronous P hit.

    Native restore already rebuilt layer zero and the draft. Reuse the existing
    per-layer archive assembly for the remaining target layers. CPU ownership is
    installed only after every GPU write completes; failures never publish a hit.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None:
        raise ValueError('Layerwise prefix cannot be restored by a shared-pressure worker')
    if worker.bindings[request_id].pressure_restore:
        raise ValueError('A compacting layerwise restore uses restore_pressure_sync')
    if not sessions_enabled() and (checkpoint.kind != 'P' or
            not getattr(worker.options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise restore requires the original frozen P window')
    chosen, frame, histories = prepare_restore(controller, request_id,
        checkpoint.authority, snapshot, worker.options)
    # Synchronous restore and causal Query both run on the model thread.
    # Reuse their one-layer arena; background pressure assembly can overlap
    # this restore and must retain its separate scratch/DMA ownership.
    state = dict(authority=checkpoint.authority, request_id=request_id,
        selected_tokens=sum(r.count for r in chosen[0]), assembly_arena='layerwise')
    # Native restore wrote its own (shared) frame; a diverged layer zero
    # (session checkpoint after a probe-mode Query) is rebuilt like the others.
    diverged = sessions_enabled() and layer0_diverged(
        tuple(checkpoint.authority.memory._store.working_records()), frame[1][0])
    first = 0 if diverged else 1
    for layer, records in enumerate(chosen[first:], first):
        controller.query.assemble_layer_pinned(state, layer, records)
    if diverged:
        _rebuild_draft(controller, state, chosen[0])
    import torch
    torch.cuda.current_stream(worker.runner.device).synchronize()
    install_restored(controller, request_id, checkpoint.authority, frame, histories, snapshot)
    emit_prefix_evidence(controller.query, 'pressure_prefix_restored', snapshot, request_id=request_id,
        actual_method='accumulated_attention_per_layer', fallback_reason=None)


def require_supported_restore(worker, request_id, checkpoint, *, asynchronous=False):
    """Reject unqualified paths before modifying any restore destination."""
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None or snapshot is None:
        raise ValueError('Layerwise pressure restore is missing its matching checkpoint/controller')
    if sessions_enabled():
        # Session checkpoints (P or M; copy, resident or compacting). An
        # asynchronous import finishes its per-layer part on the model thread
        # (finish_async_restore) before the restore is published.
        if checkpoint.kind not in ('P', 'M'):
            raise ValueError('Unknown layerwise checkpoint kind')
        return
    if (asynchronous or checkpoint.kind != 'P' or
            worker.bindings[request_id].pressure_restore or
            not getattr(worker.options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise prefix requires synchronous frozen P restore')


def oracle_record_groups(worker, request_id, authority):
    """Group equal frames for independent full-byte restore verification."""
    names = tuple(worker.runtime.names)
    shared = tuple(authority.memory._store.working_records())
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if controller is None:
        return ((shared, tuple(range(len(names)))),)
    if names[:len(controller.names)] != tuple(controller.names):
        raise ValueError('Layerwise restore oracle changed target archive order')
    frame = controller.maps.get(request_id)
    if frame is None or frame[0] != authority.history_end:
        raise ValueError('Layerwise restore oracle needs the exact restored frame')
    _, maps = _resolve(authority, frame[1], frame[0], allow_layer0_divergence=sessions_enabled())
    # The MTP draft follows layer zero (a diverged session restore rebuilds it so).
    maps += (maps[0],) * (len(names) - len(maps))
    groups = {}
    for index, records in enumerate(maps):
        groups.setdefault(records, []).append(index)
    return tuple((records, tuple(indices)) for records, indices in groups.items())


def refresh_committed_snapshot(worker, request_id, checkpoint):
    """A selection committed exactly at a live P checkpoint (non-frozen sessions).

    The pool adopts the newly committed shared window for that P; the per-layer
    snapshot is re-taken at the same position so its maps stay the windows the
    restored turn will actually resume with.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    if controller is None or getattr(checkpoint, 'layerwise_pressure', None) is None:
        return
    if not sessions_enabled():
        raise ValueError('Frozen layerwise P checkpoint cannot adopt a later shared window')
    checkpoint.layerwise_pressure = capture(controller, request_id, checkpoint.authority,
                                            checkpoint.position, committed=True)


def restore_pressure_sync(worker, request_id, checkpoint):
    """Compacting session restore (KVMEM_LAYERWISE_SESSIONS).

    The native compacting import already wrote one shared, budget-sized window
    into every layer. Every layer resumes from that shared frame (no per-layer
    map), with its own restored prefill and decode attention histories; the
    next layerwise pressure refresh selects each layer separately again.
    """
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None or snapshot is None or not sessions_enabled():
        raise ValueError('Compacting layerwise restore requires KVMEM_LAYERWISE_SESSIONS and its checkpoint')
    authority = checkpoint.authority
    if (tuple(controller.names) != snapshot.names or len(snapshot.histories) != len(snapshot.names)):
        raise ValueError('Layerwise checkpoint layer order or count changed')
    if authority.history_end != snapshot.position:
        raise ValueError('Layerwise restore must first rewind to its exact checkpoint prefix')
    if (request_id in controller.maps or request_id in controller.histories or
            request_id in controller.query.finished):
        raise ValueError('Layerwise restore requires a fresh native request owner')
    histories = []
    for rows, end, samples, chunks in snapshot.histories:
        if end > snapshot.position or any(begin >= snapshot.position for begin, _, _ in rows):
            raise ValueError('Layerwise restore cannot reuse future attention evidence')
        history = controller.new_history(worker.options)
        history.restore(({begin: (mass, exposure) for begin, mass, exposure in rows},
                         end, samples, chunks), snapshot.position)
        histories.append(history)
    if authority.action_history.prefill.snapshot() != histories[0].snapshot():
        raise ValueError('Restored layer-zero scores differ from the shared checkpoint')
    bind = getattr(controller, 'bind_history_owner', None)
    if bind is not None:
        bind(request_id, authority)
    controller.histories[request_id] = histories
    decode = restored_decode(snapshot)
    if decode is not None:
        controller.decode_histories[request_id] = decode
    shared = tuple(authority.memory._store.working_records())
    from .layerwise_gc_holds import holds_of, wait_paused
    wait_paused(authority)
    holds_of(controller).protect(request_id, authority, [shared])
    controller.query.emit('pressure_prefix_restored', request_id=request_id,
        actual_method='shared_compacted_frame_per_layer_histories', fallback_reason=None,
        accepted_history=snapshot.position, selected_tokens=sum(r[3] for r in shared))


def finish_async_restore(worker, request_id, checkpoint, *, resident, pressure=None):
    """Per-layer completion of an asynchronous session import, on the model thread.

    The background import wrote one shared window into every layer (or, for a
    resident import, left the terminal pages untouched). Before publication:
    a compacting import resumes from that shared frame; a resident one already
    holds every layer's terminal window, so only CPU state is installed; a copy
    import rebuilds target layers 1..N from the snapshot like restore_sync.
    """
    controller = getattr(getattr(worker, 'runner', None), '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if pressure is None:
        pressure = getattr(worker.bindings[request_id], 'pressure_restore', False)
    if pressure:
        restore_pressure_sync(worker, request_id, checkpoint)
        return
    if resident:
        if controller is None or snapshot is None or not sessions_enabled():
            raise ValueError('Resident layerwise import requires KVMEM_LAYERWISE_SESSIONS and its checkpoint')
        chosen, frame, histories = prepare_restore(controller, request_id,
            checkpoint.authority, snapshot, worker.options)
        if layer0_diverged(tuple(checkpoint.authority.memory._store.working_records()), frame[1][0]):
            state = dict(authority=checkpoint.authority, request_id=request_id,
                selected_tokens=sum(r.count for r in chosen[0]), assembly_arena='layerwise')
            if _rebuild_draft(controller, state, chosen[0]):
                import torch
                torch.cuda.current_stream(worker.runner.device).synchronize()
                # Verification: the draft no longer equals the terminal pages.
                worker.__dict__.setdefault('layerwise_draft_rebuilt', set()).add(request_id)
        install_restored(controller, request_id, checkpoint.authority, frame, histories, snapshot)
        from .pressure_key_reuse import restored as _reuse_restored
        _reuse_restored(controller, request_id, resident=True)
        emit_prefix_evidence(controller.query, 'pressure_prefix_restored', snapshot, request_id=request_id,
            actual_method='resident_terminal_pages_per_layer', fallback_reason=None)
        return
    restore_sync(worker, request_id, checkpoint)
