"""CPU identity/state snapshots for frozen P-prefix layerwise pressure reuse.

This module does not publish a restore or write GPU pages. The session owner
must rebuild the prepared per-layer records and wait for its completion fence
before installing the maps/histories or admitting a public Query.
"""
from dataclasses import dataclass
import hashlib
import json

from .prefill_attention import PrefillHistory


@dataclass(frozen=True)
class LayerwisePrefix:
    position: int
    names: tuple
    maps: tuple
    histories: tuple

    @property
    def bytes(self):
        # Serialized integer identities and FP64 mass/exposure, matching the
        # checkpoint pool's explicit payload accounting (not Python overhead).
        return 8 + sum(len(rows) * 40 for rows in self.maps) + sum(
            24 + len(snapshot[0]) * 24 for snapshot in self.histories)

    def evidence(self):
        return dict(accepted_history=self.position, layer_names=self.names,
            layer_tokens=[sum(r[3] for r in rows) for rows in self.maps],
            # Handles can change on rewind. Identity is original text plus its
            # dense destination, not an implementation-specific archive slot.
            map_sha256=hashlib.sha256(json.dumps([_identity(rows) for rows in self.maps]).encode()).hexdigest(),
            attention_sha256=hashlib.sha256(json.dumps(self.histories).encode()).hexdigest())


def _identity(rows):
    return tuple(tuple(row[2:]) for row in rows)


def _resolve(authority, maps, position):
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
    shared = tuple(authority.memory._store.working_records())
    if _identity(shared) != _identity(canonical[0]):
        raise ValueError('Layerwise checkpoint differs from the native layer-zero frame')
    return tuple(resolved), tuple(canonical)


def capture(controller, request_id, authority, position):
    if position != authority.history_end or position <= 0:
        raise ValueError('Layerwise checkpoint must use the exact accepted boundary')
    if request_id in controller.query.finished:
        raise ValueError('Layerwise P checkpoint must precede the final Query selection')
    histories = controller.histories.get(request_id)
    if histories is None or len(histories) != len(controller.names):
        raise ValueError('Missing independent cumulative attention histories')
    shared = tuple(authority.memory._store.working_records())
    extent = sum(r[3] for r in shared)
    maps = tuple(tuple(controller.records(request_id, authority, position, extent, i))
                 for i in range(len(controller.names)))
    _, maps = _resolve(authority, maps, position)
    snapshots = []
    for history in histories:
        values, end, samples, chunks = history.snapshot()
        if end > position or any(begin >= position for begin in values):
            raise ValueError('Layerwise checkpoint contains future attention evidence')
        snapshots.append((tuple((key, mass, exposure) for key, (mass, exposure)
                                in sorted(values.items())), end, samples, chunks))
    if authority.action_history.prefill.snapshot() != histories[0].snapshot():
        raise ValueError('Layerwise checkpoint differs from shared layer-zero scores')
    return LayerwisePrefix(position, tuple(controller.names), maps, tuple(snapshots))


def prepare_restore(controller, request_id, authority, snapshot, options):
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
    chosen, maps = _resolve(authority, snapshot.maps, snapshot.position)
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


def capture_for_pool(adapter, request_id, authority, position, *, terminal=False):
    runner = getattr(adapter, 'runner', None)
    controller = getattr(runner, '_kvmem_layerwise_pressure', None)
    if controller is None:
        return None
    worker = getattr(runner, '_kvmem_sessions', None)
    options = getattr(worker, 'options', None)
    if terminal or not getattr(options, 'session_freeze_prompt', False):
        raise ValueError('Layerwise checkpoints currently require frozen pre-Query P prefixes')
    return capture(controller, request_id, authority, position)


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
    if (checkpoint.kind != 'P' or worker.bindings[request_id].pressure_restore or
            not getattr(worker.options, 'session_freeze_prompt', False)):
        raise ValueError('Layerwise restore requires the original frozen P window')
    chosen, frame, histories = prepare_restore(controller, request_id,
        checkpoint.authority, snapshot, worker.options)
    state = dict(authority=checkpoint.authority, request_id=request_id,
        selected_tokens=sum(r.count for r in chosen[0]), assembly_arena='pressure')
    for layer, records in enumerate(chosen[1:], 1):
        controller.query.assemble_layer_pinned(state, layer, records)
    import torch
    torch.cuda.current_stream(worker.runner.device).synchronize()
    controller.maps[request_id] = frame
    controller.histories[request_id] = histories
    controller.query.emit('pressure_prefix_restored', request_id=request_id,
        actual_method='accumulated_attention_per_layer', fallback_reason=None,
        **snapshot.evidence())


def require_supported_restore(worker, request_id, checkpoint, *, asynchronous=False):
    """Reject unqualified paths before modifying any restore destination."""
    controller = getattr(worker.runner, '_kvmem_layerwise_pressure', None)
    snapshot = getattr(checkpoint, 'layerwise_pressure', None)
    if controller is None and snapshot is None:
        return
    if controller is None or snapshot is None:
        raise ValueError('Layerwise pressure restore is missing its matching checkpoint/controller')
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
    _, maps = _resolve(authority, frame[1], frame[0])
    maps += (maps[0],) * (len(names) - len(maps))
    groups = {}
    for index, records in enumerate(maps):
        groups.setdefault(records, []).append(index)
    return tuple((records, tuple(indices)) for records, indices in groups.items())
