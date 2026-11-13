"""Retain logical layer state across an exact native page swap.

The native checkpoint copies each layer's current K/V bytes. This witness does
not reconstruct those bytes or reselect history. It keeps the request's CPU
state alive and rejects a restore if that state changed while its pages were
absent. Only the explicitly qualified layerwise swap path uses these checks.
"""
from dataclasses import dataclass


def validate_modes(query, pressure):
    """Shared launch/worker guard for the qualified exact-swap replay modes."""
    from ..adapters.original_rope import enabled as original_rope
    if (original_rope() or getattr(query, 'no_probe', None) is not False or
            getattr(pressure, 'rolling_decode', None) is not False or
            getattr(query, 'mtp_selection', None) != 'global'):
        raise ValueError('Layerwise request swap requires compact positions, private Query replay, '
                         'non-rolling decode and global MTP selection')


def _controllers(runner):
    options = getattr(runner, '_kvmem_request_swap_options', None)
    if not getattr(options, 'request_swap_layerwise', False):
        return None
    query = getattr(runner, '_kvmem_layerwise_query', None)
    pressure = getattr(runner, '_kvmem_layerwise_pressure', None)
    refresh = getattr(runner, '_kvmem_async_refresh', None)
    if (query is None or pressure is None or refresh is None or
            getattr(query, 'pressure', None) is not pressure or
            getattr(pressure, 'query', None) is not query or
            getattr(query, 'mode', None) != 'sequential' or
            not getattr(query, 'optimized', False) or
            not getattr(query, 'native_batches', False) or
            not getattr(pressure, 'single_assembly', False)):
        raise RuntimeError('Layerwise swap requires its qualified live controllers')
    validate_modes(query, pressure)
    return query, pressure, refresh


def _belongs(owner, runner, request_id):
    """A borrowed layer view inherits its original assembler's page owner."""
    while owner is not None:
        binding = getattr(owner, '_runner_binding', None)
        if binding is not None and binding[0] is runner and binding[1] == request_id:
            return True
        lease = getattr(owner, '_restore_lease', None)
        if getattr(lease, 'request_id', None) == request_id:
            return True
        owner = getattr(owner, '_validated_parent', None)
    return False


def ready(runner, request_id, *, accepted=True):
    """Observe quiescence without waiting, committing, or reaping other work."""
    controllers = _controllers(runner)
    if controllers is None:
        return True
    query, _, refresh = controllers
    job = refresh.jobs.get(request_id)
    if (job is not None and not job.done) or request_id in refresh.deferred:
        return False
    if (request_id in query.pending or request_id in query.active_owners or
            query.active is not None and query.active['request_id'] == request_id):
        return False
    runtime = runner._kvmem_capture_runtime
    if accepted and not runtime.publication_ready((request_id,)):
        return False
    authority = runtime.authorities.get(request_id)
    if authority is None or getattr(authority, '_refresh_background', False):
        return False
    memory = authority.memory
    if accepted and (getattr(memory.phase, 'name', None) != 'READY' or memory._submitted or
                     memory._fence is not None or memory._plan is not None):
        return False
    for arena in ('', '_refresh', '_pressure', '_layerwise'):
        staging = getattr(runner, '_kvmem' + arena + '_assembly_staging', None)
        if staging is None:
            continue
        if _belongs(staging.owner, runner, request_id):
            return False
        binding = getattr(staging, 'fused_binding', None)
        if binding is not None and binding[1] == request_id:
            event = getattr(staging, 'fused_event', None)
            if (getattr(staging, 'fused_active', False) or
                    getattr(staging, 'fused_poisoned', False) or
                    event is not None and not event.query()):
                return False
    return True


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _history(history):
    if hasattr(history, 'snapshot'):
        return _freeze(history.snapshot())
    # DecodeHistory holds accepted mass rather than mass/exposure pairs.
    return _freeze((history.mass, history.end, history.samples, history.chunks))


def _state(runner, request_id, *, deep):
    query, pressure, refresh = _controllers(runner)
    authority = runner._kvmem_capture_runtime.authorities[request_id]
    memory = authority.memory
    window = getattr(runner, '_kvmem_window_adapter', None)
    window_state = None if window is None else tuple(
        (request_id in getattr(window, name), getattr(window, name).get(request_id))
        for name in ('offsets', 'history_ends', 'generations', 'private_capacities'))
    # Eager execution creates this global adapter lazily. A peer's first
    # pressure may create/replace it while this owner still has no window.
    # Bind its identity only once it actually carries this request's state.
    if window_state is not None and not any(present for present, _ in window_state):
        window = window_state = None
    # Deferred commit maps (KVMEM_LAYERWISE_PRESSURE_DEFERRED_MAPS) resolve
    # before identification, like deferred folds below; otherwise the first
    # use after a restore would replace this frame and fail the witness.
    resolve = getattr(pressure, 'resolved_maps', None)
    frame = resolve(request_id) if resolve is not None else pressure.maps.get(request_id)
    histories = pressure.histories.get(request_id)
    decode = pressure.decode_histories.get(request_id)
    owners = [query, pressure, refresh, authority, memory, window, frame, histories, decode]
    history_states = []
    if histories:
        # Deferred layer folds are applied before any state is identified.
        from .deferred_fold import force_all
        force_all(histories)
    for rows in (histories, decode):
        history_states.append(None if rows is None else tuple(
            _history(h) if deep else (h.end, h.samples, h.chunks) for h in rows))
        for history in rows or ():
            # Buffered values is a materializing property; retain the native
            # immutable state directly. Plain histories publish new dicts.
            state = vars(history)
            owners.extend((history, state.get('_buffered_state'),
                           state.get('values'), state.get('mass')))
    signature = (_freeze(frame) if deep else None, tuple(history_states), window_state,
                 request_id in query.finished, authority.history_end,
                 _freeze(tuple(memory._store.working_records())) if deep else None,
                 tuple(query.names), tuple(pressure.names))
    return tuple(owners), signature


@dataclass(frozen=True)
class LayerwiseSwapWitness:
    request_id: str
    owners: tuple
    signature: tuple
    fast_signature: tuple

    def validate(self, runner, *, deep=False):
        if _controllers(runner) is None:
            raise RuntimeError('Layerwise swap was disabled while a checkpoint was retained')
        if not ready(runner, self.request_id):
            raise RuntimeError('Layerwise swap owner acquired work while its pages were suspended')
        owners, signature = _state(runner, self.request_id, deep=deep)
        if (len(owners) != len(self.owners) or
                any(current is not held for current, held in zip(owners, self.owners)) or
                signature != (self.signature if deep else self.fast_signature)):
            raise RuntimeError('Layerwise swap logical state or window generation changed')


def capture(runner, request_id):
    if _controllers(runner) is None:
        return None
    if not ready(runner, request_id):
        raise RuntimeError('Layerwise swap capture requires a quiescent accepted owner')
    owners, signature = _state(runner, request_id, deep=True)
    _, fast_signature = _state(runner, request_id, deep=False)
    return LayerwiseSwapWitness(request_id, owners, signature, fast_signature)
