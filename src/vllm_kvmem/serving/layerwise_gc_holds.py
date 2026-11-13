"""Opt-in KVMEM_LAYERWISE_HISTORICAL_GC: keep every layer's window out of historical GC.

The native store protects only its own (global) frame from deletion. Layerwise
execution keeps one window per attention layer in engine maps the store does not
see, so a record still used by some layer could otherwise be retired.

Each per-layer selection is bracketed:

1. ``pause``: no new retirements for this request, requested before the record
   snapshot. ``wait_paused`` observes the worker's acknowledgement before any
   record is chosen, so every snapshot record stays live through selection.
2. ``protect``: after the last layer is chosen, one native current-window hold
   (the store's own ``protect_selection`` mechanism) covers the union of all
   layers' windows. It replaces the request's previous hold, then GC resumes.

Holds are dropped when the request finishes. Retirement of other history,
including records that left every window, is unchanged.

Stored per-layer session checkpoints take a policy hold on all their records
(``checkpoint_hold``), released with the checkpoint payload, so a later
restore always finds its sources.

The native pause is one flag per scope. Pauses are therefore owned: a scope
stays paused while any owner (a selection, a checkpoint capture) holds it.
"""
import threading
import time
import weakref

from .pressure_reader import layerwise_gc


def scope_of(authority):
    """The request's native historical GC scope when the opt-in is active."""
    if authority is None or not layerwise_gc():
        return None
    return getattr(getattr(authority, 'memory', None), '_history_scope', None)


_LOCK = threading.Lock()
_OWNERS = {}            # id(native scope) -> owner keys currently pausing it


def _own(scope, key, paused):
    with _LOCK:
        owners = _OWNERS.setdefault(id(scope), set())
        held = bool(owners)
        if paused:
            owners.add(key)
        else:
            owners.discard(key)
        active = bool(owners)
        if not active:
            del _OWNERS[id(scope)]
        if held and active and _keep_request():
            # Opt-in KVMEM_CHECKPOINT_FAST_STATE. The scope has been paused
            # without interruption since its current request. Every native
            # pause_policy call starts a new request and discards the worker's
            # acknowledgement, so repeating it made each wait_paused after an
            # early pause() sleep for another worker tick (~5 ms). One tick
            # that saw the pause already excludes an in-flight decision.
            return
        scope.pause_policy(active)


def _keep_request():
    from .layerwise_checkpoints import fast_state
    return fast_state()


def pause(authority, key='select'):
    """Request a pause of new retirements; cheap, never waits."""
    scope = scope_of(authority)
    if scope is not None:
        _own(scope, key, True)
    return scope is not None


def wait_paused(authority, timeout=30., key='select'):
    """Seconds spent until the worker acknowledged the pause (0 when inactive)."""
    scope = scope_of(authority)
    if scope is None:
        return 0.
    _own(scope, key, True)             # idempotent; also covers a missed early request
    started = time.perf_counter()
    deadline = started + timeout
    while not scope.policy_paused:
        if time.perf_counter() > deadline:
            raise TimeoutError('Historical GC did not acknowledge the layerwise selection pause')
        time.sleep(.0002)
    return time.perf_counter() - started


def resume(authority, key='select'):
    scope = scope_of(authority)
    if scope is not None:
        _own(scope, key, False)


class CheckpointHold:
    """Owner of one native policy hold for a stored per-layer checkpoint."""
    __slots__ = ('native', 'scope_key', '__weakref__')

    def __init__(self, native, scope_key):
        self.native, self.scope_key = native, scope_key
        _CHECKPOINT_HOLDS.add(self)

    def release(self):
        self.native = None


_CHECKPOINT_HOLDS = weakref.WeakSet()


def _native_handles(scope, name, pairs):
    """(native call, its argument) protecting the given (slot, generation) pairs.

    Opt-in KVMEM_SESSION_FAST_TURN: ``<name>_pairs`` takes the tuples as they
    are; otherwise one Handle per record goes to ``<name>``.
    """
    from .layerwise_checkpoints import fast_turn
    if fast_turn() and type(pairs) is list:
        fast = getattr(scope, name + '_pairs', None)
        if fast is not None:
            def call(rows):
                hold = fast(rows)
                if hold is NotImplemented:      # not plain int tuples: the original call
                    hold = getattr(scope, name)(_handles(rows))
                return hold
            return call, pairs
    return getattr(scope, name), _handles(pairs)


def _handles(pairs):
    from .. import _native
    return [_native.Handle(slot, generation) for slot, generation in pairs]


def checkpoint_hold(authority, maps, pairs=None):
    """A policy hold on every record of a stored per-layer checkpoint.

    Taken under a paused scope (no retirement decision in flight), from the
    live, window-held records of the capturing request. None when inactive.
    ``pairs`` is window_handles(maps) when the caller already computed it.
    """
    scope = scope_of(authority)
    if scope is None:
        return None
    protect, handles = _native_handles(scope, 'protect', window_handles(maps) if pairs is None else pairs)
    if not handles:
        return None
    try:
        wait_paused(authority, key='checkpoint')
        hold = protect(handles)
    finally:
        _own(scope, 'checkpoint', False)
    if hold is None:
        raise RuntimeError('A layerwise checkpoint record was retired before its capture')
    return CheckpointHold(hold, id(scope))


def window_handles(groups):
    """Unique (slot, generation) of every record in the given per-layer windows.

    ``groups`` holds records (``.handle``) or map rows (slot, generation, ...).
    """
    from .layerwise_checkpoints import fast_native
    native = fast_native()
    if native is not None:                 # opt-in KVMEM_CHECKPOINT_FAST_STATE
        handles = native.unique_handles(groups, True)
        if handles is not None:            # plain map rows only; same sorted set
            return handles
    unique = set()
    for group in groups:
        for item in group:
            handle = getattr(item, 'handle', None)
            unique.add((handle.slot, handle.generation) if handle is not None else (item[0], item[1]))
    unique.discard((-1, -1))           # unpublished partial-tail placeholders
    return sorted(unique)


_REGISTRY = weakref.WeakSet()


class WindowHolds:
    """One native current-window hold per request, covering every layer."""

    def __init__(self):
        self.holds = {}
        self.scopes = {}            # request_id -> id(native scope) of its hold
        self.counts = {}
        # KVMEM_LAYERWISE_SESSIONS: finished requests' per-layer state kept
        # (with their window hold) until the terminal capture consumes it.
        self.retired = {}
        _REGISTRY.add(self)

    def protect(self, request_id, authority, groups):
        scope = scope_of(authority)
        if scope is None:
            return False
        protect, handles = _native_handles(scope, 'protect_current', window_handles(groups))
        try:
            hold = protect(handles) if handles else None
            if handles and hold is None:
                # The selection ran under the acknowledged pause, so every
                # record was live; a refusal means a protocol violation.
                raise RuntimeError('A layerwise window record was retired during its paused selection')
            previous = self.holds.pop(request_id, None)
            self.scopes.pop(request_id, None)
            if hold is not None:
                self.holds[request_id] = hold
                self.scopes[request_id] = id(scope)
            del previous                   # release only after the new hold exists
            self.counts['protected'] = self.counts.get('protected', 0) + 1
            self.counts['held_records'] = len(handles)
        finally:
            _own(scope, 'select', False)
        return True

    def release(self, request_id):
        self.holds.pop(request_id, None)
        self.scopes.pop(request_id, None)


def drop_scope(authority):
    """Release every layerwise window hold and pause owner on an idle scope.

    Called before an evicted session's authority closes: no request of that
    scope is active and its checkpoints are gone, so nothing may still hold
    its records. Returns what was dropped (diagnostic; normally nothing).
    """
    scope = getattr(getattr(authority, 'memory', None), '_history_scope', None)
    if scope is None:
        return {}
    key, dropped = id(scope), []
    for holds in list(_REGISTRY):
        for request_id, owner in list(holds.scopes.items()):
            if owner == key:
                holds.release(request_id)
                holds.retired.pop(request_id, None)
                dropped.append(request_id)
    checkpoints = 0
    for hold in list(_CHECKPOINT_HOLDS):
        if hold.scope_key == key and hold.native is not None:
            hold.release()
            checkpoints += 1
    with _LOCK:
        owners = _OWNERS.pop(key, None)
    if owners:
        scope.pause_policy(False)
    result = {}
    if dropped:
        result['window_holds'] = dropped
    if checkpoints:
        result['checkpoint_holds'] = checkpoints
    if owners:
        result['pause_owners'] = sorted(map(str, owners))
    return result


def holds_of(controller):
    """The controller's WindowHolds, created on first use."""
    holds = getattr(controller, '_layerwise_gc_holds', None)
    if holds is None:
        holds = controller._layerwise_gc_holds = WindowHolds()
    return holds
