"""Atomic credits for bounded operations across independent resource pools.

This ledger does not allocate or free memory. Its caller must reserve before
touching resources, dispatch before publishing work, and provide completion
fences after the physical operation being accounted for. In particular, a
cancel intent and an anticipated reclaim are never credits.

Amounts use each pool's native unit (bytes, pages or slots). Shared physical
pools need their own aggregate dimension in addition to any per-group limits.
The caller identifies real backing owners; tensor views are not new owners.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import count
from threading import RLock
from uuid import uuid4
from weakref import WeakKeyDictionary


@dataclass(frozen=True)
class ResourceIdentity:
    epoch: str
    request_id: str
    generation: int
    operation_id: int


@dataclass(frozen=True, eq=False)
class ResourceGrant:
    """Opaque receipt; only the ledger that issued this instance accepts it."""
    identity: ResourceIdentity
    amounts: tuple[tuple[str, int], ...]
    emergency: bool
    _issuer: object = field(repr=False)


@dataclass
class _Reservation:
    phase: str = 'prepared'
    live: dict = field(default_factory=dict)
    promised: dict = field(default_factory=dict)
    retained: tuple | None = None


def _amounts(value, dimensions, *, nonempty=False):
    if not isinstance(value, Mapping):
        raise ValueError('Resource amounts must be a mapping')
    result = {}
    for resource, amount in value.items():
        if resource not in dimensions or type(amount) is not int or amount < 0:
            raise ValueError('Unknown resource or invalid nonnegative integer amount')
        if amount:
            result[resource] = amount
    if nonempty and not result:
        raise ValueError('A resource grant must reserve at least one resource')
    return result


class ResourceLedger:
    """Thread-safe prepare/dispatch/settle accounting with fenced reclamation.

    ``prepare`` takes all dimensions atomically or returns None. ``dispatch``
    closes the abort path before any physical allocation or queued operation.
    ``allocated`` reports cumulative physical allocations and converts promise
    to live without changing total charge. ``settle`` waits for completion and
    releases unused promise, optionally retaining only specified live owners.
    ``release`` waits for their final physical-free fence.

    Completion methods return False while their fence is pending and True after
    completion, including identical duplicate notices. A grant cannot be
    recreated from its identity: an old/foreign receipt cannot release a newer
    operation. Monotonic operation IDs are issued by this ledger, not callers.
    Active receipts are strongly retained; terminal receipts are weakly retained
    for idempotence without an ever-growing history of completed requests.

    Emergency work can borrow the reserved capacity. Ordinary work cannot.
    Its unused reserve is restored as emergency promises/live owners drain.
    Session-generation validity at *new admission* remains the session owner's
    responsibility; this ledger keeps existing generations independently owned.
    """
    def __init__(self, limits, *, emergency=None, epoch=None):
        if (not isinstance(limits, Mapping) or not limits or
                any(not isinstance(name, str) or not name or
                    type(amount) is not int or amount < 0
                    for name, amount in limits.items())):
            raise ValueError('Resource limits require names and nonnegative integer capacities')
        if epoch is not None and (not isinstance(epoch, str) or not epoch):
            raise ValueError('Resource ledger epoch must be a nonempty string')
        self.epoch = uuid4().hex if epoch is None else epoch
        self._limits = dict(limits)
        self._reserve = _amounts({} if emergency is None else emergency, self._limits)
        if any(amount > self._limits[name] for name, amount in self._reserve.items()):
            raise ValueError('Emergency reserve exceeds resource hard limit')
        self._issuer = object()
        self._sequence = count(1)
        self._active = {}
        self._terminal = WeakKeyDictionary()
        self._live, self._promised, self._emergency_used = Counter(), Counter(), Counter()
        self._lock = RLock()
        self._admissions = self._deferrals = self._aborts = 0
        self._deferrals_by_resource = Counter()

    def _available(self, resource, *, emergency):
        available = self._limits[resource] - self._live[resource] - self._promised[resource]
        if not emergency:
            available -= max(0, self._reserve.get(resource, 0) - self._emergency_used[resource])
        return available

    def prepare(self, request_id, generation, amounts, *, emergency=False):
        """Reserve an operation's additional peak demand, or return None.

        Do not include already charged backing owners in ``amounts``. A failed
        multi-resource attempt leaves no partial reservation behind.
        """
        if (not isinstance(request_id, str) or not request_id or
                type(generation) is not int or generation < 0 or type(emergency) is not bool):
            raise ValueError('Invalid request/generation/emergency identity')
        amounts = _amounts(amounts, self._limits, nonempty=True)
        with self._lock:
            blocked = [name for name, amount in amounts.items()
                       if amount > self._available(name, emergency=emergency)]
            if blocked:
                self._deferrals += 1
                self._deferrals_by_resource.update(blocked)
                return None
            identity = ResourceIdentity(self.epoch, request_id, generation, next(self._sequence))
            grant = ResourceGrant(identity, tuple(sorted(amounts.items())), emergency, self._issuer)
            self._active[grant] = _Reservation(promised=amounts)
            self._promised.update(amounts)
            if emergency:
                self._emergency_used.update(amounts)
            self._admissions += 1
            return grant

    def _entry(self, grant):
        if type(grant) is not ResourceGrant or grant._issuer is not self._issuer:
            raise ValueError('Foreign or stale resource grant')
        entry = self._active.get(grant)
        if entry is None:
            entry = self._terminal.get(grant)
        if entry is None:
            raise ValueError('Unknown or recreated resource grant')
        return entry

    def dispatch(self, grant):
        """Commit the operation before resources are touched or work is queued."""
        with self._lock:
            entry = self._entry(grant)
            if entry.phase == 'dispatched':
                return
            if entry.phase != 'prepared':
                raise ValueError('Resource grant is no longer dispatchable')
            entry.phase = 'dispatched'

    def allocated(self, grant, amounts):
        """Report cumulative, still-owned allocations; never release capacity.

        Callers must not allocate beyond the grant even if a later cancellation
        or collection could free space. Settlement, not this report, accounts
        for a decrease in physical ownership.
        """
        amounts = _amounts(amounts, self._limits)
        with self._lock:
            entry = self._entry(grant)
            if entry.phase != 'dispatched':
                raise ValueError('Physical allocation requires a dispatched grant')
            demand = dict(grant.amounts)
            if (any(amount > demand.get(name, 0) for name, amount in amounts.items()) or
                    any(amounts.get(name, 0) < amount for name, amount in entry.live.items())):
                raise ValueError('Allocation exceeds grant or decreases unfenced ownership')
            for name, amount in amounts.items():
                delta = amount - entry.live.get(name, 0)
                self._promised[name] -= delta
                self._live[name] += delta
                entry.promised[name] -= delta
            entry.live = amounts

    def _completed(self, grant, expected_phase, completion):
        """Query user fences outside the ledger lock, then callers revalidate.

        A completion may need another worker to progress. Holding the ledger
        lock while invoking it would unnecessarily couple those worker locks.
        """
        with self._lock:
            if self._entry(grant).phase != expected_phase:
                return False
        query = getattr(completion, 'query', None)
        if not callable(query):
            raise ValueError('Capacity release requires a completion fence')
        return bool(query())

    def _terminalize(self, grant, entry):
        del self._active[grant]
        self._terminal[grant] = entry

    def abort(self, grant):
        """Return an unpublished operation's credit; cancellation is not abort."""
        with self._lock:
            entry = self._entry(grant)
            if entry.phase == 'aborted':
                return True
            if entry.phase != 'prepared':
                raise ValueError('Dispatched work must drain; it cannot be aborted')
            self._promised.subtract(entry.promised)
            if grant.emergency:
                self._emergency_used.subtract(entry.promised)
            entry.promised = {}
            entry.phase = 'aborted'
            self._aborts += 1
            self._terminalize(grant, entry)
            return True

    def settle(self, grant, completion, *, retained=None):
        """After completion, return unused promises and freed transient owners.

        By default all reported live allocations remain charged. If ``retained``
        is supplied, it must describe the remaining subset; the completion fence
        must also prove that every omitted allocation has actually been freed.
        Identical duplicate settlements are harmless. Conflicting duplicates
        are rejected, including after a grant's later release.
        """
        amounts = None if retained is None else _amounts(retained, self._limits)
        signature = None if amounts is None else tuple(sorted(amounts.items()))
        with self._lock:
            entry = self._entry(grant)
            if entry.phase in ('settled', 'released'):
                if entry.retained != signature:
                    raise ValueError('Conflicting duplicate resource settlement')
                return True
            if entry.phase != 'dispatched':
                raise ValueError('Only dispatched work can settle')
        if not self._completed(grant, 'dispatched', completion):
            # A concurrent identical settlement may already have completed.
            with self._lock:
                entry = self._entry(grant)
                if entry.phase in ('settled', 'released'):
                    if entry.retained != signature:
                        raise ValueError('Conflicting duplicate resource settlement')
                    return True
            return False
        with self._lock:
            entry = self._entry(grant)
            if entry.phase in ('settled', 'released'):
                if entry.retained != signature:
                    raise ValueError('Conflicting duplicate resource settlement')
                return True
            if entry.phase != 'dispatched':
                raise ValueError('Resource operation changed during completion')
            kept = entry.live if amounts is None else amounts
            if any(amount > entry.live.get(name, 0) for name, amount in kept.items()):
                raise ValueError('Settlement retains unreported physical ownership')
            released = Counter(entry.live)
            released.subtract(kept)
            self._live.subtract(released)
            self._promised.subtract(entry.promised)
            if grant.emergency:
                self._emergency_used.subtract(released)
                self._emergency_used.subtract(entry.promised)
            entry.live = dict(kept)
            entry.promised = {}
            entry.retained = signature
            entry.phase = 'settled' if any(kept.values()) else 'released'
            if entry.phase == 'released':
                self._terminalize(grant, entry)
            return True

    def release(self, grant, completion):
        """Return settled live credit only after its physical-free fence."""
        with self._lock:
            entry = self._entry(grant)
            if entry.phase == 'released':
                return True
            if entry.phase != 'settled':
                raise ValueError('Live release requires a settled operation')
        if not self._completed(grant, 'settled', completion):
            with self._lock:
                return self._entry(grant).phase == 'released'
        with self._lock:
            entry = self._entry(grant)
            if entry.phase == 'released':
                return True
            if entry.phase != 'settled':
                raise ValueError('Resource operation changed during release')
            self._live.subtract(entry.live)
            if grant.emergency:
                self._emergency_used.subtract(entry.live)
            entry.live = {}
            entry.phase = 'released'
            self._terminalize(grant, entry)
            return True

    def snapshot(self):
        """Detached diagnostics; snapshots never grant or reclaim capacity."""
        with self._lock:
            resources = {name: dict(hard_limit=limit, live=self._live[name],
                promised=self._promised[name],
                emergency=max(0, self._reserve.get(name, 0)-self._emergency_used[name]),
                emergency_limit=self._reserve.get(name, 0),
                emergency_used=self._emergency_used[name],
                available=self._available(name, emergency=False),
                emergency_available=self._available(name, emergency=True))
                for name, limit in self._limits.items()}
            return dict(epoch=self.epoch, resources=resources, active=len(self._active),
                phases=dict(Counter(entry.phase for entry in self._active.values())),
                admissions=self._admissions, deferrals=self._deferrals, aborts=self._aborts,
                deferrals_by_resource=dict(self._deferrals_by_resource))
