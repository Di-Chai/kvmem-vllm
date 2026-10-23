"""Short selection guards and deferred graph cleanup for native history GC.

Read leases keep bytes alive; a selection hold additionally keeps the scored
candidate domain live until native execution admission. Neither is a permanent
policy pin. Cleanup runs at refresh boundaries, never in per-token GC polling.
"""
from contextlib import ExitStack, contextmanager
from threading import Event
import time


_SYNC_RETRY_PAUSE = Event()


def enabled(authority):
    return (getattr(getattr(authority, 'gc', None), 'native_history', False)
            and getattr(authority, 'block_graph', None) is not None)


def retry_selection(authority, operation):
    if not enabled(authority):
        return operation()
    from . import _native
    # A failed hold/admission has not begun assembly. Recompute the complete
    # score domain, including graph normalization, rather than repick old scores.
    for _ in range(3):
        try:
            return operation()
        except _native.HistoryAdmissionRetry:
            continue
    raise _native.HistoryAdmissionRetry('Historical graph selection must retry at a later boundary')


def synchronous_selection(authority, operation, *, deadline=None):
    """Bounded backoff only for the existing synchronous refresh/restore path.

    Asynchronous callers yield their scheduler quantum instead. A persistent
    failure remains explicit; it must never turn into an unbounded busy loop.
    """
    if not enabled(authority):
        return operation()
    from . import _native
    for attempt in range(3):
        try:
            return operation()
        except _native.HistoryAdmissionRetry:
            if attempt == 2:
                raise
            remaining = .001 if deadline is None else min(.001, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError('Historical selection exceeded restore deadline')
            _SYNC_RETRY_PAUSE.wait(remaining)


@contextmanager
def selection_guard(authority, handles):
    if not enabled(authority):
        yield
        return
    from . import _native
    memory = authority.memory
    hold = authority.gc.scope.protect_selection(handles)
    if handles and hold is None:
        raise _native.HistoryAdmissionRetry('Historical graph candidate changed before protection')
    previous = getattr(memory, '_strict_history_selection', False)
    memory._strict_history_selection = True
    try:
        yield
    finally:
        memory._strict_history_selection = previous
        # Admitted plans own execution/CPU-read leases independently. Never
        # retain this policy veto for asynchronous assembly or model execution.
        hold = None


def note_retired(authority, record):
    if not enabled(authority):
        return
    graph = authority.block_graph
    pending = getattr(authority, '_graph_gc_pending', None)
    if pending is None:
        pending = authority._graph_gc_pending = {}
    key = record.handle.slot, record.handle.generation
    if key not in pending and len(pending) >= authority.memory.config.repository_blocks:
        # Recycled physical slots cannot grow an unbounded foreground mailbox.
        # An overflow requests a live-catalog reconciliation at the next refresh.
        pending.clear()
        authority._graph_gc_reconcile = True
    pending[key] = (graph.runtime_epoch, graph.branch_id, record.history_begin)


def compact_retired(authority, live_records=None):
    if not enabled(authority):
        return
    from .memory import Phase
    graph = authority.block_graph
    if authority.memory.phase is not Phase.READY or graph._pending_update is not None:
        return
    pending = getattr(authority, '_graph_gc_pending', {})
    epoch_changed = getattr(authority, '_graph_gc_epoch', None) != graph.runtime_epoch
    reconcile = getattr(authority, '_graph_gc_reconcile', False) or epoch_changed
    # No maintenance scan when nothing retired and no checkpoint/branch was
    # restored. Query already owns a live index when reconciliation is needed.
    if not pending and not reconcile:
        return
    full = reconcile or live_records is not None
    retired = {begin for epoch, branch, begin in pending.values()
               if (epoch, branch) == (graph.runtime_epoch, graph.branch_id)}
    # A branch can replace an old generation at the same logical start. The
    # old retirement notification must never delete that new branch's node.
    current = {r.history_begin: (r.handle.slot, r.handle.generation)
               for r in authority.records.values()}
    for key, (_, _, begin) in pending.items():
        if begin in current and current[begin] != key:
            retired.discard(begin)
    if full:
        if live_records is None:
            live = set()
            for record in authority.records.values():
                lease = authority.memory.repository.pin(record.handle)
                if lease:
                    live.add(record.history_begin)
                    lease.reset()
        else:
            live = {record.history_begin for record in live_records}
        # A verified P..M replay may retain a future graph beyond the present
        # authority prefix. Its temporarily absent nodes are not GC victims.
        end = authority.history_end
        for row in graph.rows:
            retired.update(start for start in (row.source, *row.targets)
                           if start < end and start not in live)
        partial = graph.partial
        if partial is not None:
            retired.update(start for start in (partial.source, *partial.targets)
                           if start < end and start not in live)
    try:
        graph.retire_blocks(retired)
    except (BufferError, BlockingIOError):
        # Candidate masks already exclude retired generations. Graph storage
        # reclamation can wait for COW/checkpoint memory to become available.
        authority._graph_gc_reconcile = True
        return
    pending.clear()
    authority._graph_gc_reconcile = False
    authority._graph_gc_epoch = graph.runtime_epoch


@contextmanager
def pressure_candidates(authority, candidates, mandatory=(), *, max_tokens=None):
    if not enabled(authority):
        yield candidates
        return
    memory = authority.memory
    rows, handles, seen = [], [], set()
    with ExitStack() as scope:
        for row in candidates:
            slot, generation, begin, count, _ = row
            if (slot in seen or type(count) is not int or count <= 0 or
                    begin < 0 or begin % memory.config.block_tokens or
                    begin + count > authority.history_end):
                raise ValueError('Invalid graph pressure candidate extent')
            seen.add(slot)
            handle = memory._store.handle(slot)
            if handle.generation != generation:
                raise ValueError('Stale graph pressure candidate generation/count')
            record = authority.records.get((slot, generation))
            if record is not None and (record.history_begin != begin or record.count != count):
                raise ValueError('Stale graph pressure candidate generation/count')
            lease = memory.repository.pin(handle)
            if not lease:
                if record is not None:
                    note_retired(authority, record)
                continue  # An admitted old GPU window may still contain this page.
            scope.callback(lease.reset)
            if record is None:
                raise ValueError('Live graph pressure candidate has no authority record')
            rows.append(row)
            handles.append(handle)
        for handle in mandatory:
            record = authority.records.get((handle.slot, handle.generation))
            if record is None or memory._store.handle(handle.slot) != handle:
                raise ValueError('Stale graph pressure mandatory generation')
        fixed = memory._store.fixed_selection_blocks(tuple(h.slot for h in mandatory),
            pressure=True, budget=max_tokens or 0)
        guarded = {h.slot for h in handles}
        for slot in fixed:
            if slot not in guarded:
                handles.append(memory._store.handle(slot))
        with selection_guard(authority, handles):
            compact_retired(authority)
            yield tuple(rows)
