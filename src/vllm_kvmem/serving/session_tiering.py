"""Opt-in CPU -> SSD session tiering (cpu_session_ttl_ms).

A session whose turn finished and that stays idle longer than the CPU TTL has
all its archived records written to SSD and their CPU bytes dropped (native
payload arena ``spill``: same slot, same generation). Before the session's next
turn is planned, the whole session is promoted back (``restore``: identical,
checksum-verified bytes), so no plan, pressure refresh or historical-GC read
admission ever sees a spilled record. Any other reader of a spilled record
promotes it synchronously (NativeMemoryTier.borrow_records).

Spill and promotion run on one background thread; the engine thread only
submits work and polls futures (an un-promoted restore simply waits, like a
busy restore lane). Partial or unarchived records always stay in RAM.
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import threading
import time
from ..core import settings as _settings

try:                                    # vLLM's logger prints in the engine process
    from vllm.logger import init_logger
    logger = init_logger('vllm.kvmem.session_tiering')
except Exception:                       # pragma: no cover - CPU-only environments
    logger = logging.getLogger(__name__)


def tiering_prefetch():
    """KVMEM_TIERING_PREFETCH (default 1, session tiering only): start a
    session's promotion when its next turn arrives at the scheduler, so it
    overlaps the request's queue wait. 0 promotes only when its restore is
    admitted. Same records, same bytes; only the timing moves."""
    value = _settings.get('KVMEM_TIERING_PREFETCH', '1')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_TIERING_PREFETCH must be 0 or 1')
    return value == '1'


def tiering_queue():
    """KVMEM_TIERING_QUEUE (default 1, session tiering only): a request whose
    session is not in CPU RAM (spilling, on SSD or being promoted) waits in
    vLLM's waiting queue -- the connector reports no matched tokens yet -- and
    holds no GPU slot until the worker reports it back in RAM. 0 admits it at
    once and lets its restore wait on the promotion inside a GPU slot."""
    value = _settings.get('KVMEM_TIERING_QUEUE', '1')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_TIERING_QUEUE must be 0 or 1')
    return value == '1'


def tiering_threads():
    """KVMEM_TIERING_THREADS (default 8, capped at the core count): parallel
    record I/O per session spill/promotion (pread/pwrite and SHA-256 release
    the GIL). Same records, same bytes."""
    value = _settings.get('KVMEM_TIERING_THREADS', '8')
    if not value.isdigit() or not 1 <= int(value) <= 64:
        raise ValueError('KVMEM_TIERING_THREADS must be an integer in [1, 64]')
    cores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    return max(1, min(int(value), cores))


class SessionTiering:
    def __init__(self, archive, ttl_seconds, *, clock=time.monotonic, executor=None, timer=None):
        if not ttl_seconds > 0:
            raise ValueError('Session tiering needs a positive CPU TTL')
        tier = getattr(archive, 'tier', None)
        if tier is None or not hasattr(tier, 'spill'):
            raise ValueError('Session tiering requires the native CPU payload archive')
        self.archive, self.tier, self.ttl, self.clock = archive, tier, ttl_seconds, clock
        # Separate workers: a promotion (a session's next turn waits on it) never
        # queues behind spills of other idle sessions.
        self.executor = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='kvmem-tiering')
        self.promoter = executor or ThreadPoolExecutor(max_workers=1, thread_name_prefix='kvmem-promote')
        threads = tiering_threads()
        self.io = (ThreadPoolExecutor(max_workers=threads, thread_name_prefix='kvmem-tiering-io')
                   if threads > 1 and executor is None else None)
        self.idle_since = {}        # namespace -> (time, authority)
        self.spilled = {}           # namespace -> allocations on SSD
        self.jobs = {}              # namespace -> (kind, future, allocations)
        self.stats = dict(spilled_sessions=0, promoted_sessions=0, spilled_records=0,
                          promoted_records=0, promote_waits=0)
        self.durations = {'spill': [], 'promote': []}     # seconds per session job (bounded)
        self.submitted = {}         # namespace -> perf_counter() at submit (queueing time)
        self.wanted = set()         # returning sessions whose spill was already running
        # Sessions not (fully) in CPU RAM, and the 'out'/'in' transitions the
        # scheduler has not seen yet (its waiting-queue gate, tiering_queue()).
        self.out, self.events = set(), []
        # Engine thread and the TTL timer share this state. The timer makes
        # spills independent of engine steps: an idle server (no requests, so
        # no steps and no tick()) still moves idle sessions to SSD on time.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._closing = False
        self._timer = None
        if timer if timer is not None else executor is None:
            self._timer = threading.Thread(target=self._timer_loop, name='kvmem-tiering-ttl', daemon=True)
            self._timer.start()

    def _allocations(self, authority):
        namespace = authority.memory.request_id
        out = []
        entries = getattr(self.archive, '_entries', {})
        for record in list(authority.records.values()):
            if not getattr(record, 'archived', False) or record.raw is not None:
                continue
            entry = entries.get((namespace, record.handle.slot, record.handle.generation))
            if entry is not None and getattr(entry, 'ready', True):
                out.append(entry.allocation)
        return out

    def _mark_out(self, namespace):
        if namespace not in self.out:
            self.out.add(namespace)
            self.events.append(('out', namespace))

    def _mark_in(self, namespace):
        if namespace in self.out:
            self.out.discard(namespace)
            self.events.append(('in', namespace))

    def take_events(self):
        """Engine thread: tier transitions for the scheduler (oldest first)."""
        with self._lock:
            events, self.events = self.events, []
        return events

    def _reap(self, namespace):
        job = self.jobs.get(namespace)
        if job is None:
            return True
        kind, future, allocations = job
        if not future.done():
            return False
        del self.jobs[namespace]
        count, seconds, started = future.result()   # an I/O failure is fatal, never silent
        queued = max(0.0, started - self.submitted.pop(namespace, started))
        gb = count * getattr(self.tier, "record_bytes", 0) / 1e9
        logger.info('session tiering: %s %s %d records %.2f GB in %.3f s (%.2f GB/s), queued %.3f s', kind,
                    namespace, count, gb, seconds, gb / seconds if seconds > 0 else 0.0, queued)
        samples = self.durations[kind]
        samples.append(seconds)
        del samples[:-256]
        if kind == 'spill':
            self.spilled[namespace] = allocations
            self.stats['spilled_sessions'] += 1
            self.stats['spilled_records'] += count
        else:
            self.spilled.pop(namespace, None)
            self.stats['promoted_sessions'] += 1
            self.stats['promoted_records'] += count
            self._mark_in(namespace)
        return True

    def _timed(self, fn, allocations):
        started = time.perf_counter()
        spill = fn == self.tier.spill and hasattr(self.tier, 'flush_spilled')
        written = [] if spill else None
        call = (lambda part: fn(part, written=written)) if spill else fn
        try:
            if self.io is None or len(allocations) < 64:
                count = call(allocations)
            else:
                threads = self.io._max_workers
                if hasattr(getattr(self.tier, 'payloads', None), 'read_records'):
                    # Native batches: one chunk per thread (each chunk is one GIL release).
                    step = max(16, -(-len(allocations) // threads))
                else:
                    step = max(16, len(allocations) // (4 * threads))
                parts = [allocations[i:i + step] for i in range(0, len(allocations), step)]
                count = sum(f.result() for f in [self.io.submit(call, part) for part in parts])
        finally:
            if spill:
                self.tier.flush_spilled(written)        # one device sync per session
        return count, time.perf_counter() - started, started

    def idle(self, namespace, authority):
        """The session's turn finished (engine thread). Its archived records are
        listed now, so the TTL timer never reads engine-owned state."""
        with self._cond:
            self.idle_since[namespace] = (self.clock(), authority, self._allocations(authority))
            self._cond.notify()

    def forget(self, namespace):
        """The session was evicted: its records are released by the archive."""
        with self._lock:
            self.idle_since.pop(namespace, None)
            self.spilled.pop(namespace, None)
            self.wanted.discard(namespace)
            self._mark_in(namespace)            # gone: nothing left to wait for

    def _submit_due(self, now, *, fresh):
        """Start at most one spill past the TTL (caller holds the lock). One
        spill at a time, and none while a promotion runs: a promotion (a turn
        waits on it) gets the device; spills use only its idle time. ``fresh``:
        engine thread, re-list the records; otherwise use the idle() list."""
        if any(kind == 'promote' or not future.done() for kind, future, _ in self.jobs.values()):
            return False
        for namespace, (since, authority, listed) in list(self.idle_since.items()):
            if namespace in self.jobs or namespace in self.spilled or now - since < self.ttl:
                continue
            allocations = self._allocations(authority) if fresh else listed
            del self.idle_since[namespace]
            if allocations:
                self.submitted[namespace] = time.perf_counter()
                self._mark_out(namespace)
                self.jobs[namespace] = ('spill', self.executor.submit(self._timed, self.tier.spill, allocations),
                                        allocations)
                return True
        return False

    def _timer_loop(self):
        with self._cond:
            while not self._closing:
                busy = any(kind == 'promote' or not future.done() for kind, future, _ in self.jobs.values())
                due = min((since + self.ttl for since, _, _ in self.idle_since.values()), default=None)
                now = self.clock()
                if due is None:
                    self._cond.wait(1.0)
                elif busy or due > now:
                    self._cond.wait(0.05 if busy else max(0.001, due - now))
                else:
                    self._submit_due(now, fresh=False)
                    self._cond.wait(0.05)

    def _promote(self, namespace):
        self.submitted[namespace] = time.perf_counter()
        self.jobs[namespace] = ('promote', self.promoter.submit(self._timed, self.tier.promote,
                                                                self.spilled[namespace]), self.spilled[namespace])

    def prefetch(self, namespace):
        """The session's next turn arrived at the scheduler (engine thread,
        non-blocking): start bringing it back now. ready() still gates the
        restore, so this only moves the promotion earlier."""
        with self._lock:
            self._prefetch(namespace)

    def _prefetch(self, namespace):
        self.idle_since.pop(namespace, None)
        job = self.jobs.get(namespace)
        if job is not None:
            if job[0] == 'spill':
                if job[1].cancel():
                    del self.jobs[namespace]    # the spill never started: still in RAM
                    self.submitted.pop(namespace, None)
                    self.stats['spills_cancelled'] = self.stats.get('spills_cancelled', 0) + 1
                    self._mark_in(namespace)
                else:
                    self.wanted.add(namespace)  # promote once the running spill is reaped
            return
        if self.spilled.get(namespace):
            self._promote(namespace)
            self.stats['prefetched'] = self.stats.get('prefetched', 0) + 1
        else:
            self._mark_in(namespace)

    def tick(self):
        """Engine thread, once per step: reap jobs, start spills past the TTL."""
        with self._lock:
            self._tick()

    def _tick(self):
        for namespace in list(self.jobs):
            self._reap(namespace)
        for namespace in list(self.wanted):
            if namespace not in self.jobs:
                self.wanted.discard(namespace)
                if self.spilled.get(namespace):
                    self._promote(namespace)
                    self.stats['prefetched'] = self.stats.get('prefetched', 0) + 1
        self._submit_due(self.clock(), fresh=True)

    def ready(self, namespace, authority):
        """Before a session's next turn is planned (engine thread, non-blocking).
        False while its spill or promotion is still running."""
        with self._lock:
            return self._ready(namespace)

    def _ready(self, namespace):
        self.idle_since.pop(namespace, None)
        job = self.jobs.get(namespace)
        if job is not None and job[0] == 'spill' and job[1].cancel():
            del self.jobs[namespace]            # the spill never started: still in RAM
            self.submitted.pop(namespace, None)
            self.stats['spills_cancelled'] = self.stats.get('spills_cancelled', 0) + 1
            self._mark_in(namespace)
            return True
        if not self._reap(namespace):
            self.stats['promote_waits'] += 1
            return False
        self.wanted.discard(namespace)
        if not self.spilled.get(namespace):
            self._mark_in(namespace)
            return True
        self._promote(namespace)
        self.stats['promote_waits'] += 1
        return False

    def ensure(self, namespace, authority):
        """Synchronous restore paths: wait for the session to be in RAM."""
        while not self.ready(namespace, authority):
            with self._lock:
                kind, future, _ = self.jobs[namespace]
            future.result()

    def snapshot(self):
        spill_stats = getattr(self.tier, 'spill_stats', {})
        def p50(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2] if xs else None
        return dict(self.stats, idle_sessions=len(self.idle_since), spilled_now=len(self.spilled),
                    jobs=len(self.jobs), ttl_seconds=self.ttl, tier=dict(spill_stats),
                    spill_seconds_p50=p50(self.durations['spill']),
                    promote_seconds_p50=p50(self.durations['promote']))

    def close(self):
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        if self._timer is not None:
            self._timer.join(timeout=5)
        self.executor.shutdown(wait=True)
        if self.promoter is not self.executor:
            self.promoter.shutdown(wait=True)
        if self.io is not None:
            self.io.shutdown(wait=True)
