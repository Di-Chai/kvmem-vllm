"""Join all TP workers' feedback, including acknowledgements on later steps."""
from dataclasses import fields, replace

from ..core.archive_admission import ArchiveCapacity


class TensorParallelFeedback:
    def __init__(self, size):
        self.size = size
        self.commands = {}
        self.notices = {}
        self.partial_captures = set()
        self.idle_capacity = {}
        self.archive = {}
        self.publication = {}
        self.publication_dirty = set()
        self.blocked = set()
        self.usage = {}
        self.archive_sequence = self.publication_sequence = 0

    def _complete(self, pending, key, rank, value):
        group = pending.setdefault(key, {})
        if rank in group:
            raise ValueError('Duplicate tensor-parallel worker acknowledgement')
        group[rank] = value
        if len(group) == self.size:
            del pending[key]
            return tuple(group[i] for i in range(self.size))
        return None

    @staticmethod
    def _notice(values):
        first = values[0]
        capture = any(n.kind == 'capture_deferred' for n in values)
        common = dict(ready_records=min(n.ready_records for n in values),
                      allocated_records=max(n.allocated_records for n in values),
                      protected_records=max(n.protected_records for n in values),
                      resident_ready=all(n.resident_ready for n in values))
        if capture:
            common['kind'] = 'capture_deferred'
        result = replace(first, **common)
        for notice in values:
            other = replace(notice, **common)
            if other != result:
                differing = [f.name for f in fields(result) if getattr(result, f.name) != getattr(other, f.name)]
                raise ValueError(f'Tensor-parallel {result.kind} checkpoint differs in {differing}')
        return result

    @staticmethod
    def _snapshot(states, rank, seq, value):
        if rank in states and seq <= states[rank][0]:
            raise ValueError('Stale tensor-parallel worker snapshot')
        states[rank] = (seq, value)

    def collect(self, reports, pending_commands):
        from .connector import ControlCompletions
        result = ControlCompletions()
        archive_changed = publication_changed = False
        for rank, report in sorted(reports.items()):
            if not 0 <= rank < self.size or report.shards is not None:
                raise ValueError('Invalid tensor-parallel worker identity')
            if report.systemone_notices:
                raise ValueError('Unexpected fork feedback for tensor-parallel serving')
            # Scheduler timing policies count one model step per batch.
            if rank == 0:
                result.step_timings = report.step_timings
                result.restore_timings = report.restore_timings
            for rid, command in report.completed:
                if pending_commands.get(rid) != command:
                    raise ValueError('Stale or foreign worker completion')
                windows = self._complete(self.commands, (rid, command), rank,
                                         report.selected_windows.get(rid))
                if windows is not None:
                    if len(set(windows)) != 1:
                        raise ValueError('Tensor-parallel workers disagree on the selected window')
                    result.completed.append((rid, command))
                    if windows[0] is not None:
                        result.selected_windows[rid] = windows[0]
            for notice in report.sessions:
                owner = (notice.namespace, notice.generation)
                if notice.kind == 'history_capacity':
                    group = self.idle_capacity.setdefault(owner, {})
                    group[rank] = notice
                    if len(group) == self.size and len({n.history for n in group.values()}) == 1:
                        result.sessions.append(self._notice(tuple(group.values())))
                    continue
                capture = notice.kind in ('captured', 'capture_deferred')
                key = (*owner, notice.request_id, ('capture', notice.history) if capture else notice.kind)
                values = self._complete(self.notices, key, rank, notice)
                if values is not None:
                    binding = (*owner, notice.request_id)
                    if capture and len({n.kind for n in values}) > 1:
                        self.partial_captures.add(binding)
                    if notice.kind == 'retired' and binding in self.partial_captures:
                        from .session_control import SessionNotice
                        self.partial_captures.remove(binding)
                        result.sessions.append(SessionNotice('retired', *owner, notice.request_id,
                                                             discard_shards=True))
                    else:
                        result.sessions.append(self._notice(values))
                    if notice.kind in ('restored', 'evicted'):
                        self.idle_capacity.pop(owner, None)
            if report.cpu_archive_bytes is not None:
                self.usage[rank] = report.cpu_archive_bytes
                # The configured CPU budget and record width are per worker.
                result.cpu_archive_bytes = max(self.usage.values())
            if report.archive_sequence is not None:
                self._snapshot(self.archive, rank, report.archive_sequence, report.archive_capacity)
                archive_changed = True
            if report.publication_sequence is not None:
                self._snapshot(self.publication, rank, report.publication_sequence,
                               report.publication_blocked)
                self.publication_dirty.add(rank)
                publication_changed = True
        if archive_changed and len(self.archive) == self.size:
            self.archive_sequence += 1
            result.archive_sequence = self.archive_sequence
            caps = [cap for _, cap in self.archive.values()]
            for rid in set().union(*caps):
                rows = [cap.get(rid, ArchiveCapacity(0, 0)) for cap in caps]
                result.archive_capacity[rid] = ArchiveCapacity(min(c.history_end for c in rows),
                                                               min(c.ready_records for c in rows))
        if publication_changed:
            observed = set().union(*(blocked for _, blocked in self.publication.values()))
            if len(self.publication_dirty) == self.size:
                self.blocked = observed
                self.publication_dirty.clear()
            else:
                # A late shard may introduce a new fence for the same owner.
                # Widen immediately; release only on a fresh report from each.
                self.blocked.update(observed)
            self.publication_sequence += 1
            result.publication_sequence = self.publication_sequence
            result.publication_blocked = self.blocked.copy()
        return result
