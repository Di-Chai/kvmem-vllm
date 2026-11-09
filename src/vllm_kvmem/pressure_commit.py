"""An immutable, owner-bound map prepared with the original pressure ranking."""
from dataclasses import dataclass
from itertools import accumulate


@dataclass(frozen=True)
class PreparedPressureCommit:
    authority: object
    request_id: str
    history: int
    chosen: object
    maps: tuple
    layer_zero_starts: frozenset
    tokens: int

    def validate(self, job):
        if (job.authority is not self.authority or
                job.command.request_id != self.request_id or job.history != self.history or
                job.layerwise_pressure_plan[0] is not self.chosen or
                job.selected_tokens != self.tokens):
            raise RuntimeError('Prepared pressure maps lost their accepted owner/selection')


def _native_commit_map():
    import os
    value = os.getenv('KVMEM_PRESSURE_NATIVE_COMMIT_MAP', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_COMMIT_MAP must be 0 or 1')
    if value != '1':
        return None
    from . import _native
    return getattr(_native, 'commit_map', None)


def native_commit_table():
    """KVMEM_PRESSURE_NATIVE_COMMIT_TABLE: build commit maps from a native identity
    table kept with the record index. Lookups and checks run without the GIL and
    rows are created one layer at a time (same maps, same errors)."""
    import os
    value = os.getenv('KVMEM_PRESSURE_NATIVE_COMMIT_TABLE', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_PRESSURE_NATIVE_COMMIT_TABLE must be 0 or 1')
    if value != '1':
        return None
    from . import _native
    return getattr(_native, 'IdentityTable', None)


def identity_table(controller, request_id, records, record_slots):
    """The request's identity table, extended to the sorted record index; None when off."""
    cls = native_commit_table()
    if cls is None:
        return None
    tables = controller.__dict__.setdefault('_identity_tables', {})
    table = tables.get(request_id)
    if table is None or not table.matches(records):
        table = tables[request_id] = cls()
    table.extend(records, record_slots)
    return table


def prepare_commit(job, chosen, records, record_slots, selected_slots, table=None):
    """Read each selected handle's generation once, away from main-thread ACK.

    Ranking/order remain the original preview's. The published row format is
    exactly (slot, generation, history_begin, count, dense_position).
    """
    from .gil_gate import checkpoint
    rid = getattr(getattr(job, 'command', None), 'request_id', None)
    checkpoint(rid)         # opt-in KVMEM_GIL_GATE_MS (no-op on the engine thread)
    if len(chosen) != len(selected_slots) or not chosen:
        raise RuntimeError('Prepared pressure maps must cover all selected layers')
    if table is not None:
        maps, length = table.build([list(s) for s in selected_slots], [list(p) for p in chosen], records)
        return PreparedPressureCommit(job.authority, job.command.request_id, job.history,
            chosen, maps, frozenset(r.history_begin for r in chosen[0]), length)
    selected = set().union(*selected_slots)
    identities = {slot:(slot, r.handle.generation, r.history_begin, r.count)
                  for r, slot in zip(records, record_slots) if slot in selected}
    if len(identities) != len(selected):
        raise RuntimeError('Selected pressure identity is missing from the accepted archive')
    maps, lengths = [], set()
    native = _native_commit_map()
    for picked, slots in zip(chosen, selected_slots):
        checkpoint(rid)     # per layer, so a closed gate can interleave a long build
        if native is not None:
            # Same lookups, checks, errors and rows, without a Python loop per row.
            rows, length = native(list(slots), identities, list(picked))
            maps.append(rows)
            lengths.add(length)
            continue
        entries = [identities[slot] for slot in slots]
        if len(entries) != len(picked) or any(
                (entry[2], entry[3]) != (r.history_begin, r.count)
                for entry, r in zip(entries, picked)):
            raise RuntimeError('Prepared pressure map differs from the selected record order')
        offsets = accumulate((entry[3] for entry in entries), initial=0)
        maps.append(tuple((*entry, offset) for entry, offset in zip(entries, offsets)))
        lengths.add(sum(entry[3] for entry in entries))
    if len(lengths) != 1:
        raise RuntimeError('Prepared pressure maps changed the per-layer token extent')
    return PreparedPressureCommit(job.authority, job.command.request_id, job.history,
        chosen, tuple(maps), frozenset(r.history_begin for r in chosen[0]), lengths.pop())
