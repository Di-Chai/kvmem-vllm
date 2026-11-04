"""Accepted-token feedback for one native response across layerwise refreshes.

This pressure history is independent of the legacy non-thinking action parser.
Rows are sampled in original consumed-token coordinates, including thinking;
only scheduler-confirmed rows publish mass. No draft or rejected Q contributes.
"""
from dataclasses import dataclass, field

from . import _native


def sample_rows(begin, count, interval):
    if begin < 0 or count < 0 or interval <= 0:
        raise ValueError('Invalid rolling attention sampling coordinates')
    return [row for row in range(count) if (begin + row + 1) % interval == 0]


@dataclass
class DecodeHistory:
    mass: dict = field(default_factory=dict)
    samples: int = 0
    end: int = 0
    chunks: int = 0

    def prepare(self, pending, accepted, logical, masses):
        if not 0 <= accepted <= pending.candidate_count:
            raise ValueError('Invalid rolling attention accepted prefix')
        begin = pending.history_begin
        interval = pending.state.options.action_interval
        wanted = set(sample_rows(begin, accepted, interval))
        indices = [i for i, row in enumerate(pending.rows)
                   if row in wanted and begin + row >= self.end]
        if wanted - set(pending.rows):
            raise RuntimeError('Rolling attention omitted an accepted sampled row')
        if not indices:
            return self
        values = masses.numpy() if hasattr(masses, 'numpy') else masses
        mass, samples = _native.prepare_decode_attention(self.mass, self.samples,
            indices, logical, values,
            getattr(pending.state.options, 'action_commit_max_bytes', 16777216))
        return DecodeHistory(mass, samples, begin + accepted, self.chunks + 1)

    def scores(self):
        return {key: value / self.samples for key, value in self.mass.items()} if self.samples else {}


def record_map(records):
    cursor, result = 0, []
    for r in records:
        result.append((r.handle.slot, r.handle.generation, r.history_begin, r.count, cursor))
        cursor += r.count
    return tuple(result)


def segment_identities(records, begin, local, count, starts, counts):
    """Map either split or canonically merged native partial-tail segments.

    A committed partial block and its current speculative continuation may be
    two window_segments entries but one native score column. A score column
    must still belong entirely to one original block in each target layer.
    """
    from .action_attention import window_segments
    ls, lc, lh = window_segments(records, begin, local, count, 32)
    cursor = index = 0
    result = []
    for start, size in zip(starts, counts, strict=True):
        start, size = int(start), int(size)
        if start != cursor or size <= 0:
            raise RuntimeError('Rolling score segments do not partition the native window')
        end = start + size
        identity = None
        while cursor < end:
            if index >= len(ls) or not ls[index] <= cursor < ls[index]+lc[index]:
                raise RuntimeError('Rolling layer map changed physical segment geometry')
            if identity is not None and identity != lh[index]:
                raise RuntimeError('One rolling score column crosses different logical blocks')
            identity = lh[index]
            cursor = min(end, ls[index]+lc[index])
            if cursor == ls[index]+lc[index]: index += 1
        # Native segment arrays carry NumPy scalars. Logical coordinates also
        # feed the legacy graph, whose canonical key contract is Python int.
        result.append(int(identity))
    if cursor != local+count or index != len(ls):
        raise RuntimeError('Rolling score segments omit part of the native window')
    return result


def attach_packet(controller, pending, rid, authority, starts, counts, masses):
    begin, count = pending.history_begin, pending.candidate_count
    local = sum(map(int, counts)) - count
    logical = []
    for layer in range(len(controller.names)):
        records = controller.records(rid, authority, begin, local, layer)
        logical.append(segment_identities(records, begin, local, count, starts, counts))
    # Before the first decode refresh the shared frame still describes the
    # initial global Query preselection. Feedback must describe real layer zero.
    pending.history = logical[0]
    pending.layerwise_decode = controller, rid, logical, masses
    controller.decode_histories.setdefault(rid, [DecodeHistory() for _ in controller.names])


def prepare_accept(controller, pending, accepted):
    owner, rid, logical, masses = pending.layerwise_decode
    if owner is not controller or masses.shape[0] != len(controller.names):
        raise RuntimeError('Rolling feedback belongs to a different layer domain')
    histories = controller.decode_histories[rid]
    staged = [history.prepare(pending, accepted, logical[i], masses[i])
              for i, history in enumerate(histories)]
    if len({(h.samples, h.end, h.chunks) for h in staged}) != 1:
        raise RuntimeError('Rolling layers disagree on scheduler acceptance')
    return rid, staged, 'decode'


def validate_configuration(profile, *, query_mode, optimized, pressure, single_assembly, original_rope):
    kv = profile.get('kvmem', {})
    if (query_mode != 'sequential' or not optimized or not pressure or
            not single_assembly or original_rope):
        raise ValueError('Layerwise rolling requires optimized sequential Query, per-layer pressure, '
                         'single assembly and compact positions')
    if not all(kv.get(key) for key in ('rolling_active_window', 'request_planning',
                                      'async_refresh', 'action_attention', 'prefill_attention')):
        raise ValueError('Layerwise rolling requires rolling request planning and accepted attention')
    if kv.get('pressure_policy') != 'attention' or kv.get('repository_gc', False):
        raise ValueError('Layerwise rolling requires attention pressure and retained CPU history')
    if profile.get('draft_method', 'mtp') != 'mtp':
        raise ValueError('Layerwise rolling currently requires the MTP draft contract')
