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
        return _prepare_history(self, self.mass, pending, accepted, logical, masses)

    def scores(self):
        return {key: value / self.samples for key, value in self.mass.items()} if self.samples else {}


def _prepare_history(history, source, pending, accepted, logical, masses):
    """Stage the next accepted-mass state without mutating ``history``."""
    if not 0 <= accepted <= pending.candidate_count:
        raise ValueError('Invalid rolling attention accepted prefix')
    begin = pending.history_begin
    interval = pending.state.options.action_interval
    wanted = set(sample_rows(begin, accepted, interval))
    indices = [i for i, row in enumerate(pending.rows)
               if row in wanted and begin + row >= history.end]
    if wanted - set(pending.rows):
        raise RuntimeError('Rolling attention omitted an accepted sampled row')
    if not indices:
        return history
    values = masses.numpy() if hasattr(masses, 'numpy') else masses
    cap = getattr(pending.state.options, 'action_commit_max_bytes', 16777216)
    if mass_map_enabled():
        if not isinstance(source, (dict, _native.DecodeMassMap)):
            source = history.mass
        mass_map, samples = _native.prepare_decode_attention_map(source, history.samples, indices, logical,
                                                                 values, cap)
        return NativeDecodeHistory(mass_map, samples, begin + accepted, history.chunks + 1)
    if isinstance(source, _native.DecodeMassMap if hasattr(_native, 'DecodeMassMap') else ()):
        source = history.mass
    prepare = (_native.prepare_decode_attention_sparse if sparse_mass_enabled()
               else _native.prepare_decode_attention)
    mass, samples = prepare(source, history.samples, indices, logical, values, cap)
    return DecodeHistory(mass, samples, begin + accepted, history.chunks + 1)


class NativeDecodeHistory:
    """DecodeHistory whose accepted mass stays in a native DecodeMassMap
    (KVMEM_NATIVE_DECODE_MASS_MAP). ``mass`` materializes the identical dict
    (same keys, insertion order and float values) on first use, for
    checkpoints, swaps, historical GC and scores. Instances are never mutated."""

    __hash__ = None

    def __init__(self, mass_map, samples, end, chunks):
        self._map, self._mass = mass_map, None
        self.samples, self.end, self.chunks = samples, end, chunks

    @property
    def mass(self):
        if self._mass is None:
            self._mass = self._map.to_dict()
        return self._mass

    def prepare(self, pending, accepted, logical, masses):
        return _prepare_history(self, self._map, pending, accepted, logical, masses)

    def scores(self):
        return {key: value / self.samples for key, value in self.mass.items()} if self.samples else {}

    def __eq__(self, other):
        fields = ('mass', 'samples', 'end', 'chunks')
        if not all(hasattr(other, f) for f in fields):
            return NotImplemented
        return all(getattr(self, f) == getattr(other, f) for f in fields)

    def __repr__(self):
        return (f'NativeDecodeHistory(mass=<{len(self._map)} keys>, samples={self.samples}, '
                f'end={self.end}, chunks={self.chunks})')


def mass_map_enabled():
    """KVMEM_NATIVE_DECODE_MASS_MAP=1: keep accepted decode mass natively."""
    import os
    value = os.getenv('KVMEM_NATIVE_DECODE_MASS_MAP', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_MASS_MAP must be 0 or 1')
    return value == '1' and hasattr(_native, 'prepare_decode_attention_map')


def record_map(records):
    cursor, result = 0, []
    for r in records:
        result.append((r.handle.slot, r.handle.generation, r.history_begin, r.count, cursor))
        cursor += r.count
    return tuple(result)


def sparse_mass_enabled():
    """KVMEM_NATIVE_DECODE_SPARSE_MASS=1: prepare_decode_attention_sparse, which
    returns the same dictionary but writes only changed entries after copy()."""
    import os
    value = os.getenv('KVMEM_NATIVE_DECODE_SPARSE_MASS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_SPARSE_MASS must be 0 or 1')
    return value == '1' and hasattr(_native, 'prepare_decode_attention_sparse')


def native_segments_enabled():
    """KVMEM_NATIVE_DECODE_SEGMENTS=1: compute segment_identities in C++.

    The native helper returns None for any input outside its exact int64 domain
    and for every input that would raise; the Python code below then runs and
    raises its original error. Equivalence: tests/test_native_decode_segments.py.
    """
    import os
    value = os.getenv('KVMEM_NATIVE_DECODE_SEGMENTS', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_SEGMENTS must be 0 or 1')
    return value == '1' and hasattr(_native, 'decode_segment_identities')


def segment_identities(records, begin, local, count, starts, counts):
    """Map either split or canonically merged native partial-tail segments.

    A committed partial block and its current speculative continuation may be
    two window_segments entries but one native score column. A score column
    must still belong entirely to one original block in each target layer.
    """
    if native_segments_enabled():
        result = _native.decode_segment_identities(records, begin, local, count, starts, counts, 32)
        if result is not None:
            return result
    return _segment_identities_python(records, begin, local, count, starts, counts)


def _segment_identities_python(records, begin, local, count, starts, counts):
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


def layer_map_enabled():
    """KVMEM_NATIVE_DECODE_LAYER_MAP=1: per-layer decode identities straight from
    the frozen pressure frame (parsed once per frame), without building record lists."""
    import os
    value = os.getenv('KVMEM_NATIVE_DECODE_LAYER_MAP', '0')
    if value not in ('0', '1'):
        raise ValueError('KVMEM_NATIVE_DECODE_LAYER_MAP must be 0 or 1')
    return value == '1' and hasattr(_native, 'decode_identities_from_map')


_LAYER_MAP_CACHE_LIMIT = 1024


def _fused_identities(controller, rid, begin, local, count, starts, counts):
    """All layers' identities via the native fused path, or None to run Python.

    Equals segment_identities(controller.records(rid, authority, begin, local, layer), ...)
    for every layer; any input the native helper cannot reproduce exactly (including
    every error case) returns None so the original per-layer code runs and raises.
    """
    from .layerwise_pressure import LayerwisePrefillPressure
    if getattr(type(controller), 'records', None) is not LayerwisePrefillPressure.records:
        return None
    frame = controller.resolved_maps(rid)
    if frame is None:
        return None
    origin, layers = frame
    cache = controller.__dict__.setdefault('_decode_layer_maps', {})
    result = []
    for layer in range(len(controller.names)):
        records = layers[layer]
        hit = cache.get((rid, layer))
        if hit is None or hit[0] is not records:
            parsed = _native.parse_layer_map(records)
            if parsed is None:
                return None
            if len(cache) >= _LAYER_MAP_CACHE_LIMIT:
                cache.clear()
            cache[(rid, layer)] = hit = (records, parsed)
        identities = _native.decode_identities_from_map(hit[1], origin, begin, local, count, starts, counts)
        if identities is None:
            return None
        result.append(identities)
    return result


def attach_packet(controller, pending, rid, authority, starts, counts, masses):
    begin, count = pending.history_begin, pending.candidate_count
    local = sum(map(int, counts)) - count
    logical = (_fused_identities(controller, rid, begin, local, count, starts, counts)
               if layer_map_enabled() else None)
    if logical is None:
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
