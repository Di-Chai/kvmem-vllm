"""Versioned one-hop propagation over original logical block starts.

Both operators read one immutable seed vector. V1 uses stored attention mass;
V2 normalizes its surviving edges for each selection's transport mask. Neither
version changes the stored rows or refills edges excluded from their top-k.
"""
from dataclasses import dataclass, field
from bisect import bisect_right
import math
import time


@dataclass(frozen=True)
class GraphScores:
    scores: object
    diagnostics: dict
    trace: tuple = ()
    _charge: object = field(default=None, repr=False, compare=False)

    def close(self):
        # Drop owned payload before returning its bytes. Closed result objects
        # can remain referenced without retaining an unaccounted full vector.
        object.__setattr__(self, 'scores', None)
        object.__setattr__(self, 'diagnostics', None)
        object.__setattr__(self, 'trace', ())
        charge = self._charge
        object.__setattr__(self, '_charge', None)
        if charge is not None:
            charge.close()

    def __enter__(self):
        if self.scores is None:
            raise RuntimeError('Graph score result is released')
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        charge = getattr(self, '_charge', None)
        if charge is not None:
            charge.close()


def scoring_scratch_bytes(nodes, observed_sources, message_spans, *, trace=False):
    """Conservative CPython working/output bound, not serialized payload size.

    1024 bytes/index node cover two dictionaries, validation/observed-source
    sets, initial/final/incoming/outgoing vectors, sorting and returned scores;
    128 bytes/observed source cover rows missing from the eligible index;
    512 bytes/message span cover the boundary lookup's set, lists and tuples.
    Fixed 32 KiB covers diagnostics/frames; explicit trace adds 256 KiB for
    at most 64 target records with four contributors each. Keep the complete
    bound until the returned result is released, even after temporaries die.
    """
    return 32768 + nodes * 1024 + observed_sources * 128 + message_spans * 512 + (262144 if trace else 0)


class _LocalityBoundaries:
    def __init__(self, spans, ends_known=None):
        self.spans = spans
        self.starts = tuple(item[0] for item in spans)
        ends_known = (True,) * len(spans) if ends_known is None else ends_known
        self.boundaries = tuple(sorted(set([begin for begin, _end, _role in spans] +
            [end for (_begin, end, _role), known in zip(spans, ends_known) if known])))

    def coefficient(self, source, target, block_tokens, cross_message_weight):
        category = self.classify(source, target, block_tokens)
        return (cross_message_weight if category == 'cross_message' else 1.,
                category == 'unknown')

    def classify(self, source, target, block_tokens):
        low, high = min(source, target), max(source, target) + block_tokens
        index = bisect_right(self.starts, low) - 1
        if index >= 0 and high <= self.spans[index][1]:
            return 'same_message'
        boundary = bisect_right(self.boundaries, low)
        if boundary < len(self.boundaries) and self.boundaries[boundary] < high:
            return 'cross_message'
        return 'unknown'


def locality_boundary(source, target, block_tokens, spans, cross_message_weight, ends_known=None):
    """Known full containment, known internal boundary, or unknown fallback."""
    return _LocalityBoundaries(spans, ends_known).coefficient(
        source, target, block_tokens, cross_message_weight)


def _operator_weights(mode, scoring_version, alpha, locality_weight,
                      attention_weight, cross_message_weight):
    if mode not in ('off', 'collect', 'locality', 'attention', 'multi'):
        raise ValueError('Invalid graph scoring mode')
    if scoring_version not in ('v1', 'v2'):
        raise ValueError('Invalid graph scoring version')
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
            not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('Invalid graph alpha')
    if not math.isfinite(cross_message_weight) or not 0 <= cross_message_weight <= 1:
        raise ValueError('Invalid graph propagation weights')
    if scoring_version == 'v2':
        return (alpha / 2 if mode == 'multi' else alpha if mode == 'locality' else 0.,
                alpha / 2 if mode == 'multi' else alpha if mode == 'attention' else 0.)
    if (any(not math.isfinite(w) or w < 0 for w in (locality_weight, attention_weight)) or
            locality_weight + attention_weight > 1):
        raise ValueError('Invalid graph propagation weights')
    return (locality_weight if mode in ('locality', 'multi') else 0.,
            attention_weight if mode in ('attention', 'multi') else 0.)


def score_graph(raw_scores, starts, eligible, snapshot, *, mode='multi',
                scoring_version='v1', alpha=.2,
                locality_weight=.1, attention_weight=.2, cross_message_weight=.25,
                block_tokens=None, trace=False, trace_targets=()):
    """Propagate Query or pressure seeds; ``eligible`` is the transport mask.

    Ineligible seeds remain exactly unchanged, including nonzero mandatory
    seeds. Zero-score eligible nodes may receive mass. V2's alpha is shared by
    its operators; multi always assigns half to each, including identity rows.
    Legacy weight conflicts are resolved by the user-facing configuration
    parser; these compatibility arguments have no effect on the V2 operator.
    """
    if len(raw_scores) != len(starts) or len(eligible) != len(starts):
        raise ValueError('Graph scoring/index shapes differ')
    reserve = getattr(snapshot, 'reserve_scoring_scratch', None)
    wl, wa = _operator_weights(mode, scoring_version, alpha, locality_weight,
                               attention_weight, cross_message_weight)
    work = (mode in ('locality', 'attention', 'multi') and wl + wa > 0 and
            sum(bool(value) for value in eligible) > 1 and
            any(bool(keep) and value > 0 for keep, value in zip(eligible, raw_scores)))
    charge = None
    try:
        if reserve is not None and work:
            charge = reserve(scoring_scratch_bytes(len(starts), len(snapshot.rows),
                len(snapshot.message_spans), trace=trace))
        result = _score_graph(raw_scores, starts, eligible, snapshot, mode=mode,
            scoring_version=scoring_version, alpha=alpha,
            locality_weight=locality_weight, attention_weight=attention_weight,
            cross_message_weight=cross_message_weight, block_tokens=block_tokens,
            trace=trace, trace_targets=trace_targets)
        object.__setattr__(result, '_charge', charge)
        charge = None
        return result
    finally:
        if charge is not None:
            charge.close()


def _score_graph(raw_scores, starts, eligible, snapshot, *, mode='multi',
                scoring_version='v1', alpha=.2,
                locality_weight=.1, attention_weight=.2, cross_message_weight=.25,
                block_tokens=None, trace=False, trace_targets=()):
    started = time.perf_counter()
    wl, wa = _operator_weights(mode, scoring_version, alpha, locality_weight,
                               attention_weight, cross_message_weight)
    if len(raw_scores) != len(starts) or len(eligible) != len(starts):
        raise ValueError('Graph scoring/index shapes differ')
    if any(not math.isfinite(x) or x < 0 for x in raw_scores):
        raise ValueError('Graph input scores must be finite and nonnegative')
    block_tokens = snapshot.block_tokens if block_tokens is None and snapshot is not None else block_tokens
    if block_tokens is None or type(block_tokens) is not int or block_tokens <= 0:
        raise ValueError('Graph scoring requires valid logical block size')
    if any(type(p) is not int or p < 0 or p % block_tokens for p in starts):
        raise ValueError('Graph scoring requires distinct original block starts')
    eligible_count = sum(bool(value) for value in eligible)
    try:
        total = math.fsum(value for value, keep in zip(raw_scores, eligible) if bool(keep))
        raw_total = math.fsum(raw_scores)
    except OverflowError as exc:
        raise ValueError('Graph input score sum must be finite') from exc
    diagnostics = {'mode': mode, 'scoring_version': scoring_version,
                   'alpha': alpha if scoring_version == 'v2' else None,
                   'locality_weight': wl, 'attention_weight': wa,
                   'locality_share': ((.5 if mode == 'multi' else 1. if mode == 'locality' else 0.)
                                      if scoring_version == 'v2' else wl / (wl + wa) if wl + wa else 0.),
                   'attention_share': ((.5 if mode == 'multi' else 1. if mode == 'attention' else 0.)
                                       if scoring_version == 'v2' else wa / (wl + wa) if wl + wa else 0.),
                   'eligible': eligible_count, 'raw_score_sum': raw_total,
                   'raw_eligible_sum': total, 'normalized_input_sum': 1. if total else 0.,
                   'normalized_output_sum': 1. if total else 0., 'identity': True,
                   'locality_fallback_count': 0, 'locality_missing_count': 0,
                   'locality_same_message_count': 0, 'locality_cross_message_count': 0,
                   'locality_masked_count': 0, 'attention_missing_count': 0,
                   'attention_masked_count': 0, 'masked_self_mass': 0.,
                   'missing_self_mass': 0., 'attention_external_mass': 0.,
                   'attention_seed_coverage': 0., 'attention_retained_raw_mass': 0.,
                   'attention_effective_raw_mass': 0., 'attention_excluded_raw_mass': 0.,
                   'locality_external_mass': 0., 'observed_sources': 0, 'edges': 0,
                   'self_retained_mass': 1. if total else 0.,
                   'normalized_l1_change': 0., 'raw_l1_change': 0.}
    if snapshot is not None:
        diagnostics.update(revision=snapshot.revision, runtime_epoch=snapshot.epoch,
                           accepted_through=snapshot.accepted_through,
                           completed_through=snapshot.completed_through,
                           observed_sources=len(snapshot.rows),
                           edges=sum(len(row.targets) for row in snapshot.rows),
                           no_sample_sources=snapshot.completed_through // block_tokens - len(snapshot.rows),
                           partial_source=None if snapshot.partial is None else snapshot.partial.source,
                           partial_samples=0 if snapshot.partial is None else snapshot.partial.samples,
                           snapshot_bytes=snapshot.bytes)

    def finish(scores, details=()):
        diagnostics['propagated_score_sum'] = math.fsum(scores)
        # These include unavailable rows/slots and boundary attenuation. In
        # V2 discarded raw attention mass is not itself retained: a surviving
        # edge receives the complete attention share after normalization.
        diagnostics['attention_identity_mass'] = (wa if total else 0.) - diagnostics['attention_external_mass']
        diagnostics['locality_identity_mass'] = (wl if total else 0.) - diagnostics['locality_external_mass']
        for kind in ('attention_external', 'locality_external', 'self_retained'):
            diagnostics['raw_' + kind + '_mass'] = diagnostics[kind + '_mass'] * total
        locality_edges = (diagnostics['locality_fallback_count'] +
                          diagnostics['locality_same_message_count'] +
                          diagnostics['locality_cross_message_count'])
        for category, key in (('unknown', 'fallback'), ('same_message', 'same_message'),
                              ('cross_message', 'cross_message')):
            diagnostics['locality_' + category + '_fraction'] = (
                diagnostics['locality_' + key + '_count'] / locality_edges if locality_edges else 0.)
        diagnostics['scoring_seconds'] = time.perf_counter() - started
        return GraphScores(scores, diagnostics, details)

    if mode in ('off', 'collect') or eligible_count <= 1 or total == 0 or wl + wa == 0:
        return finish(raw_scores)
    if len(set(starts)) != len(starts):
        raise ValueError('Graph scoring requires distinct original block starts')
    live = {p: i for i, p in enumerate(starts)}
    allowed = {p: i for i, p in enumerate(starts) if bool(eligible[i])}
    if snapshot is None or snapshot.closed:
        raise ValueError('Graph scoring requires a live immutable snapshot')
    r = tuple(raw_scores[i] / total if bool(eligible[i]) else 0. for i in range(len(starts)))
    result = list(r)
    incoming_l, incoming_a = [0.] * len(starts), [0.] * len(starts)
    outgoing = [0.] * len(starts)
    movements = 0
    boundaries = _LocalityBoundaries(snapshot.message_spans, getattr(snapshot, 'message_end_known', None))

    def transfer(i, j, edge, weight, kind):
        nonlocal movements
        if i == j or edge == 0 or weight == 0:
            return
        delta = weight * r[i] * edge
        if scoring_version == 'v2' and delta == 0:
            return
        movements += 1
        result[i] -= delta
        result[j] += delta
        outgoing[i] += delta
        (incoming_l if kind == 'locality' else incoming_a)[j] += delta
        diagnostics[kind + '_external_mass'] += delta

    if wl:
        for source, i in allowed.items():
            for target in (source - block_tokens, source + block_tokens):
                coefficient, fallback = boundaries.coefficient(source, target, block_tokens, cross_message_weight)
                edge = .5 * coefficient
                if target not in allowed:
                    category = 'masked' if target in live else 'missing'
                    diagnostics['locality_' + category + '_count'] += 1
                    diagnostics[category + '_self_mass'] += wl * r[i] * (.5 if scoring_version == 'v2' else edge)
                    continue
                diagnostics['locality_fallback_count'] += fallback
                category = boundaries.classify(source, target, block_tokens)
                if category != 'unknown':
                    diagnostics['locality_' + category + '_count'] += 1
                transfer(i, allowed[target], edge, wl, 'locality')

    def attention_edges(row):
        if scoring_version == 'v2':
            if row.samples == 0 or row.source + block_tokens > snapshot.completed_through:
                return ()
            surviving = tuple((target, edge) for target, edge in zip(row.targets, row.weights)
                              if target in allowed and target != row.source and edge > 0)
            denominator = math.fsum(edge for _target, edge in surviving)
            return tuple((target, edge, edge / denominator) for target, edge in surviving)
        return tuple((target, edge, edge) for target, edge in zip(row.targets, row.weights)
                     if target in allowed and target != row.source)

    if wa:
        seen_sources = set()
        for row in snapshot.rows:
            if row.source in seen_sources or len(row.targets) != len(row.weights):
                raise ValueError('Invalid stored graph source row')
            seen_sources.add(row.source)
            if (len(set(row.targets)) != len(row.targets) or
                    any(not math.isfinite(v) or v < 0 or (v == 0 and scoring_version == 'v1')
                        for v in row.weights) or
                    math.fsum(row.weights) > 1 + 1e-12):
                raise ValueError('Invalid stored graph edge mass')
            if scoring_version == 'v2' and (type(row.samples) is not int or row.samples < 0):
                raise ValueError('Invalid stored graph sample count')
            if row.source not in allowed:
                continue
            i = allowed[row.source]
            if (scoring_version == 'v2' and
                    (row.samples == 0 or row.source + block_tokens > snapshot.completed_through)):
                continue
            diagnostics['attention_retained_raw_mass'] += r[i] * math.fsum(row.weights)
            for target, edge in zip(row.targets, row.weights):
                if target == row.source or edge == 0:
                    continue
                if target not in allowed:
                    category = 'masked' if target in live else 'missing'
                    diagnostics['attention_' + category + '_count'] += 1
                    if scoring_version == 'v1':
                        diagnostics[category + '_self_mass'] += wa * r[i] * edge
                    continue
            edges = attention_edges(row)
            effective_raw_mass = math.fsum(edge for _, edge, _ in edges)
            diagnostics['attention_excluded_raw_mass'] += r[i] * (math.fsum(row.weights) - effective_raw_mass)
            if edges:
                diagnostics['attention_seed_coverage'] += r[i]
                diagnostics['attention_effective_raw_mass'] += r[i] * effective_raw_mass
            for target, _raw_edge, edge in edges:
                transfer(i, allowed[target], edge, wa, 'attention')
    if movements == 0:
        return finish(raw_scores)
    actual = math.fsum(result)
    if (not math.isclose(actual, 1., rel_tol=1e-12, abs_tol=1e-12) or
            any(not math.isfinite(v) or v < -1e-14 for v in result)):
        raise RuntimeError('Graph propagation violated nonnegativity or mass conservation')
    # Clamp only roundoff at zero; never renormalize the resulting distribution.
    result = [max(0., value) for value in result]
    scores = [result[i] * total if bool(eligible[i]) else raw_scores[i] for i in range(len(starts))]
    diagnostics['identity'] = False
    diagnostics['normalized_output_sum'] = math.fsum(result)
    diagnostics['self_retained_mass'] = math.fsum(r) - math.fsum(outgoing)
    diagnostics['normalized_l1_change'] = math.fsum(abs(after-before) for after, before in zip(result, r))
    diagnostics['raw_l1_change'] = math.fsum(abs(after-before) for after, before in zip(scores, raw_scores))
    details = ()
    if trace:
        preferred = list(dict.fromkeys(allowed[p] for p in trace_targets if p in allowed))[:64]
        ranking = preferred + [i for i in sorted(allowed.values(),
                    key=lambda i: (-(incoming_l[i] + incoming_a[i]), starts[i]))
                    if i not in preferred][:64 - len(preferred)]
        contributors = {i: [] for i in ranking}

        def explanation(source, target, edge, weight, kind, raw_edge=None):
            if source not in allowed or target not in allowed or source == target:
                return
            i, j = allowed[source], allowed[target]
            contribution = weight * r[i] * edge
            if j not in contributors or not contribution:
                return
            entries = contributors[j]
            entries.append((contribution, source, edge, kind, edge if raw_edge is None else raw_edge))
            entries.sort(key=lambda item: (-item[0], item[1], item[3]))
            del entries[4:]

        # The bounded explanation pass runs only with explicit tracing. It
        # avoids retaining whole-graph contributor lists during propagation.
        if wl:
            for j in ranking:
                target = starts[j]
                for source in (target - block_tokens, target + block_tokens):
                    coefficient, _ = boundaries.coefficient(source, target, block_tokens, cross_message_weight)
                    explanation(source, target, .5 * coefficient, wl, 'locality')
        if wa:
            for row in snapshot.rows:
                for target, raw_edge, edge in attention_edges(row):
                    explanation(row.source, target, edge, wa, 'attention', raw_edge)
        details = tuple({'target': starts[i], 'direct_query_score': r[i],
                         'direct_seed_score': r[i], 'raw_seed_score': raw_scores[i],
                         'self_retained': r[i] - outgoing[i],
                         'locality_incoming': incoming_l[i], 'attention_incoming': incoming_a[i],
                         'final_score': result[i], 'raw_final_score': scores[i],
                         'sources': tuple({'source': p, 'edge_weight': e, 'raw_edge_weight': raw,
                                           'normalized_edge_weight': e, 'kind': k, 'contribution': d}
                                          for d, p, e, k, raw in contributors.get(i, ()))}
                        for i in ranking)
    return finish(scores, details)
