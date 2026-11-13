"""One-hop raw-seed max pooling; incoming scores are never added together.

Each operator pools incoming source seeds independently. No incoming edge means
identity for that operator. Multi mixes both pools equally using the single
alpha. This is a ranking operator, not a mass-conserving probability transport.
"""
import math
import time


def metadata(snapshot, mode, alpha, wl, wa):
    result = dict(mode=mode, scoring_version='v2', alpha=alpha, aggregation='max',
        pooling_source='raw_seed', score_sum_conserved=False,
        locality_weight=wl, attention_weight=wa,
        locality_share=.5 if mode == 'multi' else 1. if mode == 'locality' else 0.,
        attention_share=.5 if mode == 'multi' else 1. if mode == 'attention' else 0.)
    if snapshot is not None:
        result.update(revision=snapshot.revision, runtime_epoch=snapshot.epoch,
            accepted_through=snapshot.accepted_through, completed_through=snapshot.completed_through,
            observed_sources=len(snapshot.rows), edges=sum(len(row.targets) for row in snapshot.rows),
            no_sample_sources=snapshot.completed_through // snapshot.block_tokens - len(snapshot.rows),
            partial_source=None if snapshot.partial is None else snapshot.partial.source,
            partial_samples=0 if snapshot.partial is None else snapshot.partial.samples,
            snapshot_bytes=snapshot.bytes)
    return result


def pool_statistics(raw, scores, eligible, total, **counts):
    change = math.fsum(abs(after-before) for before, after, keep in zip(raw, scores, eligible) if keep)
    return dict(raw_score_sum=math.fsum(raw), raw_eligible_sum=total,
        propagated_score_sum=math.fsum(scores), eligible=sum(bool(x) for x in eligible),
        normalized_input_sum=1. if total else 0.,
        normalized_output_sum=math.fsum(value for value, keep in zip(scores, eligible) if keep) / total if total else 0.,
        identity=all(a == b for a, b in zip(raw, scores)), raw_l1_change=change,
        normalized_l1_change=change / total if total else 0., **counts)


def score_max(raw, starts, eligible, snapshot, *, mode, alpha, cross_message_weight,
              block_tokens=None, trace=False, trace_targets=()):
    from .graph_scoring import GraphScores, _LocalityBoundaries, _operator_weights
    started = time.perf_counter()
    wl, wa = _operator_weights(mode, 'v2', alpha, 0., 0., cross_message_weight)
    block = snapshot.block_tokens if block_tokens is None and snapshot is not None else block_tokens
    if type(block) is not int or block <= 0:
        raise ValueError('Graph scoring requires valid logical block size')
    if len(raw) != len(starts) or len(eligible) != len(starts):
        raise ValueError('Graph scoring/index shapes differ')
    if any(type(p) is not int or p < 0 or p % block for p in starts) or len(set(starts)) != len(starts):
        raise ValueError('Graph scoring requires distinct original block starts')
    if any(not math.isfinite(x) or x < 0 for x in raw):
        raise ValueError('Graph input scores must be finite and nonnegative')
    total = math.fsum(value for value, keep in zip(raw, eligible) if keep)
    math.fsum(raw)  # Reject an overflowing complete input, as in the sum path.
    d = metadata(snapshot, mode, alpha, wl, wa)
    counts = dict(locality_pool_edges=0, attention_pool_edges=0,
        locality_masked_count=0, locality_missing_count=0,
        locality_fallback_count=0, locality_same_message_count=0, locality_cross_message_count=0,
        attention_masked_count=0, attention_missing_count=0)

    def finish(scores, details=()):
        d.update(pool_statistics(raw, scores, eligible, total, **counts))
        d['scoring_seconds'] = time.perf_counter() - started
        return GraphScores(raw if d['identity'] else scores, d, details)

    if mode in ('off', 'collect') or wl + wa == 0 or total == 0 or sum(bool(x) for x in eligible) <= 1:
        return finish(raw)
    if snapshot is None or snapshot.closed:
        raise ValueError('Graph scoring requires a live immutable snapshot')
    live = {p:i for i,p in enumerate(starts)}
    allowed = {p:i for i,p in enumerate(starts) if eligible[i]}
    pools = [list(raw), list(raw)]
    seen = [[False]*len(raw), [False]*len(raw)]
    winners = [[None]*len(raw), [None]*len(raw)] if trace else None

    def offer(kind, i, j, factor, raw_edge):
        candidate = factor * raw[i]
        if not seen[kind][j] or candidate > pools[kind][j]:
            pools[kind][j] = candidate
            if trace:
                winners[kind][j] = (starts[i], factor, raw_edge)
        elif trace and candidate == pools[kind][j] and starts[i] < winners[kind][j][0]:
            winners[kind][j] = (starts[i], factor, raw_edge)
        seen[kind][j] = True

    if wl:
        boundaries = _LocalityBoundaries(snapshot.message_spans, getattr(snapshot, 'message_end_known', None))
        for source,i in allowed.items():
            for target in (source-block, source+block):
                if target not in allowed:
                    counts['locality_masked_count' if target in live else 'locality_missing_count'] += 1
                    continue
                category = boundaries.classify(source, target, block)
                counts['locality_' + ('fallback' if category == 'unknown' else category) + '_count'] += 1
                factor = cross_message_weight if category == 'cross_message' else 1.
                if factor > 0:
                    counts['locality_pool_edges'] += 1
                    offer(0, i, allowed[target], factor, factor)
    if wa:
        sources = set()
        for row in snapshot.rows:
            if (row.source in sources or len(row.targets) != len(row.weights) or
                    len(set(row.targets)) != len(row.targets) or
                    any(not math.isfinite(w) or w < 0 for w in row.weights) or
                    math.fsum(row.weights) > 1 + 1e-12 or type(row.samples) is not int or row.samples < 0):
                raise ValueError('Invalid stored graph source row/sample count or edge mass')
            sources.add(row.source)
            if row.source not in allowed or row.samples == 0 or row.source + block > snapshot.completed_through:
                continue
            for target, weight in zip(row.targets, row.weights):
                if target == row.source or weight == 0:
                    continue
                if target not in allowed:
                    counts['attention_masked_count' if target in live else 'attention_missing_count'] += 1
                    continue
                counts['attention_pool_edges'] += 1
                # Attention weight establishes a positive relation; the pool
                # compares raw source seeds, not weighted or normalized edges.
                offer(1, allowed[row.source], allowed[target], 1., weight)
    scores = [(1.-wl-wa)*value + wl*pools[0][i] + wa*pools[1][i]
              if eligible[i] and ((wl and pools[0][i] != value) or (wa and pools[1][i] != value)) else value
              for i,value in enumerate(raw)]
    if any(not math.isfinite(x) or x < 0 for x in scores):
        raise RuntimeError('Graph max pooling produced an invalid score')
    details = ()
    if trace:
        preferred = list(dict.fromkeys(allowed[p] for p in trace_targets if p in allowed))[:64]
        ranking = preferred + [i for i in sorted(allowed.values(),
            key=lambda i:(-abs(scores[i]-raw[i]), starts[i])) if i not in preferred][:64-len(preferred)]
        details = tuple(dict(target=starts[i], raw_seed_score=raw[i], raw_final_score=scores[i],
            aggregation='max', locality_pool_score=pools[0][i], attention_pool_score=pools[1][i],
            sources=tuple(dict(kind=kind, source=winner[0], pool_factor=winner[1],
                raw_edge_weight=winner[2], raw_source_score=raw[live[winner[0]]],
                contribution=weight*winner[1]*raw[live[winner[0]]])
                for kind,weight,winner in (('locality',wl,winners[0][i]),('attention',wa,winners[1][i]))
                if winner is not None)) for i in ranking)
    return finish(scores, details)
