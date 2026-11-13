"""Explicit answer experiment: content-frame token softmax, then page sum.

This slow archive-reading scorer is not a serving optimization. It sees the
current original Query and all accepted raw K, never references or gold offsets.
Selection and assembly still use the ordinary generation-qualified transaction.
"""
import time

import torch


def token_page_scores(authority, records, queries, eligible, *, chunk_records=128):
    if (queries.ndim != 4 or not records or not 0 < chunk_records <= 128 or
            len(records) != len(eligible) or not any(eligible)):
        raise ValueError('Invalid experimental Query score geometry')
    layers, rows, heads, dim = queries.shape
    kvheads = authority.shape[2]
    if (layers != authority.scoring_layers or heads % kvheads or dim != authority.shape[3]
            or rows > 512 or authority.shape[1] != 32):
        raise ValueError('Experimental Query geometry exceeds the qualified diagnostic')
    cursor = 0
    for record in records:
        if record.history_begin != cursor or not 0 < record.count <= 32:
            raise ValueError('Experimental scoring requires a complete accepted prefix')
        cursor += record.count
    if cursor != authority.history_end or cursor > 262144:
        raise ValueError('Experimental history exceeds the qualified 256K prefix')
    q = queries.to(dtype=torch.float16).float().reshape(layers, rows, kvheads, heads//kvheads, dim)
    # At most ~512 MiB for this 16-layer, <=32-row diagnostic cohort.
    size = layers * rows * heads * len(records) * 4
    if size > 512 * 1024**2:
        raise ValueError('Experimental score buffer exceeds its 512 MiB bound')
    logits = torch.empty((*q.shape[:-1], len(records)), device=q.device)
    began = time.perf_counter()
    for begin in range(0, len(records), chunk_records):
        group = records[begin:begin+chunk_records]
        tensors = authority.read_records([r.handle for r in group])
        raw = torch.zeros((layers, len(group), 32, kvheads, dim), dtype=torch.uint8)
        for i, (record, (key, _)) in enumerate(zip(group, tensors)):
            if key.dtype != torch.float8_e4m3fn or key.shape[:2] != (authority.shape[0], record.count):
                raise ValueError('Unexpected archived raw K format')
            raw[:, i, :record.count] = key[:layers].view(torch.uint8)
        keys = raw.view(torch.float8_e4m3fn).to(device=q.device, dtype=torch.float32)
        z = torch.einsum('ltkgd,lbskd->ltkgbs', q, keys) * dim**-.5
        counts = torch.tensor([r.count for r in group], device=q.device)
        valid = torch.arange(32, device=q.device)[None, :] < counts[:, None]
        z.masked_fill_(~valid, -torch.inf)
        logits[..., begin:begin+len(group)] = z.logsumexp(-1)
        del tensors, raw, keys, z
    mask = torch.tensor(eligible, dtype=torch.bool, device=q.device)
    probabilities = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    scores = probabilities.sum((0, 1, 2, 3)) / (layers * heads)
    if not bool(torch.isfinite(scores).all()) or abs(float(scores.sum())-rows) > 1e-4:
        raise ValueError('Experimental global token normalization failed')
    scores = scores.cpu().tolist()
    return scores, dict(method='content_token_softmax_page_sum', score_buffer_bytes=size,
        history_tokens=cursor, query_rows=rows, layers=layers, seconds=time.perf_counter()-began,
        score_sum=sum(scores), normalization='global eligible historical tokens per layer/Query row/Q head',
        rope=False, head_layer_reduction='mean', query_row_reduction='sum')


def select_token_queries(authority, queries, mandatory_handles, *, max_tokens):
    from .authority import query_score_frame, record_key, SCORE_DOMAIN_VERSION
    from .graph_integration import query_snapshot, emit_graph_report
    authority._require_open()
    options = authority.graph_options
    if options.graph_alpha != 0 or getattr(authority, 'query_history', None) is not None:
        raise ValueError('Token answer experiment requires Query-only, no Query history')
    with authority.index_snapshot() as (records, _), query_snapshot(authority) as snapshot:
        if snapshot is None:
            raise ValueError('Token answer experiment requires an auditable graph snapshot')
        mandatory = {record_key(h) for h in mandatory_handles}
        if not mandatory.issubset({record_key(r.handle) for r in records}):
            raise ValueError('Foreign or stale experimental mandatory handle')
        pins = tuple(slot for slot, _ in mandatory)
        with query_score_frame(authority.memory, records, pins, max_tokens) as frame:
            score_mask = tuple(frame.score_mask)
            transport_mask = tuple(frame.transport_mask)
            scores, stats = token_page_scores(authority, records, queries, score_mask)
        for record in records:
            if authority.memory._store.handle(record.handle.slot) != record.handle:
                raise ValueError('Experimental Query contains a recycled record')
        graph = authority.block_graph
        binding = (snapshot.owner, snapshot.epoch, snapshot.branch_id, snapshot.revision, snapshot.accepted_through)
        graph.validate_binding(binding)
        by_slot = {r.handle.slot:s for r,s in zip(records, scores)}
        selected = set(authority.memory.preview_selection(by_slot, pins, action=False,
            prefill_scores=None, max_tokens=max_tokens))
        stats.update(phase='query', seed_source='content_token_softmax_page_sum', mode='multi',
            alpha=0., pressure_policy=options.pressure_policy, score_domain_version=SCORE_DOMAIN_VERSION,
            score_mask_count=sum(score_mask), transport_mask_count=sum(transport_mask),
            revision=snapshot.revision, accepted_through=snapshot.accepted_through,
            selected_tokens=sum(r.count for r in records if r.handle.slot in selected),
            selected_starts=[r.history_begin for r in records if r.handle.slot in selected],
            added_blocks=0, displaced_blocks=0, diagnostic_only=True)
        authority.query_score_domain = dict(version=SCORE_DOMAIN_VERSION,
            score_mask=score_mask, transport_mask=transport_mask)
        # Emit before opening the native assembly transaction: I/O failure must
        # not leave a plan half-created. No guided or pressure call reaches here.
        emit_graph_report(authority, stats)
        graph.validate_binding(binding)
        plan = authority.memory.select(by_slot, pins, action=False,
            prefill_scores=None, max_tokens=max_tokens)
        authority.graph_selection_report = stats
        authority._graph_selection_binding = binding
        authority.memory._selection_validator = lambda: graph.validate_binding(binding)
        return plan
