"""Private Query scores replace cumulative-attention pressure, without transport."""


def select_guided_pressure(authority, records, scores, snapshot, mandatory_slots, max_tokens):
    from .authority import SCORE_DOMAIN_VERSION
    from .graph_integration import emit_graph_report
    memory = authority.memory
    graph = getattr(authority, 'block_graph', None)
    binding = graph.binding if graph is not None else None
    for record in records:
        if memory._store.handle(record.handle.slot) != record.handle:
            raise RuntimeError('Guided pressure index contains a recycled generation')
    # The pressure picker retains only sink and explicit mandatory evidence.
    # No old Action scores, cumulative-prefill scores or graph contributions.
    pairs = [(r.handle.slot, score) for r, score in zip(records, scores)]
    picked = set(memory.preview_selection({}, mandatory_slots, action=False,
        pressure_scores=pairs, max_tokens=max_tokens))
    domain = getattr(authority, 'query_score_domain', {})
    options = authority.graph_options
    report = dict(phase='pressure', seed_source='guided_query', pressure_policy='guided_query',
        scoring_version=options.graph_scoring_version, mode=options.graph_mode,
        alpha=options.graph_alpha, aggregation=getattr(options, 'graph_aggregation', 'sum'),
        propagation_applied=False, score_domain_version=SCORE_DOMAIN_VERSION,
        score_mask_count=sum(domain.get('score_mask', ())), transport_mask_count=0,
        candidate_count=len(records), budget=max_tokens or memory.config.select_budget,
        selected_tokens=sum(r.count for r in records if r.handle.slot in picked))
    if graph is not None:
        if snapshot is None or (snapshot.owner, snapshot.epoch, snapshot.branch_id,
                snapshot.revision, snapshot.accepted_through) != binding:
            raise RuntimeError('Guided pressure lost its accepted graph/index binding')
        report.update(revision=graph.revision, runtime_epoch=graph.runtime_epoch,
            accepted_through=graph.accepted_through, graph_bytes=graph.bytes,
            graph_live_bytes=graph.live_bytes,
            locality_provenance=dict(getattr(graph, 'locality_provenance', ())))
        graph.validate_binding(binding)
        emit_graph_report(authority, report)
        graph.validate_binding(binding)
    plan = memory.select({}, mandatory_slots, action=False, pressure_scores=pairs, max_tokens=max_tokens)
    authority.graph_selection_report = report
    if graph is not None:
        authority._graph_selection_binding = binding
        memory._selection_validator = lambda: graph.validate_binding(binding)
    return plan
