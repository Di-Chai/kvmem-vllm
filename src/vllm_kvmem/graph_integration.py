"""Query-only graph selection; native picking, KV ownership and budgets stay native."""
from contextlib import ExitStack, contextmanager
import json
import threading
import time


_LOG_LOCK = threading.Lock()
_MAX_LOG_BYTES = 16 * 1024 * 1024
_MAX_EVENTS = 256
_MAX_EVENT_BYTES = 65536


@contextmanager
def query_snapshot(authority):
    graph = getattr(authority, 'block_graph', None)
    if graph is None:
        yield None
        return
    snapshot = graph.snapshot()
    try:
        if snapshot.accepted_through < authority.history_end:
            raise RuntimeError('Graph accepted events have not caught up with the Query index')
        yield snapshot
    finally:
        snapshot.close()


def score_and_select(authority, records, raw_scores, eligible, snapshot,
                     mandatory_slots, max_tokens):
    """Called only for scoring modes, while the current index is pinned."""
    with ExitStack() as results:
        return _score_and_select(authority, records, raw_scores, eligible, snapshot,
            mandatory_slots, max_tokens, results)


def _score_and_select(authority, records, raw_scores, eligible, snapshot,
                      mandatory_slots, max_tokens, results):
    from .graph_scoring import score_graph
    graph = authority.block_graph
    options = authority.graph_options
    binding = graph.binding
    if (snapshot.owner, snapshot.epoch, snapshot.branch_id, snapshot.revision,
            snapshot.accepted_through) != binding:
        raise RuntimeError('Graph and Query index snapshots belong to different revisions')
    started = time.perf_counter()
    starts = [record.history_begin for record in records]
    result = results.enter_context(score_graph(raw_scores, starts, eligible, snapshot,
        mode=options.graph_mode, locality_weight=options.graph_locality_weight,
        attention_weight=options.graph_attention_weight,
        cross_message_weight=options.graph_cross_message_weight,
        block_tokens=authority.memory.config.block_tokens, trace=False))
    graph.validate_binding(binding)
    # Generation-qualified handles were pinned by index_snapshot. Verify the
    # mapping again before converting to the native selector's slot interface.
    for record in records:
        if authority.memory._store.handle(record.handle.slot) != record.handle:
            raise RuntimeError('Graph Query index contains a recycled generation')
    raw_by_slot = {record.handle.slot: score for record, score in zip(records, raw_scores)}
    baseline = authority.memory.preview_selection(raw_by_slot, mandatory_slots,
        action=False, prefill_scores=None, max_tokens=max_tokens)
    propagated = {record.handle.slot: score for record, score in zip(records, result.scores)}
    old = set(baseline)
    positions = {record.handle.slot: record.history_begin for record in records}
    trace = ()
    if options.graph_trace:
        selected = set(authority.memory.preview_selection(propagated, mandatory_slots,
            action=False, prefill_scores=None, max_tokens=max_tokens))
        # Detailed diagnostics are opt-in and bounded. Explain newly selected
        # blocks first, rather than whichever old targets had largest inflow.
        traced = results.enter_context(score_graph(raw_scores, starts, eligible, snapshot,
            mode=options.graph_mode, locality_weight=options.graph_locality_weight,
            attention_weight=options.graph_attention_weight,
            cross_message_weight=options.graph_cross_message_weight,
            block_tokens=authority.memory.config.block_tokens, trace=True,
            trace_targets=[positions[slot] for slot in selected-old]))
        trace = traced.trace
    # A diagnostic scratch-capacity failure must precede the native assembly
    # transaction. Both score results keep their accounted leases until return.
    graph.validate_binding(binding)
    plan = authority.memory.select(propagated, mandatory_slots,
        action=False, prefill_scores=None, max_tokens=max_tokens)
    selected = {row.block_id for row in plan.remaps}
    report = dict(result.diagnostics)
    report.update(mode=options.graph_mode, budget=max_tokens or authority.memory.config.select_budget,
        revision=snapshot.revision, runtime_epoch=snapshot.epoch,
        branch_id=str(snapshot.branch_id), accepted_through=snapshot.accepted_through,
        eligible_blocks=sum(bool(value) for value in eligible),
        graph_bytes=graph.bytes, graph_live_bytes=graph.live_bytes,
        snapshot_bytes=snapshot.bytes, graph_global_bytes=graph.budget.used,
        selected_tokens=plan.total_window_tokens, added_blocks=len(selected-old),
        displaced_blocks=len(old-selected), graph_selection_seconds=time.perf_counter()-started)
    if options.graph_trace:
        report['selected_starts'] = sorted(positions[slot] for slot in selected)[:256]
        report['baseline_starts'] = sorted(positions[slot] for slot in old)[:256]
        report['added_starts'] = sorted(positions[slot] for slot in selected-old)[:64]
        report['displaced_starts'] = sorted(positions[slot] for slot in old-selected)[:64]
        report['trace'] = [dict(item, selected=item['target'] in {positions[s] for s in selected},
            newly_selected=item['target'] in {positions[s] for s in selected-old}) for item in trace]
    authority.graph_selection_report = report
    authority._graph_selection_binding = binding
    # Retain only an epoch/revision token, never an unaccounted snapshot closure.
    # This guard covers both synchronous and asynchronous assembly publication.
    authority.memory._selection_validator = lambda: graph.validate_binding(binding)
    emit_graph_report(authority, report)
    return plan


def emit_graph_report(authority, report):
    """Bound both the event count per owner and shared diagnostic file size."""
    count = getattr(authority, '_graph_report_count', 0)
    if count >= _MAX_EVENTS:
        return
    path = getattr(authority, 'graph_report_path', None)
    if path is None:
        return
    row = dict(report, authority_id=authority.memory.request_id,
        request_id=getattr(authority, 'graph_request_id', authority.memory.request_id))
    encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > _MAX_EVENT_BYTES:
        row.pop('trace', None)
        row['trace_omitted_byte_limit'] = True
        encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    with _LOG_LOCK:
        if path.exists() and path.stat().st_size + len(encoded.encode()) + 1 > _MAX_LOG_BYTES:
            return
        with path.open('a') as stream:
            stream.write(encoded + '\n')
        authority._graph_report_count = count + 1


def record_query_inputs(authority, records, scores, eligible, snapshot, mandatory_slots, max_tokens):
    """Opt-in bounded replay evidence, never a per-decode graph dump."""
    options = getattr(authority, 'graph_options', None)
    path = getattr(authority, 'graph_report_path', None)
    if (snapshot is None or options is None or not options.graph_trace or path is None or
            len(records) > 2048 or getattr(authority, '_graph_input_count', 0) >= 8):
        return
    from dataclasses import asdict
    cfg = authority.memory.config
    action = getattr(authority, 'action_history', None)
    prefill = getattr(action, 'prefill', None)
    row = dict(version=1, authority_id=authority.memory.request_id,
        request_id=getattr(authority, 'graph_request_id', authority.memory.request_id),
        mode=options.graph_mode, starts=[r.history_begin for r in records],
        counts=[r.count for r in records], raw_scores=scores, eligible=eligible,
        mandatory_starts=[r.history_begin for r in records if r.handle.slot in mandatory_slots],
        budget=max_tokens or cfg.select_budget,
        config={name: getattr(cfg, name) for name in ('block_tokens', 'select_budget',
            'gen_budget', 'repository_blocks', 'sink_blocks', 'recent_blocks',
            'action_attention', 'action_fraction', 'action_combine')},
        graph=dict(rows=[asdict(item) for item in snapshot.rows],
            partial=asdict(snapshot.partial) if snapshot.partial is not None else None,
            message_spans=snapshot.message_spans, message_end_known=snapshot.message_end_known,
            accepted_through=snapshot.accepted_through, completed_through=snapshot.completed_through,
            revision=snapshot.revision),
        legacy_action_scores=getattr(action, 'scores', {}),
        prefill_scores=prefill.scores() if prefill is not None and not action.samples else None)
    encoded = json.dumps(row, separators=(',', ':'), allow_nan=False)
    if len(encoded.encode()) > 262144:
        return
    path = path.with_name('block_graph_inputs.jsonl')
    with _LOG_LOCK:
        if path.exists() and path.stat().st_size + len(encoded.encode()) + 1 > _MAX_LOG_BYTES:
            return
        with path.open('a') as stream:
            stream.write(encoded + '\n')
        authority._graph_input_count = getattr(authority, '_graph_input_count', 0) + 1
