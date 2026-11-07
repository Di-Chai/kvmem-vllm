"""One pressure transaction's unchanged native rankings over a shared snapshot.

Only native chronological enumeration and the Python/native call are batched.
Each layer keeps its original candidate order, score export and picker. This
candidate deliberately excludes concurrent historical retirement and decode.
"""
import math

from .memory import Phase


def prepare_pressure_batch(controller, job, histories, local, by_begin, mandatory,
                           max_tokens):
    authority, memory = job.authority, job.authority.memory
    if (job.command.pressure_stage != 'prefill' or memory.phase is not Phase.READY or
            getattr(authority, 'gc', None) is not None or
            getattr(memory, '_history_scope', None) is not None):
        raise ValueError('Batched pressure selection requires READY prefill without GC')
    picker = getattr(memory._store, 'pick_pressure_batch', None)
    if picker is None:
        raise ValueError('Batched pressure selection requires the rebuilt native picker')
    if len(histories) != len(controller.names) or not histories:
        raise ValueError('Batched pressure scores must cover every target attention layer')
    prepared, all_slots, all_scores = [], [], []
    for layer, history in enumerate(histories):
        mapping = controller.records(job.command.request_id, authority, job.history, local, layer)
        candidates, slots = [], []
        for _, _, begin, count, _ in mapping:
            entry = by_begin.get(begin)
            if entry is None or entry[0].count != count:
                raise RuntimeError('Layer pressure candidate is absent from accepted immutable history')
            candidates.append(entry[0])
            slots.append(entry[1])
        # Keep the qualified Python division/materialization, rather than
        # implicitly enabling the independent native score-export candidate.
        scores = history.scores()
        supplied = [scores.get(record.history_begin, 0.) for record in candidates]
        if any(not math.isfinite(score) or score < 0 for score in supplied):
            raise RuntimeError('Invalid layer cumulative attention score')
        if not any(score > 0 for score in supplied):
            raise RuntimeError('Layer pressure has no accepted attention; refusing a recency fallback')
        prepared.append((mapping, candidates, slots))
        all_slots.append(slots)
        all_scores.append(supplied)
    # The original preview calls scores([], []) before pressure selection; it
    # writes no retrieval scores. No runtime budget/action state is changed.
    # Separate scalar columns avoid retaining one GC-tracked Python pair per
    # candidate across all layers. The native adapter only zips them; no score
    # conversion, division, reduction or ranking arithmetic moves here.
    selections = picker(list(mandatory), all_slots, all_scores, max_tokens)
    if len(selections) != len(prepared):
        raise RuntimeError('Native pressure batch omitted a target layer')
    return tuple((*prepared_layer, set(selected))
                 for prepared_layer, selected in zip(prepared, selections))
