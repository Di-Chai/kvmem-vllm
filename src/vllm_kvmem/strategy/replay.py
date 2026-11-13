"""Replay a recorded decision trace through other strategies, on the CPU.

A run with strategy_trace_scores records, for every per-layer decision, each
candidate's score and what the picker needed (fixed and mandatory blocks, the
budget, the sink band). This tool picks again from that record:

- `recorded`: the recorded scores through the reference picker. It must give
  back the recorded selection; any difference is a bug in the replay or in the
  record, so it is reported first.
- every registered scorer whose needs a trace can supply (today: block
  positions), through the same reference picker.

For each, it reports how many decisions select exactly the recorded blocks and
the mean overlap with them. With --evidence, a JSON file mapping a request's
order in the run ("0", "1", ...) to its gold evidence token spans, it also
reports how much of the evidence each strategy keeps.

Later decisions depend on earlier ones, so a replay screens strategies; the
finalists still run end to end.

    python -m vllm_kvmem.strategy.replay RUN [--evidence FILE] [--strategies recency,...]

The reference pickers mirror the native store picker (third_party/qw3,
KvMemStore::pick_topk_blocks and pick_prefill_pressure_from) for the
per-layer path.
"""
import argparse
from dataclasses import dataclass
import json
import sys

from .registry import get, names, resolve
from .contract import LAYER, PRESSURE_SCORER, QUERY_SCORER, PressureLayerView, QueryLayerView, check_query_scores
from .trace import read

BLOCK = 32
REPLAYABLE_NEEDS = frozenset(('positions',))


def pick_query(candidates, scores, fixed, budget_tokens, block=BLOCK):
    """Fixed blocks, then the best scores to the budget; ties go to the later block."""
    budget = budget_tokens // block
    if len(candidates) <= budget:
        return set(candidates)
    kept = set(fixed)
    rest = sorted(((score, start) for start, score in zip(candidates, scores) if start not in kept),
                  key=lambda item: (-item[0], -item[1]))
    kept.update(start for _, start in rest[:max(0, budget - len(kept))])
    return kept


def pick_pressure(candidates, scores, mandatory, budget_tokens, sink_tokens, block=BLOCK):
    """Sink and mandatory blocks, then positive scores, then the newest window blocks."""
    budget = budget_tokens // block
    kept = set()
    for start in sorted(candidates):
        if start < sink_tokens and len(kept) < budget:
            kept.add(start)
    kept.update(mandatory)
    ranked = sorted(((score, start) for start, score in zip(candidates, scores) if start not in kept and score > 0),
                    key=lambda item: (-item[0], -item[1]))
    for _, start in ranked:
        if len(kept) >= budget:
            break
        kept.add(start)
    for start in sorted(candidates, reverse=True):
        if len(kept) >= budget:
            break
        kept.add(start)
    return kept


@dataclass(frozen=True)
class _Record:
    history_begin: int
    count: int
    handle: object


@dataclass(frozen=True)
class _Handle:
    slot: int


def replayable(decision):
    if not decision.scores or not decision.candidates or decision.budget < 0:
        return False
    return decision.kind == 'query' or decision.sink >= 0


def strategy_scores(scorer, role, decision):
    """A registered scorer's scores for a recorded decision, from what the trace holds."""
    counts = decision.counts or (BLOCK,) * len(decision.candidates)
    if role == QUERY_SCORER:
        records = [_Record(start, count, _Handle(i)) for i, (start, count) in
                   enumerate(zip(decision.candidates, counts))]
        eligible = decision.eligible or tuple(True for _ in records)
        view = QueryLayerView(decision.layer, records, eligible, None, None, decision.history, {})
        return list(check_query_scores(view, scorer.scores(view)))
    mapping = [(None, None, start, count, None) for start, count in zip(decision.candidates, counts)]
    view = PressureLayerView(decision.layer, mapping, None, None, decision.stage == 'decode')
    return list(scorer.scores(view))


def pick(decision, scores):
    if decision.kind == 'query':
        excluded = set(decision.excluded)
        scores = [-1. if start in excluded else score for start, score in zip(decision.candidates, scores)]
        return pick_query(decision.candidates, scores, decision.fixed, decision.budget)
    return pick_pressure(decision.candidates, scores, decision.fixed, decision.budget, decision.sink)


def coverage(selected, spans):
    total = sum(end - begin for begin, end in spans)
    if not total:
        return None
    kept = 0
    for begin, end in spans:
        for start in selected:
            kept += max(0, min(end, start + BLOCK) - max(begin, start))
    return kept / total


def replay(decisions, strategies=None, evidence=None):
    """Per strategy and decision kind: decisions, exact repeats, mean overlap, mean evidence coverage."""
    rows = [d for d in decisions if d.kind in ('query', 'pressure') and replayable(d)]
    roles = {'query': QUERY_SCORER, 'pressure': PRESSURE_SCORER}
    wanted = None if strategies is None else set(strategies)
    report = {}
    scorers = {}
    for role in (QUERY_SCORER, PRESSURE_SCORER):
        for name in names(role, LAYER):
            if wanted is not None and name not in wanted:
                continue
            if not set(get(role, name).needs) <= REPLAYABLE_NEEDS:
                continue
            scorers[(role, name)] = resolve(role, name, LAYER)
    for decision in rows:
        recorded = set(decision.selected)
        options = [('recorded', decision.scores)]
        for (role, name), scorer in scorers.items():
            if role == roles[decision.kind]:
                options.append((name, strategy_scores(scorer, role, decision)))
        for name, scores in options:
            chosen = pick(decision, scores)
            entry = report.setdefault(name, {}).setdefault(decision.kind, dict(
                decisions=0, exact=0, overlap=0., coverage=[], first_difference=None))
            entry['decisions'] += 1
            entry['exact'] += chosen == recorded
            entry['overlap'] += len(chosen & recorded) / max(1, len(recorded))
            if chosen != recorded and entry['first_difference'] is None:
                entry['first_difference'] = dict(request=decision.request, layer=decision.layer,
                                                 history=decision.history, missing=len(recorded - chosen),
                                                 extra=len(chosen - recorded))
            spans = (evidence or {}).get(str(decision.request))
            if spans:
                value = coverage(chosen, spans)
                if value is not None:
                    entry['coverage'].append(value)
    for kinds in report.values():
        for entry in kinds.values():
            entry['overlap'] = round(entry['overlap'] / entry['decisions'], 6)
            values = entry.pop('coverage')
            entry['evidence_coverage'] = round(sum(values) / len(values), 6) if values else None
    return dict(replayable=len(rows), not_replayable=sum(d.kind in ('query', 'pressure') for d in decisions) - len(rows),
                strategies=report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run')
    parser.add_argument('--evidence', default=None)
    parser.add_argument('--strategies', default=None, help='comma-separated registered names (default: all replayable)')
    args = parser.parse_args(argv)
    evidence = json.loads(open(args.evidence).read()) if args.evidence else None
    result = replay(read(args.run), args.strategies.split(',') if args.strategies else None, evidence)
    print(json.dumps(result, indent=1))
    recorded = result['strategies'].get('recorded', {})
    faithful = all(entry['exact'] == entry['decisions'] for entry in recorded.values())
    return 0 if result['replayable'] and faithful else 1


if __name__ == '__main__':
    sys.exit(main())
