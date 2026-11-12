"""Decision traces: what every per-layer selection chose in a serving run.

Serving already writes its per-layer decisions to capture/layerwise_query.jsonl.
This module reads them into one shape and compares two runs. It is the test
for both layers:

- a serving change keeps the strategy, so two runs must give equal traces;
- a strategy change is studied by replaying a trace recorded with
  strategy_trace_scores (candidate scores included) on the CPU.

    python -m vllm_kvmem.strategy.trace summary RUN
    python -m vllm_kvmem.strategy.trace diff RUN_A RUN_B      (exit 1 if they differ)
    python -m vllm_kvmem.strategy.trace table                 (registered strategies)

RUN is the jsonl file, a capture directory, or any directory above one.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

FILE = 'layerwise_query.jsonl'


@dataclass(frozen=True)
class Decision:
    kind: str              # 'query' or 'pressure'
    request: int           # order of the request's first decision in the run
    layer: int
    history: int           # accepted history length the decision was made at
    method: str
    selected: tuple        # history_begin of every selected block, in order
    candidates: tuple      # history_begin of every candidate, or () when not recorded
    scores: tuple          # one score per candidate, or () when not recorded


def locate(run):
    path = Path(run)
    if path.is_file():
        return path
    for candidate in (path / FILE, path / 'capture' / FILE, path / 'server' / 'capture' / FILE,
                      path / 'run' / 'server' / 'capture' / FILE):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError('No %s under %s' % (FILE, run))


def read(run):
    """Every per-layer decision of a run, in the order serving made them."""
    decisions, order, pool = [], {}, {}
    with locate(run).open() as stream:
        for line in stream:
            event = json.loads(line)
            name = event.get('event')
            if name not in ('pressure_selected', 'layer_selected', 'query_candidates'):
                continue
            request = order.setdefault(event['request_id'], len(order))
            if name == 'query_candidates':
                pool[request] = (tuple(event['candidate_starts']), int(event['history']))
            elif name == 'layer_selected':
                candidates, history = pool.get(request, ((), -1))
                scores = tuple(event.get('candidate_scores') or ())
                decisions.append(Decision(
                    'query', request, int(event['layer']), history,
                    event.get('query_scorer', 'meank'),
                    tuple(event.get('selected_starts') or event.get('selected_blocks') or ()),
                    candidates if scores else (), scores))
            else:
                for layer in event.get('layers', ()):
                    scores = tuple(layer.get('candidate_scores') or ())
                    decisions.append(Decision(
                        'pressure', request, int(layer['layer']), int(event.get('accepted_history', -1)),
                        layer.get('actual_method', ''), tuple(layer.get('selected_starts', ())),
                        tuple(layer.get('candidate_starts', ())), scores))
    return decisions


def digest(decisions):
    """One hash over what was chosen; scores and candidates are not part of it."""
    state = hashlib.sha256()
    for d in decisions:
        state.update(json.dumps([d.kind, d.request, d.layer, d.history, d.selected]).encode())
    return state.hexdigest()


def summary(decisions):
    out = dict(decisions=len(decisions), requests=len({d.request for d in decisions}), digest=digest(decisions))
    for kind in ('query', 'pressure'):
        rows = [d for d in decisions if d.kind == kind]
        out[kind] = dict(decisions=len(rows), methods=sorted({d.method for d in rows}),
                         selected_blocks=sum(len(d.selected) for d in rows),
                         with_scores=sum(bool(d.scores) for d in rows))
    return out


def diff(a, b, *, limit=10):
    """Compare what two runs chose, decision by decision in serving order."""
    out = dict(a=len(a), b=len(b), equal=0, different=[])
    for index, (x, y) in enumerate(zip(a, b)):
        if (x.kind, x.request, x.layer, x.history, x.selected) == (y.kind, y.request, y.layer, y.history, y.selected):
            out['equal'] += 1
        elif len(out['different']) < limit:
            common = len(set(x.selected) & set(y.selected))
            out['different'].append(dict(index=index, kind=x.kind, request=x.request, layer=x.layer,
                                         history=(x.history, y.history), selected=(len(x.selected), len(y.selected)),
                                         common=common))
    out['same'] = len(a) == len(b) == out['equal'] and len(a) > 0
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['summary'] and len(argv) == 2:
        print(json.dumps(summary(read(argv[1])), indent=1))
        return 0
    if argv[:1] == ['diff'] and len(argv) == 3:
        result = diff(read(argv[1]), read(argv[2]))
        print(json.dumps(result, indent=1))
        return 0 if result['same'] else 1
    if argv == ['table']:
        from . import table
        for s in table():
            print('%-16s %-11s %-8s %s' % (s.role, s.name, 'built-in' if s.native else 'strategy', s.summary))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
