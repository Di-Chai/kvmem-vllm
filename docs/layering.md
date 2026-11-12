# Layers

The package is being split along one seam. **Serving** moves blocks and schedules work.
**Strategy** decides which blocks are retrieved, kept or evicted. They meet at a small
contract, and every decision is recorded so each side can be tested without the other.

This page describes the target, what exists today and the order of the remaining moves.
The plan it belongs to is the [roadmap](roadmap.md).

## Target layers

| Layer | Owns | May import |
| --- | --- | --- |
| Apps | Benchmarks, bridges, configuration compiler | Public API only |
| Strategy | Scorers, pickers, evictors | Its own package and the standard library |
| Serving | Refresh pipeline, scheduling, sessions, tiering, tensor parallel, signal capture | Contract, core, model adapters |
| Model adapters | Per-model KV layout and state capture | Core |
| Core | KV archive, records, native code | Nothing above |

Strategy and serving never import each other's modules. Serving imports the strategy
package's contract and registry; the strategy package imports nothing from the rest of
`vllm_kvmem`. `tests/test_strategy_layer.py` checks the second rule on every run.

## The contract

`src/vllm_kvmem/strategy/contract.py` defines three roles. A fourth, eviction, is listed
in the registry but still chosen through `historical_gc.policy`.

| Role | Decides | Sees | Returns |
| --- | --- | --- | --- |
| Query scorer | How relevant each history block is to a Query, per attention layer | The candidate records, which are eligible, the layer's Query rows, the mean-key index | One score per record; 0 for ineligible records |
| Pressure scorer | Which blocks of a full window are worth keeping, per layer | The layer's window rows and its accumulated attention | One score ≥ 0 per row, at least one > 0 |
| Picker | Which blocks a set of scores selects | Scores, mandatory blocks, token budget | The selected blocks |

Rules every strategy keeps, because serving pipelines work on the strength of them:

1. **Pure.** The result depends only on the view: no hidden state, no reading of GPU
   residency, no dependence on another layer's or another request's decision.
2. **Deterministic** for equal inputs.
3. **Read-only.** A view is a few references to buffers serving already holds. Nothing
   is copied to build it and nothing reachable from it may be changed.

Serving checks every result (`check_query_scores`, `check_pressure_scores`, `check_pick`)
and stops the refresh with a `ContractError` instead of running a selection it cannot trust.

## Choosing a strategy

Three profile options under `kvmem`, all defaulting to serving's built-in code:

| Option | Default | Other shipped values |
| --- | --- | --- |
| `strategy_query_scorer` | `meank` | `recency` |
| `strategy_pressure_scorer` | `attention` | `recency` |
| `strategy_picker` | `native` | none yet |

A default name resolves to "no strategy object", and serving runs its existing path
unchanged, including the fused native pick. The `recency` scorers keep the most recent
history and ignore content; they are the lower reference for comparisons.

A scorer that is not built in still runs on the fast pipeline. Its scores feed the same
native picker, through the fused native call when that is enabled. Two shortcuts
implement the built-in strategy only and refuse anything else: pressure batch selection
and group selection.

`python -m vllm_kvmem.strategy.trace table` prints every registered strategy.

## Adding a strategy

```python
from vllm_kvmem.strategy import PRESSURE_SCORER, Strategy, register

class OldestFirst:
    name = 'oldest'

    def scores(self, view):
        newest = max(row[2] + row[3] for row in view.mapping)
        return [float(newest - row[2] + 1) for row in view.mapping]

register(Strategy(PRESSURE_SCORER, 'oldest', OldestFirst, needs=('positions',),
                  summary='Oldest window blocks stay'))
```

Then set `"strategy_pressure_scorer": "oldest"` in the profile. A strategy on the hot
path should be cheap: it runs once per layer per refresh on the refresh thread.

## The decision trace

Serving writes each per-layer decision to `capture/layerwise_query.jsonl`.
`vllm_kvmem.strategy.trace` reads those records into one shape and compares runs:

```sh
python -m vllm_kvmem.strategy.trace summary RUN
python -m vllm_kvmem.strategy.trace diff RUN_A RUN_B     # exit 1 if they differ
```

This gives each layer its own test:

- **Serving change:** the strategy is unchanged, so the trace and the generated tokens
  must be identical at one request.
- **Strategy change:** set `strategy_trace_scores` to also record every candidate's score,
  then study alternatives on the CPU from the recorded run. Later decisions depend on
  earlier ones, so this screens strategies; finalists still run end to end.

## What is in place

- The `strategy` package: contract, registry, built-in entries, `recency` scorers, trace reader.
- Scorer seams on the per-layer path: Query (`layerwise_query.py`) and pressure
  (`layerwise_pressure.py`). Prefill pressure and the final Query have run end to end with a
  non-default scorer; rolling-decode pressure shares the code but has not.
- Picker seam at `RequestMemory.preview_selection`, the funnel for global and per-layer picks.
- Four typed options replacing what would have been environment flags.
- A launch-time check that refuses a scorer the launched path would ignore. The scorer
  seams exist on the per-layer path only; the global path still follows
  `query_score_mode` and `pressure_policy`.

## What is not done yet

| Step | Content |
| --- | --- |
| Wrap the older strategies | Global MeanK, `overlap_16_8`, `key_fps_4`, graph v2, guided Query and the `pressure_policy` values are still chosen by option branches in `authority.py` and `prefill_attention.py`. Each becomes a registered scorer. |
| Signals by need | Serving still captures attention statistics for every strategy. A strategy's declared needs should switch unused signals off. |
| Capability checks | Launcher rejections for strategy combinations become one check of declared needs against what the configuration provides, which also generates the strategy × feature table. |
| Evictors | Historical-GC policies move behind the same registry. |
| Directories | Move modules into one directory per layer and extend the import rule to serving and core. Last, because it conflicts with every open change. |
| Offline replay | A CPU tool that runs a strategy over a recorded trace and reports overlap and evidence coverage. |

## Acceptance for every step

The built-in strategy must be token-identical at one request with an identical decision
trace, and throughput must stay inside the run-to-run band.
