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
  (`layerwise_pressure.py`). Prefill pressure, the final Query, the agent Query and
  rolling-decode pressure have run end to end with a non-default scorer, with sessions,
  restore and historical GC on, at one and at four concurrent sessions.
- The gate as a tracked command, `benchmarks/gate/gate.py`, with both fixtures.
- Picker seam at `RequestMemory.preview_selection`, the funnel for global and per-layer picks.
- Four typed options replacing what would have been environment flags.
- A launch-time check that refuses a scorer the launched path would ignore. The scorer
  seams exist on the per-layer path only; the global path still follows
  `query_score_mode` and `pressure_policy`.

## Remaining steps

Seven steps. Each ends at the [gate](#the-gate) below.

### 1. Prove the seam on the agent path (done 2026-11-12)

The seam had run on offline prefill only. Agents add sessions, historical GC, rolling
decode and several owners, which is where earlier restore bugs appeared.

- Built-in strategy on the agent fixture (one session, 12 turns, saved token IDs): 13 of 13
  requests token-identical to the commit before the strategy layer, and 160 of 160 layer
  decisions identical, 96 of them in rolling decode.
- The `recency` scorers on the same fixture: all 12 turns complete; every Query and pressure
  decision, including the 6 rolling-decode ones, keeps the same newest blocks in all 16
  layers, as the scorer intends.
- Four-session smoke runs, built-in and `recency`: 48 of 48 turns, no error, and no request,
  page, reservation or archive work held after the sessions are released.
- References: `mab` is the 2026-11-12 run of the commit before the strategy layer. `agent`
  is new; the earlier tool-wait runs used 24K of history and made no selection at all.
- The gate is a tracked command ([Running the gate](#running-the-gate)).

### 2. Audit and choose (done 2026-11-12)

- [strategy-audit.md](strategy-audit.md): every strategy against every path, as runs,
  rejected (at launch or late) or silent.
- Main finding: the per-layer path silently ignores every global Query strategy (window
  overlap, key FPS, the MeanK quota, both graph versions, guidance, Query history, the recall
  methods). Each shapes only the shared frame, while every attention layer still picks by
  per-layer MeanK. Five refusals come only at the first refresh.
- Chosen for step 3, the recommendation below: per-layer and global MeanK, key FPS, graph
  attention, attention and recency pressure, and the GC eviction policies. Window overlap,
  graph v1, guidance, Query history and Query token scores stay on the global path, marked
  legacy. The recall-experiment methods are left as they are. Nothing is deleted.

### 3. Port the chosen strategies

- Each becomes a registered scorer with declared needs.
- The option branches in `authority.select_for_queries` and
  `prefill_attention.select_pressure` become registry lookups, so the global path and the
  per-layer path choose strategies the same way.
- Extra condition for this step: each ported strategy's own trace is unchanged by the port.

### 4. Signals by need, capability checks

- Serving computes only the signals the chosen strategies declare. A strategy that does
  not read attention statistics stops paying for them (about 3–6% of GPU time today).
- One check of declared needs against what the configuration provides replaces the
  hand-written combination rejections in the launcher, and generates the table from step 2.

### 5. Evictors and offline replay

- Historical-GC policies are registered as evictors and chosen through the registry.
- Eviction decisions join the trace.
- A CPU tool replays a trace recorded with `strategy_trace_scores` through another
  strategy and reports overlap with the recorded choice and, where the benchmark marks
  it, whether the gold evidence was selected. The trace must also record mandatory
  blocks and the budget for this.

### 6. Typed configuration

- 217 `KVMEM_*` names are read in 94 modules. First route every read through one
  accessor, with no behavior change.
- Then move the source from environment variables to two profile namespaces: `system`
  settings never change output, `strategy` settings may. Environment variables remain as
  overrides for A/B runs; the default-on table becomes a preset.
- Extra condition: the effective settings of the reference deployments are unchanged.

### 7. Directory moves

- Move modules into one directory per layer, leave import shims at the old paths for one
  release, and extend the import rule to every layer.
- Do it in one short freeze with all other work merged and paused: it conflicts with
  every open change.
- Extra condition: the full CPU suite has the same failures before and after.

### Order

Steps 1, 2 and the replay tool in step 5 are independent and can run in parallel.
Steps 3 and 4 follow step 2. Step 6 comes before step 7 so that the move is a pure rename.

## The gate

Every step must leave the built-in strategy:

- token-identical at one request on both fixtures: 16 memory-benchmark cases (offline
  prefill, 256 Query and 4,144 pressure layer decisions) and the agent simulator;
- with an identical decision trace (`python -m vllm_kvmem.strategy.trace diff`);
- inside the run-to-run spread for throughput, measured against a repeat of the
  reference, not against its first, cold run.

When running it:

- Run from a frozen copy of the tree, never from a checkout that is being edited.
- Record a new reference whenever GPU kernels are rebuilt or re-tuned. A stale reference
  differs from the first decision on, by a few blocks, whatever the code.

### Running the gate

`benchmarks/gate/gate.py` runs both fixtures and every check:

```sh
python benchmarks/gate/gate.py freeze RUNTIME          # the working tree plus built extensions
python RUNTIME/benchmarks/gate/gate.py gate RUNTIME OUT --reference REF --autotune-dir TUNE
```

- `REF` holds four arm directories, or links to them. `mab` and `agent` are compared for
  tokens and decisions. `mab-repeat` and `agent-repeat` are repeats of them, used for
  throughput.
- The command writes `OUT/gate-result.json` and exits 0 only if both fixtures are exact
  and neither loses more than `--spread` (2%) of throughput against its repeat.
- Single steps: `gate.py run {mab,agent} RUNTIME OUT`, `gate.py compare REF RUN`,
  `gate.py measure RUN`. A reference is recorded with `gate.py run` on the reference tree.

| Fixture | File | What it runs | Throughput |
| --- | --- | --- | --- |
| `mab` | `mab16.json` | 16 memory-benchmark cases, offline prefill, 4,400 layer decisions | prefill-phase tok/s |
| `agent` | `agent.json` | One session: 48K history, then 12 turns of ~300 appended and 2,048 generated tokens; 160 layer decisions, 6 of 9 pressure events in rolling decode | output tok/s inside turns |

- Both fixtures use G = 4,096. The agent history exceeds the 36,864-token select budget, so
  the agent Query, prefill pressure and rolling-decode pressure all decide. At 24K history,
  as in the earlier tool-wait runs, the agent path makes no selection at all.
- `--sessions 4` on the agent fixture is the multi-owner smoke run. Its outputs depend on
  timing, so it checks that every turn completes, that no traceback appears and that the
  server holds no request, page, reservation or archive work after the sessions are released.
- The agent fixture keeps its own FlashInfer autotune cache (`--autotune-dir`), so tuning by
  other runs on the machine cannot change its GEMM kernels. The `mab` reference was recorded
  with the default cache and keeps it.
- The question files of `mab` are read from `--data`.

Working notes and the current reference runs are in `artifacts/layers-refactor-20261112/`,
which is local and not part of a source checkout.

## Open decisions

| Decision | Recommendation |
| --- | --- |
| Move `main` forward to the strategy-layer commit | Yes: it is additive and exact, and other work then builds on the seam |
| Comparison set for the retrieval paper | Adopted for step 3: per-layer MeanK, global MeanK, key-FPS, graph attention, recency, and the eviction policies. Whether the recall-experiment methods join it is open |
| Legacy experiments | Marked legacy in the audit, not deleted. Deleting needs a list confirmed by the user |
| When to hold the freeze for step 7 | After steps 1–6 are merged |
