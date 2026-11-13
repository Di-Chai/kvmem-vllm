# Strategy audit

Every way of choosing blocks that serving can run today, against every pipeline path. Step 2
of the [layering plan](layering.md). Code state: `f4353ea` (2026-11-12).

**Runs** means the strategy decides on that path. **Rejected** means a launch or option check
refuses the combination; **late** means the refusal only comes at the first refresh, after the
server is up. **Silent** means the combination starts and serves, but the path does not use the
strategy for its decisions.

## Paths

| Path | Enabled by | Who decides |
| --- | --- | --- |
| Global | no `--layerwise-*` flag | One selection for all layers: `authority.select_for_queries` (Query) and `prefill_attention.select_pressure` (pressure), in `async_refresh._assemble` |
| Per-layer Query | `--layerwise-query-diagnostic sequential` (+ optimized, native batches) | Each attention layer picks its own window (`layerwise_query.py`). A global plan is still made for the shared frame and, with `--layerwise-query-mtp-selection global`, the MTP layer |
| Per-layer pressure | `--layerwise-prefill-pressure` (+ single assembly, pipeline, promise, fused native pick) | `layerwise_pressure.prepare`, per layer |
| Rolling decode | `--layerwise-rolling-decode` | Per-layer pressure during decode, scored from accepted output |
| Tensor parallel | `tensor_parallel_size > 1` | Global only; every `--layerwise-*` flag is rejected at launch |

Sessions, restore and historical GC run on every path; they change what history exists, not
how blocks are scored. Their only strategy is the GC eviction policy (last rows).

## Table

| Strategy | Option | Global | Per-layer Query | Per-layer pressure | Rolling decode | Tensor parallel |
| --- | --- | --- | --- | --- | --- | --- |
| Global MeanK | `query_score_mode=mean_k` (default) | Runs | Shared frame only; layers use per-layer MeanK | n/a | n/a | Runs |
| Window overlap | `query_score_mode=overlap_16_8` | Runs (needs `graph_mode=multi`, v2, alpha 0 or 0.2) | **Silent**: layers use per-layer MeanK; overlap shapes only the shared frame | n/a | n/a | Runs |
| Key FPS | `query_score_mode=key_fps_4`, `query_fps_*` | Runs (needs graph multi v2 alpha 0, or graph off with historical GC) | **Silent**, as above | n/a | n/a | Runs |
| MeanK quota on FPS | `query_mean_quota_fraction` | Runs (key FPS with halo 1 only) | **Silent** | n/a | n/a | Runs |
| Graph propagation v1 | `graph_mode` locality, attention, multi; `graph_scoring_version=v1` | Runs (pressure must be `recency`) | **Silent**: graph metadata is kept, layers ignore it | n/a | Rejected (needs `attention` pressure) | Runs |
| Graph propagation v2 | `graph_scoring_version=v2`, `graph_alpha`, `graph_aggregation` | Runs | **Silent** | n/a | n/a | Runs |
| Guided Query | `guided_query_tokens`, `guided_reselect` | Runs | **Silent**: shared frame only | Rejected, **late** (`layerwise_pressure._history`) | Rejected, late | Runs |
| Query history | `query_history`, `query_history_decay` | Runs | **Silent**: shared frame only | Rejected, **late** | Rejected, late | Runs |
| Recall-experiment methods | `KVMEM_RECALL_METHOD` (`mean_k`, `fixed4`, `fps4`, `min_max`, `exact_mass`, value methods) | Runs (graph, guidance, history off) | **Silent**: shared frame only | **Silent** | **Silent** | Runs |
| Query token scores | `--experimental-query-token-scores` | Runs (Query-only multi, no guidance) | **Silent**: shared frame only | n/a | n/a | Runs |
| Recency pressure | `pressure_policy=recency` | Runs | Rejected, **late**, without per-layer pressure (`layerwise_query`) | Rejected, **late** | Rejected at launch | Runs |
| Attention pressure | `pressure_policy=attention` | Runs | Runs | Runs (built-in scorer) | Runs | Runs |
| Graph pressure | `pressure_policy=graph` (v2, scoring graph mode, non-thinking scope) | Runs | Rejected, late | Rejected, **late** | Rejected at launch | Runs |
| Decode scoring scope | `action_scope`, `action_interval`, `action_layers` | Runs | Runs | n/a | `action_interval` runs; scope comes from `KVMEM_LAYERWISE_DECODE_SCOPE` (`all`, `non-thinking`) | Runs |
| Per-layer Query scorer | `strategy_query_scorer` (`meank`, `recency`) | Rejected at launch | Runs | n/a | Runs (agent Query) | Rejected at launch |
| Per-layer pressure scorer | `strategy_pressure_scorer` (`attention`, `recency`) | Rejected at launch | n/a | Runs; **late** refusal with pressure batch selection | Runs | Rejected at launch |
| Picker | `strategy_picker` (`native` only) | Runs | Runs | Runs; group selection and prefill-fused picks refuse another picker | Runs | Runs |
| GC eviction: LRU | `historical_gc.policy=lru` | Runs | Runs with `KVMEM_LAYERWISE_HISTORICAL_GC=1` | Same | Same | Not verified |
| GC eviction: redundancy | `historical_gc.policy=redundancy` (thinking-only, LRU fallback) | Runs | Same | Same | Same | Not verified |

Not strategies, though listed in the plan: **Query replay** (`query_replay_budget`, compiled
replay) and **no-probe Query** (`--layerwise-query-no-probe`) change how the Query rows are
produced, not how blocks are scored. No-probe refuses guidance, Query history, repository GC
and historical GC at launch.

## Findings

1. **The per-layer path ignores the global Query strategies without a word.** With
   `overlap_16_8`, `key_fps_4`, the MeanK quota, either graph version, guidance, Query history,
   Query token scores or a recall method, a
   per-layer server starts and serves, but every attention layer still picks by per-layer
   MeanK. The configured strategy shapes only the shared frame. It still pays for its state:
   FPS caches, window caches and graph collection.
2. **Five refusals come late.** Guidance, Query history and non-attention pressure on the
   per-layer path, and a non-built-in pressure scorer with batch selection, fail at the first
   refresh instead of at launch.
3. **The recall-experiment methods form a second, parallel strategy system** (environment
   variables, their own scoring entry `select_for_event`), next to `query_score_mode`. They run
   only on the global path.
4. **Per-layer strategies cannot run on the global path.** Only the built-in MeanK and
   attention scorers exist on both.
5. Strategies are chosen in three places: profile options (`query_score_mode`,
   `pressure_policy`, graph and guidance options, `strategy_*`), launch flags, and environment
   variables (`KVMEM_RECALL_METHOD`, `KVMEM_LAYERWISE_DECODE_SCOPE`,
   `KVMEM_*_PRESSURE_METHOD`).

Evidence: `options.py` (`validate_query_score_mode`, `validate_graph_options`,
`resolve_graph_policy`), `serve.py` (`main`, `validate_strategy_scope`), `layerwise_query.py`
(selection dispatch), `layerwise_pressure.py` (`_history`, batch selection),
`layerwise_decode.validate_configuration`, `async_refresh._assemble`, `recall_experiment.py`,
`historical_gc.py`. The "Not verified" cells have no launch check either way and no recorded
run.

## Choice for the retrieval paper

The recommendation in [layering.md](layering.md#open-decisions), adopted for step 3 unless
changed:

| Ported (step 3) | Name in the registry |
| --- | --- |
| Per-layer MeanK | Query scorer `meank` (in place) |
| Global MeanK | global Query scorer `meank` |
| Key FPS (with optional MeanK quota) | global Query scorer `key_fps` |
| Graph attention (v2 propagation and graph pressure) | global Query scorer `graph`, global pressure scorer `graph` |
| Attention pressure | pressure scorer `attention` (per-layer and global) |
| Recency | Query and pressure scorers `recency` (per-layer and global) |
| GC eviction policies | evictors `lru`, `redundancy` (step 5) |

Kept on the global path, unported and marked legacy: `overlap_16_8`, graph v1, guided Query,
Query history, Query token scores. The recall-experiment methods stay as they are: they are a
comparison harness of their own, and whether the paper uses them is open. Nothing is deleted.
