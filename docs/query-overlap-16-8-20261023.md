# 16/8 overlapping content-Key scoring

This is an opt-in Query scorer. The default remains `mean_k`. This qualification
only covers 16-token windows with stride 8 and ordinary window Mean-K.

## Configuration

Add these fields to the existing `kvmem` profile, preserving the existing model,
physical KV configuration, sampling, fixed keeps, and pressure settings:

```json
{
  "query_score_mode": "overlap_16_8",
  "query_window_cache_bytes": 2147483648,
  "block_tokens": 32,
  "graph_mode": "multi",
  "graph_scoring_version": "v2",
  "graph_aggregation": "sum",
  "graph_alpha": 0.2,
  "query_history": false,
  "guided_query_tokens": 0
}
```

Set `graph_alpha` to `0` for the no-propagation ablation. This passes the exact
window seed scores to the native picker; the graph lifecycle and collection
stay enabled to keep the other execution conditions fixed. Query reports mark
`identity=true` and `propagation_applied=false`. The qualified coefficients are
currently `0` and `0.2`; other coefficients remain rejected for this scorer.

The cache limit is shared across the archive's owners, not multiplied by the
number of requests. The scheduler and archive reserve it once from the existing
CPU history/read-cache allocation. Inactive caches may be evicted; active
scoring buffers cannot be reused. Capacity exhaustion raises before opening
the native selection transaction. It does not silently switch scoring modes.
Switching back to `query_score_mode=mean_k` disables the extra cache reservation.
For this model, one complete 256K FP16 window index plus metadata occupies
about 1.02 GiB. The 2 GiB setting used here qualifies serial Query scoring;
it cannot hold two such active indexes simultaneously. Concurrent scoring
requires a larger explicit shared reservation. This round does not qualify
eight simultaneous Query scorers or concurrent throughput.

## Computation

Windows start at original logical multiples of eight. A window must contain
every original token in `[start, min(start+16, history_end))`. Interior holes,
missing portions of interior blocks, and any block excluded by the native
`score_mask` invalidate the entire window. Only the global history tail may
shorten a window. No compressed active-view coordinates are used.

Archived content Keys already have the existing raw quantization/scale. The
new index applies no extra RoPE or positioned-cache scale. It computes a plain
FP32 mean and stores the mean at the original backend's index precision:
FP16 for `qw3_cuda`, FP32 for `torch_reference`. Q retains the existing backend's
conversion rule. GQA maps each Q head to its original KV head.

For each layer, Query token and Query head independently:

1. Compute `u = Q·Mean(K_window)/sqrt(head_dim)`.
2. Max all valid intersecting window logits into each original 32-token block.
   Initialize invalid windows and block max accumulators with negative infinity.
3. For nonempty blocks with no valid window, use the original block Mean-K logit.
4. Apply the native block `score_mask`, then softmax over blocks.
5. Average over layers and heads; sum over Query tokens to obtain `x`.

Thus `[24,40)` supplies a candidate logit to both `[0,32)` and `[32,64)`.
No layer, head or Query averaging occurs before max. This window-to-block max is
separate from graph aggregation: graph aggregation remains **sum**.

The existing V2 multi operator normalizes `x` within `transport_mask`, computes
`0.8r + 0.1 P_L^T r + 0.1 P_A^T r`, and restores the normalization factor.
`transport_mask` never controls window validity or the scoring softmax.
The existing native picker/assembly transaction enforces budgets and fixed keeps.
No neighbor is added after selection. Prefill pressure continues to use the
configured cumulative attention path.

## Cache and lifetime

The cache contains derived means only, never archive leases or KV records.
Each entry is qualified by original position, native slot/generation, valid
count, and the generation/extent of any crossing window's neighbor. Changed
counts, gaps and generation reuse cause reconstruction. Masks are reapplied
for each Query. Rewind clears the derived cache before branch restoration;
request close releases its allocation.

The cache grows only within the shared bound, discarding its prior allocation
before resize. A conservative per-block metadata allowance is included. Raw
reads are batched to at most 32 rebuilt source blocks plus their dependencies.
GPU scoring tiles layers and Query rows, with an explicit 512 MiB scratch limit.
The first Query builds the index lazily; subsequent compatible queries reuse it.
This implementation qualifies semantics and is not a fused scoring kernel.

## Validation

Independent scalar-oracle tests cover GQA, FP8 raw K, mean/Q conversion, custom
scale, negative logits, max-before-aggregation, cross-boundary windows, holes,
partial tails, score versus transport masks, and original-MeanK fallback.
Native integration tests cover final graph scores/picker budget, fixed sink,
no neighbor expansion, stale graph rejection, read/capacity failures, owner
isolation, active-reader eviction protection, and rewind/branch generations.

The GPU numerical test is opt-in (`KVMEM_TEST_QUERY_WINDOW_CUDA=1`) to avoid
interfering with another experiment. It has passed on the local Pro 6000.

The saved real-Q/K component check is in
`artifacts/overlap-window-component-20261023.json`. For the cross-boundary
`niah_multikey_2-0009` example, target block seed ranks changed from 652/3224 to
491/800. Cold and cached scores were identical. This check reads its cold raw
keys from an on-disk diagnostic capture; its 54.26-second cold cost is not a
measurement of the serving CPU archive. The cached check took 1.82 seconds.
These ranks precede graph propagation and do not themselves prove answer quality.

The actual answer qualification uses only this 16/8 configuration on the same
12-case failure/control cohort, retaining original 256K inputs, seeds and
sampling. It is not a full-RULER accuracy estimate. Its graph setting is the
requested V2 multi sum with alpha .2; the earlier 12-case MeanK/token-sum runs
used alpha zero, so comparisons with those runs also change propagation.

## Completed answer qualification

The frozen run `artifacts/ruler-overlap-16-8-20261023-r1` completed all twelve
cases. Its final execution audit and GPU cleanup passed. Original input, seed,
sampling and old-answer hashes were preserved; every new selection matched
offline V2 sum propagation and native picker replay. The test service is stopped.

| Configuration | Correct answers | Complete target-value coverage |
|---|---:|---:|
| Previous MeanK, alpha 0 | 3/12 | 3/12 |
| Previous token softmax/page sum, alpha 0 | 9/12 | 9/12 |
| 16/8 window MeanK + V2 multi sum, alpha .2 | 5/12 | 5/12 |

The two new correct answers relative to MeanK are `0009` and `0018`; original
successful controls remain correct. Native picking from the window seeds alone
would cover seven target values, but the requested sum propagation displaces
the complete target-value blocks in `0008` and `0019`. That offline coverage
count is not a second answer-generation experiment or a measured 7/12 accuracy.

All twelve first-Query cache-build/read/transfer/score intervals had a median
of 40.25 seconds (P90 41.16 seconds). Each cache occupied 1.016 GiB including its
metadata allowance. Every case used a fresh owner, so these are cold-cache
measurements; the separate fixed-capture warm measurement remains 1.82 seconds.
No end-to-end performance improvement is claimed. The scorer remains opt-in.

Detailed evidence: [answer report](../artifacts/ruler-overlap-16-8-20261023-r1/report.md),
[per-case analysis](../artifacts/ruler-overlap-16-8-20261023-r1/analysis.json),
and [execution audit](../artifacts/ruler-overlap-16-8-20261023-r1/execution-audit.json).

## Completed no-propagation answer ablation

The separate frozen run `artifacts/ruler-overlap-16-8-no-graph-20261023-r1`
re-generated all twelve answers with `graph_alpha=0`. Its profile differs from
the preceding overlap run only in alpha and service port. Model/attention/
window implementation sources are unchanged; the only runtime-source changes
allow the zero coefficient and report whether propagation was applied.

All twelve original window seed vectors, logical indexes and masks matched
the previous run exactly. Actual correct answers and complete target-value
coverage improved from **5/12 to 7/12**. Cases `0008` and `0019` recovered;
no previously correct answer regressed. This is measured answer generation,
not an inference from the prior seed-only coverage preview. Case `0003` still
misses one target token across the block boundary; `0010`, `0011`, `0012`, and
`0015` do not retrieve the target value. The cohort remains failure-enriched,
so these fractions are not full-RULER accuracy estimates.

Cold window read/build/transfer/scoring took a median 39.46 seconds; request
time was a median 90.01 seconds. Different generated lengths prevent treating
whole-request timing differences as isolated propagation speedup. All case
hashes, original reference answers, native picker replay, final execution
audit and GPU cleanup passed; the service is stopped.

Evidence: [no-propagation report](../artifacts/ruler-overlap-16-8-no-graph-20261023-r1/report.md),
[paired per-case analysis](../artifacts/ruler-overlap-16-8-no-graph-20261023-r1/analysis.json),
[execution audit](../artifacts/ruler-overlap-16-8-no-graph-20261023-r1/execution-audit.json).

The historical full-context native vLLM answers for these same twelve cases are
**12/12 correct**, with matching input hashes, seeds and sampling. Its actual
concurrency was four versus one in this diagnostic; it is not a matched
throughput measurement. The five remaining overlap failures are all answered
correctly by full context. See the immutable
[native comparison](../artifacts/ruler-overlap-16-8-no-graph-20261023-r1/native-context-comparison.json).

### Why sum propagation displaced the two recovered targets

CPU replay of the original `.2` run decomposes the score changes as follows;
the replayed full selection matches the recorded selection.

| Case / target block | Seed | Self retained | Locality inflow | Attention inflow | Final | Rank |
|---|---:|---:|---:|---:|---:|---|
| 0008 / 86368 | .003021797 | .002417438 | .000141237 | .000058978 | .002617653 | 831 → 1137 |
| 0019 / 126208 | .002654105 | .002123284 | .000169501 | 0 | .002292785 | 1054 → 1493 |

Both targets transfer 20% out and receive too little back; their final scores
fall 13.37% and 13.61%. Uneven inflow to competitors also matters: the 140608
block in case 0008 receives attention mass from 48 sources and rises from
rank 4822 to 336. This is not simply uniform multiplication by .8.

The targets' surviving attention edges contain raw mass .03531657/.08779918;
V2 source normalization still distributes the full .1 attention share over
those edges. This implements the requested normalization but does not retain
absolute edge confidence. With each branch kept at its original .1 weight,
offline locality-only/attention-only picking still retains 0008, whereas
their combination loses it; either branch alone loses 0019. These last
branch comparisons are selection replays, not generated-answer experiments.
