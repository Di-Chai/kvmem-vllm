# Native attention statistics, 2026-10-19

Experimental and **default off**. Native kernel, fixed-input policy and component
cost checks pass. Four alternating full C8 model runs are complete: the median
full decode step falls from 42.895 to 40.124 ms (6.46%). Observed whole-batch
throughput increases 2.14%, with different generation/MTP trajectories. First-use
compilation makes the cold all-in result worse. Three real-model cross-turn
correctness runs pass. One original C8 snapshot gate failed and is retained below;
its executed interval passed a separate audit. This is not a default-enablement
decision.

## Mathematical contract and implementation

Mathematical equivalence is required; bitwise equality with the FP32 sampled
scorer is not. Normal tensor-core, quantization and floating-point rounding is
allowed. The visible causal context, Q/K scale semantics, per-head normalization,
head/layer weighting, logical history identities and accepted-row filtering must
remain the same.

The source-pinned FA2 prefill and SM12x warp-specialized XQA adapters extend the
native QK accumulator sites, XQA binding and call interface. They build isolated,
content-addressed FlashInfer source overlays. Installed dependencies are not
edited. `native_attention_sources.json` checks the exact installed C++ and Python
argument layouts before enabling the extension.

The producer reads the existing FP32 online-softmax weights before P*V conversion,
then emits `log(sum(weights)) + center` for sixteen-token pieces split at logical
segment boundaries. It does not load Q/K or compute another dot product. Aligned
partitions use unique writes; unaligned partition edges use stable logaddexp.
Each four-lane reduction synchronizes only its own four lanes, including divergent
causal tails and tiny partial segments.

For each Q head, the consumer normalizes over **all visible pieces**, including
pages later excluded by the transport policy. It sums `exp(piece_lse-row_lse)`
over a logical segment, averages Q heads and then the selected layers. Physical
KV pages never define historical identities.

## Scheduling, graphs and lifetime

Prefill and single-owner decode prepare the existing scope before forward.
Concurrent pure decode uses an immutable view layout and its bounded linear
append domain. Every view commit receives a monotonic generation, including
commits with unchanged offsets; free/reuse cannot reproduce an old generation.
A missing or invalidated cached view uses direct preparation.

Concurrent decode collects bounded MTP candidate rows before forward, then applies
the **original** scope parser and interval after forward. Only the same selected
rows are normalized and exposed to acceptance. Logical segments are checked
against the exact post-forward frame. Adjacent fragments belonging to the same
logical history page are combined; this preserves page mass, exposure counting,
output accumulation and graph aggregation mathematically. Chunk sizes remain
bounded even for repeated fragments of one page. Neither sampling density nor
graph/Query/pressure parameters change.

Per-layer LSE and metadata GPU addresses are reserved before compilation and
CUDA Graph capture. Every public frame clears its current LSE extent and refreshes
the row/piece maps. The consumer masks cached future pieces outside the current
normalization width, so a shorter context cannot read previous graph output.
Bounded pinned metadata has an upload fence before CPU reuse. Immutable CPU view
caches are bounded by request/context capacity and are released at retirement.

GPU normalizer checks run without host waits inside request/layer loops. A single
batch-end validation follows all scoring/download submissions. Invalid tentative
packets are discarded; the same frozen metadata and prepared scope are scored
through the original path before any packet reaches acceptance. Scope preparation
and accepted-history publication each occur once. Private Query forwards, draft
layers, public replay exclusions, rejected rows and terminal clipping retain the
existing acceptance rules.

## Controls

- `action_native_scores="off"` (default), `"verify"`, or `"reuse"`.
- Verify computes both paths, records differences and commits reference scores.
- Reuse omits sampled QK for each valid native scoring operation.
- `action_native_max_rows=256` bounds candidate Q rows across the batch.
- `action_native_max_bytes=2147483648` bounds persistent GPU and pinned payloads.
- Unsupported sources/backends/geometry, capacity, unwarmed graph variants,
  changed cached domains and invalid producer results have explicit fallback
  counters. A nonunit FA2 Q scale applied to unquantized Q, when the reference
  ignores it, is a semantic scale mismatch and falls back.
- `capture/native_attention_stats.json` records actual prepare/scoring modes,
  fallback reasons, candidate/normalized row counts, buffer bytes and verify errors.
  These are candidate work counts; accepted-sample logs remain authoritative.

## Correctness evidence

The final combined qualification passed **41 tests**: 19 native GPU cases,
18 host contracts and 4 window-buffer GPU cases (`gpu-tests-r14.txt`). Coverage
includes FA2/XQA, GQA, FP8 K scale, causal/partial/noncontiguous pages, model head
geometry, many tiny segments with different causal tails, graph replay, poisoned
future LSE, sparse nonzero sampled rows, MTP partial acceptance, exact fallback,
view generation reuse, mandatory selection and Query/pressure/checkpoint policy.
The page-coalescing tests compare original and merged output mass, prefill mass/
exposure, graph top-k rows and checkpoint restoration with both host commit paths.

The latest full host audit is **3,747 passed, 130 skipped, 5 failed**
(`host-tests-r5.txt`). The original five failures were
reproduced on unmodified `f639acf` (`baseline-existing-failures.txt`): stale expected
coordination error wording, a restore fixture missing `claim_graph_history`, and
three fake runner batches missing `positions`. Unrelated production code was not
changed to hide these failures. Earlier attempts, including test fixture failures,
the four-lane barrier blockage and the future-piece contamination, are retained.

Three frozen real-model correctness runs **passed** in `state-r2`: verify and reuse on
the unchanged two-user, three-turn collect fixture, plus reuse with the legacy
attention-quota selector. They check target/MTP/recurrent byte restores, tool
continuation, edited prefixes, accepted-token parser oracles, graph checkpoints
and explicit session release, separately from performance measurements. Verify
has passed 576 native/reference comparisons: maximum absolute page-score error
3.2783e-7 and maximum row L1 error 5.2277e-7.

| State run | Native prefill / decode operations | Target + MTP byte restores | Recurrent byte restores | Rejected-token oracle coverage | Terminal graph checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| verify, graph collect | 75 / 501 | 4 | 4 | 1,058 | 6 |
| reuse, graph collect | 75 / 516 | 4 | 4 | 1,110 | 6 |
| reuse, legacy quota | 75 / 483 | 4 | 4 | 992 | N/A, graph off |

Each run checks six mandatory selections, restores both continuation and edited
prefixes, releases all owned sessions/processes, has zero native validation
failures and only three unprepared-prefill layer fallbacks. The legacy run uses
`action_fraction=0.25`, `action_combine="quota"`; four continuation/edit Query
records actually consume output attention, each exposing 139 scored history
pages. All six expected synthetic answers are found in each run. This is a narrow
dependency fixture, not a general model-quality claim. `state-r2/final-audit.json`
independently checks the proof/result hashes, native phase use, restore evidence,
mandatory preservation, rejection oracles, quota use and cleanup. All seven model
jobs' 125 production source files match the current source (`final-source-audit.json`).

## Component cost

Ten repetitions alternate order on fixed Q/K, three actual native attention
calls, 24 Q heads / 4 KV heads, head dim 256, FP8 KV, native 16-token pages and
logical 32-token history pages. The driver executes ActionRuntime.prepare/capture,
including window-segment iteration, metadata, native side writes, clearing,
normalization, validation, layer mean and pinned D2H. Logical history mass is
compared after adding duplicate fragments.

| Requests | Phase | Native attention + bypass | Native attention + statistics | Observed reduction |
| --- | --- | ---: | ---: | ---: |
| 1 | Prefill, 1,024 Q / 32 sampled | 36.311 ms | 35.936 ms | 1.0% |
| 1 | Decode, 5 candidate Q / 1 sampled | 1.603 ms | 1.310 ms | 18.3% |
| 8 | Prefill, 1,024 Q / 32 sampled per owner | 213.770 ms | 195.440 ms | 8.6% |
| 8 | Decode, 5 candidate Q / 1 sampled per owner | 12.076 ms | 8.212 ms | 32.0% |

Evidence: `component-c1-r9.json`, `component-c8-r3.json`. The Store directory
snapshot is supplied from a fixed record list; its C++ serialization and actual
accepted-history/archive publication are excluded. The full model experiment
includes those costs. These are conditional sampling costs, not every-step or
model-throughput savings. Earlier component drivers precomputed the window
segments or omitted retained metadata; their larger reported savings are narrow
diagnostics, not complete cost gates.

## Rejected model candidates and final experiment

The first frozen C8 candidate (`model-c8-r1`) observed full decode steps of
47.153/46.875 ms for reuse versus 42.565 ms for bypass. Moving validity checks to
batch end and removing unused reference uploads still regressed: `model-c8-r2`
observed 45.841 versus 42.245 ms (+8.5%). Both candidates are rejected on performance.
Their completed cohorts were independently audited; remaining repetitions were
stopped and retained. Owned GPU/server groups were cleaned up, with a supplementary
record for the nested-controller shutdown timeout in the second attempt.

The final candidate removes the early decode scope/publication fence using the
cached logical layout. `model-c8-r3` freezes identical runtime sources across
reuse/off/reuse/off. It retains NVFP4, FP8 KV, MTP(k=4), eight original full256K
inputs with 16K accepted outputs each, K=36,864, G=16,384, batch=16,424,
chunk=1,024, 100 GB CPU history, no SSD and disabled prefix caching. A separate
read-only observer collects MTP counters at the same cadence in every arm.

All four runs completed with the same frozen production source and kernel
fingerprint `e23711467a3dafc0`. Each has eight successful 262,101-token inputs and
16,384-token outputs, an audited uninterrupted shared C8 decode interval, accepted
history/archive checks and cleanup. Results are in `model-c8-r3-analysis.json`.

| Run | Startup/JIT s | Request cold preparation s | Pure-prefill model token/s | Full decode step ms | Model decode step ms | Ready-batch s | Ready-batch output token/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| reuse-1 | 204.710 | 367.724 | 6,144.08 | 40.174 | 38.495 | 580.553 | 225.771 |
| bypass-1 | 79.858 | 370.064 | 5,968.28 | 42.830 | 41.584 | 587.103 | 223.252 |
| reuse-2 | 82.928 | 368.741 | 6,115.00 | 40.074 | 38.364 | 585.766 | 223.762 |
| bypass-2 | 81.907 | 369.758 | 5,953.23 | 42.960 | 41.674 | 604.453 | 216.844 |

Ready-batch time includes request preparation, prefill and output generation after
the engine is ready. Pure-prefill model rate counts actual model prefill work;
it is not the end-to-end input admission rate. Full decode step includes runtime
preparation and model execution, including statistics reduction, download and
submission gaps; it is not a QK multiplication timing.

| Median of two runs per mode | Bypass | Reuse | Observed change |
| --- | ---: | ---: | ---: |
| Full shared-C8 decode step | 42.895 ms | 40.124 ms | 6.46% less time |
| Model shared-C8 decode step | 41.629 ms | 38.430 ms | 7.69% less time |
| Pure-prefill model rate | 5,960.76 token/s | 6,129.54 token/s | +2.83% |
| Shared-C8 accepted output rate | 638.92 token/s | 673.55 token/s | +5.42% |
| Whole ready-batch accepted output rate | 220.05 token/s | 224.77 token/s | +2.14% |
| Whole ready-batch duration | 595.778 s | 583.160 s | 2.12% less time |

The first reuse process compiled new kernel variants; the baseline kernels were
already cached. Controller lifetimes including startup/run/cleanup were
793.033 / 675.358 / 675.417 / 693.970 s in execution order. Corresponding all-in
output rates were 165.279 / 194.078 / 194.061 / 188.873 token/s. They are bracketed
by 1 Hz observations, with about two seconds of endpoint uncertainty. Across
these two runs per arm, the median all-in rate is **6.17% lower** for reuse.
Warm-cache reuse startup is comparable to bypass, but this experiment does not
establish a cold single-process speedup or an amortization period.

Native MTP counters inside the shared C8 interval give accepted lengths
3.590 / 3.878 / 3.342 / 3.168 and draft acceptance rates
64.75% / 71.96% / 58.55% / 54.19%, in execution order. Input hashes, pressure
selections and mandatory fits match across arms. Query selections and accepted
output hashes differ; the two pairs' cross-owner Query history-page Jaccard means
are 0.9471 and 0.9447 (minimum 0.9182). Verification-step counts differ as well.
The scoring domain, sampling/graph parameters and policies are unchanged. These
results do not prove identical model trajectories, and the observed throughput
change cannot be attributed entirely to reuse. Two repetitions per arm are not a
stable causal throughput estimate.

Both reuse runs allocate 503,021,660 bytes for bounded native buffers:
502,652,592 GPU bytes and 369,068 pinned host bytes. They have no producer
validation failures and exactly three `prefill:unprepared_forward` layer-scoring
fallbacks each; all later prepared scores use native reuse. Actual prefill/decode
reuse operation counts are 6,165 / 32,202 and 6,165 / 35,328. Peak allocated memory
reported by the runtime is 89,178,371,072 bytes for reuse versus 89,177,570,816 for
bypass; peak reserved memory is 92,952,068,096 versus 92,960,456,704 bytes. Native
buffers are reserved before KV-pool profiling, so these pool-dependent peaks must
not be interpreted as the workspace size. The independent 1 Hz device-memory
peaks are 74,252 / 74,224 / 74,250 / 74,256 MiB in execution order.

The original qualification status is retained: reuse-1, bypass-1 and bypass-2
passed; **reuse-2's original job/snapshot check failed** because one delayed
scheduler histogram snapshot reported a prefill increment of two at the shared
window boundary. Its separate `reuse-2-executed-audit.json` verifies every actual
model submission in the full shared interval is pure C8 decode, that executed
public prefill counters remain constant, and that all eight outputs, histories
and releases pass. The snapshot gate was not modified or relabeled. The independent
audit supports the executed-step measurement, not a claim that all original gates
passed.

## Remaining work

The scoped implementation, four performance runs and three state proofs are
complete. Keep the option default off. Before considering default enablement,
investigate the original scheduler-snapshot discrepancy and broaden repetitions
with first-use compilation separated from warm serving. Further performance work,
not implemented here, includes selecting MTP candidate rows on the GPU
instead of retaining all bounded candidates, reducing metadata upload and CPU
logical-domain checks, reducing persistent statistics storage, and prebuilding
source-pinned variants to avoid first-use compilation. Additional backends and
hardware require their own adapters and qualification.
