# Attention maintenance and propagation optimization, 2026-10-20

All four switches remain **default off**. Implementation, fixed-input component
correctness, six real-model cross-turn state runs and component measurements are
complete. The four independent full-C8 performance comparisons are running;
end-to-end performance acceptance is pending.

This work optimizes the **existing bypass scorer** and its consumers. It is
separate from the experimental native FlashInfer/XQA QK reuse route.
Native QK reuse is disabled in every experiment reported here. No component
percentage is a per-token or whole-model speedup.

## Source ownership and independent changes

The initial HEAD was main `f639acf`. The checkout contained 76 modified/untracked
files from native QK reuse and background historical-KV work. Related chats were
idle. Their initial contents and hashes are retained under
`artifacts/attention-overhead-20261020/initial`; they were not included in these
commits. All implementation commits are on `codex/attention-overhead-20261020`:

| Change | Commit | Switch |
| --- | --- | --- |
| Dense accepted decode maintenance and graph preparation | `d77e96b` | `action_native_decode_commit` |
| Bounded pinned downloads and CUDA events | `0a1ce13` | `action_download_pool` |
| Fused bypass normalization/page reduction | `a2fc383` | `action_fused_scoring` |
| Native V2 propagation and immutable CSR cache | `aa4d76d` | `graph_native_propagation` |
| Storage-owned DLPack carrier; eliminate redundant event queries | `38beed5` | Pool follow-up |
| No score-map staging when no sampled row is accepted | `622749b` | Decode follow-up |
| Separate Query and pressure qualification evidence | `038dfb1` | Validation fix |
| Reproducible independent measurement/state/audit drivers | `c96f312` | Validation tools |

The shared working-tree snapshot is the common serving baseline across all arms,
including the pre-existing default-off features. A separate clean checkout of the
six implementation commits was built and tested without the other chats' changes.
Archived test fixtures are shared read-only with that clean checkout.

## Semantics and ownership

### Accepted decode

The scheduler-confirmed prefix is previewed before committing the parser. The
original downloaded FP32/FP64 matrix and accepted row indices pass to C++ without
materializing selected Python row lists. Native code validates and stages legacy
scores and graph input; graph memory is reserved before staging. Commit publishes
only a fully prepared candidate. Capacity, invalid mass and abort paths leave the
parser, cumulative scores and graph unchanged.

Legacy accumulation retains sequential double additions, duplicate historical
fragments and dictionary insertion order. Graph preparation retains its original
`math.fsum`-equivalent normalization and source/top-k ordering. Rows rejected by
MTP are not validated or accumulated as observations. Frames with no accepted
sample only advance parser/graph progress and retain the published score map.
The existing native prefill commit implementation is unchanged.

`action_commit_max_bytes=16777216` bounds native decode staging. Graph-enabled
staging is additionally charged to the existing shared/per-session graph budget.

### Download pool

`action_download_max_bytes=67108864` bounds all cached/in-flight pinned payload;
`action_download_max_slots=128` separately bounds event/storage slots. Both early
int64 metadata and final mass downloads use the pool. A storage-owned DLPack
lease survives Torch views/detach, NumPy aliases and every event wrapper reference.
Only the last CPU/reference release **and** DMA completion permit reuse.

Completion already observed by a consumer is remembered, avoiding another event
query. A failed/unrecordable CUDA fence quarantines its slot. Capacity exhaustion
raises before unbudgeted allocation and never waits for another owner's batch.
The cached pool lasts for the runner's lifetime; free cached bytes are still
charged. CUDA driver event bookkeeping is not claimed as an exact byte count;
the event count has its own fixed cap. Older extensions use a safe NumPy carrier,
reported as such; the measured candidate uses the native DLPack carrier.

### Bypass scorer fusion

RoPE, Q dtype/quantization, FP32 sampled QK product/reduction, scale order, sampled
layers/rows and causal mask remain the original operations. The producer stores
one logsumexp per logical segment and Q head. The consumer obtains the full-row
logsumexp over **all** segments, computes segment probabilities and averages heads.
There is no complete token-logit or token-probability matrix.

The caller must guarantee the segments partition the complete visible context;
the runtime already validates this dense working-to-logical mapping. Other API
callers lacking that contract retain the reference full softmax. Each Q head is
normalized separately over the full context, including segments not transported
by later selection. Duplicate history identities retain additive segment mass.
Ordinary FP32 softmax/logsumexp rounding differences are allowed; no narrower Q/K
precision or different scale is substituted.

### Native graph propagation

V2 reads one immutable seed vector, uses a single alpha and splits multi's alpha
equally between locality and attention. The CSR stores original logical keys and
unmodified retained weights. Every call rebuilds effective edges and source-wise
attention denominators after its **transport** mask; the scoring mask is not
substituted for that mask. Missing/masked/self edges, locality message boundaries,
zero seeds and unchanged mandatory seeds retain the reference semantics.

Each owner retains at most one immutable CSR entry, bound to account identity,
owner/epoch/branch/revision/accepted prefix and immutable data identity. Restore or
revision changes invalidate it. Index construction and per-call vectors are
reserved against graph budgets; an active scorer retains its charge independently
of cache eviction. Owner/snapshot close releases the cache. V1 and explicit trace
use the reference path with an explicit fallback diagnostic.

## Correctness qualification

- Shared working tree: **3,868 passed, 159 skipped, 13 failed** in the full host
  run (`host-tests-r2.txt`). All 13 failures reproduce on the saved **initial**
  working-tree source (`baseline-existing-failures.txt`): five original stale
  test/fake-runner issues, plus eight background-KV fixture/option assumptions.
- Clean commits, separately compiled extension: **3,788 passed, 127 skipped,
  5 baseline failures** (`standalone-host-tests-r2.txt`). The first collection
  attempt lacked an archived allocator fixture; it is retained, not relabeled.
- GPU environment: **69 cases passed** across the combined regression, explicit
  native-QK/window opt-in regression and extra wide-address case. These include
  the pool's CPU ownership cases as well as actual GPU cases. Coverage includes
  GQA, BF16/FP16/FP8 Q, nonunit scales, causal/partial/noncontiguous pages, more
  than 64 sampled rows, KV addresses past 2 GiB, large int64 coordinates, CUDA
  Graph replay and retained CPU aliases. The first pool graph test had an
  incorrect expectation before its first replay; both attempts are retained.
- Native propagation compares exact scores, diagnostics and selected pages to
  the unchanged sparse reference, plus an independent dense matrix oracle.
  Masks change between calls to a cached index. Restore, equal-name owner
  isolation, concurrent scoring, staging failure and budget rollback are tested.

Six frozen real-model state cases in `state-r2` use the same original two-user,
three-turn fixture: reference, each switch independently, then all four together.
They retain NVFP4, FP8 KV, compiled/CUDA Graph, MTP4, V2 multi/alpha .2 and graph
pressure. **All six passed, each with 6/6 expected synthetic answers**, four
target+MTP byte restores and four recurrent byte restores. Accepted-output parser
oracles, MTP rejections, terminal graph counts, actual changed selections,
mandatory contents, tool continuation, edited prefixes and explicit release pass.
This is a narrow dependency fixture, not a general model-quality result.

The original `state-r1/reference` failed because the old client counted pressure
graph reports as duplicate Query reports. Its failed status/logs remain intact;
`reference-supplementary-audit.json` rechecks the original evidence with the phase
fix. `038dfb1` keeps exactly one Query report and validates pressure separately,
including alpha and accepted-prefix bounds. Its 31 unit tests pass; all six fresh
state runs use the corrected verifier. Compact proof summaries and raw hashes are
in [`attention-overhead-state-20261020.json`](benchmarks/attention-overhead-state-20261020.json).

## Independent component costs

Ten repetitions alternate reference/optimized order, with fixed operands and
acceptance masks. Values below are mean **per eight-owner component invocation**,
not accumulated durations or per-output-token costs. Different rows measure
different scopes and must not be added or extrapolated into model throughput.

| Component | Reference | Optimized | Scope |
| --- | ---: | ---: | --- |
| Decode maintenance | 20.392 ms | 14.230 ms | Eight actual prepare/commit transactions; one sampled row per owner, MTP accepted prefix 3/5, 1,662 target fragments |
| Pool, prefill frame | 0.561 ms | 0.420 ms | One int64 metadata + eight 32-row mass downloads, allocation/event/lease, completion and CPU consumption |
| Pool, decode sampling frame | 0.195 ms | 0.181 ms | One int64 metadata + eight one-row mass downloads, same full resource lifecycle |
| Fused scorer, prefill sampling | 57.277 ms | 47.336 ms | Eight owners, three bypass scorer calls/layer mean/download, 32 sampled rows per owner |
| Fused scorer, decode sampling | 4.552 ms | 4.559 ms | Same scope, one sampled row per owner; effectively flat in this probe |
| V2 propagation, cached CSR | 57.314 ms | 6.217 ms | 8,192 nodes, 32,734 edges; changed mask on every call, all seed/filter/normalization/output costs |

Evidence: `component-{decode,pool,fused,propagation}-final.json`, with the exact
driver SHA and retained driver `final-component-driver.py`. Decode excludes
unchanged GPU scoring/download and raw archive publication. Pool excludes QK and
scope/parser preparation. Fusion excludes unchanged native model attention,
window/metadata preparation and archive publication. Its controlled three scorer
calls reuse fixed Q/K operands; it is not an emulation of all model-layer traffic.
Propagation's first index construction and memory reservation are reported
separately in its JSON and are included by the full model experiment. The native
first build plus propagation took 16.496 ms and reserved 2,628,000 index bytes.

The pool prefill reference includes a 1.781 ms allocation sample; its median is
0.426 -> 0.419 ms (1.5% lower), versus the 25.1% mean reduction. Decode medians
are 0.189 -> 0.180 ms (4.8% lower). This is a small lifecycle benefit in the
component, not evidence of a 25% steady model improvement. All sample values,
means, medians, ranges and raw hashes are preserved in
[`attention-overhead-components-20261020.json`](benchmarks/attention-overhead-components-20261020.json).

Fusion uses actual model head geometry (24 Q / 4 KV heads, D=256), BF16 Q in
**both** phases, FP8 KV, 1,616-token native pages, unit scales and 53,152 active
tokens. The final probe reduces peak temporary GPU allocation from
403,701,760 to 6,154,240 bytes for prefill and 12,615,680 to 193,024 bytes for
decode. The earlier generic FP8-Q/page16 component result is retained as a
diagnostic, not used as the model-matched performance result. The first NumPy
pool carrier's decode regression is also retained; it motivated the DLPack
follow-up rather than default enablement.

## Full C8 comparisons

`model-c8-r3` freezes 16 fresh cohorts: reference/optimized/reference/optimized for
each switch. Exactly one switch changes within each comparison; the other three
and native QK reuse are off. All arms keep the original passed V2 D scoring and
context policy: multi, alpha .2, graph pressure, identical capture density and
Mean-K Query configuration.

Each request has the full original 262,101-token input and 16,384 accepted output
tokens. All arms use NVFP4, FP8 KV, MTP(k=4), K=36,864, G=16,384,
batch=16,424, chunk=1,024, 100 GB CPU history, no historical SSD I/O and disabled
prefix caching. No acceptance mask, sampling/graph parameter or profile capacity
is reduced for an optimized arm. The original strict streamed-ID/native-window
qualifier and an independent complete executed-step audit are retained.

The first coordinator attempt (`model-c8-r1`) rejected launcher arm `D` at argument
parsing, **before any model was launched**. That failure is retained. The generic
runner accepts launcher label `multi`; its frozen manifest still supplies `D` to
the unchanged strict V2 client. `model-c8-r2` used this correction.

That attempt completed its reference cohort, but its first optimized cohort had
eight successful full HTTP outputs and only seven terminal archive publications.
The unchanged strict snapshot gate failed on a delayed scheduled-prefill count;
the client returned before its normal retirement wait and the controller then
stopped the engine. The retained snapshot had one retiring authority and
9,695,002,624 CPU authority bytes. This is a failed lifecycle qualification, not a
performance result. Both cohorts and all original failed statuses are retained
and excluded from the final comparison.

`632ccbe` moves the existing terminal and release waits before returning any
strict snapshot failure. It preserves that failure, the original input/output
checks and every strict gate. Thirty-nine related unit tests pass, including
failure-path cleanup and rejected terminal/checkpoint evidence. No serving
source or numerical path changes. `model-c8-r3` starts all 16 fresh cohorts using
the corrected client.

Results remain pending. The final analysis must separately report engine
startup/JIT, request cold preparation/input wall rate, model-prefill event rate,
shared C8 output rate, full/model step event intervals, whole ready-batch wall
throughput, native MTP acceptance, CPU RSS/PSS, GPU allocation/reservation/device
memory, actual modes and resource fallbacks. Step events do not cover every
inter-step CPU task; shared/whole-batch wall rates include those costs. CPU memory
is sampled every five seconds and device memory every second, equally in all
arms. Generation/Query/pressure identities and verification counts must be
compared explicitly. Two repetitions per arm cannot establish a stable causal
throughput estimate, and per-component savings cannot be summed.

The all-four state case establishes compatibility. No combined full-C8
performance number is inferred from the independent comparisons.
