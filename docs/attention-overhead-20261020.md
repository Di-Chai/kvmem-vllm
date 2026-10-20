# Attention maintenance and propagation optimization, 2026-10-20

All four switches remain **default off**. Implementation, fixed-input component
correctness, six real-model cross-turn state runs and component measurements are
complete. Both runs of the initial combined full-C8 comparison pass the original
strict gate and execution/lifecycle audits; whole-batch throughput is flat.
A later user-requested fresh repeat observes higher decode throughput alongside
higher MTP acceptance, but its all-on original snapshot gate fails. That repeat
is retained as diagnostic evidence, with its failure unchanged.
**Neither comparison establishes an isolated end-to-end optimization speedup.**

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
acceptance masks. Values below are means **per component invocation**: decode,
pool and fusion cover eight owners; propagation covers one 8,192-node graph.
They are not accumulated durations or per-output-token costs. Different rows
measure different scopes and must not be added or extrapolated into model throughput.

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

## Full C8 combined comparison

The final plan is **all four switches off versus all four on**, as requested by
the user after component qualification. `model-c8-combined-r1` ran only one new
all-on cohort. The primary all-off reference was declared before that launch:
`model-c8-r3/decode-reference-2`, a completed run whose original strict gate,
complete executed-window audit, eight terminal publications and resource release
all passed. Serving source/extension and C++ source hashes must match the reused
reference exactly; profiles may differ only in the four declared booleans.
Native QK reuse remains off.

The original 16-cohort independent plan was stopped after three complete cohorts
(two all-off references and one decode-only optimization). The fourth cohort was
interrupted at the user's scope change; its cleanup confirms no owned process
or GPU worker remains. The 12 subsequent individual cohorts were not run. The
completed decode-only pair is diagnostic, not the combined performance result.
All individual component results above remain independently measured.

The already completed identical all-off repetitions are reported as trajectory
noise diagnostics, without additional GPU work. Their shared decode rates were
603.08 and 743.12 output token/s, with native MTP mean lengths 3.324 and 4.109;
full step intervals were 42.906 and 42.765 ms. Query history overlap between these
same-path repetitions averaged 0.9433 and mandatory contents matched. This is
why an output-rate difference alone cannot be assigned to the optimization.

All arms retain the original passed V2 D scoring and context policy: multi,
alpha .2, graph pressure, identical capture density and Mean-K Query configuration.

Each request has the full original 262,101-token input and 16,384 accepted output
tokens. All arms use NVFP4, FP8 KV, MTP(k=4), K=36,864, G=16,384,
batch=16,424, chunk=1,024, 100 GB CPU history, no historical SSD I/O and disabled
prefix caching. Hardware is one NVIDIA RTX PRO 6000 Blackwell Server Edition
(97,887 MiB reported device memory), driver 580.159.03. No acceptance mask,
sampling/graph parameter or profile capacity
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
source or numerical path changes. `model-c8-r3` started the independent cohorts using
the corrected client; the later user-requested combined plan reuses its completed
all-off reference and keeps the same correction.

## Initial combined result

Primary evidence is in
[`attention-overhead-model-20261020.json`](benchmarks/attention-overhead-model-20261020.json).
All-off is the predeclared `model-c8-r3/decode-reference-2`; all-on is
`model-c8-combined-r1/combined-optimized-1`. Both complete eight requests with
262,101 prompt and 16,384 accepted output tokens each, pass the **original**
strict native gate and independent complete executed-window audit, publish all
eight terminal histories and release all owned history/checkpoint/graph state.
Historical SSD read/write bytes are zero. The later user-requested fresh repeat
is recorded separately below.

| Metric | All off | All on | Observed change |
| --- | ---: | ---: | ---: |
| Engine startup/JIT, s | 80.799 | 75.508 | -6.55% |
| Cold request preparation / last first-token, s | 408.457 | 404.163 | -1.05% |
| Effective input wall rate, token/s | 5,133.49 | 5,188.03 | +1.06% |
| Pure-prefill model event rate, token/s | 5,549.72 | 5,535.89 | -0.25% |
| Shared C8 accepted output rate, token/s | 743.12 | 669.63 | -9.89% |
| Full decode step event interval, ms | 42.765 | 43.099 | +0.78% |
| Model decode step event interval, ms | 41.314 | 41.770 | +1.10% |
| Whole ready-batch wall time, s | 623.712 | 623.985 | +0.04% |
| Whole ready-batch output rate, token/s | 210.148 | 210.056 | -0.04% |
| Observed controller lifetime, s | 713.023 | 707.864 | -0.72% |
| Observed output rate including startup, token/s | 183.826 | 185.166 | +0.73% |

The effective input wall rate divides all input by the maximum first-token delay;
it includes preparation, model prefill and initial Query/pressure work. The
prefill model rate uses executed prefill tokens divided by corresponding model
CUDA-event intervals, including the existing model/replay and statistics hooks.
Step events may include CPU submission gaps and exclude some inter-step work;
shared/whole-batch wall rates include those costs. Startup includes first-use
cache/JIT effects and was lower in this all-on run; its difference is not assigned
to the four optimizations. The observed controller endpoints have about two
seconds of sampling uncertainty.

### Trajectory and MTP differences

| Metric | All off | All on |
| --- | ---: | ---: |
| Native MTP mean accepted length, including bonus | 4.109 | 3.706 |
| Native draft acceptance rate | 77.73% | 67.65% |
| Total owner verification calls | 33,400 | 36,156 |
| Accepted scoped output observations | 6,492 | 7,804 |
| Accepted prefill observations | 65,384 | 65,384 |
| Candidate observations, including proposals | 74,074 | 76,252 |

Inputs match. Query, pressure-selection and accepted-output hash multisets differ.
Query history-page Jaccard is 0.9450 on average
(range 0.9246–0.9574); mandatory contents match. The
same-path all-off repetition overlap is 0.9433, so the mean overlaps are similar.
This does not prove causal identity: fixed-input component and state tests establish the intended scoring,
mask and history semantics, while closed-loop model trajectories can differ.

Shared decode output rate falls 9.89% while native MTP mean length falls
9.81%. Owner verification calls and accepted scoped sample counts increase;
the capture intervals and scoring/graph parameters remain unchanged. The output
rate change therefore does not isolate a net regression caused by
the optimization, nor rule out an effect of the optimization on trajectories.
Conversely, the lower cold/startup time cannot establish a net
optimization benefit. **Whole-batch output rate is essentially unchanged**
(210.148 → 210.056 token/s), and full step intervals are slightly higher.

### Actual graph, pool and backend costs

Across the same 112 Query/pressure graph calls, scoring takes
5.502 → 1.834 s
(-66.7%). The surrounding graph-selection block, including scoring, takes
11.514 → 7.823 s
(-32.1%). Native scoring builds 112 indexes (14 per owner) and has
**zero cache hits** because graph versions advance. These timings include index
construction and budget reservation. The hot-cache component reduction of 89.2%
is not the full-model graph reduction. Each owner's maximum reserved index is
2,623,072 bytes; the sum of per-owner maxima is 20,984,576 bytes, not an observed
concurrent process-memory peak. Graph selection does not overlap the audited
shared decode window, so its saving is not a direct shared-decode speedup.

All-on actually executes:

- 36,156 native decode commit-path calls, zero Python commit-path calls.
- 38,772 fused bypass scorer calls, zero reference-scorer fallbacks.
- 112 native V2 graph reports, zero reference-propagation fallbacks.
- The native DLPack pool carrier: 36 allocations, 18,805 reuses, 36 event slots;
  3,104,768 retained pinned bytes and 1,835,008 peak active pinned bytes, within
  the 64 MiB / 128-slot caps. Exhaustions and quarantined slots are both zero.
- Native QK reuse off. The existing early-metadata fallback occurs once in each
  arm; it is not a fallback from any of these four new optimizations.

### Memory

| Observed peak, GiB | All off | All on |
| --- | ---: | ---: |
| CUDA allocator allocated | 83.053 | 83.053 |
| CUDA allocator reserved | 86.576 | 86.576 |
| Owned process RSS, 5s sampling | 87.505 | 87.351 |
| Owned process PSS, 5s sampling | 87.123 | 86.969 |
| Device used, 1Hz sampling | 72.512 | 72.512 |

Allocator cumulative peaks and sampled device peaks are different measurements;
1Hz sampling can miss short-lived peaks. No overall GPU-memory reduction is
observed in this full run despite the separately measured scorer scratch
reduction. The roughly 0.18% process-memory difference is descriptive, not a
proven optimization benefit. Cached/in-flight pool payload and immutable indexes
have their own explicit budgets in addition to these process observations.

### Acceptance and retained evidence

Implementation and correctness qualification are complete. Component CPU costs
and scorer scratch allocation improve; this initial combined pair does **not**
demonstrate a net end-to-end throughput improvement. All four defaults stay off.
The initial single-pair scope follows the user's reduced test plan. No further
per-feature full-model cohorts were run. A subsequent explicit request to repeat
the combined comparison led to the two fresh cohorts below.

Raw failed coordinator/client attempts and the user-interrupted independent
cohort retain their original statuses and logs. They are excluded from the
primary combined comparison. Existing all-off repetitions are shown only as
trajectory-noise diagnostics. The main pair needs no exception to the original
strict gate: both original statuses are `passed`.

The final source-ownership audit confirms that 71 of the 76 initially dirty files
remain byte-identical. The other five contain separately committed additions in
shared files; the pre-existing modifications remain outside these commits. No
unexpected initial-file changes were found. Current serving source/extension and
C++ headers match the frozen off/on snapshots. Detailed hashes, original outcomes,
resource timelines and cleanup records remain under
`artifacts/attention-overhead-20261020/`; the compact tracked JSON includes raw
analysis, manifest and audit hashes. GPU/CPU component and state qualification
are independent of the earlier experimental native-QK route.

## Fresh repeat after the MTP diagnosis

The user requested another measurement to check the possible MTP effect.
`1fc751e` adds `--combined --fresh-reference`: prepare exactly one new all-off
cohort followed by one new all-on cohort, without rerunning individual switches.
The order, inputs and metrics were frozen before results in
`model-c8-repeat-r1/repeat-plan.json`. Both profiles exactly match the previous
profile except for the four declared booleans. MTP remains **k=4**; source,
extension, C++ headers, input, seed, temperature, capture density, graph policy,
full output length and capacities remain unchanged. Native QK reuse stays off.

The first preparation invocation imported an older installed serving schema
because it omitted `PYTHONPATH=src`. It failed **before any model launch**;
`model-c8-combined-r2/preparation-failure.json` retains that attempt. The corrected
preparation created `model-c8-repeat-r1`, the only new two-cohort GPU run.

### Qualification status: diagnostic repeat

The all-off original job and strict snapshot gate **pass**. The all-on original
job and strict snapshot gate **fail**, with controller return code 1. One
scheduled-prefill counter increment of 2 is observed between 402.741 and
402.994 seconds, shortly after the common SSE window begins at 402.624 seconds.
The original `prefill_increment_during_window` violation and client failure are
retained. The coordinator's `completed` status means both cohorts and independent
audits finished; it does **not** mean both original qualifications passed.

Independent complete execution audits pass for both cohorts: 3,380 / 3,285
contiguous model submissions overlapping their respective windows are pure C8,
with no prefill or Query/pressure refresh. The last actual prefill submission
ends 0.601 / 0.596 seconds before each common window. Snapshot coverage is
99.58% / 99.50%. Thus the recorded increment is consistent with delayed
publication, rather than an executed prefill step inside the window; this
interpretation does not override the original gate.

Both runs complete all eight full inputs and 16,384 accepted output tokens per
request, publish all eight terminal histories, release all owned history,
checkpoints and graph state, and perform zero historical SSD I/O. No owned
model/GPU process remains. **The repeat is diagnostic, not a fully qualified
performance acceptance.** No automatic replacement run or gate relaxation is
performed.

### Observed repeat costs

All throughput rates are totals across eight requests. The same event/wall
scopes and resource sampling limitations as the initial comparison apply.

| Metric | Fresh all off | Fresh all on (diagnostic) | Observed change |
| --- | ---: | ---: | ---: |
| Engine startup/JIT, s | 85.020 | 82.870 | -2.53% |
| Cold request preparation / last first-token, s | 409.172 | 402.175 | -1.71% |
| Effective input wall rate, token/s | 5,124.51 | 5,213.67 | +1.74% |
| Pure-prefill model event rate, token/s | 5,550.27 | 5,542.90 | -0.13% |
| Shared C8 accepted output rate, token/s | 699.26 | 785.80 | +12.38% |
| Full decode step event interval, ms | 42.787 | 42.622 | -0.39% |
| Model decode step event interval, ms | 41.469 | 41.250 | -0.53% |
| Whole ready-batch wall time, s | 627.637 | 608.902 | -2.99% |
| Whole ready-batch output rate, token/s | 208.834 | 215.260 | +3.08% |
| Observed controller lifetime, s | 719.753 | 698.680 | -2.93% |
| Observed output rate including startup, token/s | 182.107 | 187.600 | +3.02% |
| MTP mean accepted length, including bonus | 3.859 | 4.331 | +12.23% |
| Draft acceptance rate | 71.47% | 83.27% | +11.80 percentage points |
| Total owner verification calls | 35,316 | 31,702 | -10.23% |
| Accepted scoped output observations | 7,391 | 6,111 | -17.32% |
| Accepted prefill observations | 65,384 | 65,384 | unchanged |

The initial common decode drop (743.12 -> 669.63 token/s) is not reproduced in
these diagnostic measurements: the sign changes while MTP mean accepted length
also changes sign (initial 4.109 -> 3.706, repeat 3.859 -> 4.331). Full step
interval changes are below 1% in both comparisons. This supports treating MTP
and trajectory variation as material confounders. It does **not** establish that
the optimizations cannot affect those trajectories, nor justify reporting
12.38% as their net decode benefit. The measured 3.08% whole-batch increase has
the same qualification and causality limitations. No pooled causal estimate is
computed from the qualified initial pair and the diagnostic repeat.

Input hashes match, but Query/pressure selections and accepted-output hashes
differ. Query history-page Jaccard averages 0.9415 (range 0.9230–0.9541), with
matching mandatory contents. The larger MTP acceptance also reduces verification
and scoped-sampling work, so reduced total CPU costs are not all attributed to
faster implementations.

### Actual modes and memory

Both runs execute the same 112 graph selection calls. Graph scoring takes
4.974 -> 1.856 s (-62.68%); the surrounding selection block takes
11.405 -> 8.131 s (-28.71%). All-on builds 112 native indexes with zero cache
hits; index reservation is included in those times. Maximum index reservation
is 2,623,072 bytes per owner, as in the initial pair.

Actual all-on counters show 31,702 native decode commits (zero Python), 29,571
fused bypass scorer calls (zero reference), and 112 native propagation reports
(zero reference). The DLPack pool uses 34 allocations/events and 15,740 reuses,
retaining 2,580,480 pinned bytes with 1,441,792 peak active bytes. Exhaustions
and quarantined slots are zero; the 64 MiB / 128-slot limits hold. The existing
early-metadata fallback occurs once in each arm; none of the four optimized
paths falls back.

| Observed peak, GiB | Fresh all off | Fresh all on |
| --- | ---: | ---: |
| CUDA allocator allocated | 83.053 | 83.053 |
| CUDA allocator reserved | 86.576 | 86.576 |
| Owned process RSS, 5s sampling | 87.467 | 87.708 |
| Owned process PSS, 5s sampling | 87.084 | 87.327 |
| Device used, 1Hz sampling | 72.484 | 72.484 |

All four defaults remain off. The independent correctness/component results
remain unchanged; this repeat adds measurements and preserves its strict-gate
failure. The full analysis, original statuses, raw evidence hashes and source
provenance are in
[`attention-overhead-model-repeat-20261020.json`](benchmarks/attention-overhead-model-repeat-20261020.json).
The initial tracked result and all raw artifacts remain unchanged.
