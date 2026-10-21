# Static FA2: one warm pressure boundary

Evidence is in `artifacts/prefill-static-pressure-nsight-20261022-r1/`:
`trace.sqlite`, `timeline-summary.json`, `pressure-attribution.json`, and
`analysis-proof.json`. The latter preserves the analyzer and dependency hashes,
input/source hashes, successful verification of all 399 frozen runtime files,
and empty owned-process/GPU cleanup. No serving code or default changed.

The service used the completed static candidate from
`prefill-static-aligned-20261022-r1/candidate`. It first completed the same
262,045-token input without tracing, then traced one pressure boundary of a
second full request. Only the server was injected by Nsight. The actually
loaded FlashInfer cache key `c37a5d36...` contained the same 136 configurations
and identical bytes as the source candidate; no cache was manually changed.

The verified logical range is **[30,720, 63,456)**: 14,336 tokens, 2,048 tokens,
1,984 tokens, pressure selection at history 49,088, 32 replay tokens, then
14,336 tokens. This is 32,736 tokens and exactly one complete pressure lifecycle.
The diagnostic helper's former lower bound of 32,768 would omit the last large
block, so only that diagnostic restriction was relaxed. Fourteen helper CPU
tests passed. The controller exited successfully and released its shared lock.

This is an instrumented diagnostic, not another throughput measurement.
The capture gate's 13.6-second elapsed value includes profiler start overhead.
The short lifecycle below is measured from actual NVTX markers and CUDA
activity, not that gate duration.

## Where the GPU waits

| Nonoverlapping lifecycle interval | Wall time | GPU activity union | GPU idle |
| --- | ---: | ---: | ---: |
| Suspended → assembly queued | 71.074 ms | 18.798 ms | 52.276 ms |
| Queued → assembly worker started | 1.097 ms | 0 | 1.097 ms |
| Worker started → selection prepared | 84.312 ms | 0 | 84.312 ms |
| Selection prepared → assembly submitted | 101.290 ms | 15.316 ms | 85.974 ms |
| Submitted → ready marker begins | 10.779 ms | 0 | 10.779 ms |

The full lifecycle through the end of the ready marker is **268.688 ms**:
34.114 ms has CUDA activity and **234.574 ms is GPU idle**. The small difference
from summing the table is the ready marker itself. The preceding forward's GPU
tail finishes at +18.861 ms; its activity accounts for nearly all GPU work in
the first interval. Reconstruction contributes 12.330 ms H2D, 1.911 ms split
rebuild kernels, and 1.069 ms resident-V gather. These are overlapping activity
measures and must not be added to CPU ranges.

The 10.779 ms after submission is entirely GPU idle in this trace. It is not
an unfinished H2D transfer: the final reconstruction activity ends at +256.235
ms, before `assembly_submitted` at +257.772 ms. The main `refresh.step` range
includes about 7.5 ms of actual selection commit, reporting, and release work.
Simply moving the GPU completion wait later cannot hide this measured tail.

## Publication and archive work

Times below are relative to suspension. They overlap across threads; in
particular, `publish_sealed` waits for the same pack/hash operations listed here.

| Existing operation | Begin → end | Duration | Intersection with GPU idle |
| --- | ---: | ---: | ---: |
| Action acceptance preparation | 19.275 → 26.612 ms | 7.337 ms | 7.337 ms |
| Native history scatter | 30.060 → 40.142 ms | 10.081 ms | 10.081 ms |
| Native mean finalization | 41.145 → 64.108 ms | 22.964 ms | 22.964 ms |
| Native record allocation/packing | 67.474 → 128.775 ms | 61.302 ms | 61.296 ms |
| Record digest stage, configured eight workers | 128.817 → 144.957 ms | 16.139 ms | 16.139 ms |
| `publish_sealed` read barrier | 72.556 → 146.659 ms | 74.104 ms | 74.104 ms |
| Pressure selection | 147.204 → 156.453 ms | 9.250 ms | 9.250 ms |
| Eight packed history reads | separate ranges | 53.832 ms | 53.832 ms |

This is the final **1,984-token** packet, not an unprocessed 14K packet. Its
geometry implies 62 complete 32-token records, each 1,114,112 bytes
(17 layers × 32 tokens × 4 KV heads × 256 channels × raw-K/V bytes).
The packet therefore contains **65.875 MiB**, of which raw K is 32.938 MiB.
These payload counts are derived from the verified geometry and codec, not
argument-size tracing inside each native call.

The work already runs on background publication/archive threads. For this tail,
however, it begins after the preceding forward's GPU work finishes; no later
public model work can start until the required accepted prefix and immutable
archive have been published. Asynchrony alone does not overlap these particular
22.964 ms and 61.302 ms ranges with the model.

The current direct seal performs one record copy from mutable staging into
independently allocated immutable Python bytes, followed by hashes and atomic
publication. `pack_record_pairs` validates/allocates with the GIL and releases
it around the memcpy loop. Its 61.302 ms range includes both phases and page
faults; the trace does not isolate memcpy from allocation/first-touch cost.
Removing this copy requires transferring ownership of immutable storage while
preserving admission, staging reuse, record generations, failure cleanup, and
eventual reclamation; it cannot safely retain recyclable DMA/staging views.

The previously tested pack-eight-worker candidate is not an established fix:
for representative 64-record batches, full seal/read/verify/retire time was
84.342 → 84.643 ms while retirement increased 0.949 → 7.575 ms. See
`docs/cpu-seal-pack-component-20261022.md`. Any new allocation/copy candidate
must include retirement and memory accounting, not only the shorter seal range.

Assembly reads **522,518,528 bytes (498.313 MiB)** of raw K for 30,016 rebuilt
tokens in eight chunks; 2,752 tokens are skipped and V is reused on the GPU.
The archive reports 64 gather tasks and 41.914 ms cache-copy time nested in the
53.832 ms read ranges. Every read begins after the preceding chunk's GPU work
has finished; measured read/GPU overlap is zero. The 136 per-layer launch
wrappers total 20.918 ms, but only **9.888 ms** intersects GPU idle. The existing
prefetch/fused candidates' prior measurements still apply; this single trace
does not establish a new gain for either.

## Polling and the GIL

Readiness was checked 43 times before assembly admission, taking 0.200 ms.
Across the lifecycle there were 120 readiness calls totaling 0.952 ms. All
main-thread diagnostic scopes, after removing nested scopes, total 15.146 ms;
9.166 ms belongs to `refresh.step`, including real commit/report work.

The main thread already spent 127.109 ms in recorded `clock_nanosleep` calls,
with 107.041 ms overlapping GPU idle. Its recorded blocking-call union was
164.663 ms. These host elapsed ranges are not measurements of running CPU time.
There is no evidence here of a continuously spinning main thread.

Only OSRT backtraces explicitly containing `take_gil` were classified as GIL
waits. Main/publication/assembly threads respectively show 29.463/2.043/14.802
ms; their intersections with GPU idle are 29.081/2.043/10.115 ms. Assembly's
longest recorded individual GIL wait is 0.444 ms. Its GIL waits overlap all
main-thread exclusive diagnostic scopes by only about 1.30 ms, including
0.132 ms in readiness checks. Even those intersections do not establish which
thread held the GIL. Unrecorded waits remain unknown. This evidence does not
justify a new cooperative-sleep policy as the main optimization.

## Relation to the completed static model experiment

Using the original experiment's existing reference, mean long-request model
CUDA time was 38.470 → 38.092 seconds, while TTFT was 43.087 → 43.032 seconds.
The roughly 0.322-second difference between those changes is an accounting
residual, not a measured pressure regression. Pressure wall totals changed
2.997 → 3.021 seconds and Query wall totals 0.916 → 1.037 seconds; these nested
intervals must not be added to model CUDA time. The reference was previously
measured, and this is not a repeated, interleaved attribution experiment.

The diagnostic establishes substantial necessary CPU publication/sealing and
history-read work on the pressure critical path. It does not establish why a
particular small end-to-end A/B difference occurred, nor a guaranteed benefit
from optimizing one nested range.
