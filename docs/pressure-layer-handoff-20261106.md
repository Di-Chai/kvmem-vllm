# Pressure pipeline: early scheduler handoff

This follows `pressure-layer-pipeline-20261106.md`. The previous 3/288 early
consumer count was primarily a handoff timing limitation. It did not establish
that same-request layer overlap was physically unavailable.

Evidence is preserved in
`artifacts/pressure-layer-handoff-20261106-r1/`. The new execution switch remains
disabled by default. No formal MAB experiment was resumed or externally graded.

## Why the old pipeline rarely started early

Native asynchronous KV commands are enqueued at post-forward. Previously, the
pressure preparation step ran at the next pre-forward pump. The completion
acknowledgement then waited for that entire peer model batch to return. The
scheduler could only resume the owner on a subsequent batch.

In the two preceding pipeline trials, 283/288 materializations were already
fully GPU-ready at dispatch. The mean interval from submitting all KV layers
to dispatch was 0.871 and 0.809 seconds. Dispatch to the owner's next model
submission added 0.128 and 0.139 seconds. These are host intervals, not additive
GPU idle durations (`original-handoff-timeline.json`).

## Candidate implementation

`KVMEM_LAYERWISE_PRESSURE_PIPELINE_HANDOFF=1` starts one eligible pressure
preparation through the original admission/selection path at post-forward. It
waits at most 64 ms for the producer's immutable read plan to become available;
this bound covers that wait, not all selection and metadata work in the hook.

The engine may acknowledge the host selection mapping before any layer has
finished uploading. Each layer's native KV writer and attention reader still
waits on its own recorded CUDA event. An unrecorded event is never considered
ready. The original producer, source ownership, page views, scratch and
maintenance credit remain held until the final materialization fence. Whole
window checkpoints still require full completion. Producer failures propagate.

The candidate requires the existing pipeline, post-forward completion polling
and submit-on-prepare paths. It is qualified here only with static allocation.
Dynamic allocation compatibility is not established by this experiment.

## Unprofiled eight-owner measurement

Both variants use the existing layer pipeline; only the early-handoff switch
differs. Both process the same 2,490,368 complete cold-history tokens, with
GPU-confirmed completion and no final Query/decode in the timed interval.
Model loading and the separate diagnostic trace are excluded.

Controls are unchanged: Pro6000, NVFP4, Active 36864, Gen 16384, block 32,
sink 41, recent 0, original sampling/output limits, layerwise accumulated
attention pressure, sequential Mean-K Query, eight clients, compiled execution,
native batch cap 16424, 1K logical/8K physical chunks, original static GPU pool
and CPU cache. Frozen input/profile checks and unchanged autotuning passed.

| Metric | Old handoff | Early handoff |
| --- | ---: | ---: |
| Complete historical prefill time | 432.480 s | 433.684 s |
| Actual historical prefill throughput | 5758.35 tokens/s | 5742.36 tokens/s |
| Consumer submitted before final KV layer submission | 2/144 | 98/144 |
| All layers GPU-ready at dispatch | 142/144 | 1/144 |
| Mean immutable-plan-ready to dispatch | 1.674 s | 0.060 s |
| Mean suspended to resumable | 2.674 s | 0.895 s |
| Mean host assembly-start to all-layer submission | 0.758 s | 0.786 s |
| Mean sampled GPU utilization | 90.34% | 92.92% |
| GPU samples below 20% utilization | 7.18% | 4.62% |

Throughput changed by -0.28% in this single pair: there is no demonstrated
speedup. The utilization sample change is encouraging but is neither a
measurement of idle duration nor a replicated improvement.

Early release is not free. The post-forward hook took 23.107 aggregate host
seconds across 144 attempts, including the original prepare/commit work.
Per-layer host event waits totalled 9.617 seconds, versus 0.119 seconds in the
control. These durations can overlap GPU work and cannot be added as a wall
time decomposition. Model event intervals increased while whole-phase time
remained similar; the intervals themselves include submission gaps and waits,
not just kernels.

61/144 attempts acknowledged directly at post-forward. In 83 attempts the plan
missed the bounded preparation wait. Of the 46 materializations with no early
consumer, 45 missed that direct handoff; their mean dispatch-to-model interval
was 1.878 seconds. The 98 early consumers averaged 0.046 seconds. Some
pre-forward dispatches still miss the current feedback boundary and wait for
another model batch. Simply extending a blocking timeout could trade scheduler
delay for a foreground stall and has not been adopted.

## Actual CUDA overlap and remaining bottlenecks

The separate eight-owner diagnostic completed all original inputs and retained
all first answers. Its finite trace contains 20.267 seconds of observed GPU
activity span. `cuda-overlap.json` correlates actual CUPTI H2D copies and
attention-path kernels through owner/layer NVTX launch ranges. It requires the
same owner and device, different streams, a later copy layer than compute
layer, and intersecting GPU timestamps; host-range overlap is not counted.

There are 680 correlated pressure H2D copies and 51 correlated consumer
kernels. 54 copy/kernel pairs overlap across three owners, with a 17.476 ms
union of overlap intervals. For example, layer 9 KV H2D on stream 49 overlaps
layer 0's native FlashInfer attention on stream 7 for the same owner. Later
layer rebuild kernels also overlap earlier attention (52 pairs, 3.102 ms
union). This demonstrates real per-layer overlap. These are attention-path
subsets, not total model overlap or recovered throughput time; a batched
attention kernel also includes other owners.

The same trace still has 2.005 seconds with no traced GPU activity. Main-thread
OSRT waits with `take_gil` backtraces total 2.311 seconds, overlapping 0.911
seconds of those GPU gaps. This establishes continuing host submission
contention, but does not identify a unique GIL owner or prove it explains the
whole slowdown. One 108.9 ms gap ends at an assembly upload and overlaps
106.8 ms of main-thread GIL wait. Another 105.1 ms gap includes about 90.8 ms
of generation-2 Python GC inside selection-commit bookkeeping.

Some producer preparations take 74–187 ms in this instrumented window, enough
to miss the bounded post-forward plan wait. Background submission and metadata
work therefore remain relevant after early release. The next useful work is
to reduce preparation/commit cost and make ready notifications timely without
blocking the foreground for longer, rather than assuming an earlier ACK alone
will increase throughput. Existing peer requests already hide some assembly
time. Neither this observation nor the diagnostic intervals isolate each
component's net cost in the unprofiled run.

The generic diagnostic analyzer's legacy `pressure_events` field is empty
because it expects whole-window completion markers. Use the pipeline markers
in `pair-result.json` and the explicit CUDA correlation above; an empty legacy
field is not evidence of no pressure work. Diagnostic throughput is excluded
from the performance comparison.

## Correctness and disposition

Three complete original serial requests passed exact pressure/Query selected
block equality and full output-token/text/answer equality, including reused
prefix restoration (`serial-comparison.json`). The related CPU suite passed
96 checks.

The eight-owner first trajectories are preserved. They are not exact:
pressure/Query equality is 0/8 and full output equality is 1/8. The prior
baseline-versus-baseline already exhibited these same counts, so serial
equivalence must not be presented as an eight-owner exact-replay guarantee.
No answers were regenerated to obtain agreement.

The diagnostic has the same eight-owner replay limitation; its first answers
were retained without resampling. The frozen control audit passed. The
candidate remains default-off: real overlap and earlier dispatch have been
demonstrated, but throughput improvement has not. The single unprofiled pair
does not establish a repeatable GPU-utilization improvement either.

`completion-audit.json` records frozen-file verification, completed counts and
GPU release. The original formal MAB queue remains paused.
