# Pressure preparation before the native peer forward

This follows `pressure-layer-handoff-20261106.md`. The previous early ACK
opened genuine per-layer CUDA overlap, but did not improve complete eight-owner
prefill throughput. Its trace identified foreground map construction, a bounded
plan wait, and delayed batch feedback as remaining work.

The new candidates stay disabled by default:

- `KVMEM_LAYERWISE_PRESSURE_PREPARED_COMMIT=1`: construct immutable per-layer
  maps on the existing pressure producer. Read a selected native handle's
  generation once per unique record, preserve the original selected order and
  dense offsets, and bind the map to the exact authority/request/prefix/selection.
  Main-thread commit validates native layer zero and installs the prepared map.
  It still requires completed assembly or the guarded pipeline transaction.
- `KVMEM_LAYERWISE_PRESSURE_PIPELINE_PREFORWARD=1`: enqueue pure asynchronous
  prefill-pressure commands through the native connector before the peer model
  forward. Reuse the original prepare, publication, page, lane and byte checks.
  The producer may prepare/copy while the peer batch computes. Poll completion
  through the original feedback channel without the prior 64 ms foreground
  plan wait. No unrecorded layer event is considered ready.

The native pending-load receipt is consumed once, avoiding duplicate enqueue
at post-forward. Query, decode, guided pressure, mixed commands, cancellation,
retirement and session eviction retain their original control ordering. An
ordinary pressure replay reservation is accepted: its presence does not change
the original cumulative-attention selection into a private Query forward.
If the owner's publication is not ready yet, the original pump must still
advance it. An early enqueue marker alone does not prove early preparation.

The new path preserves the original producer/source/page/scratch leases and
per-layer KV writer/reader guards through the final completion fence. It does
not add asynchronous scheduler mutations or allow whole-window checkpoints to
consume unfinished KV.

## Controls and evidence

Evidence root: `artifacts/pressure-preforward-20261106-r1/`.

The comparison baseline is the preceding early-handoff implementation:
`PIPELINE=1` and `PIPELINE_HANDOFF=1` in every arm. Only the two new switches
differ. Controls remain Pro6000, NVFP4, Active 36864, Gen 16384, block32, sink41,
recent0, the original layerwise accumulated-attention pressure and sequential
Mean-K Query, original sampling/output caps, compiled execution, eight clients,
1K logical/8K physical chunks, native batch cap16424, static GPU and CPU budgets.

The corrected v2 implementation passed 122 focused CPU checks. Sixteen optional
native batch-picker checks were skipped; that independent picker remains off.
Three complete original serial requests passed exact pressure and Query block
equality and complete output-token/text/answer equality. They include two cold
prefixes and one reused-prefix request. All41 pressure commands exercised the
new native pre-forward enqueue path (`serial-v2-comparison.json`).

The first implementation conservatively excluded pressure commands with replay
tokens and therefore never exercised early enqueue. Its three first answers
and exact comparison remain preserved. Its unlaunched eight-way plan is not a
performance result. The corrected version uses a separate immutable runtime;
no running/frozen directory or first answer was overwritten.

Two unprofiled eight-owner comparisons run in ABBA order under `eight-v2/`.
Each arm executes the same2,490,368 complete historical tokens with a
GPU-confirmed end fence and a common Query/decode gate. Model load, final Query
and decode are excluded from the primary prefill interval. Concurrent first
trajectories are retained even when they differ; serial exact equality must not
be presented as an eight-owner exact-replay guarantee.

The `suspended` event moves from post-forward to pre-forward under the new
candidate. Its raw elapsed interval is therefore not a comparable end-to-end
request latency metric. Completed historical prefill wall time and the actual
GPU work counter remain comparable. Utilization samples describe sample
shares, not measured GPU idle durations.

## Two unprofiled eight-owner comparisons

Both comparisons completed in the declared ABBA order. Each arm processed
2,490,368 GPU-confirmed historical tokens before any final Query or decode.

| Pair | Earlier handoff baseline | Prepared maps + pre-forward enqueue | Change |
| --- | ---: | ---: | ---: |
| 1 | 5749.13 tokens/s | 5783.60 tokens/s | +0.60% |
| 2 (reverse order) | 5757.62 tokens/s | 5784.89 tokens/s | +0.47% |
| Pooled tokens / pooled seconds | 5753.37 tokens/s | 5784.24 tokens/s | +0.54% |

The pooled below20% GPU utilization sample share changed from6.70% to5.58%;
mean sampled utilization changed from91.13% to91.51%. These are small changes,
not evidence that the utilization valleys are solved. Two paired runs do not
establish a broad performance guarantee.

Time inside host handoff attempts fell from23.98/23.16 seconds in the controls
to9.50/10.46 seconds in the candidates. This is not equivalent to saved prefill
time: the full phase improved by only about2.31 seconds per arm on average.
The CPU intervals can overlap active GPU work.

An important tradeoff remains: the first consumer preceded the last layer's
submission in104/144 and101/144 control events, but only0/144 and2/144 candidate
events. Actual pre-forward producer attempts numbered18 and8. Most early
native receipts therefore did not start preparation at that boundary. Removing
the bounded host plan wait also changes which native feedback turn carries
the ACK; it must not be advertised as improved same-request layer pipelining.

The separate finite notification trace completed. Its timing is excluded from
the throughput table. Both candidate switches stay off by default. The formal
MAB experiment remains paused; no grading or periodic monitor was started.

Eight-owner first trajectories remain non-reproducible even within the control:
baseline1 versus baseline2 has0/8 identical pressure selections,0/8 identical
Query selections and1/8 identical complete outputs. Baseline1 versus the two
candidate runs has0/8 identical pressure/Query selections and1/8 or2/8 identical
complete outputs. All first outputs are retained. Thus the three-request serial
qualification passed, but exact eight-owner output equivalence is not established;
these runs are not an accuracy comparison and are not grounds for default rollout.

## Remaining gates and CPU cost

The new Nsight GPU/host interval covers20.57 seconds and eight pressure jobs.
The earlier handoff trace covers20.27 seconds and eight pressure jobs. In those
short windows, main-thread `committed` inclusive time fell from39.34 to9.27 ms
per job; exclusive time fell from24.64 to8.96 ms. Background `prepare` averaged
84.65 versus75.04 ms. These windows have different scheduling and GC events,
so this is localization evidence, not a separate throughput result.

Main-thread confirmed GIL waits remain2.31 versus2.32 seconds. Their overlap
with GPU idle decreased from0.911 to0.672 seconds in these windows. Moving work
to the existing producer reduces foreground work but does not eliminate Python
contention. The current trace still contains3.46 seconds inside native record
gather calls and1.70 seconds inside background native commit calls; these
overlap GPU work and must not be added as prefill overhead.

The actual gate observations identify three distinct scheduling cases:

- Commands28,33,34 encountered pending CPU publication. All three snapshots
  show scheduler confirmation, completed DMA and ready score validation, while
  the publication worker is still consuming the owner. It is unsafe to start
  selection on that incomplete accepted history.
- Commands29,31,32,35 entered `assembly_wait` before the peer forward, but
  waited1.57,1.46,1.46 and1.86 seconds respectively for the previous pipeline's
  scratch/maintenance lease. Their producer starts immediately after that
  predecessor's completion is reaped. Early receipt alone cannot free the lane.
- Command30's pre-forward prime services the preceding command29's prepared
  plan first, then exits under the existing one-job-per-turn rule. Its own
  producer starts at post-forward. This is a feedback-turn limitation.

The new trace correlates680 pressure H2D transfers, but no attention kernels
with an outstanding same-owner guarded lease. Consequently it provides no
positive evidence of same-owner later-layer-copy/earlier-layer-attention
overlap. Cross-owner compute/copy overlap may still occur. This agrees with
only2/288 early consumers in the two unprofiled candidates, versus205/288 in
the two controls. Removing every bounded plan wait was not an improvement to
that pipeline, despite the small aggregate throughput change.

Next work should separate the prepared-map optimization from the zero-wait
handoff policy, preserve a timely native feedback opportunity, and inspect
safe completion-driven release of the existing scratch lease. Accepted-history
publication and final GPU fences must remain prerequisites; adding generic CPU
threads or bypassing those fences does not address the observed queue.

Evidence: `final-result.json`, `notification-analysis.json`,
`queue-handoff-analysis.json`, `cuda-overlap.json`, and `completion-audit.json`
under the evidence root. Frozen serial, eight-way and diagnostic verification
passed after execution; the GPU was released. No first answer was resampled.
