# Isolated map preparation and native completion progress

Evidence root: `artifacts/pressure-completion-20261106-r1/`.

This follows `pressure-preforward-20261106.md`. That experiment changed map
preparation and handoff timing together. Its pooled pure-prefill improvement
was only0.54%, and removing all bounded preparation waits substantially
reduced same-request per-layer overlap. This follow-up separates the changes.

## Frozen controls

Both stages retain the original complete inputs, Pro6000/NVFP4, Active36864,
Gen16384, block32, sink41, recent0, compiled execution and eight clients.
Prefill pressure uses each layer's accepted cumulative attention; final Query
uses causal sequential Mean-K. Model sampling, thinking, answer limits,
archive source, candidate sets, ranking and mandatory rules are unchanged.
The original64 ms bounded post-forward plan handoff remains enabled.
`PIPELINE_PREFORWARD=0` throughout this experiment.

The map-only stage varies `PREPARED_COMMIT` alone. The notification stage keeps
prepared maps enabled and varies the new default-off `PIPELINE_PROGRESS`.
Both eight-owner tests use ABBA order and the same2,490,368 historical tokens.
The confirmed measurement gate excludes model load, final Query and decode.
Every first answer is preserved, including divergent concurrent trajectories.

## Implementation

The map preparation is the previous immutable owner/request/history-bound
packet, constructed on the existing pressure producer. No new implementation
or scoring backend is substituted for its isolated test.

`PIPELINE_PROGRESS` polls at two original native engine boundaries:

- after native `pre_forward`, before this batch's model submission;
- after native `finish_forward`, before ordinary sampling/feedback finishes.

These boundaries first use the existing `reap` checks: producer future done,
final CUDA event complete, exact plan/lease identity, assembler close and
maintenance-credit release. They then permit one original preparation quantum
with zero foreground plan wait. They never consume new native load receipts,
synchronize CUDA or run a private model forward. Mixed Query/control,
cancellation, eviction, retirement and current scheduled owners are excluded.

The ordinary post-forward hook retains its bounded wait. One already ready
ACK may additionally be followed by the next waiting job's original admission
attempt in the same feedback turn. The total preparation-wait budget is shared
across that call; it is not renewed for the second job. One scratch/transfer
lane remains, and an unfinished copy never loses its lease.

The notification and map-only runtimes are frozen separately. Workspace edits
do not affect an active or already frozen arm. CPU regression evidence currently
records170 passing checks covering native-boundary ordering, original receipt
ownership, partial KV, cancellation, completion fences and full wait budgets.

## Qualification and status

Map-only serial qualification passed all three original requests: pressure
and Query blocks, complete output token IDs, text and answers are exact.
It includes two cold histories and one restored prefix. Map-only ABBA completed:

| Pair | Baseline | Prepared maps alone | Change |
| --- | ---: | ---: | ---: |
| 1 | 5735.43 tokens/s | 5743.05 tokens/s | +0.13% |
| 2, reverse order | 5725.70 tokens/s | 5728.68 tokens/s | +0.05% |
| Pooled tokens / pooled time | 5730.56 tokens/s | 5735.86 tokens/s | +0.09% |

The standalone map change does not establish useful throughput improvement.
The below20% GPU sample share changed5.18% to4.84%, while sampled mean changed
92.02% to91.88%; these small changes do not demonstrate resolved idle valleys.
The notification stage passed its own three complete-input requests: pressure
and Query selection, all output token IDs, text and answers match the map-only
reference exactly. Seven of41 materializations were reaped at the additional
`target_finished` boundary; the other34 used original boundaries. The candidate
path was therefore exercised. Its independent two-pair ABBA runs completed.

The original serial summary rejected repeated `pipeline_handoff_prime` polls as
duplicate state-transition timestamps after all inference had completed. The
failed original summary/status/log remains unchanged. A read-only analyzer now
ignores only repeated readiness polls and still rejects duplicated real state
transitions;44 analyzer checks passed. `serial-analysis-recovery.json` and
`recovered-serial-measurement.json` preserve the recovered proof. No response,
frozen runtime or original event log was rewritten, and no inference was rerun.

The unlaunched notification-v1 plan is retained. A separate v2 adds an explicit
active-lease check even when no maintenance policy exists; two idle waiting
controllers were replaced before any notification GPU request. The map-only
measurement was not restarted.

The first restricted map-only launch stopped before GPU ownership checking
could complete, before any model or request. Its log is retained. The authorized
GPU execution uses a fresh serial-v2 root with the same controls.

## Authorized complete MAB follow-up

After efficiency validation, use the qualified measured execution to start
the original complete paper cohort:2071 questions,130 contexts, with the
previous two-method protocol and unchanged input/sampling/profile bindings.
Keep the paused previous queue and all its answers intact. Run the optimized
layerwise arm first. No external judge or recurring monitor is started.

The previous complete-MAB qualification exposed a Mean-K ingress guard of32768
blocks, insufficient for its >1M context. A separate qualification extends
only the host guard to65536 blocks, preserving CUDA kernel arithmetic, and
checks exact original-domain scores plus large-index softmax normalization.
That extension is excluded from every efficiency runtime. Full original
long-context/prefix-restore qualification remains required before formal MAB.

## Notification comparison and conclusion

| Pair | Prepared-map baseline | Native completion progress | Change |
| --- | ---: | ---: | ---: |
| 1 | 5741.25 tokens/s | 5749.08 tokens/s | +0.14% |
| 2, reverse order | 5721.88 tokens/s | 5738.91 tokens/s | +0.30% |
| Pooled tokens / pooled time | 5731.55 tokens/s | 5743.99 tokens/s | +0.22% |

The notification improvement is small and does not solve the prefill bottleneck.
Below20% GPU sample share changed6.10% to6.34%; mean sampled utilization changed
90.79% to91.93%. There is no stable reduction in utilization valleys. These are
sample shares over the pure-prefill interval, not measured GPU idle durations.

The two candidates returned29 and78 completions at the additional native
boundaries (pre-forward or target-finished), out of144 materializations per run.
Another4 and5 were reaped within the same post-forward feedback call. The new
path executes, but more timely completion checks are not equivalent to a
corresponding throughput gain.

Use the unchanged command-receipt anchor. Across the two baseline runs,
command-to-producer-start averaged858.20 ms; candidates averaged687.65 ms.
Command-to-host-completion averaged2441.33 versus2037.51 ms. These are
overlapping per-owner host intervals including GPU work; they must not be
summed or treated as GPU idle time. Queue-only timing is misleading: deferring
preparation moves `assembly_queued` later. For pair1, the queue-only interval
appears to drop287.56 to17.66 ms, but the true command-to-start interval drops
only759.59 to697.51 ms. Part of the apparent queue improvement is accounting.

Three complete serial requests passed exact blocks and full output for each
isolated change. Exact eight-owner replay remains unestablished: the original
baseline itself varies with concurrent execution. All original first outputs
are retained; no repeated sampling was used to obtain matching results.

Both new switches remain off by default and are off in the complete MAB run.
The selected runtime is the preceding qualified early-handoff configuration.
The selection decision is preserved in the new MAB `optimization-decision.json`.
The CPU work reduction and observed host-wait reduction are real, but these
measurements do not establish a material prefill improvement.

## Complete MAB launch and long-index qualification

The new queue is `artifacts/mab-full-optimized-20261107-r1/`:2071 questions
per method,130 contexts, independent first answers. Qualification runs
layerwise then pure Mean-K; formal inference runs layerwise then pure Mean-K.
All original complete-MAB profiles and input/sampling bindings are preserved.
The previous paused queue and its results remain intact.

The host input guard is now65536 candidate blocks. Four actual Mean-K CUDA
instruction bodies are byte-identical between old and new compiled modules.
Single-Query and tiled original-domain outputs are bitwise equal. All tested
larger indices (32769,53248,65536) match independent full-domain softmax within
the declared numeric tolerance. Twenty fixed numeric cases are retained.

An initial synthetic multi-Query legacy comparison did not match bitwise:
the old kernel's own repeated results differ by3.49e-10, the same as the
old/new difference. Its existing floating atomic accumulation order explains
this. Failed diagnostic logs and all later fixed-repeat tensors are retained;
no model answer was generated or retried. A second diagnostic stopped on an
overly restrictive path-label assertion; its completed data are retained too.
The completed v3 proof compares device instructions and keeps exact equality
where the original kernel is deterministic. It does not relax the complete
input block/output qualification or change kernel arithmetic.

Evidence: `maps-result.json`, `progress-result.json`,
`unchanged-command-anchor-initial.json`, `serial-analysis-recovery.json`,
`long-index-gpu-qualification-v3.json`, and `completion-audit.json` under the
evidence root. MAB includes its own original longest-input and restored-prefix
qualification before formal inference. No external judge or periodic monitor
was started.

Both methods subsequently passed all7 original qualification requests. The
1,665,024-token Recsys prefix and its next question completed for each method;
both cold/warm pairs passed exact KV and recurrent-state byte checks. These
qualification answers are excluded from formal scores. The queue has entered
`formal_layerwise`; `formal_mean_k` follows automatically. Final official
task metrics and assistant semantic judgments remain pending full inference.

The original pure Mean-K control uses recency pressure with attention
accumulation disabled. The layerwise method uses per-layer cumulative-attention
pressure and causal per-layer Mean-K Query. This compares the complete schemes,
including their pressure and Query differences. The original protocol is
preserved; it is not interchangeable with a shared cumulative-pressure baseline.

The final launch audit confirms actual formal GPU work and completed first
answers, correct native binary/import/execution switches, unchanged budgets and
original inputs. See `mab-formal-launch-audit.json`; the queue continues running.
