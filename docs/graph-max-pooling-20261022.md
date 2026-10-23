# Raw-seed max pooling

Protocol correction on 2026-10-23: the original `max_guided` arm placed guidance
at the final Query boundary. The user requested generated Query scores for
prefill pressure, retaining the original final Query plus graph propagation.
That old arm was stopped after 182 cases and preserved as a separate experiment.
The corrected implementation and fresh test are described in
`guided-prefill-pressure-20261023.md`; old guided scores do not measure the
requested pressure policy. The completed max and Query-only controls are reused.

Set `kvmem.graph_scoring_version="v2"` and `kvmem.graph_aggregation="max"`
to pool incoming raw Query scores. The default remains `"sum"`.

For each target, the attention pool is the maximum original seed among its
eligible incoming sources with a positive stored edge. Edge magnitude selects
the relation but does not multiply the source score. The locality pool takes
the maximum adjacent source seed, with the existing cross-message attenuation.
An operator with no effective incoming relation uses the target's own seed.

For `multi`, with locality pool L, attention pool A and original seed s:

`result = (1-alpha)*s + (alpha/2)*L + (alpha/2)*A`

Single-graph modes allocate the complete alpha to their pool. The original
seed vector is immutable throughout the call: received scores are never
forwarded again. Masked pages neither donate nor receive and keep their exact
original score. Graph versions, owner binding, top-k construction, checkpoint
restoration, native selection budgets and cumulative-attention pressure remain
under their existing policies.

Max pooling is a different ranking policy, not a precision-preserving speed
optimization. It does not conserve total score and is not renormalized after
pooling. Reports explicitly record `aggregation="max"`,
`pooling_source="raw_seed"` and `score_sum_conserved=false`. Traces identify
the winning source in each pool. Offline replay respects the aggregation field;
historical sum-policy benchmark validators reject max configurations.

Python and C++ implementations passed the incoming-neighbor oracle, masking,
cross-message, partial-source, one-hop and native/picker comparisons. A native
selector integration fixture co-selects a low-score adjacent value block
without changing K. These are component results, not evidence of a RULER score
improvement. Validation: 920 CPU tests passed; evidence is in
`artifacts/graph-max-pooling-20261022-r1/validation-tests.xml`.

The ordered Pro6000 queue is
`artifacts/ruler-pooling-queue-20261022-r1/queue.json`: max, Query-only, then
max plus guided. All arms reuse the original 260 complete prompts, sampling
and seeds, NVFP4, FP8 KV, MTP4, K=36,864/G=16,384 and up to eight owners.
Physical prefill coalescing is disabled in all three arms because it rejects
guided generation. The max/guided pair differs only in service port and the
private cap (0 versus 512). Guidance uses the previously qualified static
explicit-Query boundary path, natural completion and explicit fallback;
ordinary prefill pressure continues using cumulative attention.

Runtime, native binary and every arm's configuration are frozen on the data
disk. The queue waits for completed native-baseline cleanup, then launches
each owned controller serially. A failed arm stops the queue and preserves its
artifacts; it never silently retries or substitutes earlier answers.

The first arm completed on 2026-10-23: 260/260 cases, macro score
85.64076923076922. All 260 Query calls used native max propagation, with no
identity fallback. The final execution audit passed and owned GPU resources
were released before Query-only started. Cold preparation took 73.155 seconds;
inference took 11,916.706 seconds, for about 3h20m including preparation.
Scores and audit are under `max/run/answers/scores.json` and
`max/run/execution-audit.json`; the monitor's verification receipt is
`artifacts/ruler-pooling-monitor-20261022-r1/verified/max.json`.

The historical sum+graph reference scored 85.75, so this run is lower by
0.10923 percentage points. Multikey-3 increased from 30 to 35; multikey-2 stayed
at 55, while multivalue, CWE and FWE decreased. This run provides no overall
accuracy improvement over that historical reference. The reference is an
earlier run. Its physical prefill coalescing option was enabled, while the queue
disables it for guided compatibility. A follow-up check found that the reference
reported only `logical_prefill_stats={"enabled":true}`, with no planned or
executed group counters and no `logical_prefill_publication.jsonl`. Physical
merging did not actually execute in that reference, so this configuration flag
difference should not be presented as an observed execution difference or as an
explanation for the score difference.

Query-only also completed on 2026-10-23: 260/260 cases, macro score
85.37153846153845. The final execution audit passed; all 260 per-case reports
confirm alpha=0, an unchanged raw ranking and no added/displaced blocks from
propagation. Its 260 identity reports are intentional for this arm, with zero
native propagation calls. Ordinary pressure remained cumulative attention.
GPU cleanup completed before max+guided started. Cold preparation took 87.208
seconds and inference took 11,898.585 seconds, again about 3h20m combined.
The verification receipt is
`artifacts/ruler-pooling-monitor-20261022-r1/verified/query_only.json`.

Max scored 0.26923 percentage points above Query-only in this cohort; Query-only
was 0.37846 points below the historical sum reference. These small observed
differences do not establish a reliable accuracy gain. Max+guided remains in
progress; its first 14 completed cases all published naturally terminated
private Query tokens, so guidance is executing rather than always falling back.
