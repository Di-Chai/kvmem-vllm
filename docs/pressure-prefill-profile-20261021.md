# 256K prefill pressure profiling and generation-budget comparison

These are fixed-input performance diagnostics, not the complete RULER quality
evaluation. The cohort is the first eight task-interleaved inputs from the frozen
256K RULER fixture. Every input is complete; all four repaired optimizations are
enabled. NVFP4, FP8 KV, compiled MTP4, K=36,864, batch=16,424, chunk=1,024,
CPU history=100,000,000,000 bytes, no SSD, guided off, cumulative-attention
pressure selection and V2 multi-graph Query selection with alpha=.2 are retained.
The output cap is 8,192 and thinking cap 4,096, independently of the KV growth G.

## Attribution

The instrumented diagnostic is `artifacts/pressure-profile-20261021-r1/`.
It completed all eight inputs and owner-release checks. It is excluded from
throughput comparisons because profiler export produced a substantial stall.

- 104 pressure refreshes and eight Query refreshes were observed at G16K.
- Actual selection computation averaged about 20.2 ms per pressure refresh.
- The trace covers 16 real forwards and 63,872 public tokens. It also contains
  16 zero-token maintenance annotations; these must not be counted as forwards.
- Summed CUDA kernel time was 11.027 s, of which native FlashInfer paged prefill
  accounted for 6.630 s and fused page-score segment statistics for .328 s.
- Observed GPU kernel/copy/memset intervals occupied a union of 11.094 s within
  an 11.730 s span. This does not prove what all unobserved gaps represent.
- Small pageable metadata uploads synchronously waited for preceding GPU work.
  `cudaStreamSynchronize` CPU duration is not an independent GPU idle cost.
- The active window grew toward K+G, about 52K at G16K. A 36K selection budget
  does not mean every prefill forward attends to exactly 36K positions.

Refresh suspension includes queueing, assembly and readiness observation while
other requests can run. Sum of owner waits is not elapsed batch delay. The
analysis script joins each owner/command lifecycle and keeps these quantities
separate. Archive CPU timers likewise overlap with GPU work and other workers.

## Asynchronous metadata screening

Artifacts: `artifacts/pressure-comparison-20261021-r1/metadata-sync/` and
`metadata-async/`; only `async_action_metadata` differs between the model profiles.
Model and harness files are identical within this pair. Both runs completed
eight cases with no scoring/acceptance/download-pool fallback or owner leak.

| Measure | Synchronous | Asynchronous |
| --- | ---: | ---: |
| Public prefill before first decode, wall token/s | 5,327 | 5,646 |
| Pure-prefill model interval token/s | 5,912 | 5,961 |
| Mean pressure suspended-to-ready, s | 1.814 | 1.771 |
| Pressure refresh count | 104 | 104 |
| Accepted output tokens | 5,984 | 10,354 |
| Complete client wall time, s | 434.32 | 420.57 |

The first row includes refresh/archive/submission gaps, excludes model loading,
and stops before decode begins. The model interval includes scoring/capture and
submission gaps; it is not pure GPU multiplication time. One pair is insufficient
to establish a stable speedup. Different output lengths must not be attributed
to this optimization. Both had an actual peak prefill batch of seven despite
eight admitted clients.

Fixed-operand tests passed: 17 CPU and six real CUDA cases, including reference
and fused scoring, FP8/BF16 queries, GQA, partial pages, multiple owners and
accepted-prefix filtering. The actual metadata pool used 4 MiB pinned plus
4 MiB GPU memory with no fallback or failure. However, all 112 paired real-model
selected sets differed (mean Jaccard .903), and outputs also differed. This does
not establish end-to-end selection equivalence or prove the cause of drift.
Asynchronous metadata remains disabled in the baseline and G64K diagnostic.

## G16K to G64K diagnostic

Requested by the user to observe the tradeoff between fewer refreshes and a
larger attention window. Baseline is the synchronous arm above. The G64K run is
`artifacts/pressure-gen64-20261021-r2/`. An earlier attempt (`r1`) was rejected
by an overlooked client G16K validator before any input was submitted; its
logs are preserved. The diagnostic controller and client now both explicitly
validate the requested G64K setting without relaxing the default quality run.
All model source files remain identical to the G16K baseline.

G64K changes the maximum active window from 53,248 to 102,400 positions and
changes refresh boundaries and attention visibility. It is a configuration
tradeoff, not a guarantee of unchanged scores or answers.

The existing client reserves history credit for input + output cap + G. Its
100 GB CPU-history configuration retains 16 GiB headroom, leaving
82,820,130,816 bytes of payload admission credit. The eight G64K cases require
93,476,225,024 bytes in total; the first seven require 81,786,961,920. Therefore
the unchanged queue admits seven and delays the eighth. The comparison must
report this concurrency difference rather than presenting it as eight active
requests throughout. No CPU budget or headroom was silently increased.

`scripts/compare_gen_budget_runs.py` reports all prefill waves through completion
of the last prompt, including any overlapping earlier decode. This differs from
the first-wave metric in the metadata table. It also checks all input/sampling
and model-source hashes, records pressure latency/count, admission concurrency,
sampled device memory, system memory, output length and MTP acceptance snapshots.

Completed eight-input comparison (`pressure-gen64-20261021-r2/comparison.json`):

| Measure | G16K | G64K |
| --- | ---: | ---: |
| Complete input tokens | 2,094,967 | 2,094,967 |
| All-wave prefill wall seconds | 400.94 | 493.09 |
| All-wave prefill wall token/s | 5,225.19 | 4,248.68 |
| Pure-prefill model interval token/s | 5,912.32 | 4,669.80 |
| Pressure refresh count | 104 | 24 |
| Mean pressure suspended-to-ready seconds | 1.814 | 1.794 |
| Peak admitted client requests | 8 | 7 |
| Eighth request queue seconds | .57 | 406.78 |
| Complete client wall seconds | 434.32 | 498.33 |
| Output tokens | 5,984 | 3,374 |
| Cold model preparation seconds | 76.25 | 75.14 |
| Sampled peak device memory, GB | 77.86 | 77.51 |
| Sampled peak system used memory, GB | 113.86 | 104.78 |

Pressure count decreased 76.9%, but all-wave prefill throughput decreased 18.7%.
Model-interval throughput decreased 21.0%; the executed batch-size distribution
also changed, so this is not isolated attention-kernel attribution. Total client
wall time increased 14.7% despite shorter generated output. MTP accepted-length
snapshots were 3.81 and 4.04, respectively, but snapshots are not exact final
counters. Both cohorts scored seven of eight fully correct; `niah_multikey_3`
scored zero in both. All execution, accepted-row and owner-release audits passed.
Native KV pool capacity remained 42,253,254,656 bytes; a larger per-request
reservation need not grow a preallocated device pool. No default was changed.

At the user's request, a matched single-request pair is recorded under
`artifacts/pressure-gen-single-20261021-r1/{g16,g64}/`, using the same complete
`niah_single_1-0000` prompt, seed and sampling settings. Server limits remain
unchanged, but each cohort contains only one request, so the CPU queue admits
both immediately. This removes the multi-request admission difference above.

Completed single-request results (`pressure-gen-single-20261021-r1/comparison.json`):

| Measure | G16K | G64K |
| --- | ---: | ---: |
| Complete input tokens | 262,045 | 262,045 |
| Client time to first output token, s | 82.672 | 88.198 |
| Effective input token/s (input / TTFT) | 3,169.68 | 2,971.10 |
| Counter-observed all-prefill wall token/s | 3,129.37 | 2,933.21 |
| Pure-prefill model interval, s | 53.649 | 67.519 |
| Pure-prefill model interval token/s | 4,865.35 | 3,865.92 |
| Pressure refresh count | 13 | 3 |
| Total pressure suspended-to-ready, s | 4.407 | .952 |
| Query suspended-to-ready, s | 1.404 | 1.886 |
| Complete client wall, s | 85.149 | 90.761 |
| Accepted output tokens | 111 | 112 |
| Cold model preparation, s | 75.149 | 75.168 |
| Actual peak admitted requests | 1 | 1 |
| Sampled peak device memory, GB | 75.57 | 75.57 |
| Sampled peak system used memory, GB | 43.99 | 43.64 |

The effective input rate includes submission, prefill, pressure refreshes,
semantic Query selection and time to the first output token; it excludes cold
loading. The approximately one-second publication cadence explains the modest
difference between this rate and the counter-observed prefill rate. The model
interval covers 261,021 pure-prefill positions in each run; the final 1,024
positions are handled in a batch classified as mixed and are excluded there.

With queue delays below one millisecond in both arms, G64K reduced effective
input throughput 6.27% and increased time to first token 6.68%. Pressure waiting
decreased by 3.45 s, while the measured model interval increased by 13.87 s.
These timers have different scopes and are not an additive decomposition of
the client wall time. The larger window did not pay for itself in this pair.
This is one prompt and one measurement per arm, not a stable statistical estimate
or a quality benchmark. Both official case scores were 100, all policy/accepted-row
audits passed, and both owners released their CPU/GPU reservations.

Model source and harness hashes were identical between the single-request arms;
only `kvmem.gen_budget` differed in their profiles. MTP4 stayed enabled, but the
five-second telemetry cadence missed the brief decode interval in both arms:
zero values in the last MTP snapshots are not zero acceptance. No reliable
decode throughput or MTP acceptance comparison is inferred from this short-output
test. Defaults remain G16K; the pre-existing DSH profile is restored after testing.
