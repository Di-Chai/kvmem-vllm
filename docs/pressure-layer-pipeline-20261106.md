# Prefill pressure per-layer KV pipeline experiment

Status: implemented and measured; disabled by default. The two eight-owner
comparisons do not establish a throughput or GPU-utilization benefit.

## Execution contract

`KVMEM_LAYERWISE_PRESSURE_PIPELINE=1` permits a request to resume after its
first KV layer is ready. Each native KV writer and attention reader waits on
the recorded event for its own layer. The existing background assembler can
continue loading later layers while earlier layers compute. Selection,
accepted attention scores, mandatory blocks and original numerical kernels
are unchanged.

Early dispatch publishes the host mapping; it is not a whole-window completion
fence. The producer future, immutable read plan, page-table views, scratch and
maintenance credit remain owned until final materialization completes. Request
termination and page recycling wait for that completion. Whole-window
checkpoints cannot consume a partially materialized window. Producer errors
wake consumers and fail explicitly.

The qualified path uses static page allocation, prepared immutable reads,
single-pass assembly, reused layer views and paired native reads. Incompatible
allocation/staging options fail explicitly. This experiment does not establish
dynamic-allocation compatibility. Native PIECEWISE compute graphs are retained;
FULL graph replay is disabled while a materialization is outstanding because
it would bypass the per-layer dependency guards.

## Frozen controls and measurement

- Local Pro6000, original NVFP4 model and original sampling/output controls.
- Active KV 36864, Gen KV 16384, block 32, sink 41, recent 0.
- Layerwise cumulative-attention prefill pressure and sequential Mean-K Query.
- Eight original complete cold histories, 2,490,368 historical tokens per run.
- Same 8K physical coalescing, 1K logical submission, native batch cap 16424,
  static GPU pool and CPU cache budget in both variants.
- Run order: baseline, pipeline, pipeline, baseline. First answers preserved.
- Numerator is GPU-confirmed historical prefill work. The interval includes
  selection, assembly, dispatch and waits, and excludes model loading, final
  Query and decode. Identical frozen inputs/configuration and unchanged native
  autotuning were verified after execution.

| Repeat | Baseline tokens/s | Pipeline tokens/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 5755.44 | 5757.33 | +0.03% |
| 2 | 5792.43 | 5776.71 | -0.27% |
| Pooled tokens / pooled time | 5773.87 | 5767.00 | -0.12% |

| Repeat | Baseline samples below 20% GPU utilization | Pipeline samples below 20% |
| --- | ---: | ---: |
| 1 | 8.10% | 6.02% |
| 2 | 4.65% | 5.34% |
| Pooled | 6.38% | 5.68% |

Utilization is sampled approximately once per second. These percentages are
sample shares, not measured GPU idle durations. The direction reverses in the
second repeat, and the baseline's own variation is larger than the pooled
difference; a stable reduction has not been established.

Across the two pipeline runs, only 3 of 288 materializations recorded a first
foreground consumer submission before the final layer submission. Most were
already fully ready when resumed. Subsequent investigation found a late
handoff across the model-batch feedback boundary; this count alone does not
establish a lack of physical overlap opportunity. See
`pressure-layer-handoff-20261106.md` for the cause and early-handoff experiment.
Other requests can already compute during background assembly, and CPU
submission timestamps do not measure actual CUDA overlap duration.

## Correctness and limits

Three original complete serial inputs (two cold prefixes and one reused-prefix
request) passed exact equality for every pressure selection, final Query
selection, complete output token sequence, text and answer. The final v5
implementation also passed 207 focused checks and 23 analysis checks.

Eight-owner executions did not pass exact replay: both pipeline runs and the
second baseline differ in pressure/Query selections from the first baseline,
and only 1/8 complete answers match that first baseline. Baseline-versus-baseline
has the same 1/8 count. This experiment therefore provides serial correctness
evidence, not an eight-owner exact-equivalence qualification. No answer was
resampled to obtain agreement.

The separate restore regression run has four fixture failures both with the
preserved pre-change code and with the current code (24 cases each). They are
the same failures in existing restore fixtures lacking the worker runner; the
full repository test suite is not claimed to pass.

## Artifacts and disposition

All evidence is under `artifacts/pressure-layer-pipeline-20261106-r1/`:

- `serial-v5-comparison.json`: exact full-input qualification.
- `final-result.json`: four measurements, utilization samples, controls and
  concurrent first-answer comparisons.
- `completion-audit.json`: final resource and status audit.
- `restore-fixture-audit.json`: preserved-before/current failure comparison.
- `eight-v5b/`: immutable runtimes, inputs, first answers and raw telemetry.

The frozen legacy summary reader expects a whole-window `ready` event and
fails after all four inference arms have completed. Its error is preserved.
The updated read-only analyzer separately validates `pipeline_dispatch` and
`pipeline_completed`, without changing traces, rerunning answers or counting
negative completion waits. All four arms completed 8/8 and released their GPU
resources; the final audit and frozen-file verification passed.

Retain the implementation as an opt-in experiment. Do not enable it by default
or claim a performance gain. The paused formal MAB queue remains paused; this
experiment did not run external grading or create a periodic monitor.
