# Statistical historical GC and block graph coexistence

This extension adds an opt-in native statistical policy and an explicit semantic
evidence API. It can reclaim completed step details while retaining their result
KV. It does not infer task completion from token text, implement R-KV key
similarity, or launch a second model request.

## Configuration

Keep the existing CPU archive, request planning, asynchronous seal/retirement,
CPU history coordination and owned-session settings required by historical GC.
The legacy `repository_gc` mechanism remains incompatible. Native
`historical_gc` may now be enabled with block graph modes.

```json
"historical_gc": {
  "mode": "shadow",
  "policy": "statistical",
  "policy_params": {
    "min_observations": 3,
    "min_cold_observations": 3,
    "max_importance": 0.1,
    "proactive": true,
    "min_model_reclaim_score": -1.0
  },
  "poll_ms": 10,
  "scan_blocks": 256,
  "delete_blocks": 64
}
```

Use `delete` to commit retirement after qualifying the evidence producer and
quality evaluation. `off` retains ownership without policy decisions;
`shadow` records decisions without deleting. These defaults are conservative
heuristic gates, not a fitted statistical model or calibrated probabilities.

With `proactive=true`, eligible cold blocks can retire before capacity pressure.
With `false`, the same evidence gates also require pressure. Both modes preserve
sink/recent and P/M mandatory blocks. Unlike configurable LRU, this policy does
not expose a switch to ignore mandatory protection. A span with no observations
or no completion evidence cannot retire under this policy.

## Explicit semantic producer

The application must identify accepted-token ranges, completed steps and retained
results. The controller runs on the owning request thread at application or
maintenance boundaries, never as a Python callback on the native worker.

```python
ledger = authority.gc.open_evidence(authority.records, epoch="turn-1")
registered = ledger.register_span(
    "step-7", source_begin, source_end,
    kind="completed_step", epoch="turn-1",
)
# Only after the application has verified successful completion:
completed = ledger.complete_span(
    "step-7", result_records, epoch="turn-1",
)
# Each sample must cover every block in the registered span.
accepted = ledger.observe(
    "step-7", epoch="turn-1", observation_id=sample_id,
    importance=measured_importance, cold=measured_cold,
)
# Retry false at a later maintenance boundary; do not busy-wait during decode.
published = ledger.flush(max_blocks=32)
```

Methods that return false have not accepted the requested change. A successful
change is staged until `flush` has published it. Incomplete publication holds all
source generations against policy retirement; later flush calls retry a bounded
number of blocks. A new observation invalidates an older optional model score.

Supported semantic kinds are `thinking`, `completed_step`, `tool_detail` and
`superseded`. They are labels, not deletion rules. A closing thinking tag or a
successful tool return alone does not establish that all details can be dropped.
The producer decides whether retained results cover future dependencies.

Only full, sealed, archived blocks wholly contained in the registered range are
eligible. Mixed boundary blocks and partial tails remain. A result must consist
of full archived blocks belonging to the same authority. Result and source
candidate sets cannot overlap. Result guards are hard native guards, independent
of policy configuration, and last until ledger reset or request close.

Observation ids must increase. Missing attention coverage is unknown, not zero;
a low score for one sampled block must not be assigned to an entire span.
The caller supplies a normalized importance measure and a cold predicate;
the library does not invent attention statistics from elapsed time, GC polling
or GPU residency. Cold counts are consecutive valid cold observations, reset by
a valid hot observation. The latest valid importance must also pass the threshold.

The ledger is bounded: by default at most 64 spans, with candidate and distinct
result block sets each bounded by repository capacity. Completed spans and result
guards remain in that budget. At an application-defined safe boundary,
`ledger.reset()` withdraws live eligibility and returns false until publication
finishes; after true, `open_evidence` can create another ledger. Authority rewind
calls this reset before changing the accepted branch. Evidence revisions remain
monotone across ledgers. Closing the request releases ledger guards before native
scope closure.

Do not retire witness blocks directly through the raw repository API while the
ledger is active. Explicit repository retirement is a separate destructive
mechanism used by lifecycle operations and can override policy guards; application
branch changes must use the authority rewind/reset protocol.

## Optional asynchronous model signal

Capture `ledger.revision(span_id, epoch=epoch)` with the judge input. Supply that
exact revision to:

```python
accepted = ledger.model_signal(
    span_id, epoch=epoch, based_on_revision=captured_revision,
    decision_id=decision_id, reclaim_score=score,
)
```

A late result from an old ledger, branch, observation revision or decision id is
rejected. To require a score, configure `min_model_reclaim_score` in [0, 1]; -1
disables that gate. Unknown scores do not satisfy an enabled gate. Model signals
cannot bypass completion, real observations, mandatory protection or native
selection admission.

This implements the signal ingestion contract only. Judge scheduling, KV-prefix
sharing, prompts, calibration, teacher/student training and a per-request
inference budget remain future work. There is no extra model forward or GPU work
in the statistical policy.

## Coexistence protocol

1. Graph selection pins the live, generation-qualified candidate set with
   `protect_selection` before scoring. It holds that protection through native
   execution admission. Custom policies and LRU `keep_mandatory=false` cannot
   override this guard.
2. A failed admission rebuilds the complete candidate domain and graph scores;
   it does not reuse a score vector normalized over retired candidates. Retries
   per attempt are bounded. Async refresh yields its maintenance credit and
   retains input state for a later scheduling turn.
3. The admitted plan owns execution/CPU-read leases. Selection guards end at
   admission, so GC can retire the CPU archive while an old GPU window remains
   in flight. Ordinary fenced refresh later removes those GPU pages.
4. Retirement queues small identities. At a safe refresh boundary, graph cleanup
   uses copy-on-write to remove retired sources and targets. Original token
   positions and accepted watermarks remain unchanged; existing snapshots remain
   valid. Temporarily absent future P..M replay nodes are preserved.
5. Insufficient COW budget defers graph storage cleanup. Live-candidate filtering
   still prevents retrieval of retired blocks. Pending identities are bounded;
   overflow requests reconciliation at a later boundary.
6. Native delete intents carry the evidence revision. Commit checks the current
   generation, revision, mandatory state and hard guard under the same gate as
   evidence publication. Stale intents cannot delete using superseded evidence.

## Validation scope

The checked-in [functional report](benchmarks/statistical-gc-functional-20261023.json)
records exact suite selections, counts, exclusions and environment. Suites overlap.

Functional validation uses Python 3.12.15, torch 2.13.0+cu130 and an RTX 4090
(sm89), plus the C++ host test suite. The tests include native concurrency,
generation/revision reuse, partial evidence publication, retained results,
graph COW, admission retry, branch rewind and CPU/GPU lifetime independence.

The GPU evidence test actually runs the new statistical policy while a copy of
source KV is still used on a CUDA stream, verifies CPU archive reclamation and
checks the GPU bytes and retained result. It is a lifetime test, not an attention
kernel or model-quality benchmark.

Reproduce from an installed development checkout:

```bash
cmake -S . -B build/history-gc -G Ninja
cmake --build build/history-gc -j8
ctest --test-dir build/history-gc --output-on-failure
python -m pytest -q tests/test_historical_evidence.py tests/test_historical_statistical_native.py
KVMEM_RUN_GPU_TESTS=1 python -m pytest -q tests/test_historical_payload_gpu.py tests/test_historical_evidence.py
```

The [cloud Qwen3.8 report](qwen38-cloud-smoke-20261023.md) now records a
single-RTX-4090 NVFP4 model smoke test and five paired, serial statistical-GC
throughput measurements. All measured DELETE requests retired and released
four candidate archive blocks during decode. The paired median throughput loss
was -0.115%, with a descriptive bootstrap 95% interval of [-0.796%, 0.690%],
passing the predeclared 1% upper-bound criterion for this workload.

That result measures net single-segment throughput, including different terminal
checkpoint paths: OFF saved P+M, while DELETE deferred M and retained P. It does
not qualify next-turn replay cost. Eligibility used a synthetic diagnostic
fixture, and the GPU working window did not refresh during the measured output.
It therefore does not establish real semantic-GC quality, block-graph throughput,
or general zero overhead. The repository's older Blackwell LRU numbers do not
establish those properties either. The native attention-statistics backend still
requires SM12; its production performance needs supported hardware.

CPU-only background work avoids an extra decode-side model forward, but it can
still contend for CPU/cache/memory bandwidth. Graph protection, evidence
publication and COW add work at refresh boundaries. A throughput claim needs
paired off/shadow/delete runs with GC demonstrably active, equal task quality,
and decode throughput plus p95/p99 inter-token latency reported separately from
refresh latency.
