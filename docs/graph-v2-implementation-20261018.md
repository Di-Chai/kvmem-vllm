# Graph V2 implementation and qualification ledger

User approved this implementation/test plan on 2026-10-18 (Asia/Shanghai).
Starting commit: `c14c6bf822246a0af9c5af6739951bdf5fa5b19b`, initially clean.
Specification: `/home/chaidi/qw3/docs/kvmem_graph_scoring_v2_handoff_20261018.md`.
This ledger records implementation status; a planned check is not a passed check.

## Approved sequence

1. Freeze baseline; align Mean-K through the original native score frame. Separate
   softmax mask from actual native fixed-keep/transport mask. Rebuild baseline even
   for graph off. Preserve the c14 dense native graph-accept optimization.
2. Versioned V2 one-hop operator (alpha=.2, equal shares in multi) and graph-only
   locality protocol/real bridge producer in parallel. Keep old V1 profiles valid.
3. Restore order: validate incoming metadata, rewind/restore statistics, install
   boundaries, capture snapshot/score, assemble/publish. Then integrate read-only
   cumulative output/prefill attention seeds into all three pressure paths.
4. Mathematical, native, protocol, lifecycle and real-model qualification before
   long benchmark admission. No new recent-Q cache or additional model forward.
5. Independent performance candidates: bounded raw-K cache, Action D2H buffer
   reuse, remove residual CPU seal copies; cross-owner scoring batching only if
   detailed profiling justifies it. Preserve numerical/ownership contracts.
6. Freeze selected execution version for full A/C/D quality comparisons.

## Required qualification gates

- Mandatory denominator oracle selects M+B in actual authority/native execution;
  small history, runtime budget, sink/recent overlap, invalid counts/generations,
  observer preview, scoped restoration after success/failure.
- Independent matrix oracle, [0.175,0.220,0.605] multi example, filtered attention
  row normalization, zero/identity exact fallback, one hop and conserved mass.
- Locality metadata never changes tokens, Query/replay, mandatory or explicit
  mandatory. Actual bridge producer, UTF-8/offset mapping, accepted output IDs,
  append, unknown boundaries, conflict/edit handling and schema compatibility.
- Pressure alpha=0 matches the attention picker including zero candidates,
  output-before-prefill fallback, partial tail, runtime budgets; no cold recall.
- Sync/async/checkpoint, complete M-prefix retained across P replay, true edits,
  newly supplied metadata, cancellation/failed capacity and resource release.
- Real long input with multiple pressure selections and multi-turn tools, then
  eight-owner correctness/concurrency. Existing dense native/scalar tests remain.
- Bounded versioned diagnostics/replay uses correct Query vs pressure picker.

## Performance protocol

Fixed 256K inputs plus frozen eight-agent tool-call replay, continuous replacement;
warm up then 300s measured windows and drain. Single-variable screening, promising
candidates get at least three paired, alternating-order measurements. Proposed
selection gate approved in plan: median primary useful throughput >=5% improvement,
no consistent important latency regression, unchanged correctness and target
concurrency. Record actual batches, timeline, transfer volume, host/GPU memory and
restore latency. NVML utilization/overlapping host elapsed alone is not throughput.

## Complete quality matrix

All arms share aligned Mean-K, attention capture, execution optimizations, tokens,
sampling, K/G and batch geometry; action_fraction=0 and query_history=false.
A uses collect (no Query propagation) and legacy attention pressure; C uses V2
multi Query alpha=.2 and legacy attention pressure; D uses V2 multi Query and
attention-seeded graph pressure with the same alpha. A short graph-off performance
comparison separately measures collection overhead. No tuning using answer gold.

- NIAH: original 60 cases each at 64K/128K/256K, 180/arm, 540 total. Start with 256K;
  completed cases count toward the full run only when runtime/protocol are unchanged.
- MemoryAgentBench: original frozen 1,116/arm, 3,348 total, independent questions
  restore frozen CPU prefix without output-statistic leakage across questions.
- Independent LongMemEval-S: original 102/arm, 306 total (distinct from MAB subset).
- Total 4,194 formal answers, excluding small qualification/performance tests.
- Eight clients with continuous replacement; actual active history groups may be
  lower for MAB due CPU admission. Full CPU history, no SSD/recompute/truncation.
- Existing authorized judge protocol; never write credentials to evidence files.
  Keep original first answers and technical failures. Unknown locality for raw
  frozen token input is not evidence of semantic-boundary coverage.
- Separate selection/evidence metrics, answer quality, prefill/decode and end-to-end
  costs. Old published results are historical context, not newly aligned controls.

## Execution status

- Baseline verified; core implementation and structured sync/async model
  qualification are complete. Performance and full quality evaluation remain
  in progress; this is not a completion record for the overall goal.
- Native score-frame binding and shared native fixed-keep extraction implemented.
  Actual authority mandatory reversal, exact scope/lease cleanup, generation reuse
  and transport masks pass. Imported `kvmem_store_test.cpp` compiled and returned OK.
- V2 operator, configuration, graph-only protocol and all three pressure/restore
  entry points implemented. Combined targeted CPU run against the frozen sync
  model runtime: 695 passed (see `frozen-cpu-regression.log` in the session
  qualification directory); additional producer/protocol suites were run
  separately and are not added to this count.
- Actual qw3 renderer + local Qwen tokenizer CPU producer qualified with unchanged
  complete token IDs and independently delimited tool results; evidence at
  `artifacts/graph-v2-locality-20261018/cpu-producer.json`.
- Two native edge corrections: pressure runtime budgets no longer apply Query
  recent-band requirements; explicit candidates constrain all-fit pressure too.
  The all-fit correction is an intentional exception to literal c14 alpha=0
  equivalence: c14 returned the entire repository when it fit the budget, even
  when the explicit current-window candidate set C was smaller. Pressure now
  keeps selection within C plus native sink/mandatory pages, so cold pages are
  not recalled by that shortcut. Both attention and graph alpha=0 use this fix.
- Pressure admission and sync/async runtime fitting now use the resolved policy:
  attention/graph omit the Query recent band, while recency and Query preserve it.
  Pressure precedes the all-history bootstrap at both refresh entries. Restored
  partial tails participate in the native mandatory fit at the actual runtime
  budget; sink or explicit hard evidence cannot be silently dropped to fit them.
- Legacy `TestClient` thread startup hangs in this environment before application
  code; new gateway tests use same-loop ASGI transport. This is not GPU evidence.
- Structured real-model qualification has completed in separate frozen sync and
  async attempts under `artifacts/graph-v2-sessions-20261018/`. Each uses two
  owners and three turns, actual 131,063-token initial prompts, real tool-call
  output IDs, byte-verified restoration, and an edited Query branch. Completion
  and pass/fail are recorded per attempt; startup is not a passing result.
- A CPU-only sealing feasibility probe (not serving throughput) compared the
  current two-copy immutable packing path with independently owned record bytes.
  Median packing plus SHA time was 62.8 vs 39.9 ms for 35.7 MB, and 504.6 vs
  317.5 ms for 285.2 MB. The disabled-by-default `cpu_direct_record_seal`
  candidate is implemented; 18 new ownership/publication tests and 99 related
  CPU tests passed. Real-model A/B qualification is still pending; no serving
  throughput gain is claimed.
- Sync and async qualification passed: two owners and six turns each, five attention-seeded graph
  pressure selections per initial history, actual tool results and four verified
  M/P restores. Each restore checked 17 working attention layers including one
  MTP layer plus 96 recurrent tensors. Both owned sessions released in each
  attempt; controller cleanup verified no remaining owned or GPU processes.
- Eight-owner original 256K functional A/C/D fixtures are frozen under
  `artifacts/graph-v2-smoke8-20261018-r1/`. D passed: all eight cases completed
  and released, 7/8 exact, 96.875% coverage, 385.04 s client wall time. Actual
  executed batch peaked at seven, despite eight admitted owners. Pure-prefill
  model-step rate was 5,992 tokens/s; this includes replay work and is not
  end-to-end input throughput. Accepted-output / whole-wall rate was 4.35
  tokens/s and is not decode throughput. A also passed (7/8 exact, 96.875%
  coverage, 378.91 s, actual batch peak seven); C passed too (7/8 exact,
  96.875% coverage, 380.17 s, actual batch peak seven). All three controllers
  verified empty owned process groups and no remaining GPU processes. The small sample
  therefore establishes no quality improvement over A. This is not
  the complete quality matrix or the strict paired performance campaign.
- Profiling refinement: a custom D2H pinned allocator is not justified by the
  measured allocation cost. Sync diagnostic samples recorded 1.457 ms across 88
  allocations, compared with 1.396 s for `metadata.starts_h2d`; PyTorch already
  caches pinned allocations. These overlapping sampled host scopes are not GPU
  idle. Test the existing `async_action_metadata` flag independently instead;
  leave `native_action_metadata` unchanged so that the comparison has one cause.
- Full MAB/LME source preflight passed under
  `artifacts/graph-v2-suite-preflight-20261018/cpu-review-r1/`: 1,116 MAB and 102
  LME original inputs plus their original evaluator bundle were frozen. MAB
  contains prompts up to 757,259 tokens, so its three arms use a 1M logical
  context / 32,768 repository blocks; input truncation is forbidden. CPU credit
  admission may keep fewer than eight independent MAB histories resident.
- LME protocol difference is explicit: historical references allowed 32,768
  output tokens; all new A/C/D runs use the approved G=16,384 cap, matching the
  existing local serving harness. Preserve original sampling/inputs and report
  cap hits. Historical quality numbers are not exact output-protocol controls.
- Performance screening fixtures are frozen at
  `artifacts/graph-v2-performance-20261018/{raw-k-r1,seal-r1,metadata-r1}`.
  Schema 3 uses the actual client measurement interval for NVML, reports cache
  capacity as a gauge instead of summing it, and marks output/selection drift,
  an unexercised candidate or missing real eight-owner execution inconclusive.
  These cold screening pairs cannot establish the continuous-agent merge gate.
- The independent fixed8 natural-agent controller is frozen under
  `artifacts/graph-v2-agents-20261018/{raw-k-r1,seal-r1,metadata-r1}`. It preserves
  original input/tool/wait traces, full own-output histories and absolute Query
  locations, performs two warmup turns, measures a common 300 s window, then
  separately drains. New agent + performance CPU checks: 68 passed. Frozen
  prepare/validate passed for each candidate. Legacy extended cleanup tests
  encountered an environment thread-completion hang and were terminated; they
  are not reported as passed. No background CPU test process was left running.
- Complete formal results will be placed under `/data/chaidi/kvmem-vllm-experiments/`
  because retained original prompt IDs and answers require an estimated 25–35GB,
  exceeding comfortable headroom on the workspace filesystem. This is an
  evaluation-artifact location only; KV authority remains CPU memory.
- The first raw-K cold pair completed. Raw0 vs raw4g: 387.57 vs 396.02 s;
  observed assembly upload 71.68 vs 48.80 GB; raw4g hit 1,312,640 tokens.
  Both scored 7/8 exact and 96.875% coverage. Selection and accepted-output
  sequences differed and native batch peaked at seven, so the comparison is
  inconclusive and establishes no serving speedup.
- Capacity analysis found raw4g reserves only 2 GiB per async arena and 481
  records (15,392 tokens) per owner, 41.75% of K. Eight complete FP8 raw-K
  windows require 4.78125 GiB per arena / 9.5625 GiB total under the existing
  split. A separate derived-capacity raw0/raw10g candidate is frozen at
  `artifacts/graph-v2-performance-20261018/raw-window-r1` and the corresponding
  `graph-v2-agents-20261018/raw-window-r1`; actual allocation is capped by the
  window geometry. Candidate CPU checks: 135 passed; prepare/validate passed.
  This does not alter the already measured raw4g attempt or vLLM KV pool.
- Completion review identified required comparisons not covered by cold smoke:
  graph-off/collect overhead; controlled V1/V2 attention, attention-only/multi
  and known/unknown boundaries; sustained fixed256K after warmup. The collection
  pair is implemented and frozen under `artifacts/graph-v2-collection-20261018/r1`
  (32 new CPU checks passed), but has not run on GPU. The remaining controlled
  ablation and sustained-input harnesses are being implemented; no coverage is
  claimed from their designs alone.
- Fixed structured-input ablation is now frozen at
  `artifacts/graph-v2-fixed-ablation-20261018-r1`: four profiles, two actual
  turn-1 inputs each, matching tokens/Query/mandatory/recency pressure. Its
  92 targeted CPU checks and frozen validation passed. Full selection closure
  uses complete bounded-refresh records, followed by same-snapshot replay.
  GPU execution and interpretation are pending. Opt-in diagnostic export now
  accepts 8,192 nodes / 2 MiB per record, retaining eight-record and 16 MiB
  per-file caps; production graph_trace=false remains unaffected.
- A read-only final source audit found no deterministic blocker in the fixed
  4,194-answer campaign. The longest grouped MAB replay is 4,431 tokens; LME
  Query plus partial tail is at most 106, both below the 16,352-token replay
  allowance. This audit is not a completed formal model evaluation.
- CPU direct-seal cold pair completed under
  `artifacts/graph-v2-performance-20261018/seal-r1`. Copy/direct wall time was
  385.08/385.89 s; useful input rate 5,445/5,434 tokens/s, both 6/8 exact and
  93.75% coverage. Direct publication was exercised on 65,571 records / 73.05 GB.
  Recorded second-copy work fell from 50.11 s to zero, but pack/hash/publication
  cumulative work was approximately 170.46/172.08 s, so this serving workload
  did not reproduce the microprobe benefit. These host durations overlap GPU
  work. Selection/output sequences differed and actual native batch peaked at
  seven; the pair is inconclusive and establishes no speedup. Default remains
  disabled pending sustained evidence.
- The fixed256K sustained controller and stream qualifier are implemented;
  68 targeted stream/performance CPU checks passed. One seal pair is frozen
  and validated at `artifacts/graph-v2-stream-20261018/seal-r1`, not yet run.
  Eight independent slots refill after each release, eight completed warmup
  requests precede a fixed 300 s window, and post-window drain is excluded.
  Public-work fences, accepted SSE tokens and source-file staleness are kept
  separate; zero window output is legitimate for this long-prefill workload.
- Independent GPU checks are queued serially by the task-only artifact
  `artifacts/graph-v2-performance-20261018/queued-checks.py`: it waits for the
  running metadata pair, then runs derived raw-window capacity, collection,
  and fixed-input ablation plus its analyzer. It stops on any failure and
  never changes frozen inputs. Queue status is recorded beside the script.
- Final-profile audit corrected a preparation bug: the shared A/C/D builder
  now preserves the explicitly selected raw-K cache budget instead of silently
  resetting it to zero. Independent performance pairs explicitly reset their
  baseline knobs before changing the single candidate. Existing frozen runs
  are unchanged; targeted benchmark/performance/stream regression passed in
  `artifacts/graph-v2-performance-20261018/final-profile-regression.log`.
- Async metadata cold screening completed: sync/async wall time
  387.17/377.60 s, useful input rate 5,416/5,553 tokens/s (descriptive +2.53%).
  Both scored 7/8 exact / 96.875% coverage. The enabled path issued 2,086
  uploads totaling 22.80 MB with no reported submission/fence/query failures;
  its final per-owner snapshot still had one pending lease and is not itself
  a final cleanup proof. Selection/output drift and actual batch peak seven
  keep the pair inconclusive. The separate controller cleanup passed.
- A clean-environment stream-controller validation exposed an import bug:
  importing qualifier constants pulled in the installed serving package before
  the frozen runtime was selected. The constants now live in the dependency-light
  shared protocol. The untouched stream `seal-r1` and `metadata-r1` preparations
  are retained as pre-fix artifacts; no GPU run was started. New clean validation
  and preparation are pending the next CPU window.
- The stream bootstrap fix passed 131 targeted benchmark/performance/stream
  checks, including a subprocess that rejects any serving-package import by
  the controller. `graph-v2-stream-20261018/{metadata-r2,seal-r2}` were prepared
  and validated with PYTHONPATH removed; both passed before GPU timing began.
  Their pre-fix r1 preparations are marked `superseded_without_gpu_run`, with
  original manifests and failure logs retained. These r2 GPU tests are pending.
- Derived full-window raw-K cold pair completed: raw0/raw10g wall
  383.47/381.52 s (descriptive +0.51% useful-input rate), both 7/8 exact and
  96.875% coverage. Observed assembly upload fell 71.58 -> 40.38 GB and host
  preparation cumulative time 21.13 -> 16.09 s; 1,797,888 tokens hit raw-K.
  Actual allocated cache bytes remain unobserved by the dedicated capacity
  gauge; NVML peak was 74,028/79,122 MiB and minimum free 23,223/18,129 MiB.
  Output/selection drift and native peak seven make the pair inconclusive;
  transfer reduction does not establish serving speedup. Keep raw-K off by
  default. Async metadata is the next sustained-screen candidate based on
  its larger cold observation, without promoting it to a default or claiming
  it meets the >=5% repeated-pair admission gate.
- Collection r1 failed before any model request: its launch audit read
  lazily initialized `action_config.json` before the first forward. Cleanup
  verified empty owned process groups / no GPU processes. Original r1 status
  and logs remain failed; collect was never started. The action runtime audit
  now runs after the real cohort, retaining the strict field checks and adding
  no probe forward. Regression including the missing-before-first-request
  case: 164 passed. Collection r2 was newly frozen and validated.
- The original independent queue and dependent sustained queue stopped on that
  failure. `queued-recovery-r2.py` is the active serial continuation: collection
  r2, untouched ablation r1 plus analyzer, metadata stream r2, then untouched
  natural-agent metadata r1. It uses a frozen source path for the agent entry,
  stops on any failure and does not retry started campaigns.
- Collection r2 off arm completed all eight cases and released all owners:
  374.40 s, 7/8 exact / 96.875% coverage, actual native peak seven. The post-
  request ActionRuntime configuration audit passed; cleanup confirmed no owned
  groups or remaining GPU processes. Collect is still pending, so no collection
  overhead comparison or full-quality conclusion is established yet.
- Collection r2 finished both arms. Off/collect: 374.40/380.76 s (raw wall
  ratio 1.017), both 7/8 exact / 96.875% coverage, actual native peak seven.
  Collect graph prepare/commit observed cumulative maxima were 7.0846/0.1865 s
  across 2,562 calls each, not GPU idle. Selection, output and attention-sample
  drift make the overhead comparison inconclusive. Both cleanup files confirm
  no owned groups or GPU processes. The queue has entered fixed-input ablation.
- Implementation checkpoint c9a08c6f545265a3667e6b5b305374b01399b3e4 was
  committed on main and pushed to `Di-Chai/kvmem-vllm`; its documentation
  explicitly keeps sustained and full-quality evaluation pending.
- Fixed-input ablation passed all four model arms and its frozen analyzer:
  eight actual requests / eight full native selection matches, two canonical
  snapshots, positive normalization and genuine boundary effects in both.
  Each snapshot covered eight positive-seed cross-boundary pairs. Canonical
  V1->V2 attention changed 274/277 selected blocks; attention->multi unknown
  changed 125/130; known boundaries changed scores but not selected pages.
  No answer/retrieval labels were available. All four cleanup files confirm
  no owned groups / GPU processes. The queue entered metadata stream r2.
- Metadata stream r2 baseline passed its genuine warmup/window/drain audit:
  16 admitted requests all completed/released; individual refill happened
  before all first-eight owners retired. T1-T0=300 s, public completed prefill
  delta 1,596,896 (5,322.99/s), conservative completion bounds
  1,581,632..1,610,112. All 452 measured model steps were prefill; native peak
  six, model-event prefill 5,687.60/s. Accepted output and completed requests
  in the window were both zero. NVML mean utilization 88.62%; no pure decode
  or GPU-idle inference. Cleanup confirmed no owned/GPU processes. Candidate
  is still running, so no sustained A/B result is established yet.
- Metadata stream r2 candidate also passed, then the pair completed pending
  review. Candidate public completed prefill 5,449.49/s (bounds
  1,619,456..1,644,160 per300s), model-event prefill 5,692.30/s, native peak six,
  zero window output and zero pure-decode steps. NVML mean91.71%. Raw observed
  prefill ratio1.02377; comparison is inconclusive due output/selection drift,
  absent native C8 and no timestamped within-window metadata path proof.
  Neither flag is promoted. All admitted requests/credits and owned processes
  were released in both arms. The serial queue started natural-agent metadata
  r1 baseline after the conservative host-admission guard succeeded.
- Original reference plan rechecked directly: `attention_multilayer_prefix_reuse_20261012_r2/plan.json`
  specifies K=36,864, G=16,384 and thinking=4,096. The documented LME request-cap
  difference remains explicit; increasing G would also change pressure/prefill
  geometry. All new A/C/D arms retain the same G and served cap, with cap-hit
  audit required before interpreting comparison to historical scores.
- Natural-agent metadata r1 baseline finished its window/drain but failed the
  final release audit. Final snapshots showed no actual CPU history/checkpoints,
  no leases/reservations, 750/751 native blocks free (one null block), and
  post-stop GPU empty. `qualify_session_release` had unconditionally required
  telemetry for idle active residency, which this profile explicitly disables.
  Its default remains strict; only callers that verify the actual disabled
  launch may permit that optional field to be absent. Required resource fields
  still cannot be missing or nonzero. Offline `diagnostic-release-recheck.json`
  verifies this without reclassifying the failed attempt.
- Prelaunch audit review also corrected repository capacity vs logical ceiling:
  16,384 x32 CPU blocks means 524,288 history-token capacity, while the server
  limits each logical trajectory to262,144. The actual launch is unchanged.
  Combined agent/release/performance/stream regression:113 passed. New-source
  qualification permits only the guarded opt-in trace-cap expansion after an
  AST comparison proves the remaining graph code identical and tracing off.
  Unrun metadata r2 is superseded, retaining its manifest; metadata r3 was
  prepared/validated and has started a new complete pair after host admission.
- Agent r3 completed workload, drain, and verified explicit release, then the
  Query-event audit failed. The original fixed fixture uses an early `auto`
  Query inside K; all original Q pages are hard mandatory in pressure windows,
  and no scored Query event was observed. The audit now allows this native
  all-fit path only with the unchanged early auto plan and actual complete
  pressure hard-pin/selected-range protection. Late/forced/missing evidence
  still fails; Query propagation is explicitly marked unexercised here.
  Regression114 passed. The complete remaining r3 audit was replayed offline
  into `diagnostic-v2-evidence.json` and passed, without reclassifying r3.
  Diagnostic output150.11/s and native peak8 are not a qualified paired gain.
- Fresh natural-agent metadata r4 completed both arms and all history/config/
  retirement/release checks. Fixed300s accepted output baseline176.04/s versus
  candidate163.63/s, both actual native/decode peak8. Output/selection drift on
  126 common turns makes the comparison inconclusive; no >=5% promotion evidence.
  Freeze the verified baseline execution settings for full A/C/D: raw-K0,
  direct seal false, async action metadata false. Preserve failed r1/r3 and
  all original first answers; do not replace them with this new successful run.
  Metadata r4 retains its own frozen source, complete pair and comparison.
- Full A/C/D matrix prepared and independently validated from clean commit
  `5686617279c67468430d891eef3eed71b2a3fed5`: nine jobs,4194 original answers.
  Root `/data/chaidi/kvmem-vllm-experiments/graph-v2-full-20261018-r1`.
  Frozen controller started at1792343676 (2026-10-19 01:14 Asia/Shanghai),
  sequential fresh services with per-arm C8 continuous replacement. NIAH-A
  is running; memory jobs and original scoring remain pending. This is an
  in-progress campaign, not a successful full-quality qualification.
- Original judge availability preflight passed with the already-authorized
  credential: read-only official `/models` returned200 and both original
  `deepseek-v4-pro` / `deepseek-flash`. No benchmark content was sent by this
  probe; credential was entered without echo and never persisted in evidence.
  Metadata-only proof: `artifacts/graph-v2-suite-preflight-20261018/judge-api-probe.json`.
- First formal complete length passed: NIAH-A64K60/60exact,100% macro numeric
  coverage,60/60all needles selected,0output cap hits. Recomputed every score
  using the frozen original function and verified original served input hashes;
  bound all60 first result/response/selection/release proofs in
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-A-64k-first.json`.
  Median complete request85.39s is not decode TPS. A128K continues; no C/D
  comparison or complete4194-answer result is claimed.
- Historical LME output-length audit bound all204 first result hashes:
  attention/query request cap32768, observed maxima4234/4745, no length
  finishes or >=16384 outputs. This is only historical nonbinding evidence;
  the current served-cap difference stays documented and every new result
  still requires its own cap-hit check. Proof at
  `artifacts/graph-v2-suite-preflight-20261018/historical-lme-output-caps.json`.
- Second formal complete length passed: NIAH-A128K60/60exact,100% macro
  numeric coverage,0cap hits; all original served hashes and scores rechecked.
  Complete needle-range coverage is40/60cases, distinct from answer quality.
  Median request183.58s excludes external admission wait/post-response retirement
  and is not decode TPS. Bound first answers/evidence/releases in
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-A-128k-first.json`.
  A256K is running; C/D and both memory suites remain pending.
- Full formal NIAH-A180 passed and cleaned all owned/GPU processes; C started
  automatically. Recomputed all180 original scores/input bindings and verified
  prior64K/128K first proofs unchanged. Exact64K60/60,128K60/60,256K54/60;
  coverage100/100/97.5%; complete-needle-range coverage60/40/33cases;0cap hits.
  Wall4945.79s,original-input5563.83/s,pure-prefill GPU-event6033.41/s including
  replay. Whole output39331/4945.79=7.95/s is not decode TPS. Actual request peak8
  in7steps,decode peak7,363pure-decode and3919mixed steps. Full first proofs:
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-A-all-first.json`.
- Read-only natural-agent phase report validates all complete offered request
  token-ID/timeline/usage bindings and exact window output reconciliation.
  Baseline per-request post-first-emission stream p50/p95=40.89/87.00 tokens/s,
  duration-weighted33.88; candidate40.99/95.63,31.05. Includes mixed interference
  and offered-cohort drain, not isolated GPU decode or aggregate speedup. Only
  three pure-prefill steps in each window; no general short-prefill claim.
  Existing native samples omit per-owner reservation/progress snapshots needed
  for rigorous pure-decode wall cohorts. No proposal-count proxy is substituted.
  Proof: `artifacts/graph-v2-performance-20261018/agent-phase-rate-reading.json`.
- Formal C64K60 completed:60/60exact,100%coverage,60/60full needle intervals,
  60/60actual graph selection changes,0cap hits,median request83.37s. Recomputed
  original scores, verified same original inputs/seeds and all non-graph_mode
  profile fields versus A, and retained A first-answer hashes unchanged. A64K
  is also60/60,so no accuracy gain observed at this length; no speedup claim
  from one median difference. C128K continues; D and memory jobs still pending.
  Proof: `artifacts/graph-v2-suite-preflight-20261018/full-niah-C-64k-first.json`.
- Formal C128K60 completed:60/60exact,100%coverage,0cap hits,60/60actual graph
  selection changes. Complete needle-range coverage A40/60 -> C60/60 while
  answer exact stays60/60. All original scores/input/seed/profile controls and
  retained A hashes verified;median request182.51s is descriptive,not a speedup
  gate. C256K is running;D and both memory suites remain pending. First proofs:
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-C-128k-first.json`.
- Full C180 passed and released all owned/GPU processes;D started. Recomputed
  all original scores,verified A input/seed/non-graph-mode controls,retained
  all A and earlier C64K/128K first-proof hashes unchanged. C256K55/60exact,
  97.9167%coverage,60/60complete needle intervals,0cap hits. Paired versus A:
  two A errors corrected and one new error,net+1;no significance claim.
  C total175/180 versus A174/180;complete-needle coverage C180/180 versus A133/180.
  Wall4978.41s versus A4945.79 (+0.66%),pure-prefill GPU-event6004.64/s versus
  6033.41/s. No stable performance gain/regression inference from one run.
  Complete source-bound proofs:
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-C-all-first.json`.
  D and both memory suites/original external scoring remain pending.
- Formal D64K60 completed:60/60exact,100%coverage,60/60full needle intervals,
  0cap hits. All60pressure events use original cumulative prefill seed and
  execute nonidentity graph propagation,with actual selection changes in all
  cases. C/D profiles differ only in pressure_policy; original inputs/seeds,
  own scores and prior A/C hashes revalidated. Median request82.36s is not
  a stable speedup claim. D128K continues; first source-bound proofs:
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-D-64k-first.json`.
- Formal D128K60 completed:60/60exact,100%coverage,60/60full needle intervals,
  0cap hits. All300pressure events use prefill seed,are nonidentity,and change
  selection;all input/seed/non-pressure-policy controls and A/C hashes verified.
  Median request181.98s is descriptive;A/C/D answer exact stays identical at
  64K/128K. D256K and memory jobs remain pending. First proof:
  `artifacts/graph-v2-suite-preflight-20261018/full-niah-D-128k-first.json`.
- Real output-seed coverage checked in completed natural-agent r4 lifetime:
  baseline129prefill/117output, candidate141prefill/124output graph pressure
  events,all nonidentity and selection-changing. Includes warmup/window/drain
  and restore selections;not a300s rate or per-HTTP count. Existing source
  configuration and full graph-log hashes bound in
  `artifacts/graph-v2-suite-preflight-20261018/real-output-prefill-pressure-seeds.json`.
- Full formal D180 passed and released all owned/GPU processes; MAB-A started
  automatically. Original scorer/input/seed/non-pressure-policy controls and
  retained A/C plus D64K/128K first-proof hashes all revalidated. D256K53/60
  exact,97.0833%coverage,60/60complete needle intervals,0cap hits; paired versus
  C corrects2 and introduces4 errors (net-2),versus A corrects1/introduces2.
  All1140pressure graph events (60/300/780 per length) are nonidentity and
  selection-changing,seeded from cumulative prefill attention. Full wall5031.07s,
  useful input5469.51/s,pure-prefill GPU-event6076.01/s;actual native peak8,
  decode peak7. No stable speedup or statistical significance claim.
  Source-bound proof: `full-niah-D-all-first.json` in the suite preflight dir.
- Formal NIAH matrix now contains all540 first answers. Read-only error audit
  bound all three complete proofs: A/C/D6/5/7 errors,all256K,all exactly one
  omitted number,no extra numbers/unclosed thinking/cap hit. All C/D error
  cases retain complete needle intervals;retention is not answer correctness
  and does not diagnose the cause of the remaining generation errors.
  Evidence: `full-niah-ACD-error-reading.json`. No tuning/reruns use this gold
  analysis. Both memory suites and original judging remain pending.
- MAB-A r1 subsequently failed after69complete answers with5in-flight HTTP500s.
  Authoritative root session88261 exited1; EngineCore stack reports GraphBudget
  exhaustion during metadata installation,not GPU OOM. Owned/GPU processes
  cleaned. Original failed root,all first answers/errors and prior four-case P
  reuse proof remain intact; bound in `full-mab-A-r1-failure.json`.
- Source diagnosis: metadata-only no-op entered full `2*live_bytes` graph staging
  before detecting identical metadata. Frozen-P retained owners and longer MAB
  histories also exceed the cold-profile32MiB per-session allowance. Changed
  metadata-only preparation to a small charged validation reservation,retaining
  span/provenance/transaction validation and identical no-op publication behavior.
  Genuine updates retain full staging checks. Capacity errors now include numeric
  owner/global usage,reservation and limits.
- MAB profiles now provision a conservative8KiB per repository block for retained
  graph owners:256MiB per session/2GiB global for the1M repository and8slots.
  These are caps,not allocations;all arms match,CPU KV100GB and production/NIAH
  defaults unchanged. Profile and source preflight reject insufficient graph caps.
  352targeted CPU tests passed with process exit0,including actual1M graph rows,
  retained P/scoring/preserved snapshots,append and full release. GPU capacity
  qualification remains pending; new runtime requires a fresh frozen attempt,
  not silent retry or stitching r1 NIAH with different-runtime memory answers.
  CPU log: `artifacts/graph-v2-memory-capacity-20261019/cpu-regression.log`.
- Fresh full r2 prepared/validated from clean commit `fbfee4b26e9f6d137acfeb68fb8c3b0ba46d0c74`
  at `/data/chaidi/kvmem-vllm-experiments/graph-v2-full-20261019-r2`,all9runtime
  hashes equal,4194formal answers. Independent controls check verifies unchanged
  original rows/model/all profile fields except declared MAB graph caps.
  Evidence: `artifacts/graph-v2-memory-capacity-20261019/full-r2-controls.json`.
  Root workflow session39878 is live: new frozen structured sync/async checks
  first,then formal MAB A/C/D,then formal NIAH and independent LME A/C/D.
  Sync is currently loading; this is not a GPU pass or a complete quality result.
  Old failed session88261 is terminal and must never be restarted. Workflow
  stops on technical failure and retains attempts;external original judging
  is deliberately a separate pending step after inference completion.
- New-source structured sync/async requalification both passed and cleaned.
  Each2owners/6turns,6Query/10pressure events,4byte-verified restorations with
  M2/P2;all17attention layers including1MTP and96recurrent tensors match.
  Bound all qualification files and verified every shared core/native source
  hash equals formal r2 in `gpu-structured-requalification.json` under the
  memory-capacity artifact directory. This is not C8/long-MAB capacity or
  quality/performance proof. Workflow39878 automatically started formal MAB-A;
  original r1 remains failed and preserved,all4194 r2 answers/judging pending.
- First full MAB-A r2 group passed its source/query/frozen-P isolation audit:
  group007,60questions,P367616;59continuations,no repeated pressure or inherited
  action pages,constant prefill sample/graph source/edge counts. All first case
  files bound in `full-mab-A-r2-first-group.json`. Release proof published before
  CPU credit return,next group admitted18.98ms later while2other requests remain.
  This is CPU credit refill,not GPU resume latency/C8/quality/performance proof.
  Concurrent CPU admission order differs between attempts;passing69completed
  answers is not proof that original544K/maximum757K failure paths were exercised.
- Authorized frozen original judge preflight completed:Flash12/12calibration
  cases passed;two Pro LongMem QA protocol probes passed at thinking enabled/
  max1024. Returned models deepseek-flash/deepseek-v4-pro. Evidence under
  `artifacts/graph-v2-judge-preflight-20261019/r1/`,including raw header-free
  bodies/responses and source hashes. Credential entered hidden,memory-only.
  No formal benchmark scores or summary-JSON judge calibration are claimed;
  all original full judging remains pending after inference.
