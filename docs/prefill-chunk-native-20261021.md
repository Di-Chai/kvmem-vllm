# Single-request 256K prefill: larger chunks and native vLLM

Follow-up: `native-context-prefill-20261021.md` records warmed native 32K/256K
requests and a controlled native 2K chunk diagnostic. The subsequent experiments
leave DSH stopped at the user's request; the restoration described below belongs
to this earlier experiment only.

The 2,048-token KVMem chunk improved observed input/TTFT throughput by **21.29%**
over the 1,024-token baseline. It is **not a score-preserving configuration change**:
the sampled positions stayed identical, but chunk-dependent score weights and all
14 observed page selections changed. The default remains 1,024.

Unmodified installed vLLM processed the same complete input at approximately
3.0–3.1K token/s. With a native batch cap of about 16K, these full-context runs
did not reach a 10K+ input rate. This native comparison
does not isolate KVMem overhead: native attention retains the full 256K context,
whereas KVMem attends to a bounded active window.

## Workload and measurement

- One RTX PRO 6000 Blackwell Server Edition, one active request, TP1.
- Same local `nvidia/Qwen3.8-27B-NVFP4` weights/tokenizer, vLLM 0.30.0,
  FP8 KV, compiled full/piecewise CUDA graphs, native MTP with four draft tokens.
- Same frozen RULER `niah_single_1-0000` input: **262,045 tokens**, SHA-256
  `ed7bbc143b62c931d0d9ddf79e4101dafdb7361f65ef18cf3065a5ee793e0778`.
- Same sampling: seed 42, temperature 1, top-p .95, top-k 20, output cap 8,192,
  thinking cap 4,096. Prefix caching and asynchronous scheduling are disabled.
- KVMem: K=36,864, G=16,384, sink=1,312, recent=0, 100 GB CPU history,
  no SSD history, accumulated-attention pressure, Query multi-graph V2 alpha=.2,
  guided disabled. Four established scoring/graph optimizations enabled;
  native QK-score reuse and asynchronous action metadata disabled.
- Engine batch budget 16,424, max sequences 8, GPU memory fraction .72,
  max model length 524,288. Only one complete input was admitted in each run.
- Main rate is **original input tokens / client time to first output token**.
  It includes input submission, prefill, statistics, archive, refreshes and first
  output. Model startup/compilation/autotuning before health readiness is separate.
  It is neither a pure-kernel rate nor output-token throughput.

Native launches remove the KVMem scheduler, connector and middleware, disable
plugins, and remove inherited KVMem/import-path overrides. All other reference
arguments remain identical except the test port and the explicitly named 16,384
batch variant. All 2,681 installed vLLM Python source files with SHA-256 entries
matched their installation RECORD; no installed vLLM source was edited for this test.

## Measured results

| Run | Ordinary chunk / native batch cap | Client TTFT | Input / TTFT | Startup preparation | Output tokens | Case score |
|---|---:|---:|---:|---:|---:|---:|
| KVMem reference | 1,024 / 16,424 | 82.672 s | 3,169.68 tok/s | 75.149 s | 111 | 100 |
| KVMem larger chunk | 2,048 / 16,424 | 68.158 s | 3,844.65 tok/s | 79.151 s | 94 | 100 |
| Native first launch | 16,424 cap | 88.170 s | 2,972.06 tok/s | 204.360 s | 95 | 100 |
| Native compiler-cache-populated restart | 16,424 cap | 83.954 s | 3,121.31 tok/s | 65.128 s | 95 | 100 |
| Native aligned batch diagnostic | 16,384 cap | 87.078 s | 3,009.31 tok/s | 195.345 s | 156 | 100 |

Each row is one request, with complete input accounting and a natural stop. The
native repeat means a restart with populated compiler caches, not a second request
inside an already-serving process. Inference JIT-monitor warnings still appeared;
these measurements are not a claim of fully warmed steady-state throughput.
The two 16,424 native runs produced identical accepted output IDs. The aligned
native run and the two KVMem runs produced different output trajectories. One
correct needle answer is a smoke test, not a RULER or memory-quality result.

Both 16,424 native runs reported five unique untuned FP8/NVFP4 GEMM shapes at
that token count, falling back to a default tactic. The 16,384 run had **zero**
such warnings, but did not show a throughput gain in this measurement. Different
startup/cache conditions prevent attributing the timing difference solely to the
tactic fallback. These warnings are distinct from KVMem scoring-path fallbacks.

The native server's prefill timing was 87.959 s on the first run and 86.900 s
for the aligned run, consistent with client TTFT. These are request-wall metrics,
not CUDA kernel intervals. Native final MTP counters were 24 draft steps / 96
proposals / 73 accepted proposals for both 16,424 runs (accepted length 4.042),
and 41 / 164 / 114 for 16,384 (accepted length 3.780). Short, different outputs
do not support a controlled decode-throughput conclusion.

## What the larger KVMem chunk changed

| Metric | 1,024 | 2,048 |
|---|---:|---:|
| Committed prefill samples | 8,154 | 8,154 |
| Committed statistics chunks | 256 | 128 |
| Pure-prefill model steps | 271 | 143 |
| Mean tokens in those steps | 963.18 | 1,818.16 |
| Model interval token rate | 4,865.35/s | 5,766.98/s |
| Pressure refreshes | 13 | 13 |
| Sum of this owner's pressure waits | 4.407 s | 4.599 s |
| Query refresh wait | 1.404 s | 1.340 s |
| Fused scoring calls, including output | 774 | 387 |
| Download-pool allocated pinned bytes | 552,960 | 1,097,728 |
| Sampled device memory peak | 70.38 GiB | 70.31 GiB |
| Sampled system used-memory peak | 40.97 GiB | 39.74 GiB |

TTFT fell **17.56%**. The gain did not come from reducing pressure-refresh count
or pressure waiting time. Larger forward batches and fewer submissions/statistics
commits are the useful direction indicated by this experiment, although this pair
does not isolate each source of the gain. The CUDA model interval includes
capture/scoring/submission gaps and excludes final steps classified with decode:
its token counts are 261,021 versus 259,997, not both the complete input.

Both runs archived approximately 9.13 GB of KV payload and reported zero history
disk reads/writes. Archive pack/hash/copy timers overlap other work and must not
be added to the refresh waits to infer total overhead. Native sampled device
memory peaks were 72.33 / 73.01 / 71.83 GiB and system used-memory peaks were
27.68 / 26.08 / 27.73 GiB. All memory observations use five-second sampling;
system used memory is not model RSS, and device readings can miss short peaks.

## Semantic audit and limits

The two KVMem profiles differ only in `prefill_chunk_tokens`. The frozen model
sources are identical; four diagnostic-harness files differ. Sampling layers,
stride 32, interval 1,024, half-life 8,192, scoring precision and graph parameters
were retained. Reconstructing accepted sample positions from chunk end/row count
and the frozen sampler gives **exactly the same 8,154 logical positions**.
Their digest is `f014f16c2be1a8daf9d4582a6fe0ac005c3a293b10334cd9001a5542c44adbcd`.

All 13 pressure boundaries and the final Query boundary also match. Nevertheless,
all 14 selected page sets differ; mean page-set Jaccard is **0.8119**, and the
Query selection Jaccard is **0.6528**. KVMem policy/execution/release audits passed:
pressure used accumulated prefill attention without graph propagation, Query used
native V2 multi-graph propagation, fused scoring did not fall back to reference,
and the bounded download pool had no exhaustion/quarantine.

`PrefillHistory.accept` averages mass/exposure per **chunk**, decays previous
state at each chunk end, and retains only the observed window. Consequently,
equal sample positions do not imply equal score weights. A fixed-mass CPU
counterexample using the unchanged implementation and half-life 8,192 gives
page scores `(0.482682, 0.517318)` after two 1,024-token commits versus
`(0.5, 0.5)` after one 2,048-token commit. This is an arithmetic-policy difference,
not rounding noise. Other numerical effects of batching can also affect the
actual model run; this example does not attribute every selection change.

The existing ordinary-chunk guard limits sampling to 64 rows. With stride 32,
2,048 is the largest allowed chunk. This test did not weaken that guard or
increase the stride to run 4K/8K. A semantics-preserving follow-up must keep the
logical 1K statistics transactions, decay/exposure handling and selection
boundaries while changing physical forward batching, then verify scores/pages
and broader quality before changing defaults.

## Evidence and validation

Artifacts: `artifacts/prefill-chunk-native-20261021-r1/`, with `kvmem-2048/`,
`native-vllm/`, `native-vllm-warm/`, `native-vllm-16384/` and `comparison.json`.
The unchanged 1K reference is `artifacts/pressure-gen-single-20261021-r1/g16/`.

Harness changes explicitly gate nonstandard chunks to diagnostic cohorts;
ordinary full-suite validation still expects the original 1K configuration.
The new native controller owns its service lifecycle, records the actual launch,
verifies complete input/output accounting, and restores the recorded DSH backend.
The comparison checks frozen model identity, fixture/sampling identity, actual
launch policy, input hashes, accepted sample positions, refresh boundaries,
selected pages, execution audits and service cleanup/restoration.

Validation: 56 existing RULER/profiling/prefill tests plus 9 native-launch and
environment/idle-state guard tests passed. This turn changes diagnostic tooling
and documentation only; no production scoring/attention implementation or default
was changed. The full RULER campaign and MAB remain paused.

All diagnostic-owned processes were cleaned up. The original DSH model and
bridge both returned HTTP 200 after restoration, with zero running/waiting
requests; production remains chunk=1,024 and G=16,384. See
`final-service-health.json` in the experiment directory.
