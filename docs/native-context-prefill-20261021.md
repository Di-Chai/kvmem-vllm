# Native 32K prefill and the KVMem throughput gap

Update: the missing identical-32K-input comparison is now complete. With 2K
chunks and batch16424 in both arms, native reaches 9,593 token/s and KVMem
6,928 token/s (-27.8%), with no pressure or Query selection. See
[the matched 32K report](matched-32k-prefill-20261021.md). The historical 256K
KVMem observations below are a different workload.

Native vLLM processed the complete 32,768-token input at **10,552 token/s**
with a 16,384-token batch cap. Limiting native prefill to 2,048 tokens per step
reduced the measured rate to **9,598 token/s**, a **9.05%** decrease. Small chunks
alone therefore do not explain KVMem's previously measured 3,845 token/s on a
complete 262,045-token input. The attention domains also differ substantially,
and KVMem performs continuous statistics, history publication and restoration.

No production defaults or model kernels were changed. DSH remains stopped at
the user's request. Full RULER and MAB remain paused.

## Controlled native measurements

Same RTX PRO 6000 Blackwell Server Edition, local Qwen3.8-27B NVFP4 model,
FP8 KV, native MTP4, compiled execution, one active request. Model maximum length
remains 524,288, engine batch budget 16,384, max sequences eight, GPU fraction
.72. Prefix caching and asynchronous scheduling are disabled. The installed
vLLM has no KVMem scheduler, connector, middleware or plugin in these runs.
Other configuration is documented in `prefill-chunk-native-20261021.md`.

The 32K prompt is derived from the same frozen 256K RULER input by removing
irrelevant filler. Its first/last 256 tokens, complete needle, question and
assistant suffix are preserved. The needle remains at approximately 70.89%
depth. The exact source slices are in the manifest. SHA-256:
`fc74290ef4a52d923272ec24f19dc4ff10788e909ef0137d33385b2d97a409cd`.
This is a performance fixture, not an additional official RULER quality cohort.

The 16K-cap server ran warmup 32K, measured 32K, full 256K, measured 32K in that
order. The 2K-cap server ran warmup 32K and two measured 32K requests. Both
requests in each measured pair produced identical output token IDs. The
2K-cap launch changes only `--long-prefill-token-threshold` from 0 to 2048;
the engine batch budget, compiler budget and other model arguments stay fixed.
These are different server processes, not an interleaved same-process chunk test.

| Native request | Input tokens | Prefill cap | TTFT, seconds | Input / TTFT, token/s | Output tokens |
|---|---:|---:|---:|---:|---:|
| 32K before the long request | 32,768 | 16,384 | 3.0962 | 10,583.45 | 147 |
| Complete 256K, same server | 262,045 | 16,384 | 82.8788 | 3,161.79 | 156 |
| 32K after the long request | 32,768 | 16,384 | 3.1145 | 10,521.01 | 147 |
| 32K, smaller chunk, repeat 1 | 32,768 | 2,048 | 3.4167 | 9,590.47 | 96 |
| 32K, smaller chunk, repeat 2 | 32,768 | 2,048 | 3.4116 | 9,604.84 | 96 |

Pooled measured 32K rates are 10,552.14 and 9,597.65 token/s. Mean TTFT changes
from 3.10534 to 3.41417 seconds (+9.94%). Startup preparation was 64.118 and
67.108 seconds respectively and is excluded. Warmup requests are also excluded.
All requests naturally stopped with complete token accounting, complete thinking
and the correct needle answer. Output trajectories differ between chunk sizes;
these short outputs do not support a decode-throughput comparison.

The server's measured prefill times were 3.059/3.081 seconds for the measured
16K-cap requests and 3.370/3.368 seconds for the 2K-cap requests. The long request
reported 82.717 seconds. Queue time was below 0.03 ms and prefix-cache query/hit
counters stayed zero. There were no untuned GEMM shape warnings. Each server
reported four inference JIT warnings during its excluded warmup request.
Two short repeats establish the observed difference, not a broad confidence interval.

## Why the windows are not equivalent

A complete 32K causal prompt starts with almost no history and ends at 32K.
Its mean visible key count is `(32768 + 1) / 2 = 16,384.5` per original token.

KVMem's K=36,864 is the retained-history budget after selection. With G=16,384,
new input grows the active window toward 53,248 positions. In this actual run,
the first pressure boundary was 53,184 and subsequent boundaries were spaced
16,320 tokens apart; each retained exactly 36,864 tokens. The active context is
therefore not a constant 36K window, nor a fresh 0-to-32K prompt.

Reconstructing the original input's causal visibility from those recorded
boundaries gives **41,200.66 visible keys per token**, approximately **2.515x**
the native 32K prompt's average. This covers the original first 262,016 tokens,
excluding private/replay work and the final 29 query/template tokens. For a
span of n new tokens after retaining K tokens, the pair count is
`n*K + n*(n+1)/2`. This is a policy-based work estimate, not measured GPU time.

The native full 256K prompt has about eight times as many input tokens and
64 times as many causal pairs as the 32K prompt. Its measured TTFT is about
26.7 times longer. The model contains 16 full-attention and 48 linear-attention
layers, plus projections/FFNs: causal-pair ratios must not be applied as whole
model speed ratios. NVFP4 weight quantization does not remove the full-attention
context-length cost. Native 256K is also not a matched-window control for KVMem.

## What the KVMem evidence establishes

The existing complete-256K diagnostic remains:

| KVMem ordinary chunk | Client TTFT | Input / TTFT | Pure-prefill model interval | Pressure waits | Final Query wait |
|---|---:|---:|---:|---:|---:|
| 1,024 | 82.672 s | 3,169.68/s | 53.649 s | 4.407 s, 13 refreshes | 1.404 s |
| 2,048 | 68.158 s | 3,844.65/s | 45.084 s | 4.599 s, 13 refreshes | 1.340 s |

The model interval includes statistics/capture and submission gaps, not just
model kernels; it excludes final steps classified with decode. Do not subtract
it from client TTFT and label the remainder as archive time or GPU idle.

In the 2K run, the recorded pure-prefill host timeline spans 65.510 seconds:
43.976 seconds within model submission/capture calls, .453 seconds of measured
maintenance, and **21.081 seconds between those calls**. These three host
intervals partition that timeline. CUDA work may overlap host gaps. The CUDA
model interval and refresh waits overlap this timeline and cannot be added to it.

Additional observed work includes:

- Approximately **9.126 GB** of archived KV payload in 145 seal batches,
  with zero SSD reads/writes. CPU pack/hash/record-copy timers are
  6.189/5.679/6.859 seconds; they overlap other work and are not additive wall loss.
- 387 fused sampled-attention scoring calls, 128 native prefill score commits,
  and continuous graph preparation/publication. The score-commit CPU timer is
  .414 seconds and graph prepare/commit timers are .596/.006 seconds.
- Action capture's host timer is 21.270 seconds, but includes waiting for GPU
  work. The synchronous metadata path creates four small device tensors after
  the public forward. The earlier profiler observed pageable-copy synchronization;
  the time blocked on preceding GPU computation is not pure scoring overhead.
- Pressure suspension-to-ready averages .354 seconds. Its complete 4.599-second
  sum, even before accounting for overlap, cannot explain the entire difference
  from a hypothetical 10K+ input rate. Final Query adds a separate 1.340-second
  owner interval.

The earlier instrumented eight-request trace found native FlashInfer prefill
at 6.630 of 11.027 seconds of summed kernels, versus .328 seconds in fused
segment-score statistics. It supports attention as a hotspot, but is a different
cohort and is not a precise attribution of this single-request gap.

Thus the investigation identifies a larger average attention domain, a measured
native chunk penalty, and substantial history/host processing work. It has not
isolated the net cost of each KVMem subsystem under identical attention geometry.
An exact remaining-overhead figure requires that controlled measurement; neither
`9598 - 3845` nor all host gaps can be assigned to a single subsystem.

The next useful optimization targets are history packing/publication and
host-to-GPU submission continuity, followed by larger physical forwards while
preserving logical 1K scoring transactions. Simply changing KVMem's chunk
changes its score averaging/decay and selected pages, as the earlier audit showed;
the default remains 1K. Reducing selection count alone is not the main answer.

## Evidence and validation

- `artifacts/native-context-32k-20261021-r1/`: frozen input derivation, native
  launches, warmup/32K/256K/32K requests and complete server cleanup.
- `artifacts/native-context-32k-chunk2k-20261021-r1/`: native 2K-cap run,
  `comparison.json`, and the frozen analysis script. The comparison verifies
  runtime hashes, complete input/output counts, identical fixture/sampling,
  the sole native model-argument difference, metric deltas, empty prefix caches,
  stable within-pair outputs, completed requests and owned-process cleanup.
- `artifacts/prefill-chunk-native-20261021-r1/kvmem-2048/` and
  `artifacts/pressure-gen-single-20261021-r1/g16/`: existing KVMem measurements.
- Fourteen native diagnostic tests passed, including strict launch overrides,
  frozen suite identity, idle counters, and prevention of an unintended DSH
  restart when no existing service is present.

Both native servers were cleaned up; final GPU ownership is empty. The diagnostic
controller now supports an already-stopped DSH service explicitly without starting
one. No model-service restart or production-default change was performed afterward.
