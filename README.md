# vllm-kvmem

**Original MeanK integration, 2026-10-13:** `context_mode: "meank"` now uses
model-attention MeanK scoring, the unchanged original KVMem selector, immutable
CPU raw KV and dense re-RoPE assembly with vLLM batched CUDA Graph decode.
Accepted terminal history, partial tails, GDN and DFlash2 state survive GPU-page
release and restore. A single-session 256K history restoration passed real GPU
byte checks; eight independent users completed 32K input plus 16K output each.
With 32K input and 4K output, C1/C4/C8 steady decode measured
63.17 / 232.17 / 378.42 aggregate tok/s. The older ~592 tok/s result has not
been reproduced by the MeanK path.
This path currently uses DFlash2; the older full-context MTP results below apply
to `context_mode: "full"`. See [implementation and limits](docs/meank-integration-20261013.md)
and the [MeanK throughput and restoration evidence](docs/meank-throughput-20261013.md).
The [C8 profiling report](docs/meank-profiling-20261013.md) attributes about 85%
of sampled decode GPU activity to target verification and 11% to the DFlash graph;
KV packing/D2H is small in this workload. Instrumented timing is not a new benchmark.
The [KVMem overhead report](docs/kvmem-overhead-20261013.md) compares the same
DFlash configuration against native vLLM and measures CPU history/staging changes
adapted from qw3. At C8 with 64K input and 4K output, cumulative assembly fell
from 55.21 to 21.05 seconds, TTFT p95 from 241.45 to 187.36 seconds, and whole-batch
output throughput rose from 99.68 to 120.71 tok/s. Decode results vary with draft
acceptance; these measurements do not establish a consistent decode speedup.

**Elastic MTP preview, 2026-10-13:** `vllm-kvmem elastic-serve` adds authenticated,
versioned full-history turns with native NVFP4/FP8/MTP CUDA Graph execution,
CPU KV offload, byte admission, bounded restores, TTL and explicit selected contexts.
At 32 clients, full-context CPU recovery measured 546.28 generated tok/s over
the whole batch; an explicitly selected context measured 986.01 tok/s. Native
32K/C32 steady decode with offload enabled measured 1171.41 tok/s. See
[setup and API](docs/elastic-serving.md) and
[measurements and qualification limits](docs/elastic-throughput-20261013.md).

**Native MTP comparison, 2026-10-13:** with KVMem disabled, NVFP4 and FP8
attention KV for both target and MTP reached 1163.00 / 1178.63 aggregate steady
decode tok/s at 32K input and 32 clients for k=2 / k=3. Whole-batch throughput,
including prefill, was 246.31 / 245.55 tok/s. This is a native vLLM baseline,
not KVMem multi-user qualification. See the [configuration, exact-token evidence
and output limits](docs/native-vllm-mtp-throughput-20261013.md).

**NVFP4 preview, 2026-10-13:** the launcher also supports the NVIDIA
Qwen3.8-27B ModelOpt NVFP4/FP8 mixed checkpoint with the same BF16 DFlash2 and
multi-user isolation. All 84 baseline requests completed, including 32K input
with 32 clients and a 134K refresh case. Native CUTLASS backend tuning then
completed 102 requests: short C8 steady decode measured 386.54–392.75 tok/s,
and short C32 whole-batch throughput reached 712.80 tok/s (18 active owners).
Four independent users also passed natural-stop 32K document retrieval.
See the [NVFP4 measurements, output
quality limits and backend tuning](docs/nvfp4-throughput-20261013.md).

**Long context and 32 clients, 2026-10-13:** 32K-input steady decode measured
38.17–47.22 / 121.00 / 154.74 aggregate output tok/s at 1 / 4 / 8 clients.
All 32 long-input clients completed, with capacity queueing: 19.63 tok/s over
the whole batch, not 32 simultaneous decode streams. See the
[long-context results and measurement limits](docs/long-context-throughput-20261013.md).

**Decode scaling, 2026-10-13:** the existing multi-user Tensor Core server measures
61.01 / 86.27 / 160.57 / 269.50 aggregate output tok/s at 1 / 2 / 4 / 8 concurrent
requests in shared steady decode windows. See the [method, evidence and scope](docs/decode-throughput-20261013.md).

**Multi-user preview, 2026-10-13 (`0.1.0.dev1`):** the public launcher now supports
per-user API keys, server-owned session namespaces, bounded fair request admission,
and streaming cancellation cleanup. Session overflow retains serialization, and
serving diagnostics retain only the latest 256 completed requests. See
[multi-user setup](docs/multi-user-serving.md),
[concurrency qualification](docs/multi-user-testing.md), and the
[2026-10-13 actual-model results](docs/status-20261013.md). Both native Q8 and the
opt-in Tensor Core path completed the 32-user short-request sweep and overload,
fairness, cancellation and recovery checks. At 32 users the measured whole-window
output rates were 49.65 and 130.15 tok/s; only 8 requests are admitted at once.
Tensor Core remains disabled by default, and production task-quality and sustained
long-context acceptance remain incomplete. The model/runtime support boundaries
below still apply.

**Current status / 当前进度：** [2026-10-13 multi-user results and remaining
work](docs/status-20261013.md), with [2026-09-29 implementation and historical
results](docs/status-20260929.md) retained for context. The latest status page
supersedes corresponding pending-test statements below. This is an experimental
development snapshot, not a production release. The historical session agent run
was interrupted after its backend terminated; it is not a completed
quality/performance result.

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. A corrected long-C4 lifecycle run now passes; matched
performance and executed task-quality acceptance remain pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

Work in progress: bounded long-context KVMem for native vLLM batched execution.
This repository currently contains the extracted host core, its Python bindings,
request transactions, async NVMe storage, GPU transfer/FP8 codecs, recurrent
checkpoints, MeanK scoring and fused raw-Q/K capture. Versioned vLLM hooks qualify
the draft loader, scheduler/worker identity refresh, and continuous target capture
with native DFlash rejection counts. Raw K and FP8 V use bounded pinned transfers,
mutable partial tails and immutable complete archive records. Actual native GDN
identity restore/replay is qualified. A long compressed replay completed byte
checks, but its recurrent metadata classification required a subsequent correction;
corrected bounded target serving has now passed one real 197K-history C1 lifecycle
check and subsequent short C4. Native long C4 also passes lifecycle checks, with
severe prefill/retirement interference. DFlash2 compressed prefill plus decode
now passes a corrected long C4 lifecycle check. Exact terminal archive clipping
and async retirement pass short C1/C4 with full archive checksums. An opt-in Q8
integer Tensor Core adapter improves a short fixed-output C4 test from 88.20 to
173.11 aggregate arrival tok/s (1.96x); output streams differ, so this is not a
task-quality acceptance. On four frozen long histories, observed whole-window
arrival decode rises from 10.44 to 18.62 tok/s and service time falls from
1862.80 to 761.16 seconds. These runs retain Q8_0/FP8 and K131072/G65536;
mixed prefill/refresh blocking remains severe and changed outputs still need
executed task-quality checks. Short and frozen 134K prompt-session restore now match exact greedy token IDs.
A four-session HTTP lifecycle gate passes isolation, same-session serialization,
cancellation/recovery and idle reclamation. Concurrent output streams differ from
serial controls, so task quality remains under test. Controlled long-context repeats,
decode-time refresh and native capacity pressure remain pending.
**Production acceptance is incomplete. Installing the plugin is passive; enabling
experimental KVMem requires an explicit serving profile or qualification launcher.** See [development milestones](docs/development.md) and
[qualification boundaries](docs/support.md).

## Host core development

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest tests/test_core.py
cmake -S . -B build/core -DCMAKE_BUILD_TYPE=Debug
cmake --build build/core -j2
ctest --test-dir build/core --output-on-failure
```

Linux and a C++17 compiler are required. The host core does not require torch or
CUDA. Source provenance and the original tests are preserved under `third_party`.

## Engine qualification

Pinned vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0 and GGUF plugin revision
`e2b8ad532b8b5ea175100202c30430c1d2b5e6a8`. Q8_0 is supplied by that independent
vLLM quantization plugin. Model weights are external; never included in this repo.

For a GGUF target with a safetensors DFlash draft, the pinned V2 runner needs the
explicit draft load configuration fix. Opt in using:

```sh
python -m vllm_kvmem --draft-load-compat vllm serve ...
```

This performs a source-hash-checked import adaptation of the loader only, leaves
installed vLLM files unchanged, and propagates through the plugin into workers.
It is not an unmodified upstream reference. See `src/vllm_kvmem/compat.py` for
upstream provenance. No performance or KVMem functionality is implied by this fix.

GPU setup on this CUDA 13 / SM120 machine: `bash scripts/setup_gpu.sh`.
Explicit GPU component tests: `KVMEM_RUN_GPU_TESTS=1 .venv/bin/python -m pytest tests/test_codec_gpu.py tests/test_capture_gpu.py`.
Run only when the GPU is available; this command is not a model performance test.
See [qualification results and limitations](docs/qualification-20260928.md).
The opt-in [exact prompt-session path](docs/prompt-sessions.md) has passed short/134K
actual-model restore and C4 lifecycle gates. Official session-enabled agent tests
are running; no final quality or long-task throughput acceptance is claimed.

## Experimental profile launcher

[Installation and API usage](docs/installation.md) describe an entry that does not
require a prior benchmark artifact. Edit the external model paths in
`examples/qwen-q8-fp8-dflash.json`, then inspect a launch with:

```sh
.venv/bin/python -m vllm_kvmem serve \
  --config local-profile.json --run-dir runs/session-01 --dry-run
```

Omit `--dry-run` to start native vLLM. The public profile launcher and first-request
automatic arming passed an actual-model GPU startup check on 2026-10-13. The
multi-user entry also passed short-request load, isolation, overload and recovery
checks; the final component/HTTP suite passed 386 tests. See the
[current results and limits](docs/status-20261013.md), including exact 134K session
continuation. Production task-quality and sustained long-context acceptance
remain incomplete; no remote package release has been published.
