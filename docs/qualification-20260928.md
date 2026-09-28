# Initial qualification, 2026-09-28

These results qualify components and dependencies, not end-to-end KVMem serving.

Environment: RTX PRO 6000 Blackwell Server Edition, Python 3.12.13,
vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0,
GGUF plugin e2b8ad532b8b5ea175100202c30430c1d2b5e6a8.
Exact Python dependencies: `requirements-cu130.lock`.

## Completed checks

- Current combined suite: 30/30 Python CPU/GPU tests pass with explicit GPU opt-in.
- Original host-core CTest: 5/5 pass, Debug build (assertions enabled).
- Python host tests cover bindings, speculative accepted-prefix publication,
  completion fences, request cancellation, immutable FP8 remap planning, stale
  generations, namespaces, coalesced NVMe I/O, bounded checkpoints and MeanK math.
- GPU codec/transfer: 5/5 pass. Covers immutable-source reselection, FP8 error,
  three explicit physical cache views and pinned asynchronous copies.
- Fused capture: 3/3 pass. Original Q/K/gate tensors exactly equal the upstream
  kernel at three shapes, including the 256-dimensional Qwen gated-attention path.
- Fused immutable-K assembly: 2/2 GPU cases pass for token-major and head-major
  physical strides. K/V bytes exactly match vLLM's native FP8 cache writer for
  the tested inputs; rebuilding moved K leaves resident V unchanged.
- Native vLLM BlockPool lease ownership: pass. Capacity remains unavailable until
  the correct retiring request receives a worker completion; stale acknowledgments
  cannot free a new lease.
- Proposed per-group metadata extension: pass against actual vLLM metadata
  construction. Attention receives compressed lengths; recurrent groups retain
  complete history lengths and native model-specific fields. This hook is not
  enabled in serving yet.
- Built sdist and wheel successfully. A separate environment without torch/vLLM
  installed the wheel and passed all 14 host-only Python tests.

## Engine prerequisite runs

`scripts/qualify_engine.py` starts a localhost-only server and terminates its own
processes. Main target weights are Q8_0 GGUF, target KV FP8 e4m3, activation BF16.
DFlash uses the separate BF16 draft, 7 speculative tokens. The explicit loader
fix is active; this is not an unmodified upstream reference.

| Run | Outcome | First single request | First concurrent group |
|---|---|---:|---:|
| engine_baseline_01 | Generation passed | 128 tokens / 6.205 s | 512 tokens / 6.686 s |
| engine_dflash_01 | Generation passed | 128 tokens / 10.652 s | 512 tokens / 14.978 s |

Both use max context 8192, batch-token limit 1024, max sequences 4, eager execution,
prefix caching off, short raw prompts and a 128-token output cap. The first calls
include JIT and are **not steady-state decode benchmarks**. Startup was 298/316 s.
Post-generation device usage was 71430/71464 MiB; the KV pool deliberately fills
the configured 72% memory budget, so this is not a measurement of DFlash's
incremental memory requirement. Reported model-loading memory: 27.54/31.33 GiB.

The first single-request baseline versus DFlash token stream diverges at zero-based
token 111. This reproduces the earlier GGUF/speculation qualification issue.
Batch-size-dependent quantized GEMM is a hypothesis, not a demonstrated cause.
Fixed-prefix numerical diagnosis and quality qualification remain required.

`diagnose_q8_batch_math.py` isolates the plugin's MMVQ/MMQ dispatch on synthetic
Q8_0 matrices with identical input rows. Across 5120/17408/14336 output widths,
rows=2 matches rows=1 exactly; crossing the dispatch threshold introduces small
differences (largest tested max absolute 0.00048828125, RMS 0.0000076294).
This establishes a source of batch-dependent rounding, but does not establish
that it explains the observed full-model token divergence or rules out state bugs.

The initial runner capture probe caught a native warmup batch. The probe was then
changed to arm after HTTP readiness. `engine_capture_02` captured all 16 target
attention layers for a real 14-token request; all captured values were finite.
Its complete 128-token single-request output exactly matched `engine_baseline_01`.
This verifies target capture, not working-set reselection or long-history quality.

Full logs, commands, output token IDs and capture tensors reside in ignored
`artifacts/`. This document records results without checking model data into Git.

## Native-page archive/rebuild follow-up

`engine_rebuild_01` passed. A real 14-token initial prefill was captured across all
16 target attention layers, registered in a bounded repository, packed into one
6 MiB block record, written/read through AsyncArchive, selected without dropping
tokens, and rebuilt into the original engine-owned FP8 pages. K and V bytes were
identical in every layer. The 128-token continuation exactly matched the original
baseline. The native cache exposes logical BHN(2D) uint8 views; the observed kernel
page size is 32 tokens. A new GPU case covers this interleaved K/V view explicitly.

This is an identity refresh correctness diagnostic. It deliberately synchronizes,
does not compact history, does not exercise query replay, and is not a throughput
result. No end-to-end bounded KVMem serving claim follows from it.

The WorkingSetManager adapter passes two tests using the actual native vLLM
FullAttentionManager/BlockPool. They cover repeated refreshes as logical history
grows beyond K+G, native page return only after a completion fence, unchanged
other-request allocations, stale completion rejection and bounded long-prompt
admission. The adapter is not yet inserted into the serving scheduler.

## Remaining before any success/performance claim

Actual scheduler-owned page leases and per-request working-set commits; immutable
KV archive integration; query-conditioned refresh and DeltaNet replay; target and
draft rollback; long-history C1/C4 isolation/cancellation/capacity tests; warmed
throughput and latency comparison; actual agent-task quality and independent repeats.
