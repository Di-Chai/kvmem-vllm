# Initial qualification, 2026-09-28

These results qualify components and dependencies, not end-to-end KVMem serving.

Environment: RTX PRO 6000 Blackwell Server Edition, Python 3.12.13,
vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0,
GGUF plugin e2b8ad532b8b5ea175100202c30430c1d2b5e6a8.
Exact Python dependencies: `requirements-cu130.lock`.

## Completed checks

- Current combined suite: 57/57 Python CPU/GPU tests pass with explicit GPU opt-in
  (42 CPU/control tests and 15 GPU tests).
- Original host-core CTest: 5/5 pass, Debug build (assertions enabled).
- Python host tests cover bindings, speculative accepted-prefix publication,
  completion fences, request cancellation, immutable FP8 remap planning, stale
  generations, namespaces, coalesced NVMe I/O, bounded checkpoints and MeanK math.
- GPU codec/transfer: 5/5 pass. Covers immutable-source reselection, FP8 error,
  three explicit physical cache views and pinned asynchronous copies.
- Fused capture: 3/3 pass. Original Q/K/gate tensors exactly equal the upstream
  kernel at three shapes, including the 256-dimensional Qwen gated-attention path.
- Fused immutable-K assembly: 3/3 GPU cases pass for token-major, head-major and
  native interleaved physical strides. K/V bytes match vLLM's native FP8 writer for
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
admission. It is now inserted into the explicitly enabled diagnostic scheduler
described below; automatic compressed serving remains unimplemented.

## Scheduler/worker transaction qualification

`engine_scheduled_rebuild_02` passed with the native scheduler subclass and public
KVConnector metadata channel. The first real prefill request was suspended in
the scheduler, rebuilt by a command-only worker step, acknowledged, and resumed
with complete native block-table rows. All 16 layers' K/V bytes were unchanged;
the 128-token continuation exactly matched `engine_baseline_01`. Four subsequent
requests each produced 128 tokens successfully. This is not a simultaneous
four-request refresh test.

The diagnostic took 6.237 seconds for the first single request and 6.626 seconds
for the subsequent 512-token concurrent group. These include JIT/cold effects and
blocking diagnostic archive operations: they are **not warmed decode throughput**.
Observed device memory after generation was 71446 MiB with the configured 72%
native pool. The single archive record was 6 MiB with one write and one read syscall.

`engine_scheduled_rebuild_01` failed during warmup because native vLLM temporarily
disables its connector there. Binding the runtime only after HTTP readiness fixed
that lifecycle error. Failed logs are retained alongside the successful run.

Ten added connector/scheduler tests cover delayed completion, cancellation before
and after dispatch, out-of-order per-request acknowledgments, stale commands,
multi-group prepare rollback and pre-commit validation, protected pending pages,
and inclusion of new allocations in the block tables published on resume.
They use real connector API types and native FullAttentionManager/BlockPool.
Cancellation and capacity pressure still need end-to-end GPU qualification.

Reproduce the live identity diagnostic on an idle GPU:

```sh
.venv/bin/python scripts/qualify_engine.py --models /path/to/models \
  --output artifacts/engine_scheduled_rebuild_repeat --scheduled-rebuild-probe
```

This opt-in path requires eager TP=1, synchronous target-only execution and APC
off. It does not enable compressed retrieval or DFlash refresh semantics.

## Dense working-window metadata and slot mapping

`WindowBatchAdapter` now translates new-token positions and full-attention sequence
lengths after a committed selection, while leaving native logical counters and
recurrent metadata unchanged. Tests use real GPU `BlockTables` with different
allocator/kernel block sizes, two request offsets, reversed batch order, mixed
query lengths and padding. Each request's slots match its own dense window and
physical pages; recurrent slot-mapping rows remain disabled. A growing selection
may legitimately reduce the history-to-window offset; the adapter validates
monotonic logical history instead of incorrectly requiring a monotonic offset.

The instance-local model-state bridge and source-hash-checked metadata hook are
now enabled in the scheduled diagnostic. `engine_window_bridge_01` passed: all
16 layers byte-equal after identity refresh, 128-token continuation exactly equal
to baseline, and four subsequent 128-token requests completed. This full-model run
has zero compression offset; nonzero offsets are qualified at the GPU mapping
and native metadata levels only. It is not compressed-serving validation.

The original configuration used provisional sink=1/recent=0 defaults. This was
corrected by binding qw3's unchanged `resolve_kvmem_keep_allocation` function.
Mainline K=131072, block=128 now resolves to 11 sink blocks and 82 recent blocks
(1408 and 10496 tokens). Small mechanical tests specify explicit bands; automatic
bands are never silently shrunk to make a small budget pass. This correction was
made after `engine_window_bridge_01`; core regression tests verify both defaults
and explicit overrides. Earlier identity tests kept all tokens, so this does not
change their selection, but they do not establish long-history retrieval quality.

## Continuous accepted-prefix authority

`RequestAuthority` now extends an unsealed CPU tail across decode steps, archives
only full immutable records, and builds MeanK entries from native verifier-accepted
rows. Native GPU history positions and query offsets travel with rejection counts
in the same packed D2H packet; Python dictionary order is not used for slicing.
Completed D2H slots cannot be reused until their CPU consumer explicitly releases
them. The archive is shared and namespaced; per-request staging and global archive
capacities are bounded. A MeanK bridge feeds accepted-key scores into the original
core selector, but live compressed assembly/replay is still pending.

| Run | Outcome | Native accepted KV rows | Rejected rows excluded | Sealed records verified |
|---|---|---:|---:|---:|
| engine_archive_01 | Passed, target only | 719 | 0 | 5 |
| engine_archive_dflash_01 | Passed, DFlash | 747 | 713 | 5 |
| engine_archive_dflash_chunked_02 | Passed, chunked prefill + DFlash | 18036 | 474 | 139 |

Each run contains one single request followed by four concurrent requests. All
sealed records were read back from NVMe and their complete raw-K/FP8-V SHA256
matched the accepted capture. Every run ended with zero active authorities,
archive slots and pending downloads. The two short runs' single-request output
IDs exactly match their corresponding earlier native target/DFlash baselines.
This does not resolve the separate baseline-versus-DFlash token divergence.

The chunked run uses 3496–3506 prompt tokens and the native 1024-token batch budget.
Four outputs reached 128 tokens; the fifth naturally stopped after one token.
It processed 79 model batches. This synthetic repository-like history is a
lifecycle diagnostic, not a real SWE quality task or K128K/G64K performance test.
Cold end-to-end timings include diagnostic packing, synchronization, archive
readback and checksums and must not be presented as warmed decode throughput.

`engine_archive_dflash_chunked_01` finalized its archive successfully but failed
the harness's initial output-extent check. That check incorrectly assumed at
most one speculative block could run past termination. Native V2 asynchronous
scheduling allows two in-flight batches. The corrected harness reads actual
`max_concurrent_batches=2` and `decode_query_len=8`, matches each HTTP request to
its authority, and verifies its final native packet. The repeat above passed.
These extra native-accepted rows are retired when the request ends; persistent
sessions still require a scheduler-provided terminal prefix and recurrent replay.

Additional tests cover partial-tail sealing, exclusion of rejected rows from the
MeanK index, namespace isolation, real I/O lease retirement, shared-queue pressure
from another request, and score/selection/append continuity. They found a pybind
lifetime issue: retaining a field view kept the whole Append/Store alive. Index
records now own copied generation handles, allowing all repository leases to drain.

The native DFlash baseline and archive runs both print the same one-semaphore
resource-tracker warning at process shutdown. This was not introduced by the
archive path; it remains noted for final deployment qualification.

Reproduce the chunked lifecycle check:

```sh
.venv/bin/python scripts/qualify_engine.py --models /path/to/models \
  --output artifacts/engine_archive_chunked_repeat --archive-probe --dflash \
  --history-repeat 128
```

## Native recurrent checkpoint and replay

`NativeRecurrentState` borrows the actual convolution and SSM tensor rows for all
48 GDN layers, including every allocated speculative row and the native accepted
token counter. It validates request incarnations, state-slot indices and page IDs
before restoring. This first adapter requires APC off, Mamba cache mode `none`,
and a single worker rank. It rejects unsupported state layouts explicitly.

| Run | Initial prefill, decode, four-request forward | State bytes per single / four-request snapshot |
|---|---|---:|
| engine_recurrent_replay_01 | Output and all recurrent state byte-equal | 153944068 / 615776272 |
| engine_recurrent_replay_dflash_01 | Output and all recurrent state byte-equal | 1286602756 / 5146411024 |

Both runs also passed continuous archive verification and reclaimed all archive
resources. Their 128-token single-request outputs exactly match the corresponding
target-only and DFlash baseline outputs. These identity replays prove restoration
of actual native state; they do not prove compression quality or DFlash refresh.
The draft run includes all allocated speculation state rows: snapshotting them
has a substantial memory and kernel-launch cost that still needs optimization.

`NativeWindowAssembler` now reads selected immutable records in bounded archive
batches, uploads them through two packed staging slots and rebuilds native FP8
pages at dense positions. GPU tests force noncontiguous selection, moved blocks,
multiple chunks and an unchanged sink. Every resulting K/V byte matches an
independent RoPE plus native CUDA cache-write reference. Unselected physical
storage remains unchanged. These are small mechanical tests, not K/G benchmarks.

The complete component suite passes **67 tests**: 48 CPU and 19 explicitly enabled
GPU cases. All five unchanged C++ core tests were rerun and passed. New cases also
cover query-page retention after refresh acknowledgment, H2D destination stream
ordering, and an I/O failure after partial GPU assembly. Reproduce:

```sh
KVMEM_RUN_GPU_TESTS=1 .venv/bin/python -m pytest -q
.venv/bin/python scripts/qualify_engine.py --models /path/to/models \
  --output artifacts/recurrent_replay_repeat --archive-probe --replay-probe --dflash
```

## Real K128K/G64K target compression (2026-09-29)

`engine_compression_k128g64_02` completed its mechanical checks on the same Q8_0/FP8 target.
**Correction (2026-09-29): this run does not qualify preservation of recurrent
metadata.** The worker incorrectly classified all slot-mapped groups as attention;
native `MambaSpec` also inherits `uses_slot_mapping=True`. This included recurrent
groups in the dense-length override. Repeated execution can be byte-equal while
using the same wrong metadata. The observed counts and byte equality below remain
historical measurements, but semantic qualification is withdrawn pending a rerun.
The corrected bridge identifies actual `FullAttentionSpec` groups, including
uniform wrappers, and keeps recurrent length metadata outside window overrides. The single
request contains exactly 198656 synthetic repository-history tokens and generates
128 tokens. At logical history 195584, the original MeanK selector chooses 131072
historical tokens. The assembler rebuilds 115200 tokens in 29 bounded transfer
chunks and skips 15872 stable tokens. The 1024-token query is replayed with dense
positions and attention lengths. Native request counters retained logical history;
the correction above supersedes the earlier assertion about GDN length metadata. A second restored replay has identical output and GDN state bytes.
Only the final pass is sampled and published to the authority.

Four subsequent short requests each generate 128 tokens. Across the long C1 and
short C4 requests, 199361 native KV rows are accepted; 1556 sealed archive records
are read back and checksum-verified. Final active authorities, archive slots and
pending downloads are all zero. The long request's cold elapsed time is 735.49 s;
the final device-memory snapshot is 73336 MiB. Neither is a warmed decode metric
or a measured memory peak. The diagnostic synchronizes replay checks and performs
complete archive readback; short requests can wait behind that retirement work.

This is a mechanical integration result, **not a quality or throughput pass**.
Generated text repeats parts of the synthetic prompt. No same-input uncompressed
quality control has been run, so this cannot be attributed specifically to
compression or dismissed as a prompt-only effect. Real frozen DeepSWE inputs,
matched controls, and task execution are still required. The native allocation
also remains untrimmed; this does not demonstrate bounded GPU serving or long C4.
It uses 128-token records and a capacity-boundary refresh, not the prior qw3
32-token-record semantic-prefill policy. Keep those distinctions in comparisons.

The first attempt was manually stopped after insufficient progress visibility;
that was not evidence of an engine failure. Native NVMe arenas are anonymous,
unlinked files, so directory size is not a valid write-progress signal. The repeat
adds accepted-history and archive-slot counters and completed successfully.
Later harness runs also save a hash manifest of source files with the Git revision.

```sh
.venv/bin/python scripts/qualify_engine.py --models /path/to/models \
  --output artifacts/compression_repeat --compression-probe
```

## Remaining before any success/performance claim

Automatic bounded scheduling and compacted per-request working-set commits;
production archive admission/GC and asynchronous retirement; production query-conditioned refresh and target/draft
rollback; long-history C1/C4 isolation/cancellation/capacity tests; warmed
throughput and latency comparison; actual agent-task quality and independent repeats.
