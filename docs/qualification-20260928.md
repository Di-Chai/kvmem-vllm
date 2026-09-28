# Initial qualification, 2026-09-28

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

## Corrected bounded real-history target run (2026-09-29)

`engine_deepswe_effect_bounded_03` passes the scheduled target-only path with the
correct semantic cache-group classification. It uses the exact frozen Effect
DeepSWE input (`request_24.json`, SHA256
`55e734bd7cf8d462e3d44f5ceb6b6b3066b9792f64e495feea857c7b42cf39e2`),
197110 input tokens, Q8_0 target weights, FP8 e4m3 KV, K131072/G65536, 128-token
records, 1024-token batches, greedy sampling and natural EOS. It is not a DFlash
or executed SWE-task qualification.

The worker completed 65 scheduled refreshes: the all-history bootstrap at K and
64 actual reselections, including the final 502-token partial input chunk. Every
worker report matches a native scheduler acknowledgment and the exact retained
page count. Full-attention allocation stays at 85 native 1568-token blocks for
1024-token queries, and 84 for the final 502-token query, instead of following the
197K logical history. All three recurrent groups remain outside the dense-window
metadata override. This establishes the corrected bounded C1 lifecycle; the
older synthetic compressed replay remains subject to its correction above.

The continuation has 77 output tokens and natural EOS. It is contextually coherent
and almost identical to the uncompressed same-input control: the tool command's
separator is `====` instead of `===`. It is not token-identical, and no tools were
executed. The uncompressed control (`engine_deepswe_effect_baseline_01`) completed
in 693.691 s but had no streaming measurement; that elapsed time is not its decode
throughput.

| HTTP measurement | Corrected bounded target result |
| --- | ---: |
| Long C1 TTFT | 1877.601 s |
| Long C1 first-to-last token interval | 3.70886 s |
| Long C1 arrival decode rate, excluding first token | 20.491 tok/s |
| Long C1 HTTP service time | 1881.310 s |
| Subsequent **short** C4 output | 512 tokens total; all four hit the 128-token cap |
| Short C4 aggregate arrival decode | 76.225 tok/s |
| Short C4 request-group service time | 30.418 s |
| Final device memory snapshot | 73616 MiB; not a sampled peak |

The four short requests waited roughly 23.75 s for synchronous archive retirement
before first output. Across all five requests, 197764 accepted KV rows and 1544
sealed records were verified; active authorities, archive slots and pending
transfers ended at zero. SSE final usage agrees with emitted token IDs. These are
diagnostic service observations, not a warmed performance pass or long C4 result.
The long request performs semantic refresh throughout prefill and still pays large
host preparation and storage costs. A 25.00 s worker sample during refresh recorded
6.99 GB of actual disk reads and 24.15 CPU seconds (15.09 system, 9.06 user); it
identifies substantial host work without isolating each phase's elapsed cost.

The next opt-in candidate keeps these workload/precision/K/G settings and combines
archive-layout GPU assembly with a bounded CPU read cache. Native GPU component
checks confirm byte equality for reordered records/pages and failure fencing after
partial assembly. A lifetime GPU-page reservation adapter is now attached to the
bounded scheduler, and public connector extra config carries the options. The
combined component suite passes **102 tests** (78 CPU, 24 GPU). Full-model candidate
qualification and long C4/cancellation/capacity tests remain pending.

## Remaining before any success/performance claim

Native long C4, cancellation and capacity-pressure qualification;
production archive admission/GC and asynchronous retirement; target/draft
refresh and rollback; long-history C1/C4 isolation/cancellation/capacity tests; warmed
throughput and latency comparison; actual agent-task quality and independent repeats.


## Cached archive and packed native assembly — 2026-09-29

`artifacts/engine_deepswe_effect_cached_packed_01` passed: Q8_0 target, FP8 e4m3
KV, K131072/G65536, B128, exact Effect 197110-token input with SHA256
`55e734bd7cf8d462e3d44f5ceb6b6b3066b9792f64e495feea857c7b42cf39e2`,
target-only synchronous execution, 36 GiB global / 9 GiB per-request non-authoritative
CPU archive cache, packed assembly, and attached native lifetime GPU admission.
All 65 worker refreshes matched scheduler commits. C1 emitted the same 77 token
IDs as corrected `engine_deepswe_effect_bounded_03`, including natural EOS;
all four subsequent short completions also matched their baseline token IDs.
All request authorities, archive slots, cache bytes and pending downloads were
released. No generated tool call was executed.

| HTTP metric | Corrected bounded baseline | Cached + packed candidate |
| --- | ---: | ---: |
| C1 TTFT | 1877.601 s | 1280.156 s |
| C1 service duration | 1881.310 s | 1284.064 s |
| C1 arrival decode | 20.491 tok/s | 19.448 tok/s |
| Subsequent short C4 aggregate arrival decode | 76.225 tok/s | 75.613 tok/s |
| Short C4 group service duration | 30.418 s | 22.872 s |
| First short C4 request TTFT | 23.752 s | 16.152 s |
| End-of-run GPU memory snapshot, not peak | 73616 MiB | 73604 MiB |

C1 TTFT improved by 31.82% in this one cold comparison. Decode did not improve;
DFlash was disabled in both. The reduced delay before short C4 also includes
synchronous diagnostic retirement of C1, so it is not a long-C4 performance result.
The candidate still spent 336.225 s in window assembly (median actual refresh
5.125 s), mostly host preparation. A 10 s `/proc/3130323/io` sample late in the
run saw 4096 physical read bytes and 50,335,744 write bytes: cache hits removed
the bulk of NVMe reads in that interval, not the host assembly cost. The sample
alone does not attribute all remaining latency. Cache/packed changes were tested
together; their individual contributions are not isolated.

New harness checks reject duplicate/missing refresh transactions, validate native
page extents, require compression for each sufficiently long input, and require
actual four-request native refresh batch membership for long C4. Added acceptance
policy tests ensure speculative accepted prefixes cannot jump the aligned refresh
boundary and exclude unverified draft rows from reselection. That policy is not
yet wired into bounded DFlash serving. Latest default test run: 83 passed,
24 GPU tests skipped. Long C4 and all remaining gates in `development.md` remain
outstanding; this is not production or final performance qualification.

## Real long-history C4 — 2026-09-29

`artifacts/engine_long_c4_cached_packed_01` passed from commit `776bb7a`.
Q8_0 target / FP8 KV, no DFlash, K131072/G65536, B128, batch budget 1024,
per-request prefill cap 256, CPU read cache 36 GiB, packed assembly, full checksum
retirement. The four exact frozen inputs were request_10 (Quill, 134067 tokens),
request_22 (Effect, 132563), request_36 (DynamoDB, 132233) and request_37
(DynamoDB, 133934), with hashes retained in the run manifest. The per-request cap
changes query chunking relative to C1's 1024 and prevents a clean C1 speedup claim.

All four requests underwent actual compressed refreshes; the worker reports
include real four-request batches, including mixed prefill/decode neighbours.
All 35 worker refreshes matched native scheduler commits and bounded page extents.
652 native batches completed. Archived generations were checksum-verified and
all authorities, archive slots, pending downloads and CPU cache bytes were freed.
Outputs remained on their respective Quill, Effect and DynamoDB topics; no tools
were executed, so this does not establish SWE task success or a per-request
fixed-prefix numerical equivalence result.

| Request | Output tokens | TTFT (s) | Arrival decode (tok/s) | Max burst gap (s) | Finish |
| --- | ---: | ---: | ---: | ---: | --- |
| Quill 10 | 114 | 2067.706 | 11.480 | 3.669 | natural stop |
| Effect 22 | 128 | 1992.587 | 1.214 | 16.283 | length cap |
| DynamoDB 36 | 128 | 1971.384 | 1.105 | 21.184 | length cap |
| DynamoDB 37 | 128 | 2071.359 | 3.729 | 10.912 | length cap |

Total 498 outputs, group service 2105.441 s, cold service 0.2365 tok/s.
Aggregate arrival decode was **3.6856 tok/s** over 134.035 s. After all four first
outputs and before the first request finished, a diagnostic 6.1687 s subinterval
contained 447 arrivals (72.462 tok/s); this excludes the earlier prefill and later
retirement interference and must not replace the full-window metric.

Assembly alone consumed 170.669 s, including 148.786 s host preparation, and
uploaded 164452368384 bytes. Finished requests also synchronously verified their
archives while neighbours were active. The run demonstrates native concurrent
correctness/lifecycle progress and severe scheduling/assembly interference,
**not the final high-throughput goal**. Next priorities are compressed DFlash
qualification, matched query-granularity batching, less blocking preparation,
retirement, and persistent-session reuse.

The end GPU snapshot was 74274 MiB. A monitor attached partway through prefill
observed a maximum owned GPU-process memory of 74248 MiB and a maximum summed
API/worker RSS of 39020924 KiB; no foreign GPU processes or sampling errors were
observed. Its 5 s sampling interval and partial coverage cannot establish the
whole-run instantaneous peak. These memory definitions differ from the end
whole-device snapshot.

## Compressed DFlash2 C1 and short C4 — 2026-09-29

`engine_deepswe_effect_bounded_dflash_01` passed from launch commit `62726f0`:
Q8_0 target / FP8 e4m3 target KV, BF16 DFlash2 draft and draft KV, K131072/G65536,
B128, CPU cache 36 GiB, packed assembly and full checksum retirement. Exact Effect
input request_24 contains 197110 tokens. All 74 worker/scheduler refreshes matched,
native BF16 sliding draft group 14 was preserved, unverified draft rows were
excluded from reselection, and all five requests released their archive/cache
resources. 291 native batches completed. This qualifies compressed prefill
followed by speculative decode, not a decode-time K+G refresh or persistent reuse.

| Observation | Result |
| --- | ---: |
| C1 output | 77 tokens, natural stop |
| C1 TTFT / service | 1271.183 / 1272.072 s |
| C1 arrival decode | 85.509 tok/s, 0.888792 s window |
| Short C4 aggregate arrival decode | 91.956 tok/s, 5.524390 s window |
| Short C4 service | 18.519 s, 512 capped output tokens |
| Max owned GPU-process memory, sampled every 2 s | 74264 MiB |
| Max summed API/worker RSS, sampled | 36896540 KiB |
| End whole-device GPU snapshot | 74288 MiB |

The 77 C1 output IDs exactly match corrected target-only cached/packed C1. All
four short C4 output texts exactly match the earlier uncompressed native DFlash
and archive-DFlash runs. The raw short prompts can produce incomplete/invalid
code; this agreement is a regression/lifecycle check, not executed code or SWE
task quality. The long continuation's tool call was not executed either.

Native draft reservation reduces the requested 1024-token batch to actual
896-token aligned prefill chunks (last chunk 630), versus target C1's 1024.
Despite matching final C1 outputs, do not infer a clean DFlash speedup from these
different selection queries or claim a stable rate from less than one second of
generation. Short C4 is not long C4. The resource monitor had 702 samples, 690
with GPU ownership, no errors or foreign GPU processes; sampled maxima are not
instantaneous peaks. The pre-existing native DFlash one-semaphore shutdown warning
also recurred; this does not establish that process synchronization cleanup is fixed.

Assembly totalled 341.630 s, of which 295.243 s was host preparation. Allocating
and zeroing intermediate CPU read buffers accounted for 218.719 s. This motivates
the direct pinned-read candidate. The later complete component run at `82e504c`
passed **136 tests**, including all 28 explicit GPU cases and the direct-read
byte checks, exclusive-slot ownership and failure-after-first-chunk recovery.
JUnit evidence: `artifacts/components_82e504c.xml`. Native direct-read serving,
prefill cadence, cancellation/pressure, long DFlash C4 and the final matched
performance/quality gates remain pending.

## Corrected long C4, direct reads and asynchronous retirement (2026-09-29)

`engine_long_c4_dflash_direct_cadence_02`, frozen source `053383f`, completed with
Q8_0 target / FP8 e4m3 target KV, BF16 DFlash2, K131072/G65536, B128, CPU cache
36 GiB, packed direct-pinned assembly, prefill cadence 8, batch budget 4224 and
request prefill cap 1024. Inputs are the same four hash-checked long histories
10/22/36/37 (532797 total prompt tokens). Finish checks were publication-only.

All four requests compressed; ten worker/scheduler refreshes matched, including
six genuine history selections above K and actual native four-request batches.
All requests, physical GPU pages except the permanent null page, authority
credits, archive slots, CPU read-cache entries and retirement jobs were released.
The process exited normally. The existing native one-semaphore shutdown warning
recurred. No foreign/unknown GPU PID or monitoring error was observed.

| Frozen input | Output / finish | TTFT (s) | Arrival decode (tok/s) | Maximum burst gap (s) |
|---|---:|---:|---:|---:|
| 10 Quill | 114 / EOS | 1858.681 | 32.660 | 0.303 |
| 22 Effect | 128 / limit | 1842.087 | 6.135 | 15.680 |
| 36 Dynamo | 128 / limit | 1815.470 | 2.733 | 25.789 |
| 37 Dynamo | 128 / limit | 1858.682 | 31.052 | 0.280 |

The whole-group arrival decode rate is **10.443 tok/s over 47.303 s** (498 output
tokens, 494 after excluding one first token per request). Cold group service is
1862.795 s / 0.267 output tok/s. A separate diagnostic intersection where all
four streams overlap contains 319 arrivals over 3.2488 s (98.189 tok/s); this
short selected interval is **not the group throughput or a steady-state result**.

Native verification ran 121 request steps of eight rows. Of 847 draft candidates,
466 were rejected (sampling acceptance 44.982%); these old-worker counters still
include terminal overshoot and are not identical to final user-visible acceptance.
The four generated texts/token streams differ from the earlier target-only C4.
No tools or official task verifier executed, so no task-quality pass is implied.

Total assembly time is 15.178 s, including 11.754 s host preparation, for
36660314112 uploaded bytes. Distinct refresh intervals total 83.242 s. Each CPU
retirement reclaimed its archive in 28.2–30.6 ms; enqueue-to-native-ack was
47.6–154.4 ms. Mixed prefill/refresh blocking remains a major observed limitation;
retirement no longer introduces the previously observed multi-second pauses in
this run. The combined options and changed query sizes prevent attributing gains
to one optimization or claiming a matched speedup over historical runs.

Sampled owned GPU peak: 81222 MiB (79.32 GiB). Sampled summed process RSS peak:
41449444 KiB (39.53 GiB), with 982 GPU samples including startup. This run predates
scheduler terminal-prefix clipping, active-authority GC and canonical GDN state
continuation. It discards finished namespaces, and does not qualify persistent
sessions. Source/result/resource logs and derived summaries are in the artifact
folder; `qualification_summary.json` and `overlap_diagnostic.json` preserve the
measurement scopes explicitly.

After this run, the updated complete component suite passes **182 tests**,
including the six native accepted-prefix GDN/conv GPU comparisons. The first
attempt used an invalid multi-token non-speculative reference with a one-column
SSM table and failed four comparisons. The corrected reference executes ordinary
one-token native updates; normalization code and numeric tolerances did not
change. Both the failed artifact and corrected evidence are retained:
`components_after_long_c4.xml`, `accepted_state_gpu_correct_reference.xml`, and
`components_accepted_state_verified.xml`.

## Short native terminal and profiler gate (2026-09-29)

`artifacts/engine_terminal_dflash_profile_02`, source `1f2dd81`, passed
short C1/C4 terminal lifecycle qualification with Q8_0/FP8/DFlash2 and unchanged
K131072/G65536. Five requests ended at their 128-token output limit. Their
archived canonical lengths are exactly 141, 141, 142, 151 and 144 tokens;
respectively 1, 1, 4, 7 and 1 extra computed speculative rows were discarded.
All sealed archive records passed full checksums. No active/retiring authority,
archive slot, pending download, reservation or owned native block remains;
1161 of 1162 native blocks are free (one permanent null block). This gate does
not exercise compression, EOS/cancellation, persistent sessions or task quality.

The eight-iteration concurrent CUDA trace contains 29052 kernels. Q8_0 GGUF
`mul_mat_q8_0` consumes 1091.645 ms of 1194.157 ms summed kernel durations
(91.42%). The GPU kernel interval union is 1194.044 ms across a 1270.405 ms
first-to-last-kernel span. This identifies the current short-context batch
linear implementation as a priority; it does not establish the share of long
prefill/refresh stalls. The trace's GDN MTP kernel consumes 13.249 ms and the
Q8_1 activation quantizer 4.102 ms. Profiling export interrupts responses for
about 18.5 s, so all HTTP throughput from this instrumented run is explicitly
ineligible as performance evidence. Sampled owned GPU peak is 71442 MiB.

The non-GPU suite after the short-mode harness change passes 151 cases with
34 GPU cases skipped (`components_terminal_profile.xml`). This is not a new
full GPU-suite claim. The source-pinned recurrent continuation method remains
unwired in serving; successful terminal archive clipping does not imply it is.

## First uninstrumented short Q8 adapter comparison

Both runs use source `8d919de`, identical Q8_0/FP8/DFlash2/K/G, short prompts,
128-token limits, batch budget 1152, per-request prefill cap 1024, full archive
checksums and async retirement. Each retains C1 plus a full C4 warmup and a
measured C4; all nine requests pass exact terminal and resource-release checks.

| Artifact | C1 arrival decode | Warmup C4 aggregate | Measured C4 aggregate | Measured decode window |
| --- | ---: | ---: | ---: | ---: |
| `engine_short_warm_native_q8_01` | 89.253 tok/s | 87.652 tok/s | 88.203 tok/s | 5.759 s |
| `engine_short_warm_tensorcore_q8_01` | 89.952 tok/s | 44.668 tok/s | 89.094 tok/s | 5.702 s |

Both measured groups produce 512 tokens; arrival decode counts 508 after
excluding each first token. The observed whole-window ratio is only 1.0101x.
The candidate has a 2.799 s gap even in its second C4 group. Logs record Triton
compilation during inference; the prototype specialized matrix/reduction kernels
on exact batch size, so request departure can introduce a shape absent from the
first group. No stall time is removed to improve these figures. The subsequent
implementation uses dynamic row counts and has a cache-reuse regression test.

C1 token IDs agree exactly, but all four measured C4 outputs differ (common
prefixes 121, 3, 1 and 0 tokens). Small linear numerical errors are not proof of
model/task equivalence. This comparison therefore establishes lifecycle behavior
and a compilation problem, not accepted performance or task-quality gains.
Sampled GPU peaks are 71430/71176 MiB. Derived measurements are retained in
`artifacts/short_q8_comparison_01.json`.

### Dynamic batch-size follow-up

`engine_short_warm_tensorcore_q8_dynamic_01` at `5bc5929` passes the same native
nine-request lifecycle gate. Its retained warmup C4 is 170.961 arrival tok/s;
the measured C4 is **173.112 arrival tok/s over 2.934524 s**, again 512 emitted
tokens / 508 decode arrivals. Compared with the original 88.203 tok/s measured
group, the short-test ratio is **1.9626x**. Maximum measured burst gap is
0.1287 s (the other three requests are <=0.0875 s). No time is excluded to
produce these rates. C1 is 89.138 tok/s; its IDs agree with the native baseline.

All nine canonical terminal prefixes/checksums and all resource cleanup checks
pass. Sampled owned GPU peak is 71174 MiB. Native cache sizing chose 1143 total
blocks versus 1162 in the baseline; both use the same 0.72 memory-utilization
setting, have enough capacity for these fixtures and finish with only the null
block occupied. This is a single short fixed-output comparison, not a confidence
interval, long-context benchmark or task-quality acceptance. The four measured
C4 token streams still differ (common prefixes 121, 3, 1 and 43), so the opt-in
adapter remains experimental. The next gate uses the same four frozen long
histories as the corrected long-C4 lifecycle run.

## Long C4 with Q8 Tensor Core (2026-09-29)

`engine_long_c4_dflash_tensorcore_01` at `2eae320` passes four compressed
requests, ten matched worker/scheduler refreshes and final native/resource
cleanup. It uses the same four frozen input IDs (532797 prompt tokens),
Q8_0/FP8/DFlash2, K131072/G65536, batch4224/per-request1024, cadence8,
36 GiB CPU read cache, packed/direct assembly and async publication-only
retirement as `engine_long_c4_dflash_direct_cadence_02`.

| Metric | Earlier corrected native Q8 | Tensor Core |
| --- | ---: | ---: |
| Output tokens | 498 | 498 |
| Whole-group arrival decode | 10.443 tok/s | 18.619 tok/s |
| Arrival decode window | 47.303 s | 26.533 s |
| Whole-group service time | 1862.795 s | 761.158 s |
| Longest burst gap | 25.789 s | 14.846 s |
| Sampled owned GPU peak | 81222 MiB | 80018 MiB |

Observed ratios are 1.783x for the complete arrival-decode window and 2.447x
for service time. No prefill-interference interval is removed. The earlier
baseline predates exact terminal clipping; this is not a repeated controlled
single-change confidence estimate. Request lengths/finish reasons match, but
three output streams differ (common prefixes 5, 128 [identical], 66 and 0).
Task-quality equivalence and persistent sessions remain unqualified.

Per-input decode rates are 59.223 / 11.916 / 5.033 / 50.090 tok/s; TTFT is
758.644 / 750.400 / 734.598 / 758.597 seconds. Mixed prefill and refresh still
block already-generating requests. Distinct refresh intervals total 41.989 s;
assembly totals 12.296 s, including 10.065 s host preparation, with unchanged
36660314112 uploaded bytes. All four terminal archive lengths now equal
prompt+output-1 (494 archived generation rows overall). Native requests,
reservations, owned blocks, authorities, archive slots and pending downloads
return to zero; 1072 of 1073 physical blocks are free (one null block).

This run verifies long-context lifecycle and measures useful improvement, but
does not satisfy final long-running agent or high-throughput acceptance.
Full derived evidence: `artifacts/engine_long_c4_dflash_tensorcore_01/comparison_summary.json`.

## Agent API and authority lifecycle preflight (2026-09-29)

The first official pair (`agent_tensorcore_c2_initial_01`) made 14 calls rejected
before inference because native auto-tool parsing was not configured. They were
stopped as infrastructure failures, not scored model outcomes. The replacement
backend uncovered two launcher omissions: telemetry-directory creation and
post-warmup capture arming. All failed artifacts are retained. In particular,
`agent_backend_tensorcore_03/tool_preflight.json`'s HTTP-only success does **not**
qualify KVMem; its `qualification_correction.json` records the fatal retirement
error immediately after the HTTP response.

The strengthened preflight passes on `agent_backend_tensorcore_04` at `9cf462a`:
one parsed bash call (no tool executed), native timing/speculative metrics,
284 prompt tokens, 31 generated tokens and exactly 314 archived accepted rows.
All request/reservation/physical-page ownership, pending downloads and archive
slots return to zero, with 1072/1073 native blocks free (the null block remains).
The API remains healthy. Its 0.306-second decode interval is a tool/lifecycle
diagnostic, not a throughput benchmark or task score.

The subsequent `agent_tensorcore_c2_initial_02` is **unscored infrastructure
failure**, not a quality or clean C2 performance result. Its generated Squid
proxy timed out a silent model call after 900642 ms (`TCP_MISS_TIMEDOUT/504`).
The client retried while the old relay kept inference alive, producing three
active requests for two serial agents. Both attempts and all failure evidence
are retained; only owned processes/containers were stopped. Host tests/builds
also overlapped this run. One completed 30763-prompt/12519-output request had
30.345 tok/s decode and 19.516% draft acceptance, but cannot establish matched
framework or model-quality differences.

The replacement relay propagates disconnects and the pinned process-local Pier
entry point aligns the proxy with the existing 21600-second model-call timeout.
On `agent_backend_tensorcore_05`, actual GPU cancellation after 2048 accepted
prefill tokens releases all archive/native ownership in 1.107 s. Another 1024
in-flight rows are correctly discarded; the API remains healthy. This is active
prefill cancellation, **not compressed-refresh cancellation**. The new official
pair is `agent_tensorcore_c2_initial_03`, with matching tool and cancellation
preflight evidence; no task score is available at this checkpoint.

The latest full host suite passes 194 cases with 41 GPU cases deferred during
serving (`components_nvme_reservation_host_20260929.xml`). Strict physical NVMe
reservation and shared byte quotas pass native CPU tests; serving GC/capacity
feedback, GPU pressure/reservation and persistent sessions remain unqualified.

The corrected `_03` transport has now delivered a real termenv response in
978.242 seconds (HTTP 200, no disconnect/error), exceeding the old 900-second
proxy timeout. It contains 11367 prompt tokens and 18135 output tokens, including
17990 reasoning tokens. Native first-to-last decode is 950.649 seconds:
19.075 tok/s, with 19.689% draft acceptance. The neighbouring agent was issuing
fresh prefills, and host development/tests/builds overlapped the run. This is
transport reliability evidence and a retained workload observation, not a clean
matched performance comparison or a verifier score. The agent proceeded to its
next call. At this checkpoint 25 calls across the two tasks completed without
relay errors; both official tasks remain unscored and running.
Evidence: `agent_tensorcore_c2_initial_03/transport_long_response.json`.

The `_03` pair has now finished: both termenv and tomlkit reached the unchanged
5400-second official agent limit and received verifier reward 0. Pier exited 0
and classified both as scored `AgentTimeoutError`, rather than infrastructure
failures. This is a failed two-task quality result, not final suite acceptance.
The 32 delivered termenv calls have weighted per-request decode 27.072 tok/s;
the 28 delivered tomlkit calls have 20.067 tok/s. These divide summed output
interval tokens by summed native decode time, **not aggregate C2 throughput**.
Each task also has one undelivered final call at termination. Development and
host checks overlapped execution, and no matched native control was run; these
observations cannot isolate numerical quality from scheduling/prefill costs.
Full results and exceptions remain in `agent_tensorcore_c2_initial_03/results.json`.
All 64 backend requests (including two preflights) retired with no accepted
history, archive slots or pending downloads remaining before owned services were
stopped. Unrelated QW3 containers were left running.

The new prompt-session component GPU gate passes 13 cases, including portable
GDN/draft CUDA state copying and reconstruction into poisoned replacement native
FP8 pages (`session_components_gpu_01.xml`). These are component tests; actual
model HTTP restore and concurrent session behavior still need qualification.
