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
