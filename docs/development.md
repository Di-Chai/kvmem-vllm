# Development goal and acceptance

Develop an out-of-tree KVMem plugin in this repository, with explicit versioned
vLLM hooks where public extension points cannot represent historical working-set
reselection. The qw3 executor is a read-only semantic reference, never the serving
backend. The goal remains active until the full acceptance scope below is proved.

Primary qualification is Qwen3.8-27B Q8_0 target weights, FP8 e4m3 target KV,
DFlash2, and per-request K131072/G65536. BF16 activations and draft weights do not
change the Q8_0 target path. vLLM owns native GPU allocation, scheduling, batching,
attention and sampling. Logical history, dense working positions, physical pages
and generation-qualified repository handles must remain distinct.

## Acceptance scope and current evidence

| Requirement | Current evidence | Remaining acceptance |
| --- | --- | --- |
| Original selection/storage semantics | Unchanged host core with file hashes in `third_party/qw3/PROVENANCE.json`; MeanK score bridge; immutable raw K; dense position assembly; original sink/recent bands | Match old controller's final-query override and record granularity before declaring an equivalent-workload speedup |
| Bounded native target execution | `engine_deepswe_effect_bounded_03`: real 197110-token history, 65 matched refresh commits, bounded native attention allocation, coherent natural-EOS continuation and complete cleanup | Cancellation and GPU/CPU/NVMe capacity-pressure tests; long C4 passes but performance is poor |
| DeltaNet and transaction correctness | Native target/DFlash identity replay, C4 recurrent-state restoration, rejected candidate exclusion, per-generation leases and failure fencing | Compressed target/draft refresh, accepted-boundary checkpoints, terminal-prefix rollback and persistent sessions |
| Efficient transfers and storage | Packed D2H; fenced H2D; moved-block FP8 assembly; coalesced native NVMe I/O; opt-in archive-layout assembly and bounded CPU read cache | Independent performance repeats, overlapping preparation, background policy GC and nonblocking retirement |
| Cross-request resources | Native lifetime GPU-page reservation adapter with hybrid-manager tests; shared archive namespace isolation; bounded queue/cache components | Native pressure/cancellation proof, disk admission, CPU total-budget accounting, long-running reclamation |
| Model/quantization extensibility | Separate state, position, capture and KV-layout adapters; public connector options; explicit source/version checks | Broader qualified model, quantization and KV-format matrix; unsupported cases must remain explicit |
| Performance and task quality | Exact frozen DeepSWE token inputs and provenance; native SSE token/usage agreement; cold/arrival-decode measurements kept separate | Matched warmed C1/C4 throughput and latency, independent repeats, DFlash acceptance/verification cost, memory peaks, executed agent-task success |
| Delivery | Independent repository, reproducible GPU setup and dependency pins, local commits and qualification records | Clean install/release verification and final publication/review artifacts |

The current corrected bounded target diagnostic uses 128-token records and a
1024-token batch budget. The old qw3 frozen reference used 32-token records and
its own final-query override. Keep those differences visible; do not change
precision, K/G, histories or output stopping rules to manufacture an improvement.
There is no hard 130 tok/s or 48 GiB acceptance threshold. Substantial concurrent
throughput improvement must be supported by actual matched measurements.

## Current execution stage

The CPU/GPU component suite passed 102 tests at commit `dbd7451`; all five original
C++ tests passed before these Python-only integrations. The corrected bounded C1
run completed 77 tokens at 20.491 HTTP arrival decode tok/s **without DFlash**, with
1877.601 s TTFT. Subsequent short C4 produced 512 capped tokens at 76.225 aggregate
arrival decode tok/s, but waited about 23.75 s for synchronous archive retirement.
These are diagnostic observations, not the goal's performance pass or long C4.

The cached/packed candidate `engine_deepswe_effect_cached_packed_01` passed the
same 197K C1 and short C4 with exactly matching output token IDs and complete
cleanup. C1 TTFT decreased from 1877.601 to 1280.156 s (31.82%); arrival decode
was 19.448 rather than 20.491 tok/s. This demonstrates a cold prefill/refresh
improvement, not a decode gain or a warmed performance pass. Native lifetime
GPU-page reservation was attached. The latest CPU suite passed 83 tests; 24 GPU
component tests were skipped to avoid interfering with the live model run.

Long-history C4 now passes with actual native four-request compressed batches,
but aggregate arrival decode is 3.686 tok/s due to prefill/retirement interference.
Proceed with compressed DFlash and cancellation/pressure tests. Finish DFlash refresh and accepted-prefix/session
lifecycle before the final warmed performance and agent-quality gates. A client
submitting four HTTP requests does not itself prove four-request GPU batching.

## Evidence corrections

`engine_compression_k128g64_02` completed deterministic repeated replay but used
an incorrect cache-group classifier: native Mamba specs also advertise slot
mapping. Its recurrent-metadata semantic claim was withdrawn. The corrected
helper selects actual full-attention specs, and the corrected real bounded run
above passes. Keep the old artifact as a historical observation, not a semantic
reference. Detailed measurements and limitations are in
[the qualification record](qualification-20260928.md), with measurement definitions
in [measurement.md](measurement.md) and boundaries in [support.md](support.md).


## Experimental bounded DFlash2 wiring

The new diagnostic `scripts/qualify_engine.py --bounded-dflash` enables a narrow
contract for the existing BF16 DFlash2 draft while retaining Q8_0 target / FP8
KV and K131072/G65536. Native vLLM uses method `dflash` and selects its DFlash2
implementation from the `DFlash2DraftModel` architecture. Five upstream source
hashes pin the draft factory, position/slot handling and model implementation.

The scheduler clips speculative queries at the accepted-history checkpoint and
reserves both speculative GDN rows and recycling-aware draft SWA pages. It leaves
room for verifier and drafter lookahead. Reselection scores use known prompt rows,
or only the first already-sampled input during decode, never unverified draft
candidates. Provisional target forwards restore every batch neighbour's recurrent
state; only the final forward reaches native sampling and drafting. Draft pages
and their logical position frame remain independent of the target dense window.
The actual native draft groups are checked against BF16 SWA geometry before any
compressed refresh. Mandatory target recent context must contain the draft window.
Logical histories beyond the draft RoPE table are rejected pending an explicit
renormalization/session adapter.

Native allocator tests cover combined full attention + speculative Mamba + SWA,
future-growth credit exclusion, recycling and release. These and the admission /
transaction / harness tests pass; they are not compressed DFlash GPU evidence.
The long target C4 run `engine_long_c4_cached_packed_01` was launched from commit
`776bb7a` before these DFlash changes and continues with that loaded runtime and
its saved source manifest. Do not attribute its eventual results to later code.


A separate retirement measurement option now allows omitting diagnostic archive
readback, while preserving publication fences and complete cleanup. Both modes
are covered by real host-archive retirement tests; the publication-only mode has
not yet been measured with the native GPU server. The default still verifies
all checksums. Its implementation does not change the live C4 run, which retains
full readback, and does not satisfy the outstanding background-retirement gate.

The native cancellation diagnostic is now wired behind `--cancel-during-refresh`.
Worker transaction timestamps, native termination status and post-output capacity
snapshots support a client-disconnect test with subsequent C4 recovery. The
validator requires physical page return, not just cleared request IDs or credits.
CPU/socket and scheduler tests pass; the actual native GPU cancellation run remains
pending. Source changes made after `engine_long_c4_cached_packed_01` launched do
not affect its already loaded runtime.


`engine_deepswe_effect_bounded_dflash_01` is now running the same exact 197110-token
Effect input as the corrected target C1, from commit `62726f0`. It retains full
checksum verification and the same K/G and precision. The requested batch budget
was 1024, but native draft-slot reservation followed by KVMem alignment produces
896-token prefill chunks. It is a correctness run, not a matched 1024-query
performance comparison. Future matched C1 uses budget 1152 / request cap 1024;
C4 uses budget 4224 / cap 1024, and must verify the actual scheduled queries.
It has now passed 74 matched refreshes and complete five-request archive cleanup.
C1 generated 77 natural-stop tokens at 85.509 arrival decode tok/s over 0.889 s;
the subsequent short C4 reached 91.956 aggregate tok/s. C1 token IDs match the
cached target control and all short C4 texts match prior native DFlash controls.
This is compressed-prefill / later-decode evidence; decode-time refresh and long
DFlash C4 are still pending. See the qualification record for timing caveats.

## Direct pinned archive reads and reproducible workers

The live DFlash run's host breakdown identified repeated zero-initialization of
the CPU read buffer as a large cost: one 5.44 GB, entirely cache-hit refresh spent
2.645 s allocating buffers, 0.388 s copying cached bytes, and 3.927 s assembling
in total. These are one-round wall timings, not an isolated optimization result.

The new opt-in `--packed-assembly --direct-pinned-reads` reads immutable records
directly into the existing two bounded pinned H2D slots. It removes the separate
zeroed bytearray and second host copy. Namespace/generation checks and archive
leases remain mandatory; CPU preparation and DMA both own their staging slot.
Injected read failure after an earlier GPU chunk must retain the transaction
failure fence. The default path remains unchanged. CPU suite: 104 passed,
28 GPU tests skipped at that stage. The complete CPU/GPU suite subsequently
passed 136 tests at `82e504c`, including direct-path bytes and failed-read fences.
Direct-path native model serving/performance qualification is still pending.
`assembly_host_prepare_seconds` now includes staging copy/submission and waits
for a bounded slot in both paths; earlier reports excluded the upload call, so
use total `assembly_seconds` for comparisons spanning this instrumentation change.

Future engine runs copy the worker Python package and built native extension into
their artifact directory and import that snapshot in all child processes. Both
source and worker manifests are hashed; changing the workspace while a test runs
cannot alter a later worker import. Dependency environments remain pinned and
must not be rebuilt or updated during a run.

An experimental `--prefill-cadence N` now requests native prefill throttling on
N-1 of each N mixed scheduling steps. It does not resize queries, reorder the
running queue or override native admission/capacity gates. Prefill-only work,
unavailable decode and capacity-bound scheduling proceed normally. The default
is 1. This can reduce repeated prefill interruptions but cannot shorten one
blocking refresh or remove retirement pauses; GPU benefits remain unmeasured.
The harness records actual prefill/decode query-size histograms to detect draft
headroom and alignment changes in future comparisons. Targeted scheduler,
option and harness tests: 35 passed. Native cancellation qualification also
counts the disconnected request when waiting for archive retirement.

## Active next GPU run

`engine_long_c4_dflash_direct_cadence_01` launched from `80853e8`, with its own
frozen worker package and native extension. It uses the same four real long
inputs as the earlier target C4, Q8_0/FP8, BF16 DFlash2, K131072/G65536 and B128.
Candidates: direct pinned reads, 36 GiB CPU cache, packed assembly, prefill cadence
8, global batch budget 4224 / request cap 1024, and publication-only retirement.
It must demonstrate actual four-request native compressed batching and cleanup.
This is a combined-path qualification, not a matched speedup against the old
256-query, full-readback target C4. Actual scheduling histograms and complete
HTTP arrival windows will determine follow-up matched controls. Final outcome
is pending; the persistent goal remains active.

## Bounded asynchronous archive retirement

The opt-in `--async-archive-retirement` detaches a finalized authority only after
all native capture DMA has drained. A single bounded CPU worker publishes its
remaining records, optionally verifies every checksum, and reclaims the entire
namespace. Other native requests can keep executing. Publication, verification,
reclamation and acknowledgment timestamps are retained in the archive report.

The native connector deliberately retains the completed request's physical GPU
pages and lifetime reservation until CPU reclamation succeeds. An additional
authority-count gate caps live plus retiring requests at `max_num_seqs`, so
background cleanup cannot silently exceed the configured archive namespace
budget. This first implementation trades delayed capacity reuse for safe overlap;
it does not yet release GPU pages ahead of CPU retirement. Cancellation and
checksum/I/O failures cannot issue a successful free acknowledgment. Idle native
`no_forward` steps deliver completion even when every HTTP request has ended.
The pinned helper source hash rejects incompatible native completion semantics.

Host tests cover concurrent namespace writes during blocked checksum reading,
held generation leases, queue bounds, corrupt archive failure, admission with
spare physical pages, idle native connector completion, and graceful shutdown
ordering. Native GPU retirement/cancellation/pressure qualification remains
pending. The active long-C4 run uses its frozen pre-change package and synchronous
publication-only retirement; it cannot qualify this new option. This feature
does not implement active-history policy GC, persistent sessions or total CPU /
disk byte admission, which remain separate goal requirements.

Latest CPU regression after this integration: **120 passed, 28 GPU tests skipped**
to avoid contention with the active model run. The earlier complete CPU/GPU suite
of 136 tests qualifies the previous code, not asynchronous retirement. Evidence
for this CPU run is `artifacts/components_async_retirement_cpu.xml`.

## Correction: cold-block residency, 2026-09-29

A repeated reselection defect was reproduced after the earlier model runs:
a dropped block retained GPU residency. If its old dense position was reused by
another block, selecting it again at that position could incorrectly skip the
archive reload. `RequestMemory.commit_selection` now changes working residency
only after the assembly completion fence, preserving generation-qualified leases.

Earlier compressed model timings and lifecycle observations remain historical
measurements, but **do not establish compressed numerical correctness or accepted
performance**. The metadata classification correction described elsewhere did not
fix this separate issue. Corrected model qualification is pending. The optimized
long-C4 run `engine_long_c4_dflash_direct_cadence_01` was deliberately interrupted
before compression, and is not a completed benchmark.

The regression cycles different blocks through the same dense position and back.
Ordinary, packed and direct-pinned GPU assembly match independent native FP8 K/V
bytes. The complete updated suite passes **154 tests**, plus **5 original C++
CTest tests**. Evidence: `artifacts/components_cold_residency_fixed.xml` and
`artifacts/cold_reselection_regression.xml`. This qualifies the component fix;
it does not replace native model, task-quality or throughput qualification.

The original repository GC now has Python bindings, including generation leases,
backing flags, policy configuration and GIL-safe background callback shutdown.
Five new host tests cover protection, callback failures/retry and shutdown.
These bindings are not yet activated in the serving authority: durable backing
publication, full retrieval-snapshot leases and bundle cleanup must be integrated
before enabling active-history GC. The qualification harness now stops its owned
server before joining interrupted streaming readers.

## Authority GC integration (CPU qualified, serving activation pending)

An optional `RequestAuthority(gc_config=...)` now attaches the original policy
worker. It publishes RawBacked/ValueBacked/Indexed only after successful durable
write completion. Scoring holds generation-qualified leases over the entire
live index through native selection; returning copied handle values alone does
not protect against concurrent retirement.

Background release reclaims the generation's disk allocation and optional CPU
read-cache entry, then queues a bounded tombstone. The foreground removes old
index generations before installing recycled reservations. The callback owns no
Store/authority/GC bridge reference, touches no GPU data, and never mutates the
foreground dictionary. Finalization stops policy GC before checksum iteration;
reports distinguish previously reclaimed records from live records actually
verified. Request-step aging advances once at construction; ingestion grace can
make progress within that request without inventing per-token request steps.

Seven integration cases cover fixed-capacity reuse across 272 accepted tokens
in an eight-block repository, original sink protection, cache invalidation,
scoring leases and exception cleanup, unobserved durable writes, background
single-request ingestion, and final checksum accounting. Combined CPU regression:
**133 passed, 28 GPU tests skipped** while the model run owns the GPU. Evidence:
`artifacts/components_authority_gc_cpu.xml`. Native serving options, scheduler
backpressure on repository exhaustion, and GPU GC qualification remain pending;
this component is not activated in the current model test.

The corrected long-C4 combined-path run is
`engine_long_c4_dflash_direct_cadence_02`, launched from `053383f` with a frozen
worker snapshot. It additionally opts into asynchronous archive retirement.
Actual decode rates and final lifecycle results remain pending; no speedup is
claimed from the earlier interrupted run or from the component tests.

## Scheduler terminal prefix (CPU qualified, GPU pending)

The bounded connector now carries the finished request's scheduler-owned prefix
before native request fields are discarded. Sampling acceptance is not the final
boundary: native EOS/length checks can trim an accepted speculative burst. The
worker applies that boundary before draining the latest capture packet. Aborted
in-flight rows and rows past the final evaluated output prefix are counted as
`terminal_discarded_tokens`, separately from sampling rejections, and never enter
the raw-K/V archive or MeanK index. Late boundaries after excess publication fail
closed; this path does not silently rewind immutable records.

New bounded archive reports require exact agreement between scheduler prefix,
archive history and HTTP token usage, replacing the historical extra-burst
allowance. Legacy reports retain their old validation semantics. Finalization
also distinguishes GC-deleted records from remaining records checksum-verified
on exit. CPU tests cover EOS overshoot, ordinary decode, prefill/decode abort,
ignored requests, metadata transfer, a real archive with mixed-request capture,
and strict qualification checks. Current CPU regression: **143 passed, 28 GPU
tests skipped** (`artifacts/components_terminal_prefix_cpu.xml`).

This fixes prospective capture publication only. Native recurrent state and
DFlash sliding state may still have advanced past the final retained prefix;
persistent sessions remain disabled until final restore/replay is implemented
and qualified. The running long-C4 uses its earlier frozen source and therefore
does not qualify this new boundary path. Its finished namespaces are discarded,
not reused as sessions; its final report must retain that limitation.
